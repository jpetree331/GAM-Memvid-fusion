"""
Synthesizer Node - Runtime Abstraction for Context Injection

============================================================
THE SYNTHESIZER: Convert Raw Pearls to Detailed Abstracts
============================================================

The Librarian architecture stores FULL, RAW conversation exchanges.
But we can't inject 15 pages of text into the context window.

The Synthesizer solves this:
1. Receives raw Pearls from search results
2. Passes them to gpt-4o-mini with synthesis instructions
3. Creates detailed abstracts (200-300 words each)
4. Abstracts preserve: core arguments, key definitions, verbatim quotes, emotional tone
5. Abstracts are what get injected into the prompt context

This is a RUNTIME operation - synthesis happens during retrieval, not storage.
"""
import asyncio
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from config import config


@dataclass
class SynthesisResult:
    """Result from synthesizing a Pearl into an abstract."""
    pearl_id: str
    abstract: str
    key_points: List[str] = field(default_factory=list)
    verbatim_quotes: List[str] = field(default_factory=list)
    emotional_tone: Optional[str] = None
    word_count: int = 0
    original_word_count: int = 0
    compression_ratio: float = 0.0
    processing_time_ms: float = 0


@dataclass
class SynthesisBatch:
    """Results from synthesizing multiple Pearls."""
    abstracts: List[SynthesisResult]
    combined_context: str
    total_processing_time_ms: float = 0
    pearls_processed: int = 0
    total_original_words: int = 0
    total_abstract_words: int = 0


# =============================================================================
# Synthesis Prompts
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a Memory Synthesizer for an AI companion system.

Your task is to create a DETAILED ABSTRACT of a conversation exchange.
This abstract will be injected into the AI's context to provide relationship continuity.

CRITICAL REQUIREMENTS:
1. Preserve core logical arguments and key reasoning
2. Retain key definitions and terminology established
3. Capture the emotional tone and relational dynamics
4. Note any commitments, promises, or agreements made
5. Highlight any personal information shared

OUTPUT FORMAT (you MUST use this exact structure):

### Abstract
(A detailed, 200-300 word summary of the core philosophical/emotional arguments.
Do NOT include quotes in this section. Write as a flowing narrative that captures
the essence of the exchange for relationship continuity.)

### Verbatim Quotes
"<Exact substring from the source text - quote 1>"

"<Exact substring from the source text - quote 2>"

"<Optional 3rd quote if highly relevant>"

QUOTE REQUIREMENTS - STRICT FIDELITY:
- Quotes MUST be exact substrings copied from the source text
- Do NOT invent, paraphrase, or approximate - copy verbatim
- Do NOT add Bible verses, author names, or references not in the source
- Select quotes that represent the emotional core or central tension
- Choose quotes that capture the user's voice and personality

REMEMBER: This abstract helps the AI "remember" this exchange naturally.
Quality over brevity - capture the essence, not just the facts."""

SYNTHESIS_USER_TEMPLATE = """Create a detailed abstract of this conversation exchange.

---
EXCHANGE FROM: {timestamp}
TAGS: {tags}

{user_name}: {user_message}

AI: {ai_response}
---

OUTPUT REQUIREMENTS:
1. Write a 200-300 word abstract in the "### Abstract" section
2. Include 2-3 VERBATIM quotes in the "### Verbatim Quotes" section
3. Quotes must be EXACT substrings - copy directly from the text above, do not paraphrase

Use the exact format specified in your instructions."""


BATCH_SYNTHESIS_PROMPT = """You are synthesizing multiple conversation exchanges into a combined context.

Create a flowing narrative that weaves together the key information from these exchanges.
Organize by theme rather than chronologically if that makes more sense.
Total target: 400-600 words for the combined context.

EXCHANGES:
{exchanges}

---

Create a unified context summary that an AI can use to "remember" these past conversations naturally."""


# =============================================================================
# Synthesizer Class
# =============================================================================

class Synthesizer:
    """
    Runtime synthesizer that converts raw Pearls to detailed abstracts.

    Uses gpt-4o-mini for cost-effective, fast synthesis.
    This is called during retrieval, not during storage.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        debug: bool = False
    ):
        """
        Initialize the Synthesizer.

        Args:
            api_key: OpenAI API key (defaults to config)
            base_url: OpenAI base URL (defaults to config)
            model: Model to use for synthesis (default: gpt-4o-mini)
            timeout: Request timeout in seconds
            debug: Enable debug logging
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.model = model
        self.timeout = timeout
        self.debug = debug

        if not self.api_key:
            print("[Synthesizer] WARNING: No OpenAI API key configured")

    async def synthesize_pearl(
        self,
        pearl_id: str,
        user_message: str,
        ai_response: str,
        user_name: str = "User",
        tags: Optional[List[str]] = None,
        timestamp: Optional[str] = None
    ) -> SynthesisResult:
        """
        Synthesize a single Pearl into a detailed abstract.

        Args:
            pearl_id: ID of the Pearl being synthesized
            user_message: The full user message
            ai_response: The full AI response
            user_name: User's name for the transcript
            tags: Tags associated with the Pearl
            timestamp: When the exchange occurred

        Returns:
            SynthesisResult with the detailed abstract
        """
        start_time = time.time()

        original_words = len(user_message.split()) + len(ai_response.split())

        # Build the prompt
        prompt = SYNTHESIS_USER_TEMPLATE.format(
            timestamp=timestamp or "Unknown",
            tags=", ".join(tags or []) or "None",
            user_name=user_name,
            user_message=user_message[:8000],  # Limit for context window
            ai_response=ai_response[:8000]
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()

                result = response.json()
                abstract = result["choices"][0]["message"]["content"].strip()

                # Extract emotional tone if present
                emotional_tone = None
                if "[" in abstract and "]" in abstract:
                    # Look for [tone] at the end
                    last_bracket = abstract.rfind("[")
                    if last_bracket > len(abstract) - 100:
                        emotional_tone = abstract[last_bracket+1:abstract.rfind("]")]

                abstract_words = len(abstract.split())
                processing_time = (time.time() - start_time) * 1000

                if self.debug:
                    print(f"[Synthesizer] Pearl {pearl_id}: {original_words}→{abstract_words} words ({processing_time:.0f}ms)")

                return SynthesisResult(
                    pearl_id=pearl_id,
                    abstract=abstract,
                    emotional_tone=emotional_tone,
                    word_count=abstract_words,
                    original_word_count=original_words,
                    compression_ratio=original_words / max(abstract_words, 1),
                    processing_time_ms=processing_time
                )

        except Exception as e:
            print(f"[Synthesizer] Error synthesizing Pearl {pearl_id}: {e}")

            # Fallback: truncate the original
            fallback = f"Exchange summary: {user_name} discussed: {user_message[:200]}... AI responded about: {ai_response[:200]}..."

            return SynthesisResult(
                pearl_id=pearl_id,
                abstract=fallback,
                word_count=len(fallback.split()),
                original_word_count=original_words,
                compression_ratio=original_words / len(fallback.split()),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    async def synthesize_pearls(
        self,
        pearls: List[Dict[str, Any]],
        user_name: str = "User"
    ) -> SynthesisBatch:
        """
        Synthesize multiple Pearls into abstracts.

        Args:
            pearls: List of Pearl dicts with user_message, ai_response, etc.
            user_name: User's name for transcripts

        Returns:
            SynthesisBatch with all abstracts and combined context
        """
        start_time = time.time()

        # Synthesize each Pearl concurrently
        tasks = []
        for pearl in pearls:
            tasks.append(self.synthesize_pearl(
                pearl_id=pearl.get("id", pearl.get("pearl_id", "unknown")),
                user_message=pearl.get("user_message", ""),
                ai_response=pearl.get("ai_response", ""),
                user_name=user_name,
                tags=pearl.get("tags", []),
                timestamp=pearl.get("created_at")
            ))

        results = await asyncio.gather(*tasks)

        # Combine abstracts into a unified context
        combined_parts = []
        for result in results:
            combined_parts.append(result.abstract)

        combined_context = "\n\n---\n\n".join(combined_parts)

        total_time = (time.time() - start_time) * 1000

        return SynthesisBatch(
            abstracts=results,
            combined_context=combined_context,
            total_processing_time_ms=total_time,
            pearls_processed=len(results),
            total_original_words=sum(r.original_word_count for r in results),
            total_abstract_words=sum(r.word_count for r in results)
        )

    async def synthesize_for_context(
        self,
        pearls: List[Dict[str, Any]],
        user_name: str = "User",
        max_context_words: int = 800,
        create_unified: bool = True
    ) -> str:
        """
        Synthesize Pearls into context ready for prompt injection.

        This is the main method for the retrieval pipeline:
        1. Takes raw Pearls from search
        2. Creates detailed abstracts
        3. Optionally creates a unified narrative
        4. Returns context string for injection

        Args:
            pearls: Raw Pearls from search results
            user_name: User's name
            max_context_words: Target word limit for combined context
            create_unified: If True, create a unified narrative; else concatenate

        Returns:
            Context string ready for prompt injection
        """
        if not pearls:
            return ""

        # Convert Pearl objects to dicts if needed
        pearl_dicts = []
        for p in pearls:
            if hasattr(p, 'to_dict'):
                pearl_dicts.append(p.to_dict())
            elif hasattr(p, 'user_message'):
                pearl_dicts.append({
                    "id": getattr(p, 'id', 'unknown'),
                    "user_message": p.user_message,
                    "ai_response": p.ai_response,
                    "tags": getattr(p, 'tags', []),
                    "created_at": getattr(p, 'created_at', None)
                })
            else:
                pearl_dicts.append(p)

        batch = await self.synthesize_pearls(pearl_dicts, user_name)

        if create_unified and len(batch.abstracts) > 1:
            # Create a unified narrative from all abstracts
            return await self._create_unified_context(batch, max_context_words)
        else:
            return batch.combined_context

    async def _create_unified_context(
        self,
        batch: SynthesisBatch,
        max_words: int = 800
    ) -> str:
        """Create a unified narrative from multiple abstracts."""
        if not batch.abstracts:
            return ""

        # If only one abstract, return it directly
        if len(batch.abstracts) == 1:
            return batch.abstracts[0].abstract

        # Build exchanges text
        exchanges = "\n\n".join([
            f"[Exchange {i+1}]\n{result.abstract}"
            for i, result in enumerate(batch.abstracts)
        ])

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You synthesize multiple conversation summaries into a unified context narrative."
                            },
                            {
                                "role": "user",
                                "content": BATCH_SYNTHESIS_PROMPT.format(exchanges=exchanges)
                            }
                        ],
                        "temperature": 0.3,
                        "max_tokens": max_words * 2  # Allow for some variance
                    }
                )
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"[Synthesizer] Error creating unified context: {e}")
            # Fallback: return concatenated abstracts
            return batch.combined_context


# =============================================================================
# Convenience Functions
# =============================================================================

_synthesizer_instance: Optional[Synthesizer] = None


def get_synthesizer() -> Synthesizer:
    """Get or create the global Synthesizer instance."""
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = Synthesizer(debug=True)
    return _synthesizer_instance


async def synthesize_for_prompt(
    pearls: List[Any],
    user_name: str = "User",
    max_words: int = 800
) -> str:
    """
    Convenience function to synthesize Pearls for prompt injection.

    Usage:
        from synthesizer import synthesize_for_prompt
        from memvid_store import get_store

        store = get_store("eli")
        pearls = store.get_raw_pearls_for_synthesis("theology discussion")
        context = await synthesize_for_prompt(pearls, user_name="Jess")
        # context is now ready for prompt injection
    """
    synth = get_synthesizer()
    return await synth.synthesize_for_context(pearls, user_name, max_words)


def synthesize_for_prompt_sync(
    pearls: List[Any],
    user_name: str = "User",
    max_words: int = 800
) -> str:
    """
    Synchronous version of synthesize_for_prompt.

    For use in non-async contexts.
    """
    return asyncio.run(synthesize_for_prompt(pearls, user_name, max_words))


# =============================================================================
# Integration with MemvidStore
# =============================================================================

async def get_synthesized_context(
    model_id: str,
    query: str,
    user_name: str = "User",
    max_pearls: int = 5,
    max_words: int = 800
) -> str:
    """
    Full pipeline: Search → Synthesize → Return context.

    This is the main integration point for the Librarian architecture.

    Args:
        model_id: The AI model/persona ID
        query: Search query for relevant memories
        user_name: User's name for synthesis
        max_pearls: Maximum Pearls to retrieve
        max_words: Target word count for synthesized context

    Returns:
        Synthesized context ready for prompt injection
    """
    from memvid_store import get_store

    # 1. Search for relevant Pearls
    store = get_store(model_id)
    pearls = store.get_raw_pearls_for_synthesis(query, limit=max_pearls)

    if not pearls:
        return ""

    # 2. Synthesize into abstracts
    context = await synthesize_for_prompt(pearls, user_name, max_words)

    return context


def get_synthesized_context_sync(
    model_id: str,
    query: str,
    user_name: str = "User",
    max_pearls: int = 5,
    max_words: int = 800
) -> str:
    """Synchronous version of get_synthesized_context."""
    return asyncio.run(get_synthesized_context(
        model_id, query, user_name, max_pearls, max_words
    ))
