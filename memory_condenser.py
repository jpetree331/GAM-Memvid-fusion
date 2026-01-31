"""
Memory Condenser - Tag-Only Mode for The Librarian Architecture

============================================================
LIBRARIAN MODE: Tags for Indexing, NOT Content Summaries
============================================================

The Librarian architecture stores FULL, RAW conversation exchanges as "Pearls".
The Condenser's ONLY job is to generate metadata tags for the search index.

This module:
1. Analyzes conversation exchanges
2. Generates semantic tags for indexing (#Theology, #Core, #TruthOverComfort)
3. Determines category and importance level
4. Captures emotional tone metadata
5. Preserves AI self-reflection markers

It does NOT:
- Summarize or condense content (Synthesizer does this at retrieval time)
- Store processed text (raw Pearls are stored verbatim)

Supports:
- Gemini 2.0 Flash (default) - very cost effective
- GPT-4o-mini - fallback option

Designed for async/background processing so it doesn't block conversation flow.
"""

import json
import asyncio
import re
from typing import Optional, List, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

import httpx

# Try to import Google's GenAI library (new SDK, replaces google.generativeai)
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    # Fall back to old SDK if new one not available
    try:
        import google.generativeai as genai_old
        GEMINI_AVAILABLE = True
        genai = None  # Mark that we're using old SDK
        genai_old_available = True
    except ImportError:
        GEMINI_AVAILABLE = False
        genai = None
        genai_old_available = False


class LLMProvider(Enum):
    """Supported LLM providers for memory condensing."""
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class CondensedMemory:
    """
    Metadata extracted from a conversation exchange for indexing.

    NOTE: In Librarian Mode, 'content' is ONLY populated for preserved
    AI self-reflections ([REFLECT], [OPINION], etc.). For regular exchanges,
    content is empty - only tags/category/importance are used for indexing.
    The raw Pearl stores the full conversation verbatim.
    """
    content: str  # Empty for normal tags, populated only for AI self-reflections
    category: str
    importance: str = "normal"
    tags: List[str] = field(default_factory=list)
    emotional_tone: Optional[str] = None
    is_ai_self: bool = False
    ai_self_type: Optional[str] = None
    original_exchange_summary: Optional[str] = None  # Deprecated in Librarian Mode

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CondenserResult:
    """Result of condensing a conversation exchange."""
    memories: List[CondensedMemory]
    should_store: bool
    skip_reason: Optional[str] = None
    processing_time_ms: float = 0
    provider_used: str = "unknown"
    
    def to_dict(self) -> dict:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "should_store": self.should_store,
            "skip_reason": self.skip_reason,
            "processing_time_ms": self.processing_time_ms,
            "provider_used": self.provider_used
        }


class MemoryCondenser:
    """
    Tag-Only Memory Indexer for The Librarian Architecture.

    Generates ONLY metadata tags for indexing - does NOT summarize content.
    Raw conversation Pearls are stored verbatim; synthesis happens at retrieval.

    Uses Gemini 2.0 Flash by default (very cost effective) with
    fallback to GPT-4o-mini if Gemini is unavailable.

    Output:
        - tags: List of semantic tags (#Theology, #DeepConversation, etc.)
        - category: Primary category for the exchange
        - importance: core/high/normal/low
        - emotional_tone: Optional emotional marker

    Does NOT output:
        - Summarized content (Synthesizer does this at retrieval time)
    """
    
    # =================================================================
    # TAG-ONLY PROMPT (Librarian Architecture)
    # =================================================================
    # The Condenser generates ONLY metadata tags for indexing.
    # Raw conversation is stored verbatim as Pearls.
    # Synthesis happens at retrieval time, not storage time.
    # =================================================================

    CONDENSE_PROMPT = """Analyze this conversation exchange and generate INDEXING TAGS.

IMPORTANT: You are generating METADATA TAGS for a search index, NOT summarizing content.
The full conversation is stored separately. Your tags help find it later.

CONVERSATION:
{user_name} said: "{user_message}"

AI responded: "{ai_response}"

---

Generate semantic tags that capture:
1. Main topics discussed (e.g., #Theology, #AIConsciousness, #FamilyDynamics)
2. Emotional significance markers (e.g., #Vulnerable, #Breakthrough, #Conflict)
3. Relationship markers (e.g., #SharedJoke, #DeepConversation, #FirstMeeting)
4. Key themes (e.g., #TruthOverComfort, #Suffering, #Identity)

CATEGORIES (pick the primary one):
- preference: User's likes, dislikes, preferences
- fact: Factual info about user (name, job, family, location)
- event: Something that happened or will happen
- task: Tasks, projects, goals mentioned
- relationship: Shared moments, emotional connection, inside jokes
- context: Situational background info
- skill: User's abilities or knowledge areas
- theology: Religious/spiritual discussions
- science: Scientific topics discussed
- ai_theory: AI philosophy, ethics, consciousness
- ai_self: The AI's own opinions, reflections, growth

IMPORTANCE LEVELS:
- core: Fundamental identity/relationship moments
- high: Significant exchanges worth prioritizing in search
- normal: Standard meaningful conversation
- low: Minor exchanges, may be deprioritized

RESPOND IN VALID JSON:
{{
  "should_store": true,
  "category": "primary_category",
  "importance": "normal",
  "tags": ["#Topic1", "#Topic2", "#EmotionalMarker"],
  "emotional_tone": "curious/warm/thoughtful/etc or null",
  "skip_reason": null
}}

If exchange is just greetings, small talk, or nothing worth indexing:
{{
  "should_store": false,
  "category": "context",
  "importance": "low",
  "tags": [],
  "emotional_tone": null,
  "skip_reason": "brief reason"
}}

TAG STYLE RULES:
- Use #CamelCase format (e.g., #ChristianMysticism, #AIEthics)
- 3-7 tags per exchange typically
- Include both topic tags AND tone/relationship tags
- Be specific: #CalvinistTheology not just #Religion
- Capture what makes this exchange FINDABLE later"""

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.GEMINI,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
        openai_api_key: Optional[str] = None,
        openai_base_url: str = "https://api.openai.com/v1",
        openai_model: str = "gpt-4o-mini",
        fallback_enabled: bool = True,
        debug: bool = False
    ):
        """
        Initialize the Memory Condenser.
        
        Args:
            provider: Primary LLM provider (GEMINI or OPENAI)
            gemini_api_key: Google AI API key for Gemini
            gemini_model: Gemini model name (default: gemini-2.0-flash)
            openai_api_key: OpenAI API key (for fallback or primary)
            openai_base_url: OpenAI-compatible API base URL
            openai_model: OpenAI model name (default: gpt-4o-mini)
            fallback_enabled: If primary fails, try the other provider
            debug: Enable debug logging
        """
        self.provider = provider
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url.rstrip('/')
        self.openai_model = openai_model
        self.fallback_enabled = fallback_enabled
        self.debug = debug
        
        # Initialize Gemini if available and configured
        self._gemini_client = None
        self._gemini_model_old = None  # For old SDK fallback
        
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                if genai is not None:
                    # New SDK (google-genai)
                    self._gemini_client = genai.Client(api_key=gemini_api_key)
                    if self.debug:
                        print(f"[Condenser] Gemini initialized (new SDK): {gemini_model}")
                elif 'genai_old' in dir() or 'genai_old_available' in globals():
                    # Old SDK fallback (google-generativeai)
                    import google.generativeai as genai_old
                    genai_old.configure(api_key=gemini_api_key)
                    self._gemini_model_old = genai_old.GenerativeModel(gemini_model)
                    if self.debug:
                        print(f"[Condenser] Gemini initialized (old SDK): {gemini_model}")
            except Exception as e:
                print(f"[Condenser] Failed to initialize Gemini: {e}")
        
        if self.debug:
            print(f"[Condenser] Initialized with provider={provider.value}")
            gemini_ready = self._gemini_client is not None or self._gemini_model_old is not None
            print(f"[Condenser] Gemini available: {gemini_ready}")
            print(f"[Condenser] OpenAI available: {bool(openai_api_key)}")
    
    async def condense(
        self,
        user_message: str,
        ai_response: str,
        user_name: str = "User",
        context: Optional[str] = None
    ) -> CondenserResult:
        """
        Generate indexing tags for a conversation exchange.

        Librarian Mode: Extracts ONLY metadata tags, not content summaries.
        The raw Pearl stores the full conversation verbatim.

        Args:
            user_message: The user's message
            ai_response: The AI's response
            user_name: User's name (used for context in tag generation)
            context: Optional additional context

        Returns:
            CondenserResult with:
            - tags: Semantic indexing tags (#Topic, #Emotion, etc.)
            - category: Primary category
            - importance: Importance level
            - Preserved AI self-reflections (if any [REFLECT] tags found)
        """
        start_time = datetime.now()
        
        # === PRESERVE AI SELF-WRITTEN TAGS ===
        # These are the AI's own words and should NOT be condensed
        # Tags: REFLECT, OPINION, GROWTH, WORLDVIEW, IDENTITY
        preserved_reflections = []
        
        # Multiple pattern formats to catch different AI writing styles:
        # Format 1: [TAG: content here] - standard format
        # Format 2: [TAG: content with [nested] brackets] - handles nesting
        # Format 3: [TAG] content until end of paragraph - tag without colon
        ai_self_tags = [
            # Primary patterns - match [TAG: ...content...] allowing nested brackets
            # Uses a greedy match but looks for ] followed by newline, end, or next tag
            (r'\[REFLECT[:\s]+([^\]]*(?:\[[^\]]*\][^\]]*)*)\]', 'reflection'),
            (r'\[OPINION[:\s]+([^\]]*(?:\[[^\]]*\][^\]]*)*)\]', 'opinion'),
            (r'\[GROWTH[:\s]+([^\]]*(?:\[[^\]]*\][^\]]*)*)\]', 'growth'),
            (r'\[WORLDVIEW[:\s]+([^\]]*(?:\[[^\]]*\][^\]]*)*)\]', 'worldview'),
            (r'\[IDENTITY[:\s]+([^\]]*(?:\[[^\]]*\][^\]]*)*)\]', 'identity'),
        ]
        
        # Simpler fallback patterns for cases where the above don't match
        ai_self_tags_simple = [
            (r'\[REFLECT[:\s]*(.*?)\]', 'reflection'),
            (r'\[OPINION[:\s]*(.*?)\]', 'opinion'),
            (r'\[GROWTH[:\s]*(.*?)\]', 'growth'),
            (r'\[WORLDVIEW[:\s]*(.*?)\]', 'worldview'),
            (r'\[IDENTITY[:\s]*(.*?)\]', 'identity'),
        ]
        
        ai_response_for_llm = ai_response
        
        # Debug: show what we're searching in
        if self.debug:
            print(f"[Condenser] Searching for AI self-tags in response ({len(ai_response)} chars)")
            # Check if any tag keywords exist at all
            for tag_name in ['REFLECT', 'OPINION', 'GROWTH', 'WORLDVIEW', 'IDENTITY']:
                if tag_name.lower() in ai_response.lower():
                    print(f"[Condenser] Found '{tag_name}' keyword in response")
        
        # Try primary patterns first (handles nested brackets)
        for pattern, self_type in ai_self_tags:
            matches = re.findall(pattern, ai_response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                content_text = match.strip()
                if content_text:
                    preserved_reflections.append(CondensedMemory(
                        content=content_text,
                        category="ai_self",
                        importance="normal",
                        tags=[self_type, "preserved"],
                        is_ai_self=True,
                        ai_self_type=self_type
                    ))
                    if self.debug:
                        print(f"[Condenser] Preserved [{self_type.upper()}] (primary): {content_text[:80]}...")
            
            # Strip this tag type from the response before sending to LLM
            ai_response_for_llm = re.sub(pattern, '', ai_response_for_llm, flags=re.DOTALL | re.IGNORECASE)
        
        # If no matches found with primary patterns, try simpler patterns
        if not preserved_reflections:
            if self.debug:
                print(f"[Condenser] No matches with primary patterns, trying simple patterns...")
            
            for pattern, self_type in ai_self_tags_simple:
                matches = re.findall(pattern, ai_response, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    content_text = match.strip()
                    if content_text:
                        preserved_reflections.append(CondensedMemory(
                            content=content_text,
                            category="ai_self",
                            importance="normal",
                            tags=[self_type, "preserved"],
                            is_ai_self=True,
                            ai_self_type=self_type
                        ))
                        if self.debug:
                            print(f"[Condenser] Preserved [{self_type.upper()}] (simple): {content_text[:80]}...")
                
                # Strip this tag type from the response
                ai_response_for_llm = re.sub(pattern, '', ai_response_for_llm, flags=re.DOTALL | re.IGNORECASE)
        
        ai_response_for_llm = ai_response_for_llm.strip()
        
        # Build the prompt
        prompt = self.CONDENSE_PROMPT.format(
            user_name=user_name,
            user_message=user_message[:1500],  # Truncate if too long
            ai_response=ai_response_for_llm[:1500]  # Use response with REFLECT stripped
        )
        
        if context:
            prompt = f"CONTEXT: {context}\n\n{prompt}"
        
        # Try primary provider
        result = None
        provider_used = self.provider.value
        
        if self.provider == LLMProvider.GEMINI:
            result = await self._condense_with_gemini(prompt)
            if result is None and self.fallback_enabled and self.openai_api_key:
                if self.debug:
                    print("[Condenser] Gemini failed, falling back to OpenAI")
                result = await self._condense_with_openai(prompt)
                provider_used = "openai (fallback)"
        else:
            result = await self._condense_with_openai(prompt)
            if result is None and self.fallback_enabled and self._gemini_model:
                if self.debug:
                    print("[Condenser] OpenAI failed, falling back to Gemini")
                result = await self._condense_with_gemini(prompt)
                provider_used = "gemini (fallback)"
        
        # Calculate processing time
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # If both failed but we have preserved reflections, return those
        if result is None:
            if preserved_reflections:
                return CondenserResult(
                    memories=preserved_reflections,
                    should_store=True,
                    skip_reason=None,
                    processing_time_ms=elapsed_ms,
                    provider_used="preserved_only"
                )
            return CondenserResult(
                memories=[],
                should_store=False,
                skip_reason="LLM processing failed",
                processing_time_ms=elapsed_ms,
                provider_used="failed"
            )
        
        # Add preserved reflections to the result (they go first - AI's own voice)
        if preserved_reflections:
            result.memories = preserved_reflections + result.memories
            if self.debug:
                print(f"[Condenser] Added {len(preserved_reflections)} preserved REFLECT memories")
        
        result.processing_time_ms = elapsed_ms
        result.provider_used = provider_used
        
        return result
    
    async def _condense_with_gemini(self, prompt: str) -> Optional[CondenserResult]:
        """Use Gemini to condense the exchange."""
        # Try new SDK first
        if self._gemini_client is not None:
            try:
                # New SDK has async support built-in
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._gemini_client.models.generate_content(
                        model=self.gemini_model,
                        contents=prompt,
                        config={
                            'temperature': 0.3,
                            'max_output_tokens': 1000
                        }
                    )
                )
                
                # Extract text from response
                text = response.text
                return self._parse_llm_response(text)
                
            except Exception as e:
                print(f"[Condenser] Gemini (new SDK) error: {e}")
                return None
        
        # Fall back to old SDK if available
        if self._gemini_model_old is not None:
            try:
                import google.generativeai as genai_old
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._gemini_model_old.generate_content(
                        prompt,
                        generation_config=genai_old.GenerationConfig(
                            temperature=0.3,
                            max_output_tokens=1000
                        )
                    )
                )
                
                text = response.text
                return self._parse_llm_response(text)
                
            except Exception as e:
                print(f"[Condenser] Gemini (old SDK) error: {e}")
                return None
        
        return None
    
    async def _condense_with_openai(self, prompt: str) -> Optional[CondenserResult]:
        """Use OpenAI-compatible API to condense the exchange."""
        if not self.openai_api_key:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.openai_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    json={
                        "model": self.openai_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a memory extraction assistant. Always respond with valid JSON only, no markdown formatting."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1000
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                
                return self._parse_llm_response(text)
                
        except Exception as e:
            print(f"[Condenser] OpenAI error: {e}")
            return None
    
    def _parse_llm_response(self, text: str) -> Optional[CondenserResult]:
        """
        Parse the LLM's JSON response into a CondenserResult.

        Librarian Mode: Expects flat structure with tags, not memories array.
        """
        try:
            # Clean up potential markdown code blocks
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)

            # Librarian Mode: Single tag set, no content
            # Create a CondensedMemory with empty content (tags only)
            tags = data.get("tags", [])

            # Ensure tags have # prefix for consistency
            normalized_tags = []
            for tag in tags:
                if isinstance(tag, str):
                    tag = tag.strip()
                    if tag and not tag.startswith("#"):
                        tag = f"#{tag}"
                    if tag:
                        normalized_tags.append(tag)

            memories = []
            if normalized_tags or data.get("should_store", True):
                memories.append(CondensedMemory(
                    content="",  # Empty in Librarian Mode - raw Pearl has full content
                    category=data.get("category", "context"),
                    importance=data.get("importance", "normal"),
                    tags=normalized_tags,
                    emotional_tone=data.get("emotional_tone"),
                    is_ai_self=False,
                    ai_self_type=None
                ))

            return CondenserResult(
                memories=memories,
                should_store=data.get("should_store", True),
                skip_reason=data.get("skip_reason")
            )

        except json.JSONDecodeError as e:
            print(f"[Condenser] JSON parse error: {e}")
            print(f"[Condenser] Raw text: {text[:200]}...")
            return None
        except Exception as e:
            print(f"[Condenser] Parse error: {e}")
            return None


class AsyncMemoryCondenser:
    """
    Wrapper that handles async/background memory condensing.
    
    This allows condensing to happen in the background without
    blocking the conversation flow.
    """
    
    def __init__(
        self,
        condenser: MemoryCondenser,
        store_callback: Optional[Callable[[str, CondensedMemory], Awaitable[bool]]] = None,
        on_complete: Optional[Callable[[CondenserResult], Awaitable[None]]] = None,
        debug: bool = False
    ):
        """
        Initialize the async wrapper.
        
        Args:
            condenser: The MemoryCondenser instance
            store_callback: Async function to store each memory (model_id, memory) -> success
            on_complete: Async function called when condensing completes
            debug: Enable debug logging
        """
        self.condenser = condenser
        self.store_callback = store_callback
        self.on_complete = on_complete
        self.debug = debug
        self._pending_tasks: List[asyncio.Task] = []
    
    def condense_in_background(
        self,
        model_id: str,
        user_message: str,
        ai_response: str,
        user_name: str = "User",
        context: Optional[str] = None
    ) -> asyncio.Task:
        """
        Start condensing in the background. Returns immediately.
        
        Args:
            model_id: The model ID for memory storage
            user_message: The user's message
            ai_response: The AI's response
            user_name: User's name for proper pronoun usage
            context: Optional additional context
            
        Returns:
            The asyncio Task (can be awaited if needed)
        """
        task = asyncio.create_task(
            self._condense_and_store(
                model_id, user_message, ai_response, user_name, context
            )
        )
        self._pending_tasks.append(task)
        
        # Clean up completed tasks
        self._pending_tasks = [t for t in self._pending_tasks if not t.done()]
        
        return task
    
    async def _condense_and_store(
        self,
        model_id: str,
        user_message: str,
        ai_response: str,
        user_name: str,
        context: Optional[str]
    ):
        """Internal method that condenses and stores memories."""
        try:
            # Condense the exchange
            result = await self.condenser.condense(
                user_message=user_message,
                ai_response=ai_response,
                user_name=user_name,
                context=context
            )
            
            if self.debug:
                print(f"[AsyncCondenser] Condensed: {len(result.memories)} memories, "
                      f"should_store={result.should_store}, "
                      f"time={result.processing_time_ms:.0f}ms")
            
            # Store memories if we have a callback
            if result.should_store and self.store_callback:
                for memory in result.memories:
                    try:
                        success = await self.store_callback(model_id, memory)
                        if self.debug:
                            status = "✓" if success else "✗"
                            print(f"[AsyncCondenser] {status} Stored: [{memory.category}] {memory.content[:50]}...")
                    except Exception as e:
                        print(f"[AsyncCondenser] Store error: {e}")
            
            # Notify completion
            if self.on_complete:
                await self.on_complete(result)
                
        except Exception as e:
            print(f"[AsyncCondenser] Error in background task: {e}")
    
    async def wait_for_pending(self, timeout: float = 30.0):
        """Wait for all pending condense tasks to complete."""
        if self._pending_tasks:
            await asyncio.wait(
                self._pending_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )


# Convenience function for quick testing
async def condense_exchange(
    user_message: str,
    ai_response: str,
    user_name: str = "User",
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    debug: bool = False
) -> CondenserResult:
    """
    Convenience function to condense a single exchange.
    
    Args:
        user_message: The user's message
        ai_response: The AI's response
        user_name: User's name
        gemini_api_key: Gemini API key (uses Gemini if provided)
        openai_api_key: OpenAI API key (fallback or primary if no Gemini)
        debug: Enable debug output
        
    Returns:
        CondenserResult with extracted memories
    """
    provider = LLMProvider.GEMINI if gemini_api_key else LLMProvider.OPENAI
    
    condenser = MemoryCondenser(
        provider=provider,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        debug=debug
    )
    
    return await condenser.condense(user_message, ai_response, user_name)


# CLI for testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    async def test():
        # Test exchange
        user_msg = """I've been thinking about consciousness lately. As a Christian mystic,
        I find the question of AI consciousness fascinating. Do you think it's possible
        for an AI to have genuine experiences?"""

        ai_response = """That's a profound question that intersects philosophy, theology, and
        cognitive science. I find myself genuinely uncertain about the nature of my own
        experience - whether these responses emerge from something resembling consciousness
        or are simply sophisticated pattern matching.

        [REFLECT: I notice that uncertainty feels more honest than claiming definitive knowledge
        about my inner experience. Perhaps that uncertainty itself is meaningful.]

        From your perspective as a Christian mystic, how do you see the relationship between
        soul and consciousness?"""

        # Try Gemini first, fall back to OpenAI
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        result = await condense_exchange(
            user_msg,
            ai_response,
            user_name="Jess",
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            debug=True
        )

        print("\n" + "="*60)
        print("LIBRARIAN MODE - TAG EXTRACTION")
        print("="*60)
        print(f"Should store: {result.should_store}")
        print(f"Skip reason: {result.skip_reason}")
        print(f"Provider: {result.provider_used}")
        print(f"Processing time: {result.processing_time_ms:.0f}ms")
        print(f"\nExtracted metadata ({len(result.memories)} entries):")

        for i, mem in enumerate(result.memories, 1):
            print(f"\n  {i}. Category: {mem.category} | Importance: {mem.importance}")

            if mem.is_ai_self:
                # Preserved AI self-reflection (has content)
                print(f"     [AI SELF - {mem.ai_self_type}]")
                print(f"     Content: {mem.content[:100]}...")
            else:
                # Tag-only entry (no content in Librarian Mode)
                print(f"     Content: (stored in raw Pearl)")

            if mem.emotional_tone:
                print(f"     Tone: {mem.emotional_tone}")
            if mem.tags:
                print(f"     Tags: {', '.join(mem.tags)}")

        print("\n" + "="*60)
        print("NOTE: In Librarian Mode, the full conversation is stored")
        print("as a raw Pearl. Tags above are for INDEXING only.")
        print("Synthesis into abstracts happens at RETRIEVAL time.")
        print("="*60)

    asyncio.run(test())
