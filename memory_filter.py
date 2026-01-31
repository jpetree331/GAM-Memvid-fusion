"""
Intelligent Memory Storage Filter - Decides what's worth remembering.

Hybrid approach:
1. Fast pattern-based filter (catches obvious noise)
2. LLM-based filter for borderline cases (accurate but costs API)
3. Configurable exclusion rules

Based on proven Mem0 exclusion patterns.
"""
import re
from typing import Optional, Tuple
from dataclasses import dataclass
import httpx

from config import config


@dataclass
class FilterResult:
    """Result of memory filtering decision."""
    should_store: bool
    reason: str
    confidence: float  # 0.0 to 1.0
    extracted_fact: Optional[str] = None  # Condensed version if applicable


# =============================================================================
# Pattern-Based Fast Filters (Free, Instant)
# =============================================================================

# Messages that are definitely NOT worth storing
SKIP_PATTERNS = [
    # Greetings and pleasantries
    r"^(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening|night))[\s!.,]*$",
    r"^(bye|goodbye|see\s*you|later|take\s*care|have\s*a\s*(good|great|nice)\s*(day|one))[\s!.,]*$",
    r"^(thanks|thank\s*you|thx|ty|cheers|much\s*appreciated)[\s!.,]*$",
    r"^(you'?re\s*welcome|no\s*problem|np|anytime|my\s*pleasure)[\s!.,]*$",
    r"^(ok|okay|sure|yes|no|yep|nope|yeah|nah|alright|got\s*it|understood)[\s!.,]*$",
    
    # Generic pleasantries (from user's Mem0 rules)
    r"i\s*hope\s*(you\s*are|you're)\s*doing\s*well",
    r"hope\s*(this|that)\s*(helps|finds\s*you)",
    r"how\s*are\s*you(\s*doing)?[\s?!.,]*$",
    r"i'?m\s*(doing\s*)?(good|fine|well|great|okay)[\s!.,]*$",
    
    # Sycophantic praise (not tied to specific patterns)
    r"^(you\s*are|you're)\s*(amazing|awesome|great|wonderful|the\s*best)[\s!.,]*$",
    r"^(wow|omg|oh\s*my\s*god)[\s!.,]*$",
    r"^(that'?s?\s*)?(so\s*)?(cool|neat|nice|great|awesome|amazing)[\s!.,]*$",
    
    # Very short messages (likely not informative)
    r"^.{1,10}$",  # Less than 10 chars
    
    # Just punctuation or emoji-like
    r"^[\s\W]+$",
]

# AI response patterns to exclude
AI_SKIP_PATTERNS = [
    # Safety disclaimers
    r"as\s*an?\s*(ai|artificial\s*intelligence|language\s*model)",
    r"i'?m\s*(just\s*)?an?\s*(ai|assistant|language\s*model)",
    r"i\s*don'?t\s*have\s*(personal\s*)?(feelings|emotions|opinions|experiences)",
    r"i\s*can'?t\s*(actually|really)\s*(feel|experience|know)",
    
    # Apologies for limitations
    r"i\s*(apologize|'?m\s*sorry)\s*(for|that)\s*(any|my)\s*(confusion|limitations?|mistakes?)",
    r"i\s*don'?t\s*have\s*(access\s*to|information\s*about)\s*(the\s*)?(internet|real-?time|current)",
    
    # Generic helper phrases
    r"^(sure|of\s*course|absolutely|certainly|definitely)[,!.\s]*(i'?d\s*be\s*happy\s*to|let\s*me)",
    r"is\s*there\s*anything\s*else\s*(i\s*can|you'?d\s*like)",
    r"feel\s*free\s*to\s*ask",
    r"let\s*me\s*know\s*if\s*(you\s*)?(need|have|want)",
    
    # Low-confidence hedging (unless confirmed later)
    r"^(i\s*think\s*)?(maybe|might|perhaps|possibly|could\s*be)",
    r"i'?m\s*not\s*(entirely\s*)?(sure|certain)",
]

# Sensitive info patterns (NEVER store)
SENSITIVE_PATTERNS = [
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
    r"\bpassword\s*[:=]\s*\S+",  # Passwords
    r"\b(api[_-]?key|secret[_-]?key|auth[_-]?token)\s*[:=]\s*\S+",  # API keys
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b.*\bpassword\b",  # Email + password combo
]

# Patterns that SHOULD be stored (override skips)
ALWAYS_STORE_PATTERNS = [
    r"remember\s*(this|that)",
    r"(don'?t\s*)?forget\s*(this|that|about)",
    r"(this\s*is\s*)?(important|crucial|critical)",
    r"(my|i)\s*(name|birthday|preference|favorite)",
    r"i\s*(always|never|usually|prefer|hate|love|like|dislike)",
    r"(please\s*)?(note|keep\s*in\s*mind)",
    r"for\s*(future|later)\s*reference",
]


class MemoryStorageFilter:
    """
    Intelligent filter to decide what conversations are worth storing.
    """
    
    def __init__(
        self,
        custom_exclusions: Optional[list[str]] = None,
        use_llm_filter: bool = True,
        llm_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        min_message_length: int = 15,
        min_combined_length: int = 50
    ):
        self.custom_exclusions = custom_exclusions or []
        self.use_llm_filter = use_llm_filter
        self.llm_model = llm_model or config.GAM_MODEL_NAME
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.min_message_length = min_message_length
        self.min_combined_length = min_combined_length
        
        # Compile patterns for efficiency
        self._skip_patterns = [re.compile(p, re.IGNORECASE) for p in SKIP_PATTERNS]
        self._ai_skip_patterns = [re.compile(p, re.IGNORECASE) for p in AI_SKIP_PATTERNS]
        self._sensitive_patterns = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_PATTERNS]
        self._always_store_patterns = [re.compile(p, re.IGNORECASE) for p in ALWAYS_STORE_PATTERNS]
        self._custom_patterns = [re.compile(p, re.IGNORECASE) for p in self.custom_exclusions]
    
    def _check_sensitive(self, text: str) -> bool:
        """Check if text contains sensitive information. Returns True if sensitive."""
        for pattern in self._sensitive_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _check_always_store(self, user_msg: str, ai_msg: str) -> bool:
        """Check if exchange should always be stored."""
        combined = f"{user_msg} {ai_msg}"
        for pattern in self._always_store_patterns:
            if pattern.search(combined):
                return True
        return False
    
    def _fast_filter_user_message(self, text: str) -> Tuple[bool, str]:
        """
        Fast pattern-based filter for user messages.
        Returns (should_skip, reason).
        """
        # Length check
        if len(text.strip()) < self.min_message_length:
            return True, "Message too short"
        
        # Check skip patterns
        for pattern in self._skip_patterns:
            if pattern.search(text):
                return True, "Matches skip pattern (pleasantry/noise)"
        
        # Check custom exclusions
        for pattern in self._custom_patterns:
            if pattern.search(text):
                return True, "Matches custom exclusion"
        
        return False, ""
    
    def _fast_filter_ai_response(self, text: str) -> Tuple[bool, str]:
        """
        Fast pattern-based filter for AI responses.
        Returns (should_skip, reason).
        """
        # Check for AI disclaimers and hedging
        disclaimer_count = 0
        for pattern in self._ai_skip_patterns:
            if pattern.search(text):
                disclaimer_count += 1
        
        # If response is mostly disclaimers, skip
        if disclaimer_count >= 2:
            return True, "AI response is mostly disclaimers/hedging"
        
        return False, ""
    
    def fast_filter(
        self, 
        user_msg: str, 
        ai_msg: str
    ) -> FilterResult:
        """
        Fast pattern-based filtering. No API calls.
        Returns FilterResult with decision.
        """
        # Check for sensitive info first (absolute block)
        if self._check_sensitive(user_msg) or self._check_sensitive(ai_msg):
            return FilterResult(
                should_store=False,
                reason="Contains sensitive information",
                confidence=1.0
            )
        
        # Check always-store patterns (high priority)
        if self._check_always_store(user_msg, ai_msg):
            return FilterResult(
                should_store=True,
                reason="Contains importance marker",
                confidence=0.95
            )
        
        # Combined length check
        combined_length = len(user_msg.strip()) + len(ai_msg.strip())
        if combined_length < self.min_combined_length:
            return FilterResult(
                should_store=False,
                reason="Exchange too short to be meaningful",
                confidence=0.8
            )
        
        # Filter user message
        skip_user, reason_user = self._fast_filter_user_message(user_msg)
        if skip_user:
            return FilterResult(
                should_store=False,
                reason=f"User message: {reason_user}",
                confidence=0.85
            )
        
        # Filter AI response
        skip_ai, reason_ai = self._fast_filter_ai_response(ai_msg)
        if skip_ai:
            return FilterResult(
                should_store=False,
                reason=f"AI response: {reason_ai}",
                confidence=0.7
            )
        
        # Passed fast filters - likely worth storing or needs LLM check
        return FilterResult(
            should_store=True,
            reason="Passed fast filters",
            confidence=0.6  # Medium confidence - LLM can refine
        )
    
    async def llm_filter(
        self,
        user_msg: str,
        ai_msg: str,
        user_name: str = "User"
    ) -> FilterResult:
        """
        LLM-based filtering for borderline cases.
        More accurate but costs API calls.
        """
        prompt = f"""Analyze this conversation exchange and decide if it should be stored in long-term memory.

USER ({user_name}): {user_msg[:500]}

AI RESPONSE: {ai_msg[:500]}

---

EXCLUDE (don't store) if:
- Standard pleasantries, greetings, or goodbyes
- Generic safety disclaimers or "as an AI..." reminders
- Apologies for limitations
- Superficial small talk with no new information
- Sycophantic praise or generic compliments not tied to specific context
- Low-confidence speculation ("might", "maybe", "I'm not sure")
- The exchange is trivial and forgettable

INCLUDE (store) if:
- User shares personal preferences, facts, or experiences
- User mentions names, relationships, jobs, or interests
- There's a meaningful Q&A with useful information
- AI expresses genuine opinions or reflections
- The exchange builds relationship context
- User explicitly asks to remember something

Respond in JSON:
{{
  "store": true/false,
  "reason": "brief explanation",
  "extracted_fact": "if storing, write a concise first-person fact (from AI perspective) or null"
}}"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {"role": "system", "content": "You are a memory curation assistant. Always respond with valid JSON. Be selective - only memorable exchanges should be stored."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 200
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON
                import json
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                data = json.loads(content.strip())
                
                return FilterResult(
                    should_store=data.get("store", False),
                    reason=data.get("reason", "LLM decision"),
                    confidence=0.9,
                    extracted_fact=data.get("extracted_fact")
                )
                
        except Exception as e:
            print(f"[MemoryFilter] LLM filter error: {e}")
            # Fall back to storing (better to have noise than miss important info)
            return FilterResult(
                should_store=True,
                reason=f"LLM filter failed, defaulting to store: {str(e)[:50]}",
                confidence=0.5
            )
    
    async def should_store(
        self,
        user_msg: str,
        ai_msg: str,
        user_name: str = "User"
    ) -> FilterResult:
        """
        Hybrid filtering: Fast filter first, LLM for borderline cases.
        """
        # Step 1: Fast filter
        fast_result = self.fast_filter(user_msg, ai_msg)
        
        # If high confidence decision, use it
        if fast_result.confidence >= 0.8:
            return fast_result
        
        # Step 2: LLM filter for borderline cases
        if self.use_llm_filter and fast_result.confidence < 0.8:
            llm_result = await self.llm_filter(user_msg, ai_msg, user_name)
            return llm_result
        
        # Default to fast filter result
        return fast_result


# =============================================================================
# Singleton Instance
# =============================================================================

_filter_instance: Optional[MemoryStorageFilter] = None

def get_memory_filter(
    custom_exclusions: Optional[list[str]] = None,
    use_llm_filter: bool = True
) -> MemoryStorageFilter:
    """Get or create the memory filter instance."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = MemoryStorageFilter(
            custom_exclusions=custom_exclusions,
            use_llm_filter=use_llm_filter
        )
    return _filter_instance
