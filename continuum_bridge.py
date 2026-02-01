"""
Continuum Bridge - OpenWebUI API client for journal/scheduler.

Used by the /continuum/* routes to:
- Resolve thread_id -> model_id from OpenWebUI
- Post prompts to OpenWebUI and capture AI responses (journal entries)
- List chats/threads from OpenWebUI
- Cache thread_id -> model_id (5 min TTL) to avoid repeated API calls
"""
from typing import Optional, List
import time
import httpx

# Cache: thread_id -> (model_id, expiry_ts). TTL 5 minutes.
_thread_model_cache: dict = {}
_CACHE_TTL_SEC = 300


def _cache_get(thread_id: str) -> Optional[str]:
    now = time.time()
    entry = _thread_model_cache.get(thread_id)
    if entry is None:
        return None
    model_id, expiry = entry
    if now >= expiry:
        del _thread_model_cache[thread_id]
        return None
    return model_id


def _cache_set(thread_id: str, model_id: str) -> None:
    _thread_model_cache[thread_id] = (model_id, time.time() + _CACHE_TTL_SEC)


def list_chats(
    base_url: str,
    api_key: str,
    skip: int = 0,
    limit: int = 50,
    timeout: float = 15.0
) -> List[dict]:
    """
    List OpenWebUI chats (threads).

    GET /api/v1/chats with optional skip/limit.
    Returns list of chat objects (id, title, model, etc.).
    """
    if not base_url or not api_key:
        return []
    url = f"{base_url.rstrip('/')}/api/v1/chats"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(
                url,
                params={"skip": skip, "limit": limit},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            # Some APIs return { "chats": [...] } or { "data": [...] }
            return data.get("chats") or data.get("data") or []
    except Exception as e:
        print(f"[ContinuumBridge] list_chats error: {e}")
        return []


def get_model_id_for_thread(
    base_url: str,
    api_key: str,
    thread_id: str,
    timeout: float = 15.0,
    use_cache: bool = True
) -> Optional[str]:
    """
    Get the model_id (vault identifier) for an OpenWebUI chat/thread.

    Uses in-memory cache (5 min TTL) when use_cache=True to avoid repeated API calls.
    Calls GET /api/v1/chats/{thread_id} and extracts model from the response.
    OpenWebUI may return "model" or "models" (list); we use the first model.
    """
    if use_cache:
        cached = _cache_get(thread_id)
        if cached is not None:
            return cached
    if not base_url or not api_key:
        return None
    url = f"{base_url.rstrip('/')}/api/v1/chats/{thread_id}"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
            data = r.json()
            # OpenWebUI chat object may have "model" or "models"
            model = data.get("model")
            if model:
                if use_cache:
                    _cache_set(thread_id, model)
                return model
            models = data.get("models")
            if isinstance(models, list) and models:
                model = models[0]
                if use_cache:
                    _cache_set(thread_id, model)
                return model
            if isinstance(models, str):
                if use_cache:
                    _cache_set(thread_id, models)
                return models
            return None
    except Exception as e:
        print(f"[ContinuumBridge] get_model_id_for_thread error: {e}")
        return None


def post_message_to_thread(
    base_url: str,
    api_key: str,
    thread_id: str,
    content: str,
    timeout: float = 60.0
) -> str:
    """
    Post a user message to an OpenWebUI chat and return the AI response text.

    POST /api/v1/chats/{thread_id}/messages with body { "messages": [{ "role": "user", "content": content }] }.
    Parses the response for the assistant message (structure may vary by OpenWebUI version).
    """
    if not base_url or not api_key:
        return "Error: OpenWebUI not configured (CONTINUUM_OPENWEBUI_BASE_URL / API_KEY)."
    url = f"{base_url.rstrip('/')}/api/v1/chats/{thread_id}/messages"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": [{"role": "user", "content": content}],
                },
            )
            r.raise_for_status()
            data = r.json()
            # Try common response shapes
            if isinstance(data, str):
                return data
            text = data.get("content") or data.get("text") or data.get("message", {}).get("content")
            if text:
                return text
            # Some APIs return choices[].message.content
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {})
                if isinstance(msg, dict) and msg.get("content"):
                    return msg["content"]
            return "No response content from OpenWebUI."
    except httpx.HTTPStatusError as e:
        return f"OpenWebUI HTTP error: {e.response.status_code} {e.response.text[:200]}"
    except Exception as e:
        return f"OpenWebUI error: {e}"
