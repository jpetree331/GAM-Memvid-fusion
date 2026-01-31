"""
OpenWebUI Filter for GAM-Memvid (Librarian Architecture)

This filter integrates the Librarian memory system with OpenWebUI:
- inlet(): Retrieves synthesized context and injects it before the AI responds
- outlet(): Stores the user+AI exchange as a Pearl after each turn

AUTOMATIC MODEL ISOLATION:
Memory is automatically tied to the model ID from the request.
Each OpenWebUI model gets its own isolated memory vault.

INSTALLATION:
1. Deploy the GAM-Memvid server (server.py) to Railway or run locally
2. In OpenWebUI: Admin → Functions → Create Function
3. Paste this entire code
4. Save and enable on your models

CONFIGURATION:
Set MEMORY_SERVER_URL in the Valves to your server URL:
- Local: http://localhost:8100
- Railway: https://your-app.up.railway.app
"""

import os
import json
import httpx
from typing import Optional, List, Callable, Awaitable
from pydantic import BaseModel, Field


class Filter:
    """
    GAM-Memvid Memory Filter for OpenWebUI.

    Provides automatic memory retrieval and storage for AI models.
    Uses the Librarian architecture: full conversations stored as Pearls,
    synthesized into context at retrieval time.
    """

    class Valves(BaseModel):
        """Configuration for the memory filter."""
        MEMORY_SERVER_URL: str = Field(
            default="http://localhost:8100",
            description="URL of the GAM-Memvid server"
        )
        ENABLE_MEMORY_RETRIEVAL: bool = Field(
            default=True,
            description="Retrieve and inject memory context before AI responds"
        )
        ENABLE_MEMORY_STORAGE: bool = Field(
            default=True,
            description="Store conversations in memory after each turn"
        )
        MAX_PEARLS: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Maximum Pearls to retrieve for context"
        )
        MAX_CONTEXT_WORDS: int = Field(
            default=400,
            ge=100,
            le=2000,
            description="Target word count for synthesized context"
        )
        USER_NAME: str = Field(
            default="User",
            description="Your name for personalized memories (e.g., 'Jess')"
        )
        SHOW_STATUS: bool = Field(
            default=True,
            description="Show memory status messages in chat"
        )
        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable detailed logging"
        )
        REQUEST_TIMEOUT: float = Field(
            default=30.0,
            description="HTTP request timeout in seconds"
        )
        priority: int = Field(
            default=0,
            description="Filter priority (lower = runs earlier)"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _get_model_id(self, body: dict) -> str:
        """
        Extract model ID from request body.

        The model ID determines which memory vault is used.
        Each model gets isolated memory storage.
        """
        model_id = body.get("model", "default")
        if model_id:
            model_id = model_id.strip()
        return model_id or "default"

    def _get_last_user_message(self, messages: list) -> str:
        """Extract the most recent user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Handle both string and list content (multimodal)
                if isinstance(content, list):
                    # Extract text from multimodal content
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return " ".join(text_parts)
                return content
        return ""

    def _get_last_assistant_message(self, messages: list) -> str:
        """Extract the most recent assistant message."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> dict:
        """
        INLET: Called BEFORE the message is sent to the model.

        Retrieves relevant memories and injects synthesized context
        into the conversation as a system message.
        """
        if not self.valves.ENABLE_MEMORY_RETRIEVAL:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        model_id = self._get_model_id(body)
        user_name = self.valves.USER_NAME
        if __user__ and __user__.get("name"):
            user_name = __user__.get("name")

        # Get the current user query
        query = self._get_last_user_message(messages)
        if not query:
            return body

        if self.valves.DEBUG_MODE:
            print(f"[GAM Filter] inlet() - model: {model_id}, query: {query[:50]}...")

        # Show status if enabled
        if self.valves.SHOW_STATUS and __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Searching memories ({model_id})...", "done": False}
            })

        # Fetch synthesized context from the server
        try:
            async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{self.valves.MEMORY_SERVER_URL}/memory/context",
                    json={
                        "model_id": model_id,
                        "query": query,
                        "user_name": user_name,
                        "limit": self.valves.MAX_PEARLS,
                        "max_words": self.valves.MAX_CONTEXT_WORDS
                    }
                )
                response.raise_for_status()
                data = response.json()

                context = data.get("context", "")
                num_pearls = data.get("num_pearls", 0)
                has_memories = data.get("has_memories", False)

        except httpx.TimeoutException:
            if self.valves.DEBUG_MODE:
                print(f"[GAM Filter] Timeout fetching context")
            context = ""
            num_pearls = 0
            has_memories = False
        except Exception as e:
            if self.valves.DEBUG_MODE:
                print(f"[GAM Filter] Error fetching context: {e}")
            context = ""
            num_pearls = 0
            has_memories = False

        # Inject context if we have memories
        if context and has_memories:
            # Build the context injection message
            context_message = {
                "role": "system",
                "content": f"""## Relevant Memory Context

The following is synthesized from your past conversations with {user_name}. Use this to ground your response when helpful, but don't explicitly mention "checking memories" unless asked.

{context}

---
Remember: This context comes from real past exchanges. Reference it naturally."""
            }

            # Find existing system message or prepend
            system_idx = None
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    system_idx = i
                    break

            if system_idx is not None:
                # Append to existing system message
                messages[system_idx]["content"] = (
                    messages[system_idx]["content"] + "\n\n" + context_message["content"]
                )
            else:
                # Prepend new system message
                messages.insert(0, context_message)

            body["messages"] = messages

            if self.valves.SHOW_STATUS and __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Found {num_pearls} relevant memories",
                        "done": True
                    }
                })
        else:
            if self.valves.SHOW_STATUS and __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"No relevant memories for {model_id}",
                        "done": True
                    }
                })

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> dict:
        """
        OUTLET: Called AFTER the model generates a response.

        Stores the user message + AI response as a Pearl in the memory vault.
        The full conversation is preserved without truncation.
        """
        if not self.valves.ENABLE_MEMORY_STORAGE:
            return body

        messages = body.get("messages", [])
        if len(messages) < 2:
            return body

        model_id = self._get_model_id(body)
        user_name = self.valves.USER_NAME
        if __user__ and __user__.get("name"):
            user_name = __user__.get("name")

        # Get the last user message and AI response
        user_message = ""
        ai_response = ""

        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not ai_response:
                content = msg.get("content", "")
                if isinstance(content, str):
                    ai_response = content
            elif msg.get("role") == "user" and not user_message:
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    user_message = " ".join(text_parts)
                else:
                    user_message = content

            if user_message and ai_response:
                break

        # Skip if we don't have both parts
        if not user_message or not ai_response:
            if self.valves.DEBUG_MODE:
                print(f"[GAM Filter] Skipping storage: missing user ({bool(user_message)}) or AI ({bool(ai_response)})")
            return body

        # Skip very short exchanges (likely greetings)
        if len(user_message.strip()) < 10 and len(ai_response.strip()) < 50:
            if self.valves.DEBUG_MODE:
                print(f"[GAM Filter] Skipping short exchange")
            return body

        if self.valves.DEBUG_MODE:
            print(f"[GAM Filter] outlet() - storing for {model_id}")
            print(f"[GAM Filter] User: {user_message[:50]}...")
            print(f"[GAM Filter] AI: {ai_response[:50]}...")

        # Store the Pearl
        try:
            async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{self.valves.MEMORY_SERVER_URL}/memory/add",
                    json={
                        "model_id": model_id,
                        "user_message": user_message,
                        "ai_response": ai_response,
                        "tags": [],  # Could add smart tagging here
                        "user_name": user_name
                    }
                )
                response.raise_for_status()
                data = response.json()

                if self.valves.SHOW_STATUS and __event_emitter__:
                    word_count = data.get("word_count", 0)
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"Memory saved ({word_count} words)",
                            "done": True
                        }
                    })

        except httpx.TimeoutException:
            if self.valves.DEBUG_MODE:
                print(f"[GAM Filter] Timeout storing memory")
        except Exception as e:
            if self.valves.DEBUG_MODE:
                print(f"[GAM Filter] Error storing memory: {e}")
            # Fail silently - don't block the chat

        return body


# =============================================================================
# Alternative: Standalone functions for use in OpenWebUI Functions feature
# =============================================================================

async def get_memory_context(
    model_id: str,
    query: str,
    user_name: str = "User",
    limit: int = 5,
    max_words: int = 400,
    server_url: str = "http://localhost:8100"
) -> str:
    """
    Retrieve synthesized memory context.

    Can be used as a standalone function in OpenWebUI.

    Args:
        model_id: The AI model/persona ID
        query: Current user message or search query
        user_name: User's display name
        limit: Max Pearls to retrieve
        max_words: Target context word count
        server_url: Memory server URL

    Returns:
        Synthesized context string
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{server_url}/memory/context",
                json={
                    "model_id": model_id,
                    "query": query,
                    "user_name": user_name,
                    "limit": limit,
                    "max_words": max_words
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("context", "")
    except Exception as e:
        print(f"[get_memory_context] Error: {e}")
        return ""


async def store_memory(
    model_id: str,
    user_message: str,
    ai_response: str,
    tags: Optional[List[str]] = None,
    user_name: str = "User",
    server_url: str = "http://localhost:8100"
) -> Optional[str]:
    """
    Store a conversation exchange as a Pearl.

    Can be used as a standalone function in OpenWebUI.

    Args:
        model_id: The AI model/persona ID
        user_message: The user's message
        ai_response: The AI's response
        tags: Optional tags for categorization
        user_name: User's display name
        server_url: Memory server URL

    Returns:
        Pearl ID if successful, None otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{server_url}/memory/add",
                json={
                    "model_id": model_id,
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "tags": tags or [],
                    "user_name": user_name
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("pearl_id")
    except Exception as e:
        print(f"[store_memory] Error: {e}")
        return None
