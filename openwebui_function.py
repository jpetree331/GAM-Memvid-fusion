"""
OpenWebUI Functions for GAM-Memvid (Librarian Architecture)

This provides Functions (Tools) that can be used in OpenWebUI to:
1. Search and retrieve memory context
2. Store conversation exchanges
3. Manage memories (delete, list, etc.)

These can be used as:
- Standalone tools the AI can call
- Building blocks for custom pipes
- Manual memory operations

INSTALLATION:
1. Deploy the GAM-Memvid server (server.py)
2. In OpenWebUI: Admin → Functions → Create Function
3. Paste this code (or specific tools you want)
4. Configure the MEMORY_SERVER_URL valve

Note: For automatic memory (every turn), use the Filter instead.
These Functions are for explicit/manual memory operations.
"""

import os
import httpx
from typing import Optional, List
from pydantic import BaseModel, Field


class Tools:
    """
    GAM-Memvid Memory Tools for OpenWebUI.

    Provides explicit memory operations that the AI or user can invoke.
    Unlike the Filter (automatic), these are called on-demand.
    """

    class Valves(BaseModel):
        """Configuration for memory tools."""
        MEMORY_SERVER_URL: str = Field(
            default="http://localhost:8100",
            description="URL of the GAM-Memvid server"
        )
        DEFAULT_MODEL_ID: str = Field(
            default="",
            description="Default model ID (leave empty to require explicit model_id)"
        )
        REQUEST_TIMEOUT: float = Field(
            default=30.0,
            description="HTTP request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def search_memories(
        self,
        query: str,
        model_id: str = "",
        limit: int = 5,
        __user__: Optional[dict] = None
    ) -> str:
        """
        Search through memories for relevant past conversations.

        Use this when you need to recall something specific from past conversations,
        or when the user asks "do you remember..." or similar.

        Args:
            query: What to search for (topic, keywords, question)
            model_id: Which model's memories to search (uses default if empty)
            limit: Maximum number of results (1-20)

        Returns:
            Synthesized context from relevant past conversations
        """
        effective_model_id = model_id or self.valves.DEFAULT_MODEL_ID
        if not effective_model_id:
            return "Error: No model_id specified and no default configured."

        user_name = "User"
        if __user__ and __user__.get("name"):
            user_name = __user__.get("name")

        try:
            async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{self.valves.MEMORY_SERVER_URL}/memory/context",
                    json={
                        "model_id": effective_model_id,
                        "query": query,
                        "user_name": user_name,
                        "limit": min(max(limit, 1), 20),
                        "max_words": 600
                    }
                )
                response.raise_for_status()
                data = response.json()

                context = data.get("context", "")
                num_pearls = data.get("num_pearls", 0)

                if not context:
                    return f"No relevant memories found for: {query}"

                return f"Found {num_pearls} relevant memories:\n\n{context}"

        except httpx.TimeoutException:
            return "Error: Memory server request timed out."
        except Exception as e:
            return f"Error searching memories: {str(e)[:100]}"

    async def remember_this(
        self,
        user_message: str,
        ai_response: str,
        model_id: str = "",
        tags: str = "",
        __user__: Optional[dict] = None
    ) -> str:
        """
        Store a specific exchange in memory.

        Use this when explicitly asked to remember something, or when
        storing an important piece of information for later.

        Args:
            user_message: What the user said
            ai_response: What you (the AI) responded
            model_id: Which model's memory to store in (uses default if empty)
            tags: Comma-separated tags (e.g., "important,preference,name")

        Returns:
            Confirmation message with the memory ID
        """
        effective_model_id = model_id or self.valves.DEFAULT_MODEL_ID
        if not effective_model_id:
            return "Error: No model_id specified and no default configured."

        user_name = "User"
        if __user__ and __user__.get("name"):
            user_name = __user__.get("name")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        try:
            async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{self.valves.MEMORY_SERVER_URL}/memory/add",
                    json={
                        "model_id": effective_model_id,
                        "user_message": user_message,
                        "ai_response": ai_response,
                        "tags": tag_list,
                        "user_name": user_name
                    }
                )
                response.raise_for_status()
                data = response.json()

                pearl_id = data.get("pearl_id", "unknown")
                word_count = data.get("word_count", 0)

                return f"Memory stored successfully!\nID: {pearl_id}\nWords: {word_count}"

        except httpx.TimeoutException:
            return "Error: Memory server request timed out."
        except Exception as e:
            return f"Error storing memory: {str(e)[:100]}"

    async def forget_memory(
        self,
        pearl_id: str,
        model_id: str = "",
        reason: str = "",
        __user__: Optional[dict] = None
    ) -> str:
        """
        Delete (soft-delete) a specific memory.

        Use this when asked to forget something specific, or to remove
        incorrect information from memory.

        Args:
            pearl_id: The ID of the memory to delete
            model_id: Which model's memory (uses default if empty)
            reason: Optional reason for deletion

        Returns:
            Confirmation of deletion
        """
        effective_model_id = model_id or self.valves.DEFAULT_MODEL_ID
        if not effective_model_id:
            return "Error: No model_id specified and no default configured."

        try:
            async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{self.valves.MEMORY_SERVER_URL}/memory/{effective_model_id}/delete",
                    json={
                        "pearl_id": pearl_id,
                        "reason": reason or "User requested"
                    }
                )
                response.raise_for_status()
                data = response.json()

                if data.get("success"):
                    return f"Memory {pearl_id} has been deleted."
                else:
                    return f"Failed to delete memory: {data.get('message', 'Unknown error')}"

        except httpx.TimeoutException:
            return "Error: Memory server request timed out."
        except Exception as e:
            return f"Error deleting memory: {str(e)[:100]}"

    async def list_recent_memories(
        self,
        model_id: str = "",
        limit: int = 5,
        __user__: Optional[dict] = None
    ) -> str:
        """
        List recent memories for a model.

        Use this to show what's been remembered recently, or to find
        memory IDs for deletion.

        Args:
            model_id: Which model's memories (uses default if empty)
            limit: How many to show (1-20)

        Returns:
            List of recent memories with IDs
        """
        effective_model_id = model_id or self.valves.DEFAULT_MODEL_ID
        if not effective_model_id:
            return "Error: No model_id specified and no default configured."

        try:
            async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                response = await client.get(
                    f"{self.valves.MEMORY_SERVER_URL}/memory/{effective_model_id}/recent",
                    params={"limit": min(max(limit, 1), 20)}
                )
                response.raise_for_status()
                data = response.json()

                memories = data.get("memories", [])
                if not memories:
                    return f"No memories found for {effective_model_id}"

                lines = [f"Recent memories for {effective_model_id}:\n"]
                for mem in memories:
                    mem_id = mem.get("id", "unknown")
                    content = mem.get("content", "")[:100]
                    created = mem.get("created_at", "")[:10]
                    lines.append(f"- [{mem_id}] ({created}): {content}...")

                return "\n".join(lines)

        except httpx.TimeoutException:
            return "Error: Memory server request timed out."
        except Exception as e:
            return f"Error listing memories: {str(e)[:100]}"

    async def memory_stats(
        self,
        model_id: str = "",
        __user__: Optional[dict] = None
    ) -> str:
        """
        Get statistics about a model's memory vault.

        Shows total memories, categories, and other useful stats.

        Args:
            model_id: Which model's stats (uses default if empty)

        Returns:
            Memory statistics summary
        """
        effective_model_id = model_id or self.valves.DEFAULT_MODEL_ID
        if not effective_model_id:
            return "Error: No model_id specified and no default configured."

        try:
            async with httpx.AsyncClient(timeout=self.valves.REQUEST_TIMEOUT) as client:
                response = await client.get(
                    f"{self.valves.MEMORY_SERVER_URL}/memory/{effective_model_id}/stats"
                )
                response.raise_for_status()
                stats = response.json()

                lines = [
                    f"Memory Stats for {effective_model_id}:",
                    f"- Total Pearls: {stats.get('pearl_count', 0)}",
                    f"- Active: {stats.get('active_count', 0)}",
                    f"- Deleted: {stats.get('deleted_count', 0)}",
                    f"- Core memories: {stats.get('core_pearls', 0)}",
                    f"- Last update: {stats.get('last_update', 'unknown')}"
                ]

                by_category = stats.get("by_category", {})
                if by_category:
                    lines.append("\nBy Category:")
                    for cat, count in by_category.items():
                        lines.append(f"  - {cat}: {count}")

                return "\n".join(lines)

        except httpx.TimeoutException:
            return "Error: Memory server request timed out."
        except Exception as e:
            return f"Error getting stats: {str(e)[:100]}"


# =============================================================================
# Pipe Alternative (for use as a processing pipe instead of tools)
# =============================================================================

class Pipe:
    """
    GAM-Memvid Memory Pipe for OpenWebUI.

    This can be used as a Pipe (message processor) instead of or
    alongside the Filter. It provides the same functionality but
    in Pipe format.

    For most use cases, the Filter is recommended as it integrates
    more seamlessly with OpenWebUI's message flow.
    """

    class Valves(BaseModel):
        """Configuration for the memory pipe."""
        MEMORY_SERVER_URL: str = Field(
            default="http://localhost:8100",
            description="URL of the GAM-Memvid server"
        )
        ENABLE_RETRIEVAL: bool = Field(
            default=True,
            description="Retrieve memories before AI responds"
        )
        ENABLE_STORAGE: bool = Field(
            default=True,
            description="Store exchanges after AI responds"
        )
        MAX_PEARLS: int = Field(
            default=5,
            description="Maximum Pearls to retrieve"
        )
        MAX_CONTEXT_WORDS: int = Field(
            default=400,
            description="Target context word count"
        )
        USER_NAME: str = Field(
            default="User",
            description="Your name for personalized memories"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _get_model_id(self, body: dict) -> str:
        """Extract model ID from request."""
        model_id = body.get("model", "default")
        return model_id.strip() if model_id else "default"

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None
    ) -> dict:
        """
        Process messages through the memory system.

        This is called before the model processes the request.
        It retrieves relevant context and injects it.
        """
        if not self.valves.ENABLE_RETRIEVAL:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        model_id = self._get_model_id(body)
        user_name = self.valves.USER_NAME
        if __user__ and __user__.get("name"):
            user_name = __user__.get("name")

        # Get last user message
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                if isinstance(query, list):
                    # Handle multimodal
                    query = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in query
                    )
                break

        if not query:
            return body

        # Fetch context
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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

        except Exception as e:
            print(f"[GAM Pipe] Error fetching context: {e}")
            return body

        # Inject context
        if context:
            context_msg = {
                "role": "system",
                "content": f"## Memory Context\n\n{context}"
            }

            # Find system message or prepend
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i]["content"] += f"\n\n{context_msg['content']}"
                    break
            else:
                messages.insert(0, context_msg)

            body["messages"] = messages

        return body

    async def on_completion(
        self,
        body: dict,
        __user__: Optional[dict] = None
    ):
        """
        Called after model generates a response.

        Stores the exchange in memory.
        """
        if not self.valves.ENABLE_STORAGE:
            return

        messages = body.get("messages", [])
        if len(messages) < 2:
            return

        model_id = self._get_model_id(body)
        user_name = self.valves.USER_NAME
        if __user__ and __user__.get("name"):
            user_name = __user__.get("name")

        # Extract user message and AI response
        user_message = ""
        ai_response = ""

        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not ai_response:
                ai_response = msg.get("content", "")
            elif msg.get("role") == "user" and not user_message:
                content = msg.get("content", "")
                if isinstance(content, list):
                    user_message = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                else:
                    user_message = content

            if user_message and ai_response:
                break

        if not user_message or not ai_response:
            return

        # Store the exchange
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{self.valves.MEMORY_SERVER_URL}/memory/add",
                    json={
                        "model_id": model_id,
                        "user_message": user_message,
                        "ai_response": ai_response,
                        "user_name": user_name
                    }
                )
        except Exception as e:
            print(f"[GAM Pipe] Error storing memory: {e}")
