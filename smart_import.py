"""
Smart Import System - Intelligently parses and categorizes conversation threads.

Instead of storing an entire thread as one blob, this:
1. Splits into individual exchange pairs
2. Uses LLM to categorize each exchange
3. Extracts key facts/memories (not raw conversation)
4. Preserves timestamps
5. Detects AI_Self reflections automatically
"""
import json
import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx

from config import config
from memory_organization import MemoryCategory, ImportanceLevel, AISelfType


@dataclass
class ParsedExchange:
    """A single user-assistant exchange from a conversation."""
    user_message: str
    assistant_message: str
    timestamp: Optional[str] = None
    exchange_index: int = 0


@dataclass 
class ExtractedMemory:
    """A memory extracted from an exchange."""
    content: str
    category: str
    importance: str = ImportanceLevel.NORMAL.value
    tags: list[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    is_ai_self: bool = False
    ai_self_type: Optional[str] = None
    source_exchange: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


class SmartImporter:
    """
    Intelligently imports conversation threads with proper categorization.
    """
    
    # Prompt for the LLM to analyze exchanges
    ANALYSIS_PROMPT = """Analyze this conversation exchange and extract memories.

USER MESSAGE:
{user_message}

AI RESPONSE:
{assistant_message}

TIMESTAMP: {timestamp}
USER'S NAME: {user_name}

---

Extract distinct memories from this exchange. For each memory:
1. Write it as a concise fact from the AI's perspective
2. Categorize it appropriately
3. Determine if it's about the user OR about the AI's own views/growth

Categories available:
- preference: User's likes, dislikes, preferences
- fact: Factual information about user (name, job, family, etc.)
- event: Something that happened
- task: Tasks, projects, goals the user mentioned
- relationship: Shared jokes, relationship dynamics, emotional moments
- context: Background/situational info
- skill: User's abilities or knowledge areas
- theology: Religious/spiritual discussions
- science: Scientific topics discussed
- ai_theory: AI philosophy, ethics, consciousness discussions
- ai_self: The AI's own opinions, reflections, growth (IMPORTANT: only for AI's expressed views)

For AI_SELF memories, also specify the type:
- opinion: AI expressed a viewpoint
- reflection: AI reflected on its nature/experience
- growth: AI's perspective evolved
- worldview: How AI sees the world
- identity: Core trait AI developed

Respond in JSON format:
{{
  "memories": [
    {{
      "content": "concise fact written from AI perspective using {user_name}'s name",
      "category": "category_name",
      "importance": "normal|high|core",
      "tags": ["relevant", "tags"],
      "is_ai_self": false
    }},
    {{
      "content": "I find myself drawn to...",
      "category": "ai_self",
      "importance": "normal",
      "tags": ["reflection"],
      "is_ai_self": true,
      "ai_self_type": "reflection"
    }}
  ],
  "skip_reason": null
}}

If the exchange is just small talk with nothing memorable, return:
{{
  "memories": [],
  "skip_reason": "casual greeting/small talk"
}}

Remember:
- Use "{user_name}" (not "User" or "I") when referring to the human
- Use "I" only for the AI's own thoughts/actions
- Extract FACTS, not raw conversation
- Be concise but preserve important details"""

    def __init__(
        self,
        user_name: str = "User",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 5
    ):
        self.user_name = user_name
        self.model_name = model_name or config.GAM_MODEL_NAME
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.batch_size = batch_size  # Process this many exchanges at once
        
    def parse_thread_to_exchanges(self, thread_data: dict) -> list[ParsedExchange]:
        """
        Parse a thread into individual user-assistant exchange pairs.
        Preserves timestamps for each exchange.
        """
        messages = thread_data.get("messages", [])
        if not messages:
            return []
        
        exchanges = []
        exchange_idx = 0
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # Look for user message
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                user_ts = msg.get("timestamp")
                
                # Look for following assistant message
                assistant_msg = ""
                assistant_ts = None
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    assistant_msg = messages[i + 1].get("content", "")
                    assistant_ts = messages[i + 1].get("timestamp")
                    i += 1
                
                if user_msg and assistant_msg:
                    # Use user timestamp, fall back to assistant timestamp
                    timestamp = user_ts or assistant_ts
                    if timestamp and isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp).isoformat()
                    
                    exchanges.append(ParsedExchange(
                        user_message=user_msg,
                        assistant_message=assistant_msg,
                        timestamp=timestamp,
                        exchange_index=exchange_idx
                    ))
                    exchange_idx += 1
            
            i += 1
        
        return exchanges
    
    async def analyze_exchange(
        self, 
        exchange: ParsedExchange,
        client: httpx.AsyncClient
    ) -> list[ExtractedMemory]:
        """
        Use LLM to analyze a single exchange and extract memories.
        """
        prompt = self.ANALYSIS_PROMPT.format(
            user_message=exchange.user_message[:2000],  # Truncate if too long
            assistant_message=exchange.assistant_message[:2000],
            timestamp=exchange.timestamp or "unknown",
            user_name=self.user_name
        )
        
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a memory extraction assistant. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                },
                timeout=60.0
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content.strip())
            
            memories = []
            for mem_data in data.get("memories", []):
                memories.append(ExtractedMemory(
                    content=mem_data.get("content", ""),
                    category=mem_data.get("category", "context"),
                    importance=mem_data.get("importance", "normal"),
                    tags=mem_data.get("tags", []),
                    timestamp=exchange.timestamp,
                    is_ai_self=mem_data.get("is_ai_self", False),
                    ai_self_type=mem_data.get("ai_self_type"),
                    source_exchange=exchange.exchange_index
                ))
            
            return memories
            
        except json.JSONDecodeError as e:
            print(f"[SmartImport] JSON parse error for exchange {exchange.exchange_index}: {e}")
            return []
        except Exception as e:
            print(f"[SmartImport] Error analyzing exchange {exchange.exchange_index}: {e}")
            return []
    
    async def analyze_exchanges_batch(
        self,
        exchanges: list[ParsedExchange],
        progress_callback: Optional[callable] = None
    ) -> list[ExtractedMemory]:
        """
        Analyze multiple exchanges, processing in batches for efficiency.
        """
        all_memories = []
        
        async with httpx.AsyncClient() as client:
            for i in range(0, len(exchanges), self.batch_size):
                batch = exchanges[i:i + self.batch_size]
                
                # Process batch concurrently
                tasks = [self.analyze_exchange(ex, client) for ex in batch]
                results = await asyncio.gather(*tasks)
                
                for memories in results:
                    all_memories.extend(memories)
                
                if progress_callback:
                    progress_callback(
                        processed=min(i + self.batch_size, len(exchanges)),
                        total=len(exchanges),
                        memories_found=len(all_memories)
                    )
        
        return all_memories
    
    async def import_thread(
        self,
        thread_data: dict,
        target_model_id: str,
        gam_server_url: str = "http://localhost:8100",
        progress_callback: Optional[callable] = None,
        dry_run: bool = False
    ) -> dict:
        """
        Smart import a thread into GAM memory.
        
        Args:
            thread_data: The thread JSON data
            target_model_id: Model to import memories into
            gam_server_url: GAM server URL
            progress_callback: Called with progress updates
            dry_run: If True, analyze but don't store
        
        Returns:
            Import results with statistics
        """
        # Parse thread into exchanges
        exchanges = self.parse_thread_to_exchanges(thread_data)
        
        if not exchanges:
            return {
                "success": False,
                "error": "No valid exchanges found in thread",
                "exchanges_found": 0,
                "memories_extracted": 0
            }
        
        if progress_callback:
            progress_callback(
                status="parsing",
                message=f"Found {len(exchanges)} exchanges to analyze"
            )
        
        # Analyze exchanges
        if progress_callback:
            progress_callback(
                status="analyzing",
                message="Analyzing exchanges with LLM..."
            )
        
        memories = await self.analyze_exchanges_batch(
            exchanges,
            progress_callback=lambda **kw: progress_callback(status="analyzing", **kw) if progress_callback else None
        )
        
        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "exchanges_found": len(exchanges),
                "memories_extracted": len(memories),
                "memories": [m.to_dict() for m in memories],
                "by_category": self._count_by_category(memories),
                "ai_self_count": sum(1 for m in memories if m.is_ai_self)
            }
        
        # Store memories in GAM
        if progress_callback:
            progress_callback(
                status="storing",
                message=f"Storing {len(memories)} memories..."
            )
        
        stored_ids = []
        errors = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for mem in memories:
                try:
                    if mem.is_ai_self:
                        # Store as AI_Self memory
                        response = await client.post(
                            f"{gam_server_url}/memory/ai-self/{target_model_id}/add",
                            json={
                                "content": mem.content,
                                "ai_self_type": mem.ai_self_type or "reflection",
                                "importance": mem.importance,
                                "tags": mem.tags + ["imported", f"exchange_{mem.source_exchange}"],
                                "created_at": mem.timestamp  # Preserve original timestamp
                            }
                        )
                    else:
                        # Store as regular memory
                        response = await client.post(
                            f"{gam_server_url}/memory/add",
                            json={
                                "model_id": target_model_id,
                                "content": mem.content,
                                "category": mem.category,
                                "importance": mem.importance,
                                "tags": mem.tags + ["imported", f"exchange_{mem.source_exchange}"],
                                "timestamp": mem.timestamp
                            }
                        )
                    
                    response.raise_for_status()
                    data = response.json()
                    stored_ids.append(data.get("memory_id"))
                    
                except Exception as e:
                    errors.append(f"Failed to store memory: {str(e)[:100]}")
        
        return {
            "success": len(errors) == 0,
            "exchanges_found": len(exchanges),
            "memories_extracted": len(memories),
            "memories_stored": len(stored_ids),
            "stored_ids": stored_ids,
            "errors": errors,
            "by_category": self._count_by_category(memories),
            "ai_self_count": sum(1 for m in memories if m.is_ai_self),
            "thread_title": thread_data.get("title", "Unknown"),
            "timestamp_range": self._get_timestamp_range(exchanges)
        }
    
    def _count_by_category(self, memories: list[ExtractedMemory]) -> dict:
        """Count memories by category."""
        counts = {}
        for mem in memories:
            cat = mem.category
            counts[cat] = counts.get(cat, 0) + 1
        return counts
    
    def _get_timestamp_range(self, exchanges: list[ParsedExchange]) -> dict:
        """Get the timestamp range of exchanges."""
        timestamps = [e.timestamp for e in exchanges if e.timestamp]
        if not timestamps:
            return {"start": None, "end": None}
        return {
            "start": min(timestamps),
            "end": max(timestamps)
        }


async def smart_import_thread(
    thread_path: str,
    target_model_id: str,
    user_name: str = "User",
    gam_server_url: str = "http://localhost:8100",
    dry_run: bool = False
) -> dict:
    """
    Convenience function to smart-import a thread from a file.
    """
    # Load thread
    with open(thread_path, 'r', encoding='utf-8') as f:
        thread_data = json.load(f)
    
    # Handle nested formats
    if "chat" in thread_data:
        thread_data = thread_data["chat"]
    
    # Create importer
    importer = SmartImporter(user_name=user_name)
    
    # Run import
    def progress(status="", message="", **kwargs):
        print(f"[{status.upper()}] {message}")
        if "processed" in kwargs:
            print(f"  Progress: {kwargs['processed']}/{kwargs['total']} exchanges, {kwargs.get('memories_found', 0)} memories found")
    
    result = await importer.import_thread(
        thread_data=thread_data,
        target_model_id=target_model_id,
        gam_server_url=gam_server_url,
        progress_callback=progress,
        dry_run=dry_run
    )
    
    return result


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart import a conversation thread with automatic categorization"
    )
    parser.add_argument("file", type=str, help="Path to thread JSON file")
    parser.add_argument("--model", "-m", type=str, required=True, help="Target model ID")
    parser.add_argument("--user", "-u", type=str, default="User", help="User's name")
    parser.add_argument("--server", "-s", type=str, default="http://localhost:8100", help="GAM server URL")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without storing")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Smart Thread Import")
    print(f"{'='*60}\n")
    
    result = asyncio.run(smart_import_thread(
        thread_path=args.file,
        target_model_id=args.model,
        user_name=args.user,
        gam_server_url=args.server,
        dry_run=args.dry_run
    ))
    
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"Thread: {result.get('thread_title', 'Unknown')}")
    print(f"Exchanges found: {result.get('exchanges_found', 0)}")
    print(f"Memories extracted: {result.get('memories_extracted', 0)}")
    print(f"AI_Self memories: {result.get('ai_self_count', 0)}")
    
    if result.get("timestamp_range"):
        tr = result["timestamp_range"]
        print(f"Time range: {tr.get('start')} to {tr.get('end')}")
    
    print(f"\nBy category:")
    for cat, count in result.get("by_category", {}).items():
        print(f"  {cat}: {count}")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Memories that would be stored:")
        for mem in result.get("memories", [])[:10]:
            print(f"  [{mem['category']}] {mem['content'][:80]}...")
        if len(result.get("memories", [])) > 10:
            print(f"  ... and {len(result['memories']) - 10} more")
    else:
        print(f"\nMemories stored: {result.get('memories_stored', 0)}")
        if result.get("errors"):
            print(f"Errors: {len(result['errors'])}")
            for err in result["errors"][:5]:
                print(f"  - {err}")
