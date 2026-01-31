"""
Utility for importing historical conversations into GAM.

Supports:
- JSON conversation exports (from OpenWebUI, ChatGPT, Anthropic/Claude, etc.)
- Mem0 exports
- Custom formats via adapters
- Single thread imports
- Bulk imports with proper timestamp preservation
"""
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from memory_manager import memory_manager


@dataclass
class ConversationMessage:
    """A parsed conversation message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[datetime] = None


@dataclass 
class Conversation:
    """A parsed conversation."""
    messages: list[ConversationMessage]
    model_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None  # Conversation-level timestamp
    title: Optional[str] = None  # Conversation title if available
    tags: list[str] = field(default_factory=list)  # Any tags/categories


def parse_openwebui_export(file_path: Path) -> list[Conversation]:
    """
    Parse OpenWebUI conversation export.
    
    Expected format:
    {
        "conversations": [
            {
                "id": "...",
                "model": "model-name",
                "messages": [
                    {"role": "user", "content": "...", "timestamp": "..."},
                    {"role": "assistant", "content": "...", "timestamp": "..."}
                ],
                "created_at": "..."
            }
        ]
    }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    
    items = data.get("conversations", data.get("chats", [data] if "messages" in data else []))
    
    for item in items:
        messages = []
        for msg in item.get("messages", []):
            timestamp = None
            if "timestamp" in msg:
                try:
                    timestamp = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            
            messages.append(ConversationMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                timestamp=timestamp
            ))
        
        # Get conversation timestamp
        conv_timestamp = None
        for field in ["created_at", "timestamp", "date"]:
            if field in item:
                try:
                    conv_timestamp = datetime.fromisoformat(
                        item[field].replace("Z", "+00:00")
                    )
                    break
                except (ValueError, TypeError):
                    pass
        
        conversations.append(Conversation(
            messages=messages,
            model_id=item.get("model", item.get("model_id", "unknown")),
            user_id=item.get("user_id"),
            session_id=item.get("id", item.get("session_id")),
            timestamp=conv_timestamp
        ))
    
    return conversations


def parse_chatgpt_export(file_path: Path) -> list[Conversation]:
    """
    Parse ChatGPT conversation export.
    
    Expected format:
    [
        {
            "title": "...",
            "create_time": 1234567890.0,
            "mapping": {
                "node-id": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["..."]},
                        "create_time": 1234567890.0
                    }
                }
            }
        }
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    
    for item in data:
        messages = []
        
        # Extract messages from ChatGPT's nested format
        mapping = item.get("mapping", {})
        message_nodes = []
        
        for node_id, node in mapping.items():
            if "message" in node and node["message"]:
                msg = node["message"]
                author = msg.get("author", {})
                content = msg.get("content", {})
                
                role = author.get("role", "")
                if role not in ["user", "assistant"]:
                    continue
                
                text = ""
                parts = content.get("parts", [])
                if parts:
                    text = parts[0] if isinstance(parts[0], str) else str(parts[0])
                
                timestamp = None
                if "create_time" in msg and msg["create_time"]:
                    try:
                        timestamp = datetime.fromtimestamp(msg["create_time"])
                    except (ValueError, TypeError, OSError):
                        pass
                
                message_nodes.append((
                    msg.get("create_time", 0) or 0,
                    ConversationMessage(role=role, content=text, timestamp=timestamp)
                ))
        
        # Sort by timestamp
        message_nodes.sort(key=lambda x: x[0])
        messages = [m[1] for m in message_nodes]
        
        # Conversation timestamp
        conv_timestamp = None
        if "create_time" in item and item["create_time"]:
            try:
                conv_timestamp = datetime.fromtimestamp(item["create_time"])
            except (ValueError, TypeError, OSError):
                pass
        
        if messages:
            conversations.append(Conversation(
                messages=messages,
                model_id="chatgpt-import",  # ChatGPT doesn't specify model per conversation
                timestamp=conv_timestamp
            ))
    
    return conversations


def parse_mem0_export(file_path: Path) -> list[Conversation]:
    """
    Parse Mem0 memory export.
    
    This converts Mem0's memory format back into conversation-like structures.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    
    memories = data.get("memories", data if isinstance(data, list) else [])
    
    for mem in memories:
        content = mem.get("memory", mem.get("content", ""))
        metadata = mem.get("metadata", {})
        
        timestamp = None
        for ts_field in ["created_at", "timestamp", "updated_at"]:
            if ts_field in mem:
                try:
                    timestamp = datetime.fromisoformat(
                        mem[ts_field].replace("Z", "+00:00")
                    )
                    break
                except (ValueError, TypeError):
                    pass
        
        # Mem0 memories are typically extracted facts, treat as assistant notes
        conversations.append(Conversation(
            messages=[ConversationMessage(
                role="assistant",
                content=f"[Memory] {content}",
                timestamp=timestamp
            )],
            model_id=metadata.get("model_id", metadata.get("agent_id", "mem0-import")),
            user_id=mem.get("user_id"),
            timestamp=timestamp
        ))
    
    return conversations


def parse_anthropic_export(file_path: Path) -> list[Conversation]:
    """
    Parse Anthropic/Claude conversation export.
    
    Anthropic exports typically have format:
    {
        "conversations": [
            {
                "uuid": "...",
                "name": "Conversation Title",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T11:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "...",
                        "text": "...",
                        "sender": "human" | "assistant",
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                ]
            }
        ]
    }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    
    # Handle different Anthropic export structures
    items = data.get("conversations", data if isinstance(data, list) else [data])
    
    for item in items:
        messages = []
        
        # Get messages from various possible field names
        msg_list = item.get("chat_messages", item.get("messages", []))
        
        for msg in msg_list:
            # Anthropic uses "human"/"assistant", convert to standard
            sender = msg.get("sender", msg.get("role", ""))
            if sender == "human":
                role = "user"
            elif sender == "assistant":
                role = "assistant"
            else:
                role = sender
            
            content = msg.get("text", msg.get("content", ""))
            
            timestamp = None
            if "created_at" in msg:
                try:
                    timestamp = datetime.fromisoformat(
                        msg["created_at"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass
            
            if content and role in ["user", "assistant"]:
                messages.append(ConversationMessage(
                    role=role,
                    content=content,
                    timestamp=timestamp
                ))
        
        # Get conversation timestamp
        conv_timestamp = None
        for ts_field in ["created_at", "timestamp", "updated_at"]:
            if ts_field in item:
                try:
                    conv_timestamp = datetime.fromisoformat(
                        item[ts_field].replace("Z", "+00:00")
                    )
                    break
                except (ValueError, TypeError):
                    pass
        
        if messages:
            conversations.append(Conversation(
                messages=messages,
                model_id=item.get("model", "claude-import"),
                session_id=item.get("uuid", item.get("id")),
                timestamp=conv_timestamp,
                title=item.get("name", item.get("title"))
            ))
    
    return conversations


def parse_single_thread(data: dict, user_name: str = "User") -> Conversation:
    """
    Parse a single conversation thread from JSON data.
    
    This is useful for importing individual threads via API.
    
    Expected format:
    {
        "id": "thread-id",
        "model": "model-name",
        "title": "Conversation Title",
        "created_at": "2024-01-15T10:30:00Z",
        "messages": [
            {"role": "user", "content": "...", "timestamp": "..."},
            {"role": "assistant", "content": "...", "timestamp": "..."}
        ]
    }
    """
    messages = []
    
    for msg in data.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        timestamp = None
        for ts_field in ["timestamp", "created_at", "date"]:
            if ts_field in msg:
                try:
                    ts_val = msg[ts_field]
                    if isinstance(ts_val, (int, float)):
                        timestamp = datetime.fromtimestamp(ts_val)
                    else:
                        timestamp = datetime.fromisoformat(
                            str(ts_val).replace("Z", "+00:00")
                        )
                    break
                except (ValueError, TypeError, OSError):
                    pass
        
        if content:
            messages.append(ConversationMessage(
                role=role,
                content=content,
                timestamp=timestamp
            ))
    
    # Get conversation timestamp
    conv_timestamp = None
    for ts_field in ["created_at", "timestamp", "date", "updated_at"]:
        if ts_field in data:
            try:
                ts_val = data[ts_field]
                if isinstance(ts_val, (int, float)):
                    conv_timestamp = datetime.fromtimestamp(ts_val)
                else:
                    conv_timestamp = datetime.fromisoformat(
                        str(ts_val).replace("Z", "+00:00")
                    )
                break
            except (ValueError, TypeError, OSError):
                pass
    
    return Conversation(
        messages=messages,
        model_id=data.get("model", data.get("model_id", "imported")),
        session_id=data.get("id", data.get("uuid", data.get("session_id"))),
        timestamp=conv_timestamp,
        title=data.get("title", data.get("name")),
        tags=data.get("tags", [])
    )


def import_conversations(
    conversations: list[Conversation],
    target_model_id: Optional[str] = None,
    user_name: str = "User",
    dry_run: bool = False,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None
) -> dict:
    """
    Import conversations into GAM memory stores.
    
    Args:
        conversations: Parsed conversations to import
        target_model_id: Override the model_id for all conversations
        user_name: Name to use for the user in memories (e.g., "Jess")
        dry_run: If True, don't actually import, just report what would happen
        category: Optional category for all imported memories
        tags: Optional tags to add to all imported memories
    
    Returns:
        Import statistics
    """
    stats = {
        "total_conversations": len(conversations),
        "total_messages": 0,
        "by_model": {},
        "errors": [],
        "imported_ids": []
    }
    
    for conv in conversations:
        model_id = target_model_id or conv.model_id
        stats["total_messages"] += len(conv.messages)
        
        if model_id not in stats["by_model"]:
            stats["by_model"][model_id] = {"conversations": 0, "messages": 0}
        stats["by_model"][model_id]["conversations"] += 1
        stats["by_model"][model_id]["messages"] += len(conv.messages)
        
        if dry_run:
            continue
        
        try:
            store = memory_manager.get_store(model_id)
            
            # Build conversation text with proper pronoun perspective
            parts = []
            
            # Add title if available
            if conv.title:
                parts.append(f"[Conversation: {conv.title}]")
            
            for msg in conv.messages:
                # Use user's name for proper perspective
                if msg.role == "user":
                    role_label = user_name
                else:
                    role_label = "I (AI)"
                
                ts = msg.timestamp or conv.timestamp
                if ts:
                    parts.append(f"[{ts.isoformat()}] {role_label}: {msg.content}")
                else:
                    parts.append(f"{role_label}: {msg.content}")
            
            full_text = "\n".join(parts)
            
            # Combine tags
            all_tags = list(tags or []) + conv.tags
            if conv.title:
                # Add title words as tags for searchability
                title_tags = [w.lower() for w in conv.title.split() if len(w) > 3]
                all_tags.extend(title_tags[:5])  # Limit to 5 title tags
            
            memory_id = store.add_memory(
                content=full_text,
                user_id=conv.user_id,
                session_id=conv.session_id,
                timestamp=conv.timestamp,
                metadata={
                    "type": "imported",
                    "message_count": len(conv.messages),
                    "title": conv.title,
                    "original_session_id": conv.session_id
                },
                category=category,
                tags=all_tags if all_tags else None
            )
            
            stats["imported_ids"].append(memory_id)
        
        except Exception as e:
            stats["errors"].append(f"Error importing to {model_id}: {str(e)}")
    
    return stats


def import_single_thread(
    thread_data: dict,
    target_model_id: str,
    user_name: str = "User",
    category: Optional[str] = None,
    tags: Optional[list[str]] = None
) -> dict:
    """
    Import a single conversation thread.
    
    Args:
        thread_data: JSON data for the thread
        target_model_id: Model ID to import into
        user_name: Name to use for the user
        category: Optional category
        tags: Optional tags
    
    Returns:
        Import result
    """
    conv = parse_single_thread(thread_data, user_name)
    
    result = import_conversations(
        [conv],
        target_model_id=target_model_id,
        user_name=user_name,
        dry_run=False,
        category=category,
        tags=tags
    )
    
    return {
        "success": len(result["errors"]) == 0,
        "memory_id": result["imported_ids"][0] if result["imported_ids"] else None,
        "messages_imported": result["total_messages"],
        "errors": result["errors"]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import historical conversations into GAM memory stores"
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to conversation export file (JSON)"
    )
    parser.add_argument(
        "--format",
        choices=["openwebui", "chatgpt", "anthropic", "mem0", "auto"],
        default="auto",
        help="Export format (default: auto-detect)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="Override model ID for all imported conversations"
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default="User",
        help="User's name for proper pronoun usage (e.g., 'Jess')"
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Category for imported memories (e.g., 'relationship', 'theology')"
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated tags to add to imported memories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't import, just show what would happen"
    )
    
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1
    
    # Parse based on format
    format_type = args.format
    if format_type == "auto":
        # Try to auto-detect
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data_str = str(data)[:2000]
        if "mapping" in data_str:
            format_type = "chatgpt"
        elif "chat_messages" in data_str or '"sender": "human"' in data_str:
            format_type = "anthropic"
        elif "memories" in data or (isinstance(data, list) and data and "memory" in data[0]):
            format_type = "mem0"
        else:
            format_type = "openwebui"
        
        print(f"Auto-detected format: {format_type}")
    
    # Parse
    parsers = {
        "openwebui": parse_openwebui_export,
        "chatgpt": parse_chatgpt_export,
        "anthropic": parse_anthropic_export,
        "mem0": parse_mem0_export
    }
    
    conversations = parsers[format_type](args.file)
    print(f"Parsed {len(conversations)} conversations")
    
    # Parse tags
    tag_list = [t.strip() for t in args.tags.split(",")] if args.tags else None
    
    # Import
    if args.dry_run:
        print("\n[DRY RUN] Would import:")
    
    stats = import_conversations(
        conversations,
        target_model_id=args.model_id,
        user_name=args.user_name,
        dry_run=args.dry_run,
        category=args.category,
        tags=tag_list
    )
    
    print(f"\nTotal conversations: {stats['total_conversations']}")
    print(f"Total messages: {stats['total_messages']}")
    print("\nBy model:")
    for model_id, model_stats in stats["by_model"].items():
        print(f"  {model_id}: {model_stats['conversations']} conversations, {model_stats['messages']} messages")
    
    if stats["errors"]:
        print("\nErrors:")
        for error in stats["errors"]:
            print(f"  - {error}")
    
    if not args.dry_run:
        print("\nImport complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
