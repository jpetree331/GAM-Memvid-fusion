"""
Import OpenWebUI conversation exports to Railway-hosted GAM-Memvid server.

Usage:
    python import_to_railway.py conversation.json --model-id my-model --user-name Jess
    python import_to_railway.py conversation.json --model-id my-model --dry-run
"""
import json
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import httpx

# Railway server URL (can be overridden with --local flag)
RAILWAY_URL = "https://gam-memvid-fusion-production.up.railway.app"
LOCAL_URL = "http://localhost:8100"


@dataclass
class MessagePair:
    """A user message + AI response pair."""
    user_message: str
    ai_response: str
    timestamp: Optional[datetime] = None
    has_reasoning: bool = False


def parse_openwebui_export(file_path: Path) -> tuple[list[MessagePair], dict]:
    """
    Parse OpenWebUI's tree-structured conversation export.

    Returns:
        Tuple of (list of MessagePairs, metadata dict)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle array or single object
    if isinstance(data, list):
        thread = data[0]
    else:
        thread = data

    # Extract metadata
    metadata = {
        "thread_id": thread.get("id"),
        "user_id": thread.get("user_id"),
        "title": thread.get("title") or thread.get("chat", {}).get("title"),
    }

    # Get the chat data
    chat = thread.get("chat", thread)

    # Get model from models array
    models = chat.get("models", [])
    if models:
        metadata["model_id"] = models[0]

    # Get messages - could be in history.messages (tree) or just messages (flat)
    history = chat.get("history", {})
    messages_dict = history.get("messages", {})

    if not messages_dict:
        # Try flat format
        messages_dict = {m.get("id", str(i)): m for i, m in enumerate(chat.get("messages", []))}

    # Build ordered message list by following parent-child relationships
    ordered_messages = []

    def traverse(msg_id):
        """Traverse the message tree depth-first."""
        if msg_id not in messages_dict:
            return
        msg = messages_dict[msg_id]
        ordered_messages.append(msg)
        for child_id in msg.get("childrenIds", []):
            traverse(child_id)

    # Find root messages (no parent) and traverse
    roots = [msg for msg in messages_dict.values() if msg.get("parentId") is None]
    for root in roots:
        traverse(root.get("id"))

    # Fallback: sort by timestamp if no tree structure
    if not ordered_messages:
        ordered_messages = sorted(
            messages_dict.values(),
            key=lambda m: m.get("timestamp", 0)
        )

    # Get thread-level timestamp as fallback
    thread_ts = None
    for ts_field in ["created_at", "timestamp", "date", "createdAt"]:
        ts_val = chat.get(ts_field) or thread.get(ts_field)
        if ts_val:
            try:
                if isinstance(ts_val, (int, float)):
                    if ts_val > 1e12:
                        thread_ts = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc)
                    else:
                        thread_ts = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                else:
                    thread_ts = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
                print(f"[DEBUG] Found thread-level timestamp in {ts_field}: {thread_ts}")
                break
            except (ValueError, TypeError, OSError):
                pass

    # Pair user messages with AI responses
    pairs = []
    current_user_msg = None
    current_user_ts = None

    for msg in ordered_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if not content or not content.strip():
            continue

        # Handle multimodal content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
                else:
                    text_parts.append(str(part))
            content = " ".join(text_parts)

        # Parse timestamp
        ts = None
        ts_val = msg.get("timestamp")
        if ts_val:
            try:
                if isinstance(ts_val, (int, float)):
                    if ts_val > 1e12:  # Milliseconds
                        ts = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc)
                    else:
                        ts = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                else:
                    ts = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
            except (ValueError, TypeError, OSError) as e:
                print(f"[DEBUG] Failed to parse timestamp {ts_val!r}: {e}")
        else:
            # Try alternative timestamp fields
            for alt_field in ["created_at", "date", "createdAt"]:
                alt_val = msg.get(alt_field)
                if alt_val:
                    try:
                        if isinstance(alt_val, (int, float)):
                            if alt_val > 1e12:
                                ts = datetime.fromtimestamp(alt_val / 1000, tz=timezone.utc)
                            else:
                                ts = datetime.fromtimestamp(alt_val, tz=timezone.utc)
                        else:
                            ts = datetime.fromisoformat(str(alt_val).replace("Z", "+00:00"))
                        print(f"[DEBUG] Found timestamp in {alt_field}: {ts}")
                        break
                    except (ValueError, TypeError, OSError):
                        pass

        if role == "user":
            current_user_msg = content
            current_user_ts = ts
        elif role == "assistant" and current_user_msg:
            # Check if response contains reasoning blocks
            has_reasoning = bool(re.search(
                r'<details[^>]*type="reasoning"[^>]*>',
                content
            ))

            if content.strip():
                # Use message timestamp, fallback to thread timestamp, then None
                final_ts = current_user_ts or ts or thread_ts
                if not final_ts:
                    print(f"[DEBUG] WARNING: No timestamp found for message pair {len(pairs)+1}")
                pairs.append(MessagePair(
                    user_message=current_user_msg,
                    ai_response=content,  # Keep full response with reasoning
                    timestamp=final_ts,
                    has_reasoning=has_reasoning
                ))
            current_user_msg = None
            current_user_ts = None

    return pairs, metadata


def upload_to_railway(
    pairs: list[MessagePair],
    model_id: str,
    user_name: str = "User",
    dry_run: bool = False,
    use_local: bool = False
) -> dict:
    """
    Upload message pairs to Railway server or local server.

    Args:
        pairs: List of user/AI message pairs
        model_id: Target model ID
        user_name: User's name
        dry_run: If True, don't actually upload
        use_local: If True, use localhost:8100 instead of Railway URL

    Returns:
        Upload statistics
    """
    server_url = LOCAL_URL if use_local else RAILWAY_URL
    
    stats = {
        "total": len(pairs),
        "uploaded": 0,
        "failed": 0,
        "errors": []
    }

    if dry_run:
        print(f"\n[DRY RUN] Would upload {len(pairs)} pairs to: {server_url}")
        print(f"Model ID: {model_id}")
        print(f"User name: {user_name}")
        reasoning_count = sum(1 for p in pairs if p.has_reasoning)
        print(f"Pairs with reasoning blocks: {reasoning_count}/{len(pairs)}")
        for i, pair in enumerate(pairs[:3]):
            tags = ["imported"] + (["reasoning"] if pair.has_reasoning else [])
            ts_str = pair.timestamp.isoformat() if pair.timestamp else "None"
            print(f"\n--- Pair {i+1} (tags: {tags}) ---")
            print(f"Timestamp: {ts_str}")
            print(f"User: {pair.user_message[:80]}...")
            print(f"AI: {pair.ai_response[:80]}...")
        if len(pairs) > 3:
            print(f"\n... and {len(pairs) - 3} more")
        return stats

    print(f"\nUploading {len(pairs)} pairs to: {server_url}")
    print(f"Model ID: {model_id}")
    print(f"User name: {user_name}\n")

    with httpx.Client(timeout=60.0) as client:
        for i, pair in enumerate(pairs):
            try:
                # Build tags
                tags = ["imported"]
                if pair.has_reasoning:
                    tags.append("reasoning")

                # Include original timestamp if available
                payload = {
                    "model_id": model_id,
                    "user_message": pair.user_message,
                    "ai_response": pair.ai_response,
                    "user_name": user_name,
                    "tags": tags
                }
                if pair.timestamp:
                    payload["created_at"] = pair.timestamp.isoformat()
                    print(f"[DEBUG] Pair {i+1}: Including timestamp {payload['created_at']}")
                else:
                    print(f"[DEBUG] Pair {i+1}: WARNING - No timestamp available, will use import time")

                response = client.post(
                    f"{server_url}/memory/add",
                    json=payload
                )

                if response.status_code == 200:
                    stats["uploaded"] += 1
                    data = response.json()
                    pearl_id = data.get("pearl_id", "?")
                    r_tag = " [reasoning]" if pair.has_reasoning else ""
                    ts_info = f" @{pair.timestamp.strftime('%Y-%m-%d')}" if pair.timestamp else ""
                    print(f"  [{i+1}/{len(pairs)}] OK - {pearl_id}{r_tag}{ts_info}")
                else:
                    stats["failed"] += 1
                    stats["errors"].append(f"Pair {i+1}: HTTP {response.status_code}")
                    print(f"  [{i+1}/{len(pairs)}] FAILED - HTTP {response.status_code}")

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"Pair {i+1}: {str(e)[:50]}")
                print(f"  [{i+1}/{len(pairs)}] ERROR - {str(e)[:50]}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Import OpenWebUI conversations to Railway GAM-Memvid server"
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to OpenWebUI JSON export"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Target model ID (e.g., 'qwen-235b-a22bth-origf2')"
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default="User",
        help="User's name for memories (default: User)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and show what would be uploaded, but don't upload"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use localhost:8100 instead of Railway URL (for local testing)"
    )

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1

    # Parse
    print(f"Parsing: {args.file}")
    pairs, metadata = parse_openwebui_export(args.file)

    print(f"\nThread: {metadata.get('title', 'Unknown')}")
    print(f"Original model: {metadata.get('model_id', 'Unknown')}")
    print(f"Message pairs: {len(pairs)}")

    # Upload
    stats = upload_to_railway(
        pairs=pairs,
        model_id=args.model_id,
        user_name=args.user_name,
        dry_run=args.dry_run,
        use_local=args.local
    )

    # Summary
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Complete!")
    print(f"  Total: {stats['total']}")
    print(f"  Uploaded: {stats['uploaded']}")
    print(f"  Failed: {stats['failed']}")

    if stats["errors"]:
        print(f"\nErrors:")
        for err in stats["errors"][:5]:
            print(f"  - {err}")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
