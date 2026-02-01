"""
Import OpenWebUI conversation threads into GAM-Memvid (Librarian Architecture).

This handles OpenWebUI's tree-structured exports and stores them as Pearls
(user_message + ai_response pairs) for the Synthesizer.

Usage:
    python import_openwebui_thread.py Examples/conversation.json --model-id my-model --user-name Jess
"""
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from memvid_store import get_store


@dataclass
class MessagePair:
    """A user message + AI response pair (Pearl)."""
    user_message: str
    ai_response: str
    timestamp: Optional[datetime] = None
    user_timestamp: Optional[datetime] = None
    ai_timestamp: Optional[datetime] = None


def parse_openwebui_tree(file_path: Path) -> tuple[list[MessagePair], dict]:
    """
    Parse OpenWebUI's tree-structured conversation export.

    Returns:
        Tuple of (list of MessagePairs, metadata dict)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle array or single object
    if isinstance(data, list):
        thread = data[0]  # Take first thread
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

    # Find root messages (no parent)
    roots = [msg for msg in messages_dict.values() if msg.get("parentId") is None]

    def traverse(msg_id):
        """Traverse the message tree depth-first."""
        if msg_id not in messages_dict:
            return
        msg = messages_dict[msg_id]
        ordered_messages.append(msg)
        for child_id in msg.get("childrenIds", []):
            traverse(child_id)

    # Start from roots
    for root in roots:
        traverse(root.get("id"))

    # If no tree structure found, just use all messages sorted by timestamp
    if not ordered_messages:
        ordered_messages = sorted(
            messages_dict.values(),
            key=lambda m: m.get("timestamp", 0)
        )

    # Pair user messages with AI responses
    pairs = []
    current_user_msg = None
    current_user_ts = None

    for msg in ordered_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Skip empty messages
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
                    # Unix timestamp (might be in milliseconds or seconds)
                    if ts_val > 1e12:  # Milliseconds
                        ts = datetime.fromtimestamp(ts_val / 1000, tz=timezone.utc)
                    else:
                        ts = datetime.fromtimestamp(ts_val, tz=timezone.utc)
                else:
                    ts = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00"))
            except (ValueError, TypeError, OSError):
                pass

        if role == "user":
            # Save user message for pairing
            current_user_msg = content
            current_user_ts = ts
        elif role == "assistant" and current_user_msg:
            # Pair with previous user message
            # Clean up thinking/reasoning blocks if present
            cleaned_response = content

            # Strip <details> reasoning blocks (common in some models)
            import re
            cleaned_response = re.sub(
                r'<details[^>]*type="reasoning"[^>]*>.*?</details>',
                '',
                cleaned_response,
                flags=re.DOTALL
            )
            cleaned_response = cleaned_response.strip()

            if cleaned_response:
                pairs.append(MessagePair(
                    user_message=current_user_msg,
                    ai_response=cleaned_response,
                    timestamp=current_user_ts or ts,
                    user_timestamp=current_user_ts,
                    ai_timestamp=ts
                ))
            current_user_msg = None
            current_user_ts = None

    return pairs, metadata


def import_to_vault(
    pairs: list[MessagePair],
    model_id: str,
    user_name: str = "User",
    dry_run: bool = False,
    thread_title: Optional[str] = None
) -> dict:
    """
    Import message pairs as Pearls into the vault.

    Args:
        pairs: List of user/AI message pairs
        model_id: Target model ID for the vault
        user_name: User's name for storage
        dry_run: If True, don't actually import
        thread_title: Optional thread title for tagging

    Returns:
        Import statistics
    """
    stats = {
        "model_id": model_id,
        "total_pairs": len(pairs),
        "imported": 0,
        "skipped": 0,
        "errors": [],
        "pearl_ids": []
    }

    if dry_run:
        print(f"\n[DRY RUN] Would import {len(pairs)} message pairs to model: {model_id}")
        for i, pair in enumerate(pairs[:5]):  # Show first 5
            print(f"\n--- Pair {i+1} ---")
            print(f"User: {pair.user_message[:100]}...")
            print(f"AI: {pair.ai_response[:100]}...")
        if len(pairs) > 5:
            print(f"\n... and {len(pairs) - 5} more pairs")
        return stats

    # Get the store
    store = get_store(model_id)

    for i, pair in enumerate(pairs):
        try:
            # Build tags
            tags = ["imported"]
            if thread_title:
                # Add title words as tags
                title_words = [w.lower() for w in thread_title.split() if len(w) > 3]
                tags.extend(title_words[:3])

            # Store as a Pearl
            pearl_id = store.add_pearl(
                user_message=pair.user_message,
                ai_response=pair.ai_response,
                tags=tags,
                user_name=user_name,
                created_at=pair.timestamp.isoformat() if pair.timestamp else None
            )

            stats["imported"] += 1
            stats["pearl_ids"].append(pearl_id)

            if (i + 1) % 10 == 0:
                print(f"  Imported {i + 1}/{len(pairs)} pairs...")

        except Exception as e:
            stats["errors"].append(f"Pair {i+1}: {str(e)}")
            stats["skipped"] += 1

    # Close/seal the vault to ensure data is persisted
    store.close()
    print(f"  Vault sealed: {model_id}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Import OpenWebUI conversation into GAM-Memvid vault"
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to OpenWebUI conversation export (JSON)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Target model ID for the vault (e.g., 'qwen-235b-a22bth-origf2')"
    )
    parser.add_argument(
        "--user-name",
        type=str,
        default="User",
        help="User's name for memories (e.g., 'Jess')"
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

    print(f"Parsing: {args.file}")
    pairs, metadata = parse_openwebui_tree(args.file)

    print(f"\nThread metadata:")
    print(f"  Title: {metadata.get('title', 'Unknown')}")
    print(f"  Original model: {metadata.get('model_id', 'Unknown')}")
    print(f"  Thread ID: {metadata.get('thread_id', 'Unknown')}")
    print(f"  Message pairs found: {len(pairs)}")

    print(f"\nImporting to model: {args.model_id}")
    print(f"User name: {args.user_name}")

    stats = import_to_vault(
        pairs=pairs,
        model_id=args.model_id,
        user_name=args.user_name,
        dry_run=args.dry_run,
        thread_title=metadata.get("title")
    )

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Import complete!")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Imported: {stats['imported']}")
    print(f"  Skipped: {stats['skipped']}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            print(f"  - {err}")
        if len(stats["errors"]) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    return 0


if __name__ == "__main__":
    exit(main())
