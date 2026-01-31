#!/usr/bin/env python3
"""
Migration Script: JSON to Memvid v2 (Librarian Mode)

============================================================
THE LIBRARIAN MIGRATION: Store Full, Raw Conversations
============================================================

This migration tool imports conversation history as RAW PEARLS:
- No summarization or compression
- Full user messages preserved (even 3-page essays)
- Full AI responses preserved
- Original timestamps preserved ("Time Travel")

Supports two source formats:
  Mode A: GAM Index (memory_index.json) - Existing organized memories
  Mode B: OpenWebUI Chat Exports - Raw conversation history

Usage:
  # Dry run (see what would be imported)
  python migrate_json_to_memvid.py --model eli --dry-run

  # Import from GAM index
  python migrate_json_to_memvid.py --model eli --source gam

  # Import from OpenWebUI export (full raw messages)
  python migrate_json_to_memvid.py --model eli --source openwebui --file chat_export.json

  # Import all models
  python migrate_json_to_memvid.py --all --source gam
"""
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from config import config


@dataclass
class MigrationStats:
    """Statistics from a migration run."""
    model_id: str
    source_type: str
    source_path: str
    total_found: int = 0
    imported: int = 0
    skipped: int = 0
    errors: int = 0
    total_words: int = 0
    dry_run: bool = False
    backup_path: Optional[str] = None
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "total_found": self.total_found,
            "imported": self.imported,
            "skipped": self.skipped,
            "errors": self.errors,
            "total_words": self.total_words,
            "dry_run": self.dry_run,
            "backup_path": self.backup_path,
            "error_messages": self.error_messages[:10]
        }


@dataclass
class RawPearl:
    """A raw conversation exchange for migration."""
    pearl_id: str
    user_message: str
    ai_response: str
    tags: List[str]
    category: str = "context"
    importance: str = "normal"
    created_at: Optional[str] = None
    user_name: str = "User"
    thread_id: Optional[str] = None
    message_index: Optional[int] = None
    emotional_tone: Optional[str] = None

    @property
    def word_count(self) -> int:
        return len(self.user_message.split()) + len(self.ai_response.split())


# =============================================================================
# Mode A: GAM Index Migration (memory_index.json)
# =============================================================================

def parse_gam_index(index_path: Path) -> List[RawPearl]:
    """
    Parse a GAM memory_index.json file into RawPearls.

    Preserves FULL content and original timestamps.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"GAM index not found: {index_path}")

    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pearls = []
    memories_dict = data.get("memories", {})

    for mem_id, mem_data in memories_dict.items():
        try:
            entry_id = mem_data.get("memory_id") or mem_data.get("id") or mem_id
            content = mem_data.get("content", "")

            # CRITICAL: Preserve the original timestamp
            created_at = mem_data.get("created_at")

            # Parse content into user_message and ai_response
            user_message = content
            ai_response = ""

            # Try to split if it looks like a conversation
            for splitter in ["\n\nAI:", "\nAI:", "\nAI responded:", "AI:"]:
                if splitter in content:
                    parts = content.split(splitter, 1)
                    user_part = parts[0]
                    ai_response = parts[1].strip() if len(parts) > 1 else ""

                    # Clean up user part
                    for prefix in ["User:", "User said:", "User mentioned:"]:
                        if user_part.startswith(prefix):
                            user_part = user_part[len(prefix):].strip()
                            break
                    user_message = user_part.strip()
                    break

            # Build tags from category and existing tags
            tags = list(mem_data.get("tags", []))
            category = mem_data.get("category", "context")
            importance = mem_data.get("importance", "normal")

            # Add category as a tag for searchability
            if category and f"#{category}" not in tags:
                tags.append(f"#{category}")

            # Add importance as tag if core/high
            if importance in ["core", "high"]:
                tags.append(f"#{importance}")

            pearl = RawPearl(
                pearl_id=f"migrated_{entry_id}",
                user_message=user_message,
                ai_response=ai_response,
                tags=tags,
                category=category,
                importance=importance,
                created_at=created_at,  # PRESERVED!
                emotional_tone=mem_data.get("emotional_tone")
            )
            pearls.append(pearl)

        except Exception as e:
            print(f"  [WARN] Error parsing memory {mem_id}: {e}")

    return pearls


def find_all_gam_indexes() -> Dict[str, Path]:
    """Find all GAM memory_index.json files in data/models/"""
    models_dir = config.DATA_DIR / "models"
    indexes = {}

    if not models_dir.exists():
        return indexes

    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            index_file = model_dir / "memory_index.json"
            if index_file.exists():
                indexes[model_dir.name] = index_file

    return indexes


# =============================================================================
# Mode B: OpenWebUI Chat Export Migration (Raw Messages)
# =============================================================================

def parse_openwebui_export_raw(
    export_path: Path,
    user_name: str = "User"
) -> List[RawPearl]:
    """
    Parse an OpenWebUI chat export into RAW Pearls.

    CRITICAL: This preserves FULL messages without any compression.
    User messages that are 3 pages long stay 3 pages long.
    """
    if not export_path.exists():
        raise FileNotFoundError(f"Export file not found: {export_path}")

    with open(export_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pearls = []

    # Handle different export formats
    if isinstance(data, list):
        chats = data
    elif "chat" in data:
        chats = [data["chat"]]
    elif "chats" in data:
        chats = data["chats"]
    else:
        chats = [data]

    for chat in chats:
        chat_pearls = parse_single_chat_raw(chat, user_name)
        pearls.extend(chat_pearls)

    return pearls


def parse_single_chat_raw(chat: dict, user_name: str = "User") -> List[RawPearl]:
    """Parse a single chat into RAW Pearls (message pairs)."""
    pearls = []

    chat_id = chat.get("id", "unknown")
    chat_title = chat.get("title", "Untitled Chat")

    # Get creation timestamp
    chat_created = chat.get("created_at") or chat.get("timestamp")
    if isinstance(chat_created, (int, float)):
        chat_created = datetime.fromtimestamp(chat_created).isoformat()

    # Get messages
    messages = []
    if "messages" in chat:
        messages = chat["messages"]
    elif "history" in chat and "messages" in chat["history"]:
        msg_dict = chat["history"]["messages"]
        if isinstance(msg_dict, dict):
            messages = list(msg_dict.values())
        else:
            messages = msg_dict

    # Process messages in pairs (user + assistant)
    i = 0
    pair_index = 0

    while i < len(messages):
        user_msg = None
        ai_msg = None
        user_ts = None

        # Find next user message
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                user_ts = msg.get("timestamp") or msg.get("created_at") or chat_created
                if isinstance(user_ts, (int, float)):
                    user_ts = datetime.fromtimestamp(user_ts).isoformat()
                i += 1
                break
            i += 1

        if not user_msg:
            break

        # Find corresponding assistant response
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") == "assistant":
                ai_msg = msg.get("content", "")
                i += 1
                break
            elif msg.get("role") == "user":
                # Next user message found, no AI response for current
                break
            i += 1

        # Skip if no content
        if not user_msg or not user_msg.strip():
            continue

        # Create Pearl with FULL RAW content
        pearl_id = f"import_{chat_id}_{pair_index}_{datetime.now().strftime('%f')}"

        # Generate tags from content hints
        tags = ["#imported", f"#thread:{chat_id[:8]}"]

        # Add title-based tag
        if chat_title and chat_title != "Untitled Chat":
            safe_title = chat_title[:30].replace(" ", "_").lower()
            tags.append(f"#topic:{safe_title}")

        pearl = RawPearl(
            pearl_id=pearl_id,
            user_message=user_msg,  # FULL, RAW - no truncation
            ai_response=ai_msg or "",  # FULL, RAW - no truncation
            tags=tags,
            category="context",
            importance="normal",
            created_at=user_ts,  # PRESERVED!
            user_name=user_name,
            thread_id=chat_id,
            message_index=pair_index
        )
        pearls.append(pearl)
        pair_index += 1

    return pearls


# =============================================================================
# Migration Engine
# =============================================================================

def backup_vault(vault_path: Path) -> Optional[Path]:
    """Create a backup of an existing vault file."""
    if not vault_path.exists():
        return None

    backup_path = vault_path.with_suffix(
        f".mv2.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(vault_path, backup_path)
    print(f"  [BACKUP] Created: {backup_path}")
    return backup_path


def migrate_pearls_to_vault(
    model_id: str,
    pearls: List[RawPearl],
    user_name: str = "User",
    dry_run: bool = False,
    skip_backup: bool = False
) -> MigrationStats:
    """
    Migrate a list of RawPearls into a Memvid vault.

    Stores FULL, RAW content - no compression or summarization.
    """
    from memvid_store import MemvidStore, MEMVID_AVAILABLE

    stats = MigrationStats(
        model_id=model_id,
        source_type="pearls",
        source_path="direct",
        total_found=len(pearls),
        dry_run=dry_run
    )

    if not pearls:
        print(f"  [INFO] No Pearls to migrate for {model_id}")
        return stats

    # Calculate total words
    stats.total_words = sum(p.word_count for p in pearls)

    if dry_run:
        print(f"\n  [DRY RUN] Would import {len(pearls)} Pearls to {model_id}")
        print(f"  [DRY RUN] Total words: {stats.total_words:,}")
        print(f"\n  [DRY RUN] Sample Pearls (showing FULL content preview):")

        for pearl in pearls[:3]:
            ts = pearl.created_at[:19] if pearl.created_at else "no timestamp"
            tags = ", ".join(pearl.tags[:3])
            user_preview = pearl.user_message[:150].replace("\n", " ")
            ai_preview = pearl.ai_response[:100].replace("\n", " ") if pearl.ai_response else "(no AI response)"

            print(f"\n    Pearl: {pearl.pearl_id}")
            print(f"    Timestamp: {ts}")
            print(f"    Tags: {tags}")
            print(f"    Words: {pearl.word_count}")
            print(f"    User: {user_preview}...")
            print(f"    AI: {ai_preview}...")

        if len(pearls) > 3:
            print(f"\n    ... and {len(pearls) - 3} more Pearls")

        # Show timestamp range
        timestamps = [p.created_at for p in pearls if p.created_at]
        if timestamps:
            timestamps.sort()
            print(f"\n  [DRY RUN] Timestamp range: {timestamps[0][:10]} to {timestamps[-1][:10]}")

        # Show word count distribution
        word_counts = [p.word_count for p in pearls]
        avg_words = sum(word_counts) / len(word_counts)
        max_words = max(word_counts)
        print(f"  [DRY RUN] Word counts: avg={avg_words:.0f}, max={max_words}")

        stats.imported = len(pearls)
        return stats

    # Real migration
    if not MEMVID_AVAILABLE:
        print("  [ERROR] memvid-sdk not installed!")
        stats.errors = len(pearls)
        stats.error_messages.append("memvid-sdk not installed")
        return stats

    # Backup existing vault
    vault_path = config.VAULTS_DIR / f"{model_id}.mv2"
    if not skip_backup and vault_path.exists():
        backup_path = backup_vault(vault_path)
        stats.backup_path = str(backup_path) if backup_path else None

    # Open/create vault
    try:
        store = MemvidStore(model_id, config.VAULTS_DIR)
    except Exception as e:
        print(f"  [ERROR] Failed to open vault: {e}")
        stats.errors = len(pearls)
        stats.error_messages.append(f"Failed to open vault: {e}")
        return stats

    # Import Pearls
    print(f"  [IMPORT] Importing {len(pearls)} Pearls ({stats.total_words:,} words)...")

    for i, pearl in enumerate(pearls):
        try:
            # Store FULL, RAW content via add_pearl
            store.add_pearl(
                user_message=pearl.user_message,  # FULL
                ai_response=pearl.ai_response,    # FULL
                tags=pearl.tags,
                category=pearl.category,
                importance=pearl.importance,
                emotional_tone=pearl.emotional_tone,
                created_at=pearl.created_at,      # TIME TRAVEL!
                user_name=pearl.user_name,
                thread_id=pearl.thread_id,
                message_index=pearl.message_index,
                pearl_id=pearl.pearl_id
            )
            stats.imported += 1

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"    ... imported {i + 1}/{len(pearls)}")

        except Exception as e:
            stats.errors += 1
            if len(stats.error_messages) < 10:
                stats.error_messages.append(f"Error on {pearl.pearl_id}: {e}")

    print(f"  [DONE] Imported {stats.imported}, Errors: {stats.errors}")
    return stats


def migrate_gam_index(
    model_id: str,
    dry_run: bool = False,
    skip_backup: bool = False
) -> MigrationStats:
    """Migrate a model's GAM memory_index.json to Memvid vault."""
    index_path = config.DATA_DIR / "models" / model_id / "memory_index.json"

    stats = MigrationStats(
        model_id=model_id,
        source_type="gam_index",
        source_path=str(index_path),
        dry_run=dry_run
    )

    if not index_path.exists():
        print(f"  [SKIP] No GAM index found for {model_id}")
        return stats

    print(f"\n{'='*60}")
    print(f"Migrating: {model_id}")
    print(f"Source: {index_path}")
    print(f"Mode: RAW PEARL STORAGE (Librarian Mode)")
    print(f"{'='*60}")

    try:
        pearls = parse_gam_index(index_path)
        stats.total_found = len(pearls)
        print(f"  [FOUND] {len(pearls)} memories in GAM index")

        if not pearls:
            return stats

        return migrate_pearls_to_vault(
            model_id=model_id,
            pearls=pearls,
            dry_run=dry_run,
            skip_backup=skip_backup
        )

    except Exception as e:
        print(f"  [ERROR] Migration failed: {e}")
        stats.errors = stats.total_found or 1
        stats.error_messages.append(str(e))
        return stats


def migrate_openwebui_export(
    model_id: str,
    export_path: Path,
    user_name: str = "User",
    dry_run: bool = False,
    skip_backup: bool = False
) -> MigrationStats:
    """Migrate an OpenWebUI chat export to a Memvid vault (RAW mode)."""
    stats = MigrationStats(
        model_id=model_id,
        source_type="openwebui_export",
        source_path=str(export_path),
        dry_run=dry_run
    )

    if not export_path.exists():
        print(f"  [ERROR] Export file not found: {export_path}")
        stats.error_messages.append("Export file not found")
        return stats

    print(f"\n{'='*60}")
    print(f"Migrating OpenWebUI Export: {export_path.name}")
    print(f"Target Model: {model_id}")
    print(f"Mode: RAW PEARL STORAGE (Full conversations preserved)")
    print(f"{'='*60}")

    try:
        pearls = parse_openwebui_export_raw(export_path, user_name)
        stats.total_found = len(pearls)
        print(f"  [FOUND] {len(pearls)} conversation exchanges")

        if not pearls:
            return stats

        return migrate_pearls_to_vault(
            model_id=model_id,
            pearls=pearls,
            user_name=user_name,
            dry_run=dry_run,
            skip_backup=skip_backup
        )

    except Exception as e:
        print(f"  [ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        stats.errors = stats.total_found or 1
        stats.error_messages.append(str(e))
        return stats


def migrate_all_gam_indexes(
    dry_run: bool = False,
    skip_backup: bool = False
) -> List[MigrationStats]:
    """Migrate all found GAM memory indexes to Memvid vaults."""
    indexes = find_all_gam_indexes()

    if not indexes:
        print("No GAM memory indexes found in data/models/")
        return []

    print(f"\nFound {len(indexes)} models with GAM indexes:")
    for model_id, path in indexes.items():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            count = len(data.get("memories", {}))
        print(f"  - {model_id}: {count} memories")

    results = []
    for model_id in indexes:
        stats = migrate_gam_index(
            model_id=model_id,
            dry_run=dry_run,
            skip_backup=skip_backup
        )
        results.append(stats)

    return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate memory data to Memvid v2 (Librarian Mode - Raw Storage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The Librarian Migration stores FULL, RAW conversations:
- No summarization or compression
- 3-page user essays stay as 3 pages
- Original timestamps preserved

Examples:
  # Dry run - see what would be imported
  python migrate_json_to_memvid.py --model eli --dry-run

  # Import from GAM index (raw storage)
  python migrate_json_to_memvid.py --model eli --source gam

  # Import all models
  python migrate_json_to_memvid.py --all --source gam

  # Import from OpenWebUI export (raw messages)
  python migrate_json_to_memvid.py --model eli --source openwebui --file chat_export.json
        """
    )

    parser.add_argument("--model", "-m", help="Model ID to migrate")
    parser.add_argument("--all", "-a", action="store_true", help="Migrate all found models")
    parser.add_argument(
        "--source", "-s",
        choices=["gam", "openwebui"],
        default="gam",
        help="Source type"
    )
    parser.add_argument("--file", "-f", help="Path to OpenWebUI export file")
    parser.add_argument("--user-name", "-u", default="User", help="User's name")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview only")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup")
    parser.add_argument("--list", "-l", action="store_true", help="List models and exit")

    args = parser.parse_args()

    # List mode
    if args.list:
        indexes = find_all_gam_indexes()
        if not indexes:
            print("No GAM memory indexes found.")
            return

        print(f"\nFound {len(indexes)} models with GAM indexes:\n")
        total = 0
        total_words = 0

        for model_id, path in sorted(indexes.items()):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                memories = data.get("memories", {})
                count = len(memories)
                total += count

                # Estimate word count
                words = sum(len(m.get("content", "").split()) for m in memories.values())
                total_words += words

                timestamps = [m.get("created_at", "") for m in memories.values() if m.get("created_at")]
                if timestamps:
                    timestamps.sort()
                    ts_range = f"{timestamps[0][:10]} to {timestamps[-1][:10]}"
                else:
                    ts_range = "no timestamps"

            print(f"  {model_id:20} {count:5} memories  ~{words:,} words  ({ts_range})")

        print(f"\n  {'TOTAL':20} {total:5} memories  ~{total_words:,} words")
        return

    # Validate args
    if not args.all and not args.model:
        parser.error("Either --model or --all is required")

    if args.source == "openwebui" and not args.file:
        parser.error("--file is required for OpenWebUI imports")

    print("\n" + "=" * 60)
    print("GAM-Memvid Migration Tool (Librarian Mode)")
    print("Storage: FULL, RAW conversations (no compression)")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    results = []

    if args.all:
        results = migrate_all_gam_indexes(
            dry_run=args.dry_run,
            skip_backup=args.skip_backup
        )
    elif args.source == "gam":
        stats = migrate_gam_index(
            model_id=args.model,
            dry_run=args.dry_run,
            skip_backup=args.skip_backup
        )
        results = [stats]
    elif args.source == "openwebui":
        stats = migrate_openwebui_export(
            model_id=args.model,
            export_path=Path(args.file),
            user_name=args.user_name,
            dry_run=args.dry_run,
            skip_backup=args.skip_backup
        )
        results = [stats]

    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY (Librarian Mode)")
    print("=" * 60)

    total_found = sum(r.total_found for r in results)
    total_imported = sum(r.imported for r in results)
    total_errors = sum(r.errors for r in results)
    total_words = sum(r.total_words for r in results)

    for r in results:
        status = "DRY RUN" if r.dry_run else ("OK" if r.errors == 0 else "ERRORS")
        print(f"  {r.model_id:20} {r.imported:5}/{r.total_found:5} Pearls  ~{r.total_words:,} words  [{status}]")

    print(f"\n  {'TOTAL':20} {total_imported:5}/{total_found:5} Pearls  ~{total_words:,} words")

    if total_errors > 0:
        print(f"\n  ERRORS: {total_errors}")
        for r in results:
            for err in r.error_messages[:3]:
                print(f"    - [{r.model_id}] {err}")

    if args.dry_run:
        print("\n*** This was a DRY RUN. Run without --dry-run to actually import. ***")


if __name__ == "__main__":
    main()
