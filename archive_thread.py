"""
Archive Thread Tool - Archive OpenWebUI threads into GAM memory.

TWO MODES:
1. Simple Mode (--simple): Fast, stores as single blob (free)
2. Smart Mode (default): LLM-powered categorization (uses API calls)

TARGETS:
- Local: Stores directly to local files (default)
- Remote: Use --server URL to send to Railway or other remote GAM server

Use this when a conversation thread is at max capacity and you want to:
1. Preserve the relationship/memories
2. Start a fresh thread
3. Keep the AI's continuity

Usage:
    # Import to Railway (recommended for production)
    python archive_thread.py thread.json --model my-model --user Jess \\
        --server https://your-gam-server.up.railway.app

    # Interactive mode (will ask for server URL)
    python archive_thread.py --interactive

    # Local import (for testing only)
    python archive_thread.py thread.json --model my-model --user Jess --simple
"""
import json
import argparse
import asyncio
import sys
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Global server URL (set when using remote mode)
REMOTE_SERVER_URL: Optional[str] = None


def set_remote_server(url: str):
    """Set the remote GAM server URL."""
    global REMOTE_SERVER_URL
    REMOTE_SERVER_URL = url.rstrip('/')
    print(f"[Remote Mode] Targeting server: {REMOTE_SERVER_URL}")


async def store_memory_remote(
    model_id: str,
    content: str,
    category: Optional[str] = None,
    importance: Optional[str] = None,
    tags: Optional[list] = None,
    timestamp: Optional[datetime] = None
) -> dict:
    """Store a memory via HTTP API to remote GAM server."""
    if not REMOTE_SERVER_URL:
        raise ValueError("Remote server URL not set")
    
    payload = {
        "model_id": model_id,
        "content": content,
        "category": category,
        "importance": importance,
        "tags": tags
    }
    if timestamp:
        payload["timestamp"] = timestamp.isoformat()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{REMOTE_SERVER_URL}/memory/add",
            json=payload
        )
        response.raise_for_status()
        return response.json()


async def store_ai_self_remote(
    model_id: str,
    content: str,
    ai_self_type: str = "reflection",
    importance: str = "normal",
    tags: Optional[list] = None,
    created_at: Optional[str] = None
) -> dict:
    """Store an AI_Self memory via HTTP API."""
    if not REMOTE_SERVER_URL:
        raise ValueError("Remote server URL not set")
    
    payload = {
        "content": content,
        "ai_self_type": ai_self_type,
        "importance": importance,
        "tags": tags or []
    }
    if created_at:
        payload["created_at"] = created_at
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{REMOTE_SERVER_URL}/memory/ai-self/{model_id}/add",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def is_remote_mode() -> bool:
    """Check if we're in remote mode."""
    return REMOTE_SERVER_URL is not None


async def import_simple_remote(
    thread_data: dict,
    model_id: str,
    user_name: str = "User",
    category: Optional[str] = None,
    tags: Optional[list] = None
) -> dict:
    """Import a thread as a single blob via remote API."""
    messages = thread_data.get("messages", [])
    
    # Build conversation content
    conversation_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp")
        
        role_label = user_name if role == "user" else "AI"
        
        if timestamp:
            conversation_parts.append(f"[{timestamp}] {role_label}: {content}")
        else:
            conversation_parts.append(f"{role_label}: {content}")
    
    full_content = "\n\n".join(conversation_parts)
    
    try:
        result = await store_memory_remote(
            model_id=model_id,
            content=full_content,
            category=category or "context",
            tags=tags
        )
        return {
            "success": True,
            "memory_id": result.get("memory_id"),
            "messages_imported": len(messages),
            "errors": []
        }
    except Exception as e:
        return {
            "success": False,
            "memory_id": None,
            "messages_imported": 0,
            "errors": [str(e)]
        }


# Only import local modules if not in remote mode (lazy import)
def get_local_imports():
    from import_conversations import parse_single_thread, import_single_thread
    from memory_manager import memory_manager
    return parse_single_thread, import_single_thread, memory_manager


def load_thread_from_file(file_path: Path) -> dict:
    """Load a thread from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_thread_from_clipboard() -> Optional[dict]:
    """Try to load thread JSON from clipboard."""
    try:
        import pyperclip
        content = pyperclip.paste()
        return json.loads(content)
    except ImportError:
        print("Note: Install pyperclip for clipboard support: pip install pyperclip")
        return None
    except json.JSONDecodeError:
        print("Clipboard doesn't contain valid JSON")
        return None


def extract_openwebui_thread(data) -> dict:
    """
    Extract thread data from various OpenWebUI export formats.
    
    OpenWebUI can export in different structures:
    - Direct thread: {"id": ..., "messages": [...]}
    - Chat export: {"chat": {"id": ..., "messages": [...]}}
    - History export: {"history": {"messages": {...}}}
    - Raw messages list: [{"role": "user", "content": "..."}, ...]
    - Export list: [{conversation object}] - list with single conversation
    """
    # List format - could be messages OR a list containing one conversation export
    if isinstance(data, list):
        # Check if it's a list with a single conversation object (OpenWebUI export)
        if len(data) == 1 and isinstance(data[0], dict):
            first_item = data[0]
            # If it has 'chat' or 'history' or 'messages', it's a conversation export
            if any(key in first_item for key in ['chat', 'history', 'messages', 'title']):
                return extract_openwebui_thread(first_item)
        
        # Check if items look like messages (have 'role' field)
        if data and isinstance(data[0], dict) and 'role' in data[0]:
            return {
                "id": "imported",
                "title": "Imported Conversation",
                "messages": data,
                "model": None
            }
        
        # Multiple conversations - just take the first one
        if data and isinstance(data[0], dict):
            return extract_openwebui_thread(data[0])
    
    # Direct thread format
    if "messages" in data and isinstance(data["messages"], list):
        return data
    
    # Chat wrapper
    if "chat" in data:
        return extract_openwebui_thread(data["chat"])
    
    # History format (OpenWebUI internal)
    if "history" in data:
        history = data["history"]
        messages = []
        
        # History stores messages as dict with IDs
        if "messages" in history and isinstance(history["messages"], dict):
            msg_dict = history["messages"]
            # Sort by order if available, otherwise by key
            sorted_msgs = sorted(msg_dict.items(), key=lambda x: x[1].get("order", x[0]))
            for msg_id, msg_data in sorted_msgs:
                messages.append({
                    "role": msg_data.get("role", "user"),
                    "content": msg_data.get("content", ""),
                    "timestamp": msg_data.get("timestamp")
                })
        
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "created_at": data.get("created_at") or data.get("timestamp"),
            "messages": messages,
            "model": data.get("model") or (data.get("models", [None])[0])
        }
    
    # Already in correct format
    return data


def summarize_thread(thread: dict) -> None:
    """Print a summary of the thread."""
    messages = thread.get("messages", [])
    title = thread.get("title", "Untitled")
    
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    
    print(f"\n{'='*60}")
    print(f"Thread: {title}")
    print(f"{'='*60}")
    print(f"Total messages: {len(messages)}")
    print(f"  - User messages: {len(user_msgs)}")
    print(f"  - Assistant messages: {len(assistant_msgs)}")
    
    # Show first and last exchange
    if messages:
        print(f"\nFirst message preview:")
        first = messages[0]
        preview = first.get("content", "")[:100]
        print(f"  [{first.get('role')}]: {preview}...")
        
        if len(messages) > 2:
            print(f"\nLast message preview:")
            last = messages[-1]
            preview = last.get("content", "")[:100]
            print(f"  [{last.get('role')}]: {preview}...")
    
    print(f"{'='*60}\n")


def interactive_mode():
    """Run in interactive mode for easy archiving."""
    print("\n" + "="*60)
    print("  GAM Thread Archiver - Interactive Mode")
    print("="*60)
    
    # Ask for server target
    print("\nWhere should memories be stored?")
    print("  1. Remote server (Railway/production) - RECOMMENDED")
    print("  2. Local (this machine only)")
    
    target_choice = input("\nChoice [1/2] (default: 1): ").strip() or "1"
    
    if target_choice == "1":
        default_url = "https://dynamic-connection-production-2f3e.up.railway.app"
        server_url = input(f"\nGAM Server URL [{default_url}]: ").strip() or default_url
        set_remote_server(server_url)
        
        # Test connection
        print("Testing connection...")
        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{server_url}/health")
                if resp.status_code == 200:
                    print("✓ Connected to server successfully!")
                else:
                    print(f"⚠ Server returned status {resp.status_code}")
        except Exception as e:
            print(f"⚠ Could not connect to server: {e}")
            proceed = input("Continue anyway? [y/N]: ").strip().lower()
            if proceed != 'y':
                return 1
    else:
        print("\n[Local Mode] Memories will be stored on this machine only.")
    
    # Get thread data
    print("\nHow would you like to provide the thread?")
    print("  1. Paste JSON directly")
    print("  2. Load from file")
    print("  3. Load from clipboard")
    
    choice = input("\nChoice [1/2/3]: ").strip()
    
    thread_data = None
    
    if choice == "1":
        print("\nPaste your thread JSON (end with empty line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        try:
            thread_data = json.loads("\n".join(lines))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return 1
    
    elif choice == "2":
        file_path = input("\nEnter file path: ").strip()
        try:
            thread_data = load_thread_from_file(Path(file_path))
        except Exception as e:
            print(f"Error loading file: {e}")
            return 1
    
    elif choice == "3":
        thread_data = load_thread_from_clipboard()
        if not thread_data:
            return 1
    
    else:
        print("Invalid choice")
        return 1
    
    # Extract thread
    thread_data = extract_openwebui_thread(thread_data)
    
    # Show summary
    summarize_thread(thread_data)
    
    # Get model ID
    default_model = thread_data.get("model", "default")
    model_id = input(f"Target model ID [{default_model}]: ").strip() or default_model
    
    # Get user name
    user_name = input("Your name [User]: ").strip() or "User"
    
    # Choose import mode
    print("\nImport mode:")
    print("  1. SMART (recommended) - LLM analyzes & categorizes each exchange")
    print("  2. SIMPLE - Fast, stores as single blob under one category")
    mode_choice = input("\nChoice [1/2] (default: 1): ").strip() or "1"
    use_smart = mode_choice != "2"
    
    if use_smart:
        # Smart mode - LLM handles categorization
        print("\nSmart import will automatically:")
        print("  - Split thread into individual exchanges")
        print("  - Categorize each memory appropriately")
        print("  - Extract AI_Self reflections")
        print("  - Preserve timestamps")
        
        dry_run = input("\nDry run first? (see what would be extracted) [y/N]: ").strip().lower() == 'y'
        
        if not dry_run:
            confirm = input("\nProceed with smart import? [Y/n]: ").strip().lower()
            if confirm and confirm != 'y':
                print("Cancelled")
                return 0
        
        # Run smart import
        print("\nRunning smart import...")
        from smart_import import SmartImporter
        
        importer = SmartImporter(user_name=user_name)
        
        def progress(status="", message="", **kwargs):
            if message:
                print(f"  [{status.upper()}] {message}")
            if "processed" in kwargs:
                print(f"    Progress: {kwargs['processed']}/{kwargs['total']} exchanges, {kwargs.get('memories_found', 0)} memories")
        
        # Use remote server if set, otherwise localhost
        server_url = REMOTE_SERVER_URL or "http://localhost:8100"
        
        result = asyncio.run(importer.import_thread(
            thread_data=thread_data,
            target_model_id=model_id,
            gam_server_url=server_url,
            progress_callback=progress,
            dry_run=dry_run
        ))
        
        print(f"\n{'='*60}")
        print(f"  Results")
        print(f"{'='*60}")
        print(f"Exchanges found: {result.get('exchanges_found', 0)}")
        print(f"Memories extracted: {result.get('memories_extracted', 0)}")
        print(f"AI_Self memories: {result.get('ai_self_count', 0)}")
        
        print(f"\nBy category:")
        for cat, count in result.get("by_category", {}).items():
            print(f"  {cat}: {count}")
        
        if result.get("timestamp_range"):
            tr = result["timestamp_range"]
            if tr.get("start"):
                print(f"\nTime range: {tr['start']} to {tr['end']}")
        
        if dry_run:
            print(f"\n[DRY RUN] Preview of memories that would be stored:")
            for mem in result.get("memories", [])[:10]:
                ai_marker = " [AI_SELF]" if mem.get("is_ai_self") else ""
                print(f"  [{mem['category']}]{ai_marker} {mem['content'][:70]}...")
            if len(result.get("memories", [])) > 10:
                print(f"  ... and {len(result['memories']) - 10} more")
            
            # Ask if they want to proceed
            proceed = input("\nLooks good? Proceed with actual import? [Y/n]: ").strip().lower()
            if not proceed or proceed == 'y':
                print("\nStoring memories...")
                result = asyncio.run(importer.import_thread(
                    thread_data=thread_data,
                    target_model_id=model_id,
                    gam_server_url=server_url,
                    progress_callback=progress,
                    dry_run=False
                ))
                print(f"\n✓ Successfully stored {result.get('memories_stored', 0)} memories!")
            else:
                print("Cancelled")
                return 0
        else:
            if result.get("success"):
                print(f"\n✓ Successfully stored {result.get('memories_stored', 0)} memories!")
            else:
                print(f"\n✗ Some errors occurred:")
                for err in result.get("errors", [])[:5]:
                    print(f"  - {err}")
        
        print(f"\nYou can now safely start a fresh thread in OpenWebUI.")
        print(f"The AI will remember this conversation through categorized memories.")
        return 0
    
    else:
        # Simple mode - single blob
        print("\nAvailable categories: relationship, theology, science, ai_theory, ai_self, context")
        category = input("Category (optional): ").strip() or None
        
        tags_input = input("Tags (comma-separated, optional): ").strip()
        tags = [t.strip() for t in tags_input.split(",")] if tags_input else None
        
        print(f"\nReady to archive (simple mode):")
        print(f"  Model: {model_id}")
        print(f"  User name: {user_name}")
        print(f"  Category: {category or 'auto'}")
        print(f"  Tags: {tags or 'none'}")
        
        confirm = input("\nProceed? [Y/n]: ").strip().lower()
        if confirm and confirm != 'y':
            print("Cancelled")
            return 0
        
        print("\nArchiving thread...")
        
        if is_remote_mode():
            # Remote mode - send via HTTP API
            result = asyncio.run(import_simple_remote(
                thread_data=thread_data,
                model_id=model_id,
                user_name=user_name,
                category=category,
                tags=tags
            ))
        else:
            # Local mode
            _, import_single_thread, _ = get_local_imports()
            result = import_single_thread(
                thread_data=thread_data,
                target_model_id=model_id,
                user_name=user_name,
                category=category,
                tags=tags
            )
        
        if result["success"]:
            print(f"\n✓ Successfully archived!")
            print(f"  Memory ID: {result.get('memory_id', 'N/A')}")
            print(f"  Messages imported: {result.get('messages_imported', 'N/A')}")
            print(f"\nYou can now safely start a fresh thread in OpenWebUI.")
        else:
            print(f"\n✗ Archive failed:")
            for error in result["errors"]:
                print(f"  - {error}")
            return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Archive OpenWebUI threads into GAM memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (easiest)
  python archive_thread.py --interactive

  # Smart import (default) - LLM categorizes each exchange
  python archive_thread.py thread.json --model my-model --user Jess

  # Simple import - fast, single blob
  python archive_thread.py thread.json --model my-model --user Jess --simple

  # Dry run (preview what would be extracted)
  python archive_thread.py thread.json --model my-model --user Jess --dry-run

  # Simple mode with category and tags
  python archive_thread.py thread.json --model my-model --user Jess --simple \\
      --category relationship --tags "archived,important"
"""
    )
    
    parser.add_argument(
        "file",
        type=Path,
        nargs="?",
        help="Path to thread JSON file"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        help="Remote GAM server URL (e.g., https://your-app.up.railway.app)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Target model ID"
    )
    parser.add_argument(
        "--user", "-u",
        type=str,
        default="User",
        help="Your name (default: User)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple mode (single blob, no LLM analysis)"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="Category for the archived thread (simple mode only)"
    )
    parser.add_argument(
        "--tags", "-t",
        type=str,
        help="Comma-separated tags"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without importing"
    )
    
    args = parser.parse_args()
    
    # Set remote server if provided
    if args.server:
        set_remote_server(args.server)
    
    # Interactive mode
    if args.interactive or (not args.file and not sys.stdin.isatty()):
        return interactive_mode()
    
    # File mode
    if not args.file:
        parser.print_help()
        return 1
    
    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        return 1
    
    if not args.model:
        print("Error: --model is required in non-interactive mode")
        return 1
    
    # Load and extract thread
    thread_data = load_thread_from_file(args.file)
    thread_data = extract_openwebui_thread(thread_data)
    
    # Show summary
    summarize_thread(thread_data)
    
    # Parse tags
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
    
    if args.simple:
        # Simple mode - single blob
        if args.dry_run:
            print("[DRY RUN - SIMPLE MODE] Would import with:")
            print(f"  Model: {args.model}")
            print(f"  User: {args.user}")
            print(f"  Category: {args.category or 'context'}")
            print(f"  Tags: {tags or 'none'}")
            return 0
        
        print("Archiving thread (simple mode)...")
        
        if is_remote_mode():
            result = asyncio.run(import_simple_remote(
                thread_data=thread_data,
                model_id=args.model,
                user_name=args.user,
                category=args.category,
                tags=tags
            ))
        else:
            _, import_single_thread, _ = get_local_imports()
            result = import_single_thread(
                thread_data=thread_data,
                target_model_id=args.model,
                user_name=args.user,
                category=args.category,
                tags=tags
            )
        
        if result["success"]:
            print(f"\n✓ Successfully archived!")
            print(f"  Memory ID: {result.get('memory_id', 'N/A')}")
            print(f"  Messages imported: {result.get('messages_imported', 'N/A')}")
        else:
            print(f"\n✗ Archive failed:")
            for error in result.get("errors", ["Unknown error"]):
                print(f"  - {error}")
            return 1
    
    else:
        # Smart mode - LLM analysis
        from smart_import import SmartImporter
        
        print("Running smart import (LLM-powered categorization)...")
        
        importer = SmartImporter(user_name=args.user)
        
        def progress(status="", message="", **kwargs):
            if message:
                print(f"  [{status.upper()}] {message}")
            if "processed" in kwargs:
                print(f"    Progress: {kwargs['processed']}/{kwargs['total']} exchanges")
        
        # Use remote server if set, otherwise localhost
        server_url = REMOTE_SERVER_URL or "http://localhost:8100"
        
        result = asyncio.run(importer.import_thread(
            thread_data=thread_data,
            target_model_id=args.model,
            gam_server_url=server_url,
            progress_callback=progress,
            dry_run=args.dry_run
        ))
        
        print(f"\n{'='*60}")
        print(f"Exchanges found: {result.get('exchanges_found', 0)}")
        print(f"Memories extracted: {result.get('memories_extracted', 0)}")
        print(f"AI_Self memories: {result.get('ai_self_count', 0)}")
        
        print(f"\nBy category:")
        for cat, count in result.get("by_category", {}).items():
            print(f"  {cat}: {count}")
        
        if args.dry_run:
            print(f"\n[DRY RUN] Memories that would be stored:")
            for mem in result.get("memories", [])[:15]:
                ai_marker = " [AI_SELF]" if mem.get("is_ai_self") else ""
                ts_marker = f" ({mem['timestamp'][:10]})" if mem.get("timestamp") else ""
                print(f"  [{mem['category']}]{ai_marker}{ts_marker} {mem['content'][:60]}...")
            if len(result.get("memories", [])) > 15:
                print(f"  ... and {len(result['memories']) - 15} more")
        else:
            if result.get("success"):
                print(f"\n✓ Successfully stored {result.get('memories_stored', 0)} memories!")
            else:
                print(f"\n✗ Some errors occurred:")
                for err in result.get("errors", [])[:5]:
                    print(f"  - {err}")
                return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
