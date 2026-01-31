"""
Memory Bucket Migration Utility

Rename or merge memory buckets when model IDs don't match.
Useful when you import memories with the wrong model ID.

Usage:
    python migrate_bucket.py --from "WizardLM (OF2)" --to "wizardlm-of2"
    python migrate_bucket.py --list  # See all buckets
"""
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

from config import config


def list_buckets() -> list[dict]:
    """List all memory buckets with their memory counts."""
    models_dir = config.DATA_DIR / "models"
    if not models_dir.exists():
        return []
    
    buckets = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            index_file = model_dir / "memory_index.json"
            count = 0
            if index_file.exists():
                try:
                    data = json.loads(index_file.read_text(encoding="utf-8"))
                    count = len(data.get("memories", {}))
                except:
                    pass
            
            buckets.append({
                "model_id": model_dir.name,
                "path": str(model_dir),
                "memory_count": count
            })
    
    return sorted(buckets, key=lambda x: x["model_id"])


def rename_bucket(old_id: str, new_id: str, merge: bool = False) -> dict:
    """
    Rename a memory bucket from old_id to new_id.
    
    Args:
        old_id: Current model ID (the wrong one)
        new_id: Target model ID (the correct one)
        merge: If True and new_id exists, merge memories. If False, fail.
    
    Returns:
        Result summary
    """
    models_dir = config.DATA_DIR / "models"
    old_path = models_dir / old_id
    new_path = models_dir / new_id
    
    if not old_path.exists():
        return {"success": False, "error": f"Source bucket '{old_id}' not found"}
    
    # Load source memories
    old_index = old_path / "memory_index.json"
    if not old_index.exists():
        return {"success": False, "error": f"No memory index in '{old_id}'"}
    
    old_data = json.loads(old_index.read_text(encoding="utf-8"))
    old_memories = old_data.get("memories", {})
    old_count = len(old_memories)
    
    if new_path.exists():
        if not merge:
            return {
                "success": False, 
                "error": f"Target bucket '{new_id}' already exists. Use --merge to combine memories."
            }
        
        # Merge mode: combine memories
        new_index = new_path / "memory_index.json"
        if new_index.exists():
            new_data = json.loads(new_index.read_text(encoding="utf-8"))
            new_memories = new_data.get("memories", {})
            existing_count = len(new_memories)
            
            # Merge old into new (old memories get added)
            for mem_id, mem_data in old_memories.items():
                # Prefix old IDs to avoid conflicts
                new_mem_id = f"migrated_{mem_id}"
                new_memories[new_mem_id] = mem_data
                new_memories[new_mem_id]["id"] = new_mem_id
            
            new_data["memories"] = new_memories
            new_data["updated_at"] = datetime.now().isoformat()
            
            # Save merged index
            new_index.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
            
            # Remove old bucket
            shutil.rmtree(old_path)
            
            return {
                "success": True,
                "action": "merged",
                "from": old_id,
                "to": new_id,
                "memories_migrated": old_count,
                "existing_memories": existing_count,
                "total_memories": len(new_memories)
            }
        else:
            # Target exists but no index - just rename
            pass
    
    # Simple rename (no merge needed)
    old_path.rename(new_path)
    
    # Update model_id references in the index
    new_index = new_path / "memory_index.json"
    if new_index.exists():
        data = json.loads(new_index.read_text(encoding="utf-8"))
        data["model_id"] = new_id
        data["updated_at"] = datetime.now().isoformat()
        data["migration_note"] = f"Renamed from '{old_id}' on {datetime.now().isoformat()}"
        new_index.write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    return {
        "success": True,
        "action": "renamed",
        "from": old_id,
        "to": new_id,
        "memories_migrated": old_count
    }


def delete_bucket(model_id: str, confirm: bool = False) -> dict:
    """Delete a memory bucket entirely."""
    if not confirm:
        return {"success": False, "error": "Must pass confirm=True to delete"}
    
    models_dir = config.DATA_DIR / "models"
    bucket_path = models_dir / model_id
    
    if not bucket_path.exists():
        return {"success": False, "error": f"Bucket '{model_id}' not found"}
    
    # Count memories before deletion
    index_file = bucket_path / "memory_index.json"
    count = 0
    if index_file.exists():
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
            count = len(data.get("memories", {}))
        except:
            pass
    
    shutil.rmtree(bucket_path)
    
    return {
        "success": True,
        "action": "deleted",
        "model_id": model_id,
        "memories_deleted": count
    }


# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate or rename memory buckets"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all memory buckets"
    )
    parser.add_argument(
        "--from", "-f",
        dest="from_id",
        type=str,
        help="Source model ID (the wrong bucket name)"
    )
    parser.add_argument(
        "--to", "-t",
        type=str,
        help="Target model ID (the correct bucket name)"
    )
    parser.add_argument(
        "--merge", "-m",
        action="store_true",
        help="Merge if target exists (instead of failing)"
    )
    parser.add_argument(
        "--delete", "-d",
        type=str,
        help="Delete a bucket (use with caution!)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Confirm deletion without prompting"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Memory Bucket Migration Tool")
    print(f"{'='*60}\n")
    
    if args.list:
        buckets = list_buckets()
        if not buckets:
            print("No memory buckets found.")
        else:
            print(f"Found {len(buckets)} bucket(s):\n")
            for b in buckets:
                print(f"  • {b['model_id']}")
                print(f"    Memories: {b['memory_count']}")
                print(f"    Path: {b['path']}\n")
    
    elif args.delete:
        if not args.yes:
            confirm = input(f"Are you sure you want to DELETE '{args.delete}'? Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("Cancelled.")
                exit(0)
        
        result = delete_bucket(args.delete, confirm=True)
        if result["success"]:
            print(f"✓ Deleted bucket '{args.delete}' ({result['memories_deleted']} memories)")
        else:
            print(f"✗ Error: {result['error']}")
    
    elif args.from_id and args.to:
        print(f"Migrating: '{args.from_id}' → '{args.to}'")
        if args.merge:
            print("Mode: MERGE (will combine with existing memories)")
        else:
            print("Mode: RENAME (will fail if target exists)")
        print()
        
        result = rename_bucket(args.from_id, args.to, merge=args.merge)
        
        if result["success"]:
            print(f"✓ Success!")
            print(f"  Action: {result['action']}")
            print(f"  Memories migrated: {result['memories_migrated']}")
            if result.get('total_memories'):
                print(f"  Total in target: {result['total_memories']}")
        else:
            print(f"✗ Error: {result['error']}")
    
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python migrate_bucket.py --list')
        print('  python migrate_bucket.py --from "WizardLM (OF2)" --to "wizardlm-of2"')
        print('  python migrate_bucket.py --from "old-model" --to "new-model" --merge')
