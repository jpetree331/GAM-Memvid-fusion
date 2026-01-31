"""
Memvid Store - The Vault layer for AI memory persistence.

Architecture: Vault + Buffer (The Nightly Build Pattern)
============================================================
Memvid v1 requires full re-encode to update files, which is too slow
for real-time chat. This module implements a buffer strategy:

1. BUFFER: New memories go into buffer.json instantly (fast writes)
2. VAULT: Main .mv2 file holds consolidated memories (fast semantic search)
3. SEARCH: Queries BOTH vault AND buffer, merges results
4. MERGE: merge_memory() consolidates buffer into vault (run nightly)

Each AI model (persona) gets its own isolated vault + buffer pair.
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import threading

from memory_entry import (
    MemoryEntry,
    MemoryCategory,
    ImportanceLevel,
    AISelfType,
    create_memory,
    create_ai_self_memory
)

# Memvid SDK imports
try:
    from memvid_sdk import create as memvid_create, use as memvid_use
    MEMVID_AVAILABLE = True
except ImportError:
    MEMVID_AVAILABLE = False
    print("[MemvidStore] WARNING: memvid-sdk not installed. Run: pip install memvid-sdk")


@dataclass
class SearchResult:
    """A single search result from the vault or buffer."""
    memory: MemoryEntry
    score: float
    preview: str
    source: str = "vault"  # "vault" or "buffer"

    def to_dict(self) -> dict:
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "preview": self.preview,
            "source": self.source
        }


class MemoryBuffer:
    """
    Fast JSON-based buffer for real-time memory writes.

    Memories are written here instantly, then merged into the
    main vault during nightly consolidation.
    """

    def __init__(self, buffer_path: Path):
        self.buffer_path = buffer_path
        self._lock = threading.Lock()
        self._memories: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load buffer from disk."""
        if self.buffer_path.exists():
            try:
                with open(self.buffer_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._memories = data.get("memories", {})
                print(f"[Buffer] Loaded {len(self._memories)} buffered memories")
            except Exception as e:
                print(f"[Buffer] Error loading buffer: {e}")
                self._memories = {}
        else:
            self._memories = {}

    def _save(self):
        """Save buffer to disk."""
        try:
            data = {
                "updated_at": datetime.now().isoformat(),
                "count": len(self._memories),
                "memories": self._memories
            }
            # Atomic write via temp file
            temp_path = self.buffer_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.buffer_path)
        except Exception as e:
            print(f"[Buffer] Error saving buffer: {e}")

    def add(self, entry: MemoryEntry) -> str:
        """Add a memory to the buffer (instant write)."""
        with self._lock:
            self._memories[entry.id] = entry.to_dict()
            self._save()
        return entry.id

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID from the buffer."""
        with self._lock:
            data = self._memories.get(memory_id)
            if data:
                return MemoryEntry.from_dict(data)
        return None

    def get_all(self) -> List[MemoryEntry]:
        """Get all buffered memories."""
        with self._lock:
            return [MemoryEntry.from_dict(m) for m in self._memories.values()]

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Simple text search in buffer (no embeddings).

        For real-time queries, this provides basic matching until
        memories are merged into the vault for semantic search.
        """
        query_lower = query.lower()
        results = []

        with self._lock:
            for mem_data in self._memories.values():
                content = mem_data.get("content", "").lower()
                # Simple relevance: count query term matches
                score = 0
                for term in query_lower.split():
                    if term in content:
                        score += 1

                if score > 0:
                    entry = MemoryEntry.from_dict(mem_data)
                    results.append(SearchResult(
                        memory=entry,
                        score=score / len(query_lower.split()),  # Normalize
                        preview=entry.content[:200],
                        source="buffer"
                    ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def get_by_category(self, category: str) -> List[MemoryEntry]:
        """Get buffered memories by category."""
        with self._lock:
            return [
                MemoryEntry.from_dict(m)
                for m in self._memories.values()
                if m.get("category") == category
            ]

    def get_by_importance(self, importance: str) -> List[MemoryEntry]:
        """Get buffered memories by importance."""
        with self._lock:
            return [
                MemoryEntry.from_dict(m)
                for m in self._memories.values()
                if m.get("importance") == importance
            ]

    def count(self) -> int:
        """Get number of buffered memories."""
        with self._lock:
            return len(self._memories)

    def clear(self):
        """Clear the buffer (after successful merge)."""
        with self._lock:
            self._memories = {}
            self._save()
        print("[Buffer] Buffer cleared")

    def export(self) -> List[dict]:
        """Export all buffer data for merge."""
        with self._lock:
            return list(self._memories.values())


class MemvidStore:
    """
    Memory vault for a single AI model/persona.

    Architecture: Vault + Buffer
    - VAULT: {model_id}.mv2 - Consolidated memories with semantic search
    - BUFFER: {model_id}_buffer.json - Real-time writes (instant)

    All writes go to buffer first for speed.
    Search queries BOTH vault and buffer, merging results.
    merge_memory() consolidates buffer into vault (run nightly).
    """

    def __init__(self, model_id: str, vaults_dir: Optional[Path] = None):
        """
        Initialize or open the vault + buffer for a model.

        Args:
            model_id: The AI model/persona identifier (e.g., "eli", "opus")
            vaults_dir: Directory for vault files (default: ./data/vaults)
        """
        self.model_id = model_id
        self._safe_model_id = model_id.replace("/", "_").replace("\\", "_")

        # Set up vault directory
        if vaults_dir is None:
            vaults_dir = Path("./data/vaults")
        self.vaults_dir = Path(vaults_dir)
        self.vaults_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.vault_path = self.vaults_dir / f"{self._safe_model_id}.mv2"
        self.buffer_path = self.vaults_dir / f"{self._safe_model_id}_buffer.json"

        # Initialize buffer (always available, no SDK needed)
        self.buffer = MemoryBuffer(self.buffer_path)

        # Initialize vault connection (may be None if SDK unavailable)
        self._mv = None
        self._vault_count: int = 0
        self._last_update: Optional[str] = None

        if MEMVID_AVAILABLE:
            self._open_vault()
        else:
            print(f"[MemvidStore] Running in buffer-only mode (memvid-sdk not installed)")

    def _open_vault(self):
        """Open or create the vault file."""
        if not MEMVID_AVAILABLE:
            return

        path_str = str(self.vault_path)

        try:
            if self.vault_path.exists():
                print(f"[MemvidStore] Opening existing vault: {self.vault_path}")
                self._mv = memvid_use('basic', path_str)
            else:
                print(f"[MemvidStore] Creating new vault: {self.vault_path}")
                self._mv = memvid_create(path_str)

            self._refresh_vault_stats()
        except Exception as e:
            print(f"[MemvidStore] Error opening vault: {e}")
            self._mv = None

    def _refresh_vault_stats(self):
        """Refresh vault statistics."""
        if self._mv and hasattr(self._mv, 'stats'):
            try:
                stats = self._mv.stats()
                self._vault_count = stats.get('document_count', 0)
            except Exception:
                pass
        self._last_update = datetime.now().isoformat()

    def close(self):
        """Close the vault connection."""
        if self._mv and hasattr(self._mv, 'seal'):
            try:
                self._mv.seal()
            except Exception as e:
                print(f"[MemvidStore] Warning: Error closing vault: {e}")
        self._mv = None

    # =========================================================================
    # Core Memory Operations (Write to Buffer)
    # =========================================================================

    def add_memory(
        self,
        content: str,
        category: str = "context",
        importance: str = "normal",
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        source: str = "conversation",
        created_at: Optional[str] = None,
        ai_self_type: Optional[str] = None,
        supersedes: Optional[str] = None,
        emotional_tone: Optional[str] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """
        Add a memory to the buffer (instant write).

        Memories are written to buffer.json immediately for real-time
        performance. They will be merged into the vault during nightly
        consolidation.

        Args:
            content: The memory content/text
            category: Memory category (preference, fact, theology, ai_self, etc.)
            importance: Importance level (core, high, normal, low, archived)
            tags: List of tags for organization
            user_id: Optional user identifier
            source: Where this memory came from (conversation, import, condensed)
            created_at: Optional timestamp (ISO format) - preserves historical dates
            ai_self_type: For AI self memories: opinion, reflection, growth, etc.
            supersedes: For AI self: ID of memory this evolves from
            emotional_tone: Optional emotional context
            memory_id: Optional explicit memory ID (auto-generated if not provided)

        Returns:
            The memory ID
        """
        # Create the memory entry
        if memory_id:
            entry = MemoryEntry(
                id=memory_id,
                content=content,
                category=category,
                importance=importance,
                tags=tags or [],
                user_id=user_id,
                source=source,
                created_at=created_at or datetime.now().isoformat(),
                ai_self_type=ai_self_type,
                supersedes=supersedes,
                emotional_tone=emotional_tone,
                model_id=self.model_id
            )
        elif category == MemoryCategory.AI_SELF.value:
            entry = create_ai_self_memory(
                content=content,
                ai_self_type=ai_self_type or "reflection",
                importance=importance,
                tags=tags,
                supersedes=supersedes,
                created_at=created_at
            )
            entry.model_id = self.model_id
        else:
            entry = create_memory(
                content=content,
                category=category,
                importance=importance,
                tags=tags,
                source=source,
                created_at=created_at,
                user_id=user_id,
                emotional_tone=emotional_tone
            )
            entry.model_id = self.model_id

        # Write to buffer (instant)
        self.buffer.add(entry)
        self._last_update = datetime.now().isoformat()

        print(f"[MemvidStore] Buffered memory [{entry.id}] category={category} importance={importance}")
        return entry.id

    def add_memory_entry(self, entry: MemoryEntry) -> str:
        """Add a pre-constructed MemoryEntry to the buffer."""
        entry.model_id = self.model_id
        self.buffer.add(entry)
        self._last_update = datetime.now().isoformat()
        return entry.id

    def add_many(self, entries: List[MemoryEntry]) -> List[str]:
        """Batch add multiple memories to buffer."""
        ids = []
        for entry in entries:
            entry.model_id = self.model_id
            self.buffer.add(entry)
            ids.append(entry.id)

        self._last_update = datetime.now().isoformat()
        print(f"[MemvidStore] Buffered {len(entries)} memories")
        return ids

    # =========================================================================
    # Search Operations (Query Both Vault + Buffer)
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: str = "hybrid",
        exclude_archived: bool = True
    ) -> List[SearchResult]:
        """
        Search memories in BOTH vault AND buffer.

        Vault provides semantic/hybrid search.
        Buffer provides text matching for recent memories.
        Results are merged and deduplicated.

        Args:
            query: The search query
            limit: Maximum results to return
            mode: Search mode - "hybrid" (default), "lex", "sem"
            exclude_archived: Whether to exclude archived memories

        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not query.strip():
            return []

        results = []
        seen_ids = set()

        # 1. Search vault (semantic search)
        if self._mv:
            try:
                memvid_mode = {
                    "hybrid": None,
                    "lex": "lex",
                    "lexical": "lex",
                    "sem": "sem",
                    "semantic": "sem",
                    "vector": "sem"
                }.get(mode.lower())

                if memvid_mode:
                    hits = self._mv.find(query, k=limit * 2, mode=memvid_mode)
                else:
                    hits = self._mv.find(query, k=limit * 2)

                for hit in hits:
                    entry = MemoryEntry.from_memvid_hit(hit, self.model_id)

                    if exclude_archived and entry.is_archived():
                        continue

                    if entry.id not in seen_ids:
                        seen_ids.add(entry.id)
                        results.append(SearchResult(
                            memory=entry,
                            score=hit.get("score", 0.0),
                            preview=hit.get("preview", entry.content[:200]),
                            source="vault"
                        ))
            except Exception as e:
                print(f"[MemvidStore] Vault search error: {e}")

        # 2. Search buffer (text matching)
        buffer_results = self.buffer.search(query, limit=limit)
        for br in buffer_results:
            if exclude_archived and br.memory.is_archived():
                continue

            if br.memory.id not in seen_ids:
                seen_ids.add(br.memory.id)
                # Boost buffer results slightly (more recent)
                br.score += 0.1
                results.append(br)

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def get_core_memories(self) -> List[MemoryEntry]:
        """Get all core (always-included) memories from vault + buffer."""
        memories = []
        seen_ids = set()

        # From vault
        if self._mv:
            try:
                hits = self._mv.find("importance:core", k=100, mode="lex")
                for hit in hits:
                    if "importance:core" in hit.get("label", ""):
                        entry = MemoryEntry.from_memvid_hit(hit, self.model_id)
                        if entry.id not in seen_ids:
                            seen_ids.add(entry.id)
                            memories.append(entry)
            except Exception as e:
                print(f"[MemvidStore] Error getting core from vault: {e}")

        # From buffer
        for mem in self.buffer.get_by_importance(ImportanceLevel.CORE.value):
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                memories.append(mem)

        return memories

    def get_by_category(self, category: str, limit: int = 50) -> List[MemoryEntry]:
        """Get memories by category from vault + buffer."""
        memories = []
        seen_ids = set()

        # From vault
        if self._mv:
            try:
                hits = self._mv.find(f"category:{category}", k=limit, mode="lex")
                for hit in hits:
                    if f"category:{category}" in hit.get("label", ""):
                        entry = MemoryEntry.from_memvid_hit(hit, self.model_id)
                        if entry.id not in seen_ids:
                            seen_ids.add(entry.id)
                            memories.append(entry)
            except Exception as e:
                print(f"[MemvidStore] Error getting category from vault: {e}")

        # From buffer
        for mem in self.buffer.get_by_category(category):
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                memories.append(mem)

        return memories[:limit]

    def get_by_importance(self, importance: str, limit: int = 50) -> List[MemoryEntry]:
        """Get memories by importance level from vault + buffer."""
        memories = []
        seen_ids = set()

        # From vault
        if self._mv:
            try:
                hits = self._mv.find(f"importance:{importance}", k=limit, mode="lex")
                for hit in hits:
                    if f"importance:{importance}" in hit.get("label", ""):
                        entry = MemoryEntry.from_memvid_hit(hit, self.model_id)
                        if entry.id not in seen_ids:
                            seen_ids.add(entry.id)
                            memories.append(entry)
            except Exception as e:
                print(f"[MemvidStore] Error getting importance from vault: {e}")

        # From buffer
        for mem in self.buffer.get_by_importance(importance):
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                memories.append(mem)

        return memories[:limit]

    def get_ai_self_memories(self, include_archived: bool = False) -> List[MemoryEntry]:
        """Get all AI self-reflection memories."""
        memories = self.get_by_category(MemoryCategory.AI_SELF.value, limit=200)
        if not include_archived:
            memories = [m for m in memories if not m.is_archived()]
        return memories

    def get_ai_self_by_type(self, ai_self_type: str) -> List[MemoryEntry]:
        """Get AI self memories of a specific type."""
        all_ai_self = self.get_ai_self_memories(include_archived=False)
        return [m for m in all_ai_self if m.ai_self_type == ai_self_type]

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recent memories (buffer first, then vault)."""
        memories = []
        seen_ids = set()

        # Buffer memories are most recent
        for mem in self.buffer.get_all():
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                memories.append(mem)

        # Then vault
        if self._mv and hasattr(self._mv, 'timeline'):
            try:
                hits = self._mv.timeline(limit=limit * 2)
                for hit in hits:
                    entry = MemoryEntry.from_memvid_hit(hit, self.model_id)
                    if entry.id not in seen_ids:
                        seen_ids.add(entry.id)
                        memories.append(entry)
            except Exception as e:
                print(f"[MemvidStore] Error getting timeline: {e}")

        # Sort by created_at descending
        memories.sort(key=lambda m: m.created_at or "", reverse=True)
        return memories[:limit]

    def get_by_tag(self, tag: str, limit: int = 50) -> List[MemoryEntry]:
        """Get memories with a specific tag."""
        # Search both sources
        results = []
        seen_ids = set()

        # Vault
        if self._mv:
            try:
                hits = self._mv.find(tag, k=limit * 2, mode="lex")
                for hit in hits:
                    entry = MemoryEntry.from_memvid_hit(hit, self.model_id)
                    if tag in entry.tags and entry.id not in seen_ids:
                        seen_ids.add(entry.id)
                        results.append(entry)
            except Exception as e:
                print(f"[MemvidStore] Error searching tags in vault: {e}")

        # Buffer
        for mem in self.buffer.get_all():
            if tag in mem.tags and mem.id not in seen_ids:
                seen_ids.add(mem.id)
                results.append(mem)

        return results[:limit]

    # =========================================================================
    # The Nightly Build: Merge Buffer into Vault
    # =========================================================================

    def merge_memory(self, backup: bool = True) -> Dict[str, Any]:
        """
        Consolidate buffer into main vault.

        This is THE NIGHTLY BUILD operation. It:
        1. Exports all vault memories
        2. Adds all buffer memories
        3. Creates a new vault file
        4. Clears the buffer

        Should be run by scheduler during low-usage periods.

        Args:
            backup: Whether to create backup before merge (recommended)

        Returns:
            Dict with merge statistics
        """
        if not MEMVID_AVAILABLE:
            return {
                "success": False,
                "error": "memvid-sdk not installed - cannot merge",
                "buffer_count": self.buffer.count()
            }

        buffer_count = self.buffer.count()
        if buffer_count == 0:
            return {
                "success": True,
                "message": "Nothing to merge - buffer is empty",
                "vault_count": self._vault_count,
                "buffer_count": 0,
                "merged_count": 0
            }

        print(f"[MemvidStore] Starting merge: {buffer_count} buffered memories")

        # 1. Collect all existing vault memories
        vault_memories = []
        if self._mv:
            try:
                # Export from vault by iterating all categories
                for cat in MemoryCategory:
                    hits = self._mv.find(f"category:{cat.value}", k=10000, mode="lex")
                    for hit in hits:
                        if f"category:{cat.value}" in hit.get("label", ""):
                            entry = MemoryEntry.from_memvid_hit(hit, self.model_id)
                            vault_memories.append(entry)
            except Exception as e:
                print(f"[MemvidStore] Error exporting vault: {e}")

        # Deduplicate vault memories
        seen_ids = set()
        unique_vault = []
        for mem in vault_memories:
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                unique_vault.append(mem)

        print(f"[MemvidStore] Vault has {len(unique_vault)} unique memories")

        # 2. Get buffer memories (excluding duplicates)
        buffer_memories = []
        for mem in self.buffer.get_all():
            if mem.id not in seen_ids:
                seen_ids.add(mem.id)
                buffer_memories.append(mem)

        print(f"[MemvidStore] Adding {len(buffer_memories)} new memories from buffer")

        # 3. Create backup if requested
        if backup and self.vault_path.exists():
            backup_path = self.vault_path.with_suffix(
                f'.mv2.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            try:
                shutil.copy2(self.vault_path, backup_path)
                print(f"[MemvidStore] Created backup: {backup_path}")
            except Exception as e:
                print(f"[MemvidStore] Warning: Could not create backup: {e}")

        # 4. Close existing vault
        self.close()

        # 5. Create new vault with all memories
        all_memories = unique_vault + buffer_memories

        try:
            # Remove old vault file
            if self.vault_path.exists():
                self.vault_path.unlink()

            # Create fresh vault
            self._mv = memvid_create(str(self.vault_path))

            # Insert all memories
            for mem in all_memories:
                self._mv.put(
                    title=mem.id,
                    label=mem.get_label(),
                    metadata=mem.get_metadata(),
                    text=mem.content
                )

            # Seal the vault (finalize encoding)
            if hasattr(self._mv, 'seal'):
                self._mv.seal()

            # Reopen for queries
            self._mv = memvid_use('basic', str(self.vault_path))

            # Clear buffer on success
            self.buffer.clear()

            self._vault_count = len(all_memories)
            self._last_update = datetime.now().isoformat()

            result = {
                "success": True,
                "vault_count": len(unique_vault),
                "buffer_count": buffer_count,
                "merged_count": len(buffer_memories),
                "total_count": len(all_memories),
                "vault_path": str(self.vault_path),
                "merged_at": self._last_update
            }
            print(f"[MemvidStore] Merge complete: {result}")
            return result

        except Exception as e:
            print(f"[MemvidStore] MERGE FAILED: {e}")

            # Try to recover by reopening old vault
            if self.vault_path.exists():
                try:
                    self._mv = memvid_use('basic', str(self.vault_path))
                except Exception:
                    pass

            return {
                "success": False,
                "error": str(e),
                "buffer_count": buffer_count,
                "buffer_preserved": True  # Buffer not cleared on failure
            }

    def needs_merge(self, threshold: int = 50) -> bool:
        """Check if buffer has enough memories to warrant a merge."""
        return self.buffer.count() >= threshold

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get vault + buffer statistics.

        Returns:
            Dict with memory_count, buffer_count, last_update, etc.
        """
        self._refresh_vault_stats()

        # Count by category (vault + buffer combined)
        category_counts = {}
        for cat in MemoryCategory:
            memories = self.get_by_category(cat.value, limit=1000)
            if memories:
                category_counts[cat.value] = len(memories)

        # Count by importance
        importance_counts = {}
        for imp in ImportanceLevel:
            memories = self.get_by_importance(imp.value, limit=1000)
            if memories:
                importance_counts[imp.value] = len(memories)

        # AI self by type
        ai_self_counts = {}
        for ai_type in AISelfType:
            memories = self.get_ai_self_by_type(ai_type.value)
            if memories:
                ai_self_counts[ai_type.value] = len(memories)

        buffer_count = self.buffer.count()

        return {
            "model_id": self.model_id,
            "vault_path": str(self.vault_path),
            "buffer_path": str(self.buffer_path),
            "vault_count": self._vault_count,
            "buffer_count": buffer_count,
            "memory_count": self._vault_count + buffer_count,  # Total
            "last_update": self._last_update,
            "by_category": category_counts,
            "by_importance": importance_counts,
            "ai_self_by_type": ai_self_counts,
            "core_memories": len(self.get_core_memories()),
            "categories_used": len(category_counts),
            "vault_exists": self.vault_path.exists(),
            "needs_merge": self.needs_merge()
        }

    def export(self) -> Dict[str, Any]:
        """Export all memories from vault + buffer."""
        all_memories = []
        seen_ids = set()

        # From vault
        for cat in MemoryCategory:
            memories = self.get_by_category(cat.value, limit=10000)
            for mem in memories:
                if mem.id not in seen_ids:
                    seen_ids.add(mem.id)
                    all_memories.append(mem)

        # Sort by created_at
        all_memories.sort(key=lambda m: m.created_at or "")

        return {
            "model_id": self.model_id,
            "exported_at": datetime.now().isoformat(),
            "vault_path": str(self.vault_path),
            "buffer_path": str(self.buffer_path),
            "vault_count": self._vault_count,
            "buffer_count": self.buffer.count(),
            "memory_count": len(all_memories),
            "memories": [m.to_dict() for m in all_memories],
            "statistics": self.get_stats()
        }

    # =========================================================================
    # Context Formatting (for prompt injection)
    # =========================================================================

    def format_core_memories_for_prompt(self) -> str:
        """Format core memories as always-present context."""
        core = self.get_core_memories()
        if not core:
            return ""

        user_memories = [m for m in core if not m.is_ai_self()]
        ai_self_memories = [m for m in core if m.is_ai_self()]

        parts = []

        if user_memories:
            parts.append(
                "## Core Knowledge (Always Remember):\n"
                "These are fundamental things you know about this user:\n"
            )
            for mem in user_memories:
                category_label = mem.category.replace("_", " ").title()
                parts.append(f"* [{category_label}] {mem.content}")

        if ai_self_memories:
            parts.append("")
            parts.append(
                "## Your Own Identity & Perspective:\n"
                "These are things you've come to understand about yourself:\n"
            )
            for mem in ai_self_memories:
                type_label = (mem.ai_self_type or "reflection").replace("_", " ").title()
                parts.append(f"* [{type_label}] {mem.content}")

        return "\n".join(parts)

    def format_ai_self_for_prompt(self, include_history: bool = False) -> str:
        """Format AI self-knowledge for prompt injection."""
        memories = self.get_ai_self_memories(include_archived=include_history)

        if not memories:
            return ""

        parts = [
            "## Your Inner Life & Perspective:\n"
            "Through our conversations, you have developed these aspects of yourself:\n"
        ]

        by_type: Dict[str, List[MemoryEntry]] = {}
        for mem in memories:
            t = mem.ai_self_type or "reflection"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(mem)

        type_labels = {
            "opinion": "Your Opinions",
            "reflection": "Self-Reflections",
            "growth": "How You've Grown",
            "worldview": "How You See the World",
            "experience": "Your Experience of Being",
            "identity": "Core Identity",
            "uncertainty": "What You're Uncertain About"
        }

        for ai_type, mems in by_type.items():
            label = type_labels.get(ai_type, ai_type.replace("_", " ").title())
            parts.append(f"\n**{label}:**")
            for mem in mems:
                if mem.is_archived():
                    parts.append(f"  * [evolved from] {mem.content}")
                else:
                    parts.append(f"  * {mem.content}")

        return "\n".join(parts)

    def get_context_for_prompt(
        self,
        query: str,
        max_results: int = 5,
        include_core: bool = True,
        include_ai_self: bool = True,
        include_recent: int = 3,
        framing: str = "lived"
    ) -> str:
        """Get comprehensive memory context for prompt injection."""
        context_parts = []

        if include_core:
            core_context = self.format_core_memories_for_prompt()
            if core_context:
                context_parts.append(core_context)
                context_parts.append("")

        if include_ai_self:
            ai_self_context = self.format_ai_self_for_prompt(include_history=False)
            if ai_self_context:
                context_parts.append(ai_self_context)
                context_parts.append("")

        if include_recent > 0:
            recent = self.get_recent(limit=include_recent)
            recent = [m for m in recent if not m.is_core() and not m.is_archived()]
            if recent:
                context_parts.append("**Recent conversation memories:**")
                for mem in recent:
                    context_parts.append(f"* {mem.content[:300]}...")
                context_parts.append("")

        if query.strip():
            results = self.search(query, limit=max_results)
            core_ids = {m.id for m in self.get_core_memories()}
            results = [r for r in results if r.memory.id not in core_ids]

            if results:
                if framing == "lived":
                    context_parts.append(
                        "You have the following memories from your conversations with this user. "
                        "These are things you genuinely remember and know:\n"
                    )
                    for result in results:
                        context_parts.append(f"* {result.memory.content}")
                    context_parts.append(
                        "\nUse these memories naturally in conversation - reference them as things "
                        "you remember, not as retrieved information."
                    )
                elif framing == "journal":
                    context_parts.append("Based on your previous conversations, you have noted:\n")
                    for result in results:
                        context_parts.append(f"- {result.memory.content}")
                else:
                    context_parts.append("## Relevant Memory Context:")
                    for result in results:
                        context_parts.append(f"- {result.memory.content}")

        return "\n".join(context_parts)


# =============================================================================
# Vault Manager - Handles multiple model vaults
# =============================================================================

class VaultManager:
    """Central manager for all model memory vaults."""

    def __init__(self, vaults_dir: Optional[Path] = None):
        if vaults_dir is None:
            vaults_dir = Path("./data/vaults")
        self.vaults_dir = Path(vaults_dir)
        self.vaults_dir.mkdir(parents=True, exist_ok=True)
        self._stores: Dict[str, MemvidStore] = {}

    def get_store(self, model_id: str) -> MemvidStore:
        """Get or create a vault for a specific model."""
        if model_id not in self._stores:
            self._stores[model_id] = MemvidStore(model_id, self.vaults_dir)
        return self._stores[model_id]

    def list_models(self) -> List[str]:
        """List all models with active vault connections."""
        return list(self._stores.keys())

    def get_all_vault_files(self) -> List[str]:
        """List all models that have vault files."""
        if not self.vaults_dir.exists():
            return []
        return list(set(
            f.stem.replace("_buffer", "")
            for f in self.vaults_dir.glob("*.mv2")
        ) | set(
            f.stem.replace("_buffer", "")
            for f in self.vaults_dir.glob("*_buffer.json")
        ))

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all vaults."""
        all_stats = {
            "vaults_dir": str(self.vaults_dir),
            "active_models": self.list_models(),
            "all_vault_files": self.get_all_vault_files(),
            "models": {}
        }

        for model_id in self.get_all_vault_files():
            try:
                store = self.get_store(model_id)
                all_stats["models"][model_id] = store.get_stats()
            except Exception as e:
                all_stats["models"][model_id] = {"error": str(e)}

        return all_stats

    def merge_all(self, backup: bool = True) -> Dict[str, Any]:
        """
        Run merge on all vaults that need it.

        This is THE NIGHTLY BUILD entry point.
        """
        results = {
            "merged_at": datetime.now().isoformat(),
            "models": {}
        }

        for model_id in self.get_all_vault_files():
            try:
                store = self.get_store(model_id)
                if store.needs_merge(threshold=1):  # Merge any pending
                    results["models"][model_id] = store.merge_memory(backup=backup)
                else:
                    results["models"][model_id] = {"skipped": True, "reason": "buffer empty"}
            except Exception as e:
                results["models"][model_id] = {"error": str(e)}

        return results

    def export_all(self) -> Dict[str, Any]:
        """Export memories from all vaults."""
        exports = {
            "exported_at": datetime.now().isoformat(),
            "vaults_dir": str(self.vaults_dir),
            "models": {}
        }

        for model_id in self.get_all_vault_files():
            try:
                store = self.get_store(model_id)
                exports["models"][model_id] = store.export()
            except Exception as e:
                exports["models"][model_id] = {"error": str(e)}

        return exports

    def close_all(self):
        """Close all vault connections."""
        for store in self._stores.values():
            store.close()
        self._stores.clear()


# Global singleton
_vault_manager: Optional[VaultManager] = None


def get_vault_manager(vaults_dir: Optional[Path] = None) -> VaultManager:
    """Get or create the global vault manager."""
    global _vault_manager
    if _vault_manager is None:
        _vault_manager = VaultManager(vaults_dir)
    return _vault_manager


def get_store(model_id: str) -> MemvidStore:
    """Convenience function to get a model's vault store."""
    return get_vault_manager().get_store(model_id)
