"""
=============================================================================
LEGACY FILE - NOT USED IN PRODUCTION
=============================================================================

This file is DEPRECATED and kept only for backward compatibility with
legacy import scripts (archive_thread.py, import_conversations.py).

The new Librarian architecture uses:
- memvid_store.py (MemvidStore) for storage
- synthesizer.py (Synthesizer) for retrieval
- server.py calls these directly

Do NOT import this file in new code. Use memvid_store.py instead.

=============================================================================

Memory Manager - Thin wrapper around Memvid Store.

This provides the interface expected by legacy code while delegating
all operations to the MemvidStore (Memvid v2).
"""
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from config import config
from memvid_store import (
    MemvidStore,
    VaultManager,
    SearchResult,
    get_vault_manager,
    get_store
)
from memory_entry import (
    MemoryEntry,
    MemoryCategory,
    ImportanceLevel,
    AISelfType,
    create_memory,
    create_ai_self_memory
)


@dataclass
class MemoryResult:
    """
    Result from a memory search operation.

    This maintains backward compatibility with the old interface.
    """
    content: str
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "MemoryResult":
        """Convert from MemvidStore SearchResult."""
        timestamp = None
        if result.memory.created_at:
            try:
                timestamp = datetime.fromisoformat(result.memory.created_at)
            except (ValueError, TypeError):
                pass

        return cls(
            content=result.memory.content,
            relevance_score=result.score,
            metadata={
                "memory_id": result.memory.id,
                "category": result.memory.category,
                "importance": result.memory.importance,
                "tags": result.memory.tags,
                "source": result.memory.source
            },
            timestamp=timestamp
        )


class ModelMemoryStore:
    """
    Memory store for a single OpenWebUI model.

    This is a thin wrapper around MemvidStore that provides
    the interface expected by server.py and other components.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._store: Optional[MemvidStore] = None

    def _ensure_initialized(self):
        """Lazy initialization of the Memvid store."""
        if self._store is None:
            self._store = get_store(self.model_id)

    @property
    def store(self) -> MemvidStore:
        """Get the underlying MemvidStore."""
        self._ensure_initialized()
        return self._store

    def add_memory(
        self,
        content: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[dict] = None,
        category: Optional[str] = None,
        importance: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Add a memory to this model's vault.

        Args:
            content: The conversation or information to memorize
            user_id: Optional user identifier for namespacing
            session_id: Optional session identifier
            timestamp: Optional historical timestamp (for imports)
            metadata: Additional metadata to store
            category: Memory category (preference, fact, event, task, etc.)
            importance: Importance level (core, high, normal, low)
            tags: List of tags for organization

        Returns:
            Memory ID
        """
        self._ensure_initialized()

        # Convert timestamp to ISO string if provided
        created_at = None
        if timestamp:
            created_at = timestamp.isoformat()

        return self._store.add_memory(
            content=content,
            category=category or "context",
            importance=importance or "normal",
            tags=tags or [],
            user_id=user_id,
            source="conversation",
            created_at=created_at
        )

    def search(self, query: str, limit: int = 5) -> List[MemoryResult]:
        """
        Search memories relevant to the query.

        Args:
            query: The search query
            limit: Maximum number of results

        Returns:
            List of relevant memories
        """
        self._ensure_initialized()

        results = self._store.search(query, limit=limit)

        return [MemoryResult.from_search_result(r) for r in results]

    def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 2000,
        framing: str = "lived",
        include_core: bool = True
    ) -> str:
        """
        Get relevant memory context to inject into a prompt.

        Args:
            query: The user's current query
            max_tokens: Approximate token limit for context (advisory)
            framing: How to present memories:
                - "lived": First-person, integrated experience (recommended)
                - "rag": Traditional "retrieved documents" style
                - "journal": Second-person observational style
            include_core: Whether to include core memories (always-present)

        Returns:
            Formatted context string
        """
        self._ensure_initialized()

        return self._store.get_context_for_prompt(
            query=query,
            max_results=5,
            include_core=include_core,
            include_ai_self=True,
            include_recent=3,
            framing=framing
        )

    def get_stats(self) -> dict:
        """Get statistics for this model's vault."""
        self._ensure_initialized()
        return self._store.get_stats()

    def export(self) -> dict:
        """Export all memories from this vault."""
        self._ensure_initialized()
        return self._store.export()


class MemoryManager:
    """
    Central manager for all model memory stores.

    Handles creation and retrieval of per-model memory vaults.
    """

    def __init__(self):
        self._stores: dict[str, ModelMemoryStore] = {}
        self._vault_manager: Optional[VaultManager] = None

    def _get_vault_manager(self) -> VaultManager:
        """Get or create the vault manager."""
        if self._vault_manager is None:
            self._vault_manager = get_vault_manager(config.VAULTS_DIR)
        return self._vault_manager

    def get_store(self, model_id: str) -> ModelMemoryStore:
        """
        Get or create a memory store for a specific model.

        Args:
            model_id: The OpenWebUI model identifier (can be custom model name)

        Returns:
            The model's memory store
        """
        if model_id not in self._stores:
            self._stores[model_id] = ModelMemoryStore(model_id)
        return self._stores[model_id]

    def list_models(self) -> List[str]:
        """List all models with active memory stores."""
        return list(self._stores.keys())

    def get_all_model_dirs(self) -> List[str]:
        """
        List all models that have vault files.

        Note: This now checks the vaults directory, not the legacy models directory.
        """
        vm = self._get_vault_manager()
        return vm.get_all_vault_files()

    def export_memories(self, model_id: str, format: str = "json") -> dict:
        """
        Export all memories for a model.

        Args:
            model_id: The model to export memories from
            format: Export format ('json' supported)

        Returns:
            Exportable dictionary of memories
        """
        store = self.get_store(model_id)
        return store.export()

    def export_all_memories(self) -> dict:
        """Export memories from all models."""
        vm = self._get_vault_manager()
        return vm.export_all()

    def get_all_stats(self) -> dict:
        """Get statistics for all vaults."""
        vm = self._get_vault_manager()
        return vm.get_all_stats()


# =============================================================================
# Legacy Compatibility Layer
# =============================================================================

class MemoryOrganizer:
    """
    Compatibility layer that wraps MemvidStore.

    This provides the interface that server.py endpoints expect
    (get_by_category, get_core_memories, etc.) while using Memvid storage.

    For new code, use MemvidStore directly.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._store: Optional[MemvidStore] = None

    @property
    def store(self) -> MemvidStore:
        """Get the underlying MemvidStore."""
        if self._store is None:
            self._store = get_store(self.model_id)
        return self._store

    @property
    def _memories(self) -> dict:
        """
        Compatibility property that returns memories as a dict.

        WARNING: This loads all memories into memory. Use sparingly.
        """
        export = self.store.export()
        memories = {}
        for mem_dict in export.get("memories", []):
            entry = MemoryEntry.from_dict(mem_dict)
            memories[entry.id] = entry
        return memories

    def _save(self):
        """No-op for compatibility. Memvid auto-saves."""
        pass

    def _save_index(self):
        """No-op for compatibility. Memvid auto-saves."""
        pass

    def add(
        self,
        memory_id: str,
        content: str,
        category: str = "context",
        importance: str = "normal",
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        source: str = "conversation",
        created_at: Optional[str] = None
    ) -> MemoryEntry:
        """Add a memory via the compatibility layer."""
        self.store.add_memory(
            content=content,
            category=category,
            importance=importance,
            tags=tags,
            user_id=user_id,
            source=source,
            created_at=created_at,
            memory_id=memory_id
        )
        return MemoryEntry(
            id=memory_id,
            content=content,
            category=category,
            importance=importance,
            tags=tags or [],
            user_id=user_id,
            source=source,
            created_at=created_at or datetime.now().isoformat()
        )

    def update(
        self,
        memory_id: str,
        category: Optional[str] = None,
        importance: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[MemoryEntry]:
        """
        Update is not directly supported in append-only Memvid.

        For updates, add a new memory with supersedes field.
        """
        # Note: This is a limitation - Memvid is append-only
        # Real updates would need to add new entry and archive old
        print(f"[MemoryOrganizer] Update not directly supported in Memvid. Memory: {memory_id}")
        return None

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        # Search for the memory ID
        results = self.store.search(memory_id, limit=10, mode="lex")
        for r in results:
            if r.memory.id == memory_id:
                return r.memory
        return None

    def get_core_memories(self) -> List[MemoryEntry]:
        """Get all core (always-present) memories."""
        return self.store.get_core_memories()

    def get_by_category(self, category: str) -> List[MemoryEntry]:
        """Get all memories in a category."""
        return self.store.get_by_category(category)

    def get_by_tag(self, tag: str) -> List[MemoryEntry]:
        """Get all memories with a specific tag."""
        return self.store.get_by_tag(tag)

    def get_by_importance(self, importance: str) -> List[MemoryEntry]:
        """Get all memories at a specific importance level."""
        return self.store.get_by_importance(importance)

    def get_all_tags(self) -> List[str]:
        """Get all unique tags across all memories."""
        # This requires loading all memories
        all_tags = set()
        export = self.store.export()
        for mem_dict in export.get("memories", []):
            all_tags.update(mem_dict.get("tags", []))
        return sorted(all_tags)

    def get_all_categories(self) -> dict:
        """Get category counts."""
        stats = self.store.get_stats()
        return stats.get("by_category", {})

    def get_statistics(self) -> dict:
        """Get organization statistics."""
        return self.store.get_stats()

    def format_core_memories_for_prompt(self) -> str:
        """Format core memories for prompt injection."""
        return self.store.format_core_memories_for_prompt()

    # AI Self methods
    def add_ai_self(
        self,
        memory_id: str,
        content: str,
        ai_self_type: str = "reflection",
        importance: str = "normal",
        tags: Optional[List[str]] = None,
        supersedes: Optional[str] = None,
        created_at: Optional[str] = None
    ) -> MemoryEntry:
        """Add an AI self-reflection memory."""
        self.store.add_memory(
            content=content,
            category=MemoryCategory.AI_SELF.value,
            importance=importance,
            tags=tags,
            ai_self_type=ai_self_type,
            supersedes=supersedes,
            created_at=created_at,
            memory_id=memory_id
        )
        return MemoryEntry(
            id=memory_id,
            content=content,
            category=MemoryCategory.AI_SELF.value,
            importance=importance,
            tags=tags or [],
            ai_self_type=ai_self_type,
            supersedes=supersedes,
            created_at=created_at or datetime.now().isoformat()
        )

    def get_ai_self_memories(self) -> List[MemoryEntry]:
        """Get all AI self-reflection memories."""
        return self.store.get_ai_self_memories()

    def get_ai_self_by_type(self, ai_self_type: str) -> List[MemoryEntry]:
        """Get AI self memories of a specific type."""
        return self.store.get_ai_self_by_type(ai_self_type)

    def get_ai_opinions(self) -> List[MemoryEntry]:
        """Get all AI opinions."""
        return self.get_ai_self_by_type(AISelfType.OPINION.value)

    def get_current_ai_opinions(self) -> List[MemoryEntry]:
        """Get current (non-superseded) AI opinions."""
        opinions = self.get_ai_opinions()
        return [o for o in opinions if o.importance != ImportanceLevel.ARCHIVED.value]

    def get_ai_growth_timeline(self) -> List[MemoryEntry]:
        """Get AI self memories ordered by time."""
        ai_self = self.get_ai_self_memories()
        return sorted(ai_self, key=lambda m: m.created_at or "")

    def format_ai_self_for_prompt(self, include_history: bool = False) -> str:
        """Format AI self-knowledge for prompt injection."""
        return self.store.format_ai_self_for_prompt(include_history=include_history)

    def export(self) -> dict:
        """Export all organization data."""
        return self.store.export()


# Cache of organizers per model (for compatibility)
_organizers: dict[str, MemoryOrganizer] = {}


def get_organizer(model_id: str) -> MemoryOrganizer:
    """
    Get or create a memory organizer for a model.

    This is a compatibility function for code that used memory_organization.py.
    For new code, use get_store() directly.
    """
    if model_id not in _organizers:
        _organizers[model_id] = MemoryOrganizer(model_id)
    return _organizers[model_id]


# Global singleton
memory_manager = MemoryManager()
