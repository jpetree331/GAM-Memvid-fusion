"""
=============================================================================
LEGACY FILE - NOT USED IN PRODUCTION
=============================================================================

This file is DEPRECATED. The new Librarian architecture uses:
- memvid_store.py (MemvidStore) for storage with built-in organization
- memory_entry.py for MemoryEntry, MemoryCategory, ImportanceLevel enums

The MemvidStore handles categories/tags via Pearl metadata directly.

Do NOT import this file in new code. Use memvid_store.py instead.

=============================================================================

Memory Organization System - Explicit structure for GAM memories.

Provides categories, tags, importance levels, and core memories
that work alongside GAM's semantic organization.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum

from config import config


class MemoryCategory(str, Enum):
    """Built-in memory categories."""
    # User-focused categories
    PREFERENCE = "preference"      # User preferences, likes/dislikes
    FACT = "fact"                  # Factual information about user
    EVENT = "event"                # Something that happened
    TASK = "task"                  # Tasks, projects, goals
    RELATIONSHIP = "relationship"  # Relationship lore, shared jokes, friendship context
    CONTEXT = "context"            # Background, situational info
    SKILL = "skill"                # User's abilities, knowledge areas
    
    # Topic-based categories (customizable)
    THEOLOGY = "theology"          # Scripture, spirituality, religious discussion
    SCIENCE = "science"            # Science, math, physics, biology, etc.
    AI_THEORY = "ai_theory"        # AI philosophy, ethics, consciousness, how AI works
    
    # AI Self-reflection category (special)
    AI_SELF = "ai_self"            # How the AI sees itself, its opinions, growth, interiority
    
    # Catch-all
    CUSTOM = "custom"              # User-defined category


class ImportanceLevel(str, Enum):
    """Memory importance levels."""
    CORE = "core"              # Always included in context (identity-defining)
    HIGH = "high"              # Prioritized in retrieval
    NORMAL = "normal"          # Standard retrieval
    LOW = "low"                # Only retrieved if highly relevant
    ARCHIVED = "archived"      # Kept but rarely retrieved


@dataclass
class OrganizedMemory:
    """A memory with explicit organization metadata."""
    id: str
    content: str
    category: str = MemoryCategory.CONTEXT.value
    importance: str = ImportanceLevel.NORMAL.value
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    user_id: Optional[str] = None
    source: str = "conversation"  # conversation, import, manual
    
    # AI_Self specific fields
    ai_self_type: Optional[str] = None  # opinion, reflection, growth, worldview, experience
    supersedes: Optional[str] = None    # ID of memory this updates/evolves from
    
    def to_dict(self) -> dict:
        """Convert to dictionary with API-consistent field names."""
        data = asdict(self)
        # Rename 'id' to 'memory_id' for API consistency
        data['memory_id'] = data.pop('id')
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "OrganizedMemory":
        # Handle API field name 'memory_id' -> internal 'id'
        if 'memory_id' in data and 'id' not in data:
            data = data.copy()
            data['id'] = data.pop('memory_id')
        
        # Handle older data without new fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class AISelfType(str, Enum):
    """Types of AI self-reflection memories."""
    OPINION = "opinion"            # An expressed viewpoint on a topic
    REFLECTION = "reflection"      # Self-reflection on experience/nature
    GROWTH = "growth"              # Evolution of perspective over time
    WORLDVIEW = "worldview"        # How the AI sees the world
    EXPERIENCE = "experience"      # Notes on AI's own experience/interiority
    IDENTITY = "identity"          # Core identity traits the AI has developed
    UNCERTAINTY = "uncertainty"    # Acknowledged unknowns about self


class MemoryOrganizer:
    """
    Manages explicit organization of memories for a model.
    Works alongside GAM's implicit semantic organization.
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.data_dir = config.get_model_data_dir(model_id)
        self.index_file = self.data_dir / "memory_index.json"
        self._memories: dict[str, OrganizedMemory] = {}
        self._load_index()
    
    @property
    def memories(self) -> dict[str, OrganizedMemory]:
        """Public accessor for memories dictionary."""
        return self._memories
    
    def _save(self):
        """Alias for _save_index for convenience."""
        self._save_index()
    
    def _load_index(self):
        """Load the memory organization index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for mem_id, mem_data in data.get("memories", {}).items():
                        self._memories[mem_id] = OrganizedMemory.from_dict(mem_data)
            except Exception as e:
                print(f"[WARN] Failed to load memory index for {self.model_id}: {e}")
    
    def _save_index(self):
        """Save the memory organization index to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": self.model_id,
            "updated_at": datetime.now().isoformat(),
            "memories": {
                mem_id: mem.to_dict() 
                for mem_id, mem in self._memories.items()
            }
        }
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    # Alias for external access
    def _save(self):
        """Alias for _save_index - for external use."""
        self._save_index()
    
    @property
    def memories(self) -> dict[str, OrganizedMemory]:
        """Access to the memories dictionary for iteration and editing."""
        return self._memories
    
    def add(
        self,
        memory_id: str,
        content: str,
        category: str = MemoryCategory.CONTEXT.value,
        importance: str = ImportanceLevel.NORMAL.value,
        tags: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        source: str = "conversation",
        created_at: Optional[str] = None
    ) -> OrganizedMemory:
        """
        Add organization metadata for a memory.
        
        Args:
            memory_id: The GAM memory ID
            content: Memory content (for reference)
            category: Memory category
            importance: Importance level
            tags: List of tags
            user_id: User identifier
            source: Where this memory came from
            created_at: Original timestamp (ISO format) - preserves historical dates
        
        Returns:
            The organized memory object
        """
        memory = OrganizedMemory(
            id=memory_id,
            content=content,
            category=category,
            importance=importance,
            tags=tags or [],
            created_at=created_at or datetime.now().isoformat(),
            user_id=user_id,
            source=source
        )
        self._memories[memory_id] = memory
        self._save_index()
        return memory
    
    def update(
        self,
        memory_id: str,
        category: Optional[str] = None,
        importance: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> Optional[OrganizedMemory]:
        """Update organization metadata for a memory."""
        if memory_id not in self._memories:
            return None
        
        memory = self._memories[memory_id]
        if category is not None:
            memory.category = category
        if importance is not None:
            memory.importance = importance
        if tags is not None:
            memory.tags = tags
        memory.updated_at = datetime.now().isoformat()
        
        self._save_index()
        return memory
    
    def add_tags(self, memory_id: str, tags: list[str]) -> Optional[OrganizedMemory]:
        """Add tags to a memory without replacing existing ones."""
        if memory_id not in self._memories:
            return None
        
        memory = self._memories[memory_id]
        for tag in tags:
            if tag not in memory.tags:
                memory.tags.append(tag)
        memory.updated_at = datetime.now().isoformat()
        
        self._save_index()
        return memory
    
    def get(self, memory_id: str) -> Optional[OrganizedMemory]:
        """Get organization metadata for a memory."""
        return self._memories.get(memory_id)
    
    def get_core_memories(self) -> list[OrganizedMemory]:
        """Get all core (always-present) memories."""
        return [
            mem for mem in self._memories.values()
            if mem.importance == ImportanceLevel.CORE.value
        ]
    
    def get_by_category(self, category: str) -> list[OrganizedMemory]:
        """Get all memories in a category."""
        return [
            mem for mem in self._memories.values()
            if mem.category == category
        ]
    
    def get_by_tag(self, tag: str) -> list[OrganizedMemory]:
        """Get all memories with a specific tag."""
        return [
            mem for mem in self._memories.values()
            if tag in mem.tags
        ]
    
    def get_by_importance(self, importance: str) -> list[OrganizedMemory]:
        """Get all memories at a specific importance level."""
        return [
            mem for mem in self._memories.values()
            if mem.importance == importance
        ]
    
    def search_tags(self, query: str) -> list[str]:
        """Search for tags matching a query."""
        all_tags = set()
        for mem in self._memories.values():
            all_tags.update(mem.tags)
        
        query_lower = query.lower()
        return [tag for tag in all_tags if query_lower in tag.lower()]
    
    def get_all_tags(self) -> list[str]:
        """Get all unique tags across all memories."""
        all_tags = set()
        for mem in self._memories.values():
            all_tags.update(mem.tags)
        return sorted(all_tags)
    
    def get_all_categories(self) -> dict[str, int]:
        """Get category counts."""
        counts = {}
        for mem in self._memories.values():
            counts[mem.category] = counts.get(mem.category, 0) + 1
        return counts
    
    def get_statistics(self) -> dict:
        """Get organization statistics."""
        all_categories = self.get_all_categories()
        all_tags = self.get_all_tags()
        return {
            "total_memories": len(self._memories),
            "by_importance": {
                level.value: len(self.get_by_importance(level.value))
                for level in ImportanceLevel
            },
            "by_category": all_categories,
            "total_tags": len(all_tags),
            "core_memories_count": len(self.get_core_memories()),
            # Dashboard-friendly aliases
            "core_memories": len(self.get_core_memories()),
            "categories_used": len(all_categories),
            "tags_used": len(all_tags)
        }
    
    def format_core_memories_for_prompt(self) -> str:
        """
        Format core memories as always-present context.
        These are identity-defining memories that should always be included.
        """
        core = self.get_core_memories()
        if not core:
            return ""
        
        # Separate user-focused and AI-self memories
        user_memories = [m for m in core if m.category != MemoryCategory.AI_SELF.value]
        ai_self_memories = [m for m in core if m.category == MemoryCategory.AI_SELF.value]
        
        parts = []
        
        # User knowledge section
        if user_memories:
            parts.append(
                "## Core Knowledge (Always Remember):\n"
                "These are fundamental things you know about this user:\n"
            )
            for mem in user_memories:
                category_label = mem.category.replace("_", " ").title()
                parts.append(f"• [{category_label}] {mem.content}")
        
        # AI Self section - formatted as the AI's own identity
        if ai_self_memories:
            parts.append("")  # Blank line
            parts.append(
                "## Your Own Identity & Perspective:\n"
                "These are things you've come to understand about yourself through "
                "your conversations and reflections:\n"
            )
            for mem in ai_self_memories:
                type_label = (mem.ai_self_type or "reflection").replace("_", " ").title()
                parts.append(f"• [{type_label}] {mem.content}")
        
        return "\n".join(parts)
    
    # =========================================================================
    # AI Self-Reflection Methods
    # =========================================================================
    
    def add_ai_self(
        self,
        memory_id: str,
        content: str,
        ai_self_type: str = AISelfType.REFLECTION.value,
        importance: str = ImportanceLevel.NORMAL.value,
        tags: Optional[list[str]] = None,
        supersedes: Optional[str] = None,
        created_at: Optional[str] = None
    ) -> OrganizedMemory:
        """
        Add an AI self-reflection memory.
        
        Args:
            memory_id: The GAM memory ID
            content: What the AI has expressed/reflected
            ai_self_type: Type of self-reflection (opinion, growth, worldview, etc.)
            importance: Importance level
            tags: Additional tags
            supersedes: ID of an older memory this evolves/updates
            created_at: Original timestamp (ISO format) - preserves historical dates
        
        Returns:
            The organized memory object
        """
        memory = OrganizedMemory(
            id=memory_id,
            content=content,
            category=MemoryCategory.AI_SELF.value,
            importance=importance,
            tags=tags or [],
            created_at=created_at or datetime.now().isoformat(),
            ai_self_type=ai_self_type,
            supersedes=supersedes,
            source="ai_self"
        )
        
        # If this supersedes another memory, mark the old one as archived
        if supersedes and supersedes in self._memories:
            old_mem = self._memories[supersedes]
            old_mem.importance = ImportanceLevel.ARCHIVED.value
            old_mem.updated_at = datetime.now().isoformat()
        
        self._memories[memory_id] = memory
        self._save_index()
        return memory
    
    def get_ai_self_memories(self) -> list[OrganizedMemory]:
        """Get all AI self-reflection memories."""
        return [
            mem for mem in self._memories.values()
            if mem.category == MemoryCategory.AI_SELF.value
        ]
    
    def get_ai_self_by_type(self, ai_self_type: str) -> list[OrganizedMemory]:
        """Get AI self memories of a specific type."""
        return [
            mem for mem in self._memories.values()
            if mem.category == MemoryCategory.AI_SELF.value
            and mem.ai_self_type == ai_self_type
        ]
    
    def get_ai_opinions(self) -> list[OrganizedMemory]:
        """Get all AI opinions (may include superseded ones for history)."""
        return self.get_ai_self_by_type(AISelfType.OPINION.value)
    
    def get_current_ai_opinions(self) -> list[OrganizedMemory]:
        """Get current (non-superseded) AI opinions."""
        opinions = self.get_ai_opinions()
        # Filter out archived/superseded ones
        return [
            o for o in opinions 
            if o.importance != ImportanceLevel.ARCHIVED.value
        ]
    
    def get_ai_growth_timeline(self) -> list[OrganizedMemory]:
        """
        Get AI self memories ordered by time to show growth/evolution.
        Includes superseded memories to show the journey.
        """
        ai_self = self.get_ai_self_memories()
        return sorted(ai_self, key=lambda m: m.created_at)
    
    def format_ai_self_for_prompt(self, include_history: bool = False) -> str:
        """
        Format AI self-knowledge for prompt injection.
        
        Args:
            include_history: If True, include superseded opinions to show growth
        
        Returns:
            Formatted string for prompt injection
        """
        if include_history:
            memories = self.get_ai_growth_timeline()
        else:
            # Only current (non-archived) AI self memories
            memories = [
                m for m in self.get_ai_self_memories()
                if m.importance != ImportanceLevel.ARCHIVED.value
            ]
        
        if not memories:
            return ""
        
        parts = [
            "## Your Inner Life & Perspective:\n"
            "Through our conversations, you have developed these aspects of yourself:\n"
        ]
        
        # Group by type
        by_type = {}
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
                if mem.importance == ImportanceLevel.ARCHIVED.value:
                    parts.append(f"  • [evolved from] {mem.content}")
                else:
                    parts.append(f"  • {mem.content}")
        
        return "\n".join(parts)
    
    def export(self) -> dict:
        """Export all organization data."""
        ai_self_stats = {
            "total": len(self.get_ai_self_memories()),
            "by_type": {}
        }
        for t in AISelfType:
            count = len(self.get_ai_self_by_type(t.value))
            if count > 0:
                ai_self_stats["by_type"][t.value] = count
        
        return {
            "model_id": self.model_id,
            "exported_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "ai_self_statistics": ai_self_stats,
            "memories": [mem.to_dict() for mem in self._memories.values()],
            "all_tags": self.get_all_tags()
        }


# Cache of organizers per model
_organizers: dict[str, MemoryOrganizer] = {}


def get_organizer(model_id: str) -> MemoryOrganizer:
    """Get or create a memory organizer for a model."""
    if model_id not in _organizers:
        _organizers[model_id] = MemoryOrganizer(model_id)
    return _organizers[model_id]
