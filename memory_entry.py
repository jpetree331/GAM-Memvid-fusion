"""
Memory Entry - Lightweight dataclass for memory storage.

This replaces the gam.Page class with a standalone implementation
that has no external dependencies.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List
from enum import Enum


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


class AISelfType(str, Enum):
    """Types of AI self-reflection memories."""
    OPINION = "opinion"            # An expressed viewpoint on a topic
    REFLECTION = "reflection"      # Self-reflection on experience/nature
    GROWTH = "growth"              # Evolution of perspective over time
    WORLDVIEW = "worldview"        # How the AI sees the world
    EXPERIENCE = "experience"      # Notes on AI's own experience/interiority
    IDENTITY = "identity"          # Core identity traits the AI has developed
    UNCERTAINTY = "uncertainty"    # Acknowledged unknowns about self


@dataclass
class MemoryEntry:
    """
    A single memory entry with all metadata.

    This is a lightweight, standalone replacement for gam.Page
    that works with the Memvid storage layer.
    """
    id: str
    content: str
    category: str = MemoryCategory.CONTEXT.value
    importance: str = ImportanceLevel.NORMAL.value
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    user_id: Optional[str] = None
    source: str = "conversation"  # conversation, import, manual, condensed

    # AI Self-specific fields
    ai_self_type: Optional[str] = None  # opinion, reflection, growth, worldview, experience
    supersedes: Optional[str] = None    # ID of memory this updates/evolves from

    # Additional metadata (extensible)
    emotional_tone: Optional[str] = None
    model_id: Optional[str] = None  # Which AI model this memory belongs to

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Use memory_id for API consistency
        data['memory_id'] = data.pop('id')
        # Remove None values for cleaner output
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Create from dictionary (handles both 'id' and 'memory_id' keys)."""
        data = data.copy()

        # Handle API field name 'memory_id' -> internal 'id'
        if 'memory_id' in data and 'id' not in data:
            data['id'] = data.pop('memory_id')

        # Handle older data without new fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def get_label(self) -> str:
        """
        Generate the Memvid label for this memory.

        Format: "category:{category}|importance:{importance}"
        This enables fast filtering without full-text search.
        """
        return f"category:{self.category}|importance:{self.importance}"

    def get_metadata(self) -> dict:
        """
        Get metadata dict for Memvid storage.

        Includes all fields except content (which goes in 'text').
        """
        meta = {
            "tags": self.tags,
            "created_at": self.created_at,
            "source": self.source,
        }

        # Optional fields
        if self.updated_at:
            meta["updated_at"] = self.updated_at
        if self.user_id:
            meta["user_id"] = self.user_id
        if self.ai_self_type:
            meta["ai_self_type"] = self.ai_self_type
        if self.supersedes:
            meta["supersedes"] = self.supersedes
        if self.emotional_tone:
            meta["emotional_tone"] = self.emotional_tone
        if self.model_id:
            meta["model_id"] = self.model_id

        return meta

    @classmethod
    def from_memvid_hit(cls, hit: dict, model_id: str) -> "MemoryEntry":
        """
        Create a MemoryEntry from a Memvid search hit.

        Args:
            hit: Dict with 'frame_id', 'score', 'preview', and metadata
            model_id: The model ID this memory belongs to

        Returns:
            MemoryEntry instance
        """
        # Extract metadata (Memvid stores this in the hit)
        meta = hit.get("metadata", {})

        # Parse label to extract category and importance
        label = hit.get("label", "category:context|importance:normal")
        category = "context"
        importance = "normal"

        for part in label.split("|"):
            if part.startswith("category:"):
                category = part.replace("category:", "")
            elif part.startswith("importance:"):
                importance = part.replace("importance:", "")

        return cls(
            id=hit.get("frame_id", hit.get("title", "")),
            content=hit.get("preview", hit.get("text", "")),
            category=category,
            importance=importance,
            tags=meta.get("tags", []),
            created_at=meta.get("created_at", datetime.now().isoformat()),
            updated_at=meta.get("updated_at"),
            user_id=meta.get("user_id"),
            source=meta.get("source", "unknown"),
            ai_self_type=meta.get("ai_self_type"),
            supersedes=meta.get("supersedes"),
            emotional_tone=meta.get("emotional_tone"),
            model_id=model_id
        )

    def is_core(self) -> bool:
        """Check if this is a core (always-included) memory."""
        return self.importance == ImportanceLevel.CORE.value

    def is_ai_self(self) -> bool:
        """Check if this is an AI self-reflection memory."""
        return self.category == MemoryCategory.AI_SELF.value

    def is_archived(self) -> bool:
        """Check if this memory is archived."""
        return self.importance == ImportanceLevel.ARCHIVED.value


# Convenience functions for creating specific memory types

def create_memory(
    content: str,
    category: str = "context",
    importance: str = "normal",
    tags: Optional[List[str]] = None,
    source: str = "conversation",
    created_at: Optional[str] = None,
    **kwargs
) -> MemoryEntry:
    """
    Factory function to create a memory entry with auto-generated ID.

    Args:
        content: The memory content
        category: Memory category
        importance: Importance level
        tags: Optional list of tags
        source: Where this memory came from
        created_at: Optional timestamp (defaults to now)
        **kwargs: Additional fields (user_id, emotional_tone, etc.)

    Returns:
        MemoryEntry with unique ID
    """
    memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    return MemoryEntry(
        id=memory_id,
        content=content,
        category=category,
        importance=importance,
        tags=tags or [],
        source=source,
        created_at=created_at or datetime.now().isoformat(),
        **kwargs
    )


def create_ai_self_memory(
    content: str,
    ai_self_type: str = "reflection",
    importance: str = "normal",
    tags: Optional[List[str]] = None,
    supersedes: Optional[str] = None,
    created_at: Optional[str] = None
) -> MemoryEntry:
    """
    Factory function to create an AI self-reflection memory.

    Args:
        content: The AI's reflection, opinion, or observation
        ai_self_type: Type of self-reflection (opinion, reflection, growth, etc.)
        importance: Importance level
        tags: Optional list of tags
        supersedes: ID of memory this evolves from
        created_at: Optional timestamp (defaults to now)

    Returns:
        MemoryEntry configured for AI self-reflection
    """
    memory_id = f"ai_self_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    return MemoryEntry(
        id=memory_id,
        content=content,
        category=MemoryCategory.AI_SELF.value,
        importance=importance,
        tags=tags or [],
        source="ai_self",
        created_at=created_at or datetime.now().isoformat(),
        ai_self_type=ai_self_type,
        supersedes=supersedes
    )
