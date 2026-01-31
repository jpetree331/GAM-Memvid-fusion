"""
GAM Memory Manager - Handles per-model memory stores.

Each OpenWebUI custom model gets its own isolated memory bucket.
This includes custom models that use the same underlying LLM but have
different system prompts/instructions.
"""
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from config import config


@dataclass
class MemoryResult:
    """Result from a memory search operation."""
    content: str
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)
    timestamp: Optional[datetime] = None


class ModelMemoryStore:
    """
    Memory store for a single OpenWebUI model.
    Wraps GAM's MemoryAgent and ResearchAgent.
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.data_dir = config.get_model_data_dir(model_id)
        self._initialized = False
        
        # GAM components (lazy initialized)
        self._generator = None
        self._memory_store = None
        self._page_store = None
        self._memory_agent = None
        self._research_agent = None
        self._retrievers = {}
    
    def _ensure_initialized(self):
        """Lazy initialization of GAM components."""
        if self._initialized:
            return
        
        try:
            from gam import (
                MemoryAgent,
                ResearchAgent,
                OpenAIGenerator,
                OpenAIGeneratorConfig,
                InMemoryMemoryStore,
                InMemoryPageStore,
                DenseRetrieverConfig,
                DenseRetriever,
                IndexRetrieverConfig,
                IndexRetriever,
                BM25RetrieverConfig,
                BM25Retriever,
            )
        except ImportError:
            raise ImportError(
                "GAM is not installed. Install it with:\n"
                "pip install git+https://github.com/VectorSpaceLab/general-agentic-memory.git"
            )
        
        # Configure generator
        gen_config = OpenAIGeneratorConfig(
            model_name=config.GAM_MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            temperature=0.3,
            max_tokens=512
        )
        self._generator = OpenAIGenerator.from_config(gen_config)
        
        # Create memory and page stores
        self._memory_store = InMemoryMemoryStore()
        self._page_store = InMemoryPageStore()
        
        # Create memory agent
        self._memory_agent = MemoryAgent(
            generator=self._generator,
            memory_store=self._memory_store,
            page_store=self._page_store
        )
        
        # Set up retrievers
        self._setup_retrievers(
            IndexRetriever, IndexRetrieverConfig,
            BM25Retriever, BM25RetrieverConfig,
            DenseRetriever, DenseRetrieverConfig
        )
        
        self._initialized = True
        
        # Rebuild GAM memories from organization index (persistence layer)
        self._reload_from_organization_index()
    
    def _reload_from_organization_index(self):
        """
        Reload memories from the organization index into GAM's in-memory stores.
        
        This ensures GAM has all memories after a server restart.
        The organization index (JSON file) is our persistence layer.
        
        IMPORTANT: This bypasses the MemoryAgent.memorize() to avoid API calls.
        We directly populate the page_store since memories are already processed.
        NO API CALLS ARE MADE DURING RELOAD - this is completely FREE.
        """
        try:
            from memory_organization import get_organizer
            organizer = get_organizer(self.model_id)
            
            memories = list(organizer._memories.values())
            if not memories:
                print(f"[GAM] No memories to reload for {self.model_id}")
                return
            
            print(f"[GAM] Reloading {len(memories)} memories for {self.model_id} (FREE - no API calls)...")
            
            # Import Page from GAM (correct path: gam.schemas or gam directly)
            try:
                from gam import Page
            except ImportError:
                try:
                    from gam.schemas import Page
                except ImportError:
                    print(f"[GAM] Page class not available, skipping GAM store reload")
                    print(f"[GAM] Memories are still accessible via organization index")
                    return
            
            loaded_count = 0
            for mem in memories:
                try:
                    # Reconstruct the memory content with timestamp if available
                    content = mem.content
                    if mem.created_at:
                        content = f"[{mem.created_at}] {content}"
                    
                    # Create a header from the memory (first 100 chars or category)
                    header = f"[{getattr(mem, 'category', 'memory')}] {content[:100]}"
                    
                    # BYPASS the memory_agent.memorize() - that uses the API!
                    # Instead, directly add to page_store (this is FREE)
                    # Page expects: header (str), content (str), meta (dict)
                    page = Page(
                        header=header,
                        content=content,
                        meta={
                            "id": mem.id,
                            "category": getattr(mem, 'category', 'context'),
                            "importance": getattr(mem, 'importance', 'normal'),
                            "source": "reload",
                            "created_at": getattr(mem, 'created_at', None)
                        }
                    )
                    self._page_store.add(page)
                    loaded_count += 1
                except Exception as e:
                    # Log the error but continue trying other memories
                    if loaded_count == 0:
                        print(f"[GAM] First Page creation failed: {e}")
                        # Try one more time with minimal fields
                        try:
                            simple_page = Page(header="Memory", content=mem.content, meta={})
                            self._page_store.add(simple_page)
                            loaded_count += 1
                            print(f"[GAM] Simplified Page creation succeeded")
                        except Exception as e2:
                            print(f"[GAM] Simplified Page also failed: {e2}")
                            print(f"[GAM] Skipping GAM store reload - using organization index only")
                            return
            
            # Rebuild retrievers after loading all memories (also FREE - just indexing)
            if loaded_count > 0:
                self._rebuild_retrievers()
            
            print(f"[GAM] Successfully reloaded {loaded_count}/{len(memories)} memories (FREE)")
            
        except Exception as e:
            print(f"[GAM] Error reloading memories for {self.model_id}: {e}")
            print(f"[GAM] Memories are still accessible via organization index")
    
    def _rebuild_retrievers(self):
        """Rebuild retriever indexes after loading memories."""
        try:
            from gam import (
                DenseRetrieverConfig,
                DenseRetriever,
                IndexRetrieverConfig,
                IndexRetriever,
                BM25RetrieverConfig,
                BM25Retriever,
            )
            
            # Re-setup retrievers to index the newly loaded pages
            self._setup_retrievers(
                IndexRetriever, IndexRetrieverConfig,
                BM25Retriever, BM25RetrieverConfig,
                DenseRetriever, DenseRetrieverConfig
            )
            print(f"[GAM] Rebuilt retriever indexes for {self.model_id}")
        except Exception as e:
            print(f"[GAM] Warning: Failed to rebuild retrievers: {e}")
    
    def _setup_retrievers(
        self,
        IndexRetriever, IndexRetrieverConfig,
        BM25Retriever, BM25RetrieverConfig,
        DenseRetriever, DenseRetrieverConfig
    ):
        """Set up the retrieval system for this model's memory."""
        index_dir = self.data_dir / "indexes"
        
        # Page index retriever
        try:
            page_index_dir = index_dir / "page_index"
            if page_index_dir.exists():
                shutil.rmtree(page_index_dir)
            page_index_dir.mkdir(parents=True, exist_ok=True)
            
            index_config = IndexRetrieverConfig(index_dir=str(page_index_dir))
            index_retriever = IndexRetriever(index_config.__dict__)
            index_retriever.build(self._page_store)
            self._retrievers["page_index"] = index_retriever
        except Exception as e:
            print(f"[WARN] Page index retriever setup failed for {self.model_id}: {e}")
        
        # BM25 keyword retriever
        try:
            bm25_dir = index_dir / "bm25_index"
            if bm25_dir.exists():
                shutil.rmtree(bm25_dir)
            bm25_dir.mkdir(parents=True, exist_ok=True)
            
            bm25_config = BM25RetrieverConfig(index_dir=str(bm25_dir), threads=1)
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(self._page_store)
            self._retrievers["keyword"] = bm25_retriever
        except Exception as e:
            print(f"[WARN] BM25 retriever setup failed for {self.model_id}: {e}")
        
        # Dense vector retriever
        try:
            dense_dir = index_dir / "dense_index"
            if dense_dir.exists():
                shutil.rmtree(dense_dir)
            dense_dir.mkdir(parents=True, exist_ok=True)
            
            dense_config = DenseRetrieverConfig(
                index_dir=str(dense_dir),
                model_name=config.GAM_EMBEDDING_MODEL
            )
            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(self._page_store)
            self._retrievers["vector"] = dense_retriever
        except Exception as e:
            print(f"[WARN] Dense retriever setup failed for {self.model_id}: {e}")
    
    def _get_research_agent(self):
        """Get or create the research agent."""
        self._ensure_initialized()
        
        if self._research_agent is None:
            from gam import ResearchAgent
            
            self._research_agent = ResearchAgent(
                page_store=self._page_store,
                memory_store=self._memory_store,
                retrievers=self._retrievers,
                generator=self._generator,
                max_iters=config.GAM_MAX_RESEARCH_ITERS  # Configurable: default 2 (was 5)
            )
        
        return self._research_agent
    
    def add_memory(
        self,
        content: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[dict] = None,
        category: Optional[str] = None,
        importance: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> str:
        """
        Add a memory to this model's store.
        
        Args:
            content: The conversation or information to memorize
            user_id: Optional user identifier for namespacing
            session_id: Optional session identifier
            timestamp: Optional historical timestamp (for imports)
            metadata: Additional metadata to store
            category: Memory category (preference, fact, event, etc.)
            importance: Importance level (core, high, normal, low)
            tags: List of tags for organization
        
        Returns:
            Memory ID
        """
        self._ensure_initialized()
        
        # Build the content with metadata context
        memory_content = content
        if timestamp:
            memory_content = f"[{timestamp.isoformat()}] {content}"
        
        # Memorize using GAM
        self._memory_agent.memorize(memory_content)
        
        # Generate memory ID
        memory_id = f"{self.model_id}_{datetime.now().timestamp()}"
        
        # Always add to organization index for dashboard visibility
        from memory_organization import get_organizer
        organizer = get_organizer(self.model_id)
        organizer.add(
            memory_id=memory_id,
            content=content,
            category=category or "context",
            importance=importance or "normal",
            tags=tags or [],
            user_id=user_id,
            source="conversation",
            created_at=timestamp.isoformat() if timestamp else None
        )
        
        return memory_id
    
    def search(self, query: str, limit: int = 5) -> list[MemoryResult]:
        """
        Search memories relevant to the query.
        
        Args:
            query: The search query
            limit: Maximum number of results
        
        Returns:
            List of relevant memories
        """
        self._ensure_initialized()
        
        research_agent = self._get_research_agent()
        
        try:
            result = research_agent.research(request=query)
            
            # Convert GAM result to our format
            memories = []
            if result and result.integrated_memory:
                memories.append(MemoryResult(
                    content=result.integrated_memory,
                    relevance_score=1.0,
                    metadata={"source": "gam_research"}
                ))
            
            return memories[:limit]
        
        except Exception as e:
            print(f"[ERROR] Search failed for {self.model_id}: {e}")
            return []
    
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
            max_tokens: Approximate token limit for context
            framing: How to present memories:
                - "lived": First-person, integrated experience (recommended)
                - "rag": Traditional "retrieved documents" style
                - "journal": Second-person observational style
            include_core: Whether to include core memories (always-present)
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Always include core memories first (identity-defining)
        if include_core:
            try:
                from memory_organization import get_organizer
                organizer = get_organizer(self.model_id)
                
                total_memories = len(organizer._memories)
                print(f"[GAM] get_context_for_prompt: {self.model_id} has {total_memories} total memories")
                
                # Core memories (user knowledge + AI identity)
                core_context = organizer.format_core_memories_for_prompt()
                if core_context:
                    context_parts.append(core_context)
                    context_parts.append("")  # Blank line separator
                    print(f"[GAM] Added core memories context ({len(core_context)} chars)")
                else:
                    print(f"[GAM] No core memories found for {self.model_id}")
                
                # AI Self context (non-core but still important for continuity)
                ai_self_context = organizer.format_ai_self_for_prompt(include_history=False)
                if ai_self_context:
                    context_parts.append(ai_self_context)
                    context_parts.append("")
                    print(f"[GAM] Added AI self context ({len(ai_self_context)} chars)")
                
                # ALSO include recent memories for personality continuity
                # Get 5 most recent non-core memories
                recent_memories = sorted(
                    [m for m in organizer._memories.values() if m.importance != "core"],
                    key=lambda x: x.created_at or "",
                    reverse=True
                )[:5]
                
                if recent_memories:
                    context_parts.append("\n**Recent conversation memories:**\n")
                    for mem in recent_memories:
                        context_parts.append(f"• {mem.content[:300]}...")
                    print(f"[GAM] Added {len(recent_memories)} recent memories")
                    
            except Exception as e:
                print(f"[WARN] Failed to load core memories: {e}")
                import traceback
                traceback.print_exc()
        
        # Search for relevant memories
        memories = self.search(query, limit=5)
        
        # FALLBACK: If GAM search returns nothing, search organization index directly
        # This handles the case where GAM page_store wasn't populated (e.g., Page import failed)
        if not memories:
            try:
                from memory_organization import get_organizer
                organizer = get_organizer(self.model_id)
                
                # Simple keyword search on organization index
                query_words = set(query.lower().split())
                scored_memories = []
                
                for mem in organizer._memories.values():
                    # Skip if already in core (to avoid duplicates)
                    if mem.importance == "core":
                        continue
                    
                    content_lower = mem.content.lower()
                    # Score by keyword matches
                    score = sum(1 for word in query_words if word in content_lower and len(word) > 3)
                    if score > 0:
                        scored_memories.append((score, mem))
                
                # Sort by score descending, take top 5
                scored_memories.sort(key=lambda x: x[0], reverse=True)
                memories = [
                    MemoryResult(content=mem.content, relevance_score=score/10, metadata={"source": "org_index"})
                    for score, mem in scored_memories[:5]
                ]
                
                if memories:
                    print(f"[GAM] Fallback search found {len(memories)} memories from organization index")
                    
            except Exception as e:
                print(f"[WARN] Fallback search failed: {e}")
        
        if not memories and not context_parts:
            return ""
        
        if memories:
            if framing == "lived":
                # First-person framing - feels like genuine recall
                context_parts.append(
                    "You have the following memories from your conversations with this user. "
                    "These are things you genuinely remember and know - not external documents, "
                    "but your own experiences and learned knowledge about them:\n"
                )
                for mem in memories:
                    content = mem.content
                    context_parts.append(f"• {content}")
                
                context_parts.append(
                    "\nUse these memories naturally in conversation - reference them as things "
                    "you remember, not as retrieved information. If the user asks about something "
                    "you remember, respond as if recalling a genuine experience."
                )
            
            elif framing == "journal":
                context_parts.append("Based on your previous conversations, you have noted:\n")
                for mem in memories:
                    context_parts.append(f"- {mem.content}")
            
            else:  # "rag" - traditional style
                context_parts.append("## Relevant Memory Context:")
                for mem in memories:
                    context_parts.append(f"- {mem.content}")
        
        return "\n".join(context_parts)


class MemoryManager:
    """
    Central manager for all model memory stores.
    Handles creation and retrieval of per-model memory buckets.
    """
    
    def __init__(self):
        self._stores: dict[str, ModelMemoryStore] = {}
    
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
    
    def list_models(self) -> list[str]:
        """List all models with active memory stores."""
        return list(self._stores.keys())
    
    def get_all_model_dirs(self) -> list[str]:
        """List all models that have data directories (including inactive)."""
        models_dir = config.DATA_DIR / "models"
        if not models_dir.exists():
            return []
        return [d.name for d in models_dir.iterdir() if d.is_dir()]
    
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
        store._ensure_initialized()
        
        export_data = {
            "model_id": model_id,
            "exported_at": datetime.now().isoformat(),
            "format_version": "1.0",
            "memories": [],
            "pages": []
        }
        
        # Export from memory store
        if store._memory_store:
            try:
                memory_state = store._memory_store.load()
                if memory_state and hasattr(memory_state, 'abstracts'):
                    for abstract in memory_state.abstracts:
                        export_data["memories"].append({
                            "content": str(abstract),
                            "type": "abstract"
                        })
            except Exception as e:
                export_data["memory_export_error"] = str(e)
        
        # Export from page store
        if store._page_store:
            try:
                pages = store._page_store.load_all() if hasattr(store._page_store, 'load_all') else []
                for page in pages:
                    export_data["pages"].append({
                        "content": str(page.content) if hasattr(page, 'content') else str(page),
                        "metadata": page.metadata if hasattr(page, 'metadata') else {}
                    })
            except Exception as e:
                export_data["page_export_error"] = str(e)
        
        return export_data
    
    def export_all_memories(self) -> dict:
        """Export memories from all models."""
        all_exports = {
            "exported_at": datetime.now().isoformat(),
            "models": {}
        }
        
        for model_id in self.get_all_model_dirs():
            try:
                all_exports["models"][model_id] = self.export_memories(model_id)
            except Exception as e:
                all_exports["models"][model_id] = {"error": str(e)}
        
        return all_exports


# Global singleton
memory_manager = MemoryManager()
