"""
GAM-Memvid Server - Librarian Architecture REST API

This server exposes the Librarian memory system to OpenWebUI via REST endpoints.

Architecture:
- MemvidStore: Stores full, raw conversations as "Pearls" using Super-Index
- Synthesizer: Creates detailed abstracts at retrieval time (gpt-4o-mini)
- Per-model vault isolation (each model_id gets its own .mv2 file)

Endpoints:
- POST /memory/add - Store a new Pearl (user+AI exchange)
- POST /memory/context - Get synthesized context for prompt injection
- GET /health - Health check for Railway
- GET /models - List available models
- GET /memory/{model_id}/stats - Get vault statistics
- POST /memory/{model_id}/delete - Soft delete a Pearl
"""
import os
import asyncio
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import config
from memvid_store import get_store, get_vault_manager, MemvidStore, Pearl
from synthesizer import get_synthesizer, Synthesizer


# =============================================================================
# Request/Response Models
# =============================================================================

class AddPearlRequest(BaseModel):
    """Request to add a Pearl (conversation exchange)."""
    model_id: str = Field(..., description="Model/persona identifier (e.g., 'eli', 'opus')")
    user_message: str = Field(..., description="The user's complete message")
    ai_response: str = Field(..., description="The AI's complete response")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for indexing")
    category: Optional[str] = Field(default="context", description="Category: context, theology, ai_theory, etc.")
    importance: Optional[str] = Field(default="normal", description="Importance: core, high, normal, low")
    user_name: Optional[str] = Field(default="User", description="User's display name")


class AddPearlResponse(BaseModel):
    """Response from adding a Pearl."""
    pearl_id: str
    status: str = "ok"
    word_count: int = 0


class ContextRequest(BaseModel):
    """Request for synthesized context."""
    model_id: str = Field(..., description="Model/persona identifier")
    query: str = Field(..., description="Current user message or search query")
    user_name: Optional[str] = Field(default="User", description="User's display name")
    limit: Optional[int] = Field(default=5, ge=1, le=20, description="Max Pearls to retrieve")
    max_words: Optional[int] = Field(default=400, ge=100, le=2000, description="Target context word count")


class ContextResponse(BaseModel):
    """Response with synthesized context."""
    context: str = Field(..., description="Synthesized context for prompt injection")
    num_pearls: int = Field(..., description="Number of Pearls used")
    has_memories: bool = Field(..., description="Whether any memories were found")


class SearchRequest(BaseModel):
    """Request to search memories."""
    model_id: str = Field(..., description="Model/persona identifier")
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")


class SearchResponse(BaseModel):
    """Response from searching memories."""
    results: List[dict]
    count: int


class DeleteRequest(BaseModel):
    """Request to soft-delete a Pearl."""
    pearl_id: str = Field(..., description="ID of the Pearl to delete")
    reason: Optional[str] = Field(default=None, description="Reason for deletion")


class DeleteResponse(BaseModel):
    """Response from deleting a Pearl."""
    success: bool
    pearl_id: str
    message: str


# =============================================================================
# Shared Instances
# =============================================================================

# These are initialized at module load time
# VaultManager handles per-model vault isolation
_vault_manager = None
_synthesizer = None


def get_vault_mgr():
    """Get the global VaultManager instance."""
    global _vault_manager
    if _vault_manager is None:
        _vault_manager = get_vault_manager(config.VAULTS_DIR)
    return _vault_manager


def get_synth():
    """Get the global Synthesizer instance."""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = get_synthesizer()
    return _synthesizer


# =============================================================================
# Lifespan Handler
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Validate config on startup
    errors = config.validate()
    if errors:
        print("=" * 60)
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nCopy .env.example to .env and configure it.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("GAM-Memvid Librarian Server")
        print("=" * 60)
        print(f"Host: {config.HOST}:{config.PORT}")
        print(f"Vaults directory: {config.VAULTS_DIR}")
        print(f"Embedding model: {config.MEMVID_EMBEDDING_MODEL}")
        print(f"OpenAI configured: {'Yes' if config.OPENAI_API_KEY else 'No'}")
        print("=" * 60)

    yield

    # Cleanup on shutdown
    print("[Server] Shutting down...")
    if _vault_manager:
        _vault_manager.close_all()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="GAM-Memvid Librarian Server",
    description="AI memory system with Memvid storage and runtime synthesis",
    version="2.1.0",
    lifespan=lifespan
)

# CORS for OpenWebUI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    vault_mgr = get_vault_mgr()
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "vault_files": len(vault_mgr.get_all_vault_files()),
        "active_connections": len(vault_mgr.list_models())
    }


# =============================================================================
# Core Memory Endpoints
# =============================================================================

@app.post("/memory/add", response_model=AddPearlResponse)
async def add_pearl(request: AddPearlRequest):
    """
    Store a new Pearl (conversation exchange).

    This is called by OpenWebUI's outlet() after each turn.
    The full user message and AI response are stored without truncation.
    """
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(request.model_id)

        pearl_id = store.add_pearl(
            user_message=request.user_message,
            ai_response=request.ai_response,
            tags=request.tags or [],
            category=request.category or "context",
            importance=request.importance or "normal",
            user_name=request.user_name or "User"
        )

        word_count = len(request.user_message.split()) + len(request.ai_response.split())

        return AddPearlResponse(
            pearl_id=pearl_id,
            status="ok",
            word_count=word_count
        )

    except Exception as e:
        print(f"[Server] Error adding Pearl: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    """
    Get synthesized context for prompt injection.

    This is called by OpenWebUI's inlet() before each turn.
    It retrieves relevant Pearls and synthesizes them into a context string.
    """
    try:
        vault_mgr = get_vault_mgr()
        synthesizer = get_synth()

        # Get the store for this model
        store = vault_mgr.get_store(request.model_id)

        # Search for relevant Pearls
        pearls = store.get_raw_pearls_for_synthesis(
            query=request.query,
            limit=request.limit or 5
        )

        if not pearls:
            return ContextResponse(
                context="",
                num_pearls=0,
                has_memories=False
            )

        # Synthesize into context
        context = await synthesizer.synthesize_for_context(
            pearls=pearls,
            user_name=request.user_name or "User",
            max_context_words=request.max_words or 400
        )

        return ContextResponse(
            context=context,
            num_pearls=len(pearls),
            has_memories=True
        )

    except Exception as e:
        print(f"[Server] Error getting context: {e}")
        # Fail open - return empty context rather than error
        return ContextResponse(
            context="",
            num_pearls=0,
            has_memories=False
        )


@app.post("/memory/search", response_model=SearchResponse)
async def search_memories(request: SearchRequest):
    """Search for relevant Pearls (without synthesis)."""
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(request.model_id)

        results = store.search_pearls(
            query=request.query,
            limit=request.limit
        )

        return SearchResponse(
            results=[r.to_dict() for r in results],
            count=len(results)
        )

    except Exception as e:
        print(f"[Server] Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/{model_id}/delete", response_model=DeleteResponse)
async def delete_pearl(model_id: str, request: DeleteRequest):
    """Soft-delete a Pearl."""
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)

        success = store.soft_delete(
            pearl_id=request.pearl_id,
            reason=request.reason
        )

        return DeleteResponse(
            success=success,
            pearl_id=request.pearl_id,
            message="Pearl soft-deleted" if success else "Failed to delete Pearl"
        )

    except Exception as e:
        print(f"[Server] Error deleting Pearl: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Model/Vault Management
# =============================================================================

@app.get("/models")
async def list_models():
    """List all models with vaults."""
    vault_mgr = get_vault_mgr()
    return {
        "active_models": vault_mgr.list_models(),
        "all_vault_files": vault_mgr.get_all_vault_files()
    }


@app.get("/memory/{model_id}/stats")
async def get_model_stats(model_id: str):
    """Get statistics for a model's vault."""
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)
        return store.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{model_id}/export")
async def export_model(model_id: str):
    """Export all Pearls from a model's vault."""
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)
        return store.export()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{model_id}/recent")
async def get_recent_memories(
    model_id: str,
    limit: int = Query(default=10, ge=1, le=50)
):
    """Get most recent memories for a model."""
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)
        memories = store.get_recent(limit=limit)
        return {
            "model_id": model_id,
            "count": len(memories),
            "memories": [m.to_dict() for m in memories]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Legacy Compatibility Endpoints
# =============================================================================
# These maintain backward compatibility with the old memory_manager API

class LegacyAddRequest(BaseModel):
    """Legacy request format for adding memory."""
    model_id: str
    content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict] = None
    category: Optional[str] = None
    importance: Optional[str] = None
    tags: Optional[List[str]] = None


@app.post("/memory/add/legacy")
async def add_memory_legacy(request: LegacyAddRequest):
    """
    Legacy endpoint for backward compatibility.

    Converts the old content-based format to Pearl format.
    """
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(request.model_id)

        # Parse content into user/AI parts if possible
        user_message = request.content
        ai_response = ""

        for splitter in ["\nAI:", "\nAI responded:", "\nI (AI) responded:"]:
            if splitter in request.content:
                parts = request.content.split(splitter, 1)
                user_message = parts[0].replace("User said:", "").replace("User:", "").strip()
                ai_response = parts[1].strip()
                break

        pearl_id = store.add_pearl(
            user_message=user_message,
            ai_response=ai_response,
            tags=request.tags or [],
            category=request.category or "context",
            importance=request.importance or "normal"
        )

        return {
            "success": True,
            "memory_id": pearl_id,
            "pearl_id": pearl_id,
            "message": f"Memory added to {request.model_id}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Alias the legacy endpoint path
@app.post("/memory/add/v1")
async def add_memory_v1(request: LegacyAddRequest):
    """Alias for legacy add endpoint."""
    return await add_memory_legacy(request)


# =============================================================================
# Server Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get port from environment (Railway sets this)
    port = int(os.getenv("PORT", config.PORT))
    host = os.getenv("HOST", config.HOST)

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True
    )
