"""
GAM-Memvid Server - Librarian Architecture REST API

This server exposes the Librarian memory system to OpenWebUI via REST endpoints.

Architecture:
- MemvidStore: Stores full, raw conversations as "Pearls" using Super-Index
- Synthesizer: Creates detailed abstracts at retrieval time (gpt-4o-mini)
- Per-model vault isolation (each model_id gets its own .mv2 file)

Super-Index Format:
- text field: Compact fingerprint (summary + keywords) for embedding search
- metadata.full_payload: {"user": "...", "ai": "..."} with complete conversation

Endpoints:
- POST /memory/add - Store a new Pearl (user+AI exchange)
- POST /memory/context - Get synthesized context for prompt injection
- POST /memory/search - Search memories
- POST /memvid/search - Search memories (alias for dashboard/valve)
- GET /memvid/vaults - List available vaults
- GET /health - Health check for Railway
- GET /models - List available models
- GET /memory/{model_id}/stats - Get vault statistics
- GET /memory/{model_id}/recent - Get recent memories
- POST /memory/{model_id}/delete - Soft delete a Pearl
"""
import os
import re
import html
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from config import config
from memvid_store import get_store, get_vault_manager, MemvidStore, Pearl
from synthesizer import get_synthesizer, Synthesizer

# =============================================================================
# Logging Setup
# =============================================================================

# Allow LOG_LEVEL to be set via environment (DEBUG, INFO, WARNING, ERROR)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gam-memvid")
logger.setLevel(getattr(logging, log_level, logging.INFO))


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
    created_at: Optional[str] = Field(default=None, description="Original timestamp (ISO format) for imports")


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


class MemvidSearchRequest(BaseModel):
    """Request to search memories (memvid endpoint)."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    model_id: Optional[str] = Field(default=None, description="Model/persona identifier")


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
# Super-Index Hydration Helper
# =============================================================================

def hydrate_pearl(raw_pearl: Any) -> dict:
    """
    Convert a low-level Memvid/store hit into a client-friendly Pearl dict.

    Super-Index Architecture:
    - text field contains a compact fingerprint (for search)
    - metadata.full_payload contains {"user": "...", "ai": "..."} with full content

    This function "hydrates" the pearl by extracting the full conversation
    from full_payload if available, otherwise falls back to text field.

    Args:
        raw_pearl: Could be a Pearl object, SearchResult, dict, or raw Memvid hit

    Returns:
        A flat dictionary safe for JSON serialization
    """
    logger.debug("=" * 60)
    logger.debug("HYDRATE_PEARL INPUT")
    logger.debug(f"  Type: {type(raw_pearl).__name__}")
    logger.debug(f"  Repr (first 500): {repr(raw_pearl)[:500]}")

    # Handle None
    if raw_pearl is None:
        logger.debug("  -> INPUT IS NONE, returning empty dict")
        return {
            "id": None,
            "text": "",
            "user_message": "",
            "ai_response": "",
            "content": "",
            "metadata": {},
            "score": 0.0,
            "tags": [],
            "created_at": None,
            "category": "context",
            "importance": "normal",
            "status": "active"
        }

    # Extract fields - handle both object and dict access
    def safe_get(obj, key, default=None):
        """Safely get attribute or dict key."""
        if obj is None:
            return default
        if hasattr(obj, key):
            val = getattr(obj, key, default)
            return val if val is not None else default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    # Check if this is a SearchResult with nested pearl
    pearl_obj = safe_get(raw_pearl, "pearl", raw_pearl)
    score = safe_get(raw_pearl, "score", 0.0)

    logger.debug(f"  pearl_obj type: {type(pearl_obj).__name__}")
    logger.debug(f"  score: {score}")

    # Get metadata - could be on pearl or raw_pearl
    metadata = safe_get(pearl_obj, "metadata") or safe_get(raw_pearl, "metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    logger.debug(f"  metadata keys: {list(metadata.keys()) if metadata else 'NONE'}")
    logger.debug(f"  metadata.full_payload exists: {'full_payload' in metadata}")

    # Get text field (fingerprint in Super-Index, or content for MemoryEntry)
    text = (
        safe_get(pearl_obj, "text") or
        safe_get(raw_pearl, "text") or
        safe_get(pearl_obj, "content") or
        safe_get(raw_pearl, "content") or
        ""
    )
    logger.debug(f"  text/content field (first 200): {text[:200] if text else 'EMPTY'}")

    # Try to get user_message and ai_response directly first (Pearl objects have these)
    user_message = safe_get(pearl_obj, "user_message") or ""
    ai_response = safe_get(pearl_obj, "ai_response") or ""

    logger.debug(f"  DIRECT user_message (first 100): {user_message[:100] if user_message else 'EMPTY'}")
    logger.debug(f"  DIRECT ai_response (first 100): {ai_response[:100] if ai_response else 'EMPTY'}")

    # If not found directly, try to extract from full_payload in metadata
    full_payload = metadata.get("full_payload")
    logger.debug(f"  full_payload type: {type(full_payload).__name__ if full_payload else 'None'}")

    if isinstance(full_payload, dict):
        logger.debug(f"  full_payload keys: {list(full_payload.keys())}")
        if not user_message:
            user_message = full_payload.get("user", "")
            logger.debug(f"  FROM full_payload.user (first 100): {user_message[:100] if user_message else 'EMPTY'}")
        if not ai_response:
            ai_response = full_payload.get("ai", "")
            logger.debug(f"  FROM full_payload.ai (first 100): {ai_response[:100] if ai_response else 'EMPTY'}")

    # Also check payload_user/payload_ai format
    if not user_message:
        user_message = metadata.get("payload_user", "")
    if not ai_response:
        ai_response = metadata.get("payload_ai", "")

    # Build hydrated text - full conversation if available, else fingerprint
    if user_message or ai_response:
        hydrated_text = f"User: {user_message}\n\nAI: {ai_response}"
    else:
        hydrated_text = text or ""

    # Build combined content
    if user_message and ai_response:
        content = f"User: {user_message}\n\nAI: {ai_response}"
    elif user_message:
        content = f"User: {user_message}"
    elif ai_response:
        content = f"AI: {ai_response}"
    else:
        content = hydrated_text

    # Extract other fields
    pearl_id = (
        safe_get(pearl_obj, "id") or
        safe_get(pearl_obj, "pearl_id") or
        safe_get(raw_pearl, "id") or
        safe_get(raw_pearl, "pearl_id") or
        metadata.get("pearl_id") or
        metadata.get("id")
    )

    tags = (
        safe_get(pearl_obj, "tags") or
        safe_get(raw_pearl, "tags") or
        metadata.get("tags") or
        []
    )
    if not isinstance(tags, list):
        tags = []

    # PRIORITY: Get created_at from metadata first (preserves original timestamp from import)
    # Then fall back to object attributes
    created_at = (
        metadata.get("created_at") or
        safe_get(pearl_obj, "created_at") or
        safe_get(raw_pearl, "created_at")
    )

    category = (
        safe_get(pearl_obj, "category") or
        safe_get(raw_pearl, "category") or
        metadata.get("category") or
        "context"
    )

    importance = (
        safe_get(pearl_obj, "importance") or
        safe_get(raw_pearl, "importance") or
        metadata.get("importance") or
        "normal"
    )

    status = (
        safe_get(pearl_obj, "status") or
        safe_get(raw_pearl, "status") or
        metadata.get("status") or
        "active"
    )

    # If both user_message and ai_response are empty, try to parse from text field
    # The text field often contains "User: {message}\n\nAI: {response}" format
    if not user_message and not ai_response and text:
        logger.debug(f"  Attempting to parse user_message/ai_response from text field...")

        # Try "User: ...\n\nAI: ..." format (double newline separator)
        if "\n\nAI:" in text or "\nAI:" in text:
            # Split on AI: (with newline prefix)
            separator = "\n\nAI:" if "\n\nAI:" in text else "\nAI:"
            parts = text.split(separator, 1)
            if len(parts) == 2:
                # Extract user message (remove "User:" prefix if present)
                user_part = parts[0].strip()
                if user_part.startswith("User:"):
                    user_part = user_part[5:].strip()
                user_message = user_part

                # Extract AI response
                ai_response = parts[1].strip()

                logger.debug(f"  PARSED from text - user_message (first 100): {user_message[:100] if user_message else 'EMPTY'}")
                logger.debug(f"  PARSED from text - ai_response (first 100): {ai_response[:100] if ai_response else 'EMPTY'}")

        # Also try "User: ...\nAI: ..." format (single newline)
        elif "User:" in text and "AI:" in text:
            match = re.match(r'User:\s*(.+?)\s*AI:\s*(.+)', text, re.DOTALL)
            if match:
                user_message = match.group(1).strip()
                ai_response = match.group(2).strip()
                logger.debug(f"  PARSED (regex) from text - user_message (first 100): {user_message[:100] if user_message else 'EMPTY'}")
                logger.debug(f"  PARSED (regex) from text - ai_response (first 100): {ai_response[:100] if ai_response else 'EMPTY'}")

        # Rebuild hydrated_text and content if we successfully parsed
        if user_message or ai_response:
            hydrated_text = f"User: {user_message}\n\nAI: {ai_response}"
            if user_message and ai_response:
                content = f"User: {user_message}\n\nAI: {ai_response}"
            elif user_message:
                content = f"User: {user_message}"
            else:
                content = f"AI: {ai_response}"
            logger.debug(f"  REBUILT hydrated_text and content from parsed values")

    # Log warning if STILL both are empty after parsing attempts
    if not user_message and not ai_response:
        logger.warning(f"HYDRATE WARNING: Both user_message and ai_response are EMPTY for pearl_id={pearl_id}")
        logger.warning(f"  -> text field was: {text[:200] if text else 'EMPTY'}")
        logger.warning(f"  -> full_payload was: {full_payload}")

    # Decode HTML entities in text fields (e.g., &gt; -> >, &quot; -> ", &#x27; -> ')
    # This ensures the dashboard displays text as it appeared in original conversations
    if user_message:
        user_message = html.unescape(user_message)
    if ai_response:
        ai_response = html.unescape(ai_response)
    if hydrated_text:
        hydrated_text = html.unescape(hydrated_text)
    if content:
        content = html.unescape(content)

    result = {
        "id": pearl_id,
        "text": hydrated_text,
        "user_message": user_message,
        "ai_response": ai_response,
        "content": content,
        "metadata": metadata,
        "score": float(score) if score else 0.0,
        "tags": tags,
        "created_at": created_at,
        "category": category,
        "importance": importance,
        "status": status
    }

    logger.debug("HYDRATE_PEARL OUTPUT:")
    logger.debug(f"  id: {result['id']}")
    logger.debug(f"  user_message (first 100): {result['user_message'][:100] if result['user_message'] else 'EMPTY'}")
    logger.debug(f"  ai_response (first 100): {result['ai_response'][:100] if result['ai_response'] else 'EMPTY'}")
    logger.debug(f"  created_at: {result['created_at']}")
    logger.debug("=" * 60)

    return result


# =============================================================================
# Lifespan Handler
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    errors = config.validate()
    if errors:
        logger.warning("=" * 60)
        logger.warning("Configuration errors:")
        for error in errors:
            logger.warning(f"  - {error}")
        logger.warning("Copy .env.example to .env and configure it.")
        logger.warning("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("GAM-Memvid Librarian Server (Smart Server)")
        logger.info("=" * 60)
        logger.info(f"Host: {config.HOST}:{config.PORT}")
        logger.info(f"Vaults directory: {config.VAULTS_DIR}")
        logger.info(f"Embedding model: {config.MEMVID_EMBEDDING_MODEL}")
        logger.info(f"OpenAI configured: {'Yes' if config.OPENAI_API_KEY else 'No'}")
        logger.info("=" * 60)

    yield

    logger.info("[Server] Shutting down...")
    if _vault_manager:
        _vault_manager.close_all()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="GAM-Memvid Librarian Server",
    description="AI memory system with Memvid storage and runtime synthesis",
    version="2.2.0",
    lifespan=lifespan
)

# CORS for OpenWebUI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "status": "healthy",
        "version": "2.2.0",
        "timestamp": datetime.now().isoformat(),
        "vault_files": len(vault_mgr.get_all_vault_files()),
        "active_connections": len(vault_mgr.list_models())
    }


# =============================================================================
# Debug Endpoints
# =============================================================================

@app.post("/debug/echo")
async def debug_echo(request: AddPearlRequest):
    """
    DEBUG: Echo back exactly what the server received.
    Helps diagnose if created_at is being parsed correctly.
    """
    return {
        "received": {
            "model_id": request.model_id,
            "user_message_length": len(request.user_message),
            "ai_response_length": len(request.ai_response),
            "tags": request.tags,
            "category": request.category,
            "importance": request.importance,
            "user_name": request.user_name,
            "created_at": request.created_at,
            "created_at_type": type(request.created_at).__name__
        }
    }


# =============================================================================
# Memvid Endpoints (Dashboard & Valve)
# =============================================================================

@app.get("/memvid/vaults")
async def list_vaults():
    """
    List all available vault files.

    Scans the vaults directory for .mv2 files.
    Used by the dashboard to populate the vault selector.
    """
    try:
        vaults_dir = Path(config.VAULTS_DIR)
        if not vaults_dir.exists():
            return {"vaults": []}

        vault_files = [f.name for f in vaults_dir.glob("*.mv2")]
        return {"vaults": vault_files}

    except Exception as e:
        logger.exception("Error listing vaults")
        raise HTTPException(status_code=500, detail="Failed to list vaults")


@app.post("/memvid/search")
async def memvid_search(request: MemvidSearchRequest):
    """
    Search memories across vault(s).

    This endpoint is used by the Chat Valve and Dashboard.
    Returns hydrated Pearls with full conversation text.
    """
    try:
        vault_mgr = get_vault_mgr()

        # If no model_id specified, search first available vault
        model_id = request.model_id
        if not model_id:
            vault_files = vault_mgr.get_all_vault_files()
            if not vault_files:
                return {"items": [], "count": 0}
            model_id = vault_files[0].replace(".mv2", "")

        store = vault_mgr.get_store(model_id)

        # Search for pearls
        results = store.search_pearls(
            query=request.query,
            limit=request.limit
        )

        # Hydrate all results
        hydrated = [hydrate_pearl(r) for r in results]

        return {
            "items": hydrated,
            "count": len(hydrated),
            "model_id": model_id
        }

    except Exception as e:
        logger.exception("Error in memvid_search for query: %s", request.query)
        raise HTTPException(status_code=500, detail="Search failed")


# =============================================================================
# Core Memory Endpoints
# =============================================================================

@app.post("/memory/add", response_model=AddPearlResponse)
async def add_pearl(request: AddPearlRequest):
    """
    Store a new Pearl (conversation exchange).

    Called by OpenWebUI's outlet() after each turn.
    """
    try:
        # DEBUG: Log ALL request fields to see exactly what Pydantic parsed
        logger.info(f"POST /memory/add for model={request.model_id}")
        logger.info(f"[TIMESTAMP DEBUG] ============================================")
        logger.info(f"[TIMESTAMP DEBUG] request.created_at = {request.created_at!r}")
        logger.info(f"[TIMESTAMP DEBUG] type = {type(request.created_at).__name__}")
        logger.info(f"[TIMESTAMP DEBUG] bool(request.created_at) = {bool(request.created_at)}")
        logger.info(f"[TIMESTAMP DEBUG] Full request dict keys: {request.model_dump().keys()}")
        logger.info(f"[TIMESTAMP DEBUG] ============================================")
        logger.debug(f"  REQUEST DUMP: tags={request.tags!r}")
        logger.debug(f"  REQUEST DUMP: category={request.category!r}")
        logger.debug(f"  user_message length: {len(request.user_message)}")
        logger.debug(f"  user_message preview: {request.user_message[:100] if request.user_message else 'EMPTY'}")

        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(request.model_id)

        # Pass created_at through - MUST preserve the original timestamp!
        created_at_to_pass = request.created_at
        logger.info(f"[TIMESTAMP DEBUG] Passing to store.add_pearl: created_at={created_at_to_pass!r}")

        pearl_id = store.add_pearl(
            user_message=request.user_message,
            ai_response=request.ai_response,
            tags=request.tags or [],
            category=request.category or "context",
            importance=request.importance or "normal",
            user_name=request.user_name or "User",
            created_at=created_at_to_pass  # Use the preserved timestamp!
        )

        word_count = len(request.user_message.split()) + len(request.ai_response.split())

        logger.info(f"  -> Stored Pearl {pearl_id}, {word_count} words")

        return AddPearlResponse(
            pearl_id=pearl_id,
            status="ok",
            word_count=word_count
        )

    except Exception as e:
        logger.exception("Error adding Pearl to model %s", request.model_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    """
    Get synthesized context for prompt injection.

    Called by OpenWebUI's inlet() before each turn.
    """
    try:
        vault_mgr = get_vault_mgr()
        synthesizer = get_synth()

        store = vault_mgr.get_store(request.model_id)

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
        logger.exception("Error getting context for model %s", request.model_id)
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

        # Hydrate all results for dashboard compatibility
        hydrated = [hydrate_pearl(r) for r in results]

        return SearchResponse(
            results=hydrated,
            count=len(hydrated)
        )

    except Exception as e:
        logger.exception("Error searching model %s", request.model_id)
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
        logger.exception("Error deleting Pearl %s from model %s", request.pearl_id, model_id)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Model/Vault Management
# =============================================================================

@app.get("/models")
async def list_models():
    """List all models with vaults."""
    try:
        vault_mgr = get_vault_mgr()
        vault_files = vault_mgr.get_all_vault_files()
        model_ids = [f.replace(".mv2", "") for f in vault_files]
        return {
            "models": model_ids,
            "active_models": vault_mgr.list_models(),
            "all_vault_files": vault_files
        }
    except Exception as e:
        logger.exception("Error listing models")
        return {"models": [], "active_models": [], "all_vault_files": []}


@app.get("/memory/{model_id}/stats")
async def get_model_stats(model_id: str):
    """Get statistics for a model's vault."""
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)
        return store.get_stats()
    except Exception as e:
        logger.exception("Error getting stats for model %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to get vault statistics")


@app.get("/memory/{model_id}/export")
async def export_model(model_id: str):
    """Export all Pearls from a model's vault as JSON."""
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)
        return store.export()
    except Exception as e:
        logger.exception("Error exporting model %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to export vault")


@app.get("/memory/{model_id}/export/json")
async def export_model_json_download(model_id: str):
    """
    Export all Pearls as a downloadable JSON file.

    Returns a JSON file download with all memories for this model.
    """
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)
        export_data = store.export()

        # Return as downloadable JSON
        import json
        json_content = json.dumps(export_data, indent=2, default=str)

        return JSONResponse(
            content=export_data,
            headers={
                "Content-Disposition": f"attachment; filename={model_id}_memories.json"
            }
        )
    except Exception as e:
        logger.exception("Error exporting model %s as JSON", model_id)
        raise HTTPException(status_code=500, detail="Failed to export vault")


@app.get("/memory/{model_id}/export/mv2")
async def export_model_mv2_download(model_id: str):
    """
    Download the raw .mv2 vault file.

    Returns the actual .mv2 file for backup or transfer purposes.
    """
    try:
        vaults_dir = Path(config.VAULTS_DIR)
        vault_file = vaults_dir / f"{model_id}.mv2"

        if not vault_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Vault file not found: {model_id}.mv2"
            )

        return FileResponse(
            path=str(vault_file),
            filename=f"{model_id}.mv2",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error downloading vault file for model %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to download vault file")


@app.delete("/memory/{model_id}/vault")
async def delete_vault(model_id: str, confirm: bool = False):
    """
    Delete a model's vault file (.mv2).

    WARNING: This permanently deletes all memories for this model!

    Args:
        model_id: The model ID whose vault to delete
        confirm: Must be True to actually delete (safety check)

    Returns:
        Success/failure message
    """
    if not confirm:
        return {
            "success": False,
            "message": "Safety check: Add ?confirm=true to actually delete the vault",
            "model_id": model_id,
            "warning": "This will PERMANENTLY delete all memories for this model!"
        }

    try:
        vault_mgr = get_vault_mgr()
        vaults_dir = Path(config.VAULTS_DIR)

        # Close the store if it's open
        if model_id in vault_mgr._stores:
            vault_mgr._stores[model_id].close()
            del vault_mgr._stores[model_id]

        # Find and delete the vault file
        vault_file = vaults_dir / f"{model_id}.mv2"

        if not vault_file.exists():
            return {
                "success": False,
                "message": f"Vault file not found: {vault_file.name}",
                "model_id": model_id
            }

        # Delete the file
        vault_file.unlink()
        logger.info(f"Deleted vault file: {vault_file}")

        return {
            "success": True,
            "message": f"Vault deleted successfully",
            "model_id": model_id,
            "deleted_file": vault_file.name
        }

    except Exception as e:
        logger.exception("Error deleting vault for model %s", model_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete vault: {str(e)}")


@app.get("/memory/{model_id}/debug-raw")
async def debug_raw_vault(model_id: str, limit: int = 3):
    """
    DEBUG ENDPOINT: Show raw Memvid SDK output for diagnosing storage issues.

    This bypasses hydration to show exactly what the SDK returns.
    """
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)

        # Get raw hits from SDK
        raw_results = []

        # Try to get raw data from the internal _mv object
        if hasattr(store, '_mv') and store._mv:
            mv = store._mv

            # Try find() with empty query
            try:
                result = mv.find("", k=limit, mode="lex")
                logger.info(f"Raw find() result type: {type(result)}")
                logger.info(f"Raw find() result: {str(result)[:1000]}")

                if isinstance(result, dict):
                    hits = result.get("hits", [])
                    for i, hit in enumerate(hits[:limit]):
                        raw_results.append({
                            "index": i,
                            "type": type(hit).__name__,
                            "keys": list(hit.keys()) if isinstance(hit, dict) else "N/A",
                            "text_preview": str(hit.get("text", ""))[:200] if isinstance(hit, dict) else "N/A",
                            "metadata_type": type(hit.get("metadata")).__name__ if isinstance(hit, dict) else "N/A",
                            "metadata_keys": list(hit.get("metadata", {}).keys()) if isinstance(hit, dict) and isinstance(hit.get("metadata"), dict) else "N/A",
                            "metadata_preview": str(hit.get("metadata", ""))[:500] if isinstance(hit, dict) else "N/A",
                            "full_hit": str(hit)[:1000]
                        })
            except Exception as e:
                raw_results.append({"error": f"find() failed: {str(e)}"})

        return {
            "model_id": model_id,
            "store_type": type(store).__name__,
            "has_mv": hasattr(store, '_mv') and store._mv is not None,
            "raw_hits": raw_results
        }

    except Exception as e:
        logger.exception("Error in debug-raw for model %s", model_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{model_id}/recent")
async def get_recent_memories(
    model_id: str,
    limit: int = Query(default=20, ge=1, le=100)
):
    """
    Get most recent memories for a model.

    Returns hydrated Pearls with full conversation text.
    """
    try:
        logger.info(f"GET /memory/{model_id}/recent called with limit={limit}")

        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(model_id)

        # Get recent pearls using the new method that returns Pearls directly
        # This ensures we get ALL Pearls by searching across categories
        try:
            memories = store.get_recent_pearls(limit=limit)
            logger.info(f"store.get_recent_pearls returned {len(memories)} items")
        except AttributeError:
            # Fallback: use category-based search to get ALL Pearls (like export does)
            logger.warning("get_recent_pearls not available, using category-based search")
            memories = []
            from memory_entry import MemoryCategory
            deleted_ids = store.get_deleted_pearl_ids()
            seen_ids = set()
            
            for cat in MemoryCategory:
                try:
                    results = store.search_pearls(query=f"category:{cat.value}", limit=limit * 10, mode="lex")
                    for r in results:
                        if r.pearl.id not in seen_ids and r.pearl.id not in deleted_ids:
                            memories.append(r.pearl)
                            seen_ids.add(r.pearl.id)
                except Exception:
                    pass
            
            # Sort by created_at and limit
            memories.sort(key=lambda p: p.created_at or "", reverse=True)
            memories = memories[:limit]
            logger.info(f"category-based search returned {len(memories)} items")

        # DEBUG: Log RAW pearls before hydration
        logger.debug("=" * 80)
        logger.debug("RAW PEARLS FROM STORE (before hydration):")
        for i, m in enumerate(memories[:3]):  # Log first 3
            logger.debug(f"  RAW[{i}] type: {type(m).__name__}")
            if hasattr(m, '__dict__'):
                logger.debug(f"  RAW[{i}] __dict__ keys: {list(m.__dict__.keys())}")
            if hasattr(m, 'user_message'):
                logger.debug(f"  RAW[{i}] .user_message (first 100): {m.user_message[:100] if m.user_message else 'EMPTY'}")
            if hasattr(m, 'ai_response'):
                logger.debug(f"  RAW[{i}] .ai_response (first 100): {m.ai_response[:100] if m.ai_response else 'EMPTY'}")
            if hasattr(m, 'metadata'):
                logger.debug(f"  RAW[{i}] .metadata keys: {list(m.metadata.keys()) if isinstance(m.metadata, dict) else 'NOT A DICT'}")
            if isinstance(m, dict):
                logger.debug(f"  RAW[{i}] dict keys: {list(m.keys())}")
        logger.debug("=" * 80)

        # Hydrate all results
        hydrated = [hydrate_pearl(m) for m in memories]

        # DEBUG: Log HYDRATED results
        logger.debug("=" * 80)
        logger.debug("HYDRATED PEARLS (after hydration):")
        for i, h in enumerate(hydrated[:3]):  # Log first 3
            logger.debug(f"  HYDRATED[{i}] id: {h.get('id')}")
            logger.debug(f"  HYDRATED[{i}] user_message (first 100): {h.get('user_message', '')[:100] if h.get('user_message') else 'EMPTY'}")
            logger.debug(f"  HYDRATED[{i}] ai_response (first 100): {h.get('ai_response', '')[:100] if h.get('ai_response') else 'EMPTY'}")
            logger.debug(f"  HYDRATED[{i}] created_at: {h.get('created_at')}")
        logger.debug("=" * 80)

        # Sort by created_at (most recent first)
        hydrated.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        return {
            "model_id": model_id,
            "count": len(hydrated),
            "memories": hydrated,
            "items": hydrated  # Alias for compatibility
        }

    except Exception as e:
        logger.exception("Error getting recent memories for model %s", model_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load recent memories: {str(e)}"
        )


# =============================================================================
# Legacy Compatibility Endpoints
# =============================================================================

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
    """Legacy endpoint for backward compatibility."""
    try:
        vault_mgr = get_vault_mgr()
        store = vault_mgr.get_store(request.model_id)

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
        logger.exception("Error in legacy add for model %s", request.model_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/add/v1")
async def add_memory_v1(request: LegacyAddRequest):
    """Alias for legacy add endpoint."""
    return await add_memory_legacy(request)


# =============================================================================
# Server Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", config.PORT))
    host = os.getenv("HOST", config.HOST)

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True
    )
