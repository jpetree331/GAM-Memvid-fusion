# GAM-Memvid Project Summary

## Overview

GAM-Memvid is an AI memory system designed to integrate with OpenWebUI. It provides persistent, semantic memory for AI models using a **"Librarian" architecture** that stores full conversations and synthesizes them at retrieval time.

**Migration Status:** Migrated from GAM library to Memvid SDK v2 with "Super-Index" architecture.

---

## Current Architecture: "The Librarian"

### Core Philosophy
- **Store FULL, RAW conversations** (called "Pearls") - no summarization at write time
- **Synthesize at retrieval time** - create detailed abstracts when memories are needed
- **Super-Index strategy** - compact fingerprints for search, full payloads in metadata

### Why This Approach?
1. **No information loss** - Raw conversations preserved forever
2. **Synthesis can improve** - Better models = better abstracts without re-indexing
3. **Flexible retrieval** - Different synthesis styles for different contexts
4. **Debuggable** - Can always inspect exactly what was stored

---

## File Structure

### Core Layer (NEW - Implemented & Working)

| File | Purpose | Status |
|------|---------|--------|
| `memvid_store.py` | Main storage layer using Memvid SDK v2 with Super-Index architecture | **WORKING** |
| `synthesizer.py` | Runtime synthesis - creates detailed abstracts from raw Pearls using gpt-4o-mini | **WORKING** |
| `config.py` | Environment configuration, paths, API keys | **WORKING** |
| `debug_pipeline.py` | CLI tool to test Write → Read → Synthesize pipeline | **WORKING** |
| `dashboard_inspector.py` | Streamlit dashboard for vault inspection and testing | **WORKING** |

### OpenWebUI Integration Layer (UPDATED - Connected to Librarian)

| File | Purpose | Status |
|------|---------|--------|
| `server.py` | FastAPI server with REST endpoints | **UPDATED - uses MemvidStore + Synthesizer** |
| `openwebui_filter.py` | OpenWebUI Filter (inlet/outlet pattern) | **UPDATED - calls new endpoints** |
| `openwebui_function.py` | OpenWebUI Tools/Pipe | **UPDATED - provides memory tools** |
| `memory_filter.py` | Intelligent storage filtering logic | **AVAILABLE (optional)** |
| `memory_condenser.py` | LLM-powered memory extraction (tag generation) | **AVAILABLE (optional)** |

### Legacy Files (No Longer Used)

| File | Purpose | Status |
|------|---------|--------|
| `memory_manager.py` | OLD memory management layer | **LEGACY - replaced by memvid_store.py** |
| `memory_organization.py` | OLD JSON persistence layer | **LEGACY - replaced by Memvid SDK** |

### Support Files

| File | Purpose |
|------|---------|
| `.env.example` | Environment template with API keys |
| `.gitignore` | Git ignore patterns |
| `requirements.txt` | Python dependencies |
| `import_conversations.py` | JSON conversation import utilities |

---

## Key Technical Concepts

### 1. Super-Index Architecture

The Memvid SDK truncates the `text` field, so we use a two-part strategy:

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPER-INDEX STRATEGY                      │
├─────────────────────────────────────────────────────────────┤
│  FINGERPRINT (for embedding/search)                         │
│  ├─ Short text (<1000 chars): Use directly                  │
│  └─ Long text: LLM summary + keywords (~1500 chars max)     │
├─────────────────────────────────────────────────────────────┤
│  FULL_PAYLOAD (in metadata - never truncated)               │
│  ├─ user_message: Complete user message                     │
│  └─ ai_response: Complete AI response                       │
└─────────────────────────────────────────────────────────────┘
```

### 2. Pearl Storage Format

```python
Pearl = {
    "id": "pearl_uuid",
    "user_message": "Full user message...",
    "ai_response": "Full AI response...",
    "tags": ["theology", "philosophy"],
    "created_at": "2024-01-15T10:30:00Z",
    "status": "active"  # or "deleted" for soft delete
}
```

### 3. Synthesis Pipeline

```
Search Query → Memvid SDK find() → Raw Pearls → Synthesizer → Detailed Abstracts → Context Injection
```

The Synthesizer (`synthesizer.py`) uses gpt-4o-mini to create 200-300 word abstracts with:
- Core arguments and reasoning
- Key definitions and terminology
- **Verbatim quotes** (exact substrings from source)
- Emotional tone

### 4. Local Embeddings

Uses `fastembed` with `BAAI/bge-small-en-v1.5` model:
- Runs locally (no API calls for embeddings)
- ~50MB model download on first run
- Fast and high quality

---

## Integration Status

### What's DONE:
- [x] Memvid SDK v2 integration with Super-Index
- [x] Full payload storage (no truncation)
- [x] Runtime synthesis with gpt-4o-mini
- [x] Soft delete pattern (append-only)
- [x] Debug CLI tool (`debug_pipeline.py`)
- [x] Streamlit dashboard (`dashboard_inspector.py`)
- [x] Local embeddings with fastembed
- [x] MV005 error handling for small vaults
- [x] Structured synthesis output with verbatim quotes
- [x] **server.py connected to MemvidStore + Synthesizer**
- [x] **OpenWebUI filter updated for Librarian architecture**
- [x] **OpenWebUI functions/tools for manual memory operations**

### What's NOT YET DONE:
- [ ] Test end-to-end with OpenWebUI deployment
- [ ] JSON conversation import via new architecture
- [ ] Smart tagging in filter (optional enhancement)

---

## API Configuration

Required environment variables (see `.env.example`):

```bash
# Required for Synthesizer (runtime synthesis)
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional

# Optional for Memory Condenser (tag extraction at storage time)
GEMINI_API_KEY=AIza...
CONDENSER_PROVIDER=gemini  # or "openai"
CONDENSER_MODEL=gemini-2.0-flash

# Embedding model (local, no API key needed)
MEMVID_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Server config
HOST=0.0.0.0
PORT=8100
DATA_DIR=./data
VAULTS_DIR=./data/vaults
```

---

## REST API Endpoints

The server exposes these endpoints for OpenWebUI integration:

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memory/add` | POST | Store a Pearl (user_message + ai_response) |
| `/memory/context` | POST | Get synthesized context for prompt injection |
| `/memory/search` | POST | Search for relevant Pearls |
| `/memory/{model_id}/delete` | POST | Soft-delete a Pearl |
| `/health` | GET | Health check for Railway |

### Request/Response Examples

**POST /memory/add**
```json
{
  "model_id": "eli",
  "user_message": "What do you think about consciousness?",
  "ai_response": "Consciousness is fascinating...",
  "tags": ["philosophy", "ai_theory"],
  "user_name": "Jess"
}
```
Response: `{"pearl_id": "pearl_20240115_...", "status": "ok", "word_count": 150}`

**POST /memory/context**
```json
{
  "model_id": "eli",
  "query": "consciousness discussion",
  "user_name": "Jess",
  "limit": 5,
  "max_words": 400
}
```
Response: `{"context": "..synthesized text..", "num_pearls": 3, "has_memories": true}`

---

## Data Flow (Current Working Pipeline)

```
1. WRITE: Pearl → build_fingerprint() → Memvid SDK add()
   └─ Stores: fingerprint as text, full_payload in metadata

2. READ: Query → Memvid SDK find() → _extract_hits_from_result()
   └─ Retrieves: full_payload from metadata → reconstructs Pearl

3. SYNTHESIZE: Raw Pearls → Synthesizer → Detailed Abstracts
   └─ Creates: 200-300 word summaries with verbatim quotes
```

---

## Next Steps for OpenWebUI Integration

To connect the Librarian architecture to OpenWebUI:

1. **Update server.py** to use `memvid_store.py` instead of `memory_manager`
2. **Update endpoints** to work with Pearls (user_message + ai_response pairs)
3. **Add synthesis endpoint** for context retrieval
4. **Test with OpenWebUI filter**

The filter code in `openwebui_filter.py` should largely work once the server endpoints are updated - it already has:
- `inlet()` for memory retrieval
- `outlet()` for memory storage
- Smart filtering logic
- AI self-reflection detection

---

## Testing

### Debug Pipeline
```bash
python debug_pipeline.py
```
Tests the full Write → Read → Synthesize pipeline with a 1000+ word test essay.

### Streamlit Dashboard
```bash
streamlit run dashboard_inspector.py
```
Visual vault manager with:
- Vault selection
- Pearl listing and search
- Synthesis testing
- Soft delete functionality

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         OpenWebUI                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 openwebui_filter.py                      │    │
│  │  inlet() ──────────────────────────────── outlet()       │    │
│  │     │                                         │          │    │
│  │     ▼                                         ▼          │    │
│  │  Retrieve context                      Store exchange    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        server.py                                 │
│  /memory/context ◄────────────────────────► /memory/add         │
│         │                                          │             │
│         ▼                                          ▼             │
│  ┌─────────────┐                          ┌─────────────┐       │
│  │ synthesizer │                          │ memvid_store│       │
│  │   .py       │◄─────────────────────────│   .py       │       │
│  │ (gpt-4o-mini)│     (retrieval)          │ (Super-Index)│      │
│  └─────────────┘                          └─────────────┘       │
│                                                   │              │
│                                                   ▼              │
│                                           ┌─────────────┐       │
│                                           │ Memvid SDK  │       │
│                                           │   v2        │       │
│                                           │ (.mv2 files)│       │
│                                           └─────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

**Status:** All components are now connected. The server uses `memvid_store.py` (MemvidStore) for storage and `synthesizer.py` (Synthesizer) for context generation.

---

## Key Classes and Functions

### memvid_store.py

```python
class Pearl:
    """A single conversation exchange - the atomic unit of memory."""
    id: str
    user_message: str
    ai_response: str
    tags: List[str]
    created_at: str
    status: str  # "active" or "deleted"

class MemvidStore:
    """The Librarian's vault - stores and retrieves Pearls."""

    def add_pearl(user_message, ai_response, tags) -> str:
        """Store a conversation exchange, returns pearl_id"""

    def search(query, limit=10) -> List[Pearl]:
        """Semantic search for relevant Pearls"""

    def get_raw_pearls_for_synthesis(query, limit=5) -> List[Pearl]:
        """Get Pearls ready for the Synthesizer"""

    def soft_delete(pearl_id) -> bool:
        """Mark a Pearl as deleted (append-only)"""

def build_fingerprint(user_message, ai_response) -> str:
    """Create compact text for embedding (handles long conversations)"""
```

### synthesizer.py

```python
class Synthesizer:
    """Runtime synthesizer - converts raw Pearls to detailed abstracts."""

    async def synthesize_pearl(pearl_id, user_message, ai_response, ...) -> SynthesisResult:
        """Create a detailed abstract from a single Pearl"""

    async def synthesize_for_context(pearls, user_name, max_words) -> str:
        """Full pipeline: Pearls → Abstracts → Context string"""

async def get_synthesized_context(model_id, query, user_name, ...) -> str:
    """Convenience function: Search → Synthesize → Return context"""
```

---

## Version History

- **v1.0** - Initial GAM library integration
- **v2.0** - Migration to Memvid SDK v2
- **v2.1** - Super-Index architecture (current)
  - Fixed SDK truncation issues with full_payload in metadata
  - Added runtime synthesis with structured output
  - Local embeddings with fastembed
  - MV005 error handling
  - Safety cap on fingerprints (1500 chars)

---

## Known Issues / Edge Cases

1. **MV005 Error**: Occurs when vault is too small for timeline queries. Handled gracefully by falling back to lexical search.

2. **Long Conversations**: Conversations over 1000 chars get LLM-summarized fingerprints. Full text is always preserved in metadata.

3. **Soft Delete**: Deleted Pearls remain in the vault with `status: "deleted"`. They're filtered out during search.

---

*Last updated: January 2025*
