# Archive: Pre-Memvid v2 Code

This folder contains implementations that have been replaced by the Memvid v2 integration.

## Why Archived

1. **Buffer Strategy (memvid_store_buffer.py)**: Memvid v1 required full re-encoding to update files.
   We designed a buffer + nightly merge strategy. Memvid v2 (Jan 28, 2026) introduced "Living Memory"
   with append-only writes, making the buffer unnecessary.

2. **GAM Integration (memory_manager_gam.py)**: The original implementation used the GAM library
   (General Agentic Memory) with FAISS, BM25, and dense retrievers. Memvid v2 handles all of this
   internally with hybrid search.

3. **Memory Organization (memory_organization_legacy.py)**: The JSON-based persistence and
   organization layer. Now integrated into MemvidStore which handles categories, importance,
   tags, and AI self-reflection natively.

## Archived Files

| File | Original Purpose |
|------|-----------------|
| `memvid_store_buffer.py` | Buffer + Merge strategy for Memvid v1 |
| `memory_manager_gam.py` | GAM library wrapper (FAISS, BM25, dense retrievers) |
| `memory_organization_legacy.py` | JSON-based memory persistence and organization |

## Current Architecture (v2)

```
Writes: add_memory() → vault.mv2 (direct, real-time via Memvid v2)
Reads:  search()     → vault.mv2 (sub-5ms hybrid search)
Brain:  MemoryCondenser (Gemini Flash) → extracts memories → vault

Storage: data/vaults/{model_id}.mv2
```

## Migration Path

If you have existing data in the old formats:
- `data/models/{model_id}/memory_index.json` → Use migration script
- OpenWebUI chat exports → Use migration script

See `migrate_json_to_memvid.py` (Phase 3) for migration tools.

---
Archived: 2026-01-30
