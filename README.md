# GAM-Memvid Fusion

**AI Memory System with Memvid Storage + OpenWebUI Integration**

A persistent memory system for AI models using the "Librarian" architecture:
- **Store full, raw conversations** as "Pearls" (no truncation)
- **Synthesize at retrieval time** using gpt-4o-mini
- **Deploy to Railway** and connect to OpenWebUI via HTTP

## Architecture

```
OpenWebUI ←→ server.py ←→ MemvidStore + Synthesizer
                              ↓
                          .mv2 vault files
```

- **MemvidStore**: Stores complete conversation exchanges using Memvid SDK v2
- **Synthesizer**: Creates detailed abstracts with verbatim quotes at retrieval time
- **Super-Index**: Compact fingerprints for search, full payloads in metadata

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for detailed architecture documentation.

## Quickstart (Local)

### 1. Clone and Setup

```bash
git clone https://github.com/jpetree331/GAM-Memvid-fusion.git
cd GAM-Memvid-fusion

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Test the Pipeline

```bash
python debug_pipeline.py
```

This tests the full Write → Read → Synthesize pipeline.

### 4. Run the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8100
```

The server will be available at `http://localhost:8100`

### 5. Test Endpoints

```bash
# Health check
curl http://localhost:8100/health

# Add a memory
curl -X POST http://localhost:8100/memory/add \
  -H "Content-Type: application/json" \
  -d '{"model_id":"test","user_message":"Hello","ai_response":"Hi there!"}'

# Get context
curl -X POST http://localhost:8100/memory/context \
  -H "Content-Type: application/json" \
  -d '{"model_id":"test","query":"greeting"}'
```

## Deploy to Railway

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial GAM+Memvid Librarian server"
git remote add origin https://github.com/jpetree331/GAM-Memvid-fusion.git
git push -u origin main
```

### 2. Create Railway Service

1. Go to [Railway](https://railway.app)
2. Create new project → Deploy from GitHub repo
3. Select `GAM-Memvid-fusion`

### 3. Configure Railway

**Start Command:**
```bash
uvicorn server:app --host 0.0.0.0 --port $PORT
```

**Environment Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `MEMVID_EMBEDDING_MODEL` - `BAAI/bge-small-en-v1.5` (default)
- `DATA_DIR` - `./data`
- `VAULTS_DIR` - `./data/vaults`

Railway automatically sets `PORT`.

### 4. Get Your Railway URL

After deployment, Railway provides a URL like:
```
https://your-app-name.up.railway.app
```

## OpenWebUI Integration

### Option 1: Filter (Automatic Memory)

1. In OpenWebUI: **Admin → Functions → Create Function**
2. Copy contents of `openwebui_filter.py`
3. Set `MEMORY_SERVER_URL` valve to your Railway URL
4. Enable the filter on your models

The filter automatically:
- **inlet()**: Retrieves synthesized context before AI responds
- **outlet()**: Stores the user+AI exchange after each turn

### Option 2: Functions/Tools (Manual Memory)

1. In OpenWebUI: **Admin → Functions → Create Function**
2. Copy contents of `openwebui_function.py`
3. Set `MEMORY_SERVER_URL` valve to your Railway URL

Provides tools the AI can call:
- `search_memories(query)` - Search past conversations
- `remember_this(user_message, ai_response)` - Store explicitly
- `forget_memory(pearl_id)` - Delete a memory
- `memory_stats()` - View vault statistics

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memory/add` | POST | Store a Pearl (user_message + ai_response) |
| `/memory/context` | POST | Get synthesized context for prompt injection |
| `/memory/search` | POST | Search for relevant Pearls |
| `/memory/{model_id}/delete` | POST | Soft-delete a Pearl |
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/memory/{model_id}/stats` | GET | Vault statistics |
| `/memory/{model_id}/export` | GET | Export all Pearls |

## Dashboard (Optional)

Run the Streamlit dashboard for visual vault management:

```bash
streamlit run dashboard_inspector.py
```

## File Structure

```
├── server.py              # FastAPI server (main entrypoint)
├── memvid_store.py        # MemvidStore - Pearl storage layer
├── synthesizer.py         # Runtime synthesis (gpt-4o-mini)
├── config.py              # Environment configuration
├── openwebui_filter.py    # OpenWebUI Filter (inlet/outlet)
├── openwebui_function.py  # OpenWebUI Functions/Tools
├── debug_pipeline.py      # Test the full pipeline
├── dashboard_inspector.py # Streamlit dashboard
├── memory_entry.py        # MemoryEntry dataclass
├── memory_condenser.py    # LLM-powered tag extraction (optional)
├── memory_filter.py       # Intelligent storage filtering (optional)
├── import_conversations.py # JSON conversation import
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── .gitignore             # Git ignore patterns
├── PROJECT_SUMMARY.md     # Detailed architecture docs
└── README.md              # This file
```

## Sanity Checks

Before deploying, verify:

```bash
# Test the pipeline
python debug_pipeline.py

# Check server starts without errors
uvicorn server:app --host 0.0.0.0 --port 8100
# (Ctrl+C to stop)

# Syntax check all files
python -m py_compile server.py memvid_store.py synthesizer.py config.py
```

## License

MIT

## Links

- **GitHub**: https://github.com/jpetree331/GAM-Memvid-fusion
- **Memvid SDK**: https://github.com/memvid-ai/memvid-sdk
- **OpenWebUI**: https://github.com/open-webui/open-webui
