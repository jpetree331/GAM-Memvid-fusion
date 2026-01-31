# GAM Memory System - Setup Guide

A persistent memory system for AI models in OpenWebUI, featuring per-model memory buckets, intelligent memory categorization, and a management dashboard.

---

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Part 1: Deploy GAM Server to Railway](#part-1-deploy-gam-server-to-railway)
4. [Part 2: Configure OpenWebUI Filter](#part-2-configure-openwebui-filter)
5. [Part 3: Set Up the Dashboard](#part-3-set-up-the-dashboard)
6. [Part 4: AI Dialogue (Optional)](#part-4-ai-dialogue-optional)
7. [Usage](#usage)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Does
- **Persistent Memory**: Your AI models remember conversations across sessions
- **Per-Model Buckets**: Each custom model has its own memory storage
- **Smart Categorization**: Memories are organized by type (facts, preferences, events, etc.)
- **AI Self-Reflection**: Models can store their own growth and opinions
- **Dashboard**: Web UI to view, edit, and manage memories
- **AI Dialogue** (Optional): Three-way conversations between you and two AI models

### Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  OpenWebUI  │────▶│ GAM Server  │────▶│  Memories   │
│  (Filter)   │◀────│  (Railway)  │◀────│  (Storage)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Dashboard  │
                    │  (Local)    │
                    └─────────────┘
```

---

## Prerequisites

Before you start, you'll need:

1. **OpenWebUI** running (locally or hosted)
2. **Railway Account** (free tier works) - [railway.app](https://railway.app)
3. **GitHub Account** - to deploy from repo
4. **OpenAI API Key** - for memory processing (or compatible alternative)
5. **Python 3.10+** - for running the dashboard locally

---

## Part 1: Deploy GAM Server to Railway

### Step 1: Push Code to GitHub

1. Create a new GitHub repository
2. Push these files to it:
   ```
   server.py
   memory_manager.py
   memory_organization.py
   memory_style.py
   config.py
   requirements.txt
   Procfile
   ```

### Step 2: Create Railway Project

1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Connect your GitHub and select your repository
5. Railway will start building automatically

### Step 3: Add a Volume (Important!)

Without a volume, your memories will be lost on every redeploy!

1. In your Railway project, click **"+ New"** → **"Volume"**
2. Set the mount path to: `/app/data`
3. Click **"Create Volume"**
4. Attach it to your service

### Step 4: Configure Environment Variables

In Railway, go to your service → **Variables** tab → Add these:

| Variable | Value | Description |
|----------|-------|-------------|
| `OPENAI_API_KEY` | `sk-proj-...` | Your OpenAI API key |
| `GAM_MODEL_NAME` | `gpt-4o-mini` | Model for memory processing |
| `GAM_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `DATA_DIR` | `/app/data` | Where memories are stored |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8080` | Server port |

### Step 5: Get Your Server URL

After deployment completes:
1. Go to **Settings** → **Networking**
2. Generate a **Public Domain** or note the **Private URL** (if OpenWebUI is also on Railway)
3. Your URL will look like: `https://your-app-name.up.railway.app`

### Step 6: Verify Deployment

Open in browser: `https://your-app-name.up.railway.app/health`

You should see:
```json
{"status": "healthy", "timestamp": "...", "models_active": 0}
```

---

## Part 2: Configure OpenWebUI Filter

### Step 1: Add the Filter

1. In OpenWebUI, go to **Admin Settings** → **Functions**
2. Click **"+ New Function"**
3. Copy the entire contents of `openwebui_filter.py`
4. Paste it in and **Save**

### Step 2: Configure the Filter

1. Click the **gear icon** on the GAM filter to open **Valves**
2. Set these values:

| Setting | Value |
|---------|-------|
| `GAM_SERVER_URL` | Your Railway URL (e.g., `https://your-app.up.railway.app`) |
| `ENABLE_SMART_FILTER` | `true` (recommended) |
| `DEBUG_MODE` | `false` (set to `true` for troubleshooting) |

3. **Save** the settings

### Step 3: Enable for Models

1. Go to your custom model settings in OpenWebUI
2. Find the **Filters** section
3. Check the box for **GAM** filter
4. Save

Now when you chat with that model, memories will be stored and retrieved automatically!

---

## Part 3: Set Up the Dashboard

The dashboard lets you view, edit, and manage all your memories.

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Configure Dashboard

Create a `.env` file in the project folder:

```env
# Your Railway GAM server URL
GAM_SERVER_URL=https://your-app-name.up.railway.app
```

Or edit `dashboard.py` and update the `DEFAULT_GAM_URL` variable.

### Step 3: Run Dashboard

```bash
streamlit run dashboard.py
```

Open: `http://localhost:8501`

### Dashboard Features
- View all memories by model
- Filter by category (facts, preferences, events, etc.)
- Edit memory content
- Delete memories
- Export memories as JSON
- Manage model buckets (rename, merge, delete)

---

## Part 4: AI Dialogue (Optional)

A three-way conversation system where you chat with two AI models simultaneously.

### Step 1: Add the Pipe

1. In OpenWebUI, go to **Admin Settings** → **Functions**
2. Click **"+ New Function"**
3. Copy contents of `AIDialogue/ai_dialogue_pipe.py`
4. Paste and **Save**

### Step 2: Configure the Pipe

1. Click **gear icon** → **Valves**
2. Set these values:

| Setting | Value |
|---------|-------|
| `ai1_model_id` | Your first AI's model ID |
| `ai1_display_name` | Display name (e.g., "Wizard") |
| `ai2_model_id` | Your second AI's model ID |
| `ai2_display_name` | Display name (e.g., "Claude") |
| `gam_server_url` | Your Railway GAM URL |
| `openwebui_api_url` | Your OpenWebUI URL |
| `openwebui_api_key` | Your OpenWebUI API key |

### Step 3: Use AI Dialogue

1. Start a new chat
2. Select **"AI Dialogue (3-Way Chat)"** as the model
3. Type `/setup` to see available models and configure
4. Type `/config` to verify settings
5. Send a message - both AIs will respond!

### Commands
- `/setup` - Interactive model selection
- `/set ai1 <number>` - Set AI1 by number
- `/set ai2 <number>` - Set AI2 by number
- `/config` - Show current configuration
- `/help` - Show all commands

---

## Usage

### Basic Usage

Just chat normally! The filter automatically:
1. Retrieves relevant memories before the AI responds
2. Stores important information from the conversation
3. Categorizes memories appropriately

### Memory Categories

| Category | What's Stored |
|----------|---------------|
| `fact` | Factual information about you |
| `preference` | Your likes, dislikes, preferences |
| `event` | Things that happened, experiences |
| `relationship` | People in your life |
| `context` | General conversation context |
| `ai_self` | AI's own reflections and growth |

### AI Self-Reflection

Models can store their own thoughts using the REFLECT tag in their responses:
```
[REFLECT: I notice I'm developing a preference for philosophical discussions...]
```

These become AI_Self memories that persist across conversations.

---

## Troubleshooting

### "Storage failed" Error
- Check that `GAM_SERVER_URL` is correct in valve settings
- Verify the GAM server is running: visit `/health` endpoint
- Check Railway logs for errors

### Memories Not Showing in Dashboard
- Ensure dashboard is pointing to the correct GAM URL
- Check the model ID dropdown - memories are stored per-model
- Try refreshing or checking a different page

### AI Not Recalling Memories
- Verify the filter is enabled for your model
- Check that the model ID matches between filter and dashboard
- Look at Railway logs for retrieval errors

### Railway Deployment Issues
- Make sure the Volume is attached for persistent storage
- Check that all environment variables are set
- Look at deploy logs for errors

### HTTP 402 Error
- This means "Payment Required" from the AI provider
- Check your API credits (OpenRouter, OpenAI, etc.)
- Try a different/cheaper model

---

## File Reference

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server for GAM |
| `memory_manager.py` | Per-model memory management |
| `memory_organization.py` | Memory categorization and indexing |
| `memory_style.py` | Memory formatting rules |
| `config.py` | Configuration settings |
| `requirements.txt` | Python dependencies |
| `Procfile` | Railway deployment config |
| `openwebui_filter.py` | OpenWebUI filter (copy to Functions) |
| `dashboard.py` | Streamlit dashboard |
| `AIDialogue/ai_dialogue_pipe.py` | Three-way conversation pipe |
| `archive_thread.py` | Import conversations to GAM |
| `smart_import.py` | Intelligent memory extraction |

---

## Credits

Built with:
- [GAM (General Agentic Memory)](https://github.com/VectorSpaceLab/general-agentic-memory)
- [OpenWebUI](https://openwebui.com)
- [FastAPI](https://fastapi.tiangolo.com)
- [Streamlit](https://streamlit.io)
- [Railway](https://railway.app)

---

**Questions?** Check the troubleshooting section or review the code comments for more details.
