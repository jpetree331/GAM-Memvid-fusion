#!/usr/bin/env python3
"""
Dashboard Inspector (API Mode) - Visual Manager for GAM-Memvid

============================================================
Connects to Railway-hosted GAM-Memvid server via HTTP API
============================================================

A Streamlit application for managing and debugging Memvid vaults
via API calls (no direct file access needed).

Features:
- Vault Selector: Switch between different AI persona vaults
- Stats Panel: Pearl count, last update
- The Feed: Browse memories in reverse chronological order
- Search & Synthesis Lab: Test queries and live synthesis
- Management: Soft delete Pearls

Usage:
    streamlit run dashboard_api.py --server.port $PORT --server.address 0.0.0.0

Environment:
    MEMORY_SERVER_URL - URL of GAM-Memvid server (default: http://localhost:8100)
"""

import os
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

import streamlit as st
import httpx

# =============================================================================
# Configuration
# =============================================================================

MEMORY_SERVER_URL = os.getenv(
    "MEMORY_SERVER_URL",
    "https://gam-memvid-fusion-production.up.railway.app"
)

# HTTP client timeout
REQUEST_TIMEOUT = 30.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Pearl:
    """Represents a Pearl from the API."""
    id: str
    user_message: str
    ai_response: str
    category: str = "context"
    importance: str = "normal"
    tags: List[str] = None
    created_at: Optional[str] = None
    status: str = "active"

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SearchResult:
    """A search result with score."""
    pearl: Pearl
    score: float


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Librarian Inspector (API)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .pearl-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .tag-badge {
        background-color: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 5px;
    }
    .deleted-badge {
        background-color: #f44336;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
    }
    .score-badge {
        background-color: #2196F3;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
    }
    .stats-box {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .synthesis-box {
        background-color: #fff3e0;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API Client Functions
# =============================================================================

def api_health_check() -> dict:
    """Check API health."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            r = client.get(f"{MEMORY_SERVER_URL}/health")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def api_list_models() -> List[str]:
    """Get list of available models/vaults."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            # Try /memvid/vaults first (new endpoint)
            try:
                r = client.get(f"{MEMORY_SERVER_URL}/memvid/vaults")
                r.raise_for_status()
                data = r.json()
                vaults = data.get("vaults", [])
                # Strip .mv2 extension
                return [v.replace(".mv2", "") for v in vaults]
            except:
                pass

            # Fallback to /models
            r = client.get(f"{MEMORY_SERVER_URL}/models")
            r.raise_for_status()
            data = r.json()
            return data.get("models", [])
    except Exception as e:
        st.error(f"Failed to list models: {e}")
        return []


def api_get_stats(model_id: str) -> dict:
    """Get vault statistics."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            r = client.get(f"{MEMORY_SERVER_URL}/memory/{model_id}/stats")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_get_recent(model_id: str, limit: int = 50) -> List[Pearl]:
    """Get recent memories."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            r = client.get(
                f"{MEMORY_SERVER_URL}/memory/{model_id}/recent",
                params={"limit": limit}
            )
            r.raise_for_status()
            data = r.json()

            # Handle both "memories" and "items" keys
            items = data.get("memories") or data.get("items") or []

            pearls = []
            for m in items:
                # Extract text content - handle hydrated format
                user_msg = m.get("user_message", "")
                ai_msg = m.get("ai_response", "")

                # If not found, try to parse from "text" field
                if not user_msg and not ai_msg:
                    text = m.get("text", "") or m.get("content", "")
                    if "User:" in text and "AI:" in text:
                        parts = text.split("AI:", 1)
                        user_msg = parts[0].replace("User:", "").strip()
                        ai_msg = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        user_msg = text

                pearls.append(Pearl(
                    id=m.get("id", ""),
                    user_message=user_msg,
                    ai_response=ai_msg,
                    category=m.get("category", "context"),
                    importance=m.get("importance", "normal"),
                    tags=m.get("tags", []),
                    created_at=m.get("created_at"),
                    status=m.get("status", "active")
                ))
            return pearls
    except Exception as e:
        st.error(f"Failed to get recent memories: {e}")
        return []


def api_search(model_id: str, query: str, limit: int = 10) -> List[SearchResult]:
    """Search memories."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            # Try /memvid/search first, then fall back to /memory/search
            try:
                r = client.post(
                    f"{MEMORY_SERVER_URL}/memvid/search",
                    json={
                        "model_id": model_id,
                        "query": query,
                        "limit": limit
                    }
                )
                r.raise_for_status()
            except:
                r = client.post(
                    f"{MEMORY_SERVER_URL}/memory/search",
                    json={
                        "model_id": model_id,
                        "query": query,
                        "limit": limit
                    }
                )
                r.raise_for_status()

            data = r.json()

            # Handle both "results" and "items" keys
            items = data.get("results") or data.get("items") or []

            results = []
            for item in items:
                # Extract text content - handle hydrated format
                user_msg = item.get("user_message", "")
                ai_msg = item.get("ai_response", "")

                # If not found, try to parse from "text" field
                if not user_msg and not ai_msg:
                    text = item.get("text", "") or item.get("content", "")
                    if "User:" in text and "AI:" in text:
                        parts = text.split("AI:", 1)
                        user_msg = parts[0].replace("User:", "").strip()
                        ai_msg = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        user_msg = text

                pearl = Pearl(
                    id=item.get("id", ""),
                    user_message=user_msg,
                    ai_response=ai_msg,
                    category=item.get("category", "context"),
                    importance=item.get("importance", "normal"),
                    tags=item.get("tags", []),
                    created_at=item.get("created_at"),
                    status=item.get("status", "active")
                )
                results.append(SearchResult(
                    pearl=pearl,
                    score=item.get("score", 0.0)
                ))
            return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []


def api_get_context(model_id: str, query: str, user_name: str = "User", limit: int = 5) -> dict:
    """Get synthesized context."""
    try:
        with httpx.Client(timeout=60.0) as client:  # Longer timeout for synthesis
            r = client.post(
                f"{MEMORY_SERVER_URL}/memory/context",
                json={
                    "model_id": model_id,
                    "query": query,
                    "user_name": user_name,
                    "limit": limit,
                    "max_words": 400
                }
            )
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e), "context": ""}


def api_delete_pearl(model_id: str, pearl_id: str, reason: str = "") -> dict:
    """Soft delete a Pearl."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            r = client.post(
                f"{MEMORY_SERVER_URL}/memory/{model_id}/delete",
                json={
                    "pearl_id": pearl_id,
                    "reason": reason or "Deleted via dashboard"
                }
            )
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


# =============================================================================
# Helper Functions
# =============================================================================

def format_timestamp(ts: Optional[str]) -> str:
    """Format timestamp for display."""
    if not ts:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return ts[:16] if len(ts) > 16 else ts


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split()) if text else 0


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar."""
    st.sidebar.title("📚 Librarian Inspector")
    st.sidebar.markdown(f"*API Mode: {MEMORY_SERVER_URL}*")
    st.sidebar.markdown("---")

    # Vault selector
    st.sidebar.subheader("Vault Selection")
    models = api_list_models()

    # Allow custom model entry
    custom_model = st.sidebar.text_input(
        "Or enter model ID:",
        placeholder="e.g., qwen-235b-a22bth-origf2"
    )

    if custom_model:
        selected_vault = custom_model
    elif models:
        selected_vault = st.sidebar.selectbox(
            "Select Vault",
            options=models,
            index=0
        )
    else:
        selected_vault = st.sidebar.text_input(
            "Model ID (no vaults found)",
            value="test"
        )

    st.sidebar.markdown("---")

    # Settings
    st.sidebar.subheader("Settings")
    user_name = st.sidebar.text_input(
        "User Name (for synthesis)",
        value="User"
    )

    st.sidebar.markdown("---")

    # API Status
    st.sidebar.subheader("API Status")
    health = api_health_check()
    if health.get("status") == "healthy":
        st.sidebar.success(f"Server: Connected")
        st.sidebar.markdown(f"Version: {health.get('version', 'Unknown')}")
    else:
        st.sidebar.error(f"Server: {health.get('error', 'Disconnected')}")

    return selected_vault, user_name


# =============================================================================
# Stats Panel
# =============================================================================

def render_stats_panel(model_id: str):
    """Render the stats panel."""
    st.subheader("📊 Vault Statistics")

    stats = api_get_stats(model_id)

    if "error" in stats:
        st.warning(f"Could not load stats: {stats['error']}")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Pearls",
            value=stats.get("pearl_count", stats.get("total_memories", 0))
        )

    with col2:
        st.metric(
            label="Active",
            value=stats.get("active_count", "?")
        )

    with col3:
        st.metric(
            label="Last Updated",
            value=format_timestamp(stats.get("last_update"))
        )

    with col4:
        st.metric(
            label="Core Memories",
            value=stats.get("core_pearls", 0)
        )


# =============================================================================
# The Feed
# =============================================================================

def render_feed(model_id: str, user_name: str):
    """Render the memory feed."""
    st.subheader("📜 The Feed")

    pearls = api_get_recent(model_id, limit=50)

    if not pearls:
        st.info("No memories found in this vault. Try adding some!")
        return

    # Sort by timestamp (most recent first)
    pearls.sort(key=lambda p: p.created_at or "", reverse=True)

    st.write(f"Showing {len(pearls)} memories")

    for i, pearl in enumerate(pearls):
        render_pearl_card(pearl, i, model_id, user_name)


def render_pearl_card(pearl: Pearl, index: int, model_id: str, user_name: str):
    """Render a single Pearl card."""
    is_deleted = pearl.status == "deleted"

    with st.container():
        # Header row
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            timestamp = format_timestamp(pearl.created_at)
            status_badge = " 🗑️ DELETED" if is_deleted else ""
            st.markdown(f"**{timestamp}**{status_badge}")

        with col2:
            st.markdown(f"*{pearl.category}* | {pearl.importance}")

        with col3:
            if not is_deleted:
                if st.button("🗑️ Delete", key=f"del_{pearl.id}_{index}"):
                    result = api_delete_pearl(model_id, pearl.id)
                    if result.get("success"):
                        st.success("Deleted!")
                        st.rerun()
                    else:
                        st.error(result.get("message", "Delete failed"))

        # Tags row
        if pearl.tags:
            tags_html = " ".join([f"<span class='tag-badge'>{tag}</span>" for tag in pearl.tags[:8]])
            st.markdown(tags_html, unsafe_allow_html=True)

        # Content expander
        user_words = word_count(pearl.user_message)
        ai_words = word_count(pearl.ai_response)

        with st.expander(f"View Full Content ({user_words + ai_words:,} words)"):
            st.markdown("**User:**")
            st.text_area(
                "User message",
                value=pearl.user_message,
                height=150,
                key=f"user_{pearl.id}_{index}",
                label_visibility="collapsed"
            )

            st.markdown("**AI Response:**")
            st.text_area(
                "AI response",
                value=pearl.ai_response,
                height=200,
                key=f"ai_{pearl.id}_{index}",
                label_visibility="collapsed"
            )

        st.markdown("---")


# =============================================================================
# Search & Synthesis Lab
# =============================================================================

def render_search_lab(model_id: str, user_name: str):
    """Render the search and synthesis lab."""
    st.subheader("🔬 Search & Synthesis Lab")

    # Search box
    col1, col2 = st.columns([4, 1])

    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g., Theology, consciousness, INTJ..."
        )

    with col2:
        search_limit = st.number_input("Limit", min_value=1, max_value=20, value=5)

    if search_query:
        # Perform search
        results = api_search(model_id, search_query, limit=search_limit)

        if results:
            st.success(f"Found {len(results)} matching memories")

            for i, result in enumerate(results):
                render_search_result(result, i, model_id, user_name)

            # Synthesis section
            st.markdown("---")
            st.subheader("🧪 Test Synthesis")

            if st.button("Generate Synthesized Context"):
                with st.spinner("Synthesizing with gpt-4o-mini..."):
                    context_data = api_get_context(
                        model_id=model_id,
                        query=search_query,
                        user_name=user_name,
                        limit=search_limit
                    )

                    if context_data.get("error"):
                        st.error(f"Synthesis failed: {context_data['error']}")
                    else:
                        context = context_data.get("context", "")
                        num_pearls = context_data.get("num_pearls", 0)

                        st.markdown(f"*Used {num_pearls} pearls*")
                        st.markdown(f"<div class='synthesis-box'>{context}</div>", unsafe_allow_html=True)

                        # Word count
                        context_words = word_count(context)
                        st.markdown(f"*{context_words} words in synthesized context*")
        else:
            st.warning("No results found. Try different keywords.")


def render_search_result(result: SearchResult, index: int, model_id: str, user_name: str):
    """Render a single search result."""
    pearl = result.pearl

    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            timestamp = format_timestamp(pearl.created_at)
            st.markdown(f"**{timestamp}** | {pearl.category}")

        with col2:
            st.markdown(f"<span class='score-badge'>Score: {result.score:.3f}</span>", unsafe_allow_html=True)

        # Tags
        if pearl.tags:
            st.markdown(" ".join([f"`{tag}`" for tag in pearl.tags[:6]]))

        # Content preview
        preview = pearl.user_message[:200].replace('\n', ' ')
        st.markdown(f"*{preview}...*")

        st.markdown("---")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Sidebar
    selected_vault, user_name = render_sidebar()

    # Header
    st.title("📚 Librarian Inspector (API Mode)")
    st.markdown(f"*Inspecting vault: **{selected_vault}***")

    if not selected_vault:
        st.warning("No vault selected. Enter a model ID in the sidebar.")
        return

    # Stats Panel
    render_stats_panel(selected_vault)

    st.markdown("---")

    # Tabs
    tab1, tab2 = st.tabs(["📜 The Feed", "🔬 Search & Synthesis Lab"])

    with tab1:
        render_feed(selected_vault, user_name)

    with tab2:
        render_search_lab(selected_vault, user_name)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<small>Librarian Inspector v2.0 (API Mode) | "
        f"Server: {MEMORY_SERVER_URL}</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
