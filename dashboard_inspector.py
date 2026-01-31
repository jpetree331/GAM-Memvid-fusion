#!/usr/bin/env python3
"""
Dashboard Inspector - Visual Manager for The Librarian Architecture

============================================================
MISSION CONTROL: Inspect, Search, and Test Your Memory Vaults
============================================================

A Streamlit application for managing and debugging Memvid vaults.

Features:
- Vault Selector: Switch between different AI persona vaults
- Stats Panel: Pearl count, file size, last update
- The Feed: Browse memories in reverse chronological order
- Search & Synthesis Lab: Test queries and live abstract generation
- Management: Soft delete Pearls

Usage:
    streamlit run dashboard_inspector.py
    streamlit run dashboard_inspector.py --server.port 8502

Requirements:
    pip install streamlit
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from memvid_store import MemvidStore, Pearl, SearchResult, get_store, VaultManager
from synthesizer import Synthesizer, get_synthesizer
from config import config


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Librarian Inspector",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better formatting
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
# Helper Functions
# =============================================================================

def get_available_vaults() -> List[str]:
    """Get list of available vault files."""
    vaults_dir = Path(config.VAULTS_DIR)
    if not vaults_dir.exists():
        return []

    vaults = []
    for f in vaults_dir.glob("*.mv2"):
        vaults.append(f.stem)

    # Add common vault names even if they don't exist yet
    common_vaults = ["opus", "eli", "aria", "test_debug"]
    for v in common_vaults:
        if v not in vaults:
            vaults.append(v)

    return sorted(vaults)


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


def run_async(coro):
    """Run an async function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# =============================================================================
# Sidebar: Vault Selector
# =============================================================================

def render_sidebar():
    """Render the sidebar with vault selector and settings."""
    st.sidebar.title("📚 Librarian Inspector")
    st.sidebar.markdown("---")

    # Vault selector
    st.sidebar.subheader("Vault Selection")
    vaults = get_available_vaults()

    selected_vault = st.sidebar.selectbox(
        "Select Vault",
        options=vaults,
        index=0 if vaults else None,
        help="Choose which AI persona's memory vault to inspect"
    )

    st.sidebar.markdown("---")

    # Settings
    st.sidebar.subheader("Settings")
    show_deleted = st.sidebar.checkbox(
        "Show Deleted Pearls",
        value=False,
        help="Include soft-deleted memories in the feed"
    )

    user_name = st.sidebar.text_input(
        "User Name (for synthesis)",
        value="User",
        help="Name to use when generating abstracts"
    )

    st.sidebar.markdown("---")

    # API Status
    st.sidebar.subheader("API Status")
    if config.OPENAI_API_KEY:
        st.sidebar.success("OpenAI: Connected")
    else:
        st.sidebar.error("OpenAI: Not configured")

    return selected_vault, show_deleted, user_name


# =============================================================================
# Stats Panel
# =============================================================================

def render_stats_panel(store: MemvidStore):
    """Render the stats panel."""
    st.subheader("📊 Vault Statistics")

    stats = store.get_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Pearls",
            value=stats.get("total_memories", 0)
        )

    with col2:
        st.metric(
            label="File Size",
            value=f"{stats.get('file_size_mb', 0):.2f} MB"
        )

    with col3:
        st.metric(
            label="Last Updated",
            value=format_timestamp(stats.get("last_update"))
        )

    with col4:
        st.metric(
            label="Vault Version",
            value=stats.get("version", "Unknown")
        )


# =============================================================================
# The Feed: Data Browser
# =============================================================================

def render_feed(store: MemvidStore, show_deleted: bool, user_name: str):
    """Render the memory feed."""
    st.subheader("📜 The Feed")

    # Get all Pearls (reverse chronological)
    # Using a broad search to get recent items
    results = store.search_pearls(
        query="*",  # Match all
        limit=50,
        include_deleted=show_deleted
    )

    if not results:
        st.info("No memories found in this vault. Try adding some!")
        return

    # Sort by timestamp (most recent first)
    results.sort(key=lambda r: r.pearl.created_at or "", reverse=True)

    st.write(f"Showing {len(results)} memories")

    for i, result in enumerate(results):
        pearl = result.pearl
        render_pearl_card(pearl, i, store, user_name)


def render_pearl_card(pearl: Pearl, index: int, store: MemvidStore, user_name: str):
    """Render a single Pearl card in the feed."""
    is_deleted = pearl.status == "deleted"

    # Card container
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
            # Delete button (only for non-deleted)
            if not is_deleted:
                if st.button("🗑️ Delete", key=f"del_{pearl.id}_{index}"):
                    store.soft_delete(pearl.id, reason="Deleted via dashboard")
                    st.rerun()

        # Tags row
        if pearl.tags:
            tags_html = " ".join([f"<span class='tag-badge'>{tag}</span>" for tag in pearl.tags[:8]])
            st.markdown(tags_html, unsafe_allow_html=True)

        # Content expander
        user_words = word_count(pearl.user_message)
        ai_words = word_count(pearl.ai_response)

        with st.expander(f"View Full Content ({user_words + ai_words:,} words)"):
            # User message
            st.markdown("**User:**")
            st.text_area(
                "User message",
                value=pearl.user_message,
                height=150,
                key=f"user_{pearl.id}_{index}",
                label_visibility="collapsed"
            )

            # AI response
            st.markdown("**AI Response:**")
            st.text_area(
                "AI response",
                value=pearl.ai_response,
                height=200,
                key=f"ai_{pearl.id}_{index}",
                label_visibility="collapsed"
            )

            # Synthesis button
            if st.button("🧪 Generate Abstract", key=f"synth_{pearl.id}_{index}"):
                with st.spinner("Synthesizing..."):
                    try:
                        abstract = synthesize_pearl_sync(pearl, user_name)
                        st.session_state[f"abstract_{pearl.id}"] = abstract
                    except Exception as e:
                        st.error(f"Synthesis failed: {e}")

            # Show abstract if generated
            if f"abstract_{pearl.id}" in st.session_state:
                st.markdown("---")
                st.markdown("**📝 Generated Abstract:**")
                st.markdown(f"<div class='synthesis-box'>{st.session_state[f'abstract_{pearl.id}']}</div>", unsafe_allow_html=True)

        st.markdown("---")


# =============================================================================
# Search & Synthesis Lab
# =============================================================================

def render_search_lab(store: MemvidStore, show_deleted: bool, user_name: str):
    """Render the search and synthesis lab."""
    st.subheader("🔬 Search & Synthesis Lab")

    # Search box
    col1, col2 = st.columns([4, 1])

    with col1:
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g., Theology, consciousness, family...",
            help="Enter keywords to search for relevant memories"
        )

    with col2:
        search_mode = st.selectbox(
            "Mode",
            options=["hybrid", "semantic", "lexical"],
            help="Search mode: hybrid combines semantic + lexical"
        )

    if search_query:
        # Perform search
        results = store.search_pearls(
            query=search_query,
            limit=10,
            mode=search_mode,
            include_deleted=show_deleted
        )

        if results:
            st.success(f"Found {len(results)} matching memories")

            for i, result in enumerate(results):
                render_search_result(result, i, user_name)
        else:
            st.warning("No results found. Try different keywords.")


def render_search_result(result: SearchResult, index: int, user_name: str):
    """Render a single search result with synthesis option."""
    pearl = result.pearl

    with st.container():
        # Header with score
        col1, col2, col3 = st.columns([3, 1, 2])

        with col1:
            timestamp = format_timestamp(pearl.created_at)
            st.markdown(f"**{timestamp}** | {pearl.category}")

        with col2:
            st.markdown(f"<span class='score-badge'>Score: {result.score:.3f}</span>", unsafe_allow_html=True)

        with col3:
            if st.button("🧪 Test Abstract", key=f"search_synth_{pearl.id}_{index}"):
                st.session_state[f"synth_target_{index}"] = pearl.id

        # Tags
        if pearl.tags:
            st.markdown(" ".join([f"`{tag}`" for tag in pearl.tags[:6]]))

        # Content preview
        preview = pearl.user_message[:200].replace('\n', ' ')
        st.markdown(f"*{preview}...*")

        # Side-by-side comparison if synthesis requested
        if st.session_state.get(f"synth_target_{index}") == pearl.id:
            render_synthesis_comparison(pearl, index, user_name)

        st.markdown("---")


def render_synthesis_comparison(pearl: Pearl, index: int, user_name: str):
    """Render side-by-side comparison of raw content and abstract."""
    st.markdown("### 📊 Raw vs. Synthesized Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Raw Pearl Content:**")
        user_words = word_count(pearl.user_message)
        ai_words = word_count(pearl.ai_response)
        st.markdown(f"*{user_words + ai_words:,} total words*")

        with st.container():
            st.markdown("**User:**")
            st.text_area(
                "Raw user",
                pearl.user_message[:2000],
                height=200,
                key=f"raw_user_{index}",
                label_visibility="collapsed"
            )
            st.markdown("**AI:**")
            st.text_area(
                "Raw AI",
                pearl.ai_response[:2000],
                height=200,
                key=f"raw_ai_{index}",
                label_visibility="collapsed"
            )

    with col2:
        st.markdown("**Synthesized Abstract:**")

        # Generate abstract
        with st.spinner("Generating abstract with gpt-4o-mini..."):
            try:
                abstract = synthesize_pearl_sync(pearl, user_name)
                abstract_words = word_count(abstract)
                compression = (user_words + ai_words) / max(abstract_words, 1)

                st.markdown(f"*{abstract_words} words ({compression:.1f}x compression)*")
                st.markdown(f"<div class='synthesis-box'>{abstract}</div>", unsafe_allow_html=True)

                # Quality metrics
                st.markdown("**Quality Check:**")
                has_quotes = '"' in abstract
                st.markdown(f"- Contains quotes: {'✅' if has_quotes else '❌'}")
                in_range = 180 <= abstract_words <= 350
                st.markdown(f"- Word count (200-300): {'✅' if in_range else '⚠️'} ({abstract_words})")

            except Exception as e:
                st.error(f"Synthesis failed: {e}")
                st.markdown("**Possible causes:**")
                st.markdown("- OpenAI API key not configured")
                st.markdown("- Network connectivity issues")
                st.markdown("- API rate limits")


def synthesize_pearl_sync(pearl: Pearl, user_name: str) -> str:
    """Synchronously synthesize a Pearl into an abstract."""
    async def _synth():
        synth = get_synthesizer()
        result = await synth.synthesize_pearl(
            pearl_id=pearl.id,
            user_message=pearl.user_message,
            ai_response=pearl.ai_response,
            user_name=user_name,
            tags=pearl.tags,
            timestamp=pearl.created_at
        )
        return result.abstract

    return run_async(_synth())


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Sidebar
    selected_vault, show_deleted, user_name = render_sidebar()

    # Header
    st.title("📚 Librarian Inspector")
    st.markdown(f"*Inspecting vault: **{selected_vault}***")

    if not selected_vault:
        st.warning("No vault selected. Choose a vault from the sidebar.")
        return

    # Initialize store
    try:
        store = get_store(selected_vault)
    except Exception as e:
        st.error(f"Failed to open vault: {e}")
        return

    # Stats Panel
    render_stats_panel(store)

    st.markdown("---")

    # Tabs for different views
    tab1, tab2 = st.tabs(["📜 The Feed", "🔬 Search & Synthesis Lab"])

    with tab1:
        render_feed(store, show_deleted, user_name)

    with tab2:
        render_search_lab(store, show_deleted, user_name)

    # Footer
    st.markdown("---")
    st.markdown(
        "<small>Librarian Inspector v1.0 | "
        "Part of The Librarian Architecture for GAM-Memvid</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
