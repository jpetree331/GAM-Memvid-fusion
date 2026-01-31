"""
GAM Memory Dashboard - Visual memory management interface.

A Streamlit-based dashboard for browsing, editing, and managing memories.

Run with:
    streamlit run dashboard.py

Or from Python:
    python dashboard.py
"""
import streamlit as st
import httpx
import json
from datetime import datetime
from typing import Optional
import os

# Configuration - Default to Railway URL
# When deployed on Railway, use internal URL for faster communication
# Set GAM_INTERNAL_URL for Railway-to-Railway communication
DEFAULT_API_URL = os.getenv(
    "GAM_INTERNAL_URL",  # Railway internal URL (fastest)
    os.getenv(
        "GAM_SERVER_URL",  # Public URL fallback
        "https://dynamic-connection-production-2f3e.up.railway.app"
    )
)

# Page config
st.set_page_config(
    page_title="GAM Memory Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# API Client
# =============================================================================

class GAMClient:
    """Client for communicating with the GAM server."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or DEFAULT_API_URL
        self.client = httpx.Client(timeout=60.0)  # Increased timeout for Railway
    
    def set_base_url(self, url: str):
        """Update the base URL."""
        self.base_url = url
    
    def health_check(self) -> dict:
        """Check server health."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def list_models(self) -> list[str]:
        """Get all models with memories."""
        try:
            response = self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()
            return data.get("all_models", [])
        except Exception:
            return []
    
    def list_memories(
        self, 
        model_id: str, 
        limit: int = 50, 
        offset: int = 0,
        category: Optional[str] = None,
        importance: Optional[str] = None
    ) -> dict:
        """List memories for a model."""
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if importance:
            params["importance"] = importance
        
        try:
            response = self.client.get(
                f"{self.base_url}/memory/{model_id}/list",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e), "memories": []}
    
    def get_memory(self, model_id: str, memory_id: str) -> dict:
        """Get a specific memory."""
        try:
            response = self.client.get(
                f"{self.base_url}/memory/{model_id}/get/{memory_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def edit_memory(self, model_id: str, memory_id: str, new_content: str) -> dict:
        """Edit a memory's content."""
        try:
            response = self.client.post(
                f"{self.base_url}/memory/{model_id}/edit",
                json={"memory_id": memory_id, "new_content": new_content}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_memory(self, model_id: str, memory_id: str) -> dict:
        """Delete a memory."""
        try:
            response = self.client.post(
                f"{self.base_url}/memory/{model_id}/delete",
                json={"memory_id": memory_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def bulk_delete(self, model_id: str, memory_ids: list[str]) -> dict:
        """Delete multiple memories."""
        try:
            response = self.client.post(
                f"{self.base_url}/memory/{model_id}/bulk-delete",
                json=memory_ids
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def update_organization(
        self, 
        model_id: str, 
        memory_id: str,
        category: Optional[str] = None,
        importance: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> dict:
        """Update memory organization."""
        try:
            response = self.client.post(
                f"{self.base_url}/memory/organize/{model_id}/update",
                json={
                    "memory_id": memory_id,
                    "category": category,
                    "importance": importance,
                    "tags": tags
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self, model_id: str) -> dict:
        """Get organization statistics."""
        try:
            response = self.client.get(
                f"{self.base_url}/memory/organize/{model_id}/stats"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def validate_memory(self, model_id: str, content: str) -> dict:
        """Validate memory content for pronoun issues."""
        try:
            response = self.client.post(
                f"{self.base_url}/memory/style/{model_id}/validate",
                json={"content": content}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def export_memories(self, model_id: str) -> dict:
        """Export all memories for a model."""
        try:
            response = self.client.get(
                f"{self.base_url}/memory/export/{model_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Bucket Management
    def list_buckets(self) -> dict:
        """List all memory buckets."""
        try:
            response = self.client.get(f"{self.base_url}/buckets/list")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e), "buckets": []}
    
    def migrate_bucket(self, from_id: str, to_id: str, merge: bool = False) -> dict:
        """Migrate/rename a bucket."""
        try:
            response = self.client.post(
                f"{self.base_url}/buckets/migrate",
                json={"from_id": from_id, "to_id": to_id, "merge": merge}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_bucket(self, model_id: str) -> dict:
        """Delete a bucket entirely."""
        try:
            response = self.client.delete(
                f"{self.base_url}/buckets/{model_id}",
                params={"confirm": "true"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}


# Initialize client
client = GAMClient()


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with model selection and filters."""
    st.sidebar.title("🧠 GAM Memory Dashboard")
    
    # Server URL configuration
    with st.sidebar.expander("⚙️ Server Settings", expanded=False):
        server_url = st.text_input(
            "GAM Server URL",
            value=st.session_state.get("server_url", DEFAULT_API_URL),
            help="Enter your Railway or local GAM server URL"
        )
        if server_url != st.session_state.get("server_url"):
            st.session_state.server_url = server_url
            client.set_base_url(server_url)
            st.rerun()
    
    # Ensure client has correct URL
    current_url = st.session_state.get("server_url", DEFAULT_API_URL)
    if client.base_url != current_url:
        client.set_base_url(current_url)
    
    # Server status
    health = client.health_check()
    if health.get("status") == "healthy":
        st.sidebar.success(f"✓ Server connected")
        st.sidebar.caption(f"Active models: {health.get('models_active', 0)}")
    else:
        st.sidebar.error(f"✗ Server error: {health.get('error', 'Unknown')}")
        st.sidebar.caption(f"URL: {current_url}")
        return None, None, None
    
    st.sidebar.divider()
    
    # Model selection
    models = client.list_models()
    if not models:
        st.sidebar.warning("No models with memories found")
        return None, None, None
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        models,
        help="Each model has its own isolated memory bucket"
    )
    
    st.sidebar.divider()
    
    # Filters
    st.sidebar.subheader("Filters")
    
    categories = [
        None, "preference", "fact", "event", "task", 
        "relationship", "context", "skill", "theology",
        "science", "ai_theory", "ai_self"
    ]
    selected_category = st.sidebar.selectbox(
        "Category",
        categories,
        format_func=lambda x: "All" if x is None else x.title()
    )
    
    importance_levels = [None, "core", "high", "normal", "low", "archived"]
    selected_importance = st.sidebar.selectbox(
        "Importance",
        importance_levels,
        format_func=lambda x: "All" if x is None else x.title()
    )
    
    return selected_model, selected_category, selected_importance


def render_stats(model_id: str):
    """Render memory statistics."""
    stats_response = client.get_stats(model_id)
    
    if not stats_response.get("success"):
        return
    
    stats = stats_response.get("statistics", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Memories", stats.get("total_memories", 0))
    with col2:
        st.metric("Core Memories", stats.get("core_memories", 0))
    with col3:
        st.metric("Categories", stats.get("categories_used", 0))
    with col4:
        st.metric("Tags", stats.get("tags_used", 0))


def render_memory_card(memory: dict, model_id: str, index: int = 0):
    """Render a single memory card with edit/delete options."""
    memory_id = memory.get("memory_id", f"unknown_{index}")
    content = memory.get("content", "")
    category = memory.get("category", "context")
    importance = memory.get("importance", "normal")
    tags = memory.get("tags", [])
    created_at = memory.get("created_at", "")
    
    # Color code by importance
    importance_colors = {
        "core": "🔴",
        "high": "🟠",
        "normal": "🟢",
        "low": "🔵",
        "archived": "⚪"
    }
    importance_icon = importance_colors.get(importance, "🟢")
    
    with st.container():
        # Header row
        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
        
        with col1:
            st.markdown(f"**{importance_icon} {category.title()}**")
            if tags:
                st.caption(" | ".join([f"#{tag}" for tag in tags]))
        
        with col2:
            edit_button = st.button("✏️ Edit", key=f"edit_{memory_id}")
        
        with col3:
            delete_button = st.button("🗑️ Delete", key=f"delete_{memory_id}")
        
        # Content
        st.text_area(
            "Content",
            content,
            height=100,
            disabled=True,
            key=f"content_{memory_id}",
            label_visibility="collapsed"
        )
        
        # Metadata - format timestamp nicely if it's ISO format
        display_time = created_at
        if created_at and "T" in created_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        st.caption(f"ID: `{memory_id}` | Created: {display_time}")
        
        # Handle edit
        if edit_button:
            st.session_state[f"editing_{memory_id}"] = True
        
        # Handle delete
        if delete_button:
            st.session_state[f"confirm_delete_{memory_id}"] = True
        
        # Edit dialog
        if st.session_state.get(f"editing_{memory_id}"):
            render_edit_dialog(memory, model_id)
        
        # Delete confirmation
        if st.session_state.get(f"confirm_delete_{memory_id}"):
            render_delete_confirmation(memory_id, model_id)
        
        st.divider()


def render_edit_dialog(memory: dict, model_id: str):
    """Render the edit dialog for a memory."""
    memory_id = memory.get("memory_id")
    
    st.subheader("Edit Memory")
    
    # Content editor
    new_content = st.text_area(
        "Edit content:",
        memory.get("content", ""),
        height=150,
        key=f"edit_content_{memory_id}"
    )
    
    # Validate for pronoun issues
    if new_content:
        validation = client.validate_memory(model_id, new_content)
        if validation.get("success") and not validation.get("valid", True):
            st.warning("⚠️ Potential pronoun issues detected:")
            for issue in validation.get("issues", []):
                st.caption(f"• {issue}")
            for suggestion in validation.get("suggestions", []):
                st.caption(f"💡 {suggestion}")
    
    # Category and importance editors
    col1, col2 = st.columns(2)
    
    with col1:
        new_category = st.selectbox(
            "Category",
            ["preference", "fact", "event", "task", "relationship", 
             "context", "skill", "theology", "science", "ai_theory", "ai_self"],
            index=["preference", "fact", "event", "task", "relationship", 
                   "context", "skill", "theology", "science", "ai_theory", "ai_self"].index(
                       memory.get("category", "context")
                   ) if memory.get("category") in ["preference", "fact", "event", "task", 
                       "relationship", "context", "skill", "theology", "science", 
                       "ai_theory", "ai_self"] else 5,
            key=f"edit_category_{memory_id}"
        )
    
    with col2:
        new_importance = st.selectbox(
            "Importance",
            ["core", "high", "normal", "low", "archived"],
            index=["core", "high", "normal", "low", "archived"].index(
                memory.get("importance", "normal")
            ) if memory.get("importance") in ["core", "high", "normal", "low", "archived"] else 2,
            key=f"edit_importance_{memory_id}"
        )
    
    # Tags editor
    current_tags = ", ".join(memory.get("tags", []))
    new_tags_str = st.text_input(
        "Tags (comma-separated)",
        current_tags,
        key=f"edit_tags_{memory_id}"
    )
    new_tags = [t.strip() for t in new_tags_str.split(",") if t.strip()]
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Changes", key=f"save_{memory_id}"):
            # Update content if changed
            if new_content != memory.get("content"):
                result = client.edit_memory(model_id, memory_id, new_content)
                if not result.get("success"):
                    st.error(f"Failed to update content: {result.get('error')}")
                    return
            
            # Update organization
            result = client.update_organization(
                model_id, memory_id,
                category=new_category,
                importance=new_importance,
                tags=new_tags
            )
            
            if result.get("success"):
                st.success("✓ Memory updated!")
                st.session_state[f"editing_{memory_id}"] = False
                # Clear any cached data and force full refresh
                for key in list(st.session_state.keys()):
                    if key.startswith("editing_") or key.startswith("FormSubmitter"):
                        del st.session_state[key]
                import time
                time.sleep(0.5)  # Brief delay to ensure server saves
                st.rerun()
            else:
                st.error(f"Failed to update: {result.get('error')}")
    
    with col2:
        if st.button("Cancel", key=f"cancel_edit_{memory_id}"):
            st.session_state[f"editing_{memory_id}"] = False
            st.rerun()


def render_delete_confirmation(memory_id: str, model_id: str):
    """Render delete confirmation dialog."""
    st.warning(f"⚠️ Are you sure you want to delete this memory?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Yes, Delete", key=f"confirm_delete_yes_{memory_id}"):
            result = client.delete_memory(model_id, memory_id)
            if result.get("success"):
                st.success("✓ Memory deleted!")
                st.session_state[f"confirm_delete_{memory_id}"] = False
                st.rerun()
            else:
                st.error(f"Failed to delete: {result.get('error')}")
    
    with col2:
        if st.button("Cancel", key=f"cancel_delete_{memory_id}"):
            st.session_state[f"confirm_delete_{memory_id}"] = False
            st.rerun()


def render_bulk_actions(model_id: str, memories: list):
    """Render bulk action controls."""
    st.subheader("Bulk Actions")
    
    # Select memories for bulk operations
    if "selected_memories" not in st.session_state:
        st.session_state.selected_memories = []
    
    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    
    with col1:
        # Search within memories
        search_term = st.text_input("🔍 Search memories", key="memory_search")
    
    with col2:
        if st.button("Select All Visible"):
            st.session_state.selected_memories = [m.get("memory_id") for m in memories]
            st.rerun()
    
    with col3:
        if st.button("Clear Selection"):
            st.session_state.selected_memories = []
            st.rerun()
    
    # Bulk delete
    if st.session_state.selected_memories:
        st.info(f"Selected: {len(st.session_state.selected_memories)} memories")
        
        if st.button("🗑️ Delete Selected", type="secondary"):
            if st.session_state.get("confirm_bulk_delete"):
                result = client.bulk_delete(model_id, st.session_state.selected_memories)
                if result.get("success"):
                    st.success(f"✓ Deleted {len(result.get('deleted', []))} memories")
                    st.session_state.selected_memories = []
                    st.session_state.confirm_bulk_delete = False
                    st.rerun()
                else:
                    st.error(f"Failed: {result.get('error')}")
            else:
                st.session_state.confirm_bulk_delete = True
                st.warning("Click again to confirm deletion")
    
    return search_term


def render_export_section(model_id: str):
    """Render export functionality."""
    st.subheader("Export")
    
    if st.button("📥 Export All Memories"):
        export_data = client.export_memories(model_id)
        
        if export_data.get("success", True):  # export returns data directly
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"{model_id}_memories_export.json",
                mime="application/json"
            )
        else:
            st.error(f"Export failed: {export_data.get('error')}")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    # Sidebar
    model_id, category_filter, importance_filter = render_sidebar()
    
    if not model_id:
        st.title("🧠 GAM Memory Dashboard")
        st.info("Select a model from the sidebar to view and manage memories.")
        return
    
    # Main content
    st.title(f"Memories: {model_id}")
    
    # Stats
    render_stats(model_id)
    
    st.divider()
    
    # Pagination settings
    ITEMS_PER_PAGE = 50
    
    # Initialize page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    
    # Calculate offset
    offset = st.session_state.current_page * ITEMS_PER_PAGE
    
    # Get memories with pagination
    response = client.list_memories(
        model_id,
        limit=ITEMS_PER_PAGE,
        offset=offset,
        category=category_filter,
        importance=importance_filter
    )
    
    if not response.get("success", True):
        st.error(f"Failed to load memories: {response.get('error')}")
        return
    
    memories = response.get("memories", [])
    total = response.get("total", 0)
    total_pages = (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE  # Ceiling division
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Browse", "🔧 Bulk Actions", "📥 Export", "🗂️ Buckets"])
    
    with tab1:
        if not memories:
            st.info("No memories found for this model.")
        else:
            # Pagination controls at top
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️ First", disabled=st.session_state.current_page == 0):
                    st.session_state.current_page = 0
                    st.rerun()
            
            with col2:
                if st.button("◀️ Prev", disabled=st.session_state.current_page == 0):
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with col3:
                start_item = offset + 1
                end_item = min(offset + ITEMS_PER_PAGE, total)
                st.markdown(f"**Page {st.session_state.current_page + 1} of {total_pages}** | Showing {start_item}-{end_item} of {total}")
            
            with col4:
                if st.button("Next ▶️", disabled=st.session_state.current_page >= total_pages - 1):
                    st.session_state.current_page += 1
                    st.rerun()
            
            with col5:
                if st.button("Last ⏭️", disabled=st.session_state.current_page >= total_pages - 1):
                    st.session_state.current_page = total_pages - 1
                    st.rerun()
            
            st.divider()
            
            for idx, memory in enumerate(memories):
                render_memory_card(memory, model_id, index=idx + offset)
    
    with tab2:
        search_term = render_bulk_actions(model_id, memories)
        
        # Show filtered memories if searching
        if search_term:
            filtered = [
                m for m in memories 
                if search_term.lower() in m.get("content", "").lower()
            ]
            st.caption(f"Found {len(filtered)} memories matching '{search_term}'")
            
            for memory in filtered:
                memory_id = memory.get("memory_id")
                is_selected = memory_id in st.session_state.get("selected_memories", [])
                
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    if st.checkbox("", is_selected, key=f"select_{memory_id}"):
                        if memory_id not in st.session_state.selected_memories:
                            st.session_state.selected_memories.append(memory_id)
                    else:
                        if memory_id in st.session_state.selected_memories:
                            st.session_state.selected_memories.remove(memory_id)
                
                with col2:
                    st.text(memory.get("content", "")[:200] + "...")
    
    with tab3:
        render_export_section(model_id)
    
    with tab4:
        render_bucket_management()


def render_bucket_management():
    """Render bucket management UI."""
    st.subheader("🗂️ Bucket Management")
    st.caption("Manage memory buckets - rename, merge, or delete model memory stores")
    
    # Refresh buckets
    if st.button("🔄 Refresh Buckets"):
        st.rerun()
    
    # Get bucket list
    response = client.list_buckets()
    
    if not response.get("success", True):
        st.error(f"Failed to load buckets: {response.get('error')}")
        return
    
    buckets = response.get("buckets", [])
    
    if not buckets:
        st.info("No memory buckets found.")
        return
    
    # Display buckets
    st.markdown("### Current Buckets")
    
    for bucket in buckets:
        model_id = bucket.get("model_id", "unknown")
        count = bucket.get("memory_count", 0)
        
        col1, col2, col3 = st.columns([0.5, 0.3, 0.2])
        with col1:
            st.markdown(f"**{model_id}**")
        with col2:
            st.caption(f"{count} memories")
        with col3:
            if st.button("🗑️", key=f"del_bucket_{model_id}", help=f"Delete {model_id}"):
                st.session_state[f"confirm_delete_{model_id}"] = True
        
        # Confirm delete dialog
        if st.session_state.get(f"confirm_delete_{model_id}"):
            st.warning(f"⚠️ Are you sure you want to delete '{model_id}' and all {count} memories?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, Delete", key=f"confirm_yes_{model_id}", type="primary"):
                    result = client.delete_bucket(model_id)
                    if result.get("success"):
                        st.success(f"✓ Deleted '{model_id}'")
                        st.session_state[f"confirm_delete_{model_id}"] = False
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Failed: {result.get('error')}")
            with col_no:
                if st.button("Cancel", key=f"confirm_no_{model_id}"):
                    st.session_state[f"confirm_delete_{model_id}"] = False
                    st.rerun()
    
    st.divider()
    
    # Migration section
    st.markdown("### Migrate / Rename Bucket")
    st.caption("Move memories from one bucket to another (e.g., fix wrong model ID after import)")
    
    bucket_ids = [b.get("model_id") for b in buckets]
    
    col1, col2 = st.columns(2)
    with col1:
        from_bucket = st.selectbox(
            "From (source)",
            bucket_ids,
            key="migrate_from",
            help="The bucket to migrate FROM"
        )
    with col2:
        to_bucket = st.text_input(
            "To (target)",
            placeholder="e.g., wizardlm-of2",
            key="migrate_to",
            help="The bucket to migrate TO (can be existing or new)"
        )
    
    merge_option = st.checkbox(
        "Merge if target exists",
        value=True,
        help="If checked, memories will be combined. If unchecked, migration fails if target exists."
    )
    
    if st.button("🔀 Migrate Bucket", disabled=not to_bucket):
        if from_bucket == to_bucket:
            st.error("Source and target cannot be the same!")
        elif not to_bucket:
            st.error("Please enter a target bucket ID")
        else:
            with st.spinner(f"Migrating '{from_bucket}' → '{to_bucket}'..."):
                result = client.migrate_bucket(from_bucket, to_bucket, merge=merge_option)
            
            if result.get("success"):
                action = result.get("action", "migrated")
                count = result.get("memories_migrated", 0)
                st.success(f"✓ Successfully {action}! Moved {count} memories.")
                if result.get("total_memories"):
                    st.info(f"Target now has {result['total_memories']} total memories")
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Migration failed: {result.get('error')}")


if __name__ == "__main__":
    # Check if streamlit is being run directly or as a module
    import sys
    
    if "streamlit" in sys.modules:
        main()
    else:
        # Run streamlit
        import subprocess
        subprocess.run(["streamlit", "run", __file__])
