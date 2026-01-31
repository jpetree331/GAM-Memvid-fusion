"""
OpenSearch Export Utility

Exports GAM memories to OpenSearch-compatible formats for future migration.
This does NOT require OpenSearch to be installed - it just prepares the data.

Supported formats:
1. Bulk API format (NDJSON) - for opensearch bulk import
2. Single index format - for manual import or testing
"""
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import config
from memory_organization import get_organizer, MemoryOrganizer


def memory_to_opensearch_doc(memory: dict, model_id: str) -> dict:
    """
    Convert a GAM memory to an OpenSearch document.
    
    OpenSearch document structure optimized for:
    - Full-text search on content
    - Filtering by category, importance, tags
    - Date range queries
    - Vector search (embedding field placeholder)
    """
    return {
        # Core fields
        "memory_id": memory.get("memory_id", memory.get("id")),
        "model_id": model_id,
        "content": memory.get("content", ""),
        
        # Organization fields (keyword type for exact matching)
        "category": memory.get("category", "context"),
        "importance": memory.get("importance", "normal"),
        "tags": memory.get("tags", []),
        
        # AI Self fields
        "is_ai_self": memory.get("category") == "ai_self",
        "ai_self_type": memory.get("ai_self_type"),
        
        # Metadata
        "source": memory.get("source", "conversation"),
        "user_id": memory.get("user_id"),
        
        # Timestamps (ISO 8601 for OpenSearch date type)
        "created_at": memory.get("created_at"),
        "updated_at": memory.get("updated_at"),
        
        # Supersedes chain (for opinion evolution tracking)
        "supersedes": memory.get("supersedes"),
        
        # Placeholder for vector embedding (to be computed during import)
        # "embedding": None  # Will be populated by OpenSearch ML plugin or during import
    }


def export_bulk_format(
    model_id: str,
    index_name: str = "gam-memories",
    output_path: Optional[str] = None
) -> str:
    """
    Export memories in OpenSearch Bulk API format (NDJSON).
    
    Each memory becomes two lines:
    1. Action metadata: {"index": {"_index": "...", "_id": "..."}}
    2. Document: {the actual memory data}
    
    Usage with OpenSearch:
        curl -XPOST "localhost:9200/_bulk" -H "Content-Type: application/x-ndjson" --data-binary @export.ndjson
    """
    organizer = get_organizer(model_id)
    memories = list(organizer._memories.values())
    
    lines = []
    for mem in memories:
        mem_dict = mem.to_dict()
        doc = memory_to_opensearch_doc(mem_dict, model_id)
        
        # Action line
        action = {
            "index": {
                "_index": index_name,
                "_id": doc["memory_id"]
            }
        }
        lines.append(json.dumps(action))
        
        # Document line
        lines.append(json.dumps(doc))
    
    content = "\n".join(lines)
    if lines:
        content += "\n"  # Bulk API requires trailing newline
    
    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")
        print(f"Exported {len(memories)} memories to {output_path}")
    
    return content


def export_single_index_format(
    model_id: str,
    output_path: Optional[str] = None
) -> dict:
    """
    Export memories as a single JSON document with index mapping.
    
    Includes:
    - OpenSearch index mapping (schema)
    - All documents
    - Import instructions
    """
    organizer = get_organizer(model_id)
    memories = list(organizer._memories.values())
    
    export_data = {
        "export_info": {
            "source": "GAM Memory System",
            "model_id": model_id,
            "exported_at": datetime.now().isoformat(),
            "memory_count": len(memories),
            "format_version": "1.0"
        },
        
        # OpenSearch index mapping
        "index_mapping": {
            "mappings": {
                "properties": {
                    "memory_id": {"type": "keyword"},
                    "model_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "category": {"type": "keyword"},
                    "importance": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "is_ai_self": {"type": "boolean"},
                    "ai_self_type": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    "supersedes": {"type": "keyword"},
                    # Vector field for semantic search (k-NN)
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 384,  # all-MiniLM-L6-v2 dimension
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    }
                }
            },
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
        },
        
        # The actual documents
        "documents": [
            memory_to_opensearch_doc(mem.to_dict(), model_id)
            for mem in memories
        ],
        
        # Import instructions
        "import_instructions": {
            "step_1": "Create the index with mapping: PUT /gam-memories with index_mapping as body",
            "step_2": "For bulk import, use export_bulk_format() output with POST /_bulk",
            "step_3": "For embeddings, either: (a) use OpenSearch ML plugin, or (b) pre-compute and add to documents",
            "step_4": "Verify with: GET /gam-memories/_count"
        }
    }
    
    if output_path:
        Path(output_path).write_text(
            json.dumps(export_data, indent=2, default=str),
            encoding="utf-8"
        )
        print(f"Exported {len(memories)} memories to {output_path}")
    
    return export_data


def export_all_models(
    output_dir: str = "opensearch_export",
    format: str = "both"
) -> dict:
    """
    Export all models' memories to OpenSearch format.
    
    Args:
        output_dir: Directory to save exports
        format: "bulk", "single", or "both"
    
    Returns:
        Summary of exported data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    models_dir = config.DATA_DIR / "models"
    if not models_dir.exists():
        return {"error": "No models directory found"}
    
    model_ids = [d.name for d in models_dir.iterdir() if d.is_dir()]
    
    summary = {
        "exported_at": datetime.now().isoformat(),
        "output_dir": str(output_path),
        "models": {}
    }
    
    for model_id in model_ids:
        try:
            organizer = get_organizer(model_id)
            count = len(organizer._memories)
            
            if count == 0:
                summary["models"][model_id] = {"count": 0, "skipped": True}
                continue
            
            # Safe filename
            safe_name = model_id.replace(" ", "_").replace("/", "_")
            
            if format in ("bulk", "both"):
                bulk_path = output_path / f"{safe_name}_bulk.ndjson"
                export_bulk_format(model_id, output_path=str(bulk_path))
            
            if format in ("single", "both"):
                single_path = output_path / f"{safe_name}_full.json"
                export_single_index_format(model_id, output_path=str(single_path))
            
            summary["models"][model_id] = {"count": count, "success": True}
            
        except Exception as e:
            summary["models"][model_id] = {"error": str(e)}
    
    # Save summary
    summary_path = output_path / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return summary


# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export GAM memories to OpenSearch-compatible format"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model ID to export (omit for all models)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["bulk", "single", "both"],
        default="both",
        help="Export format (default: both)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="opensearch_export",
        help="Output directory (default: opensearch_export)"
    )
    parser.add_argument(
        "--index-name", "-i",
        type=str,
        default="gam-memories",
        help="OpenSearch index name (default: gam-memories)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  OpenSearch Export Utility")
    print(f"{'='*60}\n")
    
    if args.model:
        # Export single model
        safe_name = args.model.replace(" ", "_").replace("/", "_")
        
        if args.format in ("bulk", "both"):
            bulk_path = f"{args.output}/{safe_name}_bulk.ndjson"
            Path(args.output).mkdir(parents=True, exist_ok=True)
            export_bulk_format(args.model, args.index_name, bulk_path)
        
        if args.format in ("single", "both"):
            single_path = f"{args.output}/{safe_name}_full.json"
            Path(args.output).mkdir(parents=True, exist_ok=True)
            export_single_index_format(args.model, single_path)
        
        print(f"\n✓ Export complete for model: {args.model}")
    else:
        # Export all models
        summary = export_all_models(args.output, args.format)
        
        print(f"\n{'='*60}")
        print(f"  Export Summary")
        print(f"{'='*60}")
        for model_id, info in summary.get("models", {}).items():
            status = "✓" if info.get("success") else "⚠" if info.get("skipped") else "✗"
            count = info.get("count", 0)
            print(f"  {status} {model_id}: {count} memories")
        
        print(f"\nFiles saved to: {args.output}/")
