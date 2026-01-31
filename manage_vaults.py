"""
Vault Management Utility for GAM-Memvid.

List, delete, and manage .mv2 vault files.

Usage:
    python manage_vaults.py list                          # List all vaults
    python manage_vaults.py delete <model_id>             # Delete a vault (local)
    python manage_vaults.py delete <model_id> --remote    # Delete via API
    python manage_vaults.py info <model_id>               # Show vault info
"""
import argparse
import sys
from pathlib import Path

import httpx

from config import config


# Default Railway server URL
DEFAULT_SERVER_URL = "https://gam-memvid-fusion-production.up.railway.app"


def list_vaults(vaults_dir: Path = None, remote_url: str = None) -> list:
    """List all vault files."""
    if remote_url:
        # List from remote server
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(f"{remote_url}/memvid/vaults")
                r.raise_for_status()
                data = r.json()
                vaults = data.get("vaults", [])
                print(f"Remote vaults at {remote_url}:")
                for v in vaults:
                    model_id = v.replace(".mv2", "")
                    print(f"  - {model_id}")
                return vaults
        except Exception as e:
            print(f"Error listing remote vaults: {e}")
            return []
    else:
        # List local vaults
        if vaults_dir is None:
            vaults_dir = Path(config.VAULTS_DIR)

        if not vaults_dir.exists():
            print(f"Vaults directory does not exist: {vaults_dir}")
            return []

        vault_files = list(vaults_dir.glob("*.mv2"))
        print(f"Local vaults in {vaults_dir}:")
        for v in vault_files:
            size_kb = v.stat().st_size / 1024
            print(f"  - {v.stem} ({size_kb:.1f} KB)")
        return [v.name for v in vault_files]


def delete_vault_local(model_id: str, vaults_dir: Path = None, force: bool = False) -> bool:
    """Delete a vault file locally."""
    if vaults_dir is None:
        vaults_dir = Path(config.VAULTS_DIR)

    vault_file = vaults_dir / f"{model_id}.mv2"

    if not vault_file.exists():
        print(f"Vault not found: {vault_file}")
        return False

    size_kb = vault_file.stat().st_size / 1024
    print(f"Found vault: {vault_file.name} ({size_kb:.1f} KB)")

    if not force:
        confirm = input(f"Are you sure you want to DELETE {model_id}.mv2? (yes/no): ")
        if confirm.lower() not in ("yes", "y"):
            print("Cancelled.")
            return False

    try:
        vault_file.unlink()
        print(f"Deleted: {vault_file}")
        return True
    except Exception as e:
        print(f"Error deleting vault: {e}")
        return False


def delete_vault_remote(model_id: str, server_url: str, force: bool = False) -> bool:
    """Delete a vault via the API."""
    print(f"Deleting vault '{model_id}' via API: {server_url}")

    if not force:
        confirm = input(f"Are you sure you want to DELETE {model_id} from the server? (yes/no): ")
        if confirm.lower() not in ("yes", "y"):
            print("Cancelled.")
            return False

    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.delete(
                f"{server_url}/memory/{model_id}/vault",
                params={"confirm": "true"}
            )
            r.raise_for_status()
            data = r.json()

            if data.get("success"):
                print(f"SUCCESS: {data.get('message')}")
                print(f"  Deleted file: {data.get('deleted_file')}")
                return True
            else:
                print(f"FAILED: {data.get('message')}")
                return False

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def vault_info(model_id: str, vaults_dir: Path = None, remote_url: str = None) -> dict:
    """Get information about a vault."""
    if remote_url:
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(f"{remote_url}/memory/{model_id}/stats")
                r.raise_for_status()
                stats = r.json()
                print(f"Vault stats for '{model_id}' (remote):")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                return stats
        except Exception as e:
            print(f"Error getting vault info: {e}")
            return {}
    else:
        if vaults_dir is None:
            vaults_dir = Path(config.VAULTS_DIR)

        vault_file = vaults_dir / f"{model_id}.mv2"
        if not vault_file.exists():
            print(f"Vault not found: {vault_file}")
            return {}

        stats = {
            "model_id": model_id,
            "file": str(vault_file),
            "size_bytes": vault_file.stat().st_size,
            "size_kb": vault_file.stat().st_size / 1024,
            "modified": vault_file.stat().st_mtime
        }
        print(f"Vault info for '{model_id}' (local):")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Manage GAM-Memvid vault files"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List all vaults")
    list_parser.add_argument("--remote", "-r", action="store_true", help="List from remote server")
    list_parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Server URL for remote operations")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a vault")
    delete_parser.add_argument("model_id", help="Model ID to delete")
    delete_parser.add_argument("--remote", "-r", action="store_true", help="Delete from remote server")
    delete_parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Server URL for remote operations")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show vault info")
    info_parser.add_argument("model_id", help="Model ID to inspect")
    info_parser.add_argument("--remote", "-r", action="store_true", help="Get info from remote server")
    info_parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Server URL for remote operations")

    args = parser.parse_args()

    if args.command == "list":
        if args.remote:
            list_vaults(remote_url=args.server_url)
        else:
            list_vaults()

    elif args.command == "delete":
        if args.remote:
            success = delete_vault_remote(args.model_id, args.server_url, args.force)
        else:
            success = delete_vault_local(args.model_id, force=args.force)
        sys.exit(0 if success else 1)

    elif args.command == "info":
        if args.remote:
            vault_info(args.model_id, remote_url=args.server_url)
        else:
            vault_info(args.model_id)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
