"""
Test script to send a memory with a specific timestamp to verify timestamp preservation.

Usage:
    python test_send_memory.py
    python test_send_memory.py --server-url https://your-server.railway.app
    python test_send_memory.py --model-id test-model
"""
import argparse
import httpx
from datetime import datetime

# Default Railway server URL
DEFAULT_SERVER_URL = "https://gam-memvid-fusion-production.up.railway.app"


def send_test_memory(
    server_url: str,
    model_id: str = "timestamp-test",
    user_name: str = "Jess"
) -> dict:
    """
    Send a test memory with a specific old timestamp (Oct 31, 2025 7:35 PM EST).

    This tests whether the server correctly preserves the original created_at
    timestamp instead of overwriting it with the import time.
    """
    # Oct 31, 2025 7:35 PM EST (UTC-5)
    # ISO 8601 format with timezone
    test_timestamp = "2025-10-31T19:35:00-05:00"

    payload = {
        "model_id": model_id,
        "user_message": "This is a test message from Halloween 2025! I'm testing if timestamps are preserved correctly when importing memories.",
        "ai_response": "Hello! I see you're testing the timestamp preservation feature. This memory should show as created on October 31st, 2025 at 7:35 PM EST - Halloween evening! If you see this timestamp in the dashboard instead of today's date, the feature is working correctly.",
        "user_name": user_name,
        "tags": ["test", "timestamp-test", "halloween"],
        "created_at": test_timestamp
    }

    print(f"Sending test memory to: {server_url}/memory/add")
    print(f"Model ID: {model_id}")
    print(f"Timestamp being sent: {test_timestamp}")
    print(f"User message: {payload['user_message'][:80]}...")
    print()

    # Print raw JSON payload for debugging
    import json
    print("Raw JSON payload:")
    print(json.dumps(payload, indent=2))
    print()

    try:
        with httpx.Client(timeout=30.0) as client:
            # First, call echo endpoint to verify what server receives
            print("=" * 60)
            print("STEP 1: Calling /debug/echo to verify server receives created_at")
            print("=" * 60)
            try:
                echo_response = client.post(
                    f"{server_url}/debug/echo",
                    json=payload
                )
                if echo_response.status_code == 200:
                    echo_data = echo_response.json()
                    print("Echo response:")
                    print(json.dumps(echo_data, indent=2))
                    received_created_at = echo_data.get("received", {}).get("created_at")
                    print(f"\nServer received created_at: {received_created_at!r}")
                    if received_created_at == test_timestamp:
                        print("PASS: Server received correct timestamp!")
                    else:
                        print(f"FAIL: Expected {test_timestamp!r}, got {received_created_at!r}")
                else:
                    print(f"Echo endpoint returned: {echo_response.status_code}")
            except Exception as e:
                print(f"Echo endpoint not available: {e}")
            print()

            # Now send the actual memory
            print("=" * 60)
            print("STEP 2: Sending memory via /memory/add")
            print("=" * 60)
            response = client.post(
                f"{server_url}/memory/add",
                json=payload
            )
            print(f"Response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()

            print("SUCCESS!")
            print(f"  Pearl ID: {data.get('pearl_id')}")
            print(f"  Status: {data.get('status')}")
            print(f"  Word count: {data.get('word_count')}")
            print()
            print("Now check the dashboard to verify the timestamp shows as:")
            print("  October 31, 2025 7:35 PM (or similar)")
            print("  NOT today's date/time")

            return data

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return {"error": str(e)}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


def verify_timestamp(server_url: str, model_id: str = "timestamp-test") -> None:
    """
    Verify that the timestamp was preserved by fetching recent memories.
    """
    print(f"\nVerifying timestamp preservation...")
    print(f"Fetching recent memories from: {server_url}/memory/{model_id}/recent")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{server_url}/memory/{model_id}/recent",
                params={"limit": 5}
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("memories") or data.get("items") or []
            print(f"Found {len(items)} memories")

            for i, item in enumerate(items):
                created_at = item.get("created_at", "NOT FOUND")
                pearl_id = item.get("id", "?")
                user_msg = item.get("user_message", "")[:50] or item.get("text", "")[:50]

                print(f"\n  [{i+1}] Pearl: {pearl_id}")
                print(f"      created_at: {created_at}")
                print(f"      content: {user_msg}...")

                # Check if it's the Halloween timestamp
                if "2025-10-31" in str(created_at):
                    print("      ^ TIMESTAMP PRESERVED CORRECTLY!")

    except Exception as e:
        print(f"Error verifying: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test timestamp preservation by sending a memory with an old date"
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"Server URL (default: {DEFAULT_SERVER_URL})"
    )
    parser.add_argument(
        "--model-id",
        default="timestamp-test",
        help="Model ID for the test vault (default: timestamp-test)"
    )
    parser.add_argument(
        "--user-name",
        default="Jess",
        help="User name (default: Jess)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing memories, don't send new one"
    )

    args = parser.parse_args()

    if not args.verify_only:
        send_test_memory(
            server_url=args.server_url,
            model_id=args.model_id,
            user_name=args.user_name
        )

    verify_timestamp(args.server_url, args.model_id)


if __name__ == "__main__":
    main()
