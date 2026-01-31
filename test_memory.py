"""
Simple test script to verify GAM integration works.

Run this after setting up your .env file:
    python test_memory.py
"""
import asyncio
from datetime import datetime, timedelta

# Check if GAM is installed
try:
    import gam
    print("✓ GAM is installed")
except ImportError:
    print("✗ GAM is not installed. Run:")
    print("  pip install git+https://github.com/VectorSpaceLab/general-agentic-memory.git")
    exit(1)

from config import config

# Validate config
errors = config.validate()
if errors:
    print("\n✗ Configuration errors:")
    for error in errors:
        print(f"  - {error}")
    print("\nCreate a .env file from .env.example and configure it.")
    exit(1)

print("✓ Configuration is valid")

from memory_manager import memory_manager


def test_memory_operations():
    """Test basic memory operations."""
    print("\n--- Testing Memory Operations ---\n")
    
    # Test model: simulate an OpenWebUI custom model
    model_id = "test-model-gpt4"
    
    print(f"1. Getting memory store for model: {model_id}")
    store = memory_manager.get_store(model_id)
    print(f"   ✓ Store created, data dir: {store.data_dir}")
    
    print("\n2. Adding a memory...")
    memory_id = store.add_memory(
        content="The user's name is Alice and she prefers dark mode.",
        user_id="test-user",
        timestamp=datetime.now() - timedelta(days=7)  # Historical timestamp
    )
    print(f"   ✓ Memory added: {memory_id}")
    
    print("\n3. Adding another memory...")
    memory_id_2 = store.add_memory(
        content="Alice is working on a Python project about machine learning.",
        user_id="test-user"
    )
    print(f"   ✓ Memory added: {memory_id_2}")
    
    print("\n4. Searching memories...")
    results = store.search("What is Alice working on?", limit=5)
    print(f"   ✓ Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"   Result {i+1}: {result.content[:100]}...")
    
    print("\n5. Getting context for prompt...")
    context = store.get_context_for_prompt("Tell me about the user")
    if context:
        print(f"   ✓ Context retrieved ({len(context)} chars)")
        print(f"   Preview: {context[:200]}...")
    else:
        print("   - No context (memories may not have been indexed yet)")
    
    print("\n6. Testing second model (isolated bucket)...")
    model_id_2 = "test-model-claude"
    store_2 = memory_manager.get_store(model_id_2)
    store_2.add_memory(content="This is a different model's memory")
    print(f"   ✓ Second store created: {store_2.data_dir}")
    
    print("\n7. Listing all models...")
    models = memory_manager.list_models()
    print(f"   Active models: {models}")
    all_models = memory_manager.get_all_model_dirs()
    print(f"   All model dirs: {all_models}")
    
    print("\n--- All Tests Passed ---")


def test_server_import():
    """Test that server can be imported."""
    print("\n--- Testing Server Import ---\n")
    
    try:
        from server import app
        print("✓ Server module imports successfully")
        print(f"  App title: {app.title}")
    except Exception as e:
        print(f"✗ Server import failed: {e}")
        return False
    
    return True


def test_import_utility():
    """Test that import utility can be imported."""
    print("\n--- Testing Import Utility ---\n")
    
    try:
        from import_conversations import (
            parse_openwebui_export,
            parse_chatgpt_export,
            parse_mem0_export,
            import_conversations
        )
        print("✓ Import utility imports successfully")
    except Exception as e:
        print(f"✗ Import utility failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("GAM + OpenWebUI Integration Test")
    print("=" * 50)
    
    # Test imports first
    if not test_server_import():
        exit(1)
    
    if not test_import_utility():
        exit(1)
    
    # Test memory operations
    try:
        test_memory_operations()
    except Exception as e:
        print(f"\n✗ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run the server: python server.py")
    print("2. Test the API: curl http://localhost:8100/health")
    print("3. Configure OpenWebUI with the function in openwebui_function.py")
