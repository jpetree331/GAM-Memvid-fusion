"""
Test Script - Verify GAM setup before connecting to OpenWebUI.

Run this to check everything is working:
    python test_setup.py
"""
import sys
import json
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def ok(msg): print(f"{Colors.GREEN}✓{Colors.END} {msg}")
def fail(msg): print(f"{Colors.RED}✗{Colors.END} {msg}")
def warn(msg): print(f"{Colors.YELLOW}!{Colors.END} {msg}")
def info(msg): print(f"{Colors.BLUE}→{Colors.END} {msg}")
def header(msg): print(f"\n{Colors.BOLD}{msg}{Colors.END}")


def check_python_version():
    """Check Python version."""
    header("1. Python Version")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        ok(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        fail(f"Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_dependencies():
    """Check required packages are installed."""
    header("2. Dependencies")
    
    required = {
        "fastapi": "API server",
        "uvicorn": "ASGI server",
        "httpx": "HTTP client",
        "pydantic": "Data validation",
        "dotenv": "Environment variables (python-dotenv)",
    }
    
    optional = {
        "streamlit": "Dashboard (optional)",
    }
    
    all_ok = True
    
    for package, desc in required.items():
        try:
            if package == "dotenv":
                __import__("dotenv")
            else:
                __import__(package)
            ok(f"{package} - {desc}")
        except ImportError:
            fail(f"{package} - {desc} (pip install {package})")
            all_ok = False
    
    for package, desc in optional.items():
        try:
            __import__(package)
            ok(f"{package} - {desc}")
        except ImportError:
            warn(f"{package} - {desc} (not installed)")
    
    return all_ok


def check_gam():
    """Check if GAM is installed."""
    header("3. GAM (General Agentic Memory)")
    
    try:
        from gam import MemoryAgent, ResearchAgent
        ok("GAM is installed")
        return True
    except ImportError:
        fail("GAM not installed")
        info("Install with: pip install git+https://github.com/VectorSpaceLab/general-agentic-memory.git")
        return False


def check_env():
    """Check environment configuration."""
    header("4. Environment Configuration")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            fail(".env file missing")
            info("Copy .env.example to .env and configure it")
            return False
        else:
            fail("Neither .env nor .env.example found")
            return False
    
    ok(".env file exists")
    
    # Check for required variables
    from dotenv import dotenv_values
    env_vars = dotenv_values(".env")
    
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["GAM_MODEL_NAME", "GAM_EMBEDDING_MODEL", "HOST", "PORT"]
    
    all_ok = True
    for var in required_vars:
        value = env_vars.get(var, "")
        if value and not value.startswith("your-"):
            ok(f"{var} is set")
        else:
            fail(f"{var} not configured")
            all_ok = False
    
    for var in optional_vars:
        value = env_vars.get(var, "")
        if value:
            ok(f"{var} = {value}")
        else:
            warn(f"{var} not set (using default)")
    
    return all_ok


def check_config():
    """Check config module loads correctly."""
    header("5. Configuration Module")
    
    try:
        from config import config
        ok(f"Config loaded")
        ok(f"Data directory: {config.DATA_DIR}")
        ok(f"Server: {config.HOST}:{config.PORT}")
        
        errors = config.validate()
        if errors:
            for error in errors:
                fail(error)
            return False
        
        return True
    except Exception as e:
        fail(f"Config error: {e}")
        return False


def test_server_imports():
    """Test that server module can be imported."""
    header("6. Server Module")
    
    try:
        from server import app
        ok("Server module imports successfully")
        ok(f"FastAPI app: {app.title}")
        return True
    except Exception as e:
        fail(f"Server import error: {e}")
        return False


def test_memory_operations():
    """Test basic memory operations (without starting server)."""
    header("7. Memory Operations (Local Test)")
    
    try:
        from memory_organization import get_organizer, MemoryCategory, ImportanceLevel
        
        # Create a test organizer
        test_model = "_test_model_"
        organizer = get_organizer(test_model)
        
        # Add a test memory
        test_id = f"test_{__import__('time').time()}"
        memory = organizer.add(
            memory_id=test_id,
            content="This is a test memory",
            category=MemoryCategory.CONTEXT.value,
            importance=ImportanceLevel.NORMAL.value,
            tags=["test"]
        )
        ok(f"Memory created: {test_id}")
        
        # Retrieve it
        retrieved = organizer.get(test_id)
        if retrieved and retrieved.content == "This is a test memory":
            ok("Memory retrieved successfully")
        else:
            fail("Memory retrieval failed")
            return False
        
        # Delete test memory
        del organizer._memories[test_id]
        organizer._save_index()
        ok("Test memory cleaned up")
        
        return True
        
    except Exception as e:
        fail(f"Memory operation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_endpoint():
    """Test server is running (if started)."""
    header("8. Server Connection Test")
    
    try:
        import httpx
        from config import config
        
        url = f"http://{config.HOST}:{config.PORT}/health"
        info(f"Testing: {url}")
        
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                ok(f"Server is running! Status: {data.get('status')}")
                ok(f"Active models: {data.get('models_active', 0)}")
                return True
            else:
                warn(f"Server returned status {response.status_code}")
                return False
        except httpx.ConnectError:
            warn("Server not running (start with: python server.py)")
            return None  # Not a failure, just not running
            
    except Exception as e:
        fail(f"Connection test error: {e}")
        return False


def main():
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  GAM + OpenWebUI Integration Test{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")
    
    results = {}
    
    results["python"] = check_python_version()
    results["deps"] = check_dependencies()
    results["gam"] = check_gam()
    results["env"] = check_env()
    results["config"] = check_config()
    results["server_import"] = test_server_imports()
    results["memory_ops"] = test_memory_operations()
    results["server_running"] = test_server_endpoint()
    
    # Summary
    header("Summary")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"\n  Passed: {Colors.GREEN}{passed}{Colors.END}")
    print(f"  Failed: {Colors.RED}{failed}{Colors.END}")
    print(f"  Skipped: {Colors.YELLOW}{skipped}{Colors.END}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All checks passed!{Colors.END}")
        print("\nNext steps:")
        print("  1. Start the server:  python server.py")
        print("  2. Start dashboard:   streamlit run dashboard.py")
        print("  3. Copy openwebui_function.py code into OpenWebUI Functions")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some checks failed.{Colors.END}")
        print("Fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
