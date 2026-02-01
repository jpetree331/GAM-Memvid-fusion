"""
Configuration management for GAM-Memvid memory system.

This configuration supports:
- Memvid vault storage (The Vault)
- Memory Condenser LLM (The Brain)
- OpenWebUI integration
- Memory style guidelines
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Memory Style Guide - Prevents pronoun confusion
# =============================================================================

DEFAULT_MEMORY_STYLE_GUIDE = """
## Memory Storage Style Guide

**Perspective**: Store memories from YOUR perspective as the AI (Observer/Companion).

**Pronoun Rules (STRICT)**:
- Use the user's name or "User" to refer to the human. NEVER use "I" to refer to the user.
- Use "I" ONLY to refer to yourself (the AI).
- Use "We" for shared context or mutual experiences.

**Examples**:
- CORRECT: "User mentioned they prefer dark mode"
- CORRECT: "Jess is working on a machine learning project"
- WRONG: "I am working on a machine learning project" (when referring to user)
- CORRECT: "I noticed User seems interested in theology"
- CORRECT: "We discussed consciousness and emergence"

**Format**: Store as concise facts, not verbose narratives.
""".strip()


class Config:
    """
    Central configuration for the GAM-Memvid system.

    Environment variables can override all settings.
    """

    # =========================================================================
    # API Keys (for Condenser and fallbacks)
    # =========================================================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Gemini API Key (for Memory Condenser - recommended)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # =========================================================================
    # Memory Condenser Configuration (The Brain)
    # =========================================================================
    # Provider: "openai" (default) or "gemini"
    CONDENSER_PROVIDER: str = os.getenv("CONDENSER_PROVIDER", "openai")
    CONDENSER_MODEL: str = os.getenv("CONDENSER_MODEL", "gpt-4o-mini")

    # OpenAI model for condenser (used when provider=openai)
    OPENAI_CONDENSER_MODEL: str = os.getenv("OPENAI_CONDENSER_MODEL", "gpt-4o-mini")

    # =========================================================================
    # Memvid Vault Configuration (The Vault)
    # =========================================================================
    # Embedding model for semantic search (fastembed local embeddings)
    # Default: BAAI/bge-small-en-v1.5 (fast, high quality, runs locally)
    # Other options: BAAI/bge-base-en-v1.5, sentence-transformers/all-MiniLM-L6-v2
    MEMVID_EMBEDDING_MODEL: str = os.getenv("MEMVID_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Search settings
    MEMVID_DEFAULT_SEARCH_LIMIT: int = int(os.getenv("MEMVID_DEFAULT_SEARCH_LIMIT", "10"))
    MEMVID_CORE_MEMORY_LIMIT: int = int(os.getenv("MEMVID_CORE_MEMORY_LIMIT", "100"))

    # =========================================================================
    # Server Configuration
    # =========================================================================
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8100"))

    # =========================================================================
    # Data Storage Paths
    # =========================================================================
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))

    # Vaults directory (stores {model_id}.mv2 files)
    # Default: ./data/vaults
    VAULTS_DIR: Path = Path(os.getenv("VAULTS_DIR", "")) or DATA_DIR / "vaults"

    # Legacy: models directory (for migration compatibility)
    MODELS_DIR: Path = DATA_DIR / "models"

    # Continuum app persistence (schedules, settings) - under DATA_DIR for volume mount
    CONTINUUM_DATA_DIR: Path = DATA_DIR / "continuum"

    # =========================================================================
    # Continuum Bridge (OpenWebUI for journal/scheduler)
    # =========================================================================
    CONTINUUM_OPENWEBUI_BASE_URL: str = os.getenv("CONTINUUM_OPENWEBUI_BASE_URL", "").rstrip("/")
    CONTINUUM_OPENWEBUI_API_KEY: str = os.getenv("CONTINUUM_OPENWEBUI_API_KEY", "")
    # Optional: require Authorization: Bearer <key> or X-API-Key on /continuum/* requests
    CONTINUUM_BRIDGE_API_KEY: str = os.getenv("CONTINUUM_BRIDGE_API_KEY", "")

    # =========================================================================
    # Memory Style Guide
    # =========================================================================
    MEMORY_STYLE_GUIDE: str = os.getenv("MEMORY_STYLE_GUIDE", DEFAULT_MEMORY_STYLE_GUIDE)

    # User name for pronoun clarity (can be customized per model)
    DEFAULT_USER_NAME: str = os.getenv("DEFAULT_USER_NAME", "User")

    # =========================================================================
    # Path Helper Methods
    # =========================================================================

    @classmethod
    def get_vault_path(cls, model_id: str) -> Path:
        """
        Get the vault file path for a specific model.

        Args:
            model_id: The AI model/persona identifier (e.g., "eli", "opus")

        Returns:
            Path to the model's .mv2 vault file
        """
        safe_model_id = model_id.replace("/", "_").replace("\\", "_")
        cls.VAULTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.VAULTS_DIR / f"{safe_model_id}.mv2"

    @classmethod
    def get_model_data_dir(cls, model_id: str) -> Path:
        """
        Get the data directory for a specific model.

        This is kept for backward compatibility with migration scripts.
        New code should use get_vault_path() instead.

        Args:
            model_id: The AI model/persona identifier

        Returns:
            Path to the model's data directory
        """
        safe_model_id = model_id.replace("/", "_").replace("\\", "_")
        model_dir = cls.MODELS_DIR / safe_model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    @classmethod
    def get_vaults_dir(cls) -> Path:
        """Get the vaults directory, creating it if needed."""
        cls.VAULTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.VAULTS_DIR

    @classmethod
    def get_continuum_data_dir(cls) -> Path:
        """Get the Continuum persistence directory (schedules, settings). Create if needed."""
        cls.CONTINUUM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return cls.CONTINUUM_DATA_DIR

    # =========================================================================
    # Validation
    # =========================================================================

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate configuration and return list of errors/warnings.

        Returns:
            List of error messages (empty if config is valid)
        """
        errors = []

        # Check Condenser configuration
        if cls.CONDENSER_PROVIDER == "gemini":
            if not cls.GEMINI_API_KEY:
                errors.append(
                    "GEMINI_API_KEY is required when CONDENSER_PROVIDER=gemini "
                    "(or set CONDENSER_PROVIDER=openai)"
                )
        elif cls.CONDENSER_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                errors.append(
                    "OPENAI_API_KEY is required when CONDENSER_PROVIDER=openai"
                )

        # Check if at least one API key is available for fallback
        if not cls.GEMINI_API_KEY and not cls.OPENAI_API_KEY:
            errors.append(
                "At least one of GEMINI_API_KEY or OPENAI_API_KEY is required "
                "for the Memory Condenser to function"
            )

        # Validate embedding model choice
        # Accept fastembed model names (BAAI/*, sentence-transformers/*) or legacy short names
        valid_prefixes = ["BAAI/", "sentence-transformers/", "jinaai/"]
        legacy_models = ["bge-small", "openai", "gemini", "local"]
        is_valid = (
            cls.MEMVID_EMBEDDING_MODEL in legacy_models or
            any(cls.MEMVID_EMBEDDING_MODEL.startswith(p) for p in valid_prefixes)
        )
        if not is_valid:
            errors.append(
                f"MEMVID_EMBEDDING_MODEL must be a fastembed model (e.g., BAAI/bge-small-en-v1.5) "
                f"or one of: {legacy_models}"
            )

        # Check data directory is writable
        try:
            cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
            test_file = cls.DATA_DIR / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            errors.append(f"DATA_DIR is not writable: {e}")

        return errors

    @classmethod
    def print_config(cls):
        """Print current configuration (for debugging)."""
        print("=" * 60)
        print("GAM-Memvid Configuration")
        print("=" * 60)
        print(f"CONDENSER_PROVIDER: {cls.CONDENSER_PROVIDER}")
        print(f"CONDENSER_MODEL: {cls.CONDENSER_MODEL}")
        print(f"GEMINI_API_KEY: {'[SET]' if cls.GEMINI_API_KEY else '[NOT SET]'}")
        print(f"OPENAI_API_KEY: {'[SET]' if cls.OPENAI_API_KEY else '[NOT SET]'}")
        print(f"MEMVID_EMBEDDING_MODEL: {cls.MEMVID_EMBEDDING_MODEL}")
        print(f"DATA_DIR: {cls.DATA_DIR}")
        print(f"VAULTS_DIR: {cls.VAULTS_DIR}")
        print(f"HOST: {cls.HOST}")
        print(f"PORT: {cls.PORT}")
        print("=" * 60)


# Global config instance
config = Config()
