"""
Memory Style System - Ensures consistent pronoun usage and perspective.

Prevents the common issue where the AI confuses itself with the user
when storing memories (e.g., "I did X" when it should be "User did X").
"""
import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from config import config


@dataclass
class ModelStyleConfig:
    """Per-model style configuration."""
    user_name: str = "User"  # What to call the human
    ai_name: str = "I"       # How AI refers to itself (usually "I")
    perspective: str = "ai"  # "ai" = AI's perspective, "neutral" = third person
    custom_rules: Optional[str] = None  # Additional model-specific rules
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelStyleConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MemoryStyleManager:
    """
    Manages memory style configuration per model.
    Ensures consistent pronoun usage when storing memories.
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.data_dir = config.get_model_data_dir(model_id)
        self.config_file = self.data_dir / "style_config.json"
        self._config: Optional[ModelStyleConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load style configuration from disk."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._config = ModelStyleConfig.from_dict(data)
            except Exception as e:
                print(f"[WARN] Failed to load style config for {self.model_id}: {e}")
                self._config = ModelStyleConfig()
        else:
            self._config = ModelStyleConfig()
    
    def _save_config(self):
        """Save style configuration to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self._config.to_dict(), f, indent=2)
    
    @property
    def user_name(self) -> str:
        return self._config.user_name
    
    @property
    def ai_name(self) -> str:
        return self._config.ai_name
    
    def set_user_name(self, name: str):
        """Set the user's name for this model."""
        self._config.user_name = name
        self._save_config()
    
    def set_custom_rules(self, rules: str):
        """Set custom style rules for this model."""
        self._config.custom_rules = rules
        self._save_config()
    
    def get_style_guide(self) -> str:
        """
        Get the full style guide for this model.
        Combines default rules with model-specific customizations.
        """
        guide_parts = [config.MEMORY_STYLE_GUIDE]
        
        # Add model-specific user name
        if self._config.user_name != "User":
            guide_parts.append(f"\n**For this conversation**: The user's name is '{self._config.user_name}'. Use their name when storing memories about them.")
        
        # Add custom rules if any
        if self._config.custom_rules:
            guide_parts.append(f"\n**Additional Rules**:\n{self._config.custom_rules}")
        
        return "\n".join(guide_parts)
    
    def get_storage_prompt(self) -> str:
        """
        Get a prompt to prepend when the AI is storing memories.
        This reminds the AI of proper pronoun usage.
        """
        user = self._config.user_name
        return f"""When storing this memory, remember:
- Use "{user}" to refer to the human, never "I"
- Use "I" only when referring to yourself (the AI)
- Use "We" for shared experiences
- Store as a concise fact from your (the AI's) perspective"""
    
    def validate_memory_content(self, content: str) -> dict:
        """
        Validate memory content for pronoun issues.
        Returns a dict with validation results and suggestions.
        """
        issues = []
        suggestions = []
        user = self._config.user_name
        
        # Check for potential pronoun confusion patterns
        # These patterns suggest the AI might be referring to the user as "I"
        
        # "I am a [profession/role]" - likely about user, not AI
        profession_pattern = r'\bI am (?:a |an )?(?:software|engineer|doctor|teacher|student|developer|designer|manager|scientist|researcher|writer|artist)'
        if re.search(profession_pattern, content, re.IGNORECASE):
            issues.append("Possible pronoun confusion: 'I am a [profession]' might refer to user")
            suggestions.append(f"Consider: '{user} is a [profession]' if this refers to the human")
        
        # "I work at/for/on" - likely about user
        work_pattern = r'\bI (?:work|worked|working) (?:at|for|on|in)\b'
        if re.search(work_pattern, content, re.IGNORECASE):
            issues.append("Possible pronoun confusion: 'I work at...' might refer to user")
            suggestions.append(f"Consider: '{user} works at...' if this refers to the human")
        
        # "I live in" - likely about user
        live_pattern = r'\bI live(?:d|s)? (?:in|at)\b'
        if re.search(live_pattern, content, re.IGNORECASE):
            issues.append("Possible pronoun confusion: 'I live in...' might refer to user")
            suggestions.append(f"Consider: '{user} lives in...' if this refers to the human")
        
        # "My name is" - definitely about user (AI wouldn't store this about itself)
        name_pattern = r'\bmy name is\b'
        if re.search(name_pattern, content, re.IGNORECASE):
            issues.append("Pronoun confusion: 'My name is...' should use the user's name")
            suggestions.append(f"Consider: '{user}'s name is...' or just store the name directly")
        
        # "I have [number] [family member]" - likely about user
        family_pattern = r'\bI have (?:\d+|a|an|two|three|four|five) (?:kid|child|children|dog|cat|pet|sibling|brother|sister|son|daughter)'
        if re.search(family_pattern, content, re.IGNORECASE):
            issues.append("Possible pronoun confusion: 'I have [family/pets]' might refer to user")
            suggestions.append(f"Consider: '{user} has...' if this refers to the human")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "content": content
        }
    
    def suggest_correction(self, content: str) -> str:
        """
        Attempt to auto-correct obvious pronoun issues.
        This is a best-effort transformation, not guaranteed.
        """
        user = self._config.user_name
        corrected = content
        
        # Only apply corrections for clearly user-related statements
        # Be conservative to avoid breaking AI self-references
        
        # "My name is X" -> "User's name is X"
        corrected = re.sub(
            r'\bmy name is\b',
            f"{user}'s name is",
            corrected,
            flags=re.IGNORECASE
        )
        
        return corrected


# Cache of style managers per model
_style_managers: dict[str, MemoryStyleManager] = {}


def get_style_manager(model_id: str) -> MemoryStyleManager:
    """Get or create a style manager for a model."""
    if model_id not in _style_managers:
        _style_managers[model_id] = MemoryStyleManager(model_id)
    return _style_managers[model_id]
