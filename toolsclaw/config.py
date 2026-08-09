"""Configuration schema for toolsclaw."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """LLM provider connection settings."""

    api_key: str = ""
    api_base: str = ""


class ExecConfig(BaseModel):
    """Shell execution settings."""

    enable: bool = True
    timeout: int = 60
    deny_patterns: list[str] = Field(default_factory=lambda: [
        r"rm\s+-rf\s+/",
        r"del\s+/[fFsS]\s+[a-zA-Z]:\\",
        r"format\s+[a-zA-Z]:",
        r"mkfs\.",
        r"dd\s+if=",
        r":\(\)\{.*\|.*&\s*\};",  # fork bomb
        r"shutdown",
        r"reboot",
    ])


class MemoryCompressionConfig(BaseModel):
    """Memory compression settings for long conversations."""

    enabled: bool = False
    """Enable automatic memory compression."""

    strategy: str = "hybrid"
    """Compression strategy: truncate, drop, summarize, or hybrid."""

    threshold_tokens: int = 80_000
    """Trigger compression when estimated tokens exceed this value."""

    target_ratio: float = 0.5
    """Target compression ratio (0.0-1.0). 0.5 = compress to 50% of original."""

    min_rounds_to_keep: int = 3
    """Minimum number of recent conversation rounds to preserve intact."""

    tool_result_max_chars: int = 300
    """Max characters per tool result field after compression."""


class PersistentMemoryConfig(BaseModel):
    """Persistent memory settings for cross-session knowledge retention."""

    enabled: bool = False
    """Enable persistent memory."""

    auto_save: bool = True
    """Automatically save memories from conversations."""

    max_memories: int = 5
    """Maximum number of relevant memories to inject into context."""

    memory_dir: str = ""
    """Custom memory directory path. If empty, uses ``<workspace>/.claude/memory/``."""


class Config(BaseModel):
    """Root configuration."""

    model: str = "deepseek/deepseek-chat"
    provider: str = "auto"
    workspace: str = ""
    sandbox: bool = True
    max_iterations: int = 100
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    exec: ExecConfig = Field(default_factory=ExecConfig)
    memory: MemoryCompressionConfig = Field(default_factory=MemoryCompressionConfig)
    persistent_memory: PersistentMemoryConfig = Field(default_factory=PersistentMemoryConfig)

    def get_provider_config(self) -> ProviderConfig:
        """Resolve the provider config for the current model."""
        if self.provider != "auto" and self.provider in self.providers:
            return self.providers[self.provider]
        # auto-detect: try matching model prefix
        for name, cfg in self.providers.items():
            if cfg.api_key:
                return cfg
        return ProviderConfig()

    def get_workspace(self) -> Path:
        """Return the resolved workspace path."""
        if self.workspace:
            return Path(self.workspace).expanduser().resolve()
        return Path.cwd()


DEFAULT_CONFIG_DIR = Path.home() / ".toolsclaw"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"


def load_config(path: Path | None = None) -> Config:
    """Load config from JSON file, falling back to defaults."""
    path = path or DEFAULT_CONFIG_FILE
    p = Path(path) if isinstance(path, str) else path
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return Config(**data)
    return Config()


def save_config(config: Config, path: Path | None = None) -> None:
    """Save config to JSON file."""
    path = path or DEFAULT_CONFIG_FILE
    p = Path(path) if isinstance(path, str) else path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(config.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def init_config(workspace: str = "") -> Config:
    """Create a default config and workspace directory."""
    cfg = Config(workspace=workspace or str(DEFAULT_CONFIG_DIR / "workspace"))
    cfg.get_workspace().mkdir(parents=True, exist_ok=True)
    save_config(cfg)
    return cfg
