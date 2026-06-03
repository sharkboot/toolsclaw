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


class Config(BaseModel):
    """Root configuration."""

    model: str = "deepseek/deepseek-chat"
    provider: str = "auto"
    workspace: str = ""
    sandbox: bool = True
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    exec: ExecConfig = Field(default_factory=ExecConfig)

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
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return Config(**data)
    return Config()


def save_config(config: Config, path: Path | None = None) -> None:
    """Save config to JSON file."""
    path = path or DEFAULT_CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def init_config(workspace: str = "") -> Config:
    """Create a default config and workspace directory."""
    cfg = Config(workspace=workspace or str(DEFAULT_CONFIG_DIR / "workspace"))
    cfg.get_workspace().mkdir(parents=True, exist_ok=True)
    save_config(cfg)
    return cfg
