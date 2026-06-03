"""List directory tool with workspace boundary enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolsclaw.tool import Tool


class ListDirTool(Tool):
    """List files and directories."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List files and directories at a given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (relative to workspace). Defaults to workspace root.",
                },
            },
        }

    def _resolve(self, path: str | None) -> Path:
        if path:
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = (self._workspace / p).resolve()
            else:
                p = p.resolve()
        else:
            p = self._workspace
        if not self._is_under(p, self._workspace):
            raise PermissionError(f"Path escapes workspace: {path}")
        return p

    @staticmethod
    def _is_under(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    async def execute(self, **kwargs: Any) -> str:
        path: str | None = kwargs.get("path")

        try:
            dp = self._resolve(path)
        except PermissionError as e:
            return str(e)

        if not dp.exists():
            return f"Error: path not found: {path or '.'}"
        if not dp.is_dir():
            return f"Error: not a directory: {path or '.'}"

        try:
            entries = sorted(dp.iterdir())
            lines: list[str] = []
            for entry in entries:
                prefix = "[dir] " if entry.is_dir() else "[file]"
                lines.append(f"{prefix} {entry.name}")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {e}"
