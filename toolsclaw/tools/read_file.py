"""Read file tool with workspace boundary enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolsclaw.tool import Tool


class ReadFileTool(Tool):
    """Read a file's contents."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file and return it as text."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to workspace or absolute).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based).",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read.",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        }

    def _resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (self._workspace / p).resolve()
        else:
            p = p.resolve()
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
        path: str = kwargs.get("path", "")
        offset: int = kwargs.get("offset", 1)
        limit: int = kwargs.get("limit", 2000)

        try:
            fp = self._resolve(path)
        except PermissionError as e:
            return str(e)

        if not fp.exists():
            return f"Error: file not found: {path}"
        if not fp.is_file():
            return f"Error: not a file: {path}"

        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = text.splitlines()
        selected = lines[offset - 1 : offset - 1 + limit]
        return "\n".join(selected)
