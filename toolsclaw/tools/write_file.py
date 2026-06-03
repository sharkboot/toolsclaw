"""Write file tool with workspace boundary enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolsclaw.tool import Tool


class WriteFileTool(Tool):
    """Write content to a file."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative to workspace or absolute).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write.",
                },
            },
            "required": ["path", "content"],
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
        content: str = kwargs.get("content", "")

        try:
            fp = self._resolve(path)
        except PermissionError as e:
            return str(e)

        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {fp}"
        except Exception as e:
            return f"Error writing file: {e}"
