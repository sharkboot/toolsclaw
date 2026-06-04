"""Shell execution tool with workspace sandbox."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

from toolsclaw.tool import Tool


class ExecTool(Tool):
    """Execute shell commands with workspace containment."""

    def __init__(
        self,
        workspace: Path,
        timeout: int = 60,
        deny_patterns: list[str] | None = None,
    ) -> None:
        self._workspace = workspace.resolve()
        self._timeout = min(timeout, 600)
        self._deny_patterns = deny_patterns or []

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            "The command runs inside the workspace directory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory (relative to workspace). Defaults to workspace root.",
                },
            },
            "required": ["command"],
        }

    def _check_deny(self, command: str) -> str | None:
        """Return an error message if the command matches a deny pattern."""
        for pattern in self._deny_patterns:
            if re.search(pattern, command):
                return f"Blocked by security policy: pattern '{pattern}'"
        return None

    def _resolve_workdir(self, workdir: str | None) -> Path:
        """Resolve and validate working directory stays within workspace."""
        if workdir:
            target = (self._workspace / workdir).resolve()
        else:
            target = self._workspace
        if not self._is_under(target, self._workspace):
            raise PermissionError(f"Working directory escapes workspace: {workdir}")
        return target

    @staticmethod
    def _is_under(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs.get("command", "")
        workdir: str | None = kwargs.get("workdir")

        if not command.strip():
            return "Error: empty command"

        deny = self._check_deny(command)
        if deny:
            return deny

        try:
            cwd = self._resolve_workdir(workdir)
        except PermissionError as e:
            return str(e)

        try:
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(cwd),
                    env=self._build_env(),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=str(cwd),
                    env=self._build_env(),
                    executable="/bin/bash",
                )

            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            # truncate very long output
            if len(output) > 10000:
                output = output[:5000] + "\n... (truncated) ...\n" + output[-5000:]

            exit_info = f"\n[exit code: {proc.returncode}]" if proc.returncode != 0 else ""
            return output + exit_info

        except asyncio.TimeoutError:
            return f"Error: command timed out after {self._timeout}s"
        except Exception as e:
            return f"Error: {e}"

    def _build_env(self) -> dict[str, str]:
        """Build environment for subprocess execution — inherit full parent env."""
        return dict(os.environ)
