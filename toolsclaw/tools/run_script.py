"""Run a Python script inside a skill's isolated venv."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from toolsclaw.skills import Skill
from toolsclaw.tool import Tool


class RunScriptTool(Tool):
    """Execute a Python script using a skill's venv (for isolated dependencies)."""

    def __init__(self, workspace: Path, skills: list[Skill], timeout: int = 120) -> None:
        self._workspace = workspace.resolve()
        self._skills = {s.name: s for s in skills if s.available}
        self._timeout = min(timeout, 600)

    @property
    def name(self) -> str:
        return "run_script"

    @property
    def description(self) -> str:
        names = ", ".join(self._skills.keys()) or "(none)"
        return (
            "Execute a Python script with a skill's isolated dependencies. "
            f"Available skills with venvs: {names}. "
            "If no skill is specified, uses the system Python."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Path to the Python script (relative to workspace) or inline code.",
                },
                "skill": {
                    "type": "string",
                    "description": "Skill name whose venv to use. Omit to use system Python.",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments to pass to the script.",
                },
                "inline": {
                    "type": "boolean",
                    "description": "If true, treat 'script' as inline Python code instead of a file path.",
                },
            },
            "required": ["script"],
        }

    async def execute(self, **kwargs: Any) -> str:
        script: str = kwargs.get("script", "")
        skill_name: str = kwargs.get("skill", "")
        args: list[str] = kwargs.get("args", [])
        inline: bool = kwargs.get("inline", False)

        # resolve python executable
        python_exe = sys.executable
        if skill_name:
            skill = self._skills.get(skill_name)
            if not skill:
                return f"Error: skill '{skill_name}' not found or unavailable"
            python_exe = skill.get_python()

        # build command
        if inline:
            cmd = [python_exe, "-c", script, *args]
        else:
            script_path = self._workspace / script
            if not script_path.exists():
                return f"Error: script not found: {script}"
            cmd = [python_exe, str(script_path), *args]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._workspace),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            if len(output) > 10000:
                output = output[:5000] + "\n... (truncated) ...\n" + output[-5000:]

            exit_info = f"\n[exit code: {proc.returncode}]" if proc.returncode != 0 else ""
            return output + exit_info

        except asyncio.TimeoutError:
            return f"Error: script timed out after {self._timeout}s"
        except Exception as e:
            return f"Error: {e}"
