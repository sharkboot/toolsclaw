"""Load skill tool - directly load a skill's content by name."""

from __future__ import annotations

from typing import Any

from toolsclaw.skills import Skill
from toolsclaw.tool import Tool


class LoadSkillTool(Tool):
    """Load a skill's full content by name for immediate use."""

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = {s.name: s for s in skills}

    @property
    def name(self) -> str:
        return "load_skill"

    @property
    def description(self) -> str:
        return (
            "Load a skill's full instructions by name. Use this when you identify "
            "a relevant skill from the skills summary. Returns the complete SKILL.md "
            "content with paths resolved."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to load (from the skills summary).",
                },
            },
            "required": ["skill_name"],
        }

    async def execute(self, **kwargs: Any) -> str:
        skill_name: str = kwargs.get("skill_name", "")

        if not skill_name:
            return "Error: skill_name is required"

        skill = self._skills.get(skill_name)
        if not skill:
            available = ", ".join(self._skills.keys())
            return f"Error: skill '{skill_name}' not found. Available skills: {available}"

        if not skill.available:
            missing = []
            if skill.requires.bins:
                missing.append(f"CLI tools: {', '.join(skill.requires.bins)}")
            if skill.requires.env:
                missing.append(f"env vars: {', '.join(skill.requires.env)}")
            return f"Error: skill '{skill_name}' is unavailable. Missing: {'; '.join(missing)}"

        # Replace {SKILL_DIR} with actual path
        skill_dir = str(skill.path.parent.resolve())
        content = skill.content.replace("{SKILL_DIR}", skill_dir)

        return content
