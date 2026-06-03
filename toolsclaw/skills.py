"""Skills loader — markdown files with optional YAML frontmatter.

Each skill can declare dependencies in its frontmatter:

    ---
    name: my-skill
    description: Does cool things.
    requires:
      bins: [curl, ffmpeg]          # CLI tools that must be on PATH
      env: [MY_API_KEY]             # environment variables that must be set
      pip: [requests>=2.28, bs4]   # Python packages (installed into a per-skill venv)
    ---

- `bins` and `env` are checked at load time; missing requirements mark the skill as unavailable.
- `pip` dependencies are installed into an isolated venv under `.skills_venvs/<skill_name>/`
  on first use, so different skills can depend on conflicting package versions without issue.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillRequirements:
    """Parsed `requires` block from skill frontmatter."""

    bins: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    pip: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    description: str
    always: bool
    content: str  # body without frontmatter
    path: Path
    requires: SkillRequirements = field(default_factory=SkillRequirements)
    available: bool = True  # False if requirements are not met

    @property
    def venv_dir(self) -> Path:
        """Per-skill venv location (sibling of the skill's SKILL.md)."""
        return self.path.parent / ".venv"

    def get_python(self) -> str:
        """Return the python executable for this skill's venv, or sys.executable."""
        if self.requires.pip and self.venv_dir.is_dir():
            if sys.platform == "win32":
                return str(self.venv_dir / "Scripts" / "python.exe")
            return str(self.venv_dir / "bin" / "python")
        return sys.executable


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _check_bins(bins: list[str]) -> list[str]:
    """Return list of missing CLI tools."""
    return [b for b in bins if not shutil.which(b)]


def _check_env(env_vars: list[str]) -> list[str]:
    """Return list of missing environment variables."""
    return [v for v in env_vars if not os.environ.get(v)]


def _ensure_venv(skill: Skill) -> bool:
    """Create the per-skill venv and install pip deps if needed.

    Returns True on success, False on failure.
    """
    if not skill.requires.pip:
        return True

    venv_dir = skill.venv_dir
    pip_exe = venv_dir / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")

    # create venv if it doesn't exist or pip is missing
    if not venv_dir.is_dir() or not pip_exe.exists():
        try:
            if venv_dir.is_dir():
                import shutil
                shutil.rmtree(venv_dir)
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[toolsclaw] Failed to create venv for skill '{skill.name}': {e}")
            return False

    # install dependencies
    try:
        subprocess.run(
            [str(pip_exe), "install", "-q", *skill.requires.pip],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[toolsclaw] Failed to install deps for skill '{skill.name}': {e}")
        return False

    return True


def _parse_skill(path: Path) -> Skill | None:
    """Parse a SKILL.md file into a Skill, or None on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    meta: dict[str, Any] = {}
    content = text

    m = _FRONTMATTER_RE.match(text)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        content = text[m.end():]

    # parse requirements
    raw_req = meta.get("requires", {})
    requires = SkillRequirements(
        bins=raw_req.get("bins", []),
        env=raw_req.get("env", []),
        pip=raw_req.get("pip", []),
    )

    # check availability
    missing_bins = _check_bins(requires.bins)
    missing_env = _check_env(requires.env)
    available = not missing_bins and not missing_env

    if missing_bins:
        print(f"[toolsclaw] Skill '{meta.get('name', path.parent.name)}' missing CLI: {', '.join(missing_bins)}")
    if missing_env:
        print(f"[toolsclaw] Skill '{meta.get('name', path.parent.name)}' missing ENV: {', '.join(missing_env)}")

    return Skill(
        name=meta.get("name", path.parent.name),
        description=meta.get("description", ""),
        always=bool(meta.get("always", False)),
        content=content.strip(),
        path=path,
        requires=requires,
        available=available,
    )


def load_skills(
    workspace: Path,
    builtin_dir: Path | None = None,
    *,
    extra_dirs: list[Path] | None = None,
) -> list[Skill]:
    """Discover and load skills from workspace, built-in, and extra directories.

    Search order (first-found wins for duplicate names):
      1. workspace/skills
      2. extra_dirs (in order)
      3. builtin_dir
    """
    skills: list[Skill] = []
    seen: set[str] = set()

    search_dirs: list[Path] = [workspace / "skills"]
    if extra_dirs:
        search_dirs.extend(extra_dirs)
    search_dirs.append(builtin_dir or Path())

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for skill_md in sorted(search_dir.rglob("SKILL.md")):
            skill = _parse_skill(skill_md)
            if skill and skill.name not in seen:
                skills.append(skill)
                seen.add(skill.name)

    return skills


def ensure_skill_deps(skills: list[Skill]) -> None:
    """Ensure all pip dependencies are installed for skills that need them.

    Call this once after loading skills. Each skill's venv is created lazily
    only if the skill declares `requires.pip` and is otherwise available.
    """
    for skill in skills:
        if skill.available and skill.requires.pip:
            if not _ensure_venv(skill):
                skill.available = False


def get_always_skills(skills: list[Skill]) -> list[Skill]:
    """Return skills marked as always-on and available."""
    return [s for s in skills if s.always and s.available]


def build_skills_summary(skills: list[Skill]) -> str:
    """Build a markdown summary of available skills."""
    if not skills:
        return ""
    lines = ["## Available Skills", ""]
    for s in skills:
        if not s.available:
            continue
        flag = " (always active)" if s.always else ""
        desc = f" — {s.description}" if s.description else ""
        lines.append(f"- **{s.name}**{desc}{flag}")
    return "\n".join(lines)
