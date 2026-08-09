"""Persistent file-based memory management for toolsclaw.

Stores memories as markdown files with YAML frontmatter, managed
via a MEMORY.md index. Inspired by the Claude Code memory system.

Each memory file has::

    ---
    name: short-kebab-case-slug
    description: one-line summary
    metadata:
      type: user | feedback | project | reference
    ---

    <the fact content>

    Related: [[other-memory]], [[another-memory]]

Usage::

    store = MemoryStore(Path("/path/to/memory/dir"))
    store.save(Memory(
        name="user-preference",
        description="User prefers concise responses",
        type="user",
        content="The user likes short, direct answers.",
    ))
    memories = store.search("preference")
    index = store.rebuild_index()
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_DIR_NAME = "memory"
INDEX_FILENAME = "MEMORY.md"

VALID_TYPES = frozenset({"user", "feedback", "project", "reference"})

# Match [[name]] links in memory content
LINK_RE = re.compile(r"\[\[([\w-]+)\]\]")

# Frontmatter delimiter
_FM_DELIM = "---"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Memory:
    """A single persistent memory entry.

    Args:
        name: Short kebab-case slug (e.g. ``user-preference``).
        description: One-line summary used for relevance during recall.
        type: One of ``user``, ``feedback``, ``project``, ``reference``.
        content: The actual memory content.
        links: List of linked memory names (extracted from [[name]] in content).
        created_at: Creation timestamp (auto-set if None).
        updated_at: Last update timestamp (auto-set if None).
    """

    __slots__ = (
        "_name",
        "_description",
        "_type",
        "_content",
        "_links",
        "_created_at",
        "_updated_at",
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        type: str = "reference",
        content: str = "",
        links: list[str] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        if not re.match(r"^[a-z][a-z0-9-]*$", name):
            raise ValueError(
                f"Memory name must be a kebab-case slug, got {name!r}"
            )
        if type not in VALID_TYPES:
            raise ValueError(
                f"Invalid memory type {type!r}. Must be one of {sorted(VALID_TYPES)}"
            )

        self._name = name
        self._description = description
        self._type = type
        self._content = content
        # Extract links from content, merging with any explicitly passed links
        content_links = LINK_RE.findall(content)
        self._links = sorted(set(list(links or []) + content_links))
        now = datetime.now(timezone.utc)
        self._created_at = created_at or now
        self._updated_at = updated_at or now

    # -- read-only properties ---------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def type(self) -> str:
        return self._type

    @property
    def content(self) -> str:
        return self._content

    @property
    def links(self) -> list[str]:
        return list(self._links)

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    # -- serialisation ----------------------------------------------------

    def to_frontmatter(self) -> str:
        """Render this memory as a markdown file with YAML frontmatter."""
        lines = [_FM_DELIM]
        lines.append(f"name: {self._name}")
        if self._description:
            lines.append(f"description: {self._description}")
        lines.append("metadata:")
        lines.append(f"  type: {self._type}")
        # Only include timestamps if they are meaningful
        lines.append(f"created_at: {self._created_at.isoformat()}")
        lines.append(f"updated_at: {self._updated_at.isoformat()}")
        lines.append(_FM_DELIM)
        lines.append("")  # blank line after frontmatter
        lines.append(self._content.strip())
        lines.append("")  # trailing newline
        return "\n".join(lines)

    @classmethod
    def from_frontmatter(cls, text: str, file_path: Path | None = None) -> Memory:
        """Parse a markdown file with YAML frontmatter into a Memory.

        Args:
            text: The raw file content.
            file_path: Optional path for error messages.

        Returns:
            A new Memory instance.

        Raises:
            ValueError: If the frontmatter is malformed or required fields missing.
        """
        tag = f" ({file_path})" if file_path else ""

        # Split frontmatter from body
        lines = text.split("\n")
        if not lines or lines[0].strip() != _FM_DELIM:
            raise ValueError(f"Missing frontmatter delimiter{tag}")

        # Find closing delimiter
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == _FM_DELIM:
                end_idx = i
                break

        if end_idx is None:
            raise ValueError(f"Unclosed frontmatter{tag}")

        fm_lines = lines[1:end_idx]
        body = "\n".join(lines[end_idx + 1:]).strip()

        # Parse YAML-like frontmatter (simple key-value, no nested YAML)
        # We use a simple parser instead of pulling in pyyaml dependency
        fm_data = _parse_simple_frontmatter(fm_lines, tag)

        name = fm_data.get("name")
        if not name:
            raise ValueError(f"Missing 'name' in frontmatter{tag}")

        # Extract metadata.type
        mem_type = "reference"
        metadata = fm_data.get("metadata", {})
        if isinstance(metadata, dict):
            mem_type = metadata.get("type", "reference")
        elif isinstance(metadata, str):
            mem_type = metadata

        description = fm_data.get("description", "")

        # Parse timestamps
        created_at = None
        updated_at = None
        if "created_at" in fm_data:
            try:
                created_at = datetime.fromisoformat(fm_data["created_at"])
            except (ValueError, TypeError):
                pass
        if "updated_at" in fm_data:
            try:
                updated_at = datetime.fromisoformat(fm_data["updated_at"])
            except (ValueError, TypeError):
                pass

        # Extract links from body
        links = LINK_RE.findall(body)

        return cls(
            name=name,
            description=description,
            type=mem_type,
            content=body,
            links=links,
            created_at=created_at,
            updated_at=updated_at,
        )

    # -- helpers ----------------------------------------------------------

    def to_index_line(self) -> str:
        """Return a single line for MEMORY.md index."""
        type_tag = f"[{self._type}]"
        # Keep description concise for index
        desc = self._description or "(no description)"
        return f"- {type_tag} **{self._name}** — {desc}"

    def __repr__(self) -> str:
        return (
            f"Memory(name={self._name!r}, type={self._type!r}, "
            f"description={self._description!r})"
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MemoryStore:
    """File-based persistent memory store.

    Manages individual memory files inside a ``memory/`` directory and
    maintains a ``MEMORY.md`` index file.

    Args:
        base_dir: The root directory where the ``memory/`` folder lives.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir).expanduser().resolve()
        self._memory_dir = self._base / MEMORY_DIR_NAME
        self._index_path = self._base / INDEX_FILENAME

    # -- properties -------------------------------------------------------

    @property
    def base_dir(self) -> Path:
        return self._base

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    @property
    def index_path(self) -> Path:
        return self._index_path

    # -- CRUD -------------------------------------------------------------

    def save(self, memory: Memory) -> Path:
        """Write a memory to disk. Creates the memory directory if needed.

        Updates ``updated_at`` to now. Returns the file path written.
        """
        memory._updated_at = datetime.now(timezone.utc)
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._memory_dir / f"{memory.name}.md"
        file_path.write_text(memory.to_frontmatter(), encoding="utf-8")
        logger.info("Memory saved: %s (%s)", memory.name, file_path)
        return file_path

    def load(self, name: str) -> Memory | None:
        """Load a single memory by name. Returns None if not found."""
        file_path = self._memory_dir / f"{name}.md"
        if not file_path.exists():
            return None
        try:
            text = file_path.read_text(encoding="utf-8")
            return Memory.from_frontmatter(text, file_path=file_path)
        except (ValueError, OSError) as exc:
            logger.warning("Failed to load memory %s: %s", name, exc)
            return None

    def delete(self, name: str) -> bool:
        """Delete a memory file by name. Returns True if deleted."""
        file_path = self._memory_dir / f"{name}.md"
        if not file_path.exists():
            return False
        file_path.unlink()
        logger.info("Memory deleted: %s", name)
        return True

    def list_memories(self) -> list[Memory]:
        """List all memories in the store, sorted by name."""
        if not self._memory_dir.exists():
            return []
        memories: list[Memory] = []
        for f in sorted(self._memory_dir.iterdir()):
            if f.suffix == ".md" and f.is_file():
                try:
                    mem = Memory.from_frontmatter(
                        f.read_text(encoding="utf-8"), file_path=f
                    )
                    memories.append(mem)
                except (ValueError, OSError) as exc:
                    logger.warning("Skipping invalid memory file %s: %s", f.name, exc)
        return memories

    def exists(self, name: str) -> bool:
        """Check if a memory with the given name exists."""
        return (self._memory_dir / f"{name}.md").exists()

    # -- search -----------------------------------------------------------

    def search(self, query: str, *, max_results: int = 10) -> list[Memory]:
        """Full-text search across memory content and metadata.

        Matches against name, description, and content. Supports multi-word
        queries by splitting into individual tokens. Case-insensitive.
        """
        query = query.lower().strip()
        if not query:
            return []

        # Tokenise: split into individual words for multi-word matching
        tokens = [t for t in query.split() if len(t) > 1]
        if not tokens:
            tokens = [query]

        results: list[tuple[Memory, int]] = []

        for mem in self.list_memories():
            score = 0

            # Exact phrase match (highest priority)
            if query in mem.name.lower():
                score += 10
            if query in mem.description.lower():
                score += 5
            if query in mem.content.lower():
                score += 3

            # Individual token matches (partial)
            for token in tokens:
                if token in mem.name.lower():
                    score += 3
                if token in mem.description.lower():
                    score += 2
                if token in mem.content.lower():
                    score += 1
                if token == mem.type.lower():
                    score += 2

            if score > 0:
                results.append((mem, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in results[:max_results]]

    def find_by_type(self, type_: str, *, max_results: int = 20) -> list[Memory]:
        """Find all memories of a given type."""
        all_mems = self.list_memories()
        filtered = [m for m in all_mems if m.type == type_]
        return filtered[:max_results]

    # -- index management -------------------------------------------------

    def rebuild_index(self) -> str:
        """Rebuild the MEMORY.md index file from all stored memories.

        Returns the index content as a string.
        """
        memories = self.list_memories()
        lines: list[str] = [
            "# Memory Index",
            "",
            f"Auto-generated index. {len(memories)} memory file(s) total.",
            "",
        ]

        # Group by type
        by_type: dict[str, list[Memory]] = {}
        for mem in memories:
            by_type.setdefault(mem.type, []).append(mem)

        for type_ in sorted(by_type.keys()):
            lines.append(f"## {type_.capitalize()}")
            lines.append("")
            for mem in by_type[type_]:
                lines.append(mem.to_index_line())
            lines.append("")

        if not memories:
            lines.append("_(no memories yet)_")
            lines.append("")

        content = "\n".join(lines)
        self._index_path.write_text(content, encoding="utf-8")
        logger.info("Memory index rebuilt: %d entries", len(memories))
        return content

    def get_index_preview(self, max_lines: int = 20) -> str:
        """Read the first N lines of the index for context injection."""
        if not self._index_path.exists():
            return self.rebuild_index()
        lines = self._index_path.read_text(encoding="utf-8").split("\n")
        return "\n".join(lines[:max_lines])

    # -- link graph -------------------------------------------------------

    def find_related(self, name: str) -> list[Memory]:
        """Find memories related to the given memory via ``[[name]]`` links.

        Returns both:
        - Memories that *this* memory links to (outgoing links).
        - Memories that link *to* this memory (incoming links).

        Deduplicated and sorted by name.
        """
        target = self.load(name)
        if target is None:
            return []

        # Outgoing: load each [[link]] from the target's content
        seen: set[str] = set()
        related: list[Memory] = []

        for link_name in target.links:
            if link_name in seen:
                continue
            seen.add(link_name)
            linked = self.load(link_name)
            if linked is not None:
                related.append(linked)

        # Incoming: scan all memories for [[name]] references
        for mem in self.list_memories():
            if mem.name == name or mem.name in seen:
                continue
            if name in mem.links:
                seen.add(mem.name)
                related.append(mem)

        related.sort(key=lambda m: m.name)
        return related

    def get_link_graph(self) -> dict[str, list[str]]:
        """Build the full link graph as an adjacency dict.

        Returns a ``{memory_name: [linked_names]}`` dict. Only includes
        links where the target memory actually exists.
        """
        all_mems = self.list_memories()
        name_set = {m.name for m in all_mems}

        graph: dict[str, list[str]] = {}
        for mem in all_mems:
            resolved = [ln for ln in mem.links if ln in name_set]
            if resolved:
                graph[mem.name] = resolved
        return graph

    # -- consolidation ----------------------------------------------------

    def find_duplicates(self) -> list[tuple[Memory, Memory]]:
        """Find memory pairs that have highly similar names or content.

        Returns a list of ``(kept, duplicate)`` tuples where the duplicate
        is suggested for merging into the kept memory. Uses simple heuristics:
        exact name match after normalisation, or near-identical content.
        """
        all_mems = self.list_memories()
        pairs: list[tuple[Memory, Memory]] = []
        seen: set[str] = set()

        for i, a in enumerate(all_mems):
            if a.name in seen:
                continue
            for b in all_mems[i + 1:]:
                if b.name in seen:
                    continue

                # Same name after normalisation (e.g. tool-used-read-file vs tool-used-readfile)
                norm_a = a.name.replace("-", "").replace("_", "")
                norm_b = b.name.replace("-", "").replace("_", "")
                if norm_a == norm_b:
                    pairs.append((a, b))
                    seen.add(b.name)
                    continue

                # Near-identical content (one is substring of another)
                if len(a.content) > 20 and len(b.content) > 20:
                    if a.content in b.content or b.content in a.content:
                        pairs.append((a, b))
                        seen.add(b.name)
                        continue

        return pairs

    def consolidate(self, dry_run: bool = True) -> dict[str, Any]:
        """Merge duplicate or highly similar memories.

        Args:
            dry_run: If True, only report what would be merged without
                actually modifying anything.

        Returns:
            A dict with keys ``merged`` (count), ``kept`` (count),
            ``details`` (list of merge descriptions).
        """
        pairs = self.find_duplicates()
        details: list[dict[str, Any]] = []
        merged_count = 0
        kept_count = 0

        for kept, duplicate in pairs:
            # Merge: keep the longer description, combine content
            merged_desc = (
                kept.description
                if len(kept.description) >= len(duplicate.description)
                else duplicate.description
            )
            merged_content = kept.content
            if duplicate.content not in merged_content:
                merged_content = merged_content + "\n\n" + duplicate.content

            details.append({
                "kept": kept.name,
                "removed": duplicate.name,
                "description": merged_desc,
                "content_length": len(merged_content),
            })

            if not dry_run:
                updated = Memory(
                    name=kept.name,
                    description=merged_desc,
                    type=kept.type,
                    content=merged_content,
                    links=list(set(kept.links + duplicate.links)),
                    created_at=kept.created_at,
                )
                self.save(updated)
                self.delete(duplicate.name)
                merged_count += 1
                kept_count += 1

        if not dry_run:
            self.rebuild_index()
            logger.info(
                "Consolidation: merged %d duplicate(s), kept %d",
                merged_count,
                kept_count,
            )

        return {
            "merged": len(pairs) if dry_run else merged_count,
            "kept": len(pairs) if dry_run else kept_count,
            "dry_run": dry_run,
            "details": details,
        }

    # -- stats ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return memory store statistics.

        Returns a dict with:
        - ``total``: total memory count.
        - ``by_type``: dict of ``{type: count}``.
        - ``total_links``: total number of ``[[links]]`` across all memories.
        - ``orphan_links``: links that point to non-existent memories.
        - ``oldest``: ISO timestamp of the oldest memory.
        - ``newest``: ISO timestamp of the newest memory.
        - ``total_size_bytes``: total disk size of all memory files.
        """
        all_mems = self.list_memories()
        name_set = {m.name for m in all_mems}

        by_type: dict[str, int] = {}
        total_links = 0
        orphan_links = 0
        total_size = 0
        oldest: datetime | None = None
        newest: datetime | None = None

        for mem in all_mems:
            by_type[mem.type] = by_type.get(mem.type, 0) + 1
            total_links += len(mem.links)
            for ln in mem.links:
                if ln not in name_set:
                    orphan_links += 1
            if oldest is None or mem.created_at < oldest:
                oldest = mem.created_at
            if newest is None or mem.created_at > newest:
                newest = mem.created_at

            fpath = self._memory_dir / f"{mem.name}.md"
            if fpath.exists():
                total_size += fpath.stat().st_size

        return {
            "total": len(all_mems),
            "by_type": dict(sorted(by_type.items())),
            "total_links": total_links,
            "orphan_links": orphan_links,
            "oldest": oldest.isoformat() if oldest else None,
            "newest": newest.isoformat() if newest else None,
            "total_size_bytes": total_size,
        }

    # -- upsert -----------------------------------------------------------

    def update_or_create(
        self,
        name: str,
        *,
        description: str | None = None,
        type: str | None = None,
        content: str | None = None,
        merge_content: bool = False,
    ) -> Memory:
        """Update an existing memory or create a new one.

        Args:
            name: Memory name (kebab-case slug).
            description: If provided, replaces the existing description.
            type: If provided, replaces the existing type.
            content: If provided, replaces (or merges with) existing content.
            merge_content: If True and the memory exists, append new content
                to existing content instead of replacing it.

        Returns:
            The saved Memory instance.
        """
        existing = self.load(name)

        if existing is None:
            # Create new
            mem = Memory(
                name=name,
                description=description or "",
                type=type or "reference",
                content=content or "",
            )
            self.save(mem)
            return mem

        # Update existing
        new_desc = description if description is not None else existing.description
        new_type = type if type is not None else existing.type

        if content is not None:
            if merge_content and content not in existing.content:
                new_content = existing.content + "\n\n" + content
            else:
                new_content = content
        else:
            new_content = existing.content

        # Extract links from new content if provided
        new_links = list(existing.links)
        if content:
            new_links = list(set(new_links + LINK_RE.findall(content)))

        mem = Memory(
            name=existing.name,
            description=new_desc,
            type=new_type,
            content=new_content,
            links=new_links,
            created_at=existing.created_at,
        )
        self.save(mem)
        return mem

    # -- batch operations -------------------------------------------------

    def import_memories(self, source_dir: str | Path) -> int:
        """Import memory files from an external directory.

        Only imports files matching the frontmatter format.
        Returns the number of successfully imported memories.
        """
        source = Path(source_dir)
        if not source.is_dir():
            logger.warning("Import source not found: %s", source)
            return 0

        count = 0
        for f in sorted(source.iterdir()):
            if f.suffix == ".md" and f.is_file():
                try:
                    text = f.read_text(encoding="utf-8")
                    mem = Memory.from_frontmatter(text, file_path=f)
                    # Avoid overwrite if name already exists
                    if not self.exists(mem.name):
                        self.save(mem)
                        count += 1
                    else:
                        logger.debug("Skipping duplicate memory: %s", mem.name)
                except (ValueError, OSError) as exc:
                    logger.debug("Skipping %s: %s", f.name, exc)
        logger.info("Imported %d memories from %s", count, source)
        return count


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------


def _parse_simple_frontmatter(
    lines: list[str], tag: str = ""
) -> dict[str, Any]:
    """Parse simple YAML-like frontmatter lines into a dict.

    Handles:
    - ``key: value`` scalars
    - ``metadata:`` with nested ``  key: value``
    - Quoted values (single and double)

    Does NOT handle lists, booleans, or multi-line strings.
    """
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_nested: dict[str, Any] | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for nested lines (indented under metadata:)
        if line.startswith("  ") or line.startswith("    "):
            if current_key is not None:
                nested_match = re.match(r"  +(\w+):\s*(.*)", line)
                if nested_match:
                    k, v = nested_match.group(1), nested_match.group(2).strip()
                    v = _strip_quotes(v)
                    if current_nested is not None:
                        current_nested[k] = v
                    else:
                        result[k] = v
            continue

        # Top-level key: value
        match = re.match(r"(\w+):\s*(.*)", stripped)
        if not match:
            continue

        key, value = match.group(1), match.group(2).strip()
        value = _strip_quotes(value)

        if key == "metadata":
            # Start of metadata block
            current_key = key
            current_nested = {}
            if value:  # inline metadata: type: user
                # Parse inline metadata: key: value
                inline_match = re.match(r"(\w+):\s*(.*)", value)
                if inline_match:
                    current_nested[inline_match.group(1)] = _strip_quotes(
                        inline_match.group(2).strip()
                    )
            result[key] = current_nested
        else:
            current_key = key
            result[key] = value

    return result


def _strip_quotes(value: str) -> str:
    """Remove surrounding single or double quotes from a value."""
    if len(value) >= 2:
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
    return value