"""CLI entry point for toolsclaw."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from toolsclaw.config import DEFAULT_CONFIG_FILE, init_config, load_config
from toolsclaw.persistent_memory import Memory, MemoryStore
from toolsclaw.runner import AgentRunner

app = typer.Typer(
    name="toolsclaw",
    help="Ultra-lightweight tool-calling agent framework.",
    no_args_is_help=True,
)
console = Console()


def _resolve_memory_store(config_path: str = "") -> MemoryStore:
    """Build a MemoryStore from config (or default workspace)."""
    cfg_path = Path(config_path) if config_path else None
    cfg = load_config(cfg_path)
    pm_cfg = cfg.persistent_memory
    if pm_cfg.memory_dir:
        base = Path(pm_cfg.memory_dir)
    elif cfg.workspace:
        base = Path(cfg.workspace).expanduser().resolve() / ".claude"
    else:
        base = Path.cwd() / ".claude"
    return MemoryStore(base)


@app.command()
def init(
    workspace: str = typer.Option(
        "", "--workspace", "-w", help="Workspace directory path."
    ),
) -> None:
    """Initialize config and workspace."""
    cfg = init_config(workspace)
    console.print(f"[green]OK[/green] Config saved to {DEFAULT_CONFIG_FILE}")
    console.print(f"[green]OK[/green] Workspace: {cfg.get_workspace()}")


@app.command()
def run(
    message: str = typer.Argument(..., help="Message to send to the agent."),
    config: str = typer.Option(
        "", "--config", "-c", help="Path to config file."
    ),
) -> None:
    """Send a single message and print the response."""
    cfg_path = Path(config) if config else None
    cfg = load_config(cfg_path)
    runner = AgentRunner(cfg)
    result = asyncio.run(runner.run(message))
    console.print(result)


@app.command()
def chat(
    config: str = typer.Option(
        "", "--config", "-c", help="Path to config file."
    ),
) -> None:
    """Start an interactive chat session."""
    cfg_path = Path(config) if config else None
    cfg = load_config(cfg_path)
    runner = AgentRunner(cfg)
    asyncio.run(runner.run_interactive())


# ---------------------------------------------------------------------------
# Memory management subcommands
# ---------------------------------------------------------------------------


@app.command()
def memory_add(
    name: str = typer.Argument(..., help="Memory name (kebab-case)."),
    content: str = typer.Argument(..., help="Memory content."),
    type: str = typer.Option("reference", "--type", "-t", help="Memory type: user, feedback, project, reference."),
    description: str = typer.Option("", "--desc", "-d", help="One-line description."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Add a new memory entry."""
    store = _resolve_memory_store(config)
    if store.exists(name):
        console.print(f"[yellow]Memory '{name}' already exists. Use 'memory update' to modify it.[/yellow]")
        raise typer.Exit(1)

    mem = Memory(
        name=name,
        description=description,
        type=type,
        content=content,
    )
    store.save(mem)
    store.rebuild_index()
    console.print(f"[green]✓[/green] Memory '{name}' saved.")
    console.print(f"   Location: {store.memory_dir / name}.md")


@app.command()
def memory_update(
    name: str = typer.Argument(..., help="Memory name to update."),
    content: str = typer.Argument(None, help="New content. Leave empty to keep existing."),
    description: str = typer.Option(None, "--desc", "-d", help="New description."),
    type: str = typer.Option(None, "--type", "-t", help="New type."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Update an existing memory entry."""
    store = _resolve_memory_store(config)
    mem = store.load(name)
    if mem is None:
        console.print(f"[red]Error:[/red] Memory '{name}' not found.")
        raise typer.Exit(1)

    # Create updated memory
    updated = Memory(
        name=mem.name,
        description=description if description is not None else mem.description,
        type=type if type is not None else mem.type,
        content=content if content is not None else mem.content,
        links=mem.links,
        created_at=mem.created_at,
    )
    store.save(updated)
    store.rebuild_index()
    console.print(f"[green]✓[/green] Memory '{name}' updated.")


@app.command()
def memory_delete(
    name: str = typer.Argument(..., help="Memory name to delete."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Delete a memory entry."""
    store = _resolve_memory_store(config)
    if store.delete(name):
        store.rebuild_index()
        console.print(f"[green]✓[/green] Memory '{name}' deleted.")
    else:
        console.print(f"[yellow]Memory '{name}' not found.[/yellow]")


@app.command()
def memory_list(
    type: str = typer.Option(None, "--type", "-t", help="Filter by type (user, feedback, project, reference)."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """List all stored memories."""
    store = _resolve_memory_store(config)
    if type:
        memories = store.find_by_type(type)
    else:
        memories = store.list_memories()

    if not memories:
        console.print("[dim]No memories stored yet.[/dim]")
        return

    table = Table(title=f"Memories ({len(memories)})")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description")
    table.add_column("Links")

    for mem in memories:
        links_str = ", ".join(mem.links) if mem.links else ""
        table.add_row(mem.name, mem.type, mem.description[:60] if mem.description else "", links_str)

    console.print(table)


@app.command()
def memory_search(
    query: str = typer.Argument(..., help="Search query."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Search memories by keyword."""
    store = _resolve_memory_store(config)
    results = store.search(query)

    if not results:
        console.print(f"[dim]No memories matching '{query}'.[/dim]")
        return

    table = Table(title=f"Search results for '{query}' ({len(results)})")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Content preview")

    for mem in results:
        preview = mem.content[:100].replace("\n", " ")
        table.add_row(mem.name, mem.type, preview)

    console.print(table)


@app.command()
def memory_show(
    name: str = typer.Argument(..., help="Memory name to display."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Show full content of a memory."""
    store = _resolve_memory_store(config)
    mem = store.load(name)
    if mem is None:
        console.print(f"[red]Error:[/red] Memory '{name}' not found.")
        raise typer.Exit(1)

    console.print(f"[bold cyan]{mem.name}[/bold cyan]")
    console.print(f"  Type: {mem.type}")
    console.print(f"  Description: {mem.description}")
    if mem.links:
        console.print(f"  Links: {', '.join(mem.links)}")
    console.print(f"  Created: {mem.created_at.isoformat()}")
    console.print(f"  Updated: {mem.updated_at.isoformat()}")
    console.print()
    console.print(mem.content)


@app.command()
def memory_rebuild(
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Rebuild the MEMORY.md index."""
    store = _resolve_memory_store(config)
    content = store.rebuild_index()
    console.print(f"[green]✓[/green] Index rebuilt at {store.index_path}")
    console.print(f"   {len(store.list_memories())} memories indexed.")


@app.command()
def memory_import(
    source: str = typer.Argument(..., help="Source directory with memory files."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Import memories from an external directory."""
    store = _resolve_memory_store(config)
    count = store.import_memories(source)
    if count > 0:
        store.rebuild_index()
    console.print(f"[green]✓[/green] Imported {count} memories.")


@app.command()
def memory_stats(
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Show memory store statistics."""
    store = _resolve_memory_store(config)
    stats = store.stats()

    table = Table(title="Memory Store Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")

    table.add_row("Total memories", str(stats["total"]))
    for type_, count in stats["by_type"].items():
        table.add_row(f"  Type: {type_}", str(count))
    table.add_row("Total [[links]]", str(stats["total_links"]))
    table.add_row("Orphan links", str(stats["orphan_links"]))
    table.add_row("Oldest", stats["oldest"] or "-")
    table.add_row("Newest", stats["newest"] or "-")

    size_kb = stats["total_size_bytes"] / 1024
    if size_kb > 1024:
        table.add_row("Disk size", f"{size_kb / 1024:.1f} MB")
    else:
        table.add_row("Disk size", f"{size_kb:.1f} KB")

    console.print(table)
    console.print(f"\nMemory directory: {store.memory_dir}")


@app.command()
def memory_related(
    name: str = typer.Argument(..., help="Memory name to find relations for."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Show memories related to a given memory via [[links]]."""
    store = _resolve_memory_store(config)
    related = store.find_related(name)

    if not related:
        console.print(f"[dim]No related memories found for '{name}'.[/dim]")
        return

    # Show the memory itself first
    mem = store.load(name)
    if mem:
        console.print(f"[bold cyan]{mem.name}[/bold cyan]")
        console.print(f"  Type: {mem.type}  |  Links: {', '.join(mem.links) if mem.links else '(none)'}")
        console.print()

    table = Table(title=f"Related memories ({len(related)})")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description")
    table.add_column("Links")

    for mem in related:
        links_str = ", ".join(mem.links) if mem.links else ""
        table.add_row(mem.name, mem.type, mem.description[:60] if mem.description else "", links_str)

    console.print(table)


@app.command()
def memory_consolidate(
    dry_run: bool = typer.Option(True, "--dry-run", "-n", help="Preview merges without applying."),
    apply: bool = typer.Option(False, "--apply", "-a", help="Actually apply consolidation (overrides --dry-run)."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Merge duplicate or highly similar memories."""
    store = _resolve_memory_store(config)
    result = store.consolidate(dry_run=not apply and not dry_run is False)

    if not result["details"]:
        console.print("[green]✓[/green] No duplicate memories found.")
        return

    table = Table(
        title=f"Consolidation {'Preview' if result['dry_run'] else 'Result'} "
              f"({result['merged']} merged)"
    )
    table.add_column("Kept", style="green")
    table.add_column("Removed", style="red")
    table.add_column("Merged Description")

    for d in result["details"]:
        table.add_row(d["kept"], d["removed"], d["description"][:80])

    console.print(table)

    if result["dry_run"]:
        console.print("\n[yellow]Dry run mode[/yellow] — Use --apply to actually merge.")
    else:
        console.print(f"\n[green]✓[/green] {result['merged']} duplicate(s) merged.")
        store.rebuild_index()


@app.command()
def memory_export(
    output_dir: str = typer.Argument(..., help="Output directory for exported memories."),
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Export all memories as individual markdown files to a directory."""
    store = _resolve_memory_store(config)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    memories = store.list_memories()
    if not memories:
        console.print("[dim]No memories to export.[/dim]")
        return

    count = 0
    for mem in memories:
        file_path = output / f"{mem.name}.md"
        file_path.write_text(mem.to_frontmatter(), encoding="utf-8")
        count += 1

    console.print(f"[green]✓[/green] Exported {count} memories to {output}")


@app.command()
def memory_link_graph(
    config: str = typer.Option("", "--config", "-c", help="Path to config file."),
) -> None:
    """Show the [[link]] graph between memories."""
    store = _resolve_memory_store(config)
    graph = store.get_link_graph()

    if not graph:
        console.print("[dim]No [[links]] between memories.[/dim]")
        return

    table = Table(title="Memory Link Graph")
    table.add_column("Memory", style="cyan")
    table.add_column("Links To")

    for name, links in sorted(graph.items()):
        links_str = ", ".join(links)
        table.add_row(name, links_str)

    console.print(table)
    console.print(f"\nTotal: {len(graph)} memories with links")


if __name__ == "__main__":
    app()
