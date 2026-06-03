"""CLI entry point for toolsclaw."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from toolsclaw.config import DEFAULT_CONFIG_FILE, init_config, load_config
from toolsclaw.runner import AgentRunner

app = typer.Typer(
    name="toolsclaw",
    help="Ultra-lightweight tool-calling agent framework.",
    no_args_is_help=True,
)
console = Console()


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


if __name__ == "__main__":
    app()
