"""
Rich-formatted logging utilities for DeltaStream.
All console output goes through this module for consistent styling.
"""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

# ──────────────────────────────────────────────────────────────────────────────
# Console
# ──────────────────────────────────────────────────────────────────────────────

_THEME = Theme(
    {
        "info": "bold cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "muted": "dim white",
        "step": "bold blue",
    }
)

console = Console(theme=_THEME)


# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────


def log_info(msg: str) -> None:
    console.print(f"[info]ℹ[/info]  {msg}")


def log_success(msg: str) -> None:
    console.print(f"[success]✔[/success]  {msg}")


def log_warning(msg: str) -> None:
    console.print(f"[warning]⚠[/warning]  {msg}")


def log_error(msg: str) -> None:
    console.print(f"[error]✘[/error]  {msg}")


def log_step(step: str, detail: str = "") -> None:
    if detail:
        console.print(f"[step]▶[/step]  [step]{step}[/step]  [muted]{detail}[/muted]")
    else:
        console.print(f"[step]▶[/step]  [step]{step}[/step]")


def log_header(title: str) -> None:
    width = console.width or 80
    console.rule(f"[highlight]{title}[/highlight]", style="magenta")


# ──────────────────────────────────────────────────────────────────────────────
# Progress bars
# ──────────────────────────────────────────────────────────────────────────────


def make_progress() -> Progress:
    """Return a Rich Progress instance styled for DeltaStream."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
