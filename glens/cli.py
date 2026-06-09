from pathlib import Path

import typer

from .build_core_datasets import build_core_datasets

app = typer.Typer(help="Glens: GPCR G-alpha coupling utilities.")


@app.callback()
def main() -> None:
    """Glens CLI."""
    pass


@app.command("build")
def build_core(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config."),
) -> None:
    """Build the receptor, generic-number, and label dataset parquet files."""
    summary = build_core_datasets(config)
    typer.echo("Built:")
    for name, path in summary.items():
        typer.echo(f"  {name}: {path}")