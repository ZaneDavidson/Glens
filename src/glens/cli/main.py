import typer

from .. import __version__

app = typer.Typer(
    help=(
        """
        Glens is an open pipeline for predicting signaling propensity of G-protein coupled receptors from sequence data.
        """
    ),
    no_args_is_help=True,
)
# Nest tools inside main CLI entry point here:


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


if __name__ == "__main__":
    app()
