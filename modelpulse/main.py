import typer
from importlib.metadata import version as pkg_version, PackageNotFoundError

from modelpulse.client import cli as bridge_cli
from modelpulse.server import server as server_cli


def get_version() -> str:
    try:
        return pkg_version("modelpulse")
    except PackageNotFoundError:
        return "dev"


app = typer.Typer(
    name="modelpulse",
    help="Unified CLI for end to end partial-weight transfer pipeline.",
    add_completion=False,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def _callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the version and exit.",
        is_eager=True,
    ),
):
    if version:
        typer.echo(get_version())
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


app.add_typer(bridge_cli.app, name="bridge")
app.add_typer(server_cli.cli, name="server")


def main() -> None:
    app()


if __name__ == "__main__":
    main()