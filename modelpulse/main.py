import typer
from modelpulse.device_b import cli as bridge_cli
from modelpulse.device_a import server as server_cli

app = typer.Typer(
    name="modelpulse",
    help="Unified CLI for end to end partial-weight transfer pipeline.",
    add_completion=False,
    no_args_is_help=True,
)

# Add Device B (Bridge) subcommands under 'bridge'
app.add_typer(bridge_cli.app, name="bridge")

# Add Device A (Server) subcommands under 'server'
app.add_typer(server_cli.cli, name="server")

def main():
    app()

if __name__ == "__main__":
    main()
