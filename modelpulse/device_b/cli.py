"""
modelpulse.device_b.cli
Device B CLI — Claude Code-inspired clean TUI.

Commands
────────
  modelpulse bridge run    <host>  [options]   Full pipeline: pull → load → infer → report
  modelpulse bridge status <host>              Fetch + display latest metrics from Device A
  modelpulse bridge pull   <host>  [options]   Pull shards only (no inference)
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich import box

from modelpulse.device_b.shard_client import ShardClient
from modelpulse.device_b.bridge import ShardBridge
from modelpulse.shared.models import InferenceMetrics, ShardManifest

# ── Theme ─────────────────────────────────────────────────────────────────────
_THEME = Theme(
    {
        "ok":     "green",
        "warn":   "yellow",
        "err":    "bold red",
        "step":   "bold white",
        "val":    "cyan",
        "dim":    "dim white",
        "hdr":    "bold white",
        "tok":    "white",
    }
)

console  = Console(theme=_THEME, highlight=False)
error_console = Console(theme=_THEME, highlight=False, stderr=True)
app      = typer.Typer(
    name="bridge",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
)


# ── Print helpers ─────────────────────────────────────────────────────────────

def _header() -> None:
    console.print()
    console.print("  [hdr]◆ modelpulse bridge[/hdr]  [dim]v0.1.0[/dim]")
    console.print()


def _ok(msg: str) -> None:
    console.print(f"  [ok]✓[/ok]  {msg}")


def _step(msg: str) -> None:
    console.print(f"  [dim]>[/dim]  [step]{msg}[/step]")


def _warn(msg: str) -> None:
    console.print(f"  [warn]⚠[/warn]  {msg}")


def _err(msg: str) -> None:
    error_console.print(f"  [err]✗[/err]  {msg}")


def _rule() -> None:
    console.print()
    console.print(Rule(style="dim"))
    console.print()


def _metrics_panel(m: InferenceMetrics) -> None:
    """Render the metrics as a clean two-column table."""
    t = Table(
        box=box.SIMPLE,
        show_header=False,
        padding=(0, 2),
        expand=False,
    )
    t.add_column(style="dim", min_width=24)
    t.add_column(style="cyan")

    def _row(key: str, val: str) -> None:
        t.add_row(key, val)

    _row("load_time_s",         f"{m.load_time_s:.2f} s")
    _row("ram_delta_mb",        f"{m.ram_delta_mb:.0f} MB")
    _row("tokens_per_sec",      f"{m.tokens_per_sec:.1f}")
    _row("time_to_first_tok_s", f"{m.time_to_first_tok_s:.3f} s")
    _row("tokens_generated",    str(m.tokens_generated))
    _row("cpu_percent",         f"{m.cpu_percent:.1f} %")
    _row("ram_used_mb",         f"{m.ram_used_mb:.0f} MB")
    _row(
        "cpu_temp_c",
        f"{m.cpu_temp_c:.1f} °C" if m.cpu_temp_c is not None else "n/a",
    )
    _row("device_hw",  m.device_hw  or "n/a")
    _row("os_info",    m.os_info    or "n/a")
    if m.perplexity is not None:
        _row("perplexity", f"{m.perplexity:.2f}")

    console.print(t)


# ── Connect helper ────────────────────────────────────────────────────────────

async def _connect(host: str) -> tuple[ShardClient, ShardManifest]:
    """Verify connectivity and return (client, manifest).  Exits on failure."""
    _step(f"connecting to [val]{host}[/val]")
    client = ShardClient(host, timeout=30.0)

    try:
        rtt = await client.ping()
    except Exception as exc:
        _err(f"cannot reach {host} — {exc}")
        raise typer.Exit(1)

    _ok(f"reachable  [dim]{rtt:.0f} ms[/dim]")

    _step("fetching manifest")
    try:
        manifest = await client.fetch_manifest()
    except Exception as exc:
        await client.aclose()
        _err(f"manifest fetch failed — {exc}")
        raise typer.Exit(1)

    total_gb = manifest.total_bytes / 1e9
    _ok(
        f"[bold]{manifest.source_model}[/bold]  "
        f"[dim]{manifest.tensor_count} tensors · {total_gb:.2f} GB "
        f"· GGUF v{manifest.gguf_version}[/dim]"
    )
    console.print()
    return client, manifest


# ── Shard pull helper ─────────────────────────────────────────────────────────

async def _pull_shards(
    client: ShardClient,
    manifest: ShardManifest,
) -> dict[str, bytes]:
    """Download all shards with a live progress bar. Returns shard_data dict."""
    shard_data: dict[str, bytes] = {}
    pulled_mb  = 0.0
    t0         = time.perf_counter()

    with Progress(
        TextColumn("  "),
        SpinnerColumn(spinner_name="dots", style="dim"),
        TextColumn("[dim]{task.description}[/dim]"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("[dim]·[/dim]"),
        TextColumn("[dim]{task.fields[mb]:.0f} MB[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as prog:
        task: TaskID = prog.add_task(
            "pulling shards",
            total=manifest.tensor_count,
            mb=0.0,
        )

        for name, entry in manifest.shards.items():
            data = await client.fetch_shard(
                entry["file"],
                expected_sha256=entry["sha256"],
            )
            shard_data[name] = data
            pulled_mb += len(data) / 1_048_576
            prog.update(task, advance=1, mb=pulled_mb)

    elapsed = time.perf_counter() - t0
    _ok(
        f"all shards in memory  "
        f"[dim]{pulled_mb:.0f} MB · {elapsed:.1f} s[/dim]"
    )
    console.print()
    return shard_data


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  modelpulse bridge run                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.command()
def run(
    host: str = typer.Argument(..., metavar="HOST", help="Device A base URL  e.g. http://192.168.1.10:8000"),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p",
        help="Prompt to run  (omit for interactive input)"
    ),
    max_tokens: int = typer.Option(256, "--max-tokens", "-n", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Sampling temperature"),
    n_ctx: int = typer.Option(2048, "--ctx", help="Context window size"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip sending metrics to Device A"),
):
    """
    Full pipeline on Device B:
      pull shards → assemble in RAM → load model → run inference → report metrics
    """
    asyncio.run(_run_async(host, prompt, max_tokens, temperature, n_ctx, no_report))


async def _run_async(
    host: str,
    prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    n_ctx: int,
    no_report: bool,
) -> None:
    _header()

    # ── 1. Connect + manifest ─────────────────────────────────────────────────
    client, manifest = await _connect(host)

    # ── 2. RAM budget warning ─────────────────────────────────────────────────
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / 1_048_576
        model_mb     = manifest.total_bytes / 1_048_576
        # Peak RAM ≈ 2× model (assemble + tmpfs) before GC
        if model_mb * 2 > available_mb * 0.9:
            _warn(
                f"low RAM  [dim]model {model_mb:.0f} MB, "
                f"available {available_mb:.0f} MB  — "
                f"peak usage ≈ 2× model during assembly[/dim]"
            )
    except Exception:
        pass

    # ── 3. Pull shards ────────────────────────────────────────────────────────
    shard_data = await _pull_shards(client, manifest)

    # ── 4. Assemble + load ────────────────────────────────────────────────────
    bridge = ShardBridge(manifest, shard_data)
    del shard_data   # drop raw bytes reference early; GC may reclaim soon

    def _on_status(msg: str) -> None:
        _step(msg)

    try:
        load_time = bridge.load(n_ctx=n_ctx, on_status=_on_status)
    except RuntimeError as exc:
        _err(str(exc))
        raise typer.Exit(1)

    _ok(
        f"model loaded  "
        f"[dim]{load_time:.1f} s · RAM +{bridge.ram_delta_mb:.0f} MB[/dim]"
    )
    _rule()

    # ── 5. Prompt ─────────────────────────────────────────────────────────────
    try:
        if prompt is None:
            console.print("  [dim]Prompt[/dim] [bold]›[/bold] ", end="")
            try:
                prompt = input()
            except (EOFError, KeyboardInterrupt):
                console.print()
                bridge.cleanup()
                raise typer.Exit(0)

        if not prompt.strip():
            _warn("empty prompt — exiting")
            bridge.cleanup()
            raise typer.Exit(0)

        console.print()
        _step("generating")
        console.print()
        console.print("  ", end="")   # indent output

        output_parts: list[str] = []

        def _on_token(tok: str) -> None:
            output_parts.append(tok)
            # Indent continuation lines
            console.print(tok.replace("\n", "\n  "), end="", highlight=False)

        output, metrics = bridge.infer(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            on_token=_on_token,
        )

        console.print()   # newline after last token
        console.print()
        _ok(
            f"[dim]{metrics.tokens_generated} tokens · "
            f"{metrics.tokens_per_sec:.1f} tok/s · "
            f"TTFT {metrics.time_to_first_tok_s:.2f} s[/dim]"
        )

    finally:
        bridge.cleanup()

    # ── 6. Metrics display ────────────────────────────────────────────────────
    _rule()
    console.print("  [hdr]metrics[/hdr]")
    console.print()
    _metrics_panel(metrics)

    # ── 7. Report ─────────────────────────────────────────────────────────────
    if not no_report:
        _rule()
        _step(f"sending metrics → [val]{host}/metrics[/val]")
        metrics.server_url = host
        try:
            resp = await client.post_metrics(metrics)
            _ok(f"received  [dim](entries on Device A: {resp.get('entries', '?')})[/dim]")
        except Exception as exc:
            _warn(f"could not send metrics — {exc}")

    await client.aclose()
    console.print()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  modelpulse bridge pull                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.command()
def pull(
    host: str = typer.Argument(..., metavar="HOST", help="Device A base URL"),
    out_dir: Path = typer.Option(
        Path("./shards"),
        "--out", "-o",
        help="Directory to write .shard files + manifest",
    ),
):
    """
    Pull shards from Device A and save to disk (no inference).
    Useful for pre-staging shards on Device B before going offline.
    """
    asyncio.run(_pull_async(host, out_dir))


async def _pull_async(host: str, out_dir: Path) -> None:
    _header()
    out_dir.mkdir(parents=True, exist_ok=True)

    client, manifest = await _connect(host)
    shard_data = await _pull_shards(client, manifest)
    await client.aclose()

    _step(f"writing shards to [val]{out_dir}[/val]")
    import json

    for name, entry in manifest.shards.items():
        (out_dir / entry["file"]).write_bytes(shard_data[name])

    # Write manifest too
    import dataclasses, json as _json
    (out_dir / "manifest.json").write_text(
        _json.dumps(
            {
                "version":            manifest.version,
                "source_model":       manifest.source_model,
                "gguf_version":       manifest.gguf_version,
                "alignment":          manifest.alignment,
                "tensor_count":       manifest.tensor_count,
                "total_bytes":        manifest.total_bytes,
                "gguf_metadata_kvs":  manifest.gguf_metadata_kvs,
                "shards":             manifest.shards,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _ok(
        f"saved {len(shard_data)} shards + manifest  "
        f"[dim]→ {out_dir}[/dim]"
    )
    console.print()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  modelpulse bridge status                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.command()
def status(
    host: str = typer.Argument(..., metavar="HOST", help="Device A base URL"),
    all_results: bool = typer.Option(False, "--all", "-a", help="Show all metric entries"),
):
    """Fetch and display the latest (or all) metrics stored on Device A."""
    asyncio.run(_status_async(host, all_results))


async def _status_async(host: str, all_results: bool) -> None:
    _header()

    import httpx

    endpoint = "/results" if all_results else "/results/latest"
    _step(f"fetching {endpoint} from [val]{host}[/val]")

    try:
        async with httpx.AsyncClient(base_url=host, timeout=10.0) as c:
            resp = await c.get(endpoint)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        _err(f"request failed — {exc}")
        raise typer.Exit(1)

    if all_results:
        if not data:
            _warn("no metric entries found")
        else:
            _ok(f"{len(data)} entries")
            for i, entry in enumerate(data, 1):
                _rule()
                console.print(f"  [dim]entry {i}[/dim]")
                console.print()
                _metrics_panel(InferenceMetrics.from_dict(entry))
    else:
        _ok("latest entry")
        _rule()
        _metrics_panel(InferenceMetrics.from_dict(data))

    console.print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()