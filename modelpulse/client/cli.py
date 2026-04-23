"""
modelpulse.client.cli
Device B CLI — Claude Code-inspired clean TUI.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Optional

import typer
import websockets
import websockets.exceptions
from rich.console import Console
from rich.table import Table
from rich.theme import Theme
from rich import box
from rich.rule import Rule

from modelpulse.client.shard_client import ShardClient
from modelpulse.client.bridge import ShardBridge
from modelpulse.client.benchmarks import (
    BenchmarkResults,
    run_benchmark,
    BENCHMARK_QUESTIONS,
)
from modelpulse.shared.models import InferenceMetrics, ShardManifest
from modelpulse.shared.ws_protocol import MsgType, WsMessage

# ── Sentinel for disconnect ───────────────────────────────────────────────────
_DISCONNECT = object()

# ── Theme ─────────────────────────────────────────────────────────────────────
_THEME = Theme(
    {
        "ok":   "green",
        "warn": "yellow",
        "err":  "bold red",
        "step": "bold white",
        "val":  "cyan",
        "dim":  "dim white",
        "hdr":  "bold white",
        "trunc": "yellow",
    }
)

console       = Console(theme=_THEME, highlight=False)
error_console = Console(theme=_THEME, highlight=False, stderr=True)

app = typer.Typer(name="bridge", add_completion=False)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(msg: str)   -> None: console.print(f"[ok]✓[/ok] {msg}")
def _step(msg: str) -> None: console.print(f"[step]> {msg}[/step]")
def _warn(msg: str) -> None: console.print(f"[warn]⚠ {msg}[/warn]")
def _err(msg: str)  -> None: error_console.print(f"[err]✗ {msg}[/err]")
def _rule()         -> None: console.print(Rule(style="dim"))


def _ram_warn(manifest: ShardManifest) -> None:
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / 1e9
        model_gb     = manifest.total_bytes / 1e9
        if model_gb > available_gb * 0.8:
            _warn(
                f"Model {model_gb:.1f} GB may exceed available RAM "
                f"({available_gb:.1f} GB available)"
            )
    except Exception:
        pass


def _ram_pct(used_mb: float, total_mb: float) -> str:
    """Format RAM as 'used MB / total MB (pct%)'."""
    if total_mb > 0:
        pct = 100.0 * used_mb / total_mb
        return f"{used_mb:.0f} MB / {total_mb:.0f} MB ({pct:.1f}%)"
    return f"{used_mb:.0f} MB"


def _metrics_panel(metrics: InferenceMetrics) -> None:
    """Display single-inference metrics."""
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_row("  Load time",    f"[dim]{metrics.load_time_s:.2f} s[/dim]")
    table.add_row("  TTFT",         f"[dim]{metrics.time_to_first_tok_s:.3f} s[/dim]")
    table.add_row("  Tokens/sec",   f"[dim]{metrics.tokens_per_sec:.1f} tok/s[/dim]")
    table.add_row("  Total tokens", f"[dim]{metrics.tokens_generated}[/dim]")
    table.add_row("  RAM Δ",        f"[dim]{metrics.ram_delta_mb:.0f} MB[/dim]")
    table.add_row("  RAM used",     f"[dim]{metrics.ram_used_mb:.0f} MB[/dim]")
    if metrics.cpu_temp_c is not None:
        temp_str = f"[dim]{metrics.cpu_temp_c:.1f}°C[/dim]"
        if metrics.cpu_temp_c >= 80.0:
            temp_str += " [warn]⚠ thermal throttle risk[/warn]"
        table.add_row("  CPU temp", temp_str)
    if metrics.cpu_percent:
        table.add_row("  CPU %", f"[dim]{metrics.cpu_percent:.1f}%[/dim]")
    console.print(table)


def _benchmark_panel(results: BenchmarkResults) -> None:
    """
    Display benchmark aggregate metrics + per-question breakdown.

    Shows RAM utilisation as used/total (%), thermal throttle warning,
    truncation count, and a clean tok/s figure that excludes truncated
    questions so operators can see uncontaminated decode throughput.
    """
    # ── Warnings ──────────────────────────────────────────────────────────────
    if results.thermal_throttle_warning and results.cpu_temp_c is not None:
        _warn(
            f"CPU at {results.cpu_temp_c:.0f}°C — thermal throttling likely. "
            "tok/s figures may be lower than the model's true capability."
        )
        console.print()

    if results.truncated_count:
        _warn(
            f"{results.truncated_count} question(s) hit the token budget and "
            "were truncated. Their latency is inflated; see [trunc]T[/trunc] below."
        )
        console.print()

    # ── Aggregate summary ─────────────────────────────────────────────────────
    table = Table(show_header=False, box=box.SIMPLE)

    table.add_row(
        "  Questions",
        f"[dim]{results.success_count}/{results.question_count}[/dim]",
    )
    table.add_row("  Total tokens",   f"[dim]{results.total_tokens_generated}[/dim]")
    table.add_row("  Avg tok/q",      f"[dim]{results.avg_tokens_per_question:.1f}[/dim]")
    table.add_row("  Agg. tok/s",     f"[dim]{results.avg_tokens_per_sec:.1f} tok/s[/dim]")

    if results.avg_tokens_per_sec_clean is not None:
        label = "  Clean tok/s"
        if results.truncated_count:
            label += " (excl. truncated)"
        table.add_row(label, f"[dim]{results.avg_tokens_per_sec_clean:.1f} tok/s[/dim]")

    table.add_row("  Avg TTFT",       f"[dim]{results.avg_ttft_s:.3f} s[/dim]")
    table.add_row("  Min TTFT",       f"[dim]{results.min_ttft_s:.3f} s[/dim]")
    table.add_row("  Max TTFT",       f"[dim]{results.max_ttft_s:.3f} s[/dim]")
    table.add_row("  Avg latency",    f"[dim]{results.avg_latency_s:.3f} s[/dim]")
    table.add_row("  Median latency", f"[dim]{results.median_latency_s:.3f} s[/dim]")
    table.add_row("  p95 latency",    f"[dim]{results.p95_latency_s:.3f} s[/dim]")
    table.add_row("  RAM Δ (load)",   f"[dim]{results.ram_delta_mb:.0f} MB[/dim]")

    ram_str = _ram_pct(results.ram_used_mb, results.ram_total_mb)
    table.add_row("  RAM used",       f"[dim]{ram_str}[/dim]")

    if results.cpu_temp_c is not None:
        temp_str = f"[dim]{results.cpu_temp_c:.1f}°C[/dim]"
        if results.thermal_throttle_warning:
            temp_str += " [warn]⚠[/warn]"
        table.add_row("  CPU temp", temp_str)

    if results.avg_cpu_percent:
        table.add_row("  Avg CPU %", f"[dim]{results.avg_cpu_percent:.1f}%[/dim]")

    if results.perplexity is not None:
        table.add_row("  Perplexity", f"[dim]{results.perplexity:.2f}[/dim]")

    table.add_row("  Warmup time",  f"[dim]{results.warmup_time_s:.2f} s (excl.)[/dim]")
    table.add_row("  Inference",    f"[dim]{results.inference_time_s:.1f} s[/dim]")
    table.add_row("  Total time",   f"[dim]{results.total_time_s:.1f} s[/dim]")

    console.print(table)

    # ── Per-question breakdown ────────────────────────────────────────────────
    if results.question_results:
        console.print()
        console.print("  [hdr]per-question breakdown[/hdr]")
        console.print()

        q_table = Table(
            "q", "tok/s", "tokens", "budget", "latency", "TTFT", "flags",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold white",
        )

        for qr in results.question_results:
            flags = ""
            if qr.truncated:
                flags += "[trunc]T[/trunc] "   # hit token budget
            if qr.timed_out:
                flags += "[warn]timeout[/warn] "
            if qr.error:
                flags += "[err]err[/err] "
            flags = flags.strip() or "[dim]—[/dim]"

            if qr.timed_out or qr.error:
                q_table.add_row(
                    f"q{qr.index + 1}",
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    str(qr.max_tokens_used),
                    "[dim]—[/dim]",
                    "[dim]—[/dim]",
                    flags,
                )
            else:
                q_table.add_row(
                    f"q{qr.index + 1}",
                    f"{qr.tokens_per_sec:.1f}",
                    str(qr.tokens_generated),
                    str(qr.max_tokens_used),
                    f"{qr.latency_s:.3f} s",
                    f"{qr.time_to_first_tok_s:.3f} s",
                    flags,
                )

        console.print(q_table)

        # Legend
        if results.truncated_count:
            console.print(
                "  [dim][trunc]T[/trunc] = hit token budget (response may be incomplete)[/dim]"
            )


def _to_ws_url(http_url: str) -> str:
    http_url = http_url.rstrip("/")
    if http_url.startswith("https://"):
        return http_url.replace("https://", "wss://", 1) + "/ws"
    if http_url.startswith("http://"):
        return http_url.replace("http://", "ws://", 1) + "/ws"
    if http_url.startswith("ws://") or http_url.startswith("wss://"):
        return http_url + "/ws"
    return "ws://" + http_url + "/ws"


async def _ws_listener(ws, model_queue: asyncio.Queue) -> None:
    try:
        async for raw in ws:
            try:
                msg = WsMessage.from_json(raw)
            except Exception:
                continue

            if msg.type == MsgType.MODEL_READY:
                await model_queue.put(msg)
            elif msg.type == MsgType.PING:
                try:
                    await ws.send(WsMessage.pong(msg.payload.get("ts", 0.0)).to_json())
                except Exception:
                    pass
            elif msg.type == MsgType.ACK:
                pass
            elif msg.type == MsgType.ERROR:
                _warn(f"server: {msg.payload.get('detail', '?')}")
            elif msg.type == MsgType.BYE:
                _step("server closed the session")
                break

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as exc:
        _warn(f"listener error: {exc}")
    finally:
        await model_queue.put(_DISCONNECT)


async def _send_metrics(
    ws,
    host: str,
    current_model_id: str,
    metrics_to_send: InferenceMetrics,
) -> None:
    metrics_to_send.server_url = host
    try:
        async with ShardClient(host, timeout=10.0) as http:
            await http.post_metrics(metrics_to_send)
        return
    except Exception:
        pass
    try:
        await ws.send(
            WsMessage.metrics(
                metrics_to_send.to_dict(),
                model_id=current_model_id,
            ).to_json()
        )
    except Exception as exc:
        _warn(f"could not send metrics: {exc}")


# ── CLI Command ───────────────────────────────────────────────────────────────

@app.command()
def run(
    host: str,
    prompt: Optional[str] = None,
    benchmark: bool = typer.Option(
        False, "--benchmark", "-b",
        help="Run the built-in benchmark suite",
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", "-m",
        help=(
            "Token budget per question. When omitted the suite uses "
            "per-question defaults (recommended)."
        ),
    ),
    temperature: float = 0.7,
    n_ctx: int = 2048,
    perplexity: bool = typer.Option(
        False, "--perplexity", "-p",
        help="Compute perplexity during benchmark (requires logits_all=True)",
    ),
):
    asyncio.run(
        _run_ws_async(
            host, prompt, benchmark, max_tokens, temperature, n_ctx, perplexity
        )
    )


# ── Core Logic ────────────────────────────────────────────────────────────────

def _header() -> None:
    console.print()
    console.print("  [hdr]◆ modelpulse bridge[/hdr]  [dim]v0.2.0[/dim]")
    console.print()


async def _run_ws_async(
    host: str,
    prompt: Optional[str],
    benchmark: bool,
    max_tokens: Optional[int],
    temperature: float,
    n_ctx: int,
    compute_perplexity: bool,
) -> None:
    _header()

    ws_url    = _to_ws_url(host)
    client_id = f"bridge-{os.getpid()}"
    model_queue: asyncio.Queue = asyncio.Queue()

    _step(f"connecting to [val]{host}[/val]")
    try:
        async with websockets.connect(
            ws_url,
            ping_interval=None,
            close_timeout=5,
            max_size=2 * 1024 * 1024,
        ) as ws:
            _ok(f"WebSocket connected  [dim]{client_id}[/dim]")
            console.print()
            _step("waiting for model from server…")
            console.print()

            await ws.send(
                WsMessage.hello(client_id, capabilities={"version": "0.2.0"}).to_json()
            )

            listener_task = asyncio.create_task(_ws_listener(ws, model_queue))

            current_bridge: Optional[ShardBridge] = None
            current_model_id: str = ""

            try:
                while True:
                    model_msg = await model_queue.get()

                    if model_msg is _DISCONNECT:
                        _warn("connection lost — exiting (restart to reconnect)")
                        break

                    model_id = model_msg.payload.get("model_id", "")

                    try:
                        await ws.send(WsMessage.ack(model_msg.msg_id).to_json())
                    except Exception:
                        pass

                    if model_id and model_id == current_model_id and current_bridge:
                        _ok(f"model [dim]{model_id}[/dim] already loaded — resuming")
                        console.print()
                    else:
                        if current_bridge is not None:
                            _rule()
                            _step("new model incoming — unloading current")
                            current_bridge.cleanup()
                            current_bridge = None
                            current_model_id = ""
                            console.print()

                        _step("fetching manifest from server…")
                        try:
                            async with ShardClient(host, timeout=30.0) as http:
                                manifest = await http.fetch_manifest()
                        except Exception as exc:
                            _err(f"manifest fetch failed: {exc}")
                            continue

                        total_gb = manifest.total_bytes / 1e9
                        _ok(
                            f"[bold]{manifest.source_model}[/bold]  "
                            f"[dim]{manifest.tensor_count} tensors · {total_gb:.2f} GB "
                            f"· GGUF v{manifest.gguf_version}[/dim]"
                        )
                        console.print()
                        _ram_warn(manifest)

                        try:
                            async with ShardClient(host, timeout=120.0) as http:
                                shard_data = await _pull_shards(http, manifest)
                        except Exception as exc:
                            _err(f"shard download failed: {exc}")
                            continue

                        current_bridge = ShardBridge(
                            manifest,
                            shard_data,
                            compute_perplexity=compute_perplexity,
                        )
                        del shard_data

                        try:
                            load_time = current_bridge.load(n_ctx=n_ctx, on_status=_step)
                        except RuntimeError as exc:
                            _err(str(exc))
                            current_bridge.cleanup()
                            current_bridge = None
                            continue

                        _ok(
                            f"model loaded  "
                            f"[dim]{load_time:.1f} s · "
                            f"RAM +{current_bridge.ram_delta_mb:.0f} MB[/dim]"
                        )
                        current_model_id = model_id

                    _rule()

                    # ── Benchmark ─────────────────────────────────────────────
                    if benchmark:
                        console.print()
                        budget_note = (
                            f"max_tokens={max_tokens} (override)"
                            if max_tokens is not None
                            else "per-question budgets"
                        )
                        _step(
                            f"running benchmark suite "
                            f"({len(BENCHMARK_QUESTIONS)} questions · {budget_note})"
                        )
                        console.print()

                        def _progress(cur: int, total: int, question: str) -> None:
                            short_q = (question[:60] + "…") if len(question) > 60 else question
                            _step(f"[{cur}/{total}] {short_q}")

                        try:
                            bench_results, _all_metrics = await run_benchmark(
                                current_bridge,
                                questions=BENCHMARK_QUESTIONS,
                                max_tokens=max_tokens,   # None → per-question defaults
                                temperature=temperature,
                                on_progress=_progress,
                            )
                        except Exception as exc:
                            console.print()
                            _err(f"benchmark failed: {exc}")
                            continue

                        console.print()

                        # One-line summary with key signals
                        summary_parts = [
                            f"{bench_results.total_tokens_generated} tokens",
                            f"{bench_results.avg_tokens_per_sec:.1f} tok/s",
                            f"avg TTFT {bench_results.avg_ttft_s:.3f} s",
                            f"p95 lat {bench_results.p95_latency_s:.3f} s",
                        ]
                        if bench_results.truncated_count:
                            summary_parts.append(
                                f"[warn]{bench_results.truncated_count} truncated[/warn]"
                            )
                        if bench_results.thermal_throttle_warning:
                            summary_parts.append(
                                f"[warn]thermal {bench_results.cpu_temp_c:.0f}°C[/warn]"
                            )
                        _ok("[dim]" + " · ".join(summary_parts) + "[/dim]")

                        metrics_to_send = bench_results.to_inference_metrics()
                        await _send_metrics(ws, host, current_model_id, metrics_to_send)

                        _rule()
                        console.print("  [hdr]benchmark metrics[/hdr]")
                        console.print()
                        _benchmark_panel(bench_results)
                        _rule()
                        console.print()

                        _step("listening for server updates…")
                        console.print()
                        continue

                    # ── Single prompt ─────────────────────────────────────────
                    elif prompt is not None:
                        effective_tokens = max_tokens if max_tokens is not None else 256
                        console.print()
                        _step("generating")
                        console.print()
                        console.print("  ", end="")

                        def _on_token(tok: str) -> None:
                            console.print(
                                tok.replace("\n", "\n  "), end="", highlight=False
                            )

                        try:
                            _output, metrics = current_bridge.infer(
                                prompt,
                                max_tokens=effective_tokens,
                                temperature=temperature,
                                on_token=_on_token,
                            )
                        except Exception as exc:
                            console.print()
                            _err(f"inference failed: {exc}")
                            continue

                        console.print()
                        console.print()
                        _ok(
                            f"[dim]{metrics.tokens_generated} tokens · "
                            f"{metrics.tokens_per_sec:.1f} tok/s · "
                            f"TTFT {metrics.time_to_first_tok_s:.3f} s[/dim]"
                        )

                        if (
                            metrics.cpu_temp_c is not None
                            and metrics.cpu_temp_c >= 80.0
                        ):
                            _warn(
                                f"CPU at {metrics.cpu_temp_c:.0f}°C — "
                                "thermal throttling may be depressing tok/s"
                            )

                        await _send_metrics(ws, host, current_model_id, metrics)

                        _rule()
                        console.print("  [hdr]metrics[/hdr]")
                        console.print()
                        _metrics_panel(metrics)
                        _rule()
                        console.print()

                        _step("listening for server updates…")
                        console.print()
                        continue

                    else:
                        _step("listening for server updates…")
                        console.print()

            except KeyboardInterrupt:
                console.print()

            finally:
                listener_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await listener_task

                if current_bridge is not None:
                    current_bridge.cleanup()

                try:
                    await ws.send(WsMessage.bye("bridge shutdown").to_json())
                except Exception:
                    pass

    except Exception as exc:
        _err(f"cannot connect — {exc}")
        raise typer.Exit(1)

    console.print()


# ── Shard Pull ────────────────────────────────────────────────────────────────

async def _pull_shards(
    client: ShardClient,
    manifest: ShardManifest,
) -> dict[str, bytes]:
    data: dict[str, bytes] = {}
    for name, entry in manifest.shards.items():
        data[name] = await client.fetch_shard(entry["file"])
    return data


def main() -> None:
    app()


if __name__ == "__main__":
    main()