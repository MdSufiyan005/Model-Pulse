"""
Device B (Client) CLI — clean TUI.

Delta support (v0.3.0)
  The client maintains an in-memory shard cache keyed by model_id:

      _shard_cache: dict[str, dict[str, bytes]]
          model_id → { shard_name: raw_bytes }

  On a full model_ready:
      • Fetch all shards from /shards/*.shard  (unchanged)
      • Store a deep copy in _shard_cache[model_id]
      • Build ShardBridge, load, infer / benchmark

  On a delta model_ready:
      • Verify the base model is still cached
      • Fetch only changed shards from /shards/delta/{model_id}/*
      • Clone the base shard cache → patch changed entries → store as new model_id
      • Call bridge.apply_delta(patches) — reassembles + reloads in-process
        (no re-download of unchanged shards, no full bridge teardown)
      • Infer / benchmark as usual
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

# Sentinel for disconnect
_DISCONNECT = object()

# Theme
_THEME = Theme(
    {
        "ok":    "green",
        "warn":  "yellow",
        "err":   "bold red",
        "step":  "bold white",
        "val":   "cyan",
        "dim":   "dim white",
        "hdr":   "bold white",
        "trunc": "yellow",
        "delta": "bold magenta",   # new: highlights delta events
    }
)

console       = Console(theme=_THEME, highlight=False)
error_console = Console(theme=_THEME, highlight=False, stderr=True)

app = typer.Typer(name="bridge", add_completion=False)

# Helpers

def _ok(msg: str)    -> None: console.print(f"[ok]✓[/ok] {msg}")
def _step(msg: str)  -> None: console.print(f"[step]> {msg}[/step]")
def _warn(msg: str)  -> None: console.print(f"[warn]⚠ {msg}[/warn]")
def _err(msg: str)   -> None: error_console.print(f"[err]✗ {msg}[/err]")
def _delta(msg: str) -> None: console.print(f"[delta]Δ {msg}[/delta]")
def _rule()          -> None: console.print(Rule(style="dim"))


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
    if total_mb > 0:
        pct = 100.0 * used_mb / total_mb
        return f"{used_mb:.0f} MB / {total_mb:.0f} MB ({pct:.1f}%)"
    return f"{used_mb:.0f} MB"


def _metrics_panel(metrics: InferenceMetrics) -> None:
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

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_row("  Questions",      f"[dim]{results.success_count}/{results.question_count}[/dim]")
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

    if results.question_results:
        console.print()
        console.print("  [hdr]per-question breakdown[/hdr]")
        console.print()

        q_table = Table(
            "q", "tok/s", "tokens", "budget", "latency", "TTFT", "flags",
            box=box.SIMPLE, show_header=True, header_style="bold white",
        )

        for qr in results.question_results:
            flags = ""
            if qr.truncated:
                flags += "[trunc]T[/trunc] "
            if qr.timed_out:
                flags += "[warn]timeout[/warn] "
            if qr.error:
                flags += "[err]err[/err] "
            flags = flags.strip() or "[dim]—[/dim]"

            if qr.timed_out or qr.error:
                q_table.add_row(
                    f"q{qr.index + 1}", "[dim]—[/dim]", "[dim]—[/dim]",
                    str(qr.max_tokens_used), "[dim]—[/dim]", "[dim]—[/dim]", flags,
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


# CLI Command

@app.command()
def run(
    host: str,
    prompt: Optional[str] = None,
    benchmark: bool = typer.Option(False, "--benchmark", "-b", help="Run the built-in benchmark suite"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", "-m"),
    temperature: float = 0.7,
    n_ctx: int = 2048,
    perplexity: bool = typer.Option(False, "--perplexity", "-p"),
):
    asyncio.run(
        _run_ws_async(host, prompt, benchmark, max_tokens, temperature, n_ctx, perplexity)
    )


# Core Logic

def _header() -> None:
    console.print()
    console.print("  [hdr]◆ modelpulse bridge[/hdr]  [dim]v0.3.0[/dim]")
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

    # In-memory shard cache
    # Keyed by model_id.  Each value is a shallow dict[shard_name, bytes].
    # On a full update we store a *copy* so that a subsequent delta can
    # clone from the base without being affected by any in-place mutations
    # that apply_delta() makes to the live bridge's shard_data.
    _shard_cache: dict[str, dict[str, bytes]] = {}

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
                WsMessage.hello(client_id, capabilities={"version": "0.3.0"}).to_json()
            )

            listener_task = asyncio.create_task(_ws_listener(ws, model_queue))

            current_bridge: Optional[ShardBridge]  = None
            current_model_id: str                   = ""

            try:
                while True:
                    model_msg = await model_queue.get()

                    if model_msg is _DISCONNECT:
                        _warn("connection lost — exiting (restart to reconnect)")
                        break

                    model_id      = model_msg.payload.get("model_id", "")
                    update_type   = model_msg.payload.get("update_type", "full")
                    base_model_id = model_msg.payload.get("base_model_id")

                    try:
                        await ws.send(WsMessage.ack(model_msg.msg_id).to_json())
                    except Exception:
                        pass

                    # Already loaded
                    if model_id and model_id == current_model_id and current_bridge:
                        _ok(f"model [dim]{model_id}[/dim] already loaded — resuming")
                        console.print()

                    # Delta update
                    elif update_type == "delta" and base_model_id:
                        current_bridge, current_model_id = await _handle_delta(
                            ws, host,
                            model_id, base_model_id,
                            current_bridge, current_model_id,
                            _shard_cache,
                            n_ctx, compute_perplexity,
                        )
                        if current_bridge is None:
                            continue   # delta failed; keep listening

                    # Full model
                    else:
                        if current_bridge is not None:
                            _rule()
                            _step("new model incoming — unloading current")
                            current_bridge.cleanup()
                            current_bridge   = None
                            current_model_id = ""
                            console.print()

                        current_bridge, current_model_id = await _handle_full(
                            ws, host,
                            model_id,
                            _shard_cache,
                            n_ctx, compute_perplexity,
                        )
                        if current_bridge is None:
                            continue   # full load failed; keep listening

                    _rule()

                    # Inference / benchmark
                    if benchmark:
                        await _do_benchmark(
                            ws, host, current_model_id, current_bridge,
                            max_tokens, temperature, compute_perplexity,
                        )
                    elif prompt is not None:
                        await _do_infer(
                            ws, host, current_model_id, current_bridge,
                            prompt, max_tokens, temperature,
                        )
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


# Full model handler

async def _handle_full(
    ws,
    host: str,
    model_id: str,
    shard_cache: dict[str, dict[str, bytes]],
    n_ctx: int,
    compute_perplexity: bool,
) -> tuple[Optional[ShardBridge], str]:
    """
    Fetch manifest + all shards, build ShardBridge, populate shard_cache.
    Returns (bridge, model_id) on success or (None, "") on failure.
    """
    _step("fetching manifest from server…")
    try:
        async with ShardClient(host, timeout=30.0) as http:
            manifest = await http.fetch_manifest()
    except Exception as exc:
        _err(f"manifest fetch failed: {exc}")
        return None, ""

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
        return None, ""

    # Store a snapshot in the cache so delta updates can clone from it.
    shard_cache[model_id] = dict(shard_data)   # shallow copy is enough

    bridge = ShardBridge(manifest, shard_data, compute_perplexity=compute_perplexity)
    del shard_data

    try:
        load_time = bridge.load(n_ctx=n_ctx, on_status=_step)
    except RuntimeError as exc:
        _err(str(exc))
        bridge.cleanup()
        return None, ""

    _ok(
        f"model loaded  "
        f"[dim]{load_time:.1f} s · RAM +{bridge.ram_delta_mb:.0f} MB[/dim]"
    )
    return bridge, model_id


# Delta handler

async def _handle_delta(
    ws,
    host: str,
    model_id: str,
    base_model_id: str,
    current_bridge: Optional[ShardBridge],
    current_model_id: str,
    shard_cache: dict[str, dict[str, bytes]],
    n_ctx: int,
    compute_perplexity: bool,
) -> tuple[Optional[ShardBridge], str]:
    """
    Fetch only the changed shards and apply them to the live bridge via
    bridge.apply_delta().  If the base is not cached or no bridge exists,
    falls back to a full re-load.

    Returns (bridge, new_model_id) on success or (None, "") on failure.
    """
    _delta(
        f"delta update received  "
        f"[dim]{base_model_id}[/dim] → [dim]{model_id}[/dim]"
    )
    console.print()

    # Guard: we need both the base in cache and an active bridge
    if base_model_id not in shard_cache or current_bridge is None:
        _warn(
            f"base model [dim]{base_model_id}[/dim] not in shard cache "
            f"(cache keys: {list(shard_cache)}) — falling back to full load"
        )
        if current_bridge is not None:
            current_bridge.cleanup()
        return await _handle_full(
            ws, host, model_id, shard_cache, n_ctx, compute_perplexity
        )

    # Fetch delta manifest + changed shards only
    _step(f"fetching delta manifest for [val]{model_id}[/val]…")
    try:
        async with ShardClient(host, timeout=30.0) as http:
            delta_manifest = await http.fetch_delta_manifest(model_id)
    except Exception as exc:
        _err(f"delta manifest fetch failed: {exc} — falling back to full load")
        current_bridge.cleanup()
        return await _handle_full(
            ws, host, model_id, shard_cache, n_ctx, compute_perplexity
        )

    changed_names = list(delta_manifest.get("changed_shards", {}).keys())
    iteration     = delta_manifest.get("iteration", "?")
    _ok(
        f"delta manifest  "
        f"[dim]iteration={iteration} · {len(changed_names)} shard(s) changed[/dim]"
    )
    for name in changed_names:
        console.print(f"  [dim]Δ {name}[/dim]")
    console.print()

    _step(f"downloading {len(changed_names)} updated shard(s)…")
    try:
        async with ShardClient(host, timeout=120.0) as http:
            patches = await http.fetch_delta_shards(model_id, delta_manifest)
    except Exception as exc:
        _err(f"delta shard fetch failed: {exc} — falling back to full load")
        current_bridge.cleanup()
        return await _handle_full(
            ws, host, model_id, shard_cache, n_ctx, compute_perplexity
        )

    _ok(
        f"delta shards downloaded  "
        f"[dim]{sum(len(v) for v in patches.values()) / 1e6:.1f} MB[/dim]"
    )
    console.print()

    # Apply delta to the live bridge (reassembles + reloads in-process)
    _step("applying delta to loaded model…")
    try:
        load_time = await asyncio.to_thread(
            current_bridge.apply_delta,
            patches,
            n_ctx=n_ctx,
            on_status=_step,
        )
    except Exception as exc:
        _err(f"apply_delta failed: {exc} — falling back to full load")
        current_bridge.cleanup()
        return await _handle_full(
            ws, host, model_id, shard_cache, n_ctx, compute_perplexity
        )

    # Update the shard cache: clone base → apply patches → save as new model_id.
    # We clone the base cache entry (not the mutated bridge.shard_data) to keep
    # the base snapshot clean for potential future deltas.
    #
    # NOTE: Normalize patch keys from filenames to tensor names first, so the
    # cache remains consistent with the GGUF tensor names.
    file_to_name = {s["file"]: name for name, s in current_bridge.manifest.shards.items()}
    normalized_patches = {file_to_name.get(k, k): v for k, v in patches.items()}

    new_cache: dict[str, bytes] = dict(shard_cache[base_model_id])
    new_cache.update(normalized_patches)
    shard_cache[model_id] = new_cache

    _ok(
        f"delta applied  "
        f"[dim]{load_time:.1f} s · RAM +{current_bridge.ram_delta_mb:.0f} MB[/dim]"
    )
    console.print()

    return current_bridge, model_id


# Inference

async def _do_infer(
    ws,
    host: str,
    current_model_id: str,
    bridge: ShardBridge,
    prompt: str,
    max_tokens: Optional[int],
    temperature: float,
) -> None:
    effective_tokens = max_tokens if max_tokens is not None else 256
    console.print()
    _step("generating")
    console.print()
    console.print("  ", end="")

    def _on_token(tok: str) -> None:
        console.print(tok.replace("\n", "\n  "), end="", highlight=False)

    try:
        _output, metrics = bridge.infer(
            prompt, max_tokens=effective_tokens, temperature=temperature, on_token=_on_token,
        )
    except Exception as exc:
        console.print()
        _err(f"inference failed: {exc}")
        return

    console.print()
    console.print()
    _ok(
        f"[dim]{metrics.tokens_generated} tokens · "
        f"{metrics.tokens_per_sec:.1f} tok/s · "
        f"TTFT {metrics.time_to_first_tok_s:.3f} s[/dim]"
    )

    if metrics.cpu_temp_c is not None and metrics.cpu_temp_c >= 80.0:
        _warn(f"CPU at {metrics.cpu_temp_c:.0f}°C — thermal throttling may be depressing tok/s")

    await _send_metrics(ws, host, current_model_id, metrics)

    _rule()
    console.print("  [hdr]metrics[/hdr]")
    console.print()
    _metrics_panel(metrics)
    _rule()
    console.print()
    _step("listening for server updates…")
    console.print()


# Benchmark

async def _do_benchmark(
    ws,
    host: str,
    current_model_id: str,
    bridge: ShardBridge,
    max_tokens: Optional[int],
    temperature: float,
    compute_perplexity: bool,
) -> None:
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
            bridge,
            questions=BENCHMARK_QUESTIONS,
            max_tokens=max_tokens,
            temperature=temperature,
            on_progress=_progress,
        )
    except Exception as exc:
        console.print()
        _err(f"benchmark failed: {exc}")
        return

    console.print()

    summary_parts = [
        f"{bench_results.total_tokens_generated} tokens",
        f"{bench_results.avg_tokens_per_sec:.1f} tok/s",
        f"avg TTFT {bench_results.avg_ttft_s:.3f} s",
        f"p95 lat {bench_results.p95_latency_s:.3f} s",
    ]
    if bench_results.truncated_count:
        summary_parts.append(f"[warn]{bench_results.truncated_count} truncated[/warn]")
    if bench_results.thermal_throttle_warning:
        summary_parts.append(f"[warn]thermal {bench_results.cpu_temp_c:.0f}°C[/warn]")
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


# Shard Pull

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