"""
shard_inference.device_a.server
FastAPI shard-server for Device A.

Endpoints
─────────
  GET  /health              liveness probe
  GET  /manifest            returns manifest.json as JSON
  GET  /shards/{filename}   streams a single .shard file
  POST /metrics             Device B posts InferenceMetrics here
  GET  /results             all received metrics (JSONL → list)
  GET  /results/latest      most recent metric entry

Usage (CLI):
  shard-server ./shards_dir --port 8000

nginx tip:
  Point nginx as a reverse-proxy to uvicorn (port 8000) and set
  client_max_body_size 0 for large shard uploads if needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from rich.console import Console
from rich.table import Table
from rich import box

cli = typer.Typer(name="shard-server", add_completion=False)
_console = Console(highlight=False)


# ── App factory ──────────────────────────────────────────────────────────────

def create_app(shard_dir: Path, metrics_log: Path) -> FastAPI:
    app = FastAPI(title="shard-bridge / Device A", version="0.1.0")

    # ── health ────────────────────────────────────────────────────────────────
    @app.get("/health")
    def health_check():
        return {
            "status": "ok",
            "shard_dir": str(shard_dir),
            "manifest_present": (shard_dir / "manifest.json").exists(),
        }

    # ── manifest ──────────────────────────────────────────────────────────────
    @app.get("/manifest")
    def get_manifest():
        p = shard_dir / "manifest.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="manifest.json not found")
        return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))

    # ── individual shard ─────────────────────────────────────────────────────
    @app.get("/shards/{filename}")
    def get_shard(filename: str):
        # Reject path-traversal attempts
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        if not filename.endswith(".shard"):
            raise HTTPException(status_code=400, detail="Only .shard files served")
        p = shard_dir / filename
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Not found: {filename}")
        return FileResponse(
            path=str(p),
            media_type="application/octet-stream",
            filename=filename,
        )

    # ── receive metrics ───────────────────────────────────────────────────────
    @app.post("/metrics")
    async def post_metrics(request: Request):
        payload: dict[str, Any] = await request.json()
        with open(metrics_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        count = _count_lines(metrics_log)
        return {"status": "received", "entries": count}

    # ── query results ─────────────────────────────────────────────────────────
    @app.get("/results")
    def get_all_results():
        if not metrics_log.exists():
            return []
        return [
            json.loads(line)
            for line in metrics_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @app.get("/results/latest")
    def get_latest_result():
        if not metrics_log.exists():
            raise HTTPException(status_code=404, detail="No metrics recorded yet")
        lines = [
            l for l in metrics_log.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        if not lines:
            raise HTTPException(status_code=404, detail="No metrics recorded yet")
        return json.loads(lines[-1])

    return app


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


# ── CLI entry-point ──────────────────────────────────────────────────────────

@cli.command()
def run_server(
    shard_dir: Path = typer.Argument(
        ...,
        help="Directory that contains manifest.json and .shard files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Port to listen on"),
    metrics_log: Path = typer.Option(
        Path("metrics.jsonl"), help="File to append metrics received from Device B"
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (dev only)"),
):
    """
    Start the Device A shard server.

    Example:
      shard-server ./my_shards --port 8000
    """
    shard_dir = shard_dir.resolve()

    _console.print()
    _console.print("  [bold]◆ shard-server[/bold]  [dim]v0.1.0[/dim]")
    _console.print()

    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        _console.print(
            f"  [yellow]⚠[/yellow]  manifest.json not found in [cyan]{shard_dir}[/cyan]"
        )
    else:
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            shard_count = m.get("tensor_count", "?")
            source     = m.get("source_model", "?")
            total_gb   = m.get("total_bytes", 0) / 1e9
            _console.print(
                f"  [green]✓[/green]  [bold]{source}[/bold]"
                f"  [dim]{shard_count} shards · {total_gb:.2f} GB[/dim]"
            )
        except Exception:
            _console.print("  [yellow]⚠[/yellow]  could not parse manifest.json")

    _console.print(f"  [dim]>[/dim]  listening on [cyan]http://{host}:{port}[/cyan]")
    _console.print(f"  [dim]>[/dim]  metrics log  [dim]{metrics_log}[/dim]")
    _console.print()

    app = create_app(shard_dir, metrics_log)
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()