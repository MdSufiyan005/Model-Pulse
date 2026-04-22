from __future__ import annotations

import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from rich.console import Console

cli = typer.Typer(name="server", add_completion=False)
_console = Console(highlight=False)


def create_app(
    shard_dir: Path,
    metrics_log: Path,
    *,
    port: int = 8000,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="modelpulse / Device A", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health_check():
        return {
            "status": "ok",
            "shard_dir": str(shard_dir),
            "manifest_present": (shard_dir / "manifest.json").exists(),
        }

    @app.get("/manifest")
    def get_manifest():
        p = shard_dir / "manifest.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="manifest.json not found")
        return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))

    @app.get("/shards/{filename}")
    def get_shard(filename: str):
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

    @app.post("/metrics")
    async def post_metrics(request: Request):
        payload: dict[str, Any] = await request.json()
        with open(metrics_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return {"status": "received"}

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
            l for l in metrics_log.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if not lines:
            raise HTTPException(status_code=404, detail="No metrics recorded yet")
        return json.loads(lines[-1])

    return app


@cli.command()
def run(
    shard_dir: Path = typer.Argument(
        ...,
        help="Directory that contains manifest.json and .shard files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000, help="Port to listen on"),
    metrics_log: Path = typer.Option(
        Path("metrics.jsonl"),
        help="File to append metrics received from Device B",
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (dev only)"),
):
    shard_dir = shard_dir.resolve()

    app = create_app(
        shard_dir,
        metrics_log,
        port=port,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
