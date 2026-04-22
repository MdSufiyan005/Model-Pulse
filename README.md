# modelpulse

Shard-aware LLM inference bridge for edge devices (RPi, Jetson Nano, x86 Linux).

Device A shares model shards → Device B loads and runs inference with **zero GGUF reconstruction on disk**.

```
Device A                                  Device B
────────────────────                      ───────────────────────────────────
shard-server ./shards                     shard-bridge run http://192.168.1.10:8000
  │                                         │
  ├── GET /manifest  ◄────────────────────  │  1. fetch manifest
  ├── GET /shards/*  ◄────────────────────  │  2. pull all shards (streaming)
  │                                         │  3. assemble GGUF in RAM → /dev/shm
  │                                         │  4. llama.cpp loads from /dev/shm
  │                                         │  5. run inference, stream tokens
  └── POST /metrics  ◄────────────────────  │  6. send collected metrics
```

---

## Install

```bash
# On Device A (the model server)
pip install "modelpulse[server]"

# On Device B (the inference client)
pip install "modelpulse[client]"

# Both on one machine (dev / testing)
pip install "modelpulse[all]"
```

> **llama-cpp-python on ARM:** Follow the [official build guide](https://github.com/abetlen/llama-cpp-python)
> for RPi / Jetson.  Pre-built wheels exist for common targets.

---

## Workflow

### 1 — Prepare shards on Device A

Use `gguf_to_shards.py` from the companion tools to convert your GGUF model:

```bash
python gguf_to_shards.py convert llama-3.2-1b-instruct.Q4_K_M.gguf ./shards/
```

### 2 — Start the server on Device A

```bash
shard-server ./shards --port 8000
```

```
  ◆ shard-server  v0.1.0

  ✓  llama-3.2-1b-instruct.Q4_K_M.gguf  272 shards · 1.07 GB
  >  listening on http://0.0.0.0:8000
  >  metrics log  metrics.jsonl
```

### 3 — Run inference on Device B

```bash
shard-bridge run http://192.168.1.10:8000
```

```
  ◆ shard-bridge  v0.1.0

  >  connecting to http://192.168.1.10:8000
  ✓  reachable  11 ms
  >  fetching manifest
  ✓  llama-3.2-1b-instruct.Q4_K_M.gguf  272 tensors · 1.07 GB · GGUF v3

  ⠸ pulling shards ━━━━━━━━━━━━━━━━━━━━  [272/272]  1070 MB  28.4s
  ✓  all shards in memory  1070 MB · 28.4 s

  >  assembling GGUF in memory
  >  writing 1070 MB → /dev/shm/sb_12345.gguf
  >  loading model via llama.cpp (n_ctx=2048)
  ✓  model loaded  2.1 s · RAM +1.1 GB

  ──────────────────────────────────────────────────

  Prompt › tell me about edge computing

  >  generating

  Edge computing is a distributed computing paradigm that brings
  computation and data storage closer to the sources of data…

  ✓  156 tokens · 12.4 tok/s · TTFT 0.31 s

  ──────────────────────────────────────────────────

  metrics

    load_time_s             2.14 s
    ram_delta_mb            1131 MB
    tokens_per_sec          12.4
    time_to_first_tok_s     0.310 s
    tokens_generated        156
    cpu_percent             94.2 %
    ram_used_mb             2980 MB
    cpu_temp_c              71.3 °C
    device_hw               Raspberry Pi 4 Model B Rev 1.4
    os_info                 Linux 6.1.21-v8+ aarch64

  ──────────────────────────────────────────────────

  >  sending metrics → http://192.168.1.10:8000/metrics
  ✓  received  (entries on Device A: 1)
```

---

## Commands

### Device A

| Command | Description |
|---|---|
| `shard-server <dir>` | Start the shard server |
| `--port INT` | Port (default: 8000) |
| `--host STR` | Bind address (default: 0.0.0.0) |
| `--metrics-log PATH` | Where to append received metrics (default: metrics.jsonl) |

### Device B

| Command | Description |
|---|---|
| `shard-bridge run <host>` | Full pipeline: pull → infer → report |
| `shard-bridge pull <host>` | Pull shards to disk (no inference) |
| `shard-bridge status <host>` | Display latest metrics from Device A |

#### `run` options

| Flag | Default | Description |
|---|---|---|
| `--prompt / -p` | *(interactive)* | Prompt string |
| `--max-tokens / -n` | 256 | Tokens to generate |
| `--temp / -t` | 0.7 | Sampling temperature |
| `--ctx` | 2048 | Context window |
| `--no-report` | false | Skip sending metrics |

---

## Metrics collected

| Metric | Description |
|---|---|
| `load_time_s` | Time from shard pull to model ready |
| `ram_delta_mb` | RAM consumed by the model |
| `tokens_per_sec` | Inference throughput |
| `time_to_first_tok_s` | Latency to first token |
| `cpu_temp_c` | CPU temperature (RPi / Jetson thermal zone) |
| `cpu_percent` | CPU utilisation during inference |
| `ram_used_mb` | Total RAM used on device |
| `perplexity` | (optional) quality proxy |
| `device_hw` | Auto-detected hardware string |
| `os_info` | OS + kernel + arch |

---

## Zero-disk strategy

```
shard_data  ─── assemble_gguf_bytes() ──► gguf_bytes (RAM)
                                              │
                                       write_bytes()
                                              │
                                        /dev/shm/sb_<pid>.gguf   ← tmpfs, never hits disk
                                              │
                                       del gguf_bytes            ← Python bytes freed
                                              │
                                    Llama(model_path=...)        ← mmap from tmpfs
                                              │
                                       cleanup() → unlink()
```

Peak RAM ≈ **2× model size** during assembly (Python bytes + tmpfs copy).
After llama.cpp loads, Python bytes are GC'd and usage drops back to **~1× model size**.

---

## nginx reverse-proxy (optional)

```nginx
server {
    listen 8000;
    client_max_body_size 0;

    location / {
        proxy_pass         http://127.0.0.1:8001;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

Run uvicorn on port 8001, nginx on 8000.

---

## Project layout

```
shard_inference/
├── shared/
│   └── models.py          ShardManifest, InferenceMetrics
├── device_a/
│   └── server.py          FastAPI server  (shard-server CLI)
└── device_b/
    ├── shard_client.py    Async HTTP client
    ├── bridge.py          In-memory GGUF assembly + llama.cpp wrapper
    └── cli.py             TUI CLI  (shard-bridge CLI)
```