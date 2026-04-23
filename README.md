# ModelPulse 🚀

**End-to-end partial-weight transfer pipeline for edge LLM inference.**

ModelPulse enables a unique "Zero-Disk" inference strategy: **Device A** (Server) serves model shards over the network, while **Device B** (Client/Bridge) reconstructs the model entirely in RAM and runs inference via `llama.cpp` without ever writing the full GGUF to physical storage.

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Server (Device A)                       │
│                  FastAPI @ 0.0.0.0:8000                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  WebSocket /ws (Control Plane)   HTTP (Data Plane)          │
│  ├─ MODEL_READY                  ├─ GET /manifest           │
│  ├─ PING/PONG                    ├─ GET /shards/*           │
│  ├─ METRICS                      └─ POST /metrics           │
│  └─ ACK/BYE                                                 │
│                                                             │
│  /models/upload (Multipart)                                 │
│  └─ Accept manifest.json + *.shard files                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↑                              ↑
         │                              │
         │ WS connect                   │ HTTP GET/POST
         │ + MODEL_READY signal         │ + shard stream
         │                              │
┌────────┴──────────────────────────────┴─────────────────────┐
│                   Client (Device B)                         │
│                       Bridge CLI                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Connect WebSocket → Send HELLO                          │
│  2. Receive MODEL_READY → Fetch manifest (HTTP)             │
│  3. Download shards (HTTP streaming)                        │
│  4. Assemble GGUF in /dev/shm                               │
│  5. Load with llama.cpp                                     │
│  6. Run inference                                           │
│  7. Send METRICS → Loop back to step 2                      │
│     (no restart, listen for next model)                     │ 
│                                                             │
└─────────────────────────────────────────────────────────────┘
```



---

## ✨ Key Features

- **🛡️ Zero-Disk Strategy**: Models are assembled in `tmpfs` (`/dev/shm`), ensuring no persistent GGUF footprint on the client's disk.
- **🔄 Dynamic Model Swapping**: Upload new models to the server at runtime; connected clients automatically unload, pull, and reload the new model without a restart.
- **📊 Real-time Telemetry**: Detailed inference metrics (TTFT, tok/s, RAM delta, CPU temp) are streamed back to the server for centralized monitoring.
- **🛠️ Integrated Benchmarking**: Built-in suite to stress-test edge devices and validate performance across different quantization levels.
- **🌐 Network Agnostic**: Works seamlessly over local networks, Tailscale, or any HTTP/WS-capable connection.

---

## 📦 Installation

Install `ModelPulse` from PyPI:

```bash
pip install modelpulse
```

Alternatively, install directly from the repository for the latest dev features:

```bash
pip install git+https://github.com/MdSufiyan005/ModelPulse.git
```

*Note: Ensure you have `llama-cpp-python` dependencies installed on your system (e.g., `build-essential`, `python3-dev`).*

---

## 🔄 Workflow

### 1. Prepare Shards
Convert a monolithic `.gguf` file into a shard directory using the companion tool:

```bash
python tools/gguf_to_shards.py convert my_model.gguf ./my-shards/
```

### 2. Start the Server
Start the control plane on Device A. It will default to using `./models-storage` for storing model data.

```bash
modelpulse server run --host 0.0.0.0 --port 8000
```

### 3. Run the Bridge
Connect your edge device to the server. It will wait for a model to be assigned.

```bash
modelpulse bridge run http://<server-ip>:8000
```

### 4. Dynamic Upload
Upload your prepared shards to the server. All connected bridges will instantly receive the update.

```bash
./upload_model.sh "qwen-3.5-2b" "./my-shards/"
```

---

## 📋 Command Reference

### `modelpulse server run`
Start the FastAPI control plane.

| Option | Default | Description |
| :--- | :--- | :--- |
| `--shard-dir`, `-d` | `./models-storage` | Root directory for model storage |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Listening port |
| `--metrics-log` | `metrics.jsonl` | File to append received telemetry |

### `modelpulse bridge run`
Connect to a server and enter the inference loop.

| Option | Default | Description |
| :--- | :--- | :--- |
| `host` | (Required) | Server URL (e.g., `http://100.64.0.5:8000`) |
| `--prompt`, `-p` | (Interactive) | Run a single prompt and exit |
| `--benchmark`, `-b`| `false` | Run the standard benchmark suite |
| `--max-tokens`, `-m`| `256` | Token generation limit |
| `--temperature` | `0.7` | Sampling temperature |
| `--n-ctx` | `2048` | Context window size |

---

## 📁 Project Layout

```bash
modelpulse/
├── modelpulse/             # Core package
│   ├── server/             
│   │   └── server.py       # FastAPI + WebSocket control plane
│   ├── client/             # Bridge (Device B) logic
│   │   ├── cli.py          # Claude-inspired terminal UI
│   │   ├── bridge.py       # RAM GGUF assembly & llama.cpp loading
│   │   ├── shard_client.py # Async HTTP downloader for shards
│   │   └── benchmarks.py   # Built-in performance testing suite
│   ├── shared/             # Cross-component protocol definitions
│   │   ├── ws_protocol.py  # WebSocket message schemas
│   │   └── models.py       # ShardManifest & InferenceMetrics models
│   └── main.py             # Unified CLI entry point
├── tools/                  # Model preparation utilities
│   ├── gguf_to_shards.py   # GGUF → Shard converter (tensor-level)
│   └── gguf_parser.py      # Low-level GGUF format metadata reader
├── upload_model.sh         # Script for dynamic model assignment
├── TEST_WORKFLOW.md        # Step-by-step end-to-end testing guide
├── pyproject.toml          # Project metadata & dependencies
└── metrics.jsonl           # Appends log for inference telemetry
```

---

## 💾 The Zero-Disk Strategy

ModelPulse leverages the Linux `tmpfs` (RAM-backed filesystem) to satisfy `llama.cpp`'s requirement for a file path while keeping the actual data off physical storage:

1. **Pull**: Bridge fetches `manifest.json`.
2. **Stream**: Bridge pulls `.shard` files (tensor by tensor) into memory.
3. **Assemble**: Bridge calculates GGUF layout and writes bytes to `/dev/shm/sb_<pid>.gguf`.
4. **Load**: `llama-cpp-python` loads the model via `mmap` from the RAM-backed file.
5. **Clean**: Once the model is unloaded, the virtual file is unlinked and memory is reclaimed.

---

## 📡 Networking (Tailscale)

For easy cross-device connectivity without port forwarding, Tailscale is highly recommended:

```bash
# Get IP on Server
tailscale ip  # e.g., 100.66.170.100

# Connect Bridge
modelpulse bridge run http://100.66.170.100:8000
```

---

<p align="center">
  <i>Built with ❤️ for Edge AI and Decentralized Inference.</i>
</p>
