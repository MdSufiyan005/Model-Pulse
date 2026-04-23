# ModelPulse Testing Workflow 🧪

This guide provides a step-by-step workflow to test the ModelPulse end-to-end pipeline. It assumes you have installed `modelpulse` via pip or are running from the source.

---

## 🚀 Quick Start (5 Minutes)

### Terminal 1: Start the Control Plane (Server)
Start the server to manage model distribution and telemetry. It will automatically create a `./models-storage` directory.

```bash
modelpulse server run --host 0.0.0.0 --port 8000
```

### Terminal 2: Start the Bridge (Client)
Connect an edge device to the server. The bridge will wait for a model assignment.

```bash
# Replace <server-ip> with your actual server IP (e.g., localhost or Tailscale IP)
modelpulse bridge run http://<server-ip>:8000
```

### Terminal 3: Upload a Model
Assign a model to the server. All connected bridges will instantly start pulling shards.

```bash
# Usage: ./upload_model.sh <model_id> <shard_directory>
./upload_model.sh "qwen-2.5-2b" "./path/to/my-shards"
```

---

## 🧪 Testing Scenarios

### Scenario 1: Interactive Multi-Prompt
Start the bridge without any flags. It will stay connected, allowing you to enter multiple prompts.

```bash
modelpulse bridge run http://<server-ip>:8000
# › What is the speed of light?
# › Tell me a joke about robots.
```

### Scenario 2: Dynamic Hot-Reload
While a bridge is connected and a model is loaded, upload a **different** model via the upload script.
1. The server notifies the bridge.
2. The bridge automatically unloads the old model.
3. The bridge pulls the new manifest and shards.
4. The bridge reloads the new model—**no manual restart required**.

### Scenario 3: Batch Inference
Run a single prompt and exit immediately after receiving the response and reporting metrics.

```bash
modelpulse bridge run http://<server-ip>:8000 --prompt "Explain quantum entanglement in one sentence."
```

### Scenario 4: Performance Benchmarking
Run the built-in benchmark suite to measure TTFT, throughput (tok/s), and resource usage.

```bash
modelpulse bridge run http://<server-ip>:8000 --benchmark
```

---

## 🛠️ Manual API Testing (cURL)

You can interact with the server API directly using `curl`.

### Check Health & Active Model
```bash
curl http://<server-ip>:8000/health | jq
```

### List Available Models in Storage
```bash
curl http://<server-ip>:8000/models | jq
```

### View Latest Telemetry
```bash
curl http://<server-ip>:8000/results/latest | jq
```

### Manually Notify Clients
If you manually move files into the storage directory, you can force a notification to all connected bridges:
```bash
curl -X POST http://<server-ip>:8000/models/notify
```

---

## 🔍 Troubleshooting

| Issue | Potential Cause | Fix |
| :--- | :--- | :--- |
| `Connection refused` | Server not running or wrong IP | Verify server status and firewall/Tailscale settings. |
| `Stuck at "waiting..."` | No model active on server | Use `./upload_model.sh` to assign a model. |
| `Out of memory` | Model too large for RAM | Use a higher quantization (e.g., Q4_K_M) or smaller parameter count. |
| `No /dev/shm access` | Environment restrictions | ModelPulse will fallback to `/tmp`, though performance may decrease. |

