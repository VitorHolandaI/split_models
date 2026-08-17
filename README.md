# Split Models

Distributed LLM inference across trusted LAN machines, in two independent
engines:

- **llama.cpp RPC appliance** (`split-llama`) — the current focus: a stateless
  Arch ISO that boots into a fleet of inference nodes with one-command
  orchestration, model splitting over RPC, and load-balanced replication.
- **PyTorch vertical split** (`split-cluster`) — the original research engine:
  layers of a model split across machines over a custom control protocol.

Both engines share the same philosophy: no shared storage, no orchestration
daemon, everything discoverable over UDP broadcast on a trusted LAN.

> **Security:** the RPC backend has no authentication or encryption. Anyone
> who reaches the ports runs code on your nodes. Use only on an isolated LAN.

---

## 1. llama.cpp appliance (`split-llama`)

Boot the ISO on N machines and one command runs a fleet benchmark:

```bash
split-llama fleet bench \
  --scenario replicated \
  --model small \
  --profile users-2k \
  --auto-discover
```

The ISO embeds everything: llama.cpp binaries, the Python controller, the
fleet SSH key (mode `0600`), and HAProxy.

### Modes (no reboot needed)

Each node can switch role at runtime via `split-llama mode`:

| Mode | What the node does |
|---|---|
| `worker` | Runs `ggml-rpc-server`, advertises over UDP, holds layers |
| `replica` | Runs a full local `llama-server` (same model, whole node) |
| `distributed-master` | One `llama-server` spanning the fleet via RPC layers |
| `idle` | No inference; still reachable for fleet control |

```bash
split-llama mode status
split-llama mode replica --model small
split-llama mode distributed-master --model small --rpc 192.168.0.11:50052,192.168.0.12:50052
split-llama mode worker
```

Transitions are supervised: locked, health-checked, and rolled back to
`worker` on failure.

### Always-on fleet control

Discovery for control is separate from RPC discovery, so nodes stay
reachable in every mode:

```bash
split-llama nodes              # only live RPC workers
split-llama fleet nodes        # every node + current mode (replicas, masters, idle)
```

### Fleet commands (one coordinator node, no manual IPs)

```bash
split-llama become_master      # stop local inference, keep control plane

split-llama fleet bench \
  --scenario replicated --model small --profile users-2k --auto-discover

split-llama fleet bench \
  --scenario distributed --model small --profile users-2k --auto-discover

split-llama fleet reboot       # reboot every discovered node
```

What `fleet bench --auto-discover` does automatically:

1. Discover every node over the always-on control UDP.
2. Return all nodes to `worker` (supervised cleanup of the previous role).
3. `replicated`: promote all to replicas, start local HAProxy (round-robin,
   active `/health` checks, SSE tunnel timeout).
4. `distributed`: promote exactly one master with the others as RPC workers
   (layer split).
5. Run the capacity profile and write `benchmark-results/capacity-summary.csv`.

Scenarios can switch in any order — `replicated` to `distributed` and back —
without rebooting or typing IPs.

### Building the ISO

Requires Arch Linux packages: `cmake`, `shaderc`, `glslc`, `vulkan-headers`,
`spirv-headers`, `openssl`, plus the `archiso` tooling:

```bash
./deploy/build_llamacpp_archiso.sh
```

Output: `dist/llamacpp-rpc/split-models-llamacpp-rpc-<date>-x86_64.iso`.

The build is pinned to a single llama.cpp commit (`deploy/llama-version.env`)
so RPC protocol versions always match across the fleet. When `glslc` changes,
the build invalidates the Vulkan shader cache automatically.

Test the ISO locally with 3 QEMU/KVM VMs:

```bash
SPLIT_VM_MEMORY_MB=2048 ./deploy/run_test_vm.sh 1 dist/llamacpp-rpc/*.iso
SPLIT_VM_MEMORY_MB=2048 ./deploy/run_test_vm.sh 2 dist/llamacpp-rpc/*.iso
SPLIT_VM_MEMORY_MB=2048 ./deploy/run_test_vm.sh 3 dist/llamacpp-rpc/*.iso
```

The full operational guide lives in
[`README-FLEET-BENCH.md`](README-FLEET-BENCH.md).

---

## 2. PyTorch vertical split (`split-cluster`)

Layer-wise splitting for CPU inference across trusted machines. The complete
wire protocol, startup sequence and failure behavior are documented in
[`docs/protocol.md`](docs/protocol.md).

### Worker

```bash
uv run split-cluster worker
```

Listens on `29500/udp` (discovery), `29510/tcp` (control) and `9999/tcp`
while a job is active.

### Interactive master

```bash
uv run split-cluster master \
  --discover \
  --broadcast 192.168.0.255 \
  --head-address 192.168.0.7 \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

Every answer ends with a metrics line and appends to
`benchmark-results/generation.jsonl`:

```text
METRICS split | prompt=128 tokens | generated=50 | total=4.000s | tok/s=12.50 | TTFT=500.0ms | avg_inter_token=71.4ms
```

### Local baseline

```bash
uv run python tools/benchmark_single_model.py \
  --model Qwen/Qwen2.5-0.5B \
  --max-new-tokens 50 \
  --ignore-eos
```

Local and split runs write the same JSONL schema, distinguished by `mode`.

### Benchmark profiles

Named profiles live in `benchmarks.toml`:

```bash
uv run split-cluster benchmarks
uv run split-cluster benchmark-local --benchmark-profile context-quick
uv run split-cluster master \
  --discover --broadcast 192.168.0.255 --head-address 192.168.0.7 \
  --benchmark-profile context-quick
```

Optional TorchAO INT8/INT4 weight-only quantization:

```bash
uv sync --extra quant
```

Profiles: `context-quick`, `context-quick-int8`, `context-quick-int4`,
`context-realistic`, `poem-realistic`.

---

## Development

```bash
uv sync --extra plots
uv run --extra plots python -m unittest discover -s tests
```

The suite covers both engines: fleet coordination, mode transitions, HAProxy
generation, capacity profiles, discovery protocols, ISO staging, and the
PyTorch split cluster — all against named fakes, no network or hardware
required.

## Documentation

- `README-FLEET-BENCH.md` — operator manual for the llama.cpp appliance
- `docs/protocol.md` — PyTorch split cluster wire protocol
- `docs/heterogeneous-split.md` — heterogeneous worker memory planning
- `docs/memory-bandwidth-roof.md` — CPU inference performance model
- `docs/deployment-memory.md` — RAM budgets for the appliance
