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
split-llama mode distributed-master --model small --rpc 192.0.2.11:50052,192.0.2.12:50052
split-llama mode distributed-master --model small --auto-discover
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

For an interactive distributed chat, including immediately after a replicated
run:

```bash
split-llama master --discover \
  --hf unsloth/Qwen3.5-9B-GGUF:Q6_K \
  -- -c 32768
```

`master --discover` first finds every control agent, returns every node to
`worker` over fleet SSH, and then discovers the live RPC endpoints. The master
downloads the complete GGUF; workers receive only their assigned layer tensors
and retain the weights and KV cache for those layers. The model cache keeps only
the selected model, preserving it when selected again and deleting it when a
different model is chosen.

#### Hardware measurement and capacity planning

Before splitting, `master --discover` measures every discovered node (SSH) and
prints a capacity table:

```text
NODE            RAM GiB    TOPS    MIPS  tok/s CPU tok/s Vulkan
10.0.0.2           14.0    8.00    5000       3.00        10.00
10.0.0.4           14.0    8.00    5000       3.00        20.00
```

Then it asks how many machines to use in total (master included), validated
between the safe minimum and the discovered maximum. Workers are ordered by
Vulkan throughput; machines without a Vulkan driver fall back to their CPU
tok/s and rank last. Unselected nodes stay out of the RPC endpoints.

`split-llama hardware-bench` runs the same measurements on one machine and
explains the units:

- **TOPS** — iGPU INT8 matrix math (vkpeak), tera-operations per second.
- **TFLOPS** — iGPU FP16 matrix math (vkpeak), tera-floating-point ops/s.
- **MIPS** — CPU integer rating from `7z b`; kept in its native unit, never
  renamed to TOPS.
- **tok/s CPU / Vulkan** — effective llama.cpp throughput with the official
  small model (`unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M`, prompt 128, gen 32): what
  actually matters for the split.

```bash
split-llama hardware-bench          # human table
split-llama hardware-bench --json   # one record per node, for the fleet
```

#### Model cache: RAM by default, USB for the split master

All runtime paths use `/run`, which is RAM-backed in the live ISO. The original
disk is never formatted, mounted, or used for model data. In a distributed run,
only the master may use an explicitly selected USB ext4 partition for the full
GGUF; workers receive their layer tensors over RPC and keep them in RAM.
Replicas also keep their complete model in RAM.

Select a specific partition when more than one USB is present:

```bash
split-llama master --discover --usb-device /dev/sda1 \
  --hf unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
split-llama mode distributed-master --model small --auto-discover \
  --usb-device /dev/sda1
```

Without `--usb-device`, the selected server mode uses the RAM-backed cache.
The free space is validated before any USB download starts — the previous model
is only erased after the new one is known to fit, including a 0.5 GB reserve.

#### One large model, end to end

A 16 GB USB ext4 holds the catalog's largest checked GGUF
(`ggml-org/gpt-oss-20b-GGUF:MXFP4`, 12.11 GB) with room to spare:

```bash
split-llama master --discover \
  --usb-device /dev/sda1 \
  --hf ggml-org/gpt-oss-20b-GGUF:MXFP4 \
  -- -c 32768
```

Preview the fleet plan before changing any node or downloading the model. With
no `--machines`, the dry-run selects the smallest safe master-first prefix:

```bash
split-llama master --discover --dry-run \
  --hf ggml-org/gpt-oss-20b-GGUF:MXFP4
```

The output reports each node's safe RAM budget, estimated model share, and
contiguous layer range. Known models provide their layer count automatically.
Layer ranges are proportional estimates; llama.cpp can adjust them at runtime
from the actual tensor sizes. For an uncatalogued GGUF, pass `--model-size-gb N`
and, when layer ranges are wanted, `--model-layers N`.

### Fleet commands (one coordinator node, no manual IPs)

```bash
split-llama become_master      # stop local inference, keep control plane

split-llama fleet bench \
  --scenario replicated --model small --profile users-2k --auto-discover

split-llama fleet bench \
  --scenario distributed --model small --profile users-2k --auto-discover

split-llama fleet reboot       # reboot every discovered node
split-llama fleet reset --auto-discover  # return every node to worker without reboot
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
  --broadcast 192.0.2.255 \
  --head-address 192.0.2.7 \
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
  --discover --broadcast 192.0.2.255 --head-address 192.0.2.7 \
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
