# Split Models

Vertical layer splitting for CPU inference across trusted LAN machines.

## Worker Agent

Each worker discovers itself with a random ID and waits for validated jobs:

```bash
uv run split-cluster worker
```

The worker listens on `29500/udp` for discovery, `29510/tcp` for control, and
`9999/tcp` while an inference job is active.

## Interactive Master

```bash
uv run split-cluster master \
  --discover \
  --broadcast 192.168.0.255 \
  --head-address 192.168.0.7 \
  --model Qwen/Qwen2.5-0.5B
```

Every answer ends with:

```text
METRICS split | prompt=128 tokens | generated=50 | total=4.000s | tok/s=12.50 | TTFT=500.0ms | avg_inter_token=71.4ms
```

Records are appended to `benchmark-results/generation.jsonl`.

## Local Baseline

```bash
uv run python tools/benchmark_single_model.py \
  --model Qwen/Qwen2.5-0.5B \
  --max-new-tokens 50 \
  --ignore-eos
```

The local and split commands write the same JSONL schema with different
`mode` values.

## Context Sweep

For two machines with a 5 GiB application budget each, use the already tested
`Qwen/Qwen2.5-3B-Instruct`. Its BF16 checkpoint is approximately 6.17 GB and
has 36 layers. The split loader preserves BF16 instead of expanding weights to
FP32.

Start conservatively:

```bash
uv run split-cluster master \
  --discover \
  --broadcast 192.168.0.255 \
  --head-address 192.168.0.7 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-new-tokens 16 \
  --context-sizes 128,256,512,1024,2048
```

Each completed point updates:

```text
benchmark-results/generation.jsonl
benchmark-results/context-sweep.csv
benchmark-results/context-sweep.svg
```

The sweep uses exact input-token counts, resets KV cache between points, keeps
the model loaded, and ignores EOS so every point generates the same number of
tokens. Add `4096` and `8192` only after checking memory use.
