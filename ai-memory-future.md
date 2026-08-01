# Split Models Project State

Updated: 2026-07-30

## Goal

Build a stateless Archiso appliance that boots each machine as a vertical
split-model worker, discovers peers on the LAN, and runs CPU inference while
keeping all runtime state in RAM.

The appliance must support both:

- Native BIOS/UEFI boot from a USB drive or virtual CD-ROM.
- Immutable LCC GRUB entries that expect Ubuntu/Casper paths and parameters.

## Repository State

- Repository: `/home/vitor/tinker_git/split_models`
- Branch: `feature/operator-errors-tmux-help`
- Model distribution branch: `feature/master-model-distribution` at `30688ba`
- Latest pushed commit: `30688ba Harden stateless Archiso appliance`
- Structured errors, the split tmux console, and generated command examples are
  implemented in the worktree but not committed or rebuilt into the ISO yet.
- Do not modify or include unrelated user work such as
  `experiments/tensor_parallel/README.md`, `CLAUDE.md`, `hand.md`, or
  `studies/` unless explicitly requested.

## Stateless Appliance

The live system does not install itself, format disks, or require a persistent
VM disk.

- Application: `/opt/split-models`
- Fresh production virtualenv: `/opt/split-models/.venv`
- Hugging Face cache: `/run/split-models/huggingface`
- Benchmark results: `/run/split-models/benchmark-results`
- Temporary state: `/run/split-models`
- Models and results disappear after reboot.
- Checkpoints are not embedded in the ISO.
- The repository `.venv` is never copied into the image.
- The build uses `uv sync --frozen --no-dev --no-editable`.

Boot flow:

```text
live ISO
  -> DHCP and DNS
  -> split-models.service
  -> tmux session named split-models
  -> split-cluster worker
```

The smoke-test minimum is 4096 MiB RAM per VM. A 3072 MiB VM was OOM-killed
while the RAM-backed model cache and Python tensors existed simultaneously.

## Console and Autologin

The physical `tty1` automatically logs in as root and attaches the
`split-models` tmux session.

Systemd drop-in:

```ini
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin root --noclear %I $TERM
```

The tmux window is split side by side: the left 65% pane runs the worker and
shows its logs; the right 35% pane runs `bpytop`. `Ctrl-b` plus arrow keys move
between panes. Pressing `q` replaces `bpytop` with a shell in the right pane.
The fixed status line shows `role`, stable `node_id`, and current local IPv4.
The login profile waits up to 30 seconds for the service-created session and
attaches only on `/dev/tty1`; SSH, serial consoles, and other TTYs do not attach.

- The worker pane remains a shell after the worker exits, allowing promotion
  to master in the visible session.
- No `sudo` is needed because the login is already root.
- Getty does not wait for the network. The worker independently waits for
  `network-online.target`.

ArchWiki reference:

https://wiki.archlinux.org/title/Getty#Automatic_login_to_virtual_console

## Native and Casper Boot

The ISO contains:

```text
/casper/vmlinuz
/casper/vmlinuz.efi
/casper/initrd
/casper/initrd.img
/casper/initrd.lz
/casper/initrd.gz
```

The `casper-compat` mkinitcpio hook recognizes:

```text
iso-scan/filename=
fromiso=
isofrom=
findiso=
img_loop
```

It locates the filesystem containing the ISO, mounts it read-only, creates a
read-only loop device for the ISO, and exports that loop as `archisodevice` for
the normal Archiso hook.

It remains inactive during native Archiso boot when the kernel command line
already contains `archisolabel`, `archisodevice`, or `archisosearchuuid`.

Hook order:

```text
HOOKS=(base udev modconf casper-compat archiso block filesystems)
```

`switch_root(8)` moves `/proc`, `/sys`, `/dev`, and `/run` to the new root.
Therefore `/run/casper-compat` survives the switch in the same way as
`/run/archiso/img_dev`.

## Local Disk Safety

A plain `mount -o ro` can replay an ext journal and write metadata to a dirty
filesystem. The Casper hook now selects safe options using the filesystem type
reported by `blkid`:

```text
ext2/ext3/ext4 -> ro,noload
btrfs          -> ro,rescue=nologreplay
xfs            -> ro,norecovery
vfat/exfat     -> ro
ntfs3          -> ro
```

For older Btrfs kernels only, `ro,rescue=nologreplay` falls back to the
deprecated alias `ro,nologreplay`. Ext and XFS never fall back to unsafe plain
`ro`. Devices with no filesystem type are skipped.

`archiso_shutdown` does not explicitly unmount `/run/casper-compat`. This was
accepted because the mount is read-only and the only expected effect is
possible harmless shutdown noise.

## Initramfs Validation

The final ISO uses kernel `7.1.4-arch1-1` with:

```text
CONFIG_EXT4_FS=y
CONFIG_BTRFS_FS=y
CONFIG_XFS_FS=m
CONFIG_NTFS3_FS=m
CONFIG_VFAT_FS=m
CONFIG_EXFAT_FS=m
CONFIG_ISO9660_FS=m
CONFIG_BLK_DEV_LOOP=m
```

Ext4 and Btrfs are built into this kernel and therefore do not have `.ko`
files in the initramfs. The initramfs contains:

```text
loop.ko.zst
isofs.ko.zst
xfs.ko.zst
vfat.ko.zst
exfat.ko.zst
ntfs3.ko.zst
```

It also contains:

```text
hooks/casper-compat
hooks/archiso
usr/bin/blkid
usr/bin/losetup
usr/bin/modprobe
usr/bin/mount
```

## Final ISO Artifact

- Path: `dist/split-models-2026.07.30-x86_64.iso`
- Build timestamp: `2026-07-30 16:00:32 -0300`
- Size: 848,691,200 bytes, approximately 809 MiB
- SHA-256: `a280e0e0e48b30fd99f22352bfad4f811050002492029993a2ca0294aab9bc93`
- Hybrid BIOS/UEFI and El Torito records are present.
- `/var/lib/libvirt/images/split-models.iso` was updated to the final artifact.

## VM Validation

The final ISO was booted in a temporary diskless libvirt VM with 4096 MiB RAM
and two vCPUs.

Confirmed:

- systemd reached `running`.
- DHCP configured the VirtIO network interface.
- `split-models.service` was active.
- The `split-models` tmux session existed.
- `python -m split_lm.cluster.cli worker` was running.
- Root autologin occurred on `tty1`.
- `bpytop` was running on `tty1`.
- The only block devices were read-only `sr0` and read-only `loop0`.
- No writable VM disk was attached.

The final ISO was then tested with two diskless 4 GiB VMs. One VM was promoted
to master by stopping its automatic worker; the second remained a worker.

Inference result:

```text
Model: Qwen/Qwen2.5-0.5B-Instruct
Prompt: Reply with exactly: FINAL ISO OK
Output: FINAL ISO OK
Generated tokens: 3
Total latency: 2491.6 ms
Time to first token: 2224.5 ms
Throughput: 1.20 tokens/s
```

Both temporary VMs were destroyed and undefined after validation. No test
domains are currently defined.

The environment-specific test that remains is booting through the real LCC
entry from its actual ext4, Btrfs, XFS, VFAT, exFAT, or NTFS3 host filesystem.

## Master Promotion Requirement

Every appliance starts as a worker. Before starting the master on one machine,
its local worker must be stopped as documented in
`/root/README-split-models.txt`.

If the local worker remains discoverable, both the head process and its local
worker shard try to bind inference port 9999 and startup fails with:

```text
[Errno 98] Address already in use
```

Validated workflow:

```text
1. Choose one appliance as master.
2. Stop its worker, currently by interrupting the worker session or stopping
   split-models.service.
3. Leave workers running on the other appliances.
4. Run /root/split-master.sh on the promoted master.
```

Automatic local-worker suspension and restoration is not implemented.

## Vertical Layer Split

`split_lm/cluster/plan.py` splits the model into contiguous, end-exclusive
ranges. The head receives the first range and workers receive the remaining
ranges in network order.

For a 24-layer model with one worker:

```text
Head:   (0, 12)  -> original layers 0..11
Worker: (12, 24) -> original layers 12..23
```

Each process creates a smaller instance of the same model architecture:

```python
cfg.num_hidden_layers = layer_end - layer_start
model = AutoModelForCausalLM.from_config(cfg)
```

Checkpoint tensors are not numerically changed. Their state-dict keys are
renamed because each local model numbers its own layers from zero:

```python
local_index = original_index - layer_start
```

Example for a worker starting at original layer 12:

```text
model.layers.12.self_attn.q_proj.weight
    -> model.layers.0.self_attn.q_proj.weight

model.layers.23.self_attn.q_proj.weight
    -> model.layers.11.self_attn.q_proj.weight
```

The head owns embeddings and the first layers. Intermediate workers own only
their transformer layers. The final worker also owns final normalization and
`lm_head`.

Weights remain fixed during inference. The runner uses
`torch.inference_mode()` and the model is in `.eval()` mode. Hidden states,
KV caches, positions, and generated token IDs change; weights do not.

## Hidden-State Data Path

The head computes its local slice:

```python
hidden = _forward(head_model, None, device, True, False, chunk, cache_state)
```

It sends the resulting tensor to the first worker:

```python
_send(conn, hidden, net_stats, reset_cache)
```

The protocol serializes tensor dtype, shape, flags, and contiguous payload over
TCP. The worker accepts the upstream connection and reconstructs the tensor:

```python
h, reset_cache = _recv(upstream, device, net_stats)
```

Within each local slice, every transformer layer returns a new hidden state:

```python
for layer in model.model.layers:
    out = layer(hidden, ...)
    hidden = out[0] if isinstance(out, (tuple, list)) else out
```

Non-final workers forward their result:

```python
_send(downstream, h, net_stats, reset_cache)
```

The final worker applies normalization and `lm_head`, selects the next token,
and sends only that token ID back to the head:

```python
nid = h[0, -1].argmax()
_send(token_back, nid.unsqueeze(0), net_stats)
```

The head receives, decodes, displays, and appends the token before the next
generation step.

## Cluster Protocol

Default ports:

```text
UDP 29500 -> discovery
TCP 29510 -> worker control agent
TCP 9999  -> hidden-state inference stream and token-back endpoint
TCP 10001 -> head control and READY/ERROR messages
```

Worker jobs are started in reverse pipeline order so downstream listeners are
ready before upstream nodes connect. Control messages report `READY`, `ERROR`,
and process lifecycle events.

JSON framing is isolated in `split_lm/io/framing.py`. Binary tensor framing is
in `split_lm/io/protocol.py`. Lightweight CLI commands do not import Torch or
Transformers until inference is actually started.

## Current Model Download Behavior

`split_lm/model/loader.py` first tries to download `model.safetensors`.

- For a monolithic checkpoint, every participating machine downloads the same
  complete file even though it loads only its assigned tensors.
- For a sharded checkpoint, the index is read and each node downloads only the
  shard files containing weights needed by its assigned layers.
- A shard remains the minimum download unit; individual tensors are not
  downloaded separately.
- The Hugging Face cache is RAM-backed, so files disappear after reboot.

## Future Model Distribution Proposal

Proposed flow: Hugging Face -> one master download -> LAN distribution -> each
worker loads its assigned tensors. Rsync reduces WAN traffic but fresh nodes
still receive the whole monolithic file. Sharded checkpoints can distribute
only required shards. Avoiding full copies of monolithic safetensors requires
per-node safetensors extraction or a tensor-aware service. Authentication,
checksums, failure cleanup, master RAM, and startup synchronization remain open.
This proposal is not implemented.

## Integrated GPU Feasibility

The current runtime is CPU-only: `pyproject.toml` selects PyTorch CPU wheels,
the CLI defaults to `cpu`, and TorchAO quantization rejects other devices.

- Intel UHD 730 is not in PyTorch's validated XPU client list, which currently
  names Arc and Core Ultra Arc hardware. OpenVINO officially supports the Intel
  UHD family, but Optimum's `OVModelForCausalLM` is a complete compiled graph;
  using it for this layer pipeline requires a new partial-graph/KV-cache backend.
- Radeon Vega 6 is absent from the official ROCm compatibility matrix. The
  practical accelerator path is llama.cpp with Vulkan, not this PyTorch worker.
- llama.cpp officially supports Vulkan and an RPC backend that distributes
  weights and KV cache across local and remote devices. Its RPC documentation
  labels the backend proof-of-concept, fragile, and insecure.

Keep the validated CPU split as the production path. The best cross-vendor
experiment is a separate GGUF llama.cpp Vulkan+RPC spike on an isolated LAN,
starting with a 0.5B model and comparing CPU, each iGPU, and two-node RPC.
OpenVINO should be evaluated separately as an Intel-only backend.

## Model and Cluster Limits

- Default interactive model: `Qwen/Qwen2.5-0.5B-Instruct`.
- The base `Qwen/Qwen2.5-0.5B` model repeated bad text even under full local
  Transformers inference; this was not a vertical-split bug.
- Maximum planned cluster size: 24 total nodes, including the head.
- The nominal aggregate budget is 96 GiB at 4 GiB per node.
- More than 23 discovered workers currently aborts planning instead of
  selecting a subset automatically.
- Existing 3B benchmark profiles are unsafe for 4 GiB VMs.
- Secure Boot enforcement is unsupported because the GRUB/kernel path is not
  signed for that environment.
- Encrypted filesystems and unsupported RAID/LVM arrangements cannot be scanned
  by the Casper compatibility hook.

## Tests

The source passed:

```text
uv run --extra plots python -m unittest discover -s tests
102 tests passed
```

Also validated:

```text
bash -n on Casper, tmux, monitor, and login scripts; git diff --check
```

The test suite includes regressions for stateless runtime paths, root autologin,
`bpytop`, Casper layout, initramfs hook ordering, filesystem anti-replay mount
options, classified errors, command examples, cluster defaults, lightweight
imports, and framing behavior.

## Relevant Files

- Build and boot: `deploy/build_archiso.sh`, `deploy/archiso/profiledef.sh`, and
  `deploy/archiso/packages.x86_64`.
- Casper: `deploy/archiso/airootfs/etc/initcpio/{hooks,install}/casper-compat`.
- Runtime: `split-models.service`, `/root/.bash_profile`,
  `/root/split-master.sh`, and `/root/README-split-models.txt` under `airootfs`.
- Protocol: `docs/protocol.md`, `split_lm/io/framing.py`, and
  `split_lm/io/protocol.py`.
- Operator UX: `split_lm/errors.py`, `split_lm/command_examples.py`, and the
  `split-models-{tmux,monitor}` appliance scripts.
- Split runtime: `split_lm/cluster/{plan,master}.py`,
  `split_lm/model/{loader,runner}.py`, and `split_lm/node/{head,worker}.py`.
- Regressions: `tests/test_archiso.py` and `tests/test_cluster.py`.

## Next Steps

1. Rebuild the ISO so the new tmux, errors, and help behavior is included.
2. Test the rebuilt ISO and the actual immutable LCC GRUB entry.
3. Decide whether `/root/split-master.sh` should automatically stop and later
   restore the local worker during master promotion.
4. Decide whether the first model-distribution implementation should use
   rsync, an HTTP service, or a master-owned Hugging Face cache proxy.
5. If full monolithic LAN copies are still too expensive, design per-node
   safetensors extraction and distribution.
6. Fix the master launcher's initial status line so CLI `--max-new-tokens`
   overrides are displayed accurately.
7. Implement automatic worker subset selection when discovery returns more
   nodes than the model has layers or the 24-node limit allows.
8. Commit and push only the intended project changes after reviewing the dirty
   worktree and excluding unrelated user files.
