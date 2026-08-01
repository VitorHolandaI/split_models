# Split Models Archiso

This is a stateless live appliance. The compressed root filesystem, project,
Python environment, and worker service are part of the ISO. All runtime writes
use RAM and disappear after reboot. No local disk is mounted or formatted.

## Build

Install the host build tools once:

```bash
sudo pacman -S --needed archiso uv trash-cli
```

Build from the repository root:

```bash
./deploy/build_archiso.sh
```

The build script copies only project source and lock metadata, creates a fresh
`.venv` inside the staged ISO with `uv sync --frozen --no-dev --no-editable`,
enables networkd/resolved and `split-models.service`, and writes the ISO to
`dist/`. It never copies the repository's local `.venv`.

## Boot flow

```text
live ISO rootfs
  -> DHCP and DNS
  -> network-online.target
  -> split-models.service
  -> RAM directories under /run/split-models
  -> tmux session split-models
  -> split-cluster worker
```

Runtime paths are ephemeral:

- `/run/split-models/huggingface`: models downloaded on demand;
- `/run/split-models/benchmark-results`: JSONL, CSV, and SVG outputs;
- `/run/split-models/tmp`: temporary files.

The physical console on `tty1` logs in as root and opens `bpytop` while the
worker continues in the detached tmux session. Press `q` to leave `bpytop` and
use the shell. Serial consoles and SSH do not auto-launch the monitor.

Inspect a running node:

```bash
systemctl status split-models
tmux attach -t split-models
```

Use `Ctrl-C` inside tmux to stop the worker and promote that node to master.
Rebooting restores the original worker state and clears all runtime data.

The root home contains a shortcut that detects the LAN broadcast and starts the
default Qwen2.5-0.5B-Instruct master:

```bash
/root/split-master.sh
```

Override defaults with `SPLIT_BROADCAST`, `SPLIT_MODEL`, or
`SPLIT_MAX_NEW_TOKENS` when needed.

An operational command reference is available at
`/root/README-split-models.txt`. The detected broadcast follows the default
network interface, so the shortcut is not tied to the libvirt address range.

## VM smoke test

The VM needs no writable disk:

```bash
./deploy/run_test_vm.sh 1
./deploy/run_test_vm.sh 2
```

With libvirt, create each VM using only `--cdrom` and the `default` NAT network.
For this host, use 2 vCPUs and at least 4096 MiB per VM, and test inference with
`Qwen/Qwen2.5-0.5B-Instruct`. A 3072 MiB VM was OOM-killed while holding the RAM-backed
model download and approximately 1.7 GiB of anonymous Python memory.

`virt-install --cdrom` may eject live media after the first shutdown because it
treats the boot as an installation. Reinsert the ISO into both the running and
persistent definitions before resetting the VM:

```bash
virsh --connect qemu:///system change-media split-live-1 sda \
  /var/lib/libvirt/images/split-models-stateless.iso --insert --live --config
```

## Casper-compatible loopback boot

Some LCC deployments copy the ISO onto an existing filesystem and use an
immutable GRUB entry written for Debian/Ubuntu. The build publishes real files
at `/casper/vmlinuz` and `/casper/initrd`; common alternate initrd names are
Rock Ridge symlinks to `/casper/initrd`.

The `casper-compat` initramfs hook accepts `iso-scan/filename`, `fromiso`,
`isofrom`, `findiso`, and the Archiso `img_loop` parameter. It mounts the
underlying filesystem read-only, creates a loop device for the ISO, then passes
that loop as `archisodevice` to the normal Archiso hook. Native USB BIOS/UEFI
boot stays on the original Archiso path and does not invoke the shim.

The shim can only see storage available in the initramfs. Encrypted filesystems,
unsupported RAID/storage drivers, paths containing shell-breaking whitespace,
and Secure Boot enforcement remain outside this compatibility layer. Validate
the final image against the exact LCC GRUB entry before rollout.

The compatibility path was validated with an external GRUB rescue image using
exactly `boot=casper iso-scan/filename=/split-models.iso`. The booted guest
mounted the outer filesystem, created `/dev/loop0` for the nested ISO, mounted
that loop at `/run/archiso/bootmnt`, and then mounted the inner EROFS rootfs.
