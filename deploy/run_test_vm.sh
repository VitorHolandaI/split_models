#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    printf 'uso: %s <1|2> [caminho-da-iso]\n' "$0" >&2
}

latest_iso() {
    local candidates=("$repo_root"/dist/split-models-*.iso)
    if [[ ! -e "${candidates[0]}" ]]; then
        printf 'ISO ausente em %s/dist; execute ./deploy/build_archiso.sh\n' "$repo_root" >&2
        exit 1
    fi
    printf '%s\n' "${candidates[${#candidates[@]} - 1]}"
}

main() {
    local node="${1:-}"
    [[ "$node" == "1" || "$node" == "2" ]] || { usage; exit 1; }
    [[ -e /dev/kvm ]] || { printf '/dev/kvm ausente; esperado KVM habilitado\n' >&2; exit 1; }
    local iso_path="${2:-$(latest_iso)}"
    [[ -f "$iso_path" ]] || { printf 'ISO invalida: %s\n' "$iso_path" >&2; exit 1; }

    local ssh_port=$((2220 + node))
    local private_ip="192.168.76.$((9 + node))"

    printf 'node %s: configure a LAN privada com:\n' "$node"
    printf '  ip -brief link\n  ip address add %s/24 dev enp0s4\n' "$private_ip"
    printf 'saida do QEMU: Ctrl-a x\n'

    exec qemu-system-x86_64 \
        -name "split-node-$node" -machine q35,accel=kvm -cpu host \
        -smp "${SPLIT_VM_CPUS:-4}" -m "${SPLIT_VM_MEMORY_MB:-4096}" \
        -boot d -cdrom "$iso_path" \
        -netdev "user,id=wan,hostfwd=tcp:127.0.0.1:$ssh_port-:22" \
        -device "virtio-net-pci,netdev=wan,mac=52:54:00:76:0$node:01,addr=0x3" \
        -netdev "socket,id=lan,mcast=230.0.0.1:1234" \
        -device "virtio-net-pci,netdev=lan,mac=52:54:00:76:0$node:02,addr=0x4" \
        -display none -serial mon:stdio
}

main "$@"
