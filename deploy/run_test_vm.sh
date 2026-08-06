#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=deploy/lib_iso_paths.sh
source "$repo_root/deploy/lib_iso_paths.sh"

usage() {
    printf 'uso: %s <1|2> [caminho-da-iso]\n' "$0" >&2
}

wifi_interface_type() {
    local interface="$1" info key value
    command -v iw >/dev/null || {
        printf 'iw ausente; esperado para validar a interface Wi-Fi %s\n' "$interface" >&2
        return 1
    }
    info="$(iw dev "$interface" info)" || {
        printf 'interface Wi-Fi invalida ou inacessivel: %s\n' "$interface" >&2
        return 1
    }
    while read -r key value _; do
        [[ "$key" == "type" ]] && { printf '%s\n' "$value"; return 0; }
    done <<< "$info"
    printf 'tipo ausente na interface Wi-Fi %s; esperado managed\n' "$interface" >&2
    return 1
}

wifi_global_ipv4() {
    local interface="$1" output line cidr
    local -a addresses=()
    output="$(ip -4 -o address show dev "$interface" scope global up)" || {
        printf 'nao foi possivel consultar IPv4 global de %s\n' "$interface" >&2
        return 1
    }
    while read -r line; do
        [[ -n "$line" ]] || continue
        read -r _ _ _ cidr _ <<< "$line"
        addresses+=("${cidr%%/*}")
    done <<< "$output"
    (( ${#addresses[@]} == 1 )) || {
        printf '%s deve ter exatamente um IPv4 global; encontrei %s\n' \
            "$interface" "${#addresses[@]}" >&2
        return 1
    }
    printf '%s\n' "${addresses[0]}"
}

resolve_vm_bind_address() {
    local interface="${SPLIT_VM_WIFI_INTERFACE:-}"
    [[ -n "$interface" ]] || { printf '127.0.0.1\n'; return 0; }
    local interface_type
    interface_type="$(wifi_interface_type "$interface")" || return 1
    [[ "$interface_type" == "managed" ]] || {
        printf 'interface %s tem tipo %s; esperado managed\n' \
            "$interface" "$interface_type" >&2
        return 1
    }
    wifi_global_ipv4 "$interface"
}

vm_wan_netdev() {
    local node="$1" bind_address="$2"
    local ssh_port=$((2220 + node)) rpc_port=$((50050 + node))
    local http_port=$((8080 + node))
    printf 'user,id=wan,hostfwd=tcp:%s:%s-:22,' "$bind_address" "$ssh_port"
    printf 'hostfwd=tcp:%s:%s-:50052,' "$bind_address" "$rpc_port"
    printf 'hostfwd=tcp:%s:%s-:8080\n' "$bind_address" "$http_port"
}

# A LAN entre VMs era socket multicast (230.0.0.1): depende da rota multicast
# do host, que com VPN/ethernet policy-routing sai pela interface errada e os
# frames nunca voltam para a outra VM (ARP FAILED, RX=0). listen/connect por
# loopback nao depende de rota nenhuma; por isso o no 1 precisa subir primeiro.
vm_lan_netdev() {
    local node="$1" port="${SPLIT_VM_LAN_PORT:-1234}"
    if [[ "$node" == "1" ]]; then
        printf 'socket,id=lan,listen=127.0.0.1:%s\n' "$port"
    else
        printf 'socket,id=lan,connect=127.0.0.1:%s\n' "$port"
    fi
}

main() {
    local node="${1:-}"
    [[ "$node" == "1" || "$node" == "2" ]] || { usage; exit 1; }
    [[ -e /dev/kvm ]] || { printf '/dev/kvm ausente; esperado KVM habilitado\n' >&2; exit 1; }
    # Sem o "|| exit", o antigo "exit 1" de latest_iso morria na subshell da
    # substituicao de comando e o erro real virava "ISO invalida: ".
    local iso_path="${2:-}"
    [[ -n "$iso_path" ]] || iso_path="$(latest_iso "$repo_root")" || exit 1
    [[ -f "$iso_path" ]] || { printf 'ISO invalida: %s\n' "$iso_path" >&2; exit 1; }

    local ssh_port=$((2220 + node))
    local rpc_port=$((50050 + node))
    local http_port=$((8080 + node))
    local private_ip="192.168.76.$((9 + node))"
    local bind_address wan_netdev lan_netdev
    bind_address="$(resolve_vm_bind_address)" || exit 1
    wan_netdev="$(vm_wan_netdev "$node" "$bind_address")"
    lan_netdev="$(vm_lan_netdev "$node")"

    printf 'node %s: configure a LAN privada com:\n' "$node"
    printf '  ip -brief link\n  ip address add %s/24 dev enp0s4\n' "$private_ip"
    [[ "$node" != "1" ]] || printf 'LAN listen em 127.0.0.1:%s; suba o node 1 antes do 2.\n' \
        "${SPLIT_VM_LAN_PORT:-1234}"
    printf 'SSH exposto em %s:%s\n' "$bind_address" "$ssh_port"
    printf 'RPC exposto em %s:%s\n' "$bind_address" "$rpc_port"
    printf 'HTTP exposto em %s:%s\n' "$bind_address" "$http_port"
    [[ "$bind_address" == "127.0.0.1" ]] || \
        printf 'AVISO: RPC sem autenticacao exposto na LAN confiavel.\n'
    printf 'saida do QEMU: Ctrl-a x\n'

    # A porta RPC tambem sai no host: assim da para apontar um master direto,
    # por endereco, sem depender do link multicast entre as VMs.
    exec qemu-system-x86_64 \
        -name "split-node-$node" -machine q35,accel=kvm -cpu host \
        -smp "${SPLIT_VM_CPUS:-4}" -m "${SPLIT_VM_MEMORY_MB:-4096}" \
        -boot d -cdrom "$iso_path" \
        -netdev "$wan_netdev" \
        -device "virtio-net-pci,netdev=wan,mac=52:54:00:76:0$node:01,addr=0x3" \
        -netdev "$lan_netdev" \
        -device "virtio-net-pci,netdev=lan,mac=52:54:00:76:0$node:02,addr=0x4" \
        -display none -serial "${SPLIT_VM_SERIAL:-mon:stdio}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
