#!/usr/bin/bash
set -euo pipefail

# Promove este no a master: descobre os workers RPC da LAN e roda llama-cli
# atraves deles. O broadcast sai da rota padrao, entao funciona em bare metal
# com DHCP sem depender do range do libvirt.

detect_broadcast() {
    local interface broadcast
    interface="$(ip -4 route show default | awk '
        NR == 1 { for (i = 1; i <= NF; i++) if ($i == "dev") print $(i + 1) }')"
    broadcast="$(ip -4 -o address show dev "$interface" scope global | awk '
        { for (i = 1; i <= NF; i++) if ($i == "brd") print $(i + 1) }')"
    printf '%s\n' "${broadcast:-255.255.255.255}"
}

broadcast="${SPLIT_BROADCAST:-$(detect_broadcast)}"
model="${SPLIT_LLAMA_MODEL:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"

install -d -m 0755 /run/split-models
printf 'master\n' > /run/split-models/role
restore_role() {
    printf 'shell\n' > /run/split-models/role
}
trap restore_role EXIT

printf 'master llama.cpp: broadcast=%s model=%s\n' "$broadcast" "$model"
split-llama master \
    --discover \
    --broadcast "$broadcast" \
    --hf "$model" \
    "$@"
