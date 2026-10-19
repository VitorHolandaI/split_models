#!/usr/bin/bash
set -euo pipefail

detect_broadcast() {
    local interface broadcast
    interface="$(ip -4 route show default | awk '
        NR == 1 { for (i = 1; i <= NF; i++) if ($i == "dev") print $(i + 1) }')"
    broadcast="$(ip -4 -o address show dev "$interface" scope global | awk '
        { for (i = 1; i <= NF; i++) if ($i == "brd") print $(i + 1) }')"
    printf '%s\n' "${broadcast:-255.255.255.255}"
}

broadcast="${SPLIT_BROADCAST:-$(detect_broadcast)}"
model="${SPLIT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
max_new_tokens="${SPLIT_MAX_NEW_TOKENS:-50}"

install -d -m 0755 /run/split-models
printf 'master\n' > /run/split-models/role
restore_role() {
    printf 'shell\n' > /run/split-models/role
}
trap restore_role EXIT

printf 'master: broadcast=%s model=%s max_new_tokens=%s\n' \
    "$broadcast" "$model" "$max_new_tokens"
split-cluster master \
    --discover \
    --broadcast "$broadcast" \
    --model "$model" \
    --max-new-tokens "$max_new_tokens" \
    "$@"
