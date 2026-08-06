#!/usr/bin/bash
set -euo pipefail

# Promove este no a master de um turno so: descobre os workers RPC da LAN e
# roda llama-cli atraves deles. O broadcast sai da rota padrao, entao funciona
# em bare metal com DHCP sem depender do range do libvirt.
#
# Atalho interativo, fora do controle de modos: o papel supervisionado
# (distributed-master com llama-server, readiness e rollback) e
# `split-llama mode distributed-master`. Este script nao grava estado porque
# /run/split-models/state.json e a fonte oficial do papel do no.

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

# O download do GGUF vai para o model-cache em tmpfs, como no modo
# supervisionado; /root/.cache estouraria no overlay de 256 MB da live ISO.
export LLAMA_CACHE=/run/split-models/model-cache
split-llama-model-cache

printf 'master llama.cpp: broadcast=%s model=%s\n' "$broadcast" "$model"
split-llama master \
    --discover \
    --broadcast "$broadcast" \
    --hf "$model" \
    "$@"
