#!/usr/bin/env bash
set -uo pipefail

# Mede a lista de modelos de referencia com llama-bench, local e em split.
#
# Nao usa "set -e" de proposito: modelo grande demais para a maquina faz o
# llama-bench sair com erro, e isso e resultado -- a varredura registra "nao
# coube" e segue para o proximo em vez de abortar a bateria inteira.
#
# E longo. Baixa cada modelo que faltar e roda repeticoes. Deixe rodando.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
llama_prefix="${SPLIT_LLAMA_PREFIX:-$repo_root/build/llamacpp-rpc/llamacpp-src/dest-b10216}"
export LD_LIBRARY_PATH="$llama_prefix/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
llama_bench="$llama_prefix/usr/bin/llama-bench"

# Mesma lista de split_lm/llama_rpc/examples.py, com a quantizacao que aquele
# arquivo documenta. Cobre 0.53 GB a 12.11 GB e quatro familias de proposito:
# o interesse esta em onde cada maquina para de aguentar.
models=(
    "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M"
    "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
    "ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
    "unsloth/Qwen3.5-9B-GGUF:Q6_K"
    "unsloth/Ornith-1.0-9B-GGUF:Q6_K"
    "unsloth/gemma-4-12b-it-GGUF:Q5_K_M"
    "ggml-org/gpt-oss-20b-GGUF:MXFP4"
)

# A ordem de -ts e RPC primeiro, local depois (src/llama.cpp:264 insere os
# workers no inicio do vetor). Entao "10/90" carrega o dispositivo local e
# "100/0" o remoto -- o inverso do que a saida de --list-devices sugere.
#
# Quatro pontos porque a curva importa mais que qualquer ponto isolado: dois
# perto de uma ponta so mostrariam a inclinacao ali. 100/0 isola o pedagio da
# rede, 10/90 mostra o custo de carregar o dispositivo lento, e os do meio
# dizem se a queda e linear.
ratios=("100/0" "90/10" "50/50" "10/90")

# Modelos MoE lem so os experts ativados por token, nao todos os pesos: o
# "A3B"/"A4B" no nome de outros modelos e literalmente "active 3B". Entao
# capacidade cobra os parametros TOTAIS e banda cobra so os ATIVOS.
#
# --n-cpu-moe tira os experts da memoria rapida e deixa la so a atencao, que e
# pequena e usada em todo token. Num host de GPU dedicada isso faz um modelo que
# nao cabe na VRAM passar a caber -- alternativa ao split, nao variacao dele.
#
# Nao ajuda em iGPU: o Vulkan reporta uma=1 nesta maquina, entao "CPU" e "GPU"
# sao a mesma DRAM e mover experts nao muda a banda. Medir mesmo assim, para o
# numero existir em vez de ser suposicao.
moe_models=("ggml-org/gpt-oss-20b-GGUF:MXFP4")
moe_cpu_layers=(99)

is_moe() {
    local candidate="$1" known
    for known in "${moe_models[@]}"; do
        [[ "$candidate" == "$known" ]] && return 0
    done
    return 1
}

usage() {
    printf 'uso: %s [saida.csv] [host:porta-do-worker]\n' "$0" >&2
    printf '  sem worker mede so esta maquina\n' >&2
    printf '  com worker mede tambem o split nas razoes %s\n' "${ratios[*]}" >&2
}

require_llama_bench() {
    [[ -x "$llama_bench" ]] || {
        printf 'llama-bench ausente em %s\n' "$llama_bench" >&2
        printf 'compile antes: ./deploy/build_llamacpp_binaries.sh\n' >&2
        exit 1
    }
}

# Uma linha por tentativa, escrita na hora: a bateria leva horas e uma queda no
# meio nao pode custar as medicoes que ja terminaram.
record_attempt() {
    local model="$1" label="$2" status="$3" output_csv="$4"
    printf '%s,%s,%s\n' "$model" "$label" "$status" >> "$output_csv"
}

run_case() {
    local model="$1" label="$2" output_csv="$3"
    shift 3
    printf '\n=== %s [%s] ===\n' "$model" "$label" >&2
    if "$llama_bench" -hf "$model" -ngl 99 -p 128 -n 32 -r 2 -o csv "$@" \
        | tail -n +2 | sed "s|^|$model,$label,|" >> "$output_csv"; then
        record_attempt "$model" "$label" "ok" "$output_csv.status"
        return 0
    fi
    printf 'nao coube ou falhou: %s [%s]\n' "$model" "$label" >&2
    record_attempt "$model" "$label" "falhou" "$output_csv.status"
}

main() {
    [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
    require_llama_bench
    local output_csv="${1:-benchmark-results/llamacpp-models.csv}"
    local worker="${2:-}"

    mkdir -p "$(dirname -- "$output_csv")"
    : > "$output_csv"
    : > "$output_csv.status"
    printf 'modelo,configuracao,linha_llama_bench\n' >> "$output_csv"
    printf 'modelo,configuracao,status\n' >> "$output_csv.status"

    local ratio layers
    for model in "${models[@]}"; do
        run_case "$model" "local" "$output_csv"
        if is_moe "$model"; then
            for layers in "${moe_cpu_layers[@]}"; do
                run_case "$model" "local-cpumoe-$layers" "$output_csv" -ncmoe "$layers"
            done
        fi
        [[ -n "$worker" ]] || continue
        for ratio in "${ratios[@]}"; do
            run_case "$model" "rpc-$ratio" "$output_csv" \
                --rpc "$worker" -sm layer -ts "$ratio"
        done
    done

    printf '\nresultados em %s\n' "$output_csv" >&2
    printf 'quem coube e quem nao: %s\n' "$output_csv.status" >&2
}

main "$@"
