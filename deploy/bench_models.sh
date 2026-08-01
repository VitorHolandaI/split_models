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
# arquivo documenta. Cobre 0.49 GB a 12.11 GB e quatro familias de proposito:
# o interesse esta em onde cada maquina para de aguentar.
models=(
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m"
    "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
    "ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
    "unsloth/Qwen3.5-9B-GGUF:Q6_K"
    "unsloth/Ornith-1.0-9B-GGUF:Q6_K"
    "unsloth/gemma-4-12b-it-GGUF:Q5_K_M"
    "ggml-org/gpt-oss-20b-GGUF:MXFP4"
)

usage() {
    printf 'uso: %s [saida.csv] [host:porta-do-worker]\n' "$0" >&2
    printf '  sem worker mede so esta maquina\n' >&2
    printf '  com worker mede tambem o split, razoes 100/0 e 90/10\n' >&2
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

    for model in "${models[@]}"; do
        run_case "$model" "local" "$output_csv"
        [[ -n "$worker" ]] || continue
        # 100/0 isola o custo da rede: nenhum trabalho local, so o pedagio de
        # trazer os logits a cada token. 90/10 e a melhor razao medida ate agora.
        run_case "$model" "rpc-100/0" "$output_csv" --rpc "$worker" -sm layer -ts 100/0
        run_case "$model" "rpc-90/10" "$output_csv" --rpc "$worker" -sm layer -ts 90/10
    done

    printf '\nresultados em %s\n' "$output_csv" >&2
    printf 'quem coube e quem nao: %s\n' "$output_csv.status" >&2
}

main "$@"
