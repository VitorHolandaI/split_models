#!/usr/bin/env bash
set -euo pipefail

# Compila o llama.cpp fixado e instala numa arvore local, sem gerar ISO.
#
# Existe para as maquinas que nao vao bootar a appliance: clona o repositorio,
# roda isto, e o host vira um worker RPC ou um master. E o mesmo commit e as
# mesmas opcoes de build da imagem -- o backend RPC nao negocia versao entre
# master e worker, entao misturar builds falha em runtime, longe da causa.
#
# RPC nao tem autenticacao nem criptografia. Use somente em LAN isolada.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${SPLIT_LLAMA_BUILD_ROOT:-$repo_root/build/llamacpp-rpc}"
# shellcheck source=deploy/lib_llamacpp_build.sh
source "$repo_root/deploy/lib_llamacpp_build.sh"

usage() {
    printf 'uso: %s [prefixo-de-instalacao]\n' "$0" >&2
    printf '  padrao: %s\n' "$llama_destdir" >&2
}

print_next_steps() {
    local prefix="$1"
    printf '\nBinarios em %s/usr/bin\n' "$prefix"
    printf 'Para usar sem instalar no sistema:\n'
    printf '  export LD_LIBRARY_PATH=%s/usr/lib\n' "$prefix"
    printf '  export PATH=%s/usr/bin:$PATH\n\n' "$prefix"
    printf 'Worker (a maquina que empresta GPU ou CPU):\n'
    printf '  ggml-rpc-server --host 0.0.0.0 --port 50052\n\n'
    printf 'Master (a maquina que roda a conversa):\n'
    printf '  llama-cli --rpc <ip-do-worker>:50052 --list-devices\n\n'
    printf 'Confira que a GPU aparece em --list-devices. Se nao aparecer, falta\n'
    printf 'a ICD Vulkan: vulkan-intel, vulkan-radeon ou nvidia-utils.\n'
}

main() {
    [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }
    require_llama_build_commands
    fetch_llama_source
    build_llama_binaries
    [[ -n "${1:-}" ]] && llama_destdir="$1"
    install_llama_to_destdir
    print_next_steps "$llama_destdir"
}

main "$@"
