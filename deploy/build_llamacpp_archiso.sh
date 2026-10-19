#!/usr/bin/env bash
set -euo pipefail

# Build experimental: llama.cpp com backend RPC e Vulkan. Sai em
# dist/llamacpp-rpc/ com prefixo proprio, para que nenhum glob de limpeza
# alcance a ISO estavel em dist/. As guardas estao em
# deploy/lib_archiso_build.sh:assert_variant_isolation.
#
# RPC nao tem autenticacao nem criptografia. Use somente em LAN isolada.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=deploy/lib_archiso_build.sh
source "$repo_root/deploy/lib_archiso_build.sh"

variant_name="llamacpp-rpc"
build_root="$repo_root/build/llamacpp-rpc"

# Precisa vir depois da lib base: sobrescreve o stage_variant_extras sem-op dela
# com a compilacao do llama.cpp. Depende de build_root, definido acima.
# shellcheck source=deploy/lib_llamacpp_build.sh
source "$repo_root/deploy/lib_llamacpp_build.sh"

# Antes de encenar qualquer coisa: sem cmake ou headers Vulkan o build so
# quebraria depois do uv sync, minutos adiante e longe da causa.
require_llama_build_commands
refuse_cuda_for_image

profile_source="$repo_root/deploy/archiso"
overlay_source="$repo_root/deploy/archiso-llamacpp-rpc"
output_dir="$repo_root/dist/llamacpp-rpc"
iso_prefix="split-models-llamacpp-rpc"

build_archiso_variant
