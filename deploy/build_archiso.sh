#!/usr/bin/env bash
set -euo pipefail

# Build estavel. Produz dist/split-models-<data>-x86_64.iso, o artefato pinado
# do projeto. Variantes experimentais ficam em subdiretorios de dist/ e usam
# outro prefixo; ver deploy/build_llamacpp_archiso.sh.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=deploy/lib_archiso_build.sh
source "$repo_root/deploy/lib_archiso_build.sh"

variant_name="stable"
profile_source="$repo_root/deploy/archiso"
overlay_source=""
build_root="$repo_root/build"
output_dir="$repo_root/dist"
iso_prefix="split-models"

build_archiso_variant
