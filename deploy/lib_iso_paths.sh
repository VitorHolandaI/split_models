#!/usr/bin/env bash
# Resolucao de caminho da ISO, compartilhada pelos lancadores QEMU.
# Este arquivo e feito para "source", nao para execucao direta.

# Ultima ISO construida em dist/, em ordem de glob (a data no nome ordena).
# Uso: iso_path="$(latest_iso "$repo_root")" || exit 1
latest_iso() {
    local repo_root="$1"
    local candidates=("$repo_root"/dist/split-models-*.iso)
    if [[ ! -e "${candidates[0]}" ]]; then
        printf 'ISO ausente em %s/dist; execute ./deploy/build_archiso.sh\n' \
            "$repo_root" >&2
        return 1
    fi
    printf '%s\n' "${candidates[${#candidates[@]} - 1]}"
}

# vim: set ft=sh:
