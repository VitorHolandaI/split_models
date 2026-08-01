#!/usr/bin/env bash
# Corpo comum dos builds Archiso, compartilhado pelo build estavel e pelas
# variantes experimentais. Feito para "source", nao para execucao direta.
#
# O chamador define, antes de chamar build_archiso_variant:
#   repo_root       raiz do repositorio
#   variant_name    "stable" ou o nome da variante experimental
#   profile_source  profile Archiso base
#   overlay_source  diretorio sobreposto ao profile ("" no estavel)
#   build_root      onde criar os diretorios temporarios
#   output_dir      destino final da ISO
#   iso_prefix      prefixo do arquivo, tem que casar com iso_name do profiledef

releng_root=/usr/share/archiso/configs/releng/airootfs

require_build_commands() {
    local command_name
    for command_name in mkarchiso sudo trash uv xorriso; do
        command -v "$command_name" >/dev/null || {
            printf 'comando ausente: %s\n' "$command_name" >&2
            exit 1
        }
    done
}

# A ISO estavel e o artefato pinado do projeto. Nenhuma variante experimental
# pode escrever em dist/ direto, usar o prefixo split-models ou sair do
# repositorio; sem estas guardas um glob de limpeza apagaria a ISO estavel.
assert_variant_isolation() {
    [[ "$variant_name" == "stable" ]] && return 0
    [[ "$output_dir" != "$repo_root/dist" ]] || {
        printf 'variante %s nao pode escrever direto em dist/\n' "$variant_name" >&2
        exit 1
    }
    [[ "$iso_prefix" != "split-models" ]] || {
        printf 'variante %s nao pode usar o prefixo split-models\n' "$variant_name" >&2
        exit 1
    }
    [[ "$output_dir" == "$repo_root/dist/"* ]] || {
        printf 'output_dir fora de %s/dist: %s\n' "$repo_root" "$output_dir" >&2
        exit 1
    }
}

copy_releng_link() {
    local relative_path="$1"
    local source_path="$releng_root/$relative_path"
    local target_path="$staged_profile/airootfs/$relative_path"
    [[ -L "$source_path" ]] || {
        printf 'link Archiso ausente: %s\n' "$source_path" >&2
        exit 1
    }
    mkdir -p "$(dirname -- "$target_path")"
    cp -a "$source_path" "$target_path"
}

copy_releng_network_links() {
    local wants=etc/systemd/system
    copy_releng_link "$wants/multi-user.target.wants/systemd-networkd.service"
    copy_releng_link "$wants/multi-user.target.wants/systemd-resolved.service"
    copy_releng_link "$wants/network-online.target.wants/systemd-networkd-wait-online.service"
    copy_releng_link "$wants/sockets.target.wants/systemd-networkd.socket"
    copy_releng_link "$wants/dbus-org.freedesktop.network1.service"
    copy_releng_link "$wants/dbus-org.freedesktop.resolve1.service"
    copy_releng_link etc/resolv.conf
}

# packages.x86_64 do overlay e aditivo: a variante acrescenta pacotes em vez de
# substituir a lista base, senao um esquecimento removeria uv ou bpytop.
merge_overlay_packages() {
    local base_packages="$1"
    [[ -f "$overlay_source/packages.x86_64" ]] || return 0
    printf '%s\n' "$base_packages" > "$staged_profile/packages.x86_64"
    cat "$overlay_source/packages.x86_64" >> "$staged_profile/packages.x86_64"
}

# O overlay acrescenta ao profiledef em vez de substitui-lo, senao a variante
# teria que duplicar file_permissions e os campos de boot do profile base e
# ficaria dessincronizada em silencio. mkarchiso:46 faz
# "declare -A file_permissions=()" antes de dar source, entao o += funciona.
append_overlay_profiledef() {
    [[ -f "$overlay_source/profiledef.append.sh" ]] || return 0
    cat "$overlay_source/profiledef.append.sh" >> "$staged_profile/profiledef.sh"
    rm -f "$staged_profile/profiledef.append.sh"
}

stage_profile() {
    cp -a "$profile_source/." "$staged_profile/"
    [[ -n "$overlay_source" ]] || return 0
    [[ -d "$overlay_source" ]] || {
        printf 'overlay ausente: %s\n' "$overlay_source" >&2
        exit 1
    }
    local base_packages
    base_packages="$(cat "$staged_profile/packages.x86_64")"
    cp -a "$overlay_source/." "$staged_profile/"
    merge_overlay_packages "$base_packages"
    append_overlay_profiledef
}

# Descasamento entre iso_prefix e o profiledef faria a limpeza de imagens
# antigas usar um glob que nunca casa, acumulando ISOs em silencio.
assert_profile_iso_name() {
    local staged_name
    staged_name="$(grep -oP '^iso_name="\K[^"]+' "$staged_profile/profiledef.sh")"
    [[ "$staged_name" == "$iso_prefix" ]] || {
        printf 'iso_name do profile (%s) difere de iso_prefix (%s)\n' \
            "$staged_name" "$iso_prefix" >&2
        exit 1
    }
}

stage_application() {
    app_root="$staged_profile/airootfs/opt/split-models"
    mkdir -p "$app_root/docs"
    cp -a \
        "$repo_root/benchmarks.toml" \
        "$repo_root/pyproject.toml" \
        "$repo_root/uv.lock" \
        "$repo_root/split_lm" \
        "$repo_root/tools" \
        "$app_root/"
    cp -a "$repo_root/docs/protocol.md" "$app_root/docs/"
    ln -s /run/split-models/benchmark-results "$app_root/benchmark-results"
    uv sync --project "$app_root" --python /usr/bin/python \
        --frozen --no-dev --no-editable
}

enable_worker_service() {
    local wants_dir="$staged_profile/airootfs/etc/systemd/system/multi-user.target.wants"
    mkdir -p "$wants_dir"
    ln -s ../split-models.service "$wants_dir/split-models.service"
}

# Bootloaders fora do nosso controle (LCC) esperam o layout Casper do Ubuntu:
# /casper/vmlinuz e /casper/initrd. Publicamos esses caminhos na ISO; o hook
# initcpio casper-compat cuida da outra metade, traduzindo iso-scan/filename.
stage_casper_aliases() {
    local source_iso="$1"
    mkdir -p "$casper_stage/casper"
    xorriso -osirrox on -indev "$source_iso" \
        -extract /arch/boot/x86_64/vmlinuz-linux "$casper_stage/casper/vmlinuz" \
        -extract /arch/boot/x86_64/initramfs-linux.img "$casper_stage/casper/initrd"
    ln -s vmlinuz "$casper_stage/casper/vmlinuz.efi"
    local initrd_alias
    for initrd_alias in initrd.img initrd.lz initrd.gz; do
        ln -s initrd "$casper_stage/casper/$initrd_alias"
    done
}

# "-boot_image any replay" preserva El Torito, EFI e MBR isohibrido da imagem
# original; sem isso o remaster deixa de dar boot nativo em USB.
add_casper_aliases() {
    local source_iso="$1"
    casper_image="$casper_output_dir/$(basename -- "$source_iso")"
    stage_casper_aliases "$source_iso"
    xorriso -indev "$source_iso" -outdev "$casper_image" \
        -boot_image any replay \
        -follow off \
        -map "$casper_stage/casper" /casper
    [[ -f "$casper_image" ]] || {
        printf 'remaster Casper falhou: %s\n' "$casper_image" >&2
        exit 1
    }
}

make_build_directories() {
    mkdir -p "$build_root" "$output_dir"
    staged_profile="$(mktemp -d "$build_root/archiso-profile.XXXXXX")"
    work_dir="$(mktemp -d "$build_root/archiso-work.XXXXXX")"
    build_output_dir="$(mktemp -d "$build_root/archiso-output.XXXXXX")"
    casper_stage="$(mktemp -d "$build_root/casper-stage.XXXXXX")"
    casper_output_dir="$(mktemp -d "$build_root/casper-output.XXXXXX")"
}

discard_build_directories() {
    sudo /usr/bin/trash \
        "$staged_profile" "$work_dir" "$build_output_dir" \
        "$casper_stage" "$casper_output_dir"
}

single_built_image() {
    shopt -s nullglob
    local built_images=("$build_output_dir/$iso_prefix"-*.iso)
    (( ${#built_images[@]} == 1 )) || {
        printf 'esperava uma ISO %s-*, encontrei %s em %s\n' \
            "$iso_prefix" "${#built_images[@]}" "$build_output_dir" >&2
        exit 1
    }
    printf '%s\n' "${built_images[0]}"
}

# O glob e sempre ancorado em output_dir e iso_prefix, os dois campos que
# assert_variant_isolation ja validou.
discard_old_images() {
    shopt -s nullglob
    local old_images=("$output_dir/$iso_prefix"-*.iso)
    (( ${#old_images[@]} > 0 )) || return 0
    trash "${old_images[@]}"
}

build_archiso_variant() {
    require_build_commands
    assert_variant_isolation
    make_build_directories
    trap discard_build_directories EXIT
    stage_profile
    assert_profile_iso_name
    stage_application
    copy_releng_network_links
    enable_worker_service
    sudo mkarchiso -v -w "$work_dir" -o "$build_output_dir" "$staged_profile"
    add_casper_aliases "$(single_built_image)"
    discard_old_images
    mv "$casper_image" "$output_dir/"
}

# vim: set ft=sh:
