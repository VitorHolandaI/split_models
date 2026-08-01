#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile_source="$repo_root/deploy/archiso"
build_root="$repo_root/build"
output_dir="$repo_root/dist"
releng_root=/usr/share/archiso/configs/releng/airootfs

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

for command_name in mkarchiso sudo trash uv xorriso; do
    command -v "$command_name" >/dev/null || {
        printf 'comando ausente: %s\n' "$command_name" >&2
        exit 1
    }
done

mkdir -p "$build_root" "$output_dir"
staged_profile="$(mktemp -d "$build_root/archiso-profile.XXXXXX")"
work_dir="$(mktemp -d "$build_root/archiso-work.XXXXXX")"
build_output_dir="$(mktemp -d "$build_root/archiso-output.XXXXXX")"
casper_stage="$(mktemp -d "$build_root/casper-stage.XXXXXX")"
casper_output_dir="$(mktemp -d "$build_root/casper-output.XXXXXX")"

cleanup() {
    sudo /usr/bin/trash \
        "$staged_profile" "$work_dir" "$build_output_dir" \
        "$casper_stage" "$casper_output_dir"
}
trap cleanup EXIT

cp -a "$profile_source/." "$staged_profile/"
app_root="$staged_profile/airootfs/opt/split-models"
mkdir -p "$app_root"
cp -a \
    "$repo_root/benchmarks.toml" \
    "$repo_root/pyproject.toml" \
    "$repo_root/uv.lock" \
    "$repo_root/split_lm" \
    "$repo_root/tools" \
    "$app_root/"
mkdir -p "$app_root/docs"
cp -a "$repo_root/docs/protocol.md" "$app_root/docs/"
ln -s /run/split-models/benchmark-results "$app_root/benchmark-results"

uv sync \
    --project "$app_root" \
    --python /usr/bin/python \
    --frozen \
    --no-dev \
    --no-editable

copy_releng_link etc/systemd/system/multi-user.target.wants/systemd-networkd.service
copy_releng_link etc/systemd/system/multi-user.target.wants/systemd-resolved.service
copy_releng_link etc/systemd/system/network-online.target.wants/systemd-networkd-wait-online.service
copy_releng_link etc/systemd/system/sockets.target.wants/systemd-networkd.socket
copy_releng_link etc/systemd/system/dbus-org.freedesktop.network1.service
copy_releng_link etc/systemd/system/dbus-org.freedesktop.resolve1.service
copy_releng_link etc/resolv.conf

wants_dir="$staged_profile/airootfs/etc/systemd/system/multi-user.target.wants"
mkdir -p "$wants_dir"
ln -s ../split-models.service "$wants_dir/split-models.service"

sudo mkarchiso -v -w "$work_dir" -o "$build_output_dir" "$staged_profile"

shopt -s nullglob
built_images=("$build_output_dir"/split-models-*.iso)
if (( ${#built_images[@]} != 1 )); then
    printf 'esperava uma ISO, encontrei %s em %s\n' \
        "${#built_images[@]}" "$build_output_dir" >&2
    exit 1
fi

add_casper_aliases "${built_images[0]}"
built_images=("$casper_image")

old_images=("$output_dir"/split-models-*.iso)
if (( ${#old_images[@]} > 0 )); then
    trash "${old_images[@]}"
fi
mv "${built_images[0]}" "$output_dir/"
