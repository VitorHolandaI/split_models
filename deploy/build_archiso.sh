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

for command_name in mkarchiso sudo trash uv; do
    command -v "$command_name" >/dev/null || {
        printf 'comando ausente: %s\n' "$command_name" >&2
        exit 1
    }
done

mkdir -p "$build_root" "$output_dir"
staged_profile="$(mktemp -d "$build_root/archiso-profile.XXXXXX")"
work_dir="$(mktemp -d "$build_root/archiso-work.XXXXXX")"
build_output_dir="$(mktemp -d "$build_root/archiso-output.XXXXXX")"

cleanup() {
    sudo /usr/bin/trash "$staged_profile" "$work_dir" "$build_output_dir"
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

old_images=("$output_dir"/split-models-*.iso)
if (( ${#old_images[@]} > 0 )); then
    trash "${old_images[@]}"
fi
mv "${built_images[0]}" "$output_dir/"
