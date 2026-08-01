#!/usr/bin/env bash
set -euo pipefail

# Reproduz o caminho de boot do LCC sem depender da maquina do amigo: a ISO vai
# como arquivo dentro de uma particao ext4, e um GRUB externo (que nao podemos
# editar no cenario real) a inicia com o layout Ubuntu.
#
# O hospedeiro precisa ser ext4, nao outra ISO. Com ISO aninhada o filesystem
# hospedeiro seria iso9660, o unico modulo que o shim carregava historicamente,
# e o teste passaria enquanto o cenario real falharia.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=deploy/lib_iso_paths.sh
source "$repo_root/deploy/lib_iso_paths.sh"

# Sobrescrito em main com o nome real da ISO. O nome importa: a varredura sem
# iso-scan/filename usa o glob split-models-*.iso (casper-compat:131), que exige
# o hifen antes da data. Um nome generico faria a variante scan falhar por culpa
# do harness.
iso_in_guest="/isos/split-models.iso"
work_dir=""

usage() {
    printf 'uso: %s [ext4|alias|scan] [caminho-da-iso]\n' "$0" >&2
    printf '  ext4   caminho LCC completo (padrao)\n' >&2
    printf '  alias  initrd (loop)/casper/initrd.lz, valida symlinks Rock Ridge\n' >&2
    printf '  scan   sem iso-scan/filename, exercita a varredura por nome\n' >&2
}

cleanup() {
    [[ -n "$work_dir" ]] && rm -rf -- "$work_dir"
}

# mke2fs -d popula o filesystem a partir de um diretorio sem mount e sem root,
# ao contrario de mkfs seguido de "mount -o loop".
build_host_disk() {
    local iso_path="$1" disk_img="$2" stage size_mb
    stage="$work_dir/ext4-stage"
    mkdir -p "$stage$(dirname -- "$iso_in_guest")"
    cp -- "$iso_path" "$stage$iso_in_guest"
    size_mb=$(( $(stat -c %s -- "$iso_path") / 1048576 + 512 ))
    mke2fs -q -t ext4 -L SPLITHOST -d "$stage" "$disk_img" "${size_mb}m"
}

# console=ttyS0 e acrescimo do harness para captura headless; nao toca em
# nenhum caminho do shim.
grub_boot_lines() {
    local variant="$1" initrd_name="initrd" iso_scan="iso-scan/filename=$iso_in_guest"
    [[ "$variant" == "alias" ]] && initrd_name="initrd.lz"
    [[ "$variant" == "scan" ]] && iso_scan=""
    printf '    linux (loop)/casper/vmlinuz boot=casper %s console=ttyS0\n' "$iso_scan"
    printf '    initrd (loop)/casper/%s\n' "$initrd_name"
}

write_grub_config() {
    local variant="$1" grub_root="$2"
    mkdir -p "$grub_root/boot/grub"
    {
        printf 'serial --unit=0 --speed=115200\n'
        printf 'terminal_input serial console\nterminal_output serial console\n'
        printf 'set timeout=0\nset default=0\n'
        printf 'menuentry "casper-compat" {\n'
        printf '    search --no-floppy --file --set=root %s\n' "$iso_in_guest"
        printf '    loopback loop %s\n' "$iso_in_guest"
        grub_boot_lines "$variant"
        printf '}\n'
    } > "$grub_root/boot/grub/grub.cfg"
}

build_grub_iso() {
    local variant="$1" grub_iso="$2" grub_root="$work_dir/grub-root"
    write_grub_config "$variant" "$grub_root"
    grub-mkrescue -o "$grub_iso" "$grub_root" >/dev/null
}

print_acceptance_criteria() {
    printf 'variante %s: procure no serial, nesta ordem:\n' "$1"
    printf '  casper-compat: %s em /dev/vda -> /dev/loopN\n' "$iso_in_guest"
    printf '  :: Mounting /dev/loopN to /run/archiso/bootmnt\n'
    printf '  split-models.service ativo e login disponivel\n'
    printf 'saida do QEMU: Ctrl-a x\n'
}

run_vm() {
    local variant="$1" grub_iso="$2" disk_img="$3"
    qemu-system-x86_64 \
        -name "casper-$variant" -machine q35,accel=kvm -cpu host \
        -smp "${SPLIT_VM_CPUS:-4}" -m "${SPLIT_VM_MEMORY_MB:-4096}" \
        -boot d -drive "file=$grub_iso,media=cdrom" \
        -drive "file=$disk_img,format=raw,if=virtio" \
        -display none -serial "${SPLIT_VM_SERIAL:-mon:stdio}"
}

main() {
    local variant="${1:-ext4}"
    case "$variant" in ext4|alias|scan) ;; *) usage; exit 1 ;; esac
    [[ -e /dev/kvm ]] || { printf '/dev/kvm ausente; esperado KVM habilitado\n' >&2; exit 1; }
    local iso_path="${2:-}"
    [[ -n "$iso_path" ]] || iso_path="$(latest_iso "$repo_root")" || exit 1
    [[ -f "$iso_path" ]] || { printf 'ISO invalida: %s\n' "$iso_path" >&2; exit 1; }
    iso_in_guest="/isos/$(basename -- "$iso_path")"

    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/casper-boot-test.XXXXXX")"
    trap cleanup EXIT
    build_host_disk "$iso_path" "$work_dir/host-ext4.img"
    build_grub_iso "$variant" "$work_dir/grub.iso"
    print_acceptance_criteria "$variant"
    run_vm "$variant" "$work_dir/grub.iso" "$work_dir/host-ext4.img"
}

# Executa so quando chamado direto, para que os testes possam dar source e
# exercitar grub_boot_lines sem subir VM.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi

# vim: set ft=sh:
