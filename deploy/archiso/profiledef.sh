#!/usr/bin/env bash
# shellcheck disable=SC2034

iso_name="split-models"
iso_label="ARCH_$(date --date="@${SOURCE_DATE_EPOCH:-$(date +%s)}" +%Y%m)"
iso_publisher="Arch Linux <https://archlinux.org>"
iso_application="Split Models cluster appliance"
iso_version="$(date --date="@${SOURCE_DATE_EPOCH:-$(date +%s)}" +%Y.%m.%d)"
install_dir="arch"
buildmodes=('iso')
bootmodes=('bios.syslinux'
           'uefi.grub')
pacman_conf="pacman.conf"
airootfs_image_type="erofs"
airootfs_image_tool_options=('-zlzma,109' -E 'ztailpacking')
bootstrap_tarball_compression=(xz -9e)
file_permissions=(
  ["/root/split-master.sh"]="0:0:755"
  ["/etc/shadow"]="0:0:400"
  ["/usr/local/bin/split-cluster"]="0:0:755"
  ["/usr/local/bin/split-lm"]="0:0:755"
  ["/usr/local/bin/split-models-python"]="0:0:755"
  ["/usr/local/bin/split-models-session"]="0:0:755"
)
