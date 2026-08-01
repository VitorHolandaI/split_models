import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARCHISO = ROOT / "deploy" / "archiso"
# O corpo do build vive na lib desde que a variante llamacpp-rpc passou a
# compartilha-lo; build_archiso.sh so declara a configuracao da variante.
BUILD_LIB = ROOT / "deploy" / "lib_archiso_build.sh"


class ArchisoInstallFlowTest(unittest.TestCase):
    def test_build_creates_fresh_environment_without_copying_local_venv(self):
        packages = (ARCHISO / "packages.x86_64").read_text(encoding="utf-8").splitlines()
        build_lib = BUILD_LIB.read_text(encoding="utf-8")

        self.assertIn("uv", packages)
        self.assertIn("uv sync", build_lib)
        self.assertIn("--no-editable", build_lib)
        self.assertIn('"$repo_root/docs/protocol.md"', build_lib)
        self.assertNotIn('"$repo_root/.venv"', build_lib)

    def test_build_replaces_iso_only_after_success(self):
        build_lib = BUILD_LIB.read_text(encoding="utf-8")

        build = build_lib.index("sudo mkarchiso")
        discard_old = build_lib.index("discard_old_images\n    mv")
        self.assertLess(build, discard_old)
        self.assertIn('-o "$build_output_dir"', build_lib)

    def test_runtime_state_uses_ram(self):
        environment = (ARCHISO / "airootfs" / "etc" / "split-models.env").read_text(
            encoding="utf-8")
        session = (ARCHISO / "airootfs" / "usr" / "local" / "bin" /
                   "split-models-session").read_text(encoding="utf-8")
        python_wrapper = (ARCHISO / "airootfs" / "usr" / "local" / "bin" /
                          "split-models-python").read_text(encoding="utf-8")

        self.assertIn("HF_HOME=/run/split-models/huggingface", environment)
        self.assertIn("/run/split-models/benchmark-results", session)
        self.assertNotIn("SPLIT_DATA", environment + session)
        self.assertIn("cd /opt/split-models", python_wrapper)

    def test_worker_starts_after_network_without_install_service(self):
        system_dir = ARCHISO / "airootfs" / "etc" / "systemd" / "system"
        service = (system_dir / "split-models.service").read_text(encoding="utf-8")

        self.assertIn("After=network-online.target", service)
        self.assertNotIn("split-models-install.service", service)
        self.assertFalse((system_dir / "split-models-install.service").exists())

    def test_vm_launcher_does_not_attach_persistent_storage(self):
        launcher = (ROOT / "deploy" / "run_test_vm.sh").read_text(encoding="utf-8")

        self.assertNotIn("SPLIT_DATA", launcher)
        self.assertNotIn("-drive", launcher)
        self.assertIn('SPLIT_VM_MEMORY_MB:-4096', launcher)

    def test_root_home_has_master_shortcut(self):
        shortcut = ARCHISO / "airootfs" / "root" / "split-master.sh"
        guide = ARCHISO / "airootfs" / "root" / "README-split-models.txt"
        profile = (ARCHISO / "profiledef.sh").read_text(encoding="utf-8")

        script = shortcut.read_text(encoding="utf-8")
        commands = guide.read_text(encoding="utf-8")
        self.assertIn("split-cluster master", script)
        self.assertIn("--discover", script)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", script)
        self.assertNotIn("192.168.122.255", script)
        self.assertIn('["/root/split-master.sh"]="0:0:755"', profile)
        self.assertIn("/root/split-master.sh", commands)
        self.assertIn("/opt/split-models/docs/protocol.md", commands)
        self.assertIn("split-cluster nodes", commands)
        self.assertIn("split-cluster benchmarks", commands)

    def test_physical_console_autologs_in_and_opens_bpytop(self):
        packages = (ARCHISO / "packages.x86_64").read_text(encoding="utf-8").splitlines()
        autologin = (ARCHISO / "airootfs" / "etc" / "systemd" / "system" /
                     "getty@tty1.service.d" / "autologin.conf")
        profile = ARCHISO / "airootfs" / "root" / ".bash_profile"
        service = (ARCHISO / "airootfs" / "etc" / "systemd" / "system" /
                   "split-models.service")
        tmux_launcher = (ARCHISO / "airootfs" / "usr" / "local" / "bin" /
                         "split-models-tmux")
        monitor = (ARCHISO / "airootfs" / "usr" / "local" / "bin" /
                   "split-models-monitor")

        self.assertIn("bpytop", packages)
        self.assertIn("--autologin root", autologin.read_text(encoding="utf-8"))
        startup = profile.read_text(encoding="utf-8")
        self.assertIn("/dev/tty1", startup)
        self.assertIn("tmux attach-session -t split-models", startup)
        self.assertNotIn("bpytop", startup)
        self.assertIn("split-models-tmux", service.read_text(encoding="utf-8"))
        layout = tmux_launcher.read_text(encoding="utf-8")
        self.assertIn("new-session", layout)
        self.assertIn("split-window -h -p 35", layout)
        monitor_script = monitor.read_text(encoding="utf-8")
        self.assertIn("bpytop", monitor_script)
        self.assertIn("exec /usr/bin/bash", monitor_script)

    def test_tmux_status_keeps_role_node_id_and_ip_visible(self):
        bin_dir = ARCHISO / "airootfs" / "usr" / "local" / "bin"
        status_script = bin_dir / "split-models-status"
        session_script = (bin_dir / "split-models-session").read_text(encoding="utf-8")
        tmux_script = (bin_dir / "split-models-tmux").read_text(encoding="utf-8")
        master_script = (ARCHISO / "airootfs" / "root" /
                         "split-master.sh").read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as runtime:
            Path(runtime, "role").write_text("worker\n", encoding="utf-8")
            Path(runtime, "node-id").write_text("node-abcd1234\n", encoding="utf-8")
            result = subprocess.run(
                ["bash", str(status_script)], check=True, capture_output=True, text=True,
                env=os.environ | {"SPLIT_RUNTIME_DIR": runtime,
                                  "SPLIT_NODE_IP": "192.168.50.12"})

        self.assertEqual(
            result.stdout.strip(),
            "role=worker | node=node-abcd1234 | ip=192.168.50.12")
        self.assertIn("--node-id", session_script)
        self.assertIn("/run/split-models/node-id", session_script)
        self.assertIn("status-right", tmux_script)
        self.assertIn("split-models-status", tmux_script)
        self.assertIn("printf 'master", master_script)
        self.assertIn("/run/split-models/role", master_script)

    def test_build_enables_archiso_network_services(self):
        build_lib = BUILD_LIB.read_text(encoding="utf-8")

        self.assertIn("multi-user.target.wants/systemd-networkd.service", build_lib)
        self.assertIn("multi-user.target.wants/systemd-resolved.service", build_lib)
        self.assertIn("systemd-networkd-wait-online.service", build_lib)
        self.assertIn("sockets.target.wants/systemd-networkd.socket", build_lib)
        self.assertIn("copy_releng_link etc/resolv.conf", build_lib)

    def test_iso_accepts_casper_style_loopback_boot(self):
        hook = ARCHISO / "airootfs" / "etc" / "initcpio" / "hooks" / "casper-compat"
        hook_install = (ARCHISO / "airootfs" / "etc" / "initcpio" /
                        "install" / "casper-compat")
        mkinitcpio = (ARCHISO / "airootfs" / "etc" / "mkinitcpio.conf.d" /
                      "archiso.conf").read_text(encoding="utf-8")
        profile = (ARCHISO / "profiledef.sh").read_text(encoding="utf-8")
        build_script = BUILD_LIB.read_text(encoding="utf-8")
        hook_script = hook.read_text(encoding="utf-8")

        parsed = subprocess.run(
            ["bash", "-c", (
                f"source {hook}; "
                "casper_iso_path_from_cmdline 'boot=casper "
                "iso-scan/filename=/isos/split-models.iso'")],
            check=False, capture_output=True, text=True)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.assertEqual(parsed.stdout.strip(), "/isos/split-models.iso")
        mount_options = subprocess.run(
            ["bash", "-c", (
                f"source {hook}; "
                "for fs in ext4 btrfs xfs ntfs3; do "
                "casper_mount_options $fs; printf '\\n'; done")],
            check=True, capture_output=True, text=True)
        self.assertEqual(
            mount_options.stdout.splitlines(),
            ["x-initrd.mount,ro,noload",
             "x-initrd.mount,ro,rescue=nologreplay",
             "x-initrd.mount,ro,norecovery",
             "x-initrd.mount,ro"])
        self.assertTrue(hook_install.is_file())
        self.assertIn("losetup --find --show --read-only", hook_script)
        self.assertIn("archisodevice=", hook_script)
        self.assertIn("ro,nologreplay", hook_script)
        # The host partition must land where archiso_loop_mnt puts it, so the
        # loop device survives switch_root and shutdown unmounts in order.
        self.assertIn("/run/archiso/img_dev", hook_script)
        self.assertNotIn("/run/casper-compat", hook_script)
        # archiso re-reads archisodevice via getarg, which greps /tmp/cmdline;
        # an exported variable would be clobbered. See hooks/archiso:229.
        self.assertIn("/tmp/cmdline", hook_script)
        for filesystem in ("ext4", "btrfs", "xfs", "vfat", "exfat", "ntfs3"):
            self.assertIn(filesystem, hook_script)
            self.assertIn(filesystem, hook_install.read_text(encoding="utf-8"))
        self.assertIn("casper-compat archiso", mkinitcpio)
        self.assertIn('["/etc/initcpio/hooks/casper-compat"]="0:0:755"', profile)
        self.assertIn("/casper/vmlinuz", build_script)
        self.assertIn("/casper/initrd", build_script)
        self.assertIn("-boot_image any replay", build_script)
        self.assertIn('casper_stage="$(mktemp -d', build_script)
        self.assertIn('casper_output_dir="$(mktemp -d', build_script)
        # A imagem publicada e a remasterizada, nunca a saida crua do mkarchiso.
        self.assertIn('mv "$casper_image" "$output_dir/"', build_script)

    def test_casper_boot_harness_hosts_iso_on_ext4(self):
        harness = ROOT / "deploy" / "run_casper_boot_test.sh"
        harness_script = harness.read_text(encoding="utf-8")

        self.assertTrue(harness.is_file())
        self.assertTrue(os.access(harness, os.X_OK))
        # ISO dentro de ISO faria o hospedeiro ser iso9660 e o teste passaria
        # enquanto o cenario real (LCC copia para ext4/NTFS) falharia.
        self.assertIn("mke2fs -q -t ext4", harness_script)
        self.assertNotIn("mkarchiso", harness_script)
        self.assertIn("lib_iso_paths.sh", harness_script)
        self.assertIn(
            "lib_iso_paths.sh",
            (ROOT / "deploy" / "run_test_vm.sh").read_text(encoding="utf-8"))

        boot_lines = subprocess.run(
            ["bash", "-c", (
                f"source {harness}; "
                "iso_in_guest=/isos/split-models.iso; "
                "for v in ext4 alias scan; do grub_boot_lines $v; done")],
            check=True, capture_output=True, text=True)
        emitted = [line.strip() for line in boot_lines.stdout.splitlines()]
        self.assertEqual(emitted, [
            "linux (loop)/casper/vmlinuz boot=casper "
            "iso-scan/filename=/isos/split-models.iso console=ttyS0",
            "initrd (loop)/casper/initrd",
            "linux (loop)/casper/vmlinuz boot=casper "
            "iso-scan/filename=/isos/split-models.iso console=ttyS0",
            "initrd (loop)/casper/initrd.lz",
            "linux (loop)/casper/vmlinuz boot=casper  console=ttyS0",
            "initrd (loop)/casper/initrd"])


if __name__ == "__main__":
    unittest.main()
