from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARCHISO = ROOT / "deploy" / "archiso"


class ArchisoInstallFlowTest(unittest.TestCase):
    def test_build_creates_fresh_environment_without_copying_local_venv(self):
        packages = (ARCHISO / "packages.x86_64").read_text(encoding="utf-8").splitlines()
        build_script = (ROOT / "deploy" / "build_archiso.sh").read_text(encoding="utf-8")

        self.assertIn("uv", packages)
        self.assertIn("uv sync", build_script)
        self.assertIn("--no-editable", build_script)
        self.assertIn('"$repo_root/docs/protocol.md"', build_script)
        self.assertNotIn('"$repo_root/.venv"', build_script)

    def test_build_replaces_iso_only_after_success(self):
        build_script = (ROOT / "deploy" / "build_archiso.sh").read_text(
            encoding="utf-8")

        build = build_script.index("sudo mkarchiso")
        discard_old = build_script.index('trash "${old_images[@]}"')
        self.assertLess(build, discard_old)
        self.assertIn('-o "$build_output_dir"', build_script)

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

        self.assertIn("bpytop", packages)
        self.assertIn("--autologin root", autologin.read_text(encoding="utf-8"))
        startup = profile.read_text(encoding="utf-8")
        self.assertIn("/dev/tty1", startup)
        self.assertIn("bpytop", startup)

    def test_build_enables_archiso_network_services(self):
        build_script = (ROOT / "deploy" / "build_archiso.sh").read_text(
            encoding="utf-8")

        self.assertIn("multi-user.target.wants/systemd-networkd.service", build_script)
        self.assertIn("multi-user.target.wants/systemd-resolved.service", build_script)
        self.assertIn("systemd-networkd-wait-online.service", build_script)
        self.assertIn("sockets.target.wants/systemd-networkd.socket", build_script)
        self.assertIn("copy_releng_link etc/resolv.conf", build_script)

    def test_iso_accepts_casper_style_loopback_boot(self):
        hook = ARCHISO / "airootfs" / "etc" / "initcpio" / "hooks" / "casper-compat"
        hook_install = (ARCHISO / "airootfs" / "etc" / "initcpio" /
                        "install" / "casper-compat")
        mkinitcpio = (ARCHISO / "airootfs" / "etc" / "mkinitcpio.conf.d" /
                      "archiso.conf").read_text(encoding="utf-8")
        profile = (ARCHISO / "profiledef.sh").read_text(encoding="utf-8")
        build_script = (ROOT / "deploy" / "build_archiso.sh").read_text(
            encoding="utf-8")
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
            ["ro,noload", "ro,rescue=nologreplay", "ro,norecovery", "ro"])
        self.assertTrue(hook_install.is_file())
        self.assertIn("losetup --find --show --read-only", hook_script)
        self.assertIn("archisodevice=", hook_script)
        self.assertIn("ro,nologreplay", hook_script)
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
        self.assertIn('built_images=("$casper_image")', build_script)


if __name__ == "__main__":
    unittest.main()
