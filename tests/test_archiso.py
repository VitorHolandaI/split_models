from pathlib import Path
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

        self.assertIn("HF_HOME=/run/split-models/huggingface", environment)
        self.assertIn("/run/split-models/benchmark-results", session)
        self.assertNotIn("SPLIT_DATA", environment + session)

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
        self.assertIn("split-cluster nodes", commands)
        self.assertIn("split-cluster benchmarks", commands)

    def test_build_enables_archiso_network_services(self):
        build_script = (ROOT / "deploy" / "build_archiso.sh").read_text(
            encoding="utf-8")

        self.assertIn("multi-user.target.wants/systemd-networkd.service", build_script)
        self.assertIn("multi-user.target.wants/systemd-resolved.service", build_script)
        self.assertIn("systemd-networkd-wait-online.service", build_script)
        self.assertIn("sockets.target.wants/systemd-networkd.socket", build_script)
        self.assertIn("copy_releng_link etc/resolv.conf", build_script)


if __name__ == "__main__":
    unittest.main()
