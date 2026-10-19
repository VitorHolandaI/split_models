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
VM_LAUNCHER = ROOT / "deploy" / "run_test_vm.sh"


def run_vm_launcher_helper(script: str) -> subprocess.CompletedProcess[str]:
    """Executa uma funcao do launcher sem iniciar QEMU.

    Exemplo: ``run_vm_launcher_helper("resolve_vm_bind_address")``.
    """
    return subprocess.run(
        ["bash", "-c", f"source {VM_LAUNCHER}\n{script}"],
        check=False, capture_output=True, text=True)


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
        launcher = VM_LAUNCHER.read_text(encoding="utf-8")

        self.assertNotIn("SPLIT_DATA", launcher)
        self.assertNotIn("-drive", launcher)
        self.assertIn('SPLIT_VM_MEMORY_MB:-4096', launcher)


class VmLanForwardingTest(unittest.TestCase):
    def test_default_is_bound_to_loopback_only(self):
        result = run_vm_launcher_helper(
            "unset SPLIT_VM_WIFI_INTERFACE\nresolve_vm_bind_address")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "127.0.0.1")

    def test_managed_wifi_resolves_its_single_global_ipv4(self):
        result = run_vm_launcher_helper("""
iw() { printf 'Interface wlan0\\n\\ttype managed\\n'; }
ip() { printf '3: wlan0 inet 192.0.2.6/24 brd 192.0.2.255 scope global wlan0\\n'; }
SPLIT_VM_WIFI_INTERFACE=wlan0 resolve_vm_bind_address
""")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "192.0.2.6")

    def test_non_managed_wifi_fails_closed(self):
        result = run_vm_launcher_helper("""
iw() { printf 'Interface wlan0\\n\\ttype AP\\n'; }
ip() { printf '3: wlan0 inet 192.0.2.6/24 scope global wlan0\\n'; }
SPLIT_VM_WIFI_INTERFACE=wlan0 resolve_vm_bind_address
""")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wlan0", result.stderr)
        self.assertIn("managed", result.stderr)
        self.assertIn("AP", result.stderr)

    def test_missing_global_ipv4_fails_closed(self):
        result = run_vm_launcher_helper("""
iw() { printf 'Interface wlan0\\n\\ttype managed\\n'; }
ip() { return 0; }
SPLIT_VM_WIFI_INTERFACE=wlan0 resolve_vm_bind_address
""")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wlan0", result.stderr)
        self.assertIn("exatamente um IPv4 global", result.stderr)

    def test_multiple_global_ipv4_addresses_fail_closed(self):
        result = run_vm_launcher_helper("""
iw() { printf 'Interface wlan0\\n\\ttype managed\\n'; }
ip() {
    printf '3: wlan0 inet 192.0.2.6/24 scope global wlan0\\n'
    printf '3: wlan0 inet 192.0.2.7/24 scope global secondary wlan0\\n'
}
SPLIT_VM_WIFI_INTERFACE=wlan0 resolve_vm_bind_address
""")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("encontrei 2", result.stderr)

    def test_forwarding_covers_ssh_rpc_and_http_without_wildcard(self):
        result = run_vm_launcher_helper(
            "vm_wan_netdev 2 192.0.2.6")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), (
            "user,id=wan,hostfwd=tcp:192.0.2.6:2222-:22,"
            "hostfwd=tcp:192.0.2.6:50052-:50052,"
            "hostfwd=tcp:192.0.2.6:8082-:8080"))
        self.assertNotIn("0.0.0.0", result.stdout)
        self.assertNotIn("tcp::", result.stdout)

    def test_lan_netdev_is_listen_on_node1_and_connect_on_node2(self):
        # Regressao: a LAN era socket multicast (230.0.0.1), que depende da
        # rota multicast do host e morria em silencio com VPN/ethernet no
        # caminho. listen/connect por loopback nao depende de rota alguma.
        node1 = run_vm_launcher_helper("vm_lan_netdev 1")
        node2 = run_vm_launcher_helper("vm_lan_netdev 2")

        self.assertEqual(node1.returncode, 0, node1.stderr)
        self.assertEqual(node1.stdout.strip(), "socket,id=lan,listen=127.0.0.1:1234")
        self.assertEqual(node2.returncode, 0, node2.stderr)
        self.assertEqual(node2.stdout.strip(), "socket,id=lan,connect=127.0.0.1:1234")

    def test_lan_netdev_node3_connects_to_second_listener(self):
        # O listener do no 1 aceita UMA conexao: cada no adicional conecta no
        # listener (N-1). Sem isso o no 3+ ficava fora do segmento (ARP FAILED).
        node3 = run_vm_launcher_helper("vm_lan_netdev 3")

        self.assertEqual(node3.returncode, 0, node3.stderr)
        self.assertEqual(node3.stdout.strip(), "socket,id=lan,connect=127.0.0.1:1235")

    def test_lan_netdev_honours_split_vm_lan_port(self):
        result = run_vm_launcher_helper(
            "SPLIT_VM_LAN_PORT=2345 vm_lan_netdev 2")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "socket,id=lan,connect=127.0.0.1:2345")

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

    def test_physical_console_autologs_in_and_opens_btop(self):
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

        self.assertIn("btop", packages)
        self.assertIn("--autologin root", autologin.read_text(encoding="utf-8"))
        startup = profile.read_text(encoding="utf-8")
        self.assertIn("/dev/tty1", startup)
        self.assertIn("tmux attach-session -t split-models", startup)
        self.assertNotIn("btop", startup)
        self.assertIn("split-models-tmux", service.read_text(encoding="utf-8"))
        layout = tmux_launcher.read_text(encoding="utf-8")
        self.assertIn("new-session", layout)
        self.assertIn("split-window -h -p 35", layout)
        monitor_script = monitor.read_text(encoding="utf-8")
        self.assertIn("btop", monitor_script)
        self.assertIn("exec /usr/bin/bash", monitor_script)

    def test_serial_console_autologs_in_for_headless_vm(self):
        # A VM do harness fala pelo serial-getty, nao pelo getty do tty1: sem
        # este override a VM parava no prompt de login (regressao do boot de
        # 2026-08-06). E o .bash_profile precisa aceitar ttyS0, senao mesmo
        # logado o tmux nao anexa.
        autologin = (ARCHISO / "airootfs" / "etc" / "systemd" / "system" /
                     "serial-getty@ttyS0.service.d" / "autologin.conf")
        profile = ARCHISO / "airootfs" / "root" / ".bash_profile"

        content = autologin.read_text(encoding="utf-8")
        self.assertIn("--autologin root", content)
        self.assertIn("115200", content)
        startup = profile.read_text(encoding="utf-8")
        self.assertIn("/dev/ttyS0", startup)
        self.assertIn("/dev/tty1", startup)

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
                                  "SPLIT_NODE_IP": "203.0.113.12"})

        self.assertEqual(
            result.stdout.strip(),
            "role=worker | node=node-abcd1234 | ip=203.0.113.12")
        self.assertIn("--node-id", session_script)
        self.assertIn("/run/split-models/node-id", session_script)
        self.assertIn("status-right", tmux_script)
        self.assertIn("split-models-status", tmux_script)
        self.assertIn("printf 'master", master_script)
        self.assertIn("/run/split-models/role", master_script)

    def test_root_ssh_accepts_the_appliance_key_and_refuses_passwords(self):
        ssh_dir = ARCHISO / "airootfs" / "etc" / "ssh" / "sshd_config.d"
        authorized = (ARCHISO / "airootfs" / "root" / ".ssh" /
                      "authorized_keys").read_text(encoding="utf-8")
        profile = (ARCHISO / "profiledef.sh").read_text(encoding="utf-8")
        configs = sorted(path.name for path in ssh_dir.glob("*.conf"))
        settings = (ssh_dir / configs[0]).read_text(encoding="utf-8")

        # sshd usa o primeiro valor obtido de cada palavra-chave, e o Include e
        # a linha 2 do sshd_config: quem manda e o arquivo de menor prefixo.
        self.assertEqual(configs[0], "10-archiso.conf")
        self.assertIn("PermitRootLogin prohibit-password", settings)
        self.assertIn("PasswordAuthentication no", settings)
        self.assertIn("PubkeyAuthentication yes", settings)
        self.assertRegex(authorized, r"(?m)^ssh-ed25519 \S+ split-models-appliance$")
        # StrictModes descarta authorized_keys em silencio se a permissao
        # estiver frouxa, e o login falha sem pista nenhuma.
        self.assertIn('["/root/.ssh"]="0:0:700"', profile)
        self.assertIn('["/root/.ssh/authorized_keys"]="0:0:600"', profile)

    def test_appliance_private_key_stays_out_of_version_control(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        tracked = subprocess.run(
            ["git", "ls-files", "deploy/keys"], cwd=ROOT,
            check=True, capture_output=True, text=True)

        self.assertIn("deploy/keys/*", ignored)
        self.assertIn("!deploy/keys/*.pub", ignored)
        # Commit de chave privada nao se desfaz: so se revoga trocando o par.
        for path in tracked.stdout.split():
            self.assertTrue(path.endswith(".pub"), f"chave privada versionada: {path}")

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
