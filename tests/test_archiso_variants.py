import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "deploy" / "lib_archiso_build.sh"
OVERLAY = ROOT / "deploy" / "archiso-llamacpp-rpc"


class FakeVariantTree:
    """Repositorio de mentira com a ISO estavel e a experimental ja em disco.

    Existe para provar, sem rodar mkarchiso, que a limpeza da variante nao
    alcanca dist/split-models-*.iso.
    """

    stable_iso = "dist/split-models-2026.07.31-x86_64.iso"
    variant_iso = ("dist/llamacpp-rpc/"
                   "split-models-llamacpp-rpc-2026.08.01-x86_64.iso")

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        for relative in (self.stable_iso, self.variant_iso):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"iso")

    def survives(self, relative: str) -> bool:
        return (self.root / relative).exists()


def run_lib(script: str, check: bool = True) -> subprocess.CompletedProcess:
    """Executa um trecho bash com a lib carregada e trash trocado por rm."""
    preamble = f"set -euo pipefail\nsource {LIB}\ntrash() {{ rm -f \"$@\"; }}\n"
    return subprocess.run(["bash", "-c", preamble + script],
                          check=check, capture_output=True, text=True)


def variant_config(root: str, name: str, output_dir: str, prefix: str) -> str:
    return (f"repo_root={root}\nvariant_name={name}\n"
            f"output_dir={output_dir}\niso_prefix={prefix}\n")


class VariantIsolationTest(unittest.TestCase):
    def test_stable_build_keeps_its_output_path_and_prefix(self):
        stable = (ROOT / "deploy" / "build_archiso.sh").read_text(encoding="utf-8")

        self.assertIn("lib_archiso_build.sh", stable)
        self.assertIn('variant_name="stable"', stable)
        self.assertIn('output_dir="$repo_root/dist"', stable)
        self.assertIn('iso_prefix="split-models"', stable)

    def test_experimental_build_writes_to_its_own_subdirectory(self):
        experimental = (ROOT / "deploy" / "build_llamacpp_archiso.sh").read_text(
            encoding="utf-8")

        self.assertIn('output_dir="$repo_root/dist/llamacpp-rpc"', experimental)
        self.assertIn('iso_prefix="split-models-llamacpp-rpc"', experimental)
        self.assertIn('overlay_source="$repo_root/deploy/archiso-llamacpp-rpc"',
                      experimental)

    def test_variant_cleanup_never_reaches_the_stable_iso(self):
        with tempfile.TemporaryDirectory() as root:
            tree = FakeVariantTree(root)
            run_lib(variant_config(root, "llamacpp-rpc",
                                   f"{root}/dist/llamacpp-rpc",
                                   "split-models-llamacpp-rpc")
                    + "assert_variant_isolation\ndiscard_old_images\n")

            self.assertTrue(tree.survives(tree.stable_iso))
            self.assertFalse(tree.survives(tree.variant_iso))

    def test_variant_may_not_write_straight_into_dist(self):
        with tempfile.TemporaryDirectory() as root:
            rejected = run_lib(
                variant_config(root, "llamacpp-rpc", f"{root}/dist",
                               "split-models-llamacpp-rpc")
                + "assert_variant_isolation\n", check=False)

        self.assertEqual(rejected.returncode, 1)
        self.assertIn("nao pode escrever direto em dist/", rejected.stderr)

    def test_variant_may_not_borrow_the_stable_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            rejected = run_lib(
                variant_config(root, "llamacpp-rpc",
                               f"{root}/dist/llamacpp-rpc", "split-models")
                + "assert_variant_isolation\n", check=False)

        self.assertEqual(rejected.returncode, 1)
        self.assertIn("nao pode usar o prefixo split-models", rejected.stderr)

    def test_variant_may_not_escape_the_repository(self):
        with tempfile.TemporaryDirectory() as root:
            rejected = run_lib(
                variant_config(root, "llamacpp-rpc", "/tmp/elsewhere",
                               "split-models-llamacpp-rpc")
                + "assert_variant_isolation\n", check=False)

        self.assertEqual(rejected.returncode, 1)
        self.assertIn("output_dir fora de", rejected.stderr)


class OverlayMergeTest(unittest.TestCase):
    def test_overlay_packages_extend_the_base_list(self):
        with tempfile.TemporaryDirectory() as staged:
            Path(staged, "packages.x86_64").write_text("base\nuv\n", encoding="utf-8")
            merged = run_lib(
                f"staged_profile={staged}\noverlay_source={OVERLAY}\n"
                'merge_overlay_packages "$(cat '
                f'{staged}/packages.x86_64)"\n'
                f"cat {staged}/packages.x86_64\n")

        self.assertIn("uv", merged.stdout)
        self.assertIn("vulkan-icd-loader", merged.stdout)
        self.assertIn("vulkan-radeon", merged.stdout)

    def test_iso_name_check_reads_the_override_not_both_assignments(self):
        """Reproduz o profile encenado de verdade: base + overlay anexado.

        O overlay deixa dois iso_name no arquivo e so o ultimo vale. Uma
        checagem por grep casa os dois e compara as duas linhas contra o
        prefixo, reprovando um profile correto.
        """
        base = (ROOT / "deploy" / "archiso" / "profiledef.sh").read_text(
            encoding="utf-8")
        overlay = (OVERLAY / "profiledef.append.sh").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as staged:
            Path(staged, "profiledef.sh").write_text(base + overlay,
                                                     encoding="utf-8")
            checked = subprocess.run(
                ["bash", "-c",
                 f"set -euo pipefail\nsource {LIB}\n"
                 f"staged_profile={staged}\n"
                 "iso_prefix=split-models-llamacpp-rpc\n"
                 "assert_profile_iso_name\necho ACEITO\n"],
                check=False, capture_output=True, text=True)

        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("ACEITO", checked.stdout)

    def test_overlay_profiledef_renames_the_iso_without_losing_permissions(self):
        base = (ROOT / "deploy" / "archiso" / "profiledef.sh").read_text(
            encoding="utf-8")
        with tempfile.TemporaryDirectory() as staged:
            Path(staged, "profiledef.sh").write_text(base, encoding="utf-8")
            resolved = run_lib(
                f"staged_profile={staged}\noverlay_source={OVERLAY}\n"
                "append_overlay_profiledef\n"
                "declare -A file_permissions=()\n"
                f"source {staged}/profiledef.sh\n"
                'printf \'%s\\n\' "$iso_name" "${file_permissions['
                '"/etc/initcpio/hooks/casper-compat"]}"\n')

        self.assertEqual(resolved.stdout.split(),
                         ["split-models-llamacpp-rpc", "0:0:755"])


class LlamaApplianceOverlayTest(unittest.TestCase):
    def overlay_file(self, relative: str) -> str:
        return (OVERLAY / "airootfs" / relative).read_text(encoding="utf-8")

    def test_boot_starts_the_rpc_worker_not_the_pytorch_one(self):
        service = self.overlay_file("etc/systemd/system/split-models.service")
        session = self.overlay_file("usr/local/bin/split-llama-session")

        # Mesmo nome de unidade da imagem estavel de proposito: reaproveita o
        # link em multi-user.target.wants que lib_archiso_build.sh ja cria.
        self.assertIn("ExecStart=/usr/local/bin/split-llama-tmux", service)
        self.assertIn("split-llama worker", session)
        self.assertNotIn("split-cluster worker", session)

    def test_worker_panel_leaves_a_shell_instead_of_closing(self):
        session = self.overlay_file("usr/local/bin/split-llama-session")

        # Fechar o painel tiraria da tela o log que explica por que o worker
        # morreu, que e a unica pista no console fisico.
        self.assertIn("exec /usr/bin/bash", session)
        self.assertIn("/root/split-llama-master.sh", session)

    def test_insecure_transport_is_on_screen_not_only_in_the_docs(self):
        layout = self.overlay_file("usr/local/bin/split-llama-tmux")
        guide = self.overlay_file("root/README-split-llama.txt")

        self.assertIn("EXPERIMENTAL RPC - LAN ISOLADA", layout)
        self.assertIn("LAN isolada", guide)
        self.assertIn("executa codigo", guide)

    def test_master_shortcut_discovers_instead_of_hardcoding_addresses(self):
        shortcut = self.overlay_file("root/split-llama-master.sh")

        self.assertIn("split-llama master", shortcut)
        self.assertIn("--discover", shortcut)
        self.assertIn("detect_broadcast", shortcut)
        self.assertNotIn("192.168.122.255", shortcut)

    def test_variant_scripts_are_declared_executable(self):
        appended = (OVERLAY / "profiledef.append.sh").read_text(encoding="utf-8")

        for script in ("/root/split-llama-master.sh",
                       "/usr/local/bin/split-llama-session",
                       "/usr/local/bin/split-llama-tmux"):
            self.assertIn(f'["{script}"]="0:0:755"', appended)


if __name__ == "__main__":
    unittest.main()
