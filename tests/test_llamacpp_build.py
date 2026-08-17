import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLAMA_LIB = ROOT / "deploy" / "lib_llamacpp_build.sh"
ARCHISO_LIB = ROOT / "deploy" / "lib_archiso_build.sh"
VERSION_ENV = ROOT / "deploy" / "llama-version.env"


def read_pin() -> dict[str, str]:
    return dict(re.findall(r"(?m)^(LLAMA_\w+)=(\S+)$",
                           VERSION_ENV.read_text(encoding="utf-8")))


def run_llama_lib(script: str) -> subprocess.CompletedProcess:
    """Carrega a lib do llama.cpp com um repo_root e build_root de mentira."""
    with tempfile.TemporaryDirectory() as scratch:
        preamble = (f"set -euo pipefail\nrepo_root={ROOT}\nbuild_root={scratch}\n"
                    f"source {LLAMA_LIB}\n")
        return subprocess.run(["bash", "-c", preamble + script],
                              check=True, capture_output=True, text=True)


class LlamaVersionPinTest(unittest.TestCase):
    def test_pin_names_a_tag_a_commit_and_a_digest(self):
        pin = VERSION_ENV.read_text(encoding="utf-8")

        self.assertRegex(pin, r"(?m)^LLAMA_TAG=b\d+$")
        self.assertRegex(pin, r"(?m)^LLAMA_COMMIT=[0-9a-f]{40}$")
        self.assertRegex(pin, r"(?m)^LLAMA_SHA256=[0-9a-f]{64}$")

    def test_build_reads_the_pin_instead_of_repeating_it(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        self.assertIn("source \"$repo_root/deploy/llama-version.env\"", recipe)
        for value in read_pin().values():
            self.assertNotIn(value, recipe)

    def test_cached_tarball_is_verified_on_every_build(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")
        download = recipe.index("curl -fsSL")
        verify = recipe.index("sha256sum -c")

        # Um tarball em cache corrompido passaria se so o download fosse checado.
        self.assertLess(download, verify)


class LlamaBuildOptionsTest(unittest.TestCase):
    def setUp(self):
        self.options = run_llama_lib(
            'printf "%s\\n" "${llama_cmake_options[@]}"').stdout.split()

    def test_rpc_and_vulkan_backends_are_enabled(self):
        for required in ("-DGGML_RPC=ON", "-DGGML_VULKAN=ON",
                         "-DGGML_BACKEND_DL=ON", "-DGGML_CPU=ON"):
            self.assertIn(required, self.options)

    def test_binaries_run_on_hosts_other_than_the_build_host(self):
        # A ISO e compilada numa maquina e bootada em outras; NATIVE=ON faria o
        # binario faltar instrucao em qualquer CPU mais velha da frota.
        self.assertIn("-DGGML_CPU_ALL_VARIANTS=ON", self.options)
        self.assertIn("-DGGML_NATIVE=OFF", self.options)

    def test_rpc_server_is_actually_installed(self):
        # tools/rpc/CMakeLists.txt so instala ggml-rpc-server sob esta opcao, e
        # sem ela o build passa limpo com o binario faltando na imagem.
        self.assertIn("-DLLAMA_TOOLS_INSTALL=ON", self.options)
        self.assertIn("-DLLAMA_BUILD_TOOLS=ON", self.options)

    def test_rdma_is_pinned_off_rather_than_auto_detected(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        # ggml/src/ggml-rpc/CMakeLists.txt:12-22 procura libibverbs no host de
        # build e liga RDMA sozinho. O host de desenvolvimento tem rdma-core, a
        # ISO nao: sem este pin, libggml-rpc.so sai linkada contra uma lib
        # ausente da imagem e o backend so falha no boot da maquina fisica.
        self.assertIn("-DGGML_RPC_RDMA=OFF", self.options)
        self.assertIn("libibverbs", recipe)

    def test_first_image_carries_no_vendor_sdk_backends(self):
        for excluded in ("-DGGML_CUDA=ON", "GGML_HIP", "GGML_SYCL", "GGML_CANN",
                         "GGML_MUSA", "GGML_METAL"):
            self.assertNotIn(excluded, self.options)

    def test_requested_cuda_must_actually_arrive(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        # Backend e plugin: sem libggml-cuda.so o worker sobe, cai em Vulkan e
        # serve a mesma GPU mais devagar, sem nada indicando que CUDA sumiu.
        self.assertIn("libggml-cuda.so", recipe)
        self.assertIn("assert_requested_cuda_backend", recipe)

    def test_cuda_is_opt_in_and_off_by_default(self):
        without = run_llama_lib('llama_cuda_options').stdout.strip()
        with_cuda = run_llama_lib(
            'SPLIT_LLAMA_CUDA=1\nllama_cuda_options').stdout.strip()

        self.assertEqual(without, "")
        self.assertEqual(with_cuda, "-DGGML_CUDA=ON")

    def test_cuda_is_refused_for_the_iso_before_anything_is_staged(self):
        variant = (ROOT / "deploy" / "build_llamacpp_archiso.sh").read_text(
            encoding="utf-8")
        guard = variant.index("refuse_cuda_for_image")
        staging = variant.index("build_archiso_variant")

        # O backend CUDA compila para a GPU do host de build; numa imagem que
        # roda em outras maquinas isso nao serve. Checado depois, a recusa vinha
        # so apos o uv sync, minutos gastos por causa de uma variavel.
        self.assertLess(guard, staging)

    def test_cuda_guard_actually_stops_the_image_build(self):
        refused = subprocess.run(
            ["bash", "-c",
             f"set -euo pipefail\nrepo_root={ROOT}\nbuild_root=/tmp\n"
             f"source {LLAMA_LIB}\nSPLIT_LLAMA_CUDA=1\nrefuse_cuda_for_image\n"],
            check=False, capture_output=True, text=True)

        self.assertEqual(refused.returncode, 1)
        self.assertIn("nao vale para a ISO", refused.stderr)


class VulkanShaderCacheTest(unittest.TestCase):
    def test_changed_glslc_signature_cleans_incremental_outputs(self):
        result = run_llama_lib(
            'llama_tree="$build_root/tree"\n'
            'vulkan_toolchain_stamp="$llama_tree/build/toolchain"\n'
            'mkdir -p "$llama_tree/build"\n'
            'touch "$llama_tree/build/CMakeCache.txt"\n'
            'printf "old\\n" > "$vulkan_toolchain_stamp"\n'
            'glslc() { printf "new\\n"; }\n'
            'cmake() { printf "cmake %s\\n" "$*"; }\n'
            'invalidate_vulkan_shader_cache')

        self.assertIn("--target clean", result.stdout)
        self.assertIn("glslc mudou", result.stderr)

    def test_unchanged_glslc_signature_keeps_incremental_outputs(self):
        result = run_llama_lib(
            'llama_tree="$build_root/tree"\n'
            'vulkan_toolchain_stamp="$llama_tree/build/toolchain"\n'
            'mkdir -p "$llama_tree/build"\n'
            'touch "$llama_tree/build/CMakeCache.txt"\n'
            'printf "same\\n" > "$vulkan_toolchain_stamp"\n'
            'glslc() { printf "same\\n"; }\n'
            'cmake() { printf "cmake %s\\n" "$*"; }\n'
            'invalidate_vulkan_shader_cache')

        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_signature_is_recorded_only_after_successful_build(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")
        build = recipe.index('cmake --build "$llama_tree/build" --config Release')
        record = recipe.index("record_vulkan_toolchain_signature", build)

        self.assertLess(build, record)


class LlamaStagingTest(unittest.TestCase):
    def test_variant_hook_is_a_noop_for_the_stable_build(self):
        archiso_lib = ARCHISO_LIB.read_text(encoding="utf-8")

        self.assertIn("stage_variant_extras() { :; }", archiso_lib)
        staged = archiso_lib.index("stage_application\n    stage_variant_extras")
        packaged = archiso_lib.index("sudo mkarchiso")
        self.assertLess(staged, packaged)

    def test_llamacpp_variant_overrides_the_hook_after_the_base_lib(self):
        variant = (ROOT / "deploy" / "build_llamacpp_archiso.sh").read_text(
            encoding="utf-8")
        base = variant.index("lib_archiso_build.sh")
        override = variant.index("lib_llamacpp_build.sh")

        # Ordem invertida deixaria o sem-op da lib base valendo e a ISO sairia
        # sem nenhum binario do llama.cpp, sem erro nenhum.
        self.assertLess(base, override)
        self.assertIn("stage_variant_extras", LLAMA_LIB.read_text(encoding="utf-8"))

    def test_fleet_key_is_embedded_at_the_fleet_path(self):
        # Sem a privada em /root/.ssh/fleet_ed25519 o `fleet bench` nao roda
        # de dentro do no. A chave vem de deploy/keys (gitignored) e vai para
        # o caminho fixo que o SshKeyRunner procura.
        lib = LLAMA_LIB.read_text(encoding="utf-8")

        self.assertIn("embed_fleet_key", lib)
        self.assertIn("stage_variant_extras", lib)
        self.assertIn("/root/.ssh/fleet_ed25519", lib)
        self.assertIn("deploy/keys/split-models-appliance", lib)

    def test_missing_build_tools_stop_the_run_before_any_staging(self):
        variant = (ROOT / "deploy" / "build_llamacpp_archiso.sh").read_text(
            encoding="utf-8")
        preflight = variant.index("require_llama_build_commands")
        staging = variant.index("build_archiso_variant")

        # Checado depois, o build so quebraria passado o uv sync, minutos
        # adiante e longe da causa.
        self.assertLess(preflight, staging)

    def test_install_is_checked_for_the_binaries_that_matter(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        for binary in ("usr/bin/llama-cli", "usr/bin/llama-server",
                       "usr/bin/llama-bench", "usr/bin/llama-batched-bench",
                       "usr/bin/ggml-rpc-server"):
            self.assertIn(binary, recipe)
        # Backend e plugin com GGML_BACKEND_DL e fica ao lado do executavel, em
        # usr/bin. Procurar em usr/lib daria ausente num build correto; e um que
        # nao compilou vira "sem aceleracao" em runtime, nunca erro de build.
        self.assertIn("usr/bin/libggml-rpc.so", recipe)
        self.assertIn("usr/bin/libggml-vulkan.so", recipe)

    def test_install_layout_is_the_same_on_every_distribution(self):
        options = run_llama_lib(
            'printf "%s\\n" "${llama_cmake_options[@]}"').stdout.split()

        # Debian/Ubuntu instala em usr/lib/x86_64-linux-gnu por padrao e o Arch
        # em usr/lib. A divergencia quebra o LD_LIBRARY_PATH que o script
        # imprime e, pior, faz o glob da checagem de linkagem casar nada.
        self.assertIn("-DCMAKE_INSTALL_LIBDIR=lib", options)

    def test_linkage_check_fails_when_it_finds_nothing_to_check(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        # Glob sem correspondencia percorre zero arquivos e passa silenciosamente.
        self.assertIn("found=0", recipe)
        self.assertIn("(( found ))", recipe)

    def test_linkage_is_checked_on_the_artifact_not_only_on_the_flag(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        # Uma opcao pode ser removida por engano; o ELF construido e a evidencia.
        self.assertIn("readelf -d", recipe)
        # A checagem tem que estar pendurada no caminho do install, senao fica
        # definida e nunca executada.
        self.assertIn("    assert_no_host_only_libraries\n}", recipe)
        self.assertIn("    assert_llama_artifacts\n}", recipe)

    def test_compiled_binaries_get_the_execute_bit_back(self):
        """mkarchiso:316 copia airootfs sem preservar modo.

        Sem declarar em file_permissions, ggml-rpc-server e llama-cli chegam
        na imagem em 644: o build passa limpo e so falha rodando dentro da ISO.
        """
        with tempfile.TemporaryDirectory() as scratch:
            destdir = Path(scratch, "dest/usr/bin")
            destdir.mkdir(parents=True)
            for name in ("ggml-rpc-server", "llama-cli", "libggml-vulkan.so"):
                (destdir / name).touch()
            staged = Path(scratch, "staged")
            staged.mkdir()
            (staged / "profiledef.sh").write_text(
                "declare -A file_permissions=()\n", encoding="utf-8")

            generated = subprocess.run(
                ["bash", "-c",
                 f"set -euo pipefail\nrepo_root={ROOT}\nbuild_root={scratch}\n"
                 f"source {LLAMA_LIB}\n"
                 f"llama_destdir={scratch}/dest\nstaged_profile={staged}\n"
                 "append_llama_file_permissions\n"
                 f"source {staged}/profiledef.sh\n"
                 'for key in "${!file_permissions[@]}"; do '
                 'printf "%s=%s\\n" "$key" "${file_permissions[$key]}"; done'],
                check=True, capture_output=True, text=True)

        self.assertEqual(
            sorted(generated.stdout.split()),
            ["/usr/bin/ggml-rpc-server=0:0:755",
             "/usr/bin/libggml-vulkan.so=0:0:755",
             "/usr/bin/llama-cli=0:0:755"])

    def test_missing_dependency_names_packages_for_both_distributions(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        # A frota mistura Arch (a appliance) e Ubuntu (maquinas com GPU que nao
        # bootam a ISO). Dizer so "pacman" manda o operador do Ubuntu procurar
        # um pacote que nao existe ali.
        self.assertIn("sudo pacman -S --needed", recipe)
        self.assertIn("sudo apt install -y", recipe)
        self.assertIn("libvulkan-dev", recipe)

    def test_https_support_is_checked_before_compiling(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        # Sem os headers do OpenSSL o llama.cpp compila sem HTTPS e so recusa
        # na hora de baixar o modelo, com a compilacao inteira ja gasta.
        self.assertIn("/usr/include/openssl/ssl.h", recipe)
        self.assertIn("libssl-dev", recipe)

    def test_bare_metal_build_shares_the_pin_with_the_image(self):
        script = (ROOT / "deploy" / "build_llamacpp_binaries.sh").read_text(
            encoding="utf-8")

        # Maquinas que nao bootam a appliance tem que sair do mesmo commit: o
        # backend RPC nao negocia versao, entao builds diferentes so falham em
        # runtime, longe da causa.
        self.assertIn("lib_llamacpp_build.sh", script)
        self.assertNotIn("cmake -B", script)
        self.assertIn("LAN isolada", script)

    def test_image_records_the_upstream_commit_on_disk(self):
        recipe = LLAMA_LIB.read_text(encoding="utf-8")

        self.assertIn("build-info.txt", recipe)
        self.assertIn("$LLAMA_COMMIT", recipe)


if __name__ == "__main__":
    unittest.main()
