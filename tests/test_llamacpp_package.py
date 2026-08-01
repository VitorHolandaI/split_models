import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "deploy" / "packages" / "split-llama-cpp-rpc"
PKGBUILD = PACKAGE / "PKGBUILD"
VERSION_ENV = PACKAGE / "llama-version.env"


def read_pkgbuild_variable(name: str) -> str:
    """Le uma variavel do PKGBUILD dando source nele, como o makepkg faria."""
    resolved = subprocess.run(
        ["bash", "-c", f'source ./PKGBUILD; printf "%s" "{name}"'],
        cwd=PACKAGE, check=True, capture_output=True, text=True)
    return resolved.stdout


class LlamaVersionPinTest(unittest.TestCase):
    def test_pin_names_a_tag_a_commit_and_a_digest(self):
        pin = VERSION_ENV.read_text(encoding="utf-8")

        self.assertRegex(pin, r"(?m)^LLAMA_TAG=b\d+$")
        self.assertRegex(pin, r"(?m)^LLAMA_COMMIT=[0-9a-f]{40}$")
        self.assertRegex(pin, r"(?m)^LLAMA_SHA256=[0-9a-f]{64}$")

    def test_pkgbuild_reads_the_pin_instead_of_repeating_it(self):
        recipe = PKGBUILD.read_text(encoding="utf-8")
        pin_values = re.findall(r"^LLAMA_\w+=(\S+)$", VERSION_ENV.read_text(
            encoding="utf-8"), re.MULTILINE)

        self.assertIn("source ./llama-version.env", recipe)
        for value in pin_values:
            self.assertNotIn(value, recipe)

    def test_pkgbuild_resolves_the_pinned_source_and_digest(self):
        pin = dict(re.findall(r"^(LLAMA_\w+)=(\S+)$",
                              VERSION_ENV.read_text(encoding="utf-8"),
                              re.MULTILINE))

        self.assertEqual(read_pkgbuild_variable("$pkgver"), pin["LLAMA_TAG"])
        self.assertEqual(read_pkgbuild_variable("${sha256sums[0]}"),
                         pin["LLAMA_SHA256"])
        self.assertIn(f"/refs/tags/{pin['LLAMA_TAG']}.tar.gz",
                      read_pkgbuild_variable("${source[0]}"))


class LlamaBuildOptionsTest(unittest.TestCase):
    def test_rpc_and_vulkan_backends_are_enabled(self):
        options = read_pkgbuild_variable('${_cmake_options[@]}')

        for required in ("-DGGML_RPC=ON", "-DGGML_VULKAN=ON",
                         "-DGGML_BACKEND_DL=ON", "-DGGML_CPU=ON"):
            self.assertIn(required, options)

    def test_binaries_run_on_hosts_other_than_the_build_host(self):
        options = read_pkgbuild_variable('${_cmake_options[@]}')

        # GGML_CPU_ALL_VARIANTS escolhe a ISA em runtime e exige NATIVE=OFF;
        # sem o par a ISO quebra em qualquer CPU mais velha que a do build.
        self.assertIn("-DGGML_CPU_ALL_VARIANTS=ON", options)
        self.assertIn("-DGGML_NATIVE=OFF", options)

    def test_rpc_server_is_actually_installed(self):
        options = read_pkgbuild_variable('${_cmake_options[@]}')

        # tools/rpc/CMakeLists.txt so instala ggml-rpc-server sob esta opcao.
        self.assertIn("-DLLAMA_TOOLS_INSTALL=ON", options)
        self.assertIn("-DLLAMA_BUILD_TOOLS=ON", options)

    def test_first_image_carries_no_vendor_sdk_backends(self):
        recipe = PKGBUILD.read_text(encoding="utf-8")

        # Cada um destes vira outra variante; aqui inchariam a ISO e trariam
        # dependencia de driver proprietario.
        for excluded in ("GGML_CUDA", "GGML_HIP", "GGML_SYCL", "GGML_CANN",
                         "GGML_MUSA", "GGML_METAL"):
            self.assertNotIn(excluded, recipe)

    def test_package_records_the_upstream_commit_on_disk(self):
        recipe = PKGBUILD.read_text(encoding="utf-8")

        self.assertIn("build-info.txt", recipe)
        self.assertIn("${LLAMA_COMMIT}", recipe)


if __name__ == "__main__":
    unittest.main()
