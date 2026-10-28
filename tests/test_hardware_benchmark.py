import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from split_lm.llama_rpc.cli import _build_parser, run_hardware
from split_lm.llama_rpc.hardware_benchmark import (
    HardwareBenchmarkError, HardwareScore, parse_7zip_mips,
    parse_llama_tokens_per_second, parse_vkpeak, run_hardware_benchmark)


class FakeBenchmarkRun:
    """Return realistic output for each benchmark without running hardware."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str],
                 **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        if command[0] == "vkpeak":
            return subprocess.CompletedProcess(command, 0,
                "device = Fake Vulkan\nfp16-matrix = 8000 GFLOPS\n"
                "int8-matrix = 16000 GIOPS\n", "")
        if command[0] == "7z":
            return subprocess.CompletedProcess(
                command, 0, "Tot: 99 6000 5900\n", "")
        speed = 5.0 if command[command.index("-ngl") + 1] == "0" else 20.0
        return subprocess.CompletedProcess(command, 0, json.dumps([
            {"n_prompt": 0, "n_gen": 32, "avg_ts": speed}]), "")


class FakeCpuOnlyBenchmarkRun:
    """Simulate a machine with no Vulkan driver and no int8-matrix support."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str],
                 **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        if command[0] == "vkpeak":
            return subprocess.CompletedProcess(
                command, 2, "no vulkan driver found\n", "")
        if command[0] == "7z":
            return subprocess.CompletedProcess(
                command, 0, "Tot: 99 6000 5900\n", "")
        if command[command.index("-ngl") + 1] == "99":
            return subprocess.CompletedProcess(command, 1, "", "no device\n")
        return subprocess.CompletedProcess(command, 0, json.dumps([
            {"n_prompt": 0, "n_gen": 32, "avg_ts": 5.0}]), "")


class FakeCpuFallbackBenchmarkRun(FakeCpuOnlyBenchmarkRun):
    """Simulate llama-bench accepting -ngl 99 but silently using the CPU."""

    def __call__(self, command: list[str],
                 **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "llama-bench" and command[command.index("-ngl") + 1] == "99":
            self.commands.append(list(command))
            return subprocess.CompletedProcess(command, 0, json.dumps([
                {"n_prompt": 0, "n_gen": 32, "avg_ts": 5.1}]), "")
        return super().__call__(command, **kwargs)


class HardwareBenchmarkParserTest(unittest.TestCase):
    def test_hardware_benchmark_is_exposed_by_the_appliance_cli(self):
        options = _build_parser().parse_args(["hardware-bench", "--json"])

        self.assertEqual(options.command, "hardware-bench")
        self.assertTrue(options.json)

    def test_full_benchmark_uses_small_model_for_cpu_and_vulkan(self):
        fake = FakeBenchmarkRun()
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "speed-cache"

            score = run_hardware_benchmark(cache_dir=cache, run=fake)

            self.assertFalse(cache.exists())
        llama_commands = [command for command in fake.commands
                          if command[0] == "llama-bench"]
        self.assertEqual(len(llama_commands), 2)
        self.assertTrue(all("Qwen3.5-0.8B" in " ".join(command)
                            for command in llama_commands))
        self.assertEqual(score.cpu_tokens_per_second, 5.0)
        self.assertEqual(score.vulkan_tokens_per_second, 20.0)

    def test_vkpeak_reports_real_tops_and_tflops_units(self):
        output = (
            "device       = Intel(R) Graphics (ARL)\n"
            "fp16-matrix  = 8761.15 GFLOPS\n"
            "int8-matrix  = 17757.51 GIOPS\n")

        device, fp16_tflops, int8_tops = parse_vkpeak(output)

        self.assertEqual(device, "Intel(R) Graphics (ARL)")
        self.assertAlmostEqual(fp16_tflops, 8.76115)
        self.assertAlmostEqual(int8_tops, 17.75751)

    def test_7zip_total_rating_is_kept_as_mips_not_renamed_tops(self):
        output = "Avr: x\nTot:              99   6056   6022\n"

        self.assertEqual(parse_7zip_mips(output), 6022)

    def test_llama_generation_row_supplies_tokens_per_second(self):
        output = json.dumps([
            {"n_prompt": 128, "n_gen": 0, "avg_ts": 300.0},
            {"n_prompt": 0, "n_gen": 32, "avg_ts": 19.25},
        ])

        self.assertEqual(parse_llama_tokens_per_second(output), 19.25)

    def test_invalid_vkpeak_output_names_the_expected_metrics(self):
        with self.assertRaises(HardwareBenchmarkError) as refused:
            parse_vkpeak("device = GPU sem matriz")

        self.assertIn("fp16-matrix", str(refused.exception))
        self.assertIn("int8-matrix", str(refused.exception))

    def test_machine_without_vulkan_still_scores_cpu_honestly(self):
        fake = FakeCpuOnlyBenchmarkRun()
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "speed-cache"

            score = run_hardware_benchmark(cache_dir=cache, run=fake)

        self.assertEqual(score.device, "sem-vulkan")
        self.assertEqual(score.fp16_tflops, 0.0)
        self.assertEqual(score.int8_tops, 0.0)
        self.assertEqual(score.vulkan_tokens_per_second, 0.0)
        self.assertEqual(score.cpu_tokens_per_second, 5.0)
        self.assertEqual(score.cpu_mips, 5900)

    def test_cpu_fallback_is_not_reported_as_vulkan_throughput(self):
        fake = FakeCpuFallbackBenchmarkRun()
        with tempfile.TemporaryDirectory() as temporary:
            score = run_hardware_benchmark(
                cache_dir=Path(temporary) / "speed-cache", run=fake)

        self.assertEqual(score.device, "sem-vulkan")
        self.assertEqual(score.vulkan_tokens_per_second, 0.0)

    def test_cli_records_the_score_for_the_nodes_own_panel(self):
        fake = FakeBenchmarkRun()
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "run"
            options = argparse.Namespace(
                cache_dir=Path(temporary) / "cache",
                runtime_dir=runtime, json=False)

            with patch("split_lm.llama_rpc.cli.run_hardware_benchmark",
                       return_value=HardwareScore(
                           "Fake Vulkan", 8.0, 16.0, 5000, 3.0, 20.0)):
                result = run_hardware(options)

            score_file = runtime / "hardware-score.json"
            self.assertEqual(result, 0)
            self.assertTrue(score_file.exists())
            self.assertIn('"int8_tops": 16.0', score_file.read_text(
                encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
