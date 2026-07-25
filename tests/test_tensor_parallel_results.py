import tempfile
import unittest
from pathlib import Path

from experiments.tensor_parallel.results import (
    build_result,
    effective_network_mbps,
    result_lines,
    write_sweep_outputs,
)


class TensorParallelResultsTest(unittest.TestCase):
    def test_effective_network_speed_uses_bytes_per_iteration(self):
        speed = effective_network_mbps(114_688, 0.003951)

        self.assertAlmostEqual(speed, 232.22, places=2)

    def test_report_explains_network_formula_and_error(self):
        result = build_result(
            "mlp-tcp", 32, 0.004546, 0.005851, 0.001900,
            114_688, 369.164, 0.0020045, 3, 30, 60, 3_441_720,
        )

        report = "\n".join(result_lines(result))

        self.assertIn("effective_network_mbps=232.3", report)
        self.assertIn("network_formula=(114724 bytes × 8)", report)
        self.assertIn("max_error=369.164", report)
        self.assertIn("Maior diferença absoluta", report)
        self.assertIn("DESEMPENHO", report)
        self.assertIn("REDE", report)
        self.assertIn("menor é melhor.\n\ndistributed_ms", report)

    def test_color_can_be_enabled(self):
        result = build_result(
            "mlp-tcp", 1, 0.002, 0.001, 0.0008,
            1_024, 0.0, 0.0, 3, 1, 2, 1_060,
        )

        report = "\n".join(result_lines(result, color=True))

        self.assertIn("\033[1;36mRESULTADO", report)
        self.assertIn("\033[36mbaseline_ms", report)
        self.assertIn("\033[35mdistributed_ms", report)
        self.assertIn("\033[32mspeedup", report)

    def test_sweep_writes_csv_and_png(self):
        losing = build_result(
            "mlp-tcp", 32, 0.004, 0.006, 0.002,
            100_000, 0.0, 0.0, 3, 10, 20, 1_000_000,
        )
        winning = build_result(
            "mlp-tcp", 128, 0.032, 0.015, 0.007,
            400_000, 0.0, 0.0, 3, 10, 20, 4_000_000,
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "threshold")
            csv_path, png_path, speedup_png_path = write_sweep_outputs(
                [losing, winning], prefix
            )

            self.assertIn("baseline_ms", csv_path.read_text(encoding="utf-8"))
            self.assertGreater(png_path.stat().st_size, 1_000)
            self.assertEqual(speedup_png_path.name, "threshold_speedup.png")
            self.assertGreater(speedup_png_path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
