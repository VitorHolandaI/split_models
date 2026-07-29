import tempfile
import unittest
from pathlib import Path

from split_lm.context_benchmark import parse_context_sizes, write_context_outputs
from split_lm.metrics import build_generation_metrics


class ContextBenchmarkTest(unittest.TestCase):
    def test_context_sizes_must_grow_within_model_limit(self):
        self.assertEqual(parse_context_sizes("128,256,512", 1024), (128, 256, 512))
        with self.assertRaisesRegex(ValueError, "crescente"):
            parse_context_sizes("128,512,256", 1024)
        with self.assertRaisesRegex(ValueError, "max_context=1024"):
            parse_context_sizes("128,2048", 1024)

    def test_sweep_writes_csv_and_svg_without_plot_dependency(self):
        results = (
            build_generation_metrics("split", "Qwen/test", 128, 1, 3, (2, 3), 1),
            build_generation_metrics("split", "Qwen/test", 256, 1, 5, (3, 4, 5), 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "context-sweep"

            csv_path, svg_path = write_context_outputs(results, prefix)

            self.assertIn("prompt_tokens", csv_path.read_text(encoding="utf-8"))
            svg = svg_path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("Latencia total (ms)", svg)
            self.assertGreater(svg_path.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
