import json
import tempfile
import unittest
from pathlib import Path

from split_lm.metrics import (
    append_metrics, build_generation_metrics, format_metrics)


class GenerationMetricsTest(unittest.TestCase):
    def test_metrics_separate_prefill_from_inter_token_latency(self):
        metrics = build_generation_metrics(
            mode="split", model="Qwen/test", prompt_tokens=128,
            started_at=10.0, finished_at=14.0,
            token_times=(12.0, 13.0, 14.0), workers=1,
            recorded_at="2026-07-29T17:00:00+00:00")

        self.assertEqual(metrics.generated_tokens, 3)
        self.assertEqual(metrics.total_seconds, 4.0)
        self.assertEqual(metrics.time_to_first_token_seconds, 2.0)
        self.assertEqual(metrics.average_inter_token_seconds, 1.0)
        self.assertEqual(metrics.tokens_per_second, 0.75)

    def test_empty_generation_has_zero_rates(self):
        metrics = build_generation_metrics(
            "split", "Qwen/test", 128, 10.0, 11.0, (), 1)

        self.assertEqual(metrics.generated_tokens, 0)
        self.assertEqual(metrics.tokens_per_second, 0.0)
        self.assertEqual(metrics.time_to_first_token_seconds, 0.0)

    def test_jsonl_storage_keeps_local_and_split_records(self):
        local = build_generation_metrics("local", "Qwen/test", 64, 1, 3, (2, 3), 0)
        split = build_generation_metrics("split", "Qwen/test", 64, 1, 2, (2,), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"

            append_metrics(path, local)
            append_metrics(path, split)

            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["mode"] for record in records], ["local", "split"])

    def test_terminal_report_includes_requested_latencies(self):
        metrics = build_generation_metrics("split", "Qwen/test", 64, 1, 3, (2, 3), 1)

        report = format_metrics(metrics)

        self.assertIn("tokens_per_second       1.00 tok/s", report)
        self.assertIn("total_latency           2000.0 ms", report)
        self.assertIn("average_inter_token     1000.0 ms", report)
        self.assertGreaterEqual(len(report.splitlines()), 8)

    def test_terminal_report_can_color_metric_values(self):
        metrics = build_generation_metrics("split", "Qwen/test", 64, 1, 3, (2, 3), 1)

        report = format_metrics(metrics, color=True)

        self.assertIn("\033[1;36m", report)
        self.assertIn("\033[1;32m1.00 tok/s\033[0m", report)


if __name__ == "__main__":
    unittest.main()
