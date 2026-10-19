import tempfile
import unittest
from pathlib import Path

from split_lm.benchmark_profiles import (
    BenchmarkProfileError, format_profiles, load_profile, load_profiles,
    profile_overrides)


VALID_TOML = """
[profiles.context-quick]
description = "Quick context sweep"
model = "Qwen/Qwen2.5-3B-Instruct"
context_sizes = [128, 256, 512]
prompt = ""
max_new_tokens = 16
warmup_runs = 1
precision = "bfloat16"
metrics_file = "benchmark-results/generation.jsonl"
output_prefix = "benchmark-results/context-quick"
"""


class BenchmarkProfileTest(unittest.TestCase):
    def _load(self, body: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmarks.toml"
            path.write_text(body, encoding="utf-8")
            return load_profiles(path)

    def test_profile_loads_typed_values(self):
        profile = self._load(VALID_TOML)["context-quick"]

        self.assertEqual(profile.context_sizes, (128, 256, 512))
        self.assertEqual(profile.max_new_tokens, 16)
        self.assertEqual(profile.warmup_runs, 1)
        self.assertEqual(profile.precision, "bfloat16")

    def test_natural_prompt_profile_can_omit_context_sweep(self):
        prompt_profile = (VALID_TOML
                          .replace("context_sizes = [128, 256, 512]", "context_sizes = []")
                          .replace('prompt = ""', 'prompt = "Write a poem about distributed systems."'))

        profile = self._load(prompt_profile)["context-quick"]

        self.assertEqual(profile.context_sizes, ())
        self.assertIn("poem", profile.prompt)

    def test_unknown_precision_reports_profile_and_allowed_values(self):
        invalid = VALID_TOML.replace('precision = "bfloat16"', 'precision = "int3"')

        with self.assertRaisesRegex(BenchmarkProfileError, "context-quick.*int3.*int4"):
            self._load(invalid)

    def test_unknown_field_is_rejected(self):
        invalid = VALID_TOML + 'command = "touch /tmp/owned"\n'

        with self.assertRaisesRegex(BenchmarkProfileError, "command"):
            self._load(invalid)

    def test_list_output_shows_name_precision_and_contexts(self):
        output = format_profiles(self._load(VALID_TOML))

        self.assertIn("context-quick", output)
        self.assertIn("bfloat16", output)
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", output)
        self.assertIn("128,256,512", output)

    def test_missing_profile_lists_available_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmarks.toml"
            path.write_text(VALID_TOML, encoding="utf-8")

            with self.assertRaisesRegex(BenchmarkProfileError, "missing.*context-quick"):
                load_profile(path, "missing")

    def test_local_overrides_keep_shared_metrics_and_separate_graph(self):
        profile = self._load(VALID_TOML)["context-quick"]

        overrides = profile_overrides(profile, local=True)

        self.assertEqual(overrides["metrics_file"], "benchmark-results/generation.jsonl")
        self.assertEqual(overrides["benchmark_output"], "benchmark-results/context-quick-local")
        self.assertEqual(overrides["prompt"], "")


if __name__ == "__main__":
    unittest.main()
