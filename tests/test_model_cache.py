import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from split_lm.llama_rpc.cli import run_master
from split_lm.llama_rpc.model_cache import (
    ModelCacheError, ensure_model_fits_storage, select_cached_model)


class FakeCompletedProcess:
    """Successful subprocess result for the two llama-cli invocations."""

    returncode = 0


class FakeSubprocessRun:
    """Record subprocess environments without starting llama.cpp."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, command: list[str], **kwargs: object) -> FakeCompletedProcess:
        self.calls.append((list(command), dict(kwargs)))
        return FakeCompletedProcess()


class SingleModelCacheTest(unittest.TestCase):
    def test_16_gb_usb_accepts_the_12_gb_model_with_reserve(self):
        ensure_model_fits_storage(15_500_000_000, 12.11)

    def test_storage_smaller_than_model_plus_reserve_is_refused(self):
        with self.assertRaises(ModelCacheError) as refused:
            ensure_model_fits_storage(12_000_000_000, 12.11)

        self.assertIn("12.11", str(refused.exception))
        self.assertIn("livres", str(refused.exception))

    def test_selecting_another_model_removes_the_previous_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            old_model = cache / "old-model.gguf"
            old_model.write_text("weights", encoding="utf-8")
            (cache / ".selected-model").write_text(
                "org/old-model:Q4_K_M", encoding="utf-8")

            select_cached_model(cache, "org/new-model:Q6_K")

            self.assertFalse(old_model.exists())
            self.assertEqual(
                (cache / ".selected-model").read_text(encoding="utf-8"),
                "org/new-model:Q6_K")

    def test_selecting_the_same_model_preserves_its_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            model = "org/current-model:Q4_K_M"
            downloaded = cache / "current-model.gguf"
            downloaded.write_text("weights", encoding="utf-8")
            (cache / ".selected-model").write_text(model, encoding="utf-8")

            cache_hit = select_cached_model(cache, model)

            self.assertEqual(downloaded.read_text(encoding="utf-8"), "weights")
            self.assertTrue(cache_hit)

    def test_refused_download_preserves_the_previous_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            downloaded = cache / "old-model.gguf"
            downloaded.write_text("weights", encoding="utf-8")
            (cache / ".selected-model").write_text(
                "org/old-model:Q4_K_M", encoding="utf-8")
            fake_usage = type("Usage", (), {"free": 12_000_000_000})()

            with patch("split_lm.llama_rpc.model_cache.shutil.disk_usage",
                       return_value=fake_usage), \
                    self.assertRaises(ModelCacheError):
                select_cached_model(cache, "org/gpt-oss:MXFP4",
                                    model_gigabytes=12.11)

            self.assertEqual(downloaded.read_text(encoding="utf-8"), "weights")
            self.assertEqual(
                (cache / ".selected-model").read_text(encoding="utf-8"),
                "org/old-model:Q4_K_M")

    def test_master_cli_selects_one_model_in_the_configured_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            (cache / "old.gguf").write_text("weights", encoding="utf-8")
            fake_run = FakeSubprocessRun()
            options = argparse.Namespace(
                rpc="10.0.0.2:50052", hf="org/new:Q6_K", llama_args=[])

            with patch.dict(os.environ, {"LLAMA_CACHE": str(cache)}), \
                    patch("split_lm.llama_rpc.cli.subprocess.run", fake_run):
                result = run_master(options)

            self.assertEqual(result, 0)
            self.assertFalse((cache / "old.gguf").exists())
            environment = fake_run.calls[-1][1]["env"]
            self.assertIsInstance(environment, dict)
            self.assertEqual(environment, {**os.environ,
                                            "LLAMA_CACHE": str(cache)})


if __name__ == "__main__":
    unittest.main()
