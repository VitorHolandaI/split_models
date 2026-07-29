import argparse
import unittest
import tempfile
from pathlib import Path

import torch

from tools.benchmark_single_model import TokenTimingStreamer, _measure_generation


class FakeClock:
    def __init__(self):
        self.values = iter((10.0, 11.0, 12.0))

    def __call__(self) -> float:
        return next(self.values)


class FakeGenerateModel:
    def generate(self, inputs: torch.Tensor, **options: object) -> torch.Tensor:
        streamer = options["streamer"]
        assert isinstance(streamer, TokenTimingStreamer)
        streamer.put(inputs)
        streamer.put(torch.tensor([4]))
        streamer.put(torch.tensor([5]))
        return torch.cat([inputs, torch.tensor([[4, 5]])], dim=1)


class FakeGenerateTokenizer:
    eos_token_id = 99


class SingleModelBenchmarkTest(unittest.TestCase):
    def test_streamer_skips_prompt_and_times_generated_tokens(self):
        streamer = TokenTimingStreamer(FakeClock())

        streamer.put(torch.tensor([[1, 2, 3]]))
        streamer.put(torch.tensor([4]))
        streamer.put(torch.tensor([5]))

        self.assertEqual(streamer.token_times, [10.0, 11.0])

    def test_warmup_measurement_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            args = argparse.Namespace(
                max_new_tokens=2, ignore_eos=True, context_sizes="8",
                model="Qwen/test", metrics_file=str(path), precision="bfloat16")

            _measure_generation(
                FakeGenerateModel(), FakeGenerateTokenizer(),
                torch.tensor([[1, 2]]), args, record=False)

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
