import unittest

import torch

from tools.benchmark_single_model import TokenTimingStreamer


class FakeClock:
    def __init__(self):
        self.values = iter((10.0, 11.0, 12.0))

    def __call__(self) -> float:
        return next(self.values)


class SingleModelBenchmarkTest(unittest.TestCase):
    def test_streamer_skips_prompt_and_times_generated_tokens(self):
        streamer = TokenTimingStreamer(FakeClock())

        streamer.put(torch.tensor([[1, 2, 3]]))
        streamer.put(torch.tensor([4]))
        streamer.put(torch.tensor([5]))

        self.assertEqual(streamer.token_times, [10.0, 11.0])


if __name__ == "__main__":
    unittest.main()
