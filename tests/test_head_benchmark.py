import unittest
from types import SimpleNamespace

from split_lm.context_benchmark import build_context_ids


class FakeBenchmarkTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[41, 42, 43])


class HeadBenchmarkTest(unittest.TestCase):
    def test_context_ids_repeat_seed_to_exact_requested_size(self):
        ids = build_context_ids(FakeBenchmarkTokenizer(), 8, "cpu")

        self.assertEqual(tuple(ids.shape), (1, 8))
        self.assertEqual(ids.tolist()[0], [41, 42, 43, 41, 42, 43, 41, 42])


if __name__ == "__main__":
    unittest.main()
