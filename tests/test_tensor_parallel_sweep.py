import unittest

from experiments.tensor_parallel.sweep import estimate_rank_bytes, parse_token_sweep


class TensorParallelSweepTest(unittest.TestCase):
    def test_token_sweep_must_grow(self):
        self.assertEqual(parse_token_sweep("1,4,16,64", 1), (1, 4, 16, 64))
        with self.assertRaisesRegex(ValueError, "crescente"):
            parse_token_sweep("1,16,4", 1)

    def test_memory_estimate_grows_and_rank_zero_includes_baseline(self):
        rank_zero = estimate_rank_bytes(32, 896, 4864, 2, 4, rank=0)
        rank_one = estimate_rank_bytes(32, 896, 4864, 2, 4, rank=1)
        larger = estimate_rank_bytes(128, 896, 4864, 2, 4, rank=0)

        self.assertGreater(rank_zero, rank_one)
        self.assertGreater(larger, rank_zero)


if __name__ == "__main__":
    unittest.main()
