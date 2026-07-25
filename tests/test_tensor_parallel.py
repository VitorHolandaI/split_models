import unittest

import torch

from experiments.tensor_parallel.ops import column_parallel, mlp_parallel, row_parallel


class TensorParallelTest(unittest.TestCase):
    def test_column_parallel_matches_full_matmul(self):
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)

        actual = column_parallel(inputs, weight, shards=2)

        self.assertTrue(torch.equal(actual, inputs @ weight))

    def test_row_parallel_matches_full_matmul_with_uneven_shards(self):
        inputs = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        actual = row_parallel(inputs, weight, shards=3)

        self.assertTrue(torch.equal(actual, inputs @ weight))

    def test_sharded_mlp_keeps_intermediate_hidden_split(self):
        torch.manual_seed(3)
        inputs = torch.randn(2, 4)
        gate = torch.randn(4, 6)
        up = torch.randn(4, 6)
        down = torch.randn(6, 4)
        expected = (torch.nn.functional.silu(inputs @ gate) * (inputs @ up)) @ down

        actual = mlp_parallel(inputs, gate, up, down, shards=2)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
