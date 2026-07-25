import socket
import threading
import unittest

import torch

from experiments.tensor_parallel.benchmark import (
    _config,
    _mlp_minimum_network_bytes,
    _parse_args,
    _tcp_all_reduce,
)
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

    def test_physical_nodes_keep_all_local_threads(self):
        args = _parse_args(["--nodes", "2", "--threads", "8", "--rank", "0"])

        local = _config(args, local_spawn=True)
        physical = _config(args, local_spawn=False)

        self.assertEqual(local.worker_threads, 4)
        self.assertEqual(physical.worker_threads, 8)
        self.assertEqual(physical.baseline_threads, 8)

    def test_worker_threads_can_be_tuned_without_changing_baseline(self):
        args = _parse_args([
            "--nodes", "2", "--threads", "14", "--worker-threads", "4",
            "--rank", "0",
        ])

        config = _config(args, local_spawn=False)

        self.assertEqual(config.baseline_threads, 14)
        self.assertEqual(config.worker_threads, 4)

    def test_steady_state_mlp_only_communicates_final_hidden(self):
        network_bytes = _mlp_minimum_network_bytes(
            tokens=1, hidden_size=896, element_size=4, nodes=2
        )

        self.assertEqual(network_bytes, 7_168)

    def test_tcp_collective_can_be_selected(self):
        args = _parse_args(["--collective", "tcp"])

        config = _config(args, local_spawn=True)

        self.assertEqual(config.collective, "tcp")

    def test_tcp_collective_returns_sum_to_both_ranks(self):
        left, right = socket.socketpair()
        remote_result: list[torch.Tensor] = []
        remote = threading.Thread(
            target=lambda: remote_result.append(
                _tcp_all_reduce(torch.tensor([2.0]), right, None)
            )
        )
        try:
            remote.start()
            local_result = _tcp_all_reduce(torch.tensor([1.0]), left, None)
            remote.join(timeout=2)
        finally:
            left.close()
            right.close()

        self.assertFalse(remote.is_alive())
        self.assertEqual(local_result.item(), 3.0)
        self.assertEqual(remote_result[0].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
