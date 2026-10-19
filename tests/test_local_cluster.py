import unittest
from types import SimpleNamespace

from tools.benchmark_local_cluster import estimate_node_bytes, split_layers


class LocalClusterTest(unittest.TestCase):
    def test_layers_are_distributed_without_gaps(self):
        self.assertEqual(
            split_layers(total_layers=10, nodes=3),
            [(0, 4), (4, 7), (7, 10)],
        )

    def test_memory_estimate_includes_head_and_tail_weights(self):
        config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=2,
            vocab_size=32,
            attention_bias=False,
        )

        node_bytes = estimate_node_bytes(config, nodes=2, bytes_per_parameter=4)

        self.assertEqual(len(node_bytes), 2)
        self.assertGreater(node_bytes[0], 0)
        self.assertGreater(node_bytes[1], 0)
        self.assertEqual(sum(node_bytes), 6_816)


if __name__ == "__main__":
    unittest.main()
