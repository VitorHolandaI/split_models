import threading
import unittest

from split_lm.cluster.cli import format_nodes
from split_lm.cluster.discovery import (
    DiscoveryReplyError, decode_reply, discover_nodes, encode_reply, local_node,
    serve_discovery)
from split_lm.cluster.plan import (
    APP_MEMORY_PER_NODE, MAX_CLUSTER_NODES, ClusterNode,
    ModelCapacityError, build_vertical_plan, required_cluster_nodes)


GIB = 1024 ** 3


class VerticalClusterPlanTest(unittest.TestCase):
    def test_master_builds_contiguous_worker_chain(self):
        workers = (
            ClusterNode("node-b", "worker-b", "192.168.0.11", 29510, 8, 8_000),
            ClusterNode("node-c", "worker-c", "192.168.0.12", 29510, 4, 4_000),
        )

        plan = build_vertical_plan(10, "192.168.0.8", workers)

        self.assertEqual(plan.head_layers, (0, 4))
        self.assertEqual(plan.first_worker, "192.168.0.11:9999")
        self.assertEqual(plan.workers[0].layers, (4, 7))
        self.assertEqual(plan.workers[0].connect_to, "192.168.0.12:9999")
        self.assertIsNone(plan.workers[0].token_back_to)
        self.assertEqual(plan.workers[1].layers, (7, 10))
        self.assertIsNone(plan.workers[1].connect_to)
        self.assertEqual(plan.workers[1].token_back_to, "192.168.0.8:9999")

    def test_plan_rejects_more_than_twenty_four_total_nodes(self):
        workers = tuple(
            ClusterNode(f"node-{index}", f"worker-{index}", "192.168.0.11",
                        29510, 8, 8 * GIB)
            for index in range(MAX_CLUSTER_NODES)
        )

        with self.assertRaisesRegex(ValueError, "maximo=24"):
            build_vertical_plan(48, "192.168.0.8", workers)


class ClusterCapacityTest(unittest.TestCase):
    def test_twenty_four_nodes_provide_ninety_six_gib(self):
        self.assertEqual(APP_MEMORY_PER_NODE * MAX_CLUSTER_NODES, 96 * GIB)

    def test_required_nodes_rounds_up_with_two_node_minimum(self):
        self.assertEqual(required_cluster_nodes(1), 2)
        self.assertEqual(required_cluster_nodes(8 * GIB + 1), 3)

    def test_model_above_ninety_six_gib_is_rejected(self):
        with self.assertRaisesRegex(ModelCapacityError, "96.0 GiB"):
            required_cluster_nodes(96 * GIB + 1)


class DiscoveryReplyTest(unittest.TestCase):
    def test_reply_roundtrip_takes_address_from_the_sender(self):
        node = ClusterNode("node-b", "worker-b", "", 29510, 8, 8_000)

        decoded = decode_reply(encode_reply(node), "192.168.0.11")

        self.assertEqual(decoded, ClusterNode(
            "node-b", "worker-b", "192.168.0.11", 29510, 8, 8_000))

    def test_malformed_reply_reports_payload_and_expected_shape(self):
        with self.assertRaises(DiscoveryReplyError) as raised:
            decode_reply(b'{"node_id": "node-b"}', "192.168.0.11")

        message = str(raised.exception)
        self.assertIn("192.168.0.11", message)
        self.assertIn("node_id", message)
        self.assertIn("available_memory", message)


class NodeTableTest(unittest.TestCase):
    def test_table_lists_one_row_per_node(self):
        nodes = (ClusterNode("node-b", "worker-b", "192.168.0.11", 29510, 8, 8_000),)

        lines = format_nodes(nodes).splitlines()

        self.assertIn("node-b", lines[1])
        self.assertIn("192.168.0.11:29510", lines[1])

    def test_table_states_when_nothing_answered(self):
        self.assertEqual(format_nodes(()), "nenhum no respondeu ao discovery")


class DiscoveryTest(unittest.TestCase):
    def _serve(self, node, port):
        stop = threading.Event()
        thread = threading.Thread(
            target=serve_discovery, args=(node, port, stop), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(stop.set)

    def test_local_node_reports_usable_resources(self):
        node = local_node("node-a", 29510)

        self.assertEqual(node.node_id, "node-a")
        self.assertEqual(node.control_port, 29510)
        self.assertGreater(node.cpu_count, 0)
        self.assertGreater(node.available_memory, 0)

    def test_discover_finds_serving_node(self):
        self._serve(local_node("node-a", 29510), 29500)

        found = discover_nodes(29500, timeout=1.0, broadcast_address="127.0.0.1")

        self.assertEqual([node.node_id for node in found], ["node-a"])
        self.assertEqual(found[0].address, "127.0.0.1")
        self.assertEqual(found[0].control_port, 29510)

    def test_discover_returns_empty_without_nodes(self):
        self.assertEqual(
            discover_nodes(29501, timeout=0.3, broadcast_address="127.0.0.1"), ())

    def test_discover_reports_each_node_once(self):
        self._serve(local_node("node-a", 29511), 29502)

        found = discover_nodes(29502, timeout=1.0, broadcast_address="127.0.0.1",
                               probes=3)

        self.assertEqual([node.node_id for node in found], ["node-a"])


if __name__ == "__main__":
    unittest.main()
