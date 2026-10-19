import contextlib
import io
import subprocess
import sys
import threading
import time
import unittest

from split_lm.cluster.cli import (
    _build_parser, format_nodes, local_benchmark_command, main as cluster_main, random_node_id,
    source_address_for)
from split_lm.cluster.discovery import (
    DiscoveryReplyError, decode_reply, discover_nodes, encode_reply, local_node,
    serve_discovery)
from split_lm.cluster.plan import (
    APP_MEMORY_PER_NODE, MAX_CLUSTER_NODES, ClusterNode,
    ModelCapacityError, build_vertical_plan, required_cluster_nodes)
from split_lm.constants import DEFAULT_CHAT_MODEL


GIB = 1024 ** 3


class VerticalClusterPlanTest(unittest.TestCase):
    def test_master_builds_contiguous_worker_chain(self):
        workers = (
            ClusterNode("node-b", "worker-b", "192.0.2.11", 29510, 8, 8_000),
            ClusterNode("node-c", "worker-c", "192.0.2.12", 29510, 4, 4_000),
        )

        plan = build_vertical_plan(10, "192.0.2.8", workers)

        self.assertEqual(plan.head_layers, (0, 4))
        self.assertEqual(plan.first_worker, "192.0.2.11:9999")
        self.assertEqual(plan.workers[0].layers, (4, 7))
        self.assertEqual(plan.workers[0].connect_to, "192.0.2.12:9999")
        self.assertIsNone(plan.workers[0].token_back_to)
        self.assertEqual(plan.workers[1].layers, (7, 10))
        self.assertIsNone(plan.workers[1].connect_to)
        self.assertEqual(plan.workers[1].token_back_to, "192.0.2.8:9999")

    def test_plan_rejects_more_than_twenty_four_total_nodes(self):
        workers = tuple(
            ClusterNode(f"node-{index}", f"worker-{index}", "192.0.2.11",
                        29510, 8, 8 * GIB)
            for index in range(MAX_CLUSTER_NODES)
        )

        with self.assertRaisesRegex(ValueError, "maximo=24"):
            build_vertical_plan(48, "192.0.2.8", workers)


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

        decoded = decode_reply(encode_reply(node), "192.0.2.11")

        self.assertEqual(decoded, ClusterNode(
            "node-b", "worker-b", "192.0.2.11", 29510, 8, 8_000))

    def test_malformed_reply_reports_payload_and_expected_shape(self):
        with self.assertRaises(DiscoveryReplyError) as raised:
            decode_reply(b'{"node_id": "node-b"}', "192.0.2.11")

        message = str(raised.exception)
        self.assertIn("192.0.2.11", message)
        self.assertIn("node_id", message)
        self.assertIn("available_memory", message)


class NodeTableTest(unittest.TestCase):
    def test_cluster_cli_import_does_not_load_torch(self):
        check = subprocess.run(
            [sys.executable, "-c", (
                "import sys; import split_lm.cluster.cli; "
                "print('torch' in sys.modules)")],
            check=True, capture_output=True, text=True)

        self.assertEqual(check.stdout.strip(), "False")

    def test_interactive_master_defaults_to_instruction_model(self):
        args = _build_parser().parse_args(["master", "--discover"])

        self.assertEqual(args.model, DEFAULT_CHAT_MODEL)
        self.assertTrue(args.model.endswith("-Instruct"))

    def test_local_benchmark_help_is_specific_to_that_command(self):
        output = io.StringIO()

        with self.assertRaises(SystemExit), contextlib.redirect_stdout(output):
            _build_parser().parse_args(["benchmark-local", "--help"])

        help_text = output.getvalue()
        self.assertIn("Run one named profile on a full local model", help_text)
        self.assertIn("--benchmark-profile", help_text)
        self.assertNotIn("--broadcast", help_text)

    def test_master_help_explains_model_tokens_memory_and_examples(self):
        output = io.StringIO()

        with self.assertRaises(SystemExit), contextlib.redirect_stdout(output):
            _build_parser().parse_args(["master", "--help"])

        help_text = output.getvalue()
        self.assertIn("repositorio Hugging Face", help_text)
        self.assertIn("limite de tokens", help_text)
        self.assertIn("memoria", help_text)
        self.assertIn("split-cluster exemplos", help_text)

    def test_local_benchmark_command_uses_shared_profile_file(self):
        command = local_benchmark_command(
            "benchmarks.toml", "context-realistic", executable="/venv/python")

        self.assertEqual(command[:3], ["/venv/python", "-m", "tools.benchmark_single_model"])
        self.assertIn("benchmarks.toml", command)
        self.assertIn("context-realistic", command)

    def test_examples_flag_and_command_show_default_model_and_all_profiles(self):
        command_output = io.StringIO()
        flag_output = io.StringIO()

        with contextlib.redirect_stdout(command_output):
            cluster_main(["exemplos"])
        with contextlib.redirect_stdout(flag_output):
            cluster_main(["--comandos-exemplos"])

        self.assertEqual(flag_output.getvalue(), command_output.getvalue())
        examples = command_output.getvalue()
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", examples)
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", examples)
        for profile in ("context-quick", "context-deep", "context-realistic",
                        "poem-realistic", "decode-long", "context-quick-int8",
                        "context-quick-int4"):
            self.assertIn(
                f"split-cluster master --discover --benchmark-profile {profile}",
                examples)
            self.assertIn(
                f"split-cluster benchmark-local --benchmark-profile {profile}",
                examples)

    def test_default_worker_ids_are_random_and_safe(self):
        first = random_node_id()
        second = random_node_id()

        self.assertRegex(first, r"^node-[0-9a-f]{8}$")
        self.assertNotEqual(first, second)

    def test_table_lists_one_row_per_node(self):
        nodes = (ClusterNode("node-b", "worker-b", "192.0.2.11", 29510, 8, 8_000),)

        lines = format_nodes(nodes).splitlines()

        self.assertIn("node-b", lines[1])
        self.assertIn("192.0.2.11:29510", lines[1])

    def test_table_states_when_nothing_answered(self):
        self.assertEqual(format_nodes(()), "nenhum no respondeu ao discovery")

    def test_source_address_uses_the_route_to_the_worker(self):
        self.assertEqual(source_address_for("127.0.0.1"), "127.0.0.1")


class DiscoveryTest(unittest.TestCase):
    def _serve(self, node, port):
        """Sobe o discovery e so retorna com a porta ja ligada.

        Sem esperar por `ready` o teste corre com o bind, e um probe que chega
        antes some sem deixar rastro: UDP para porta nao ligada nao tem fila.
        """
        stop, ready = threading.Event(), threading.Event()
        thread = threading.Thread(
            target=serve_discovery, args=(node, port, stop, ready), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(stop.set)
        self.assertTrue(ready.wait(5), f"discovery nao ligou udp/{port} em 5s")

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

    def test_discovery_reaches_a_node_that_binds_after_the_first_probe(self):
        """Um worker ainda subindo tem que ser achado dentro da janela.

        E o caso real de `split-cluster nodes` logo apos ligar as maquinas.
        Com os probes em rajada todos saem antes do bind e some tudo; so o
        espalhamento pela janela alcanca esse no.
        """
        stop, ready = threading.Event(), threading.Event()
        node = local_node("node-lento", 29514)

        def bind_late():
            time.sleep(0.5)
            serve_discovery(node, 29504, stop, ready)

        thread = threading.Thread(target=bind_late, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(stop.set)

        found = discover_nodes(29504, timeout=3.0, broadcast_address="127.0.0.1",
                               probes=6)

        self.assertEqual([entry.node_id for entry in found], ["node-lento"])

    def test_discover_reports_each_node_once(self):
        self._serve(local_node("node-a", 29511), 29502)

        found = discover_nodes(29502, timeout=1.0, broadcast_address="127.0.0.1",
                               probes=3)

        self.assertEqual([node.node_id for node in found], ["node-a"])


if __name__ == "__main__":
    unittest.main()
