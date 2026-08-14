import argparse
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from unittest.mock import patch

from split_lm.llama_rpc.cli import (
    _build_parser, _configure_distributed, _start_replicated, _stop_proxy,
    format_fleet_nodes)
from split_lm.llama_rpc.fleet_discovery import (
    FleetNode, decode_reply, discover_fleet_nodes, encode_reply,
    serve_fleet_discovery)
from split_lm.llama_rpc.fleet_ssh import SshResult
from split_lm.llama_rpc.mode_state import Mode, ModeConfig, NodeState
from split_lm.llama_rpc.mode_store import StateStore


class FakeSshRunner:
    """Record fleet transitions without contacting lab machines."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, host: str, argv: Sequence[str]) -> SshResult:
        self.calls.append((host, list(argv)))
        return SshResult("", "")


class FakeStuckProxy:
    """HAProxy process that ignores terminate until the forced kill."""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> int:
        raise subprocess.TimeoutExpired("haproxy", timeout)

    def kill(self) -> None:
        self.killed = True


class FleetDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = StateStore(Path(self.temporary.name))

    def _serve(self, port: int) -> None:
        stop, ready = threading.Event(), threading.Event()
        thread = threading.Thread(
            target=serve_fleet_discovery,
            args=("node-a", self.store),
            kwargs={"port": port, "stop": stop, "ready": ready}, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(stop.set)
        self.assertTrue(ready.wait(5), f"controle nao ligou udp/{port}")

    def test_reply_round_trip_uses_udp_source_address(self):
        node = FleetNode("node-a", "host-a", "", "replica")

        restored = decode_reply(encode_reply(node), "192.168.0.10")

        self.assertEqual(restored, FleetNode(
            "node-a", "host-a", "192.168.0.10", "replica"))

    def test_control_discovery_survives_worker_to_replica_transition(self):
        self._serve(29541)
        before = discover_fleet_nodes(
            29541, timeout=0.3, broadcast_address="127.0.0.1")
        self.store.write_state(NodeState.from_config(ModeConfig(
            Mode.REPLICA, "repo:model")))

        after = discover_fleet_nodes(
            29541, timeout=0.3, broadcast_address="127.0.0.1")

        self.assertEqual(before[0].mode, "worker")
        self.assertEqual(after[0].mode, "replica")
        self.assertEqual(after[0].node_id, "node-a")

    def test_control_listing_includes_replica_mode(self):
        listing = format_fleet_nodes((
            FleetNode("node-a", "host-a", "10.0.0.2", "replica"),))

        self.assertIn("10.0.0.2", listing)
        self.assertIn("replica", listing)


class AutoDistributedTransitionTest(unittest.TestCase):
    def test_all_nodes_return_to_worker_and_only_local_node_becomes_master(self):
        runner = FakeSshRunner()
        options = argparse.Namespace(
            master="", rpc_port=50052, model="small", parallel=4, port=8080)

        with patch("split_lm.llama_rpc.cli.lan_address",
                   return_value="10.0.0.2"):
            endpoint = _configure_distributed(
                runner, ("10.0.0.2", "10.0.0.3", "10.0.0.4"), options)

        self.assertEqual(endpoint, "http://10.0.0.2:8080")
        self.assertEqual(runner.calls[:3], [
            ("10.0.0.2", ["split-llama", "mode", "worker"]),
            ("10.0.0.3", ["split-llama", "mode", "worker"]),
            ("10.0.0.4", ["split-llama", "mode", "worker"]),
        ])
        promotions = [call for call in runner.calls
                      if "distributed-master" in call[1]]
        self.assertEqual(len(promotions), 1)
        self.assertEqual(promotions[0][0], "10.0.0.2")
        self.assertIn("10.0.0.3:50052,10.0.0.4:50052", promotions[0][1])

    def test_replicated_then_distributed_resets_every_role(self):
        runner = FakeSshRunner()
        with tempfile.TemporaryDirectory() as output_dir:
            replicated = argparse.Namespace(
                model="small", parallel=4, port=8080, haproxy_port=8088,
                output_dir=Path(output_dir))
            with patch("split_lm.llama_rpc.cli.subprocess.Popen"):
                endpoint, _proxy = _start_replicated(
                    runner, ("10.0.0.2", "10.0.0.3", "10.0.0.4"), replicated)

            self.assertEqual(endpoint, "http://127.0.0.1:8088")
            config = Path(output_dir, "haproxy.cfg").read_text(encoding="utf-8")
            self.assertIn("bind *:8088", config)

        distributed = argparse.Namespace(
            master="", rpc_port=50052, model="small", parallel=4, port=8080)
        with patch("split_lm.llama_rpc.cli.lan_address",
                   return_value="10.0.0.2"):
            _configure_distributed(
                runner, ("10.0.0.2", "10.0.0.3", "10.0.0.4"), distributed)

        worker_calls = [call for call in runner.calls
                        if call[1] == ["split-llama", "mode", "worker"]]
        replica_calls = [call for call in runner.calls if "replica" in call[1]]
        master_calls = [call for call in runner.calls
                        if "distributed-master" in call[1]]
        self.assertEqual(len(worker_calls), 6)
        self.assertEqual(len(replica_calls), 3)
        self.assertEqual(len(master_calls), 1)

    def test_become_master_command_is_exposed_exactly_as_requested(self):
        options = _build_parser().parse_args(["become_master"])

        self.assertEqual(options.command, "become_master")

    def test_proxy_is_force_killed_when_it_ignores_cleanup(self):
        proxy = FakeStuckProxy()

        _stop_proxy(cast(subprocess.Popen[bytes], proxy))

        self.assertTrue(proxy.terminated)
        self.assertTrue(proxy.killed)


if __name__ == "__main__":
    unittest.main()
