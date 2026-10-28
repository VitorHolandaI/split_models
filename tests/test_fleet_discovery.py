import argparse
import io
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import cast
from unittest.mock import patch

from split_lm.llama_rpc.cli import (
    _build_parser, _configure_direct_mode_discovery, _configure_distributed,
    _reset_hosts, _run_fleet_reboot, _run_master_dry_run,
    _resolve_endpoints, _restore_discovered_workers, _select_master_hosts,
    _start_replicated, _stop_proxy, format_capacity_plan, format_capacity_table,
    format_fleet_nodes)
from split_lm.llama_rpc.discovery import RpcWorker
from split_lm.llama_rpc.distributed_planner import RankedNode
from split_lm.llama_rpc.fleet_discovery import (
    FleetNode, decode_reply, discover_fleet_nodes, encode_reply,
    serve_fleet_discovery)
from split_lm.llama_rpc.fleet_coordinator import FleetCoordinator
from split_lm.llama_rpc.fleet_ssh import SshResult
from split_lm.llama_rpc.fleet_inventory import NodeInventory
from split_lm.llama_rpc.hardware_benchmark import HardwareScore
from split_lm.llama_rpc.mode_state import Mode, ModeConfig, NodeState
from split_lm.llama_rpc.mode_store import StateStore


class FakeSshRunner:
    """Record fleet transitions without contacting lab machines."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, host: str, argv: Sequence[str]) -> SshResult:
        self.calls.append((host, list(argv)))
        return SshResult("", "")


class FakePlanningRunner:
    """Return RAM inventory and hardware scores for master node selection."""

    def __init__(self, speeds: dict[str, float]) -> None:
        self._speeds = speeds
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, host: str, argv: Sequence[str]) -> SshResult:
        command = list(argv)
        self.calls.append((host, command))
        if "hardware-bench" in command:
            score = HardwareScore("GPU", 4.0, 8.0, 5000, 3.0,
                                  self._speeds[host])
            return SshResult(score.to_json(), "")
        if "SECTION hostname" in " ".join(command):
            return SshResult(_inventory_output(host), "")
        return SshResult("", "")


def _inventory_output(host: str) -> str:
    return (
        f"SECTION hostname\n{host}\n"
        f"SECTION ip\n{host}\n"
        "SECTION memtotal\nMemTotal: 14680064 kB\n"
        f"SECTION status\nrole=worker | node=node-{host} | ip={host}\n"
        "SECTION build\nabc123\n"
        "SECTION devices\nVulkan0\n")


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
    def test_mode_parser_exposes_ip_free_discovery_and_usb_selection(self):
        options = _build_parser().parse_args([
            "mode", "distributed-master", "--model", "small",
            "--auto-discover", "--broadcast", "192.168.76.255",
            "--usb-device", "/dev/sda1"])

        self.assertTrue(options.auto_discover)
        self.assertEqual(options.broadcast, "192.168.76.255")
        self.assertEqual(options.usb_device, "/dev/sda1")

    def test_master_parser_exposes_capacity_dry_run(self):
        options = _build_parser().parse_args([
            "master", "--discover", "--dry-run", "--model-layers", "24"])

        self.assertTrue(options.dry_run)
        self.assertEqual(options.model_layers, 24)

    def test_master_help_shows_dry_run_without_manual_layer_count(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(output):
            _build_parser().parse_args(["master", "--help"])

        help_text = output.getvalue()
        self.assertIn("split-llama master --discover --dry-run", help_text)
        self.assertNotIn("--model-layers 24", help_text)

    def test_direct_mode_discovery_resets_remote_nodes_and_sets_rpc(self):
        options = argparse.Namespace(
            target="distributed-master", discovery_timeout=3.0,
            broadcast="192.168.76.255", rpc="")
        found = (
            FleetNode("node-local", "a", "10.0.0.2", "replica"),
            FleetNode("node-remote", "b", "10.0.0.3", "replica"),
        )
        workers = (RpcWorker("node-remote", "b", "10.0.0.3", 50052),)
        runner = FakeSshRunner()

        with patch("split_lm.llama_rpc.cli._local_node_id",
                   return_value="node-local"), \
                patch("split_lm.llama_rpc.cli.discover_fleet_nodes",
                      return_value=found), \
                patch("split_lm.llama_rpc.cli.SshKeyRunner",
                      return_value=runner), \
                 patch("split_lm.llama_rpc.cli.discover_workers",
                       return_value=workers) as discover:
            _configure_direct_mode_discovery(options)

        self.assertEqual(options.rpc, "10.0.0.3:50052")
        discover.assert_called_once_with(29520, 3.0, "192.168.76.255")
        self.assertEqual(runner.calls, [
            ("10.0.0.3", ["split-llama", "mode", "worker"])])

    def test_capacity_table_shows_ram_tops_mips_and_tokens_per_second(self):
        node = RankedNode(
            NodeInventory("10.0.0.2", "node-a", "host-a", "10.0.0.2",
                          14 * 1024 * 1024, ("Vulkan",), "abc", "worker"),
            HardwareScore("Intel(R) Graphics (ARL)", 8.76, 17.76, 6022,
                          3.0, 20.0))

        table = format_capacity_table((node,))

        self.assertIn("10.0.0.2", table)
        self.assertIn("14.0", table)
        self.assertIn("17.76", table)
        self.assertIn("6022", table)
        self.assertIn("20.00", table)

    def test_rpc_discovery_excludes_unselected_workers(self):
        options = argparse.Namespace(
            rpc="", discover=True, discovery_port=29520, timeout=3.0,
            broadcast="192.168.0.255")
        workers = (
            RpcWorker("a", "a", "10.0.0.2", 50052),
            RpcWorker("b", "b", "10.0.0.3", 50052),
            RpcWorker("c", "c", "10.0.0.4", 50052),
        )

        with patch("split_lm.llama_rpc.cli._discover_master_hosts",
                   return_value=("10.0.0.2", "10.0.0.4")), \
                patch("split_lm.llama_rpc.cli.discover_workers",
                      return_value=workers):
            endpoints = _resolve_endpoints(options)

        self.assertEqual(endpoints, "10.0.0.2:50052,10.0.0.4:50052")

    def test_master_uses_requested_total_and_fastest_worker(self):
        nodes = (
            FleetNode("node-a", "a", "10.0.0.2", "replica"),
            FleetNode("node-b", "b", "10.0.0.3", "replica"),
            FleetNode("node-c", "c", "10.0.0.4", "replica"),
        )
        runner = FakePlanningRunner({
            "10.0.0.2": 10.0, "10.0.0.3": 5.0, "10.0.0.4": 20.0})
        options = argparse.Namespace(
            hf="ggml-org/gpt-oss-20b-GGUF:MXFP4",
            model_size_gb=None, machines=2, hardware_timeout=900)

        with patch("split_lm.llama_rpc.cli.lan_address",
                   return_value="10.0.0.2"):
            selected = _select_master_hosts(options, nodes, runner)

        self.assertEqual(selected, ("10.0.0.2", "10.0.0.4"))

    def test_master_dry_run_selects_safe_minimum_without_resetting_nodes(self):
        nodes = (
            FleetNode("node-a", "a", "10.0.0.2", "replica"),
            FleetNode("node-b", "b", "10.0.0.3", "replica"),
            FleetNode("node-c", "c", "10.0.0.4", "replica"),
        )
        runner = FakePlanningRunner({
            "10.0.0.2": 10.0, "10.0.0.3": 5.0, "10.0.0.4": 20.0})
        options = argparse.Namespace(
            discover=True, discovery_port=29520, timeout=3.0,
            broadcast="192.168.0.255", hf="ggml-org/gpt-oss-20b-GGUF:MXFP4",
            model_size_gb=None, hardware_timeout=900, machines=None,
            model_layers=24)
        output = io.StringIO()

        with patch("split_lm.llama_rpc.cli.discover_fleet_nodes",
                   return_value=nodes), \
             patch("split_lm.llama_rpc.cli.SshKeyRunner",
                   return_value=runner), \
             patch("split_lm.llama_rpc.cli.lan_address",
                   return_value="10.0.0.2"), \
             redirect_stdout(output):
            result = _run_master_dry_run(options)

        self.assertEqual(result, 0)
        self.assertIn("DRY-RUN", output.getvalue())
        self.assertIn("10.0.0.2", output.getvalue())
        self.assertIn("10.0.0.4", output.getvalue())
        self.assertIn("0-13", output.getvalue())
        self.assertIn("fora do plano minimo: 10.0.0.3", output.getvalue())
        self.assertNotIn("mode worker", " ".join(
            " ".join(command) for _, command in runner.calls))

    def test_fleet_reset_can_discover_control_nodes_without_manual_ips(self):
        options = _build_parser().parse_args(
            ["fleet", "reset", "--auto-discover"])
        nodes = (
            FleetNode("node-a", "host-a", "10.0.0.2", "replica"),
            FleetNode("node-b", "host-b", "10.0.0.3", "replica"),
        )

        with patch("split_lm.llama_rpc.cli.discover_fleet_nodes",
                   return_value=nodes):
            hosts = _reset_hosts(options)

        self.assertEqual(hosts, ("10.0.0.2", "10.0.0.3"))

    def test_interactive_master_restores_replicas_before_rpc_discovery(self):
        runner = FakeSshRunner()
        options = argparse.Namespace(
            timeout=3.0, broadcast="192.168.0.255")
        nodes = (
            FleetNode("node-a", "host-a", "10.0.0.2", "replica"),
            FleetNode("node-b", "host-b", "10.0.0.3", "replica"),
        )

        with patch("split_lm.llama_rpc.cli.discover_fleet_nodes",
                   return_value=nodes):
            restored = _restore_discovered_workers(options, runner)

        self.assertEqual(restored, ("10.0.0.2", "10.0.0.3"))
        self.assertEqual(runner.calls, [
            ("10.0.0.2", ["split-llama", "mode", "worker"]),
            ("10.0.0.3", ["split-llama", "mode", "worker"]),
        ])

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

    def test_distributed_then_replicated_resets_every_role(self):
        runner = FakeSshRunner()
        distributed = argparse.Namespace(
            master="", rpc_port=50052, model="small", parallel=4, port=8080)
        with patch("split_lm.llama_rpc.cli.lan_address",
                   return_value="10.0.0.2"):
            _configure_distributed(
                runner, ("10.0.0.2", "10.0.0.3", "10.0.0.4"), distributed)

        with tempfile.TemporaryDirectory() as output_dir:
            replicated = argparse.Namespace(
                model="small", parallel=4, port=8080, haproxy_port=8088,
                output_dir=Path(output_dir))
            with patch("split_lm.llama_rpc.cli.subprocess.Popen"):
                _start_replicated(
                    runner, ("10.0.0.2", "10.0.0.3", "10.0.0.4"), replicated)

        replica_calls = [call for call in runner.calls if "replica" in call[1]]
        master_calls = [call for call in runner.calls
                        if "distributed-master" in call[1]]
        self.assertEqual(len(replica_calls), 3)
        self.assertEqual(len(master_calls), 1)

    def test_reboot_discovers_control_nodes_without_manual_ips(self):
        runner = FakeSshRunner()
        options = argparse.Namespace(
            discovery_port=29521, discovery_timeout=3.0,
            broadcast_address="192.168.76.255")
        nodes = (
            FleetNode("node-a", "a", "10.0.0.2", "replica"),
            FleetNode("node-b", "b", "10.0.0.3", "distributed-master"),
        )

        with patch("split_lm.llama_rpc.cli.discover_fleet_nodes",
                   return_value=nodes), \
                patch("split_lm.llama_rpc.fleet_ssh.SshKeyRunner",
                      return_value=runner):
            result = _run_fleet_reboot(cast(FleetCoordinator, None), options)

        self.assertEqual(result, 0)
        self.assertEqual(runner.calls, [
            ("10.0.0.2", ["reboot"]), ("10.0.0.3", ["reboot"])])

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
