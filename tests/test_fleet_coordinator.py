import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from split_lm.llama_rpc.cli import _build_parser
from split_lm.llama_rpc.fleet_coordinator import (
    FleetCoordinator, FleetError, FleetRunRecord)
from split_lm.llama_rpc.fleet_inventory import (
    FleetInventory, inventory_sections_command)
from split_lm.llama_rpc.fleet_ssh import SshError, SshResult
from split_lm.llama_rpc.health import HttpHealthProbe

TIMESTAMP = "2026-08-05T12:00:00+00:00"
INVENTORY_FILE = "inventory-2026-08-05T12-00-00-00-00.json"
RUN_FILE = "run-2026-08-05T12-00-00-00-00.json"


def node_sections(ip: str, hostname: str, role: str = "worker",
                  memtotal_kb: int = 16384000) -> str:
    return (
        f"SECTION hostname\n{hostname}\n"
        f"SECTION ip\n{ip}\n"
        f"SECTION memtotal\nMemTotal: {memtotal_kb} kB\n"
        f"SECTION status\nrole={role} | node=node-{hostname} | ip={ip}\n"
        "SECTION build\nabc1234def567890\n"
        "SECTION devices\nllama_1 device: CPU\n"
    )


class FakeSshRunner:
    """SSH de mentira: responde stdout por host e grava argv para asserts.

    Um host com `Exception` na resposta simula falha de conexao/execucao.
    """

    def __init__(self, replies: dict[str, str | Exception]) -> None:
        self._replies = replies
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, host: str, argv: Sequence[str]) -> SshResult:
        self.calls.append((host, list(argv)))
        reply = self._replies[host]
        if isinstance(reply, Exception):
            raise reply
        return SshResult(reply, "")


class FakeHealthProbe(HttpHealthProbe):
    """Readiness HTTP programado: ready=False faz o health check estourar."""

    def __init__(self, ready: bool = True) -> None:
        self._ready = ready
        self.checked: list[str] = []

    def ready(self, url: str) -> bool:
        self.checked.append(url)
        return self._ready


class FakeClock:
    """Relogio de mentira: sleep avanca o tempo sem esperar de verdade."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FleetCoordinatorTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.run_dir = Path(self._temporary.name) / "runs"
        self.health = FakeHealthProbe()
        self.clock = FakeClock()
        self.ssh = FakeSshRunner({})

    def coordinator(self, replies: dict[str, str | Exception]) -> FleetCoordinator:
        self.ssh = FakeSshRunner(replies)
        return FleetCoordinator(
            self.ssh, health=self.health, sleep=self.clock.sleep,
            clock=self.clock.monotonic, run_dir=self.run_dir,
            health_timeout=10.0, now=lambda: TIMESTAMP)

    def read_json(self, relative: str) -> dict[str, object]:
        path = self.run_dir / relative
        self.assertTrue(path.exists(), f"arquivo esperado: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def master_replies() -> dict[str, str | Exception]:
        return {
            "master": node_sections("10.0.0.2", "m-host"),
            "w1": node_sections("10.0.0.3", "w1-host"),
            "w2": node_sections("10.0.0.4", "w2-host"),
        }


class InventoryRunTest(FleetCoordinatorTestBase):
    def test_every_run_writes_the_exact_inventory_json(self):
        coordinator = self.coordinator(self.master_replies())

        inventory_path, nodes = coordinator.collect_inventory(("master", "w1"))

        self.assertEqual(inventory_path, self.run_dir / INVENTORY_FILE)
        self.assertEqual([node.node_id for node in nodes],
                         ["node-m-host", "node-w1-host"])
        record = self.read_json(INVENTORY_FILE)
        self.assertEqual(record["timestamp"], TIMESTAMP)
        nodes = cast(list[object], record["nodes"])
        first = cast(dict[str, object], nodes[0])
        self.assertEqual(first["mem_total_kb"], 16384000)
        # Um comando SSH por no, exatamente.
        self.assertEqual(len(self.ssh.calls), 2)

    def test_inventory_round_trips_from_the_saved_file(self):
        coordinator = self.coordinator(self.master_replies())

        inventory_path, nodes = coordinator.collect_inventory(("master", "w1"))

        restored = FleetInventory.from_json(
            inventory_path.read_text(encoding="utf-8"))
        self.assertEqual(restored.nodes, nodes)

    def test_duplicate_hosts_are_refused_with_the_offending_value(self):
        coordinator = self.coordinator(self.master_replies())

        with self.assertRaises(FleetError) as refused:
            coordinator.collect_inventory(("w1", "w1"))

        self.assertIn("w1", str(refused.exception))
        self.assertEqual(self.ssh.calls, [])

    def test_ssh_failure_propagates_with_the_host(self):
        coordinator = self.coordinator(
            {"master": SshError("ssh master: recusado")})

        with self.assertRaises(SshError) as failed:
            coordinator.collect_inventory(("master",))

        self.assertIn("master", str(failed.exception))


class PromoteMasterTest(FleetCoordinatorTestBase):
    def test_master_gets_the_mode_config_with_worker_endpoints(self):
        coordinator = self.coordinator(self.master_replies())

        result = coordinator.promote_master(
            "master", ("w1", "w2"), "large", parallel=8, port=8080)

        # 3 inventarios + 1 comando de modo no master, nessa ordem.
        self.assertEqual([call[0] for call in self.ssh.calls],
                         ["master", "w1", "w2", "master"])
        self.assertEqual(self.ssh.calls[0][1], inventory_sections_command())
        self.assertEqual(
            self.ssh.calls[3][1],
            ["split-llama", "mode", "distributed-master", "--model", "large",
             "--parallel", "8", "--port", "8080",
             "--rpc", "10.0.0.3:50052,10.0.0.4:50052"])
        self.assertEqual(result.endpoints,
                         ("10.0.0.3:50052", "10.0.0.4:50052"))

    def test_promotion_reports_which_node_contributes_each_endpoint(self):
        coordinator = self.coordinator(self.master_replies())

        result = coordinator.promote_master(
            "master", ("w1", "w2"), "large", parallel=8, port=8080)

        self.assertEqual(
            result.contributions,
            (("10.0.0.3:50052", "node-w1-host", "w1-host"),
             ("10.0.0.4:50052", "node-w2-host", "w2-host")))

    def test_coordinator_waits_for_the_master_health_on_its_lan_ip(self):
        coordinator = self.coordinator(self.master_replies())

        coordinator.promote_master("master", ("w1", "w2"), "large",
                                   parallel=8, port=8080)

        self.assertEqual(self.health.checked,
                         ["http://10.0.0.2:8080/health"])

    def test_run_record_references_the_exact_inventory_file(self):
        coordinator = self.coordinator(self.master_replies())

        coordinator.promote_master("master", ("w1", "w2"), "large",
                                   parallel=8, port=8080)

        record = self.read_json(RUN_FILE)
        self.assertEqual(record["scenario"], "distributed")
        self.assertEqual(record["inventory_file"],
                         str(self.run_dir / INVENTORY_FILE))
        self.assertEqual(record["nodes"], ["master", "w1", "w2"])
        params = cast(dict[str, object], record["params"])
        self.assertEqual(params["model"], "large")

    def test_master_inside_the_worker_list_is_refused(self):
        coordinator = self.coordinator(self.master_replies())

        with self.assertRaises(FleetError) as refused:
            coordinator.promote_master("master", ("master", "w1"), "large",
                                       parallel=8, port=8080)

        self.assertIn("master", str(refused.exception))
        self.assertIn("workers", str(refused.exception))
        self.assertEqual(self.ssh.calls, [])

    def test_empty_worker_list_is_refused(self):
        coordinator = self.coordinator(self.master_replies())

        with self.assertRaises(FleetError) as refused:
            coordinator.promote_master("master", (), "large", parallel=8,
                                       port=8080)

        self.assertIn("worker", str(refused.exception))

    def test_worker_in_server_mode_blocks_the_promotion(self):
        replies = self.master_replies()
        replies["w1"] = node_sections("10.0.0.3", "w1-host", role="replica")
        coordinator = self.coordinator(replies)

        with self.assertRaises(FleetError) as refused:
            coordinator.promote_master("master", ("w1", "w2"), "large",
                                       parallel=8, port=8080)

        reported = str(refused.exception)
        self.assertIn("node-w1-host", reported)
        self.assertIn("replica", reported)

    def test_master_without_ram_for_the_model_is_refused_by_name(self):
        replies = self.master_replies()
        replies["master"] = node_sections("10.0.0.2", "m-host",
                                          memtotal_kb=3 * 1024 * 1024)
        coordinator = self.coordinator(replies)

        with self.assertRaises(FleetError) as refused:
            coordinator.promote_master("master", ("w1", "w2"), "large",
                                       parallel=8, port=8080)

        # O GGUF inteiro mora no tmpfs do master: 3 GB nao comportam o
        # Qwen3.5-4B medido (2.74 GB) mais o overhead de 2 GB.
        reported = str(refused.exception)
        self.assertIn("node-m-host", reported)
        self.assertIn("GB", reported)

    def test_unknown_model_profile_is_refused_with_the_valid_set(self):
        coordinator = self.coordinator(self.master_replies())

        with self.assertRaises(FleetError) as refused:
            coordinator.promote_master("master", ("w1",), "gigante",
                                       parallel=8, port=8080)

        self.assertIn("small", str(refused.exception))
        self.assertIn("large", str(refused.exception))

    def test_health_timeout_names_the_url_and_window(self):
        coordinator = self.coordinator(self.master_replies())
        self.health = FakeHealthProbe(ready=False)
        coordinator = FleetCoordinator(
            self.ssh, health=self.health, sleep=self.clock.sleep,
            clock=self.clock.monotonic, run_dir=self.run_dir,
            health_timeout=10.0, now=lambda: TIMESTAMP)

        with self.assertRaises(FleetError) as failed:
            coordinator.promote_master("master", ("w1",), "large",
                                       parallel=8, port=8080)

        reported = str(failed.exception)
        self.assertIn("http://10.0.0.2:8080/health", reported)
        self.assertIn("10.0", reported)
        self.assertGreaterEqual(self.clock.now, 10.0)


class ConfigureReplicasTest(FleetCoordinatorTestBase):
    def replicas(self) -> dict[str, str | Exception]:
        return {
            "r1": node_sections("10.0.0.2", "r1-host"),
            "r2": node_sections("10.0.0.3", "r2-host"),
        }

    def test_each_replica_runs_the_local_server_without_rpc(self):
        coordinator = self.coordinator(self.replicas())

        result = coordinator.configure_replicas(
            ("r1", "r2"), "small", parallel=4, port=8080,
            haproxy_path=self.run_dir / "haproxy.cfg")

        self.assertEqual(
            self.ssh.calls[2][1],
            ["split-llama", "mode", "replica", "--model", "small",
             "--parallel", "4", "--port", "8080"])
        self.assertEqual(self.ssh.calls[3][1], self.ssh.calls[2][1])
        for call in self.ssh.calls[2:]:
            self.assertNotIn("--rpc", call[1])
        self.assertEqual(result.haproxy_path, self.run_dir / "haproxy.cfg")

    def test_health_is_checked_per_replica_before_haproxy_points_at_it(self):
        coordinator = self.coordinator(self.replicas())

        coordinator.configure_replicas(
            ("r1", "r2"), "small", parallel=4, port=8080,
            haproxy_path=self.run_dir / "haproxy.cfg")

        self.assertEqual(self.health.checked,
                         ["http://10.0.0.2:8080/health",
                          "http://10.0.0.3:8080/health"])

    def test_haproxy_file_covers_every_replica_and_run_record_refers_to_it(self):
        coordinator = self.coordinator(self.replicas())

        coordinator.configure_replicas(
            ("r1", "r2"), "small", parallel=4, port=8080,
            haproxy_path=self.run_dir / "haproxy.cfg")

        config = (self.run_dir / "haproxy.cfg").read_text(encoding="utf-8")
        self.assertIn("server node-r1-host 10.0.0.2:8080 check", config)
        self.assertIn("server node-r2-host 10.0.0.3:8080 check", config)
        record = self.read_json(RUN_FILE)
        self.assertEqual(record["scenario"], "replicated")
        params = cast(dict[str, object], record["params"])
        self.assertEqual(params["haproxy_file"],
                         str(self.run_dir / "haproxy.cfg"))

    def test_replica_health_timeout_names_the_node(self):
        coordinator = self.coordinator(self.replicas())
        self.health = FakeHealthProbe(ready=False)
        coordinator = FleetCoordinator(
            self.ssh, health=self.health, sleep=self.clock.sleep,
            clock=self.clock.monotonic, run_dir=self.run_dir,
            health_timeout=5.0, now=lambda: TIMESTAMP)

        with self.assertRaises(FleetError) as failed:
            coordinator.configure_replicas(
                ("r1",), "small", parallel=4, port=8080,
                haproxy_path=self.run_dir / "haproxy.cfg")

        self.assertIn("10.0.0.2", str(failed.exception))

    def test_empty_replica_list_is_refused(self):
        coordinator = self.coordinator(self.replicas())

        with self.assertRaises(FleetError) as refused:
            coordinator.configure_replicas(
                (), "small", parallel=4, port=8080,
                haproxy_path=self.run_dir / "haproxy.cfg")

        self.assertIn("replica", str(refused.exception))


class ResetFleetTest(FleetCoordinatorTestBase):
    def test_every_node_returns_to_worker_mode(self):
        coordinator = self.coordinator(self.master_replies())

        run_path = coordinator.reset(("master", "w1", "w2"))

        for host in ("master", "w1", "w2"):
            self.assertIn(
                (host, ["split-llama", "mode", "worker"]), self.ssh.calls)
        record = json.loads(run_path.read_text(encoding="utf-8"))
        self.assertEqual(record["scenario"], "reset")
        self.assertEqual(record["inventory_file"],
                         str(self.run_dir / INVENTORY_FILE))


class FleetRunRecordTest(unittest.TestCase):
    def test_run_record_round_trips(self):
        record = FleetRunRecord(
            "distributed", TIMESTAMP, "/runs/inventory-1.json",
            ("m", "w1"), (("model", "large"),))

        restored = FleetRunRecord.from_json(record.to_json())

        self.assertEqual(restored, record)


class FleetCliParseTest(unittest.TestCase):
    def test_fleet_subcommands_parse_their_options(self):
        parser = _build_parser()

        inventory = parser.parse_args(["fleet", "inventory", "--nodes", "h1,h2"])
        self.assertEqual(inventory.fleet_command, "inventory")
        self.assertEqual(inventory.nodes, "h1,h2")

        distributed = parser.parse_args(
            ["fleet", "distributed", "--master", "h1", "--workers", "h2,h3",
             "--model", "large", "--parallel", "8"])
        self.assertEqual(distributed.fleet_command, "distributed")
        self.assertEqual(distributed.master, "h1")
        self.assertEqual(distributed.parallel, 8)

        replicated = parser.parse_args(
            ["fleet", "replicated", "--replicas", "r1,r2", "--model", "small"])
        self.assertEqual(replicated.fleet_command, "replicated")

        reset = parser.parse_args(["fleet", "reset", "--nodes", "h1,h2"])
        self.assertEqual(reset.fleet_command, "reset")

    def test_fleet_without_a_subcommand_is_refused(self):
        parser = _build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["fleet"])


if __name__ == "__main__":
    unittest.main()
