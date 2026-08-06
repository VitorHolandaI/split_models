import tempfile
import unittest
from pathlib import Path

from split_lm.llama_rpc.controller import ModeController, ModeTransitionError
from split_lm.llama_rpc.mode_state import (
    MODEL_PROFILES, Mode, ModeConfig, NodeState)
from split_lm.llama_rpc.mode_store import ModeLockedError, StateStore
from split_lm.llama_rpc.server import llama_server_command
from split_lm.llama_rpc.units import (
    SERVER_UNIT, WORKER_UNIT, UnitCommandError)


class FakeUnitManager:
    """Systemd de mentira: registra a ordem das acoes e falha sob demanda."""

    def __init__(self, fail_on: tuple[tuple[str, str], ...] = ()):
        self.calls: list[tuple[str, str]] = []
        self.active: set[str] = set()
        self._fail_on = set(fail_on)

    def start(self, unit: str) -> None:
        self.calls.append(("start", unit))
        if ("start", unit) in self._fail_on:
            raise UnitCommandError(f"start {unit}: induzido pelo teste")
        self.active.add(unit)

    def stop(self, unit: str) -> None:
        self.calls.append(("stop", unit))
        if ("stop", unit) in self._fail_on:
            raise UnitCommandError(f"stop {unit}: induzido pelo teste")
        self.active.discard(unit)


class FakeHealthProbe:
    """Readiness HTTP programado, sem servidor nenhum."""

    def __init__(self, ready: bool = True):
        self._ready = ready
        self.checked: list[str] = []

    def ready(self, url: str) -> bool:
        self.checked.append(url)
        return self._ready


class FakePortProbe:
    """Estado programado da porta RPC, sem socket."""

    def __init__(self, is_open: bool = False):
        self._is_open = is_open
        self.checked: list[int] = []

    def open(self, port: int) -> bool:
        self.checked.append(port)
        return self._is_open


class FakeClock:
    """Relogio de mentira: sleep avanca o tempo sem esperar de verdade."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class ModeControllerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.runtime = Path(self._temporary.name)
        self.store = StateStore(self.runtime)
        self.units = FakeUnitManager()
        self.health = FakeHealthProbe()
        self.port = FakePortProbe()
        self.clock = FakeClock()
        self.controller = ModeController(
            self.store, self.units, health=self.health, port_probe=self.port,
            sleep=self.clock.sleep, clock=self.clock.monotonic,
            health_timeout=10.0, port_timeout=5.0)

    def master_config(self) -> ModeConfig:
        return ModeConfig(Mode.DISTRIBUTED_MASTER, MODEL_PROFILES["large"],
                          parallel=8,
                          rpc_endpoints=("10.0.0.2:50052", "10.0.0.3:50052"))

    def replica_config(self) -> ModeConfig:
        return ModeConfig(Mode.REPLICA, MODEL_PROFILES["small"], parallel=4)

    def seed_model_cache(self) -> Path:
        cache = self.store.model_cache_dir
        cache.mkdir(parents=True)
        leftover = cache / "modelo-parcial.gguf"
        leftover.write_text("download incompleto", encoding="utf-8")
        return leftover


class WorkerTransitionTest(ModeControllerTestBase):
    def test_worker_stops_the_server_and_supervises_the_rpc_worker(self):
        state = self.controller.apply(ModeConfig(Mode.WORKER))

        self.assertIs(state.mode, Mode.WORKER)
        self.assertEqual(self.units.calls,
                         [("stop", SERVER_UNIT), ("start", WORKER_UNIT)])
        self.assertEqual(self.store.read_state().mode, Mode.WORKER)

    def test_returning_to_worker_frees_the_full_model_from_ram(self):
        leftover = self.seed_model_cache()

        self.controller.apply(ModeConfig(Mode.WORKER))

        # O GGUF inteiro mora em tmpfs: so apagar devolve a RAM do no.
        self.assertFalse(leftover.exists())

    def test_hot_cache_option_keeps_the_model_between_runs(self):
        leftover = self.seed_model_cache()

        self.controller.apply(ModeConfig(Mode.WORKER), keep_model_cache=True)

        self.assertTrue(leftover.exists())


class IdleTransitionTest(ModeControllerTestBase):
    def test_idle_stops_both_supervised_units(self):
        state = self.controller.apply(ModeConfig(Mode.IDLE))

        self.assertIs(state.mode, Mode.IDLE)
        self.assertEqual(self.units.active, set())
        self.assertIn(("stop", WORKER_UNIT), self.units.calls)
        self.assertIn(("stop", SERVER_UNIT), self.units.calls)


class MasterTransitionTest(ModeControllerTestBase):
    def test_master_transition_follows_the_planned_order(self):
        state = self.controller.apply(self.master_config())

        self.assertIs(state.mode, Mode.DISTRIBUTED_MASTER)
        self.assertEqual(
            self.units.calls,
            [("stop", WORKER_UNIT), ("start", SERVER_UNIT)])
        self.assertEqual(self.units.active, {SERVER_UNIT})

    def test_master_waits_for_the_rpc_port_to_close_before_serving(self):
        self.controller.apply(self.master_config())

        # Subir o master com a porta RPC local ainda aberta deixaria o no se
        # anunciar como worker e como master ao mesmo tempo.
        self.assertEqual(self.port.checked, [50052])

    def test_master_records_state_and_effective_command_atomically(self):
        self.controller.apply(self.master_config())

        recorded = self.store.read_command()
        self.assertIn("--rpc", recorded)
        state = self.store.read_state()
        self.assertEqual(state.rpc_endpoints,
                         ("10.0.0.2:50052", "10.0.0.3:50052"))
        self.assertEqual(recorded, llama_server_command(self.master_config()))

    def test_master_waits_for_health_on_the_configured_port(self):
        self.controller.apply(self.master_config())

        self.assertEqual(self.health.checked,
                         ["http://127.0.0.1:8080/health"])


class ReplicaTransitionTest(ModeControllerTestBase):
    def test_replica_serves_without_rpc(self):
        state = self.controller.apply(self.replica_config())

        self.assertIs(state.mode, Mode.REPLICA)
        self.assertNotIn("--rpc", self.store.read_command())
        self.assertEqual(self.units.active, {SERVER_UNIT})


class RollbackTest(ModeControllerTestBase):
    def test_health_timeout_restores_the_worker_and_cleans_up(self):
        leftover = self.seed_model_cache()
        self.health = FakeHealthProbe(ready=False)
        self.controller = ModeController(
            self.store, self.units, health=self.health, port_probe=self.port,
            sleep=self.clock.sleep, clock=self.clock.monotonic,
            health_timeout=5.0, port_timeout=5.0)

        with self.assertRaises(ModeTransitionError) as failed:
            self.controller.apply(self.master_config())

        # Falha induzida: o no nao pode ficar sem papel nem com o servidor
        # morto segurando o modelo em RAM.
        self.assertIn("health", str(failed.exception))
        self.assertEqual(self.units.active, {WORKER_UNIT})
        self.assertEqual(self.store.read_state(), NodeState.boot_default())
        self.assertFalse(self.store.command_path.exists())
        self.assertFalse(leftover.exists())

    def test_server_start_failure_rolls_back_to_worker(self):
        units = FakeUnitManager(fail_on=(("start", SERVER_UNIT),))
        controller = ModeController(
            self.store, units, health=self.health, port_probe=self.port,
            sleep=self.clock.sleep, clock=self.clock.monotonic,
            health_timeout=10.0, port_timeout=5.0)

        with self.assertRaises(ModeTransitionError):
            controller.apply(self.replica_config())

        self.assertEqual(units.active, {WORKER_UNIT})
        self.assertEqual(self.store.read_state().mode, Mode.WORKER)

    def test_rollback_names_the_failed_stage(self):
        units = FakeUnitManager(fail_on=(("start", SERVER_UNIT),))
        controller = ModeController(
            self.store, units, health=self.health, port_probe=self.port,
            sleep=self.clock.sleep, clock=self.clock.monotonic,
            health_timeout=10.0, port_timeout=5.0)

        with self.assertRaises(ModeTransitionError) as failed:
            controller.apply(self.replica_config())

        self.assertIn("server.start", str(failed.exception))
        self.assertIn("replica", str(failed.exception))


class ModeLockingTest(ModeControllerTestBase):
    def test_transition_while_locked_is_refused_without_side_effects(self):
        with self.store.lock():
            with self.assertRaises(ModeLockedError):
                self.controller.apply(ModeConfig(Mode.IDLE))

        self.assertEqual(self.units.calls, [])

    def test_status_reads_the_recorded_state(self):
        self.controller.apply(self.replica_config())

        self.assertEqual(self.controller.status().mode, Mode.REPLICA)


if __name__ == "__main__":
    unittest.main()
