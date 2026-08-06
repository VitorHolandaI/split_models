import contextlib
import email.message
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from split_lm.llama_rpc.cli import _build_parser, main as llama_main
from split_lm.llama_rpc.health import HttpHealthProbe, TcpPortProbe, wait_until
from split_lm.llama_rpc.mode_state import (
    MODEL_PROFILES, Mode, ModeConfig, ModeConfigError, NodeState)
from split_lm.llama_rpc.mode_store import (
    ModeLockedError, StateError, StateStore)
from split_lm.llama_rpc.server import llama_server_command


class FakeOpener:
    """Resposta de HTTP programada, sem abrir socket de verdade."""

    def __init__(self, failure: Exception | None = None):
        self._failure = failure
        self.requested: list[str] = []

    def __call__(self, url: str) -> object:
        self.requested.append(url)
        if self._failure is not None:
            raise self._failure
        return object()


class FakeClock:
    """Relogio monotonico de mentira: sleep faz o tempo andar sem esperar."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class ModeParseTest(unittest.TestCase):
    def test_every_planned_mode_round_trips_from_its_cli_name(self):
        for name in ("worker", "distributed-master", "replica", "idle"):
            self.assertIs(Mode.from_text(name), Mode(name))

    def test_unknown_mode_names_the_expected_set(self):
        with self.assertRaises(ModeConfigError) as refused:
            Mode.from_text("master")

        # O nome antigo do atalho nao pode virar modo por engano: a mensagem
        # tem que dizer o que e valido, nao so que falhou.
        self.assertIn("master", str(refused.exception))
        self.assertIn("distributed-master", str(refused.exception))


class ModeConfigValidationTest(unittest.TestCase):
    def test_distributed_master_requires_explicit_rpc_endpoints(self):
        with self.assertRaises(ModeConfigError) as refused:
            ModeConfig(Mode.DISTRIBUTED_MASTER, model=MODEL_PROFILES["large"])

        # Sem --rpc o master baixaria o GGUF inteiro e subiria sem worker
        # nenhum, parecendo distribuido sem estar.
        self.assertIn("rpc_endpoints", str(refused.exception))

    def test_server_modes_require_a_model(self):
        for mode in (Mode.DISTRIBUTED_MASTER, Mode.REPLICA):
            with self.assertRaises(ModeConfigError) as refused:
                ModeConfig(mode, rpc_endpoints=("10.0.0.2:50052",))

            self.assertIn("model", str(refused.exception))

    def test_worker_and_idle_carry_no_server_configuration(self):
        with self.assertRaises(ModeConfigError):
            ModeConfig(Mode.WORKER, model=MODEL_PROFILES["small"])
        with self.assertRaises(ModeConfigError):
            ModeConfig(Mode.IDLE, rpc_endpoints=("10.0.0.2:50052",))

    def test_parallel_and_port_must_be_usable(self):
        with self.assertRaises(ModeConfigError):
            ModeConfig(Mode.WORKER, parallel=0)
        with self.assertRaises(ModeConfigError):
            ModeConfig(Mode.WORKER, port=70000)

    def test_valid_configs_are_accepted(self):
        master = ModeConfig(Mode.DISTRIBUTED_MASTER, MODEL_PROFILES["large"],
                            parallel=8, rpc_endpoints=("10.0.0.2:50052",))
        replica = ModeConfig(Mode.REPLICA, MODEL_PROFILES["small"], parallel=4)

        self.assertEqual(master.parallel, 8)
        self.assertEqual(replica.port, 8080)


class NodeStateSerializationTest(unittest.TestCase):
    def test_state_survives_a_json_round_trip(self):
        state = NodeState(Mode.DISTRIBUTED_MASTER, MODEL_PROFILES["large"], 8,
                          ("10.0.0.2:50052", "10.0.0.3:50052"), 8080)

        restored = NodeState.from_json(state.to_json())

        self.assertEqual(restored, state)

    def test_boot_default_is_a_plain_worker(self):
        # O estado pode nao existir ainda no primeiro boot; o painel e o
        # status precisam de uma resposta mesmo assim.
        self.assertIs(NodeState.boot_default().mode, Mode.WORKER)

    def test_corrupt_state_names_the_offending_shape(self):
        with self.assertRaises(ValueError) as refused:
            NodeState.from_json('{"mode": "worker"}')

        self.assertIn("mode", str(refused.exception))

    def test_unknown_mode_in_state_is_refused(self):
        payload = json.dumps({"mode": "shell", "model": None, "parallel": 4,
                              "rpc_endpoints": [], "port": 8080})

        with self.assertRaises(ValueError) as refused:
            NodeState.from_json(payload)

        self.assertIn("shell", str(refused.exception))


class ServerCommandTest(unittest.TestCase):
    def test_master_command_spans_the_explicit_endpoints_by_layer(self):
        config = ModeConfig(Mode.DISTRIBUTED_MASTER, MODEL_PROFILES["large"],
                            parallel=8,
                            rpc_endpoints=("10.0.0.2:50052", "10.0.0.3:50052"))

        command = llama_server_command(config)

        self.assertEqual(command[:3],
                         ["llama-server", "-hf", MODEL_PROFILES["large"]])
        self.assertEqual(command[command.index("--rpc") + 1],
                         "10.0.0.2:50052,10.0.0.3:50052")
        self.assertEqual(command[command.index("--split-mode") + 1], "layer")
        self.assertEqual(command[command.index("--parallel") + 1], "8")

    def test_server_exposes_batching_metrics_and_reachable_http(self):
        config = ModeConfig(Mode.REPLICA, MODEL_PROFILES["small"], parallel=4)

        command = llama_server_command(config)

        # Continuous batching e default mas a flag vai explicita: o benchmark
        # confirma batching real por n_busy_slots_per_decode.
        self.assertIn("--cont-batching", command)
        self.assertIn("--metrics", command)
        # HAProxy externo precisa alcancar a replica; o default 127.0.0.1 a
        # esconderia da LAN.
        self.assertEqual(command[command.index("--host") + 1], "0.0.0.0")
        self.assertIn("--no-mmproj", command)

    def test_local_replica_omits_rpc_entirely(self):
        config = ModeConfig(Mode.REPLICA, MODEL_PROFILES["small"], parallel=4)

        command = llama_server_command(config)

        # Uma replica com --rpc dependeria de workers que nao existem e
        # deixaria de ser uma copia completa local.
        self.assertNotIn("--rpc", command)
        self.assertNotIn("--split-mode", command)

    def test_non_server_modes_have_no_server_command(self):
        with self.assertRaises(ModeConfigError):
            llama_server_command(ModeConfig(Mode.WORKER))


class StateStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.store = StateStore(Path(self._temporary.name))

    def test_missing_state_reads_as_the_boot_default(self):
        self.assertEqual(self.store.read_state(), NodeState.boot_default())

    def test_state_and_command_round_trip_atomically(self):
        state = NodeState(Mode.REPLICA, MODEL_PROFILES["small"], 4, (), 8080)
        command = llama_server_command(
            ModeConfig(Mode.REPLICA, MODEL_PROFILES["small"], parallel=4))

        self.store.write_state(state)
        self.store.write_command(command)

        self.assertEqual(self.store.read_state(), state)
        self.assertEqual(self.store.read_command(), command)
        # Escrita atomica: nada de arquivo temporario esquecido no diretorio.
        leftovers = [p.name for p in Path(self._temporary.name).iterdir()
                     if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_read_command_without_a_recorded_one_is_an_explicit_error(self):
        with self.assertRaises(StateError) as missing:
            self.store.read_command()

        self.assertIn("effective-command", str(missing.exception))

    def test_command_with_non_string_argument_is_refused(self):
        self.store.write_command(["llama-server", "--parallel", 4])

        with self.assertRaises(ValueError):
            self.store.read_command()

    def test_clear_removes_state_and_command(self):
        self.store.write_state(NodeState.boot_default())
        self.store.write_command(["llama-server"])

        self.store.clear()

        self.assertFalse(self.store.state_path.exists())
        self.assertFalse(self.store.command_path.exists())

    def test_caches_live_in_separate_directories(self):
        self.assertEqual(self.store.model_cache_dir.name, "model-cache")
        self.assertEqual(self.store.rpc_cache_dir.name, "rpc-cache")


class ModeLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.store = StateStore(Path(self._temporary.name))

    def test_second_transition_is_refused_while_the_lock_is_held(self):
        with self.store.lock():
            with self.assertRaises(ModeLockedError):
                with self.store.lock():
                    pass

    def test_lock_is_available_again_after_release(self):
        with self.store.lock():
            pass
        with self.store.lock():
            pass


class HealthProbeTest(unittest.TestCase):
    def test_http_200_means_ready(self):
        probe = HttpHealthProbe(opener=FakeOpener())

        self.assertTrue(probe.ready("http://127.0.0.1:8080/health"))

    def test_loading_model_and_refused_connection_mean_not_ready(self):
        loading = HttpHealthProbe(opener=FakeOpener(
            urllib.error.HTTPError("u", 503, "loading", email.message.Message(),
                                   None)))
        refused = HttpHealthProbe(opener=FakeOpener(
            urllib.error.URLError("connection refused")))

        self.assertFalse(loading.ready("http://127.0.0.1:8080/health"))
        self.assertFalse(refused.ready("http://127.0.0.1:8080/health"))

    def test_tcp_probe_distinguishes_open_from_closed(self):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            probe = TcpPortProbe("127.0.0.1")

            self.assertTrue(probe.open(port))
        self.assertFalse(probe.open(port))

    def test_wait_until_polls_until_the_check_turns_true(self):
        clock = FakeClock()
        attempts = iter([False, False, True])

        result = wait_until(lambda: next(attempts), timeout=10.0,
                            sleep=clock.sleep, clock=clock.monotonic)

        self.assertTrue(result)
        self.assertGreater(clock.now, 0.0)

    def test_wait_until_gives_up_after_the_timeout(self):
        clock = FakeClock()

        result = wait_until(lambda: False, timeout=5.0,
                            sleep=clock.sleep, clock=clock.monotonic)

        self.assertFalse(result)
        self.assertGreaterEqual(clock.now, 5.0)


class ModeCliInterfaceTest(unittest.TestCase):
    def test_mode_subcommand_parses_every_target(self):
        parser = _build_parser()

        for target in ("worker", "distributed-master", "replica", "idle",
                       "status"):
            options = parser.parse_args(["mode", target])
            self.assertEqual(options.target, target)

    def test_master_options_parse_endpoints_and_parallel(self):
        options = _build_parser().parse_args(
            ["mode", "distributed-master", "--model", "large", "--parallel", "8",
             "--rpc", "10.0.0.2:50052,10.0.0.3:50052"])

        self.assertEqual(options.model, "large")
        self.assertEqual(options.parallel, 8)
        self.assertEqual(options.rpc, "10.0.0.2:50052,10.0.0.3:50052")

    def test_invalid_mode_config_exits_non_zero_without_touching_anything(self):
        with tempfile.TemporaryDirectory() as runtime:
            code = llama_main(["mode", "distributed-master", "--model", "large",
                               "--runtime-dir", runtime])

        # Sem --rpc o master e recusado na validacao, antes de qualquer
        # systemctl: falhar fechado e o comportamento seguro.
        self.assertEqual(code, 1)

    def test_status_on_a_fresh_runtime_dir_reports_the_boot_default(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as runtime:
            with contextlib.redirect_stdout(output):
                code = llama_main(["mode", "status", "--runtime-dir", runtime])

        self.assertEqual(code, 0)
        self.assertIn("worker", output.getvalue())


if __name__ == "__main__":
    unittest.main()
