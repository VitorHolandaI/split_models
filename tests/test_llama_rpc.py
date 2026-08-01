import subprocess
import threading
import unittest

from split_lm.errors import ClusterError
from split_lm.llama_rpc.cli import _build_parser, format_workers, main as llama_main
from split_lm.llama_rpc.discovery import (
    PROTOCOL, RpcReplyError, RpcWorker, decode_reply, discover_workers,
    encode_reply, local_worker, rpc_endpoints, serve_discovery)
from split_lm.llama_rpc.examples import GgufModel, KNOWN_MODELS, format_llama_examples
from split_lm.llama_rpc.master import (
    DEFAULT_GGUF_REPO, list_devices_command, llama_cli_command)
from split_lm.llama_rpc.process import (
    assert_clean_exit, rpc_server_command, serve_until_server_exits)


class FakeRpcServer:
    """Stand-in for the ggml-rpc-server subprocess, without spawning one."""

    def __init__(self, exit_code: int = 0, output: str = "", delay: float = 0.0):
        self._exit_code = exit_code
        self._delay = delay
        self.stdout = FakeServerOutput(output)
        self.terminated = False

    def wait(self) -> int:
        if self._delay:
            threading.Event().wait(self._delay)
        return self._exit_code

    def terminate(self) -> None:
        self.terminated = True


class FakeServerOutput:
    """Readable stand-in for the subprocess stdout pipe."""

    def __init__(self, text: str):
        self._text = text

    def read(self) -> str:
        return self._text


class RpcDiscoveryTest(unittest.TestCase):
    def _serve(self, worker: RpcWorker, port: int) -> None:
        stop, ready = threading.Event(), threading.Event()
        thread = threading.Thread(
            target=serve_discovery, args=(worker, port, stop, ready), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(stop.set)
        self.assertTrue(ready.wait(5), f"discovery nao ligou udp/{port}")

    def test_reply_survives_a_round_trip(self):
        worker = local_worker("node-abcd1234", 50052)

        restored = decode_reply(encode_reply(worker), "192.168.0.11")

        self.assertEqual(restored.node_id, "node-abcd1234")
        self.assertEqual(restored.address, "192.168.0.11")
        self.assertEqual(restored.rpc_port, 50052)
        self.assertEqual(restored.endpoint, "192.168.0.11:50052")

    def test_reply_from_another_protocol_names_the_node_to_reinstall(self):
        stale = b'{"protocol": 999, "node_id": "node-a", "hostname": "h", "rpc_port": 1}'

        with self.assertRaises(RpcReplyError) as refused:
            decode_reply(stale, "192.168.0.12")

        # A mensagem tem que levar o endereco e o esperado: o operador precisa
        # saber qual maquina reconstruir, nao so que houve divergencia.
        self.assertIn("999", str(refused.exception))
        self.assertIn("192.168.0.12", str(refused.exception))
        self.assertIn(str(PROTOCOL), str(refused.exception))

    def test_malformed_reply_reports_the_offending_payload(self):
        with self.assertRaises(RpcReplyError) as refused:
            decode_reply(b"nao e json", "192.168.0.13")

        self.assertIn("nao e json", str(refused.exception))
        self.assertIn("192.168.0.13", str(refused.exception))

    def test_discovery_finds_a_worker_and_reports_it_once(self):
        self._serve(local_worker("node-a", 50052), 29530)

        found = discover_workers(29530, timeout=1.0, broadcast_address="127.0.0.1",
                                 probes=3)

        self.assertEqual([worker.node_id for worker in found], ["node-a"])
        self.assertEqual(found[0].address, "127.0.0.1")


class RpcEndpointTest(unittest.TestCase):
    def test_endpoints_follow_node_id_order(self):
        workers = (RpcWorker("node-b", "hb", "10.0.0.3", 50052),
                   RpcWorker("node-a", "ha", "10.0.0.2", 50052))

        # Ordenar e o que torna a atribuicao de layers reproduzivel entre runs.
        ordered = tuple(sorted(workers, key=lambda worker: worker.node_id))

        self.assertEqual(rpc_endpoints(ordered), "10.0.0.2:50052,10.0.0.3:50052")

    def test_empty_fleet_is_refused_with_the_expected_shape(self):
        with self.assertRaises(ValueError) as refused:
            rpc_endpoints(())

        self.assertIn("ao menos um", str(refused.exception))


class RpcServerCommandTest(unittest.TestCase):
    def test_server_keeps_its_cache_in_ram_and_picks_its_own_devices(self):
        command = rpc_server_command("192.168.0.11")

        self.assertIn("--cache", command)
        self.assertIn("/run/split-models/llama-cache", command)
        # Sem --device o servidor enumera o que achar e cai em CPU num host sem
        # Vulkan, que e o que permite uma imagem so servir os dois tipos de no.
        self.assertNotIn("--device", command)
        self.assertEqual(command[:5],
                         ["ggml-rpc-server", "--host", "192.168.0.11", "--port", "50052"])


class MasterCommandTest(unittest.TestCase):
    def test_master_splits_by_layer_and_sizes_the_offload_at_run_time(self):
        command = llama_cli_command(DEFAULT_GGUF_REPO, "10.0.0.2:50052")

        self.assertEqual(command[:3], ["llama-cli", "-hf", DEFAULT_GGUF_REPO])
        self.assertIn("--rpc", command)
        self.assertEqual(command[command.index("--split-mode") + 1], "layer")
        self.assertEqual(command[command.index("-ngl") + 1], "auto")
        self.assertEqual(command[command.index("--fit") + 1], "on")

    def test_master_does_not_pin_a_static_tensor_split(self):
        command = llama_cli_command(DEFAULT_GGUF_REPO, "10.0.0.2:50052")

        # As GPUs alvo compartilham memoria com o host, entao uma razao fixa so
        # valeria na maquina onde foi medida.
        self.assertNotIn("--tensor-split", command)

    def test_run_without_a_terminal_takes_one_turn_and_exits(self):
        scripted = llama_cli_command(DEFAULT_GGUF_REPO, "10.0.0.2:50052",
                                     interactive=False)
        at_console = llama_cli_command(DEFAULT_GGUF_REPO, "10.0.0.2:50052",
                                       interactive=True)

        # llama-cli abre um chat por padrao e espera stdin para sempre quando
        # nao ha terminal; medido girando um core a 93% com RPC no caminho.
        self.assertIn("-st", scripted)
        self.assertNotIn("-st", at_console)

    def test_extra_arguments_reach_llama_cli_unchanged(self):
        command = llama_cli_command(DEFAULT_GGUF_REPO, "10.0.0.2:50052",
                                    ["-p", "ola", "-n", "16"])

        self.assertEqual(command[-4:], ["-p", "ola", "-n", "16"])

    def test_device_listing_targets_the_same_endpoints(self):
        command = list_devices_command("10.0.0.2:50052,10.0.0.3:50052")

        self.assertEqual(command,
                         ["llama-cli", "--rpc", "10.0.0.2:50052,10.0.0.3:50052",
                          "--list-devices"])


class RpcLifetimeTest(unittest.TestCase):
    def test_discovery_stops_when_the_rpc_server_exits(self):
        worker = RpcWorker("node-a", "host-a", "127.0.0.1", 50052)

        exit_code = serve_until_server_exits(worker, FakeRpcServer(), 29531)

        # Um worker que continuasse respondendo depois do servidor morrer
        # seguiria recebendo layers que nao consegue servir.
        self.assertEqual(exit_code, 0)
        self.assertEqual(discover_workers(29531, timeout=0.3,
                                          broadcast_address="127.0.0.1"), ())

    def test_failed_server_names_the_machine_stage_and_output(self):
        worker = RpcWorker("node-a", "host-a", "10.0.0.2", 50052)
        server = FakeRpcServer(exit_code=3, output="vulkan: device lost")

        with self.assertRaises(ClusterError) as failure:
            assert_clean_exit(worker, server, 3)

        reported = str(failure.exception)
        self.assertIn("node-a", reported)
        self.assertIn("10.0.0.2", reported)
        self.assertIn("rpc_server.run", reported)
        self.assertIn("vulkan: device lost", reported)

    def test_clean_exit_raises_nothing(self):
        worker = RpcWorker("node-a", "host-a", "10.0.0.2", 50052)

        assert_clean_exit(worker, FakeRpcServer(), 0)


class LlamaExampleTest(unittest.TestCase):
    def test_examples_cover_models_that_need_the_split_and_ones_that_do_not(self):
        repos = [model.repo for model in KNOWN_MODELS]

        self.assertIn(DEFAULT_GGUF_REPO, repos)
        # Sem os dois lados a lista nao mostra por que o split existe.
        self.assertTrue(any(model.needs_split for model in KNOWN_MODELS))
        self.assertTrue(any(not model.needs_split for model in KNOWN_MODELS))
        self.assertGreater(len({repo.split("/")[0] for repo in repos}), 1)

    def test_every_model_carries_a_measured_size(self):
        for model in KNOWN_MODELS:
            # Tamanho lido da API, nao estimado: e o numero que decide se cabe
            # num no de 8 GB, entao chute aqui vira decisao errada la.
            self.assertGreater(model.gigabytes, 0.0)
            self.assertTrue(model.quant)

    def test_master_ram_caps_the_model_no_matter_how_many_nodes(self):
        big = GgufModel("x/y", "Q4_K_M", 12.11, "")
        medium = GgufModel("x/y", "Q8_0", 9.53, "")

        # O master baixa o GGUF inteiro e da mmap; numa appliance stateless esse
        # cache e tmpfs, entao o arquivo ocupa RAM do master enquanto carregado.
        # Somar workers nao levanta esse teto.
        self.assertFalse(big.fits_master(12.0))
        self.assertTrue(medium.fits_master(12.0))

    def test_split_threshold_leaves_room_for_the_rest_of_the_node(self):
        # 8 GB e o total do host; o OS, o servidor RPC e o KV cache saem do
        # mesmo orcamento, entao o corte fica abaixo disso.
        borderline = GgufModel("x/y", "Q4_K_M", 6.5, "")
        comfortable = GgufModel("x/y", "Q4_K_M", 5.0, "")

        self.assertTrue(borderline.needs_split)
        self.assertFalse(comfortable.needs_split)

    def test_examples_warn_about_the_misleading_401(self):
        rendered = format_llama_examples()

        # Repositorio inexistente responde 401 "Invalid username or password",
        # que manda o operador procurar um problema de login inexistente.
        self.assertIn("401", rendered)
        self.assertIn("huggingface.co", rendered)

    def test_examples_explain_when_the_run_is_interactive(self):
        rendered = format_llama_examples()

        self.assertIn("-st", rendered)
        self.assertIn("/exit", rendered)

    def test_examples_are_reachable_from_the_command_line(self):
        self.assertEqual(llama_main(["exemplos"]), 0)
        self.assertEqual(llama_main(["--comandos-exemplos"]), 0)


class LlamaCliInterfaceTest(unittest.TestCase):
    def test_every_subcommand_is_reachable(self):
        parser = _build_parser()

        for command in ("worker", "nodes", "master"):
            self.assertEqual(parser.parse_args([command]).command, command)

    def test_master_forwards_arguments_written_after_the_separator(self):
        options = _build_parser().parse_args(
            ["master", "--discover", "--", "-p", "ola"])

        self.assertEqual(options.llama_args, ["-p", "ola"])
        self.assertTrue(options.discover)

    def test_empty_fleet_listing_exits_non_zero(self):
        code = llama_main(["nodes", "--broadcast", "127.0.0.1", "--timeout", "0.2"])

        self.assertEqual(code, 1)

    def test_worker_and_nodes_report_the_insecure_transport(self):
        helped = subprocess.run(
            ["python", "-m", "split_lm.llama_rpc.cli", "--help"],
            check=True, capture_output=True, text=True)

        self.assertIn("LAN isolada", helped.stdout)

    def test_empty_listing_says_so_instead_of_printing_a_bare_header(self):
        self.assertIn("nenhum worker", format_workers(()))


if __name__ == "__main__":
    unittest.main()
