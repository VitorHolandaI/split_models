import unittest

from split_lm.llama_rpc.bench import (
    DEFAULT_DEPTHS, DEFAULT_RATIOS, bench_command, device_speed_command)


class BenchCommandTest(unittest.TestCase):
    def test_sweep_is_delegated_to_llama_bench(self):
        command = bench_command("modelo.gguf", "10.0.0.2:50052", DEFAULT_RATIOS)

        # llama-bench ja repete, aquece e imprime CSV. Uma varredura propria
        # significaria reimplementar isso pior, o que ja foi tentado uma vez.
        self.assertEqual(command[0], "llama-bench")
        self.assertEqual(command[command.index("-o") + 1], "csv")
        self.assertIn("-r", command)

    def test_ratio_separator_is_the_one_llama_bench_expects(self):
        command = bench_command("modelo.gguf", "10.0.0.2:50052", ("90/10", "50/50"))

        # llama-bench separa dispositivos com barra e razoes com virgula; o
        # llama-cli usa virgula dentro da razao. Trocar os dois mede outra
        # configuracao sem dar erro.
        self.assertEqual(command[command.index("-ts") + 1], "90/10,50/50")

    def test_local_run_carries_no_ratio(self):
        command = bench_command("modelo.gguf", "", DEFAULT_RATIOS)

        # Sem worker remoto nao ha o que dividir; passar razao mediria uma
        # configuracao que nao existe.
        self.assertNotIn("-ts", command)
        self.assertNotIn("--rpc", command)

    def test_depths_sweep_the_kv_cache_axis(self):
        command = bench_command("modelo.gguf", depths=(0, 4096, 16384))

        # Profundidade e o KV cache ja preenchido: o eixo em que a frota soma
        # capacidade, porque os pesos sao fixos e o cache cresce com o contexto.
        self.assertEqual(command[command.index("-d") + 1], "0,4096,16384")

    def test_default_depths_start_at_zero(self):
        self.assertEqual(DEFAULT_DEPTHS[0], 0)


class ModelBatteryScriptTest(unittest.TestCase):
    def setUp(self):
        from pathlib import Path
        self.script = (Path(__file__).resolve().parents[1] / "deploy" /
                       "bench_models.sh").read_text(encoding="utf-8")

    def test_battery_covers_the_documented_model_list(self):
        from split_lm.llama_rpc.examples import KNOWN_MODELS

        # A lista do script tem que acompanhar a de examples.py, senao a
        # bateria mede um conjunto e a ajuda ao operador anuncia outro.
        for model in KNOWN_MODELS:
            self.assertIn(model.repo, self.script)

    def test_a_model_that_does_not_fit_is_recorded_not_fatal(self):
        # Modelo grande demais faz o llama-bench sair com erro, e isso e
        # resultado: "nao coube" e o dado que a bateria existe para produzir.
        self.assertNotIn("set -euo pipefail", self.script)
        self.assertIn("nao coube ou falhou", self.script)

    def test_ratios_span_both_ends_of_the_curve(self):
        # Dois pontos perto de uma ponta so mostram a inclinacao ali. 100/0
        # isola o pedagio da rede, 10/90 mostra o custo do dispositivo lento, e
        # os do meio dizem se a queda entre eles e linear.
        for ratio in ("100/0", "90/10", "50/50", "10/90"):
            self.assertIn(f'"{ratio}"', self.script)

    def test_results_are_written_as_they_happen(self):
        # A bateria leva horas; uma queda no meio nao pode custar o que ja
        # terminou.
        self.assertIn('>> "$output_csv"', self.script)


class DeviceSpeedTest(unittest.TestCase):
    def test_ranking_run_is_short_enough_to_run_at_startup(self):
        quick = device_speed_command("modelo.gguf")
        full = bench_command("modelo.gguf", "10.0.0.2:50052", DEFAULT_RATIOS)

        self.assertEqual(quick[quick.index("-d") + 1], "0")
        self.assertLess(int(quick[quick.index("-r") + 1]),
                        int(full[full.index("-r") + 1]))
        self.assertLess(int(quick[quick.index("-n") + 1]),
                        int(full[full.index("-n") + 1]))

    def test_ranking_run_compares_machines_not_splits(self):
        quick = device_speed_command("modelo.gguf", "10.0.0.2:50052")

        # A pergunta e qual maquina deve ser master, nao qual proporcao usar.
        self.assertNotIn("-ts", quick)


if __name__ == "__main__":
    unittest.main()
