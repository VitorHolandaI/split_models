import unittest

from split_lm.llama_rpc.context_sweep import (
    ContextResult, format_sweep, kv_cache_bytes, largest_loaded, parse_rates,
    sweep_command)


def loaded(context: int, rate: float, endpoints: str = "") -> ContextResult:
    return ContextResult(context, endpoints, True, rate * 4, rate, "ok")


def refused(context: int, detail: str, endpoints: str = "") -> ContextResult:
    return ContextResult(context, endpoints, False, 0.0, 0.0, detail)


class KvCacheSizeTest(unittest.TestCase):
    def test_cache_grows_linearly_with_context(self):
        small = kv_cache_bytes(36, 2, 128, 2048)
        large = kv_cache_bytes(36, 2, 128, 131072)

        self.assertEqual(large, small * 64)

    def test_long_context_dwarfs_a_small_model(self):
        # Os pesos sao fixos, o KV nao. Em 128k tokens o cache de um 3B passa de
        # 4 GB, mais que muitos modelos inteiros -- e por isso que contexto, e
        # nao tamanho de modelo, e o eixo em que a frota soma capacidade.
        at_128k = kv_cache_bytes(36, 2, 128, 131072)

        self.assertGreater(at_128k, 4 * 1024 ** 3)

    def test_impossible_geometry_is_refused_with_the_values(self):
        with self.assertRaises(ValueError) as refused_call:
            kv_cache_bytes(36, 0, 128, 2048)

        self.assertIn("kv_heads=0", str(refused_call.exception))


class SweepCommandTest(unittest.TestCase):
    def test_local_run_carries_no_rpc_flags(self):
        command = sweep_command("modelo.gguf", 8192)

        self.assertNotIn("--rpc", command)
        self.assertEqual(command[command.index("-c") + 1], "8192")

    def test_split_run_targets_the_endpoints_by_layer(self):
        command = sweep_command("modelo.gguf", 8192, "10.0.0.2:50052")

        self.assertEqual(command[command.index("--rpc") + 1], "10.0.0.2:50052")
        self.assertEqual(command[command.index("--split-mode") + 1], "layer")

    def test_sweep_never_negotiates_its_way_into_a_fit(self):
        command = sweep_command("modelo.gguf", 8192, "10.0.0.2:50052")

        # --fit on procura por tentativa e erro, reabrindo uma sessao RPC por
        # tentativa e podendo nao convergir. Cada ponto do sweep tem que
        # carregar ou falhar rapido, nao negociar.
        self.assertNotIn("--fit", command)
        self.assertEqual(command[command.index("-ngl") + 1], "99")

    def test_run_is_not_interactive(self):
        command = sweep_command("modelo.gguf", 8192)

        self.assertIn("-st", command)


class ModelLimitTest(unittest.TestCase):
    def test_sweep_stops_at_the_model_limit(self):
        from split_lm.llama_rpc.context_sweep import run_context_sweep

        attempted: list[int] = []

        def fake_measure(_path, context, _endpoints="", _timeout=0.0):
            attempted.append(context)
            return loaded(context, 10.0)

        import split_lm.llama_rpc.context_sweep as module
        original = module.measure_context
        module.measure_context = fake_measure
        try:
            # Acima de n_ctx_train o modelo responde pior sem medir nada novo,
            # e cada ponto custa uma carga inteira do modelo.
            run_context_sweep("m.gguf", "", (4096, 8192, 16384), max_context=8192)
        finally:
            module.measure_context = original

        self.assertEqual(attempted, [4096, 8192])


class RateParsingTest(unittest.TestCase):
    def test_reads_both_rates_from_the_summary_line(self):
        prompt, generation = parse_rates(
            "[ Prompt: 60.8 t/s | Generation: 42.6 t/s ]")

        self.assertEqual((prompt, generation), (60.8, 42.6))

    def test_output_without_a_summary_reports_zero(self):
        self.assertEqual(parse_rates("failed to allocate Vulkan0 buffer"), (0.0, 0.0))


class SweepReportTest(unittest.TestCase):
    def test_largest_loaded_ignores_the_failing_point(self):
        results = (loaded(4096, 12.0), loaded(8192, 9.0),
                   refused(16384, "sem memoria: failed to allocate"))

        self.assertEqual(largest_loaded(results), 8192)

    def test_nothing_loaded_reports_zero_rather_than_crashing(self):
        self.assertEqual(largest_loaded((refused(4096, "sem memoria"),)), 0)

    def test_report_states_where_the_sweep_stopped(self):
        rendered = format_sweep((loaded(4096, 12.0),
                                 refused(8192, "sem memoria: failed to allocate")))

        self.assertIn("maior contexto que carregou: 4096", rendered)
        self.assertIn("sem memoria", rendered)

    def test_empty_sweep_says_so(self):
        self.assertIn("nenhuma medicao", format_sweep(()))


if __name__ == "__main__":
    unittest.main()
