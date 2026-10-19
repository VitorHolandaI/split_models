import unittest

from split_lm.llama_rpc.capacity_load import (
    HttpCompletionClient, RequestRecord, exact_prompt)
from split_lm.llama_rpc.capacity_metrics import (
    PointSummary, detect_saturation, percentile, summarize_point,
    write_request_jsonl, write_summary_csv)
from split_lm.llama_rpc.capacity_profiles import (
    PROFILES, CapacityProfileError, CapacityPoint, get_profile)


class FakeCounter:
    """Token counter over a known text/unit mapping."""

    def __init__(self, unit_tokens: int) -> None:
        self._unit_tokens = unit_tokens
        self.calls = 0

    def count(self, text: str) -> int:
        self.calls += 1
        units = len(text) // 22
        return units * self._unit_tokens


class CapacityProfilesTest(unittest.TestCase):
    def test_exactly_four_profiles_with_the_plan_names(self):
        self.assertEqual(
            set(PROFILES),
            {"users-2k", "context-2users", "nodes-8users-2k",
             "capacity-envelope"})

    def test_profiles_carry_the_official_model_and_three_measured_waves(self):
        for profile in PROFILES.values():
            self.assertEqual(profile.generated, 128)
            self.assertEqual(profile.warmup_waves, 1)
            self.assertEqual(profile.measured_waves, 3)

    def test_each_profile_varies_only_its_planned_axis(self):
        axes = {
            "users-2k": ("users", (2, 4, 8, 16, 32)),
            "context-2users": ("context", (512, 1024, 2048, 4096, 8192)),
            "nodes-8users-2k": ("nodes", (1, 2, 4, 8)),
        }
        for name, (axis, expected) in axes.items():
            points = PROFILES[name].points
            varying = sorted({getattr(point, axis) for point in points})
            fixed = {field for field in ("users", "context", "nodes")
                     if field != axis}
            self.assertEqual(varying, list(expected))
            for field in fixed:
                values = {getattr(point, field) for point in points}
                self.assertEqual(len(values), 1,
                                 f"{name} deveria variar so {axis}")

    def test_unknown_profile_names_the_valid_set(self):
        with self.assertRaises(CapacityProfileError) as refused:
            get_profile("users-9999")
        self.assertIn("users-9999", str(refused.exception))
        self.assertIn("users-2k", str(refused.exception))

    def test_malformed_grid_is_refused(self):
        from split_lm.llama_rpc.capacity_profiles import CapacityProfile

        with self.assertRaises(CapacityProfileError) as refused:
            CapacityProfile("users-2k", "m", 0, 1, 3, (CapacityPoint(1, 1, 1),))
        self.assertIn("generated=0", str(refused.exception))


class ExactPromptTest(unittest.TestCase):
    def test_reaches_the_exact_target_token_count(self):
        prompt = exact_prompt(64, FakeCounter(2))

        self.assertEqual(FakeCounter(2).count(prompt), 64)

    def test_bounded_steps_on_an_impossible_target(self):
        # O tokenizador pode nunca contar exatamente `target_tokens` para o
        # bloco de texto repetitivo; a funcao aceita ate 5% (capped 6) de
        # tolerancia e retorna o melhor ajuste em vez de lancar excecao.
        counter = FakeCounter(3)

        prompt = exact_prompt(61, counter, max_steps=6)
        self.assertLessEqual(abs(FakeCounter(3).count(prompt) - 61), 6)
        self.assertLessEqual(counter.calls, 8)


class PercentileTest(unittest.TestCase):
    def test_nearest_rank_matches_hand_computed_values(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]

        self.assertEqual(percentile(values, 50), 0.3)
        self.assertEqual(percentile(values, 95), 0.5)

    def test_empty_series_yields_zero(self):
        self.assertEqual(percentile([], 95), 0.0)


class SaturationTest(unittest.TestCase):
    @staticmethod
    def _row(users: int, tok_s: float, p95: float) -> PointSummary:
        return PointSummary("users-2k", "distributed", users, 2048, 1,
                            users, 0, users * 128, 0.1, 0.1, 0.1,
                            1.0, p95, p95, tok_s, 2, False)

    def test_throughput_that_stops_scaling_is_saturated(self):
        rows = [self._row(2, 100.0, 2.0), self._row(4, 195.0, 2.5),
                self._row(8, 200.0, 3.0)]

        flags = detect_saturation(rows)

        self.assertEqual(flags, [False, False, True])

    def test_latency_bound_saturates_even_while_scaling(self):
        rows = [self._row(2, 100.0, 2.0), self._row(4, 200.0, 90.0)]

        flags = detect_saturation(rows)

        self.assertEqual(flags, [False, True])


class ArtifactWritersTest(unittest.TestCase):
    def test_jsonl_round_trips_request_records(self):
        import json
        from pathlib import Path
        import tempfile

        record = RequestRecord("users-2k", "distributed", 2, 2048, 1, 1, 0,
                               0.1, 1.0, 128, 128.0, 200, "")
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "requests.jsonl"
            write_request_jsonl([record], path)
            parsed = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(parsed["users"], 2)
        self.assertEqual(parsed["tok_s"], 128.0)

    def test_csv_has_header_and_saturation_flag(self):
        import csv
        from pathlib import Path
        import tempfile

        summary = PointSummary("users-2k", "distributed", 2, 2048, 1,
                               2, 0, 256, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0,
                               100.0, 2, True)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "summary.csv"
            write_summary_csv([summary], path)
            rows = list(csv.DictReader(path.open(encoding="utf-8")))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["profile"], "users-2k")
        self.assertEqual(rows[0]["saturated"], "True")


class SseClientTest(unittest.TestCase):
    def test_http_client_counts_sse_events_and_ttft(self):
        import io
        import json
        from pathlib import Path

        events = [
            b"data: " + json.dumps({"content": "a"}).encode() + b"\n\n",
            b"data: " + json.dumps({"content": "b"}).encode() + b"\n\n",
            b"data: " + json.dumps({"stop": True}).encode() + b"\n\n",
        ]

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        def opener(_request: object) -> FakeResponse:
            return FakeResponse(b"".join(events))

        times = iter([0.0, 0.3, 0.5])
        client = HttpCompletionClient("http://x:8080", opener=opener,
                                      clock=lambda: next(times))

        result = client.generate("p", 2)

        self.assertTrue(result.ok)
        self.assertEqual(result.tokens, 2)
        self.assertAlmostEqual(result.ttft_s, 0.3, places=6)
        self.assertAlmostEqual(result.e2e_s, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
