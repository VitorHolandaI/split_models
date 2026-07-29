import unittest

from split_lm.node.profiling import _profile_lines


class ProfilingFormatTest(unittest.TestCase):
    def test_profile_uses_aligned_millisecond_lines(self):
        lines = _profile_lines(
            "head", {"total": 2.0, "tokens": 4, "forward": 1.25}, color=False)
        report = "\n".join(lines)

        self.assertIn("total_latency           2000.0 ms", report)
        self.assertIn("average_per_token       500.0 ms", report)
        self.assertIn("forward                 1250.0 ms", report)
        self.assertGreaterEqual(len(lines), 7)

    def test_profile_can_color_values(self):
        lines = _profile_lines("head", {"total": 1.0, "tokens": 2}, color=True)

        self.assertIn("\033[1;36m", lines[0])
        self.assertTrue(any("\033[1;32m2.00 tok/s\033[0m" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
