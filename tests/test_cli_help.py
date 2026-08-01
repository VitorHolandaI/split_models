import contextlib
import io
import unittest

from split_lm.cli import _run_cli


class SplitCliHelpTest(unittest.TestCase):
    def test_low_level_help_explains_layer_ranges_and_operator_entrypoint(self):
        output = io.StringIO()

        with self.assertRaises(SystemExit), contextlib.redirect_stdout(output):
            _run_cli(["--help"])

        help_text = output.getvalue()
        self.assertIn("intervalo global", help_text)
        self.assertIn("repositorio Hugging Face", help_text)
        self.assertIn("split-cluster exemplos", help_text)


if __name__ == "__main__":
    unittest.main()
