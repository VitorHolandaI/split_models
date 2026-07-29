import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from split_lm.model.runner import _final_logits
from split_lm.node.head import _read_prompt


class RecordingLmHead:
    def __init__(self):
        self.input_shape: tuple[int, ...] | None = None

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        self.input_shape = tuple(hidden.shape)
        return torch.zeros((*hidden.shape[:-1], 10))


class RunnerTest(unittest.TestCase):
    def test_final_projection_only_receives_last_sequence_position(self):
        lm_head = RecordingLmHead()
        model = SimpleNamespace(
            model=SimpleNamespace(norm=lambda hidden: hidden), lm_head=lm_head)

        logits = _final_logits(model, torch.zeros((1, 2048, 8)))

        self.assertEqual(lm_head.input_shape, (1, 1, 8))
        self.assertEqual(tuple(logits.shape), (1, 1, 10))

    def test_eof_returns_no_prompt_instead_of_crashing_chat(self):
        with patch("builtins.input", side_effect=EOFError):
            self.assertIsNone(_read_prompt())


if __name__ == "__main__":
    unittest.main()
