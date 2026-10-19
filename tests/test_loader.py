import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from split_lm.model.loader import (
    _drop_unused_modules, _ensure_precision_available, _model_init_kwargs,
    _quantize_model, _wanted_weight)


class LoaderTest(unittest.TestCase):
    def test_model_initialization_preserves_checkpoint_dtype(self):
        config = SimpleNamespace(dtype=torch.bfloat16)

        self.assertEqual(_model_init_kwargs(config, "auto"), {"dtype": torch.bfloat16})

    def test_model_initialization_uses_framework_default_without_dtype(self):
        self.assertEqual(_model_init_kwargs(SimpleNamespace(), "auto"), {})

    def test_explicit_precision_overrides_checkpoint_dtype(self):
        config = SimpleNamespace(dtype=torch.bfloat16)

        self.assertEqual(_model_init_kwargs(config, "float32"), {"dtype": torch.float32})

    def test_bfloat16_does_not_require_quantization_backend(self):
        model = object()

        self.assertIs(_quantize_model(model, "bfloat16", "cpu"), model)

    def test_quantization_rejects_non_cpu_device_before_import(self):
        with self.assertRaisesRegex(ValueError, "device='cpu'"):
            _quantize_model(object(), "int4", "cuda")

    def test_quantization_extra_is_checked_before_model_load(self):
        with patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "uv sync --extra quant"):
                _ensure_precision_available("int8")

    def test_tail_loads_real_lm_head_when_embeddings_are_not_tied(self):
        config = SimpleNamespace(tie_word_embeddings=False)

        self.assertTrue(_wanted_weight("lm_head.weight", 18, 28, 28, config))
        self.assertFalse(
            _wanted_weight("model.embed_tokens.weight", 18, 28, 28, config)
        )

    def test_unused_modules_are_removed_from_intermediate_slice(self):
        model = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=object(), norm=object()),
            lm_head=object(),
        )

        _drop_unused_modules(model, is_head=False, is_last=False)

        self.assertIsNone(model.model.embed_tokens)
        self.assertIsNone(model.model.norm)
        self.assertIsNone(model.lm_head)


if __name__ == "__main__":
    unittest.main()
