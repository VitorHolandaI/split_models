import unittest
from types import SimpleNamespace

from split_lm.model.loader import _drop_unused_modules, _wanted_weight


class LoaderTest(unittest.TestCase):
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
