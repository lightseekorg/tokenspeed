import json
import os
import sys
import tempfile
import unittest

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer  # noqa: E402


class TestHfTransformersUtils(unittest.TestCase):
    def _write_deepseek_v3_like_tokenizer(self, model_dir: str) -> list[int]:
        vocab = {
            "MLA": 0,
            "Ġattention": 1,
            ",": 2,
            "Ġor": 3,
            "Ġmore": 4,
            "Ġspecifically": 5,
            ".ĊĊ": 6,
            "<｜begin▁of▁sentence｜>": 7,
            "<｜end▁of▁sentence｜>": 8,
        }
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.save(os.path.join(model_dir, "tokenizer.json"))

        tokenizer_config = {
            "tokenizer_class": "LlamaTokenizerFast",
            "legacy": True,
            "clean_up_tokenization_spaces": False,
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>",
            "unk_token": None,
        }
        with open(os.path.join(model_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f)

        return [0, 1, 2, 3, 4, 5, 6]

    def test_deepseek_v3_uses_verbatim_fast_tokenizer(self):
        with tempfile.TemporaryDirectory() as model_dir:
            token_ids = self._write_deepseek_v3_like_tokenizer(model_dir)

            for architecture in (
                "DeepseekV3ForCausalLM",
                "DeepseekV3ForCausalLMNextN",
            ):
                with self.subTest(architecture=architecture):
                    tokenizer = get_tokenizer(
                        model_dir,
                        trust_remote_code=True,
                        architectures=[architecture],
                    )

                    self.assertEqual(
                        tokenizer.decode(token_ids),
                        "MLA attention, or more specifically.\n\n",
                    )


if __name__ == "__main__":
    unittest.main()
