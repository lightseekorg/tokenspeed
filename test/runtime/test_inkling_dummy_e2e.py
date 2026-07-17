"""Inkling dummy-weight e2e smoke test (text-only v1).

Launches the in-process Engine on a tiny synthetic Inkling checkpoint with
``--load-format dummy`` and generates from prompts that cross the SWA window
and conv-state boundaries. Validates the full serving path: FA4 relative-bias
score_mod attention, engine-side sconv state pool, sigmoid/logsigmoid MoE
gate, dense+MoE layers, batching and request lifecycle.

NOTE: intentionally NOT registered in CI suites while the Inkling port is
confidential/local-only. Requires a Blackwell GPU (FA4 score_mod).

Run:
  CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH=python:tokenspeed-kernel/python \
  python3 -m pytest test/runtime/test_inkling_dummy_e2e.py -q
"""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.models.inkling_fixtures import make_inkling_dummy_checkpoint  # noqa: E402


def _has_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


@unittest.skipUnless(_has_blackwell(), "Inkling e2e needs a Blackwell GPU (FA4)")
class TestInklingDummyE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Runs the real MoELayer path (flashinfer cutlass_unquant,
        # precomputed topk). Set INKLING_TORCH_MOE=1 to fall back to torch-native
        # experts, e.g. when the flashinfer JIT cache is cold or broken.
        # INKLING_E2E_TP=N runs the same test tensor-parallel over N GPUs
        # (attention TP; MoE layout resolves from world size), and
        # INKLING_E2E_EP=N adds expert parallelism.
        cls._tmpdir = tempfile.TemporaryDirectory()
        ckpt = make_inkling_dummy_checkpoint(cls._tmpdir.name, tiny=True)

        from tokenspeed.runtime.entrypoints.engine import Engine

        extra = {}
        tp = int(os.environ.get("INKLING_E2E_TP", "1"))
        if tp > 1:
            extra["attn_tp_size"] = tp
        ep = int(os.environ.get("INKLING_E2E_EP", "1"))
        if ep > 1:
            extra["ep_size"] = ep
        cls.engine = Engine(
            model=str(ckpt),
            load_format="dummy",
            attention_backend="fa4",
            enable_prefix_caching=False,
            disable_kvstore=True,
            # CUDA graphs on: covers score_mod decode capture, sconv_decode
            # under graph replay, and the InklingAttnBackend capture hooks.
            enforce_eager=False,
            dtype="bfloat16",
            gpu_memory_utilization=0.3,
            max_model_len=2048,
            max_num_seqs=8,
            log_level=os.environ.get("INKLING_E2E_LOG", "warning"),
            **extra,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        cls._tmpdir.cleanup()

    def test_batch_generation(self):
        sampling_params = {"temperature": 0.0, "max_new_tokens": 16}
        prompts = [
            "The quick brown fox",
            # Crosses the tiny config's SWA window (32) and conv boundaries.
            "counting " + " ".join(str(i) for i in range(120)),
            "hi",
        ]
        out = self.engine.generate(prompt=prompts, sampling_params=sampling_params)
        self.assertEqual(len(out), len(prompts))
        for o in out:
            meta = o["meta_info"]
            self.assertEqual(meta["completion_tokens"], 16)
            self.assertEqual(meta["finish_reason"]["type"], "length")
            self.assertIsInstance(o["text"], str)
            # Padded-vocab logits are masked in the model: sampled ids must
            # all be real (< unpadded_vocab_size) tokens.
            for token_id in o.get("output_ids") or []:
                self.assertLess(token_id, 2000)
        # The long prompt tokenized past the SWA window.
        self.assertGreater(out[1]["meta_info"]["prompt_tokens"], 64)

    def test_second_batch_reuses_slots(self):
        sampling_params = {"temperature": 0.0, "max_new_tokens": 8}
        out = self.engine.generate(
            prompt="another round", sampling_params=sampling_params
        )
        result = out if isinstance(out, dict) else out[0]
        self.assertEqual(result["meta_info"]["completion_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
