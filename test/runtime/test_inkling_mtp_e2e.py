"""Inkling MTP speculative-decoding dummy-weight e2e.

Launches the in-process Engine on the same tiny synthetic Inkling
checkpoint (with ``mtp_config``) and dummy weights — once plain, once with
``--speculative-algorithm MTP`` (CUDA graphs AND eager) — and asserts
greedy outputs are IDENTICAL: speculative decoding must be lossless
regardless of draft quality (the dummy draft proposes junk; the target
accepts/rejects exactly).

Byte-exact parity is asserted at TP1 ONLY. At TP>1 the per-rank sharded
GEMM shapes differ between the k-token verify pass and 1-token decode, so
BF16 rounding flips greedy argmaxes on the tiny config's near-flat dummy
logits (verified empirically: the divergence appears identically in eager
and graph MTP, persists with sconv disabled entirely, and the plain
baseline itself flips under batch-shape changes on the real checkpoint).
With INKLING_E2E_TP>1 the test only sanity-checks completion.

Exercises the full MTP path: draft-worker config rename, NextN model
construction, draft KV + conv pools, target-verify conv stash/rollback,
draft catch-up valid-length conv update, and multi-step drafting.

NOTE: intentionally NOT registered in CI while the Inkling port is
confidential/local-only. Requires a Blackwell GPU (FA4 score_mod).

Run:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=python:tokenspeed-kernel/python \
  python3 -m unittest runtime.test_inkling_mtp_e2e -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.models.inkling_fixtures import (  # noqa: E402
    TINY_AUDIO_PLACEHOLDER_TOKEN_ID,
    TINY_IMAGE_PLACEHOLDER_TOKEN_ID,
    make_inkling_dummy_checkpoint,
)

PROMPTS = [
    "The quick brown fox",
    "counting " + " ".join(str(i) for i in range(80)),
    "hi",
]
MAX_NEW_TOKENS = 32
NUM_DEPTHS = 2


def _has_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _make_mtp_ckpt(tmpdir: str) -> Path:
    ckpt = make_inkling_dummy_checkpoint(tmpdir, tiny=True, mm_towers=True)
    cfg = json.loads((ckpt / "config.json").read_text())
    cfg["mtp_config"] = {
        "num_nextn_predict_layers": NUM_DEPTHS,
        "chain_hidden_post_norm": True,
    }
    (ckpt / "config.json").write_text(json.dumps(cfg, indent=2))
    return ckpt


def _make_engine(ckpt: Path, spec: bool, enforce_eager: bool = False):
    from tokenspeed.runtime.entrypoints.engine import Engine

    extra = {}
    tp = int(os.environ.get("INKLING_E2E_TP", "1"))
    if tp > 1:
        extra["attn_tp_size"] = tp
    if spec:
        extra.update(
            speculative_algorithm="MTP",
            speculative_num_steps=NUM_DEPTHS,
            speculative_num_draft_tokens=NUM_DEPTHS + 1,
            speculative_eagle_topk=1,
        )
    return Engine(
        model=str(ckpt),
        load_format="dummy",
        attention_backend="fa4",
        enable_prefix_caching=False,
        disable_kvstore=True,
        enforce_eager=enforce_eager,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=2048,
        max_num_seqs=8,
        log_level=os.environ.get("INKLING_E2E_LOG", "warning"),
        **extra,
    )


def _generate(ckpt: Path, spec: bool, enforce_eager: bool = False):
    engine = _make_engine(ckpt, spec, enforce_eager)
    try:
        out = engine.generate(
            prompt=PROMPTS,
            sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS},
        )
        return [(o["output_ids"], o["meta_info"]) for o in out]
    finally:
        engine.shutdown()


def _multimodal_requests():
    from tokenspeed.runtime.engine.io_struct import GenerateReqInput
    from tokenspeed.runtime.multimodal.inputs import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )

    def request(input_ids, items):
        return GenerateReqInput(
            input_ids=input_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": 8},
            precomputed_multimodal_inputs=MultimodalInputs(mm_items=items),
        )

    image_feature = torch.arange(2 * 1 * 4 * 4 * 3, dtype=torch.float32).reshape(
        2, 1, 4, 4, 3
    )
    audio_feature = torch.arange(2 * 8, dtype=torch.int64).reshape(2, 8) % 4

    image = request(
        [11, 12, TINY_IMAGE_PLACEHOLDER_TOKEN_ID, TINY_IMAGE_PLACEHOLDER_TOKEN_ID],
        [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=image_feature.clone(),
                offsets=[(2, 3)],
            )
        ],
    )
    audio = request(
        [21, TINY_AUDIO_PLACEHOLDER_TOKEN_ID, TINY_AUDIO_PLACEHOLDER_TOKEN_ID],
        [
            MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=audio_feature.clone(),
                offsets=[(1, 2)],
            )
        ],
    )
    mixed = request(
        [
            31,
            TINY_IMAGE_PLACEHOLDER_TOKEN_ID,
            TINY_IMAGE_PLACEHOLDER_TOKEN_ID,
            32,
            TINY_AUDIO_PLACEHOLDER_TOKEN_ID,
            TINY_AUDIO_PLACEHOLDER_TOKEN_ID,
        ],
        [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=image_feature.clone(),
                offsets=[(1, 2)],
            ),
            MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=audio_feature.clone(),
                offsets=[(4, 5)],
            ),
        ],
    )
    return [image, audio, mixed]


def _generate_multimodal(ckpt: Path, spec: bool):
    engine = _make_engine(ckpt, spec)
    try:
        return [
            list(engine.llm.generate(req)["output_ids"])
            for req in _multimodal_requests()
        ]
    finally:
        engine.shutdown()


@unittest.skipUnless(_has_blackwell(), "Inkling e2e needs a Blackwell GPU (FA4)")
class TestInklingMTPDummyE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.ckpt = _make_mtp_ckpt(cls._tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_mtp_greedy_matches_baseline(self):
        strict = int(os.environ.get("INKLING_E2E_TP", "1")) == 1
        baseline = _generate(self.ckpt, spec=False)
        # MTP under CUDA graphs (the production configuration) and eager
        # (the fallback / debug path) must BOTH match the baseline exactly
        # at TP1; at TP>1 kernel-shape numerics make byte parity
        # unattainable on flat dummy logits (see module docstring).
        for label, eager in (("graphs", False), ("eager", True)):
            mtp = _generate(self.ckpt, spec=True, enforce_eager=eager)
            self.assertEqual(len(baseline), len(mtp))
            for i, ((base_ids, _), (mtp_ids, mtp_meta)) in enumerate(
                zip(baseline, mtp)
            ):
                if strict:
                    self.assertEqual(
                        list(base_ids),
                        list(mtp_ids),
                        f"prompt {i} [{label}]: MTP output diverges from "
                        f"baseline (speculative decoding must be lossless)",
                    )
                self.assertEqual(mtp_meta["completion_tokens"], MAX_NEW_TOKENS)

    def test_mtp_multimodal_image_audio_and_mixed_match_baseline(self):
        baseline = _generate_multimodal(self.ckpt, spec=False)
        mtp = _generate_multimodal(self.ckpt, spec=True)
        self.assertEqual(
            baseline,
            mtp,
            "MTP must remain lossless for image, audio, and mixed prompts",
        )


if __name__ == "__main__":
    unittest.main()
