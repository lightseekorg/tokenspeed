"""Inkling multimodal tests: tower units + engine-only e2e with synthetic features.

The SMG gateway normally does all media preprocessing; the engine only embeds
already-extracted features at pre-expanded placeholder offsets. These tests
exercise exactly that engine surface:

* Unit: InklingAudioTower / InklingHMLPPatchEncoder parity against hand-rolled torch
  references on shared weights, plan_out_scales schedules, MM config flags,
  and the M-RoPE no-op gate.
* E2E: an in-process Engine on the tiny tower-enabled dummy checkpoint, fed
  ``GenerateReqInput(input_ids=..., precomputed_multimodal_inputs=
  MultimodalInputs(...))`` through ``engine.llm.generate`` — the same
  low-level API the SMG gRPC servicer uses (``_build_generate_req``).

NOTE: intentionally NOT registered in CI suites while the Inkling port is
confidential/local-only. The e2e needs a Blackwell GPU (FA4); unit tests need
any CUDA GPU (runtime RMSNorm kernels are GPU-only).

Run:
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=python:tokenspeed-kernel/python \
  python3 -m pytest test/runtime/models/test_inkling_multimodal.py -q
"""

import os
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

# Add project root directory to path for importing test.* helpers.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
from test.runtime.models.inkling_fixtures import (  # noqa: E402
    TINY_AUDIO_PLACEHOLDER_TOKEN_ID,
    TINY_AUDIO_TOWER_CONFIG,
    TINY_IMAGE_PLACEHOLDER_TOKEN_ID,
    TINY_MM_TOWERS_CONFIG,
    TINY_VISION_TOWER_CONFIG,
    make_inkling_dummy_checkpoint,
)


def _has_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


def _naive_hmlp_forward(encoder, x: torch.Tensor) -> torch.Tensor:
    """Independent torch reimplementation of the hMLP forward on the same
    weights (explicit reshape/permute fold, F.rms_norm, F.gelu)."""

    def fold(x, t_fold, hw_fold):
        B, T, H, W, C = x.shape
        x = x.reshape(
            B, T // t_fold, t_fold, H // hw_fold, hw_fold, W // hw_fold, hw_fold, C
        )
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        return x.reshape(
            B, T // t_fold, H // hw_fold, W // hw_fold, t_fold * hw_fold * hw_fold * C
        )

    num_patches = x.shape[0]
    for i, (start, end) in enumerate(zip(encoder.scales[:-1], encoder.scales[1:])):
        t_fold, hw_fold = end[0] // start[0], end[1] // start[1]
        if t_fold > 1 or hw_fold > 1:
            x = fold(x, t_fold, hw_fold)
        x = x @ encoder.layers[f"linear_{i}"].weight.t()
        if i < encoder.n_layers - 1:
            norm = encoder.layers[f"norm_{i}"]
            x = F.rms_norm(x, (x.shape[-1],), norm.weight, norm.variance_epsilon)
            x = F.gelu(x)
    if encoder.final_norm is not None:
        x = F.rms_norm(
            x,
            (x.shape[-1],),
            encoder.final_norm.weight,
            encoder.final_norm.variance_epsilon,
        )
    return x.reshape(num_patches, -1)


class TestInklingPlanOutScales(unittest.TestCase):
    def test_tiny_fixture_schedule(self):
        from tokenspeed.runtime.models.inkling import inkling_plan_out_scales

        self.assertEqual(
            inkling_plan_out_scales(1, 4, 1, 3), [(1, 1, 1, 3), (1, 4, 4, 64)]
        )

    def test_real_config_schedule(self):
        # Cross-checked against the reference scipy implementation.
        from tokenspeed.runtime.models.inkling import inkling_plan_out_scales

        self.assertEqual(
            inkling_plan_out_scales(2, 40, 4, 3),
            [
                (1, 1, 1, 3),
                (1, 5, 5, 128),
                (1, 10, 10, 320),
                (1, 40, 40, 4800),
                (2, 40, 40, 9600),
            ],
        )


class TestInklingMultimodalConfig(unittest.TestCase):
    def test_arch_flags(self):
        from tokenspeed.runtime.configs.model_config import (
            is_audio_model,
            is_multimodal_model,
        )

        arch = ["InklingForConditionalGeneration"]
        self.assertTrue(is_multimodal_model(arch))
        self.assertTrue(is_audio_model(arch))
        self.assertFalse(is_audio_model(["Qwen3_5ForConditionalGeneration"]))

    def test_placeholder_token_ids(self):
        from tokenspeed.runtime.configs.inkling_config import (
            INKLING_AUDIO_PLACEHOLDER_TOKEN_ID,
            INKLING_IMAGE_PLACEHOLDER_TOKEN_ID,
            INKLING_MODEL_END_SAMPLING_TOKEN_ID,
            InklingMMConfig,
        )

        cfg = InklingMMConfig(**TINY_MM_TOWERS_CONFIG)
        self.assertEqual(
            cfg.image_placeholder_token_id, TINY_IMAGE_PLACEHOLDER_TOKEN_ID
        )
        self.assertEqual(
            cfg.audio_placeholder_token_id, TINY_AUDIO_PLACEHOLDER_TOKEN_ID
        )
        # Explicit checkpoint EOS remains authoritative for tiny fixtures.
        self.assertEqual(cfg.eos_token_id, 1)
        self.assertEqual(cfg.text_config.eos_token_id, 1)
        # Checkpoint soft-placeholder IDs also give the text-only MTP draft
        # safe, modality-specific embeddings for media feature positions.
        default = InklingMMConfig()
        self.assertEqual(
            default.image_placeholder_token_id,
            INKLING_IMAGE_PLACEHOLDER_TOKEN_ID,
        )
        self.assertEqual(
            default.audio_placeholder_token_id,
            INKLING_AUDIO_PLACEHOLDER_TOKEN_ID,
        )
        self.assertEqual(default.eos_token_id, INKLING_MODEL_END_SAMPLING_TOKEN_ID)
        self.assertEqual(
            default.text_config.eos_token_id,
            INKLING_MODEL_END_SAMPLING_TOKEN_ID,
        )
        self.assertIsNone(default.audio_config.decoder_dmodel)
        self.assertIsNone(default.vision_config.decoder_dmodel)
        self.assertEqual(default.audio_config.dmel_min_value, -7.0)
        self.assertEqual(default.audio_config.audio_rms_norm_floor, 0.01)
        # Transport placeholder IDs must remain valid unsigned token IDs.
        with self.assertRaises(ValueError):
            InklingMMConfig(image_placeholder_token_id=-1)

    def test_mrope_is_noop_for_tml(self):
        """The precomputed-MM input path calls compute_mrope_positions
        unconditionally; Inkling is not an M-RoPE architecture, so it must
        no-op (input_processor then skips all mrope_* fields)."""
        from tokenspeed.runtime.configs.inkling_config import InklingMMConfig
        from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
        from tokenspeed.runtime.multimodal.mrope import compute_mrope_positions

        cfg = InklingMMConfig(**TINY_MM_TOWERS_CONFIG)
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.zeros(2, 1, 4, 4, 3),
            offsets=[(1, 2)],
        )
        positions, delta = compute_mrope_positions(
            cfg, [7, TINY_IMAGE_PLACEHOLDER_TOKEN_ID] * 2, [item]
        )
        self.assertIsNone(positions)
        self.assertIsNone(delta)


@unittest.skipUnless(torch.cuda.is_available(), "runtime RMSNorm kernels need CUDA")
class TestInklingAudioTower(unittest.TestCase):
    def test_matches_reference(self):
        from tokenspeed.runtime.configs.inkling_config import InklingAudioConfig
        from tokenspeed.runtime.models.inkling import InklingAudioTower

        torch.manual_seed(0)
        cfg = InklingAudioConfig(**TINY_AUDIO_TOWER_CONFIG)
        tower = InklingAudioTower(cfg).cuda()
        n_bins, vocab = cfg.n_mel_bins, cfg.mel_vocab_size

        dmel = torch.randint(0, vocab, (13, n_bins), device="cuda")
        out = tower(dmel)
        self.assertEqual(out.shape, (13, cfg.decoder_dmodel))

        # Hand-rolled reference: per-bin offset indexing + sum over bins.
        emb = tower.encoder.weight
        ref = torch.stack(
            [
                sum(emb[b * vocab + int(dmel[t, b])] for b in range(n_bins))
                for t in range(dmel.shape[0])
            ]
        )
        ref = F.rms_norm(ref, (cfg.decoder_dmodel,), tower.final_norm.weight, 1e-6)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_no_norm_variant(self):
        from tokenspeed.runtime.configs.inkling_config import InklingAudioConfig
        from tokenspeed.runtime.models.inkling import InklingAudioTower

        torch.manual_seed(1)
        cfg = InklingAudioConfig(**{**TINY_AUDIO_TOWER_CONFIG, "use_audio_norm": False})
        tower = InklingAudioTower(cfg).cuda()
        self.assertIsNone(tower.final_norm)
        dmel = torch.randint(0, cfg.mel_vocab_size, (3, cfg.n_mel_bins), device="cuda")
        emb = tower.encoder.weight
        ref = torch.stack(
            [
                sum(
                    emb[b * cfg.mel_vocab_size + int(dmel[t, b])]
                    for b in range(cfg.n_mel_bins)
                )
                for t in range(3)
            ]
        )
        torch.testing.assert_close(tower(dmel), ref, atol=1e-5, rtol=1e-5)

    def test_peak_memory_does_not_scale_with_n_mel_bins(self):
        """dMel embedding must fuse lookup+sum.

        The unfused ``encoder(idx).reshape(...).sum(1)`` form materializes an
        intermediate ``n_mel_bins`` times the size of the output (~0.95 MB per
        audio token at the released ``decoder_dmodel=6144``), which lets a
        single long clip OOM the engine. Assert the peak stays within a small
        multiple of the output instead.
        """
        from tokenspeed.runtime.configs.inkling_config import InklingAudioConfig
        from tokenspeed.runtime.models.inkling import InklingAudioTower

        torch.manual_seed(2)
        cfg = InklingAudioConfig(
            decoder_dmodel=512,
            n_mel_bins=80,
            mel_vocab_size=16,
            use_audio_norm=True,
            audio_mode="dmel",
        )
        tower = InklingAudioTower(cfg).cuda().eval()
        n_tokens = 4096
        dmel = torch.randint(
            0, cfg.mel_vocab_size, (n_tokens, cfg.n_mel_bins), device="cuda"
        )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            out = tower(dmel)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base

        out_bytes = out.numel() * out.element_size()
        unfused_bytes = out_bytes * cfg.n_mel_bins
        # Generous ceiling: comfortably above the fused path, far below unfused.
        self.assertLess(
            peak,
            out_bytes * 8,
            f"peak {peak / 2**20:.1f} MB suggests the unfused lookup+sum is "
            f"back (unfused would be ~{unfused_bytes / 2**20:.1f} MB, output is "
            f"{out_bytes / 2**20:.1f} MB)",
        )


@unittest.skipUnless(torch.cuda.is_available(), "runtime RMSNorm kernels need CUDA")
class TestInklingHMLPPatchEncoder(unittest.TestCase):
    def test_tiny_shape_and_parity(self):
        from tokenspeed.runtime.configs.inkling_config import InklingVisionConfig
        from tokenspeed.runtime.models.inkling import InklingHMLPPatchEncoder

        torch.manual_seed(0)
        cfg = InklingVisionConfig(**TINY_VISION_TOWER_CONFIG)
        enc = InklingHMLPPatchEncoder(cfg).cuda()
        x = torch.randn(
            9,
            cfg.temporal_patch_size,
            cfg.patch_size,
            cfg.patch_size,
            cfg.n_channels,
            device="cuda",
        )
        out = enc(x)
        self.assertEqual(out.shape, (9, cfg.decoder_dmodel))
        torch.testing.assert_close(
            out, _naive_hmlp_forward(enc, x), atol=1e-4, rtol=1e-4
        )

    def test_real_size_multilayer_parity(self):
        """Real checkpoint geometry: 4 layers, patch 40, temporal 2 — covers
        the RMSNorm+GELU inner layers and the temporal fold."""
        from tokenspeed.runtime.configs.inkling_config import InklingVisionConfig
        from tokenspeed.runtime.models.inkling import InklingHMLPPatchEncoder

        torch.manual_seed(0)
        cfg = InklingVisionConfig(
            vision_encoder_type="hmlp",
            decoder_dmodel=512,
            patch_size=40,
            temporal_patch_size=2,
            n_channels=3,
            n_layers=4,
            use_vision_norm=True,
        )
        enc = InklingHMLPPatchEncoder(cfg).cuda()
        x = torch.randn(3, 2, 40, 40, 3, device="cuda")
        out = enc(x)
        self.assertEqual(out.shape, (3, 512))
        torch.testing.assert_close(
            out, _naive_hmlp_forward(enc, x), atol=1e-3, rtol=1e-3
        )


@unittest.skipUnless(_has_blackwell(), "Inkling e2e needs a Blackwell GPU (FA4)")
class TestInklingMultimodalE2E(unittest.TestCase):
    """Engine-only e2e with synthetic features (no gateway).

    Mirrors the SMG servicer's precomputed-multimodal entry point:
    ``GenerateReqInput(input_ids=..., precomputed_multimodal_inputs=...)``
    submitted through the ``LLM`` facade over ``AsyncLLM.generate_request``.
    """

    IMAGE_TOKENS = 8  # one token per patch (hMLP folds patch interior)
    AUDIO_TOKENS = 6

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        ckpt = make_inkling_dummy_checkpoint(
            cls._tmpdir.name, tiny=True, mm_towers=True
        )

        from tokenspeed.runtime.entrypoints.engine import Engine

        cls.engine = Engine(
            model=str(ckpt),
            load_format="dummy",
            attention_backend="fa4",
            enable_prefix_caching=False,
            disable_kvstore=True,
            enforce_eager=False,
            dtype="bfloat16",
            gpu_memory_utilization=0.3,
            max_model_len=2048,
            max_num_seqs=8,
            enable_output_logprobs=True,
            log_level="warning",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        cls._tmpdir.cleanup()

    # -- helpers ----------------------------------------------------------

    # The placeholder run ends the prompt on purpose: the first sampled
    # token's logits come from the LAST prompt position's hidden state, so a
    # trailing mm position makes the spliced embedding directly observable.
    # With uniform(+-1e-3) dummy weights, information does NOT survive
    # cross-position propagation — attention residuals (~1e-7) round away
    # against bf16 hidden states (~1e-3) — so an interior run followed by
    # text suffix yields byte-identical logits with real vs. placeholder
    # embeddings. (Not an issue with real weights.)

    @staticmethod
    def _image_prompt(num_tokens: int) -> tuple[list[int], int]:
        """Pre-expanded prompt: text then a trailing placeholder run.
        Returns (input_ids, run_start)."""
        prefix = [11, 45, 260, 132]
        ids = prefix + [TINY_IMAGE_PLACEHOLDER_TOKEN_ID] * num_tokens
        return ids, len(prefix)

    @staticmethod
    def _audio_prompt(num_tokens: int) -> tuple[list[int], int]:
        prefix = [21, 33, 407]
        ids = prefix + [TINY_AUDIO_PLACEHOLDER_TOKEN_ID] * num_tokens
        return ids, len(prefix)

    def _generate(self, input_ids, mm_items=None, max_new_tokens=8):
        """Submit one request the way the SMG servicer does and return
        ``(output_ids, output_logprobs)``."""
        from tokenspeed.runtime.engine.io_struct import GenerateReqInput
        from tokenspeed.runtime.multimodal.inputs import MultimodalInputs

        precomputed = (
            MultimodalInputs(mm_items=mm_items) if mm_items is not None else None
        )
        obj = GenerateReqInput(
            input_ids=list(input_ids),
            sampling_params={"temperature": 0.0, "max_new_tokens": max_new_tokens},
            return_logprob=True,
            precomputed_multimodal_inputs=precomputed,
        )
        out = self.engine.llm.generate(obj)
        meta = out["meta_info"]
        self.assertEqual(meta["completion_tokens"], max_new_tokens)
        logprobs = [pair[0] for pair in meta["output_token_logprobs"]]
        output_ids = [pair[1] for pair in meta["output_token_logprobs"]]
        for token_id in output_ids:
            self.assertLess(token_id, 2000)  # padded-vocab mask still active
        return output_ids, logprobs

    @staticmethod
    def _image_item(seed: int, num_tokens: int, run_start: int):
        from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

        g = torch.Generator().manual_seed(seed)
        feature = torch.randn(num_tokens, 1, 4, 4, 3, generator=g)
        return MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=[(run_start, run_start + num_tokens - 1)],
        )

    @staticmethod
    def _audio_item(seed: int, num_tokens: int, run_start: int):
        from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

        g = torch.Generator().manual_seed(seed)
        feature = torch.randint(0, 4, (num_tokens, 8), generator=g)
        return MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=feature,
            offsets=[(run_start, run_start + num_tokens - 1)],
        )

    # -- tests --------------------------------------------------------------

    def test_image_feature_replaces_placeholder_embeddings(self):
        ids, start = self._image_prompt(self.IMAGE_TOKENS)

        base_ids, base_lp = self._generate(ids)  # no mm: placeholders as text
        a_ids, a_lp = self._generate(
            ids, [self._image_item(1, self.IMAGE_TOKENS, start)]
        )
        a2_ids, a2_lp = self._generate(
            ids, [self._image_item(1, self.IMAGE_TOKENS, start)]
        )
        b_ids, b_lp = self._generate(
            ids, [self._image_item(2, self.IMAGE_TOKENS, start)]
        )

        # Same feature => deterministic.
        self.assertEqual(a_ids, a2_ids)
        self.assertEqual(a_lp, a2_lp)
        # The mm feature must actually reach the LM: with vs without, and
        # feature A vs feature B, first-token logits (hence logprobs or the
        # greedy pick) must differ.
        self.assertTrue(
            a_ids != base_ids or a_lp != base_lp,
            "image feature did not change the model output",
        )
        self.assertTrue(
            a_ids != b_ids or a_lp != b_lp,
            "different image features produced identical outputs",
        )

    def test_audio_feature_replaces_placeholder_embeddings(self):
        ids, start = self._audio_prompt(self.AUDIO_TOKENS)

        base_ids, base_lp = self._generate(ids)
        a_ids, a_lp = self._generate(
            ids, [self._audio_item(3, self.AUDIO_TOKENS, start)]
        )
        a2_ids, a2_lp = self._generate(
            ids, [self._audio_item(3, self.AUDIO_TOKENS, start)]
        )
        b_ids, b_lp = self._generate(
            ids, [self._audio_item(4, self.AUDIO_TOKENS, start)]
        )

        self.assertEqual(a_ids, a2_ids)
        self.assertEqual(a_lp, a2_lp)
        self.assertTrue(
            a_ids != base_ids or a_lp != base_lp,
            "audio feature did not change the model output",
        )
        self.assertTrue(
            a_ids != b_ids or a_lp != b_lp,
            "different audio features produced identical outputs",
        )

    def test_mixed_image_and_audio_request(self):
        """Interior image run + trailing audio run in one request: both
        encoders execute in the same forward; the trailing audio feature
        must be observable in the first-token logits."""
        prefix = [5, 9]
        mid = [17]
        img_start = len(prefix)
        aud_start = img_start + self.IMAGE_TOKENS + len(mid)
        ids = (
            prefix
            + [TINY_IMAGE_PLACEHOLDER_TOKEN_ID] * self.IMAGE_TOKENS
            + mid
            + [TINY_AUDIO_PLACEHOLDER_TOKEN_ID] * self.AUDIO_TOKENS
        )
        items = [
            self._image_item(5, self.IMAGE_TOKENS, img_start),
            self._audio_item(6, self.AUDIO_TOKENS, aud_start),
        ]
        a_ids, a_lp = self._generate(ids, items)
        self.assertEqual(len(a_ids), 8)
        items_b = [
            self._image_item(5, self.IMAGE_TOKENS, img_start),
            self._audio_item(7, self.AUDIO_TOKENS, aud_start),
        ]
        b_ids, b_lp = self._generate(ids, items_b)
        self.assertTrue(
            a_ids != b_ids or a_lp != b_lp,
            "changing the trailing audio feature did not change the output",
        )

    def test_text_only_prompt_still_works(self):
        # Plain text through the same engine (no precomputed mm): the
        # text-only fast path must keep working alongside mm requests.
        out = self.engine.generate(
            prompt="the quick brown fox",
            sampling_params={"temperature": 0.0, "max_new_tokens": 8},
        )
        result = out if isinstance(out, dict) else out[0]
        self.assertEqual(result["meta_info"]["completion_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
