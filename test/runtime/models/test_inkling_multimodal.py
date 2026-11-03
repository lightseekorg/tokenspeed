"""Inkling multimodal tests: tower units + engine-only e2e with synthetic features.

The SMG gateway normally does all media preprocessing; the engine only embeds
already-extracted features at pre-expanded placeholder offsets. These tests
exercise exactly that engine surface:

* Unit: InklingAudioTower / InklingHMLPPatchEncoder parity against hand-rolled torch
  references on shared weights, plan_out_scales schedules, MM config flags,
  and the M-RoPE no-op gate.
* E2E: an engine on the released config (layer-truncated, experts shrunk,
  dummy weights; towers are on in the real config), driven over its wire
  (``WireDriver``) with requests carrying ``MultimodalInputs`` the way a
  frontend that preprocesses media sends them.

The e2e needs a Blackwell GPU (FA4); unit tests need any CUDA GPU (runtime
RMSNorm kernels are GPU-only).

Run:
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=python:tokenspeed-kernel/python \
  python3 -m pytest test/runtime/models/test_inkling_multimodal.py -q
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Add project root directory to path for importing test.* helpers.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from test.runtime.models.inkling_fixtures import (  # noqa: E402
    INKLING_BF16,
    load_inkling_config,
    truncation_hf_overrides,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=120,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*", "h100-*"],
)
from tokenspeed.runtime.configs.inkling_config import (  # noqa: E402
    INKLING_AUDIO_PLACEHOLDER_TOKEN_ID,
    INKLING_IMAGE_PLACEHOLDER_TOKEN_ID,
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


class TestInklingPlanOutScales:
    def test_tiny_fixture_schedule(self):
        from tokenspeed.runtime.models.inkling import inkling_plan_out_scales

        assert (inkling_plan_out_scales(1, 4, 1, 3)) == ([(1, 1, 1, 3), (1, 4, 4, 64)])

    def test_real_config_schedule(self):
        # Cross-checked against the reference scipy implementation.
        from tokenspeed.runtime.models.inkling import inkling_plan_out_scales

        assert (inkling_plan_out_scales(2, 40, 4, 3)) == (
            [
                (1, 1, 1, 3),
                (1, 5, 5, 128),
                (1, 10, 10, 320),
                (1, 40, 40, 4800),
                (2, 40, 40, 9600),
            ]
        )


class TestInklingMultimodalConfig:
    def test_arch_flags(self):
        from tokenspeed.runtime.configs.model_config import (
            is_audio_model,
            is_multimodal_model,
        )

        arch = ["InklingForConditionalGeneration"]
        assert is_multimodal_model(arch)
        assert is_audio_model(arch)
        assert not (is_audio_model(["Qwen3_5ForConditionalGeneration"]))

    def test_placeholder_token_ids(self):
        from tokenspeed.runtime.configs.inkling_config import (
            INKLING_MODEL_END_SAMPLING_TOKEN_ID,
            InklingMMConfig,
        )

        cfg = load_inkling_config()
        assert (cfg.image_placeholder_token_id) == (INKLING_IMAGE_PLACEHOLDER_TOKEN_ID)
        assert (cfg.audio_placeholder_token_id) == (INKLING_AUDIO_PLACEHOLDER_TOKEN_ID)
        assert (cfg.eos_token_id) == (INKLING_MODEL_END_SAMPLING_TOKEN_ID)
        assert (cfg.text_config.eos_token_id) == (INKLING_MODEL_END_SAMPLING_TOKEN_ID)
        # Checkpoint soft-placeholder IDs also give the text-only MTP draft
        # safe, modality-specific embeddings for media feature positions.
        default = InklingMMConfig()
        assert (default.image_placeholder_token_id) == (
            INKLING_IMAGE_PLACEHOLDER_TOKEN_ID
        )
        assert (default.audio_placeholder_token_id) == (
            INKLING_AUDIO_PLACEHOLDER_TOKEN_ID
        )
        assert (default.eos_token_id) == (INKLING_MODEL_END_SAMPLING_TOKEN_ID)
        assert (default.text_config.eos_token_id) == (
            INKLING_MODEL_END_SAMPLING_TOKEN_ID
        )
        assert (default.audio_config.decoder_dmodel) is None
        assert (default.vision_config.decoder_dmodel) is None
        assert (default.audio_config.dmel_min_value) == (-7.0)
        assert (default.audio_config.audio_rms_norm_floor) == (0.01)
        # Transport placeholder IDs must remain valid unsigned token IDs.
        with pytest.raises((ValueError)):
            InklingMMConfig(image_placeholder_token_id=-1)

    def test_mrope_is_noop_for_tml(self):
        """Inkling is not an M-RoPE architecture: compute_mrope_positions
        gives it no positions, so its requests carry no mrope_* fields."""
        from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
        from tokenspeed.runtime.multimodal.mrope import compute_mrope_positions

        cfg = load_inkling_config()
        v = cfg.vision_config
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.zeros(
                2, v.temporal_patch_size, v.patch_size, v.patch_size, v.n_channels
            ),
            offsets=[(1, 2)],
        )
        positions, delta = compute_mrope_positions(
            cfg, [7, INKLING_IMAGE_PLACEHOLDER_TOKEN_ID] * 2, [item]
        )
        assert (positions) is None
        assert (delta) is None


@pytest.mark.skipif(
    not (torch.cuda.is_available()), reason=("runtime RMSNorm kernels need CUDA")
)
class TestInklingAudioTower:
    def test_matches_reference(self):
        from tokenspeed.runtime.models.inkling import InklingAudioTower

        torch.manual_seed(0)
        cfg = load_inkling_config().audio_config
        tower = InklingAudioTower(cfg).cuda()
        n_bins, vocab = cfg.n_mel_bins, cfg.mel_vocab_size

        dmel = torch.randint(0, vocab, (13, n_bins), device="cuda")
        out = tower(dmel)
        assert (out.shape) == ((13, cfg.decoder_dmodel))

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
        from tokenspeed.runtime.models.inkling import InklingAudioTower

        torch.manual_seed(1)
        cfg = load_inkling_config().audio_config
        cfg.use_audio_norm = False
        tower = InklingAudioTower(cfg).cuda()
        assert (tower.final_norm) is None
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
        _pytest_a_1 = peak
        _pytest_b_2 = out_bytes * 8
        _pytest_msg_3 = (
            f"peak {peak / 2**20:.1f} MB suggests the unfused lookup+sum is "
            f"back (unfused would be ~{unfused_bytes / 2**20:.1f} MB, output is "
            f"{out_bytes / 2**20:.1f} MB)"
        )
        assert _pytest_a_1 < _pytest_b_2, _pytest_msg_3


@pytest.mark.skipif(
    not (torch.cuda.is_available()), reason=("runtime RMSNorm kernels need CUDA")
)
class TestInklingHMLPPatchEncoder:
    def test_released_config_shape_and_parity(self):
        from tokenspeed.runtime.models.inkling import InklingHMLPPatchEncoder

        torch.manual_seed(0)
        cfg = load_inkling_config().vision_config
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
        assert (out.shape) == ((9, cfg.decoder_dmodel))
        torch.testing.assert_close(
            out, _naive_hmlp_forward(enc, x), atol=1e-3, rtol=1e-3
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
        assert (out.shape) == ((3, 512))
        torch.testing.assert_close(
            out, _naive_hmlp_forward(enc, x), atol=1e-3, rtol=1e-3
        )


@pytest.mark.skipif(
    not (_has_blackwell()), reason=("Inkling e2e needs a Blackwell GPU (FA4)")
)
class TestInklingMultimodalE2E:
    """Engine-only e2e with synthetic features (no gateway).

    Requests carry their items as a preprocessing frontend sends them: each
    item's pad value over its placeholder run, the original ids beside.
    """

    IMAGE_TOKENS = 8  # one token per patch (hMLP folds patch interior)
    AUDIO_TOKENS = 6

    @classmethod
    def setup_class(cls):
        from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

        # Released config (towers on), layer-truncated + expert-shrunk via
        # hf_overrides; dummy weights.
        cls.text = load_inkling_config().get_text_config()
        cls.vision = load_inkling_config().vision_config
        cls.audio = load_inkling_config().audio_config
        cls.engine = WireDriver(
            [
                "--disable-autotune",
                "--model",
                INKLING_BF16,
                "--hf-overrides",
                truncation_hf_overrides(),
                "--load-format",
                "dummy",
                "--attention-backend",
                "fa4",
                "--disable-prefix-caching",
                "--disable-kvstore",
                "--dtype",
                "bfloat16",
                "--gpu-memory-utilization",
                "0.3",
                "--max-model-len",
                "2048",
                "--max-num-seqs",
                "8",
                "--enable-output-logprobs",
                "--log-level",
                "warning",
            ],
            num_engines=1,
            routing="round_robin",
            startup_timeout_s=3600,
        )

    @classmethod
    def teardown_class(cls):
        cls.engine.shutdown()

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
        ids = prefix + [INKLING_IMAGE_PLACEHOLDER_TOKEN_ID] * num_tokens
        return ids, len(prefix)

    @staticmethod
    def _audio_prompt(num_tokens: int) -> tuple[list[int], int]:
        prefix = [21, 33, 407]
        ids = prefix + [INKLING_AUDIO_PLACEHOLDER_TOKEN_ID] * num_tokens
        return ids, len(prefix)

    def _generate(self, input_ids, mm_items=None, max_new_tokens=8):
        """Submit one request as a preprocessing frontend does and return
        ``(output_ids, output_logprobs)``."""
        from tokenspeed.runtime.entrypoints.wire_driver import generate_request
        from tokenspeed.runtime.multimodal.embedder import pad_input_tokens
        from tokenspeed.runtime.multimodal.inputs import MultimodalInputs

        fields = {}
        if mm_items is not None:
            mm_inputs = MultimodalInputs(mm_items=mm_items)
            mm_inputs.ensure_pad_values()
            fields = {
                "multimodal_inputs": mm_inputs,
                "input_ids_unpadded": list(input_ids),
            }
            input_ids = pad_input_tokens(list(input_ids), mm_inputs)
        [completion] = self.engine.generate(
            [
                generate_request(
                    input_ids,
                    {"temperature": 0.0, "max_new_tokens": max_new_tokens},
                    return_logprob=True,
                    **fields,
                )
            ]
        )
        assert (completion.completion_tokens) == (max_new_tokens)
        logprobs = completion.output_token_logprobs
        output_ids = completion.output_ids
        for token_id in output_ids:
            # padded-vocab mask still active
            assert (token_id) < (self.text.unpadded_vocab_size)
        return output_ids, logprobs

    @classmethod
    def _image_item(cls, seed: int, num_tokens: int, run_start: int):
        from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

        g = torch.Generator().manual_seed(seed)
        v = cls.vision
        feature = torch.randn(
            num_tokens,
            v.temporal_patch_size,
            v.patch_size,
            v.patch_size,
            v.n_channels,
            generator=g,
        )
        return MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=[(run_start, run_start + num_tokens - 1)],
        )

    @classmethod
    def _audio_item(cls, seed: int, num_tokens: int, run_start: int):
        from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

        g = torch.Generator().manual_seed(seed)
        a = cls.audio
        feature = torch.randint(
            0, a.mel_vocab_size, (num_tokens, a.n_mel_bins), generator=g
        )
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

        # Same feature => deterministic.
        assert (a_ids) == (a2_ids)
        assert (a_lp) == (a2_lp)
        # The splice must reach the LM (with vs without differs). Image
        # feature CONTENT is not assertable under dummy weights — the deep
        # hMLP contracts inputs below bf16 visibility; the audio A-vs-B
        # check covers content observability.
        assert (
            a_ids != base_ids or a_lp != base_lp
        ), "image feature did not change the model output"

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

        assert (a_ids) == (a2_ids)
        assert (a_lp) == (a2_lp)
        assert (
            a_ids != base_ids or a_lp != base_lp
        ), "audio feature did not change the model output"
        assert (
            a_ids != b_ids or a_lp != b_lp
        ), "different audio features produced identical outputs"

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
            + [INKLING_IMAGE_PLACEHOLDER_TOKEN_ID] * self.IMAGE_TOKENS
            + mid
            + [INKLING_AUDIO_PLACEHOLDER_TOKEN_ID] * self.AUDIO_TOKENS
        )
        items = [
            self._image_item(5, self.IMAGE_TOKENS, img_start),
            self._audio_item(6, self.AUDIO_TOKENS, aud_start),
        ]
        a_ids, a_lp = self._generate(ids, items)
        assert (len(a_ids)) == (8)
        items_b = [
            self._image_item(5, self.IMAGE_TOKENS, img_start),
            self._audio_item(7, self.AUDIO_TOKENS, aud_start),
        ]
        b_ids, b_lp = self._generate(ids, items_b)
        assert (
            a_ids != b_ids or a_lp != b_lp
        ), "changing the trailing audio feature did not change the output"

    def test_text_only_prompt_still_works(self):
        # Plain text through the same engine (no mm items): the text-only
        # fast path must keep working alongside mm requests.
        output_ids, _ = self._generate([11, 45, 260, 132])
        assert (len(output_ids)) == (8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
