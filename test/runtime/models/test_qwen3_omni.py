import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tokenspeed.runtime.configs.model_config import (
    ModelConfig,
    get_hf_text_config,
    is_audio_model,
    is_multimodal_model,
)
from tokenspeed.runtime.models.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeTextModel,
    _shared_vision_config,
)
from tokenspeed.runtime.multimodal.embedder import (
    EncodePlan,
    EncoderSpec,
    MultimodalEmbedder,
    ScatterRange,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
    MultimodalInputs,
)
from tokenspeed.runtime.multimodal.mrope import compute_mrope_positions


def _omni_config():
    vision = SimpleNamespace(spatial_merge_size=2)
    text = SimpleNamespace(hidden_size=2)
    thinker = SimpleNamespace(
        vision_config=vision,
        text_config=text,
        position_id_per_seconds=13,
        seconds_per_chunk=2,
    )
    return SimpleNamespace(
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
        thinker_config=thinker,
    )


class TestQwen3OmniConfig(unittest.TestCase):
    def test_architecture_flags_cover_asr_and_omni(self):
        for architecture in (
            "Qwen3ASRForConditionalGeneration",
            "Qwen3OmniMoeForConditionalGeneration",
        ):
            with self.subTest(architecture=architecture):
                self.assertTrue(is_multimodal_model([architecture]))
                self.assertTrue(is_audio_model([architecture]))

        direct_thinker = "Qwen3OmniMoeThinkerForConditionalGeneration"
        self.assertFalse(is_multimodal_model([direct_thinker]))
        self.assertFalse(is_audio_model([direct_thinker]))

    def test_model_config_unwraps_omni_thinker_text(self):
        config = _omni_config()
        self.assertIs(get_hf_text_config(config), config.thinker_config.text_config)

    def test_encode_disaggregation_rejects_audio_models(self):
        hf_config = SimpleNamespace(architectures=["Qwen3ASRForConditionalGeneration"])
        server_args = SimpleNamespace(
            mapping=None,
            language_model_only=False,
            disaggregation_mode="encode",
        )
        with (
            patch(
                "tokenspeed.runtime.configs.model_config.get_config",
                return_value=hf_config,
            ),
            patch(
                "tokenspeed.runtime.configs.model_config.get_generation_config",
                return_value=None,
            ),
            patch(
                "tokenspeed.runtime.configs.model_config.get_hf_text_config",
                return_value=SimpleNamespace(),
            ),
            self.assertRaisesRegex(ValueError, "does not support audio models"),
        ):
            ModelConfig(
                "stub",
                model_override_args="{}",
                server_args=server_args,
            )


class TestQwen3OmniMultimodalEmbedder(unittest.TestCase):
    @staticmethod
    def _plan_for(item: MultimodalDataItem) -> EncodePlan:
        return EncodePlan(
            scatter_ranges=[
                ScatterRange(
                    flat_dst_start=0,
                    flat_dst_end=0,
                    item=item,
                    item_src_start=0,
                    item_src_end=0,
                )
            ]
        )

    def test_same_hash_does_not_alias_across_modalities(self):
        audio = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=7,
            offsets=[(0, 0)],
        )
        video = MultimodalDataItem(
            modality=Modality.VIDEO,
            hash=7,
            offsets=[(1, 1)],
        )
        ctx = MultimodalForwardContext(
            mm_inputs=[MultimodalInputs(mm_items=[audio, video])],
            extend_prefix_lens=[0],
            extend_seq_lens=[2],
        )

        plan = MultimodalEmbedder()._plan(ctx)

        self.assertEqual(plan.misses_by_modality[Modality.AUDIO], [audio])
        self.assertEqual(plan.misses_by_modality[Modality.VIDEO], [video])
        self.assertEqual([r.item for r in plan.scatter_ranges], [audio, video])
        self.assertFalse(plan.aliases_by_canonical)

    def test_deepstack_buffer_only_for_active_deepstack_modality(self):
        embedder = MultimodalEmbedder()
        text_embedding = nn.Embedding(2, 2)
        input_ids = torch.tensor([0])
        encoders = {
            Modality.AUDIO: EncoderSpec(fn=lambda _: torch.empty(0, 2)),
            Modality.IMAGE: EncoderSpec(fn=lambda _: torch.empty(0, 2), deepstack=True),
        }
        model = SimpleNamespace(deepstack_visual_indexes=[0])

        audio = MultimodalDataItem(
            modality=Modality.AUDIO,
            encoded=torch.tensor([[1.0, 2.0]]),
        )
        _, audio_kwargs = embedder._assemble(
            input_ids,
            text_embedding,
            self._plan_for(audio),
            encoders,
            model,
        )
        self.assertNotIn("input_deepstack_embeds", audio_kwargs)

        image = MultimodalDataItem(
            modality=Modality.IMAGE,
            encoded=torch.tensor([[1.0, 2.0]]),
            encoded_deepstack=torch.tensor([[3.0, 4.0]]),
        )
        _, image_kwargs = embedder._assemble(
            input_ids,
            text_embedding,
            self._plan_for(image),
            encoders,
            model,
        )
        torch.testing.assert_close(
            image_kwargs["input_deepstack_embeds"],
            image.encoded_deepstack,
        )


class TestQwen3OmniMrope(unittest.TestCase):
    def test_mixed_modalities_follow_authored_offsets(self):
        audio = MultimodalDataItem(
            modality=Modality.AUDIO,
            offsets=[(2, 4)],
        )
        image = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(8, 11)],
            model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 4]])},
        )
        video = MultimodalDataItem(
            modality=Modality.VIDEO,
            offsets=[(15, 22)],
            model_specific_data={
                "video_grid_thw": torch.tensor([[2, 4, 4]]),
                "video_second_per_grid": torch.tensor([1.0]),
            },
        )

        # Pass items out of order: item offsets, not modality batches, define
        # the authored image/audio/video sequence.
        positions, delta = compute_mrope_positions(
            _omni_config(), list(range(25)), [video, audio, image]
        )

        self.assertEqual(positions.shape, (3, 25))
        self.assertTrue(torch.equal(positions[:, :8], torch.arange(8).expand(3, -1)))
        self.assertEqual(
            positions[:, 8:12].tolist(),
            [
                [8, 8, 8, 8],
                [8, 8, 9, 9],
                [8, 9, 8, 9],
            ],
        )
        self.assertTrue(
            torch.equal(positions[:, 12:15], torch.arange(10, 13).expand(3, -1))
        )
        self.assertEqual(positions[0, 15:23].tolist(), [13] * 4 + [26] * 4)
        self.assertEqual(positions[1, 15:23].tolist(), [13, 13, 14, 14] * 2)
        self.assertEqual(positions[2, 15:23].tolist(), [13, 14, 13, 14] * 2)
        self.assertTrue(
            torch.equal(positions[:, 23:], torch.arange(27, 29).expand(3, -1))
        )
        self.assertEqual(delta.tolist(), [[4]])

    def test_multiple_videos_support_fractional_seconds_per_grid(self):
        first = MultimodalDataItem(
            modality=Modality.VIDEO,
            offsets=[(2, 9)],
            model_specific_data={
                "video_grid_thw": torch.tensor([[2, 4, 4]]),
                "video_second_per_grid": torch.tensor([0.5]),
            },
        )
        second = MultimodalDataItem(
            modality=Modality.VIDEO,
            offsets=[(12, 19)],
            model_specific_data={
                "video_grid_thw": torch.tensor([[2, 4, 4]]),
                "video_second_per_grid": torch.tensor([1.5]),
            },
        )
        positions, delta = compute_mrope_positions(
            _omni_config(), list(range(22)), [second, first]
        )

        self.assertEqual(positions[0, 2:10].tolist(), [2] * 4 + [8] * 4)
        self.assertEqual(positions[0, 12:20].tolist(), [11] * 4 + [30] * 4)
        self.assertEqual(delta.tolist(), [[11]])

    def test_rejects_audio_in_video_interleaving(self):
        video = MultimodalDataItem(
            modality=Modality.VIDEO,
            offsets=[(1, 4)],
            model_specific_data={
                "video_grid_thw": torch.tensor([[1, 4, 4]]),
                "use_audio_in_video": torch.tensor([True]),
            },
        )
        with self.assertRaisesRegex(ValueError, "use_audio_in_video=true"):
            compute_mrope_positions(_omni_config(), list(range(6)), [video])

    def test_rejects_placeholder_grid_mismatch(self):
        image = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(1, 3)],
            model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 4]])},
        )
        with self.assertRaisesRegex(ValueError, "grid requires 4"):
            compute_mrope_positions(_omni_config(), list(range(5)), [image])


class _FakeCommManager:
    def final_norm(self, hidden_states, residual, ctx, norm):
        return hidden_states, None


class _IdentityDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.comm_manager = _FakeCommManager()

    def forward(
        self,
        positions,
        hidden_states,
        ctx,
        out_cache_loc,
        residual,
        cos_sin=None,
    ):
        return hidden_states, residual


class TestQwen3OmniModelHelpers(unittest.TestCase):
    def test_language_only_mode_rejects_multimodal_context(self):
        model = Qwen3OmniMoeForConditionalGeneration.__new__(
            Qwen3OmniMoeForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.is_multimodal_active = False
        multimodal_context = SimpleNamespace(has_extend_inputs=lambda: True)
        ctx = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False)
        )

        with self.assertRaisesRegex(RuntimeError, "encoders are disabled"):
            model.forward(
                ctx,
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
                multimodal_context=multimodal_context,
            )

    def test_deepstack_is_injected_after_early_decoder_layers(self):
        model = Qwen3OmniMoeTextModel.__new__(Qwen3OmniMoeTextModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=2)
        model.embed_tokens = nn.Embedding(4, 2)
        model.layers = nn.ModuleList([_IdentityDecoderLayer(), _IdentityDecoderLayer()])
        model.norm = nn.Identity()
        ctx = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))

        hidden_states, _ = model(
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
            ctx=ctx,
            out_cache_loc=torch.tensor([0]),
            input_embeds=torch.zeros(1, 2),
            input_deepstack_embeds=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        )
        self.assertTrue(torch.equal(hidden_states, torch.tensor([[4.0, 6.0]])))

    def test_visual_checkpoint_names_map_to_shared_qwen3_tower(self):
        map_weight = Qwen3OmniMoeForConditionalGeneration._map_visual_weight
        self.assertEqual(
            map_weight("visual.blocks.0.attn.qkv.weight"),
            "visual.blocks.0.attn.qkv_proj.weight",
        )
        self.assertEqual(
            map_weight("visual.merger_list.2.ln_q.weight"),
            "visual.deepstack_merger_list.2.norm.weight",
        )
        self.assertEqual(
            map_weight("visual.merger.mlp.2.bias"),
            "visual.merger.linear_fc2.bias",
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.cuda is not None,
        "Qwen3 vision parity requires an NVIDIA GPU",
    )
    def test_shared_vision_tower_matches_transformers_omni(self):
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeVisionEncoder,
        )

        from tokenspeed.runtime.distributed.mapping import Mapping
        from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
        from tokenspeed.runtime.models.qwen3_vision import Qwen3VLMoeVisionModel

        config = Qwen3OmniMoeVisionEncoderConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=4,
            deepstack_visual_indexes=[0],
        )
        torch.manual_seed(1)
        reference = Qwen3OmniMoeVisionEncoder(config).cuda().bfloat16().eval()
        tower = (
            Qwen3VLMoeVisionModel(
                _shared_vision_config(config),
                Mapping(rank=0, world_size=1),
                prefix="visual",
            )
            .cuda()
            .bfloat16()
            .eval()
        )
        params = dict(tower.named_parameters(remove_duplicate=False))
        for name, weight in reference.state_dict().items():
            mapped = Qwen3OmniMoeForConditionalGeneration._map_visual_weight(
                f"visual.{name}"
            ).removeprefix("visual.")
            self.assertIn(mapped, params)
            param = params[mapped]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)

        patches = torch.randn(16, 24, device="cuda", dtype=torch.bfloat16)
        grid = torch.tensor([[1, 4, 4]], device="cuda")
        with torch.no_grad():
            reference_output = reference(patches, grid)
            tokens = tower.prepare_patch_embed(patches, grid)
            output = tower.forward_blocks(tokens, tower.prepare_metadata(grid))
        expected = torch.cat(
            [
                reference_output.pooler_output,
                *reference_output.deepstack_features,
            ],
            dim=-1,
        )
        torch.testing.assert_close(output, expected, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
