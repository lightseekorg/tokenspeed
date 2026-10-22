# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Nemotron-H Omni encoders against independent references.

C-RADIO runs packed images and video tubelets against a per-sequence torch
ViT on the same weights; Parakeet runs against transformers'
``ParakeetEncoder``, unpadded and as a padded batch.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="Nemotron-H Omni is validated on NVIDIA only",
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from transformers import ParakeetEncoderConfig  # noqa: E402
from transformers.models.parakeet.modeling_parakeet import (  # noqa: E402
    ParakeetEncoder as HFParakeetEncoder,
)
from transformers.models.parakeet.modeling_parakeet import (  # noqa: E402
    ParakeetEncoderRelPositionalEncoding,
)

from tokenspeed.runtime.configs.nemotron_omni_config import (  # noqa: E402
    NemotronHOmniConfig,
    ParakeetAudioConfig,
    RadioVisionConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.models import nemotron_omni  # noqa: E402
from tokenspeed.runtime.models.parakeet import (  # noqa: E402
    ParakeetEncoder,
    _relative_positions,
)
from tokenspeed.runtime.models.radio import (  # noqa: E402
    RadioEncoderCudaGraphAdapter,
    RadioVisionModel,
)
from tokenspeed.runtime.multimodal.encoder_cudagraph import (  # noqa: E402
    EncoderForwardStepRunner,
)

DEVICE = "cuda"
PATCH = 16
GRID = 8  # position-embedding grid side


def _radio_config(final_norm: bool, temporal_patch_size: int) -> RadioVisionConfig:
    return RadioVisionConfig(
        args={
            "model": "vit_base_patch16_224",
            "cpe_max_size": GRID * PATCH,
            "cls_token_per_teacher": True,
            "teachers": [{"name": "clip"}, {"name": "siglip"}, {"name": "clip"}],
            "register_multiple": 4,
            "model_norm": final_norm,
            # Video passes then hold two 4x6 tubelets.
            "max_num_patches": 24,
        },
        patch_size=PATCH,
        video_temporal_patch_size=temporal_patch_size,
        # The checkpoint repeats the planner's budgets outside args.
        max_num_patches=24,
    )


def _radio(final_norm: bool) -> RadioVisionModel:
    config = _radio_config(final_norm, temporal_patch_size=2)
    torch.manual_seed(0)
    model = RadioVisionModel(
        config,
        Mapping(rank=0, world_size=1),
        mm_attention_backend=None,
        prefix="visual",
    )
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.05)
    return model.to(DEVICE, torch.bfloat16)


def _reference_vit(model: RadioVisionModel, embedded: torch.Tensor) -> torch.Tensor:
    """Pre-norm ViT in the model dtype with independent per-sequence math attention."""
    heads = model.blocks[0].attn.num_attention_heads_per_partition
    x = torch.cat([model.cls_token, embedded.to(model.dtype)])
    for block in model.blocks:
        h = F.layer_norm(
            x, x.shape[-1:], block.norm1.weight, block.norm1.bias, block.norm1.eps
        )
        qkv = F.linear(h, block.attn.qkv_proj.weight, block.attn.qkv_proj.bias)
        q, k, v = (
            t.unflatten(-1, (heads, -1)).transpose(0, 1) for t in qkv.chunk(3, -1)
        )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attn = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).flatten(1)
        x = x + F.linear(attn, block.attn.proj.weight, block.attn.proj.bias)
        h = F.layer_norm(
            x, x.shape[-1:], block.norm2.weight, block.norm2.bias, block.norm2.eps
        )
        mlp = block.mlp
        h = F.gelu(F.linear(h, mlp.fc1.weight, mlp.fc1.bias))
        x = x + F.linear(h, mlp.fc2.weight, mlp.fc2.bias)
    if model.norm is not None:
        x = F.layer_norm(
            x, x.shape[-1:], model.norm.weight, model.norm.bias, model.norm.eps
        )
    return x[model.num_prefix_tokens :].float()


def _reference_position(model: RadioVisionModel, rows: int, cols: int) -> torch.Tensor:
    grid = model.pos_embed.float().view(1, GRID, GRID, -1).permute(0, 3, 1, 2)
    side = max(rows, cols)
    grid = F.interpolate(grid, size=(side, side), mode="bilinear", align_corners=False)
    return grid[0, :, :rows, :cols].flatten(1).T.to(model.dtype).float()


def _assert_close_bf16(out: torch.Tensor, ref: torch.Tensor) -> None:
    # bf16 encoder against independent references on the same weights.
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


def test_radio_adds_a_full_register_multiple_when_already_aligned():
    config = _radio_config(final_norm=False, temporal_patch_size=2)
    config.args["teachers"] = [{"name": name} for name in ("a", "b", "c", "d")]
    assert config.num_prefix_tokens == 8


@pytest.mark.parametrize("final_norm", [False, True])
def test_radio_packs_images_of_any_size_like_one_at_a_time(final_norm: bool):
    model = _radio(final_norm)
    assert (model.norm is not None) == final_norm
    # Two distinct teachers, padded to a multiple of 4.
    assert model.num_prefix_tokens == 4
    # Smaller, wider and taller than the position grid.
    grids = [(4, 6), (10, 4), (GRID, GRID)]
    images = [
        torch.randn(rows * cols, 3 * PATCH * PATCH, device=DEVICE)
        for rows, cols in grids
    ]
    with torch.no_grad():
        out = model.embed_images(torch.cat(images), grids)
        refs = [
            _reference_vit(
                model,
                F.linear(image.to(model.dtype), model.embedder.weight)
                + _reference_position(model, rows, cols).to(model.dtype),
            )
            for image, (rows, cols) in zip(images, grids)
        ]
    _assert_close_bf16(out, torch.cat(refs))


def test_radio_position_embedding_equals_resize_then_crop():
    model = _radio(final_norm=False)
    # Thin and wide grids far past the position grid read only part of it.
    for rows, cols in [
        (2, 50),
        (2, 200),
        (37, 3),
        (GRID, 3 * GRID + 1),
        (9, 9),
        (3, 5),
    ]:
        out = model.position_embedding(rows, cols)
        assert torch.equal(out.float(), _reference_position(model, rows, cols))


def test_radio_refuses_video_without_a_tubelet_embedder():
    with pytest.raises(NotImplementedError):
        RadioVisionModel(
            _radio_config(final_norm=False, temporal_patch_size=1),
            Mapping(rank=0, world_size=1),
            mm_attention_backend=None,
            prefix="visual",
        )


def test_radio_video_pairs_frames_and_repeats_an_odd_last_frame():
    model = _radio(final_norm=False)
    frames, rows, cols = 3, 4, 6
    patches = torch.randn(frames, rows * cols, 3 * PATCH * PATCH, device=DEVICE)
    with torch.no_grad():
        out = model.embed_video(patches.flatten(0, 1), frames, rows, cols)
        tubelets = [patches[[0, 1]], patches[[2, 2]]]
        refs = [
            _reference_vit(
                model,
                F.linear(
                    torch.cat([pair[0], pair[1]], dim=-1).to(model.dtype),
                    model.video_embedder.weight,
                )
                + _reference_position(model, rows, cols).to(model.dtype),
            )
            for pair in tubelets
        ]
    _assert_close_bf16(out, torch.cat(refs))


def test_radio_video_runs_tubelets_in_bounded_passes_bit_for_bit():
    model = _radio(final_norm=False)
    # Three 4x6 tubelets: passes of two, then one.
    frames, rows, cols = 5, 4, 6
    patches = torch.randn(frames * rows * cols, 3 * PATCH * PATCH, device=DEVICE)
    with torch.no_grad():
        passes = model.embed_video(patches, frames, rows, cols)
        model.video_patches_per_pass = frames * rows * cols
        whole = model.embed_video(patches, frames, rows, cols)
    assert torch.equal(passes, whole)


SOUND_CONFIG = {
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "conv_kernel_size": 9,
    "convolution_bias": False,
    "subsampling_conv_channels": 16,
    "subsampling_conv_kernel_size": 3,
    "subsampling_conv_stride": 2,
    "subsampling_factor": 8,
    "num_mel_bins": 32,
    "projection_hidden_size": 64,
    "projection_bias": False,
    "sampling_rate": 16000,
}


def test_config_round_trips_through_its_dict():
    config = NemotronHOmniConfig(
        llm_config={"hidden_size": 64},
        vision_config={"args": {"model": "vit_base_patch16_224"}},
        sound_config=SOUND_CONFIG,
        sound_context_token_id=27,
    )
    rebuilt = NemotronHOmniConfig.from_dict(config.to_dict())
    assert rebuilt.text_config.hidden_size == 64
    assert rebuilt.to_dict() == config.to_dict()
    assert (rebuilt.image_token_id, rebuilt.audio_token_id) == (18, 27)


def test_config_reads_a_transformers_format_radio_config():
    # The ViT shape stated beside args, no args.model_norm, and an MTP language model.
    vision = {
        "args": {
            "model": "vit_huge_patch16_224",
            "cls_token_per_teacher": True,
            "teachers": [{"name": 0}, {"name": 1}, {"name": 2}],
            "register_multiple": 10,
        },
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "hidden_act": "gelu",
        "qkv_bias": True,
        "use_swiglu_ffn": False,
        "layerscale_value": 1.0,
        "num_cls_tokens": 3,
        "num_registers": 7,
        "layer_norm_eps": 1e-5,
    }
    config = NemotronHOmniConfig(
        llm_config={"num_nextn_predict_layers": 1}, vision_config=vision
    )
    radio = config.vision_config
    assert (radio.hidden_size, radio.num_prefix_tokens) == (1280, 10)
    assert (radio.layer_norm_eps, radio.final_norm) == (1e-5, False)
    assert config.vision_final_layernorm
    assert not NemotronHOmniConfig(vision_config=vision).vision_final_layernorm
    rebuilt = NemotronHOmniConfig.from_dict(config.to_dict())
    assert rebuilt.to_dict() == config.to_dict()
    for name, value in [("hidden_size", 1024), ("num_registers", 6)]:
        with pytest.raises(ValueError, match=name):
            RadioVisionConfig(**{**vision, name: value})
    with pytest.raises(ValueError, match="layer_norm_eps"):
        RadioVisionConfig(
            **{**vision, "args": {**vision["args"], "layer_norm_eps": 1e-6}}
        )


def test_config_refuses_a_sound_config_without_the_encoder_shape():
    sound = {k: v for k, v in SOUND_CONFIG.items() if k != "conv_kernel_size"}
    with pytest.raises(ValueError, match="conv_kernel_size"):
        NemotronHOmniConfig(sound_config=sound)


def test_encoders_honor_or_refuse_settings_they_do_not_implement():
    config = _radio_config(final_norm=True, temporal_patch_size=2)
    config.args["layer_norm_eps"] = 1e-3
    model = RadioVisionModel(
        config, Mapping(rank=0, world_size=1), mm_attention_backend=None, prefix="v"
    )
    norms = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert norms and all(norm.eps == 1e-3 for norm in norms)
    with pytest.raises(ValueError, match="hidden_act"):
        ParakeetAudioConfig(hidden_act="relu")
    for name, value in [
        ("hidden_act", "relu"),
        ("qkv_bias", False),
        ("qk_normalization", True),
        ("norm_type", "rms_norm"),
    ]:
        with pytest.raises(ValueError, match=f"args.{name}"):
            RadioVisionConfig(args={**config.args, name: value})
    with pytest.raises(NotImplementedError, match="flashinfer_cudnn"):
        RadioVisionModel(
            config,
            Mapping(rank=0, world_size=1),
            mm_attention_backend="flashinfer_cudnn",
            prefix="v",
        )


def test_pixel_shuffle_orders_each_block_by_row_column_channel():
    rows, cols, channels = 4, 6, 3
    features = torch.arange(rows * cols * channels).view(rows * cols, channels)
    merged = nemotron_omni.pixel_shuffle(features, rows, cols)
    grid = features.view(rows, cols, channels)
    # Block (1, 2) covers rows 2-3 and columns 4-5.
    expected = torch.cat([grid[2, 4], grid[2, 5], grid[3, 4], grid[3, 5]])
    assert merged.shape == (rows * cols // 4, 4 * channels)
    assert torch.equal(merged[1 * (cols // 2) + 2], expected)
    # The ref engine's v2 shuffle: blocks row-major, each block's patches row-major.
    reference = grid.view(rows // 2, 2, cols // 2, 2, channels).permute(0, 2, 1, 3, 4)
    assert torch.equal(merged, reference.reshape(-1, 4 * channels))


def _projector(input_size: int) -> nemotron_omni.NemotronOmniProjector:
    torch.manual_seed(1)
    projector = nemotron_omni.NemotronOmniProjector(
        input_size, 64, 32, bias=False, eps=1e-5
    )
    with torch.no_grad():
        for param in projector.parameters():
            param.normal_(std=0.1)
    return projector.to(DEVICE, torch.bfloat16)


def _reference_projector(projector, x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    hidden = F.linear(
        x * projector.norm.weight.float(), projector.linear1.weight.float()
    )
    return F.linear(torch.relu(hidden).square(), projector.linear2.weight.float())


def test_projector_is_norm_linear_squared_relu_linear():
    projector = _projector(48)
    x = torch.randn(10, 48, device=DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        _assert_close_bf16(projector(x), _reference_projector(projector, x))


def _omni_glue(**modules) -> SimpleNamespace:
    """The model's feature methods bound to stand-in modules."""
    model = SimpleNamespace(**{"vision_final_norm": None, **modules})
    model._project_vision = lambda features, grids: (
        nemotron_omni.NemotronH_Nano_Omni_Reasoning_V3._project_vision(
            model, features, grids
        )
    )
    return model


def test_video_feature_projects_each_tubelet_on_its_grid():
    radio = _radio(final_norm=False)
    frames, rows, cols = 3, 4, 6
    projector = _projector(4 * radio.blocks[0].norm1.normalized_shape[0])
    model = _omni_glue(visual=radio, vision_projector=projector)
    patches = torch.randn(frames * rows * cols, 3 * PATCH * PATCH, device=DEVICE)
    item = SimpleNamespace(
        feature=patches, video_grid_thw=torch.tensor([frames, rows, cols])
    )
    get_video_feature = nemotron_omni.NemotronH_Nano_Omni_Reasoning_V3.get_video_feature
    with torch.no_grad():
        out = get_video_feature(model, [item])
        features = radio.embed_video(patches, frames, rows, cols)
        expected = projector(
            torch.cat(
                [
                    nemotron_omni.pixel_shuffle(tubelet, rows, cols)
                    for tubelet in features.split(rows * cols)
                ]
            )
        )
    # Two tubelets (the odd last frame repeats), rows * cols / 4 embeddings each.
    assert out.shape[0] == 2 * rows * cols // 4
    assert torch.equal(out, expected)


def test_image_feature_shuffles_each_image_on_its_own_grid():
    radio = _radio(final_norm=False)
    projector = _projector(4 * radio.blocks[0].norm1.normalized_shape[0])
    model = _omni_glue(visual=radio, vision_projector=projector)
    grids = [(4, 6), (6, 2)]
    patches = [
        torch.randn(rows * cols, 3 * PATCH * PATCH, device=DEVICE)
        for rows, cols in grids
    ]
    items = [
        SimpleNamespace(feature=p, image_grid_hw=torch.tensor(grid))
        for p, grid in zip(patches, grids)
    ]
    get_image_feature = nemotron_omni.NemotronH_Nano_Omni_Reasoning_V3.get_image_feature
    with torch.no_grad():
        out = get_image_feature(model, items)
        features = radio.embed_images(torch.cat(patches), grids)
        shuffled = [
            nemotron_omni.pixel_shuffle(f, rows, cols)
            for f, (rows, cols) in zip(features.split([r * c for r, c in grids]), grids)
        ]
        expected = projector(torch.cat(shuffled))
    assert torch.equal(out, expected)


def test_image_feature_layer_norms_the_vit_output_before_shuffling():
    radio = _radio(final_norm=False)
    hidden = radio.blocks[0].norm1.normalized_shape[0]
    projector = _projector(4 * hidden)
    norm = torch.nn.LayerNorm(hidden, eps=1e-6).to(DEVICE, torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
        norm.bias.normal_(std=0.1)
    model = _omni_glue(visual=radio, vision_projector=projector, vision_final_norm=norm)
    rows, cols = 4, 6
    patches = torch.randn(rows * cols, 3 * PATCH * PATCH, device=DEVICE)
    item = SimpleNamespace(feature=patches, image_grid_hw=torch.tensor([rows, cols]))
    get_image_feature = nemotron_omni.NemotronH_Nano_Omni_Reasoning_V3.get_image_feature
    with torch.no_grad():
        out = get_image_feature(model, [item])
        features = norm(radio.embed_images(patches, [(rows, cols)]))
        expected = projector(nemotron_omni.pixel_shuffle(features, rows, cols))
    assert torch.equal(out, expected)


def _image_items(grids: list[tuple[int, int]]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            feature=torch.randn(rows * cols, 3 * PATCH * PATCH, device=DEVICE),
            image_grid_hw=torch.tensor([rows, cols]),
        )
        for rows, cols in grids
    ]


def test_radio_encoder_graph_matches_the_eager_encoder():
    radio = _radio(final_norm=True)
    # 44, 28 and 68 packed rows replay budget graphs; 148 exceeds them all and runs eagerly.
    grids = [(10, 4), (4, 6), (12, 12), (8, 8)]
    items = _image_items(grids)
    adapter = RadioEncoderCudaGraphAdapter(
        tower=radio,
        post_encode=lambda features, grids: features,
        max_image_tokens=12 * 12 + radio.num_prefix_tokens,
        capture_tp_size=1,
        capture_tp_group=None,
    )
    runner = EncoderForwardStepRunner(
        adapter=adapter, budget_range=(32, 128), max_metadata_sequences_per_batch=8
    )
    with torch.no_grad():
        out = runner(items)
        expected = radio.embed_images(
            torch.cat([item.feature for item in items]), grids
        )
    assert sorted(runner.budget_graphs) == [32, 64, 128]
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


def test_image_encoder_graph_serves_get_image_feature():
    config = _radio_config(final_norm=False, temporal_patch_size=2)
    config.args["min_num_patches"] = 6
    radio = _radio(final_norm=False)
    projector = _projector(4 * radio.blocks[0].norm1.normalized_shape[0])
    model = _omni_glue(
        visual=radio,
        vision_projector=projector,
        omni_config=SimpleNamespace(vision_config=config),
    )
    omni = nemotron_omni.NemotronH_Nano_Omni_Reasoning_V3
    wrappers = omni.make_encoder_cudagraph_wrappers(
        model, Mapping(rank=0, world_size=1)
    )
    items = _image_items([(4, 6), (6, 2)])
    with torch.no_grad():
        out = wrappers["image_encoder"](items)
        expected = omni.get_image_feature(model, items)
    # One smallest image is 6 patches plus 4 prefix tokens.
    assert wrappers["image_encoder"].encoder_output_token_budgets == [
        10,
        20,
        40,
        80,
        160,
        320,
    ]
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


def _parakeet_pair() -> tuple[ParakeetEncoder, HFParakeetEncoder]:
    fields = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        attention_bias=False,
        conv_kernel_size=9,
        convolution_bias=False,
        subsampling_conv_channels=16,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        subsampling_factor=8,
        num_mel_bins=32,
    )
    torch.manual_seed(0)
    reference = HFParakeetEncoder(
        ParakeetEncoderConfig(**fields, scale_input=False)
    ).eval()
    with torch.no_grad():
        for name, tensor in reference.state_dict().items():
            if name.endswith("running_var"):
                tensor.uniform_(0.5, 2.0)
            elif name.endswith("running_mean"):
                tensor.normal_()
            elif ".self_attn." in name:
                # At the default init the relative-position terms barely move the output.
                tensor.normal_(std=0.5)
    ours = ParakeetEncoder(ParakeetAudioConfig(**fields))
    for name, tensor in reference.state_dict().items():
        ours.load_weight(name, tensor)
    return ours.to(DEVICE), reference.to(DEVICE)


def test_parakeet_matches_transformers_unpadded_and_as_a_padded_batch():
    ours, reference = _parakeet_pair()
    lengths = [80, 53, 80]
    clips = [torch.randn(n, 32, device=DEVICE) for n in lengths]
    with torch.no_grad():
        encoded = ours.encode_clips(clips)
        padded = torch.zeros(len(clips), max(lengths), 32, device=DEVICE)
        mask = torch.zeros(len(clips), max(lengths), dtype=torch.long, device=DEVICE)
        for i, clip in enumerate(clips):
            padded[i, : len(clip)] = clip
            mask[i, : len(clip)] = 1
        batched = reference(input_features=padded, attention_mask=mask)
    for i, clip in enumerate(clips):
        with torch.no_grad():
            alone = reference(input_features=clip[None]).last_hidden_state[0]
        assert encoded[i].shape == alone.shape  # one row per 8 frames, rounded up
        torch.testing.assert_close(encoded[i], alone, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            encoded[i], batched.last_hidden_state[i, : len(alone)], atol=1e-4, rtol=1e-4
        )


def test_parakeet_relative_positions_stay_fp32_for_bf16_long_clips():
    # A 30-second clip has 375 encoder positions; offsets past 256 are not exact in
    # bf16, so the sinusoids must be computed in fp32 and only the result cast.
    length, hidden = 375, 1024
    like = torch.zeros(1, length, hidden, dtype=torch.bfloat16, device=DEVICE)
    reference = ParakeetEncoderRelPositionalEncoding(
        ParakeetEncoderConfig(hidden_size=hidden), device=DEVICE
    )
    assert torch.equal(_relative_positions(length, hidden, like), reference(like)[0])


def test_audio_feature_encodes_every_clip_of_every_item_in_order():
    ours, _ = _parakeet_pair()
    ours = ours.to(torch.bfloat16)
    projector = _projector(64)
    model = SimpleNamespace(audio_tower=ours, audio_projector=projector)
    # Two clips in the first item, one in the second.
    clip_frames = [[80, 53], [40]]
    mels = [torch.randn(sum(frames), 32, device=DEVICE) for frames in clip_frames]
    items = [
        SimpleNamespace(feature=mel, audio_clip_frames=torch.tensor(frames))
        for mel, frames in zip(mels, clip_frames)
    ]
    get_audio_feature = nemotron_omni.NemotronH_Nano_Omni_Reasoning_V3.get_audio_feature
    with torch.no_grad():
        out = get_audio_feature(model, items)
        clips = [
            clip for mel, frames in zip(mels, clip_frames) for clip in mel.split(frames)
        ]
        expected = projector(torch.cat(ours.encode_clips(clips)))
    # One row per 8 frames of each clip, rounded up.
    assert out.shape[0] == 10 + 7 + 5
    assert torch.equal(out, expected)


def _mtp_omni_config() -> NemotronHOmniConfig:
    """A small Nemotron-H Omni config whose language model has one MTP layer."""
    llm = dict(
        hybrid_override_pattern="M*E",
        hidden_size=256,
        mamba_num_heads=8,
        mamba_head_dim=4,
        n_groups=2,
        ssm_state_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        moe_shared_expert_intermediate_size=256,
    )
    return NemotronHOmniConfig(llm_config=llm)


def test_mtp_draft_reads_the_language_model_mtp_tensors():
    from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype

    with torch.device(DEVICE), set_default_torch_dtype(torch.bfloat16):
        draft = nemotron_omni.NemotronH_Omni_Reasoning_V3NextN(
            _mtp_omni_config(), Mapping(rank=0, world_size=1)
        )
    names = [
        "language_model.mtp.layers.0.enorm.weight",
        "language_model.backbone.norm_f.weight",
        "vision_model.radio_model.model.blocks.0.norm1.weight",
    ]
    assert [draft.checkpoint_weight_name_filter(n) for n in names] == [
        True,
        False,
        False,
    ]
    draft.load_weights([(name, torch.full((256,), 2.0)) for name in names])
    assert torch.all(draft.enorm.weight == 2)


def test_kv_cache_scales_reach_the_language_model_and_its_mtp_head():
    from tokenspeed_kernel.ops.attention.prologue import FP8TensorScales

    from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype

    config = _mtp_omni_config()
    mapping = Mapping(rank=0, world_size=1)
    with torch.device(DEVICE), set_default_torch_dtype(torch.bfloat16):
        target = nemotron_omni.NemotronH_Omni_Reasoning_V3(
            config,
            mapping,
            None,
            is_multimodal_active=False,
            mm_attention_backend=None,
        )
        draft = nemotron_omni.NemotronH_Omni_Reasoning_V3NextN(config, mapping)
    scales = {
        "language_model.backbone.layers.1.mixer.k_proj.k_scale": 0.25,
        "language_model.backbone.layers.1.mixer.v_proj.v_scale": 0.5,
        "language_model.mtp.layers.0.mixer.k_proj.k_scale": 0.125,
        "language_model.mtp.layers.0.mixer.v_proj.v_scale": 0.0625,
    }
    scales = {name: torch.tensor([s]) for name, s in scales.items()}
    target.load_kv_cache_scales(scales)
    draft.load_kv_cache_scales(scales)
    assert target.model.layers[1].mixer.attn.fp8_kv_scales == FP8TensorScales(0.25, 0.5)
    assert draft.layers[0].mixer.attn.fp8_kv_scales == FP8TensorScales(0.125, 0.0625)


def test_checkpoint_tensors_route_to_their_modules():
    calls = []

    def recorder(owner):
        return SimpleNamespace(load_weight=lambda name, _: calls.append((owner, name)))

    model = SimpleNamespace(
        visual=recorder("visual"),
        vision_final_norm=torch.nn.LayerNorm(1),
        vision_projector=recorder("vision_projector"),
        audio_tower=recorder("audio_tower"),
        audio_projector=recorder("audio_projector"),
    )
    tensor = torch.full((1,), 2.0)
    names = [
        "language_model.backbone.layers.0.mixer.in_proj.weight",
        "vision_model.radio_model.model.blocks.3.attn.qkv.weight",
        "mlp1.3.weight",
        "sound_encoder.encoder.layers.1.conv.norm.running_var",
        "sound_projection.linear1.weight",
        "vision_model.radio_model.input_conditioner.norm_mean",
        "sound_encoder.encoder.feature_extractor.featurizer.fb",
        "vision_model.summary_idxs",
        "vision_projector.vision_final_layernorm.bias",
    ]
    route = nemotron_omni.NemotronH_Nano_Omni_Reasoning_V3._route_encoder_weights
    language = [name for name, _ in route(model, [(n, tensor) for n in names])]
    assert language == ["backbone.layers.0.mixer.in_proj.weight"]
    assert calls == [
        ("visual", "blocks.3.attn.qkv.weight"),
        ("vision_projector", "linear2.weight"),
        ("audio_tower", "layers.1.conv.norm.running_var"),
        ("audio_projector", "linear1.weight"),
    ]
    assert model.vision_final_norm.bias.item() == 2.0
    with pytest.raises(KeyError):
        list(route(model, [("unknown.weight", tensor)]))
    # Without MTP layers the model builds no vision LayerNorm to load it into.
    model.vision_final_norm = None
    with pytest.raises(KeyError):
        list(route(model, [("vision_projector.vision_final_layernorm.bias", tensor)]))
    # With the encoders disabled their tensors are skipped, not unknown names.
    text_only = SimpleNamespace(visual=None)
    language = [name for name, _ in route(text_only, [(n, tensor) for n in names])]
    assert language == ["backbone.layers.0.mixer.in_proj.weight"]
    with pytest.raises(KeyError):
        list(route(text_only, [("unknown.weight", tensor)]))
