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
    est_time=60, suite="runtime-1gpu", disabled_on_runners=["amd-*", "h100-*"]
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from transformers import ParakeetEncoderConfig  # noqa: E402
from transformers.models.parakeet.modeling_parakeet import (  # noqa: E402
    ParakeetEncoder as HFParakeetEncoder,
)

from tokenspeed.runtime.configs.nemotron_omni_config import (  # noqa: E402
    ParakeetAudioConfig,
    RadioVisionConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.models import nemotron_omni  # noqa: E402
from tokenspeed.runtime.models.parakeet import ParakeetEncoder  # noqa: E402
from tokenspeed.runtime.models.radio import RadioVisionModel  # noqa: E402

DEVICE = "cuda"
PATCH = 16
GRID = 8  # position-embedding grid side


def _radio(final_norm: bool) -> RadioVisionModel:
    config = RadioVisionConfig(
        args={
            "model": "vit_base_patch16_224",
            "cpe_max_size": GRID * PATCH,
            "cls_token_per_teacher": True,
            "teachers": [{"name": "clip"}, {"name": "siglip"}, {"name": "clip"}],
            "register_multiple": 4,
            "model_norm": final_norm,
        },
        patch_size=PATCH,
        video_temporal_patch_size=2,
    )
    torch.manual_seed(0)
    model = RadioVisionModel(
        config,
        Mapping(rank=0, world_size=1),
        final_norm=config.final_norm,
        mm_attention_backend=None,
        prefix="visual",
    )
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.05)
    return model.to(DEVICE, torch.bfloat16)


def _reference_vit(model: RadioVisionModel, embedded: torch.Tensor) -> torch.Tensor:
    """Pre-norm ViT over one embedded sequence in fp32, prefix tokens dropped."""
    heads = model.blocks[0].attn.num_attention_heads_per_partition
    x = torch.cat([model.cls_token.float(), embedded])
    for block in model.blocks:
        h = F.layer_norm(
            x, x.shape[-1:], block.norm1.weight.float(), block.norm1.bias.float(), 1e-6
        )
        qkv = F.linear(
            h, block.attn.qkv_proj.weight.float(), block.attn.qkv_proj.bias.float()
        )
        q, k, v = (
            t.unflatten(-1, (heads, -1)).transpose(0, 1) for t in qkv.chunk(3, -1)
        )
        attn = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).flatten(1)
        x = x + F.linear(
            attn, block.attn.proj.weight.float(), block.attn.proj.bias.float()
        )
        h = F.layer_norm(
            x, x.shape[-1:], block.norm2.weight.float(), block.norm2.bias.float(), 1e-6
        )
        mlp = block.mlp
        x = x + F.linear(
            F.gelu(F.linear(h, mlp.fc1.weight.float(), mlp.fc1.bias.float())),
            mlp.fc2.weight.float(),
            mlp.fc2.bias.float(),
        )
    if model.norm is not None:
        x = F.layer_norm(
            x, x.shape[-1:], model.norm.weight.float(), model.norm.bias.float(), 1e-6
        )
    return x[model.num_prefix_tokens :]


def _reference_position(model: RadioVisionModel, rows: int, cols: int) -> torch.Tensor:
    grid = model.pos_embed.float().view(1, GRID, GRID, -1).permute(0, 3, 1, 2)
    side = max(rows, cols)
    grid = F.interpolate(grid, size=(side, side), mode="bilinear", align_corners=False)
    return grid[0, :, :rows, :cols].flatten(1).T.bfloat16().float()


def _assert_close_bf16(out: torch.Tensor, ref: torch.Tensor) -> None:
    # bf16 encoder against an fp32 reference on the same weights.
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


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
                F.linear(image.bfloat16().float(), model.embedder.weight.float())
                + _reference_position(model, rows, cols),
            )
            for image, (rows, cols) in zip(images, grids)
        ]
    _assert_close_bf16(out, torch.cat(refs))


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
                    torch.cat([pair[0], pair[1]], dim=-1).bfloat16().float(),
                    model.video_embedder.weight.float(),
                )
                + _reference_position(model, rows, cols),
            )
            for pair in tubelets
        ]
    _assert_close_bf16(out, torch.cat(refs))


def test_pixel_shuffle_orders_each_block_by_row_column_channel():
    rows, cols, channels = 4, 6, 3
    features = torch.arange(rows * cols * channels).view(rows * cols, channels)
    merged = nemotron_omni.pixel_shuffle(features, rows, cols)
    grid = features.view(rows, cols, channels)
    # Block (1, 2) covers rows 2-3 and columns 4-5.
    expected = torch.cat([grid[2, 4], grid[2, 5], grid[3, 4], grid[3, 5]])
    assert merged.shape == (rows * cols // 4, 4 * channels)
    assert torch.equal(merged[1 * (cols // 2) + 2], expected)


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


def test_checkpoint_tensors_route_to_their_modules():
    calls = []

    def recorder(owner):
        return SimpleNamespace(load_weight=lambda name, _: calls.append((owner, name)))

    model = SimpleNamespace(
        visual=recorder("visual"),
        vision_projector=recorder("vision_projector"),
        audio_tower=recorder("audio_tower"),
        audio_projector=recorder("audio_projector"),
    )
    tensor = torch.zeros(1)
    names = [
        "language_model.backbone.layers.0.mixer.in_proj.weight",
        "vision_model.radio_model.model.blocks.3.attn.qkv.weight",
        "mlp1.3.weight",
        "sound_encoder.encoder.layers.1.conv.norm.running_var",
        "sound_projection.linear1.weight",
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
    with pytest.raises(KeyError):
        list(route(model, [("unknown.weight", tensor)]))
