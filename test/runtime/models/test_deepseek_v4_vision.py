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

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.configs.deepseek_v4_config import DeepseekV4Config
from tokenspeed.runtime.models.deepseek_v4_vision import (
    IMAGE,
    DeepseekV4VisionAligner,
    DeepseekV4VisionAttention,
    DeepseekV4VisionMetadataError,
    DeepseekV4VisionRowCountError,
    DeepseekV4VisionTransformer,
    apply_vision_rotary,
    build_dsv4_vision_forward,
    build_image_block,
    encode_image_items,
    get_vision_cos_sin,
    merge_image_block,
)

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "deepseek_v4_vision"
PATCH_ROOT = (
    Path(os.environ["DSV4_VISION_FIXTURE_ROOT"]) / "patches"
    if "DSV4_VISION_FIXTURE_ROOT" in os.environ
    else None
)


def _small_config(**overrides):
    values = {
        "vision_patch_size": 2,
        "vision_dim": 8,
        "vision_n_heads": 2,
        "vision_inter_dim": 12,
        "vision_n_layers": 2,
        "vision_rope_theta": 10000.0,
        "vision_downsample_ratio": 2,
        "hidden_size": 16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _item(
    start: int,
    end: int,
    *,
    n_vit_h: int = 2,
    n_vit_w: int = 2,
    n_llm_h: int = 1,
    n_llm_w: int = 1,
    compress_pad: int | None = None,
    feature: torch.Tensor | None = None,
):
    if compress_pad is None:
        compress_pad = 3 - start % 4
    return SimpleNamespace(
        modality=SimpleNamespace(name="IMAGE"),
        offsets=[(start, end)],
        feature=(
            feature if feature is not None else torch.zeros(n_vit_h * n_vit_w, 3, 2, 2)
        ),
        model_specific_data={
            "n_vit_h": torch.tensor([n_vit_h], dtype=torch.uint32),
            "n_vit_w": torch.tensor([n_vit_w], dtype=torch.uint32),
            "n_llm_h": torch.tensor([n_llm_h], dtype=torch.uint32),
            "n_llm_w": torch.tensor([n_llm_w], dtype=torch.uint32),
            "dsv4_compress_pad": torch.tensor([compress_pad], dtype=torch.uint32),
        },
    )


def _mm_context(items_by_request, prefixes, lengths):
    return SimpleNamespace(
        mm_inputs=[
            None if items is None else SimpleNamespace(mm_items=items)
            for items in items_by_request
        ],
        extend_prefix_lens=prefixes,
        extend_seq_lens=lengths,
    )


def test_config_publishes_official_vision_defaults():
    config = DeepseekV4Config()
    assert config.vision_n_layers == 0
    assert config.vision_dim == 1024
    assert config.vision_n_heads == 16
    assert config.vision_inter_dim == 2816
    assert config.vision_patch_size == 14
    assert config.vision_rope_theta == 10000.0
    assert config.vision_downsample_ratio == 3
    assert config.vision_max_n_token == 384
    assert config.vision_min_pixels == 147456
    assert config.vision_max_wh_ratio == 8


def test_all_fixture_layouts_match_every_prompt_residue():
    fixture = json.loads((FIXTURE_ROOT / "layout.json").read_text())
    preprocessing = json.loads((FIXTURE_ROOT / "preprocessing.json").read_text())[
        "fixtures"
    ]
    assert set(fixture["fixtures"]) == {f"F{index}" for index in range(1, 12)}
    for case_name, case in fixture["fixtures"].items():
        for residue in range(4):
            expected = case[str(residue)]
            n_llm_h = preprocessing[case_name]["n_llm_h"]
            n_llm_w = preprocessing[case_name]["n_llm_w"]
            types, permutation = build_image_block(n_llm_h, n_llm_w, residue)
            assert types.tolist() == expected["types"]
            assert permutation.tolist() == expected["perm"]
            assert types.numel() == expected["block_length"]
            assert types.numel() - expected["num_tokens"] == expected["compress_pad"]


def test_all_transported_bf16_patch_payloads_round_trip_bit_exact():
    if PATCH_ROOT is None:
        pytest.skip("set DSV4_VISION_FIXTURE_ROOT for external patch fixtures")
    preprocessing = json.loads((FIXTURE_ROOT / "preprocessing.json").read_text())[
        "fixtures"
    ]
    for fixture_id, metadata in preprocessing.items():
        raw = (PATCH_ROOT / f"{fixture_id}.bf16.bin").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == metadata["patch_sha256"]
        expected_elements = 1
        for extent in metadata["patch_shape"]:
            expected_elements *= extent
        assert len(raw) == expected_elements * 2
        transported = torch.frombuffer(bytearray(raw), dtype=torch.uint16)
        assert torch.equal(
            transported.view(torch.bfloat16).view(torch.uint16), transported
        )


def test_rope_tables_and_rotary_match_canonical_fp32_construction():
    n_h, n_w, dim, theta = 3, 5, 4, 10000.0
    cos, sin = get_vision_cos_sin(n_h, n_w, dim, theta)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = (
        torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    ).flatten(1)
    assert cos.dtype == sin.dtype == torch.float32
    assert torch.equal(cos, freqs.cos().unsqueeze(1))
    assert torch.equal(sin, freqs.sin().unsqueeze(1))

    source = torch.arange(n_h * n_w * 4 * dim, dtype=torch.float32).reshape(
        n_h * n_w, 2, 2 * dim
    )
    x1, x2 = source.chunk(2, dim=-1)
    expected = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    assert torch.equal(apply_vision_rotary(source, cos, sin), expected)


def test_attention_uses_native_rank_three_sdpa(monkeypatch):
    attention = DeepseekV4VisionAttention(_small_config())
    calls = []

    def fake_sdpa(q, k, v):
        calls.append((q.shape, k.shape, v.shape))
        return v

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    hidden = torch.randn(7, 8)
    cos, sin = get_vision_cos_sin(1, 7, 2, 10000.0)
    output = attention(hidden, cos, sin)
    assert output.shape == (7, 8)
    assert calls == [
        (torch.Size([2, 7, 4]),) * 3,
    ]


def test_aligner_uses_chw_unfold_and_exact_gelu():
    aligner = DeepseekV4VisionAligner(
        _small_config(vision_dim=1, vision_downsample_ratio=2, hidden_size=4)
    )
    with torch.no_grad():
        aligner.w1.weight.copy_(torch.eye(4))
        aligner.w1.bias.zero_()
        aligner.w2.weight.copy_(torch.eye(4))
        aligner.w2.bias.zero_()
    hidden = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    expected_window = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = F.gelu(expected_window, approximate="none")
    assert torch.equal(aligner(hidden, 2, 2), expected)
    assert not torch.equal(
        aligner(hidden, 2, 2), F.gelu(expected_window, approximate="tanh")
    )


def test_official_32_block_parameter_scope_and_large_shapes_on_meta():
    config = _small_config(
        vision_patch_size=14,
        vision_dim=1024,
        vision_n_heads=16,
        vision_inter_dim=2816,
        vision_n_layers=32,
        vision_downsample_ratio=3,
        hidden_size=4096,
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            vision = DeepseekV4VisionTransformer(config)
            aligner = DeepseekV4VisionAligner(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    params = dict(vision.named_parameters())
    assert len(vision.blocks) == 32
    assert len(params) == 259
    assert params["patch_embed.proj.weight"].shape == (1024, 588)
    assert params["blocks.0.attn.wqkv.weight"].shape == (3072, 1024)
    assert params["blocks.31.attn.wo.weight"].shape == (1024, 1024)
    assert params["blocks.31.mlp.w1.weight"].shape == (5632, 1024)
    assert params["blocks.31.mlp.w2.weight"].shape == (1024, 2816)
    assert aligner.w1.weight.shape == (4096, 9216)
    assert aligner.w2.weight.shape == (4096, 4096)
    assert all(
        parameter.dtype == torch.bfloat16
        for name, parameter in params.items()
        if name.endswith("norm1.weight")
        or name.endswith("norm2.weight")
        or name == "norm.weight"
    )


def test_merge_block_places_sentinels_and_permuted_rows():
    types, permutation = build_image_block(2, 2, 0)
    aligned = torch.arange(4, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    sentinels = tuple(
        torch.full((3,), value, dtype=torch.float32)
        for value in (10.0, 20.0, 30.0, 40.0)
    )
    block = merge_image_block(
        aligned,
        types,
        permutation,
        sentinels,
        placeholder_count=types.numel(),
    )
    assert torch.equal(block[types == IMAGE], aligned[permutation])
    assert torch.equal(block[types == 0][0], sentinels[0])
    assert torch.equal(block[types == 4][0], sentinels[1])
    assert torch.equal(block[types == 3][0], sentinels[2])
    assert torch.equal(block[types == 1][0], sentinels[3])


def test_row_count_mismatch_is_named_before_scatter():
    types, permutation = build_image_block(1, 1, 0)
    sentinels = tuple(torch.zeros(2) for _ in range(4))
    with pytest.raises(DeepseekV4VisionRowCountError, match="placeholder-token"):
        merge_image_block(
            torch.zeros(permutation.numel(), 2),
            types,
            permutation,
            sentinels,
            placeholder_count=types.numel() - 1,
        )


class _IdentityVision(nn.Module):
    def forward(self, patches, n_h, n_w):
        assert patches.shape[0] == n_h * n_w
        return patches[:, :1, 0, 0]


class _OneRowAligner(nn.Module):
    def forward(self, hidden, n_h, n_w):
        del n_h, n_w
        return hidden[:1].repeat(1, 2)


def test_encoder_cross_checks_all_three_compress_pad_sources():
    types, _ = build_image_block(1, 1, 0)
    good = _item(0, types.numel() - 1)
    sentinels = tuple(torch.zeros(2) for _ in range(4))
    assert encode_image_items(
        [good], _IdentityVision(), _OneRowAligner(), sentinels
    ).shape == (types.numel(), 2)

    for transmitted, end_delta in ((2, 0), (3, -1)):
        bad = _item(0, types.numel() - 1 + end_delta, compress_pad=transmitted)
        with pytest.raises(
            DeepseekV4VisionMetadataError, match="compress_pad mismatch"
        ):
            encode_image_items([bad], _IdentityVision(), _OneRowAligner(), sentinels)


def test_forward_payload_uses_absolute_atomic_and_visibility_spans():
    image = _item(5, 10)
    context = _mm_context([[image], [image]], [0], [12])
    payload = build_dsv4_vision_forward(
        context,
        num_tokens=13,
        device="cpu",
        max_image_tokens=384,
    )
    assert payload is not None and payload.intersects_span
    assert payload.atomic_spans == [[(5, 10)], [(5, 10)]]
    assert payload.visibility_spans == [[(7, 10)], [(7, 10)]]
    assert payload.atomic_spans_in_chunk == [[(5, 10)], []]
    assert payload.visibility_spans_in_chunk == [[(7, 10)], []]
    assert payload.image_mask.tolist() == [False] * 5 + [True] * 6 + [False] * 2
    assert payload.left.tolist()[5:11] == [0, 0, 0, 1, 2, 3]
    assert payload.right.tolist()[5:11] == [0, 0, 3, 2, 1, 0]
    assert not payload.image_mask[-1]
    assert payload.left[-1] == payload.right[-1] == 0


def test_forward_payload_keeps_equal_media_offsets_and_mixed_decode_rows_distinct():
    shared_feature = torch.zeros(4, 3, 2, 2)
    first = _item(2, 6, feature=shared_feature)
    second = _item(9, 13, feature=shared_feature)
    context = _mm_context([[first, second], None], [0], [14])
    payload = build_dsv4_vision_forward(
        context,
        num_tokens=16,
        device="cpu",
        max_image_tokens=384,
    )
    assert payload is not None
    assert payload.atomic_spans == [[(2, 6), (9, 13)], []]
    assert payload.atomic_spans_in_chunk == [[(2, 6), (9, 13)], []]
    assert payload.image_mask[:14].tolist() == (
        [False] * 2 + [True] * 5 + [False] * 2 + [True] * 5
    )
    assert not payload.image_mask[14:].any()
    assert not payload.left[14:].any()
    assert not payload.right[14:].any()


def test_single_three_chunk_two_image_and_mixed_payload_corpus():
    image = _item(5, 20)
    three_chunks = []
    for prefix, length in ((0, 8), (8, 8), (16, 8)):
        payload = build_dsv4_vision_forward(
            _mm_context([[image]], [prefix], [length]),
            num_tokens=length,
            device="cpu",
            max_image_tokens=384,
        )
        three_chunks.append(payload is not None and payload.intersects_span)
    assert three_chunks == [True, True, True]

    two_images = build_dsv4_vision_forward(
        _mm_context([[_item(1, 5), _item(7, 11)]], [0], [12]),
        num_tokens=12,
        device="cpu",
        max_image_tokens=384,
    )
    assert two_images is not None
    assert two_images.atomic_spans_in_chunk == [[(1, 5), (7, 11)]]

    single = build_dsv4_vision_forward(
        _mm_context([[_item(1, 5)]], [0], [6]),
        num_tokens=6,
        device="cpu",
        max_image_tokens=384,
    )
    mixed = build_dsv4_vision_forward(
        _mm_context([[_item(1, 5)], None], [0], [6]),
        num_tokens=7,
        device="cpu",
        max_image_tokens=384,
    )
    assert single is not None and single.intersects_span
    assert mixed is not None and mixed.intersects_span
    assert not mixed.image_mask[-1]


def test_forward_payload_is_none_for_all_false_later_chunk():
    image = _item(5, 10)
    context = _mm_context([[image]], [11], [4])
    assert (
        build_dsv4_vision_forward(
            context,
            num_tokens=4,
            device="cpu",
            max_image_tokens=384,
        )
        is None
    )


def test_forward_payload_never_accepts_or_inspects_input_ids():
    assert "input_ids" not in inspect.signature(build_dsv4_vision_forward).parameters
    context = _mm_context([[_item(4, 8)]], [0], [9])
    first = build_dsv4_vision_forward(
        context, num_tokens=9, device="cpu", max_image_tokens=384
    )
    second = build_dsv4_vision_forward(
        context, num_tokens=9, device="cpu", max_image_tokens=384
    )
    assert first is not None and second is not None
    assert torch.equal(first.image_mask, second.image_mask)
    assert torch.equal(first.left, second.left)
    assert torch.equal(first.right, second.right)
