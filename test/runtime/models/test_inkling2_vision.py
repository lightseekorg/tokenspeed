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

"""PE patch geometry and exact CPU arithmetic."""

import pytest
import torch

from tokenspeed.runtime.configs.inkling2 import Inkling2VisionConfig
from tokenspeed.runtime.models.inkling2_vision import (
    Inkling2VisionAttention,
    Inkling2VisionEncoder,
    _position_embeddings,
    _rotary,
)


def _config() -> Inkling2VisionConfig:
    return Inkling2VisionConfig(
        temporal_patch_size=2,
        native_patch_size=2,
        adapter_merge_size=2,
        projection_hidden_dim=32,
        projector_bias=True,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        mlp_ratio=2.0,
        posemb_grid_size=4,
        layer_norm_eps=1e-5,
    )


def _encoder() -> Inkling2VisionEncoder:
    torch.manual_seed(7)
    encoder = Inkling2VisionEncoder(
        _config(), text_hidden_size=24, mm_attention_backend=None
    )
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            if name.endswith("gamma"):
                parameter.fill_(0.1)
            elif "ln_" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0, std=0.1)
    return encoder


def test_rotary_axes_and_class_token() -> None:
    values = torch.ones(3, 2, 8, dtype=torch.bfloat16)
    positions = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int32)
    q, k = _rotary(values, values, positions)
    torch.testing.assert_close(q, k, rtol=0, atol=0)
    torch.testing.assert_close(q[0], values[0], rtol=0, atol=0)
    torch.testing.assert_close(q[1, :, :4], values[1, :, :4], rtol=0, atol=0)
    torch.testing.assert_close(q[2, :, 4:], values[2, :, 4:], rtol=0, atol=0)
    angle = torch.tensor(1.0)
    expected = torch.stack(
        (angle.cos() - angle.sin(), angle.sin() + angle.cos())
    ).bfloat16()
    torch.testing.assert_close(q[2, 0, :2], expected, rtol=0, atol=0)


def test_position_embedding_uses_pixel_centers() -> None:
    grid = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4, 1)
    result = _position_embeddings(
        grid,
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 3]),
        torch.tensor([2, 2, 4]),
        torch.tensor([2, 2, 4]),
    )
    torch.testing.assert_close(
        result[:, 0], torch.tensor([2.5, 12.5, 3], dtype=torch.bfloat16), rtol=0, atol=0
    )


def test_native_patch_rgb_order_and_positions() -> None:
    encoder = _encoder()
    patches = (
        torch.arange(6 * 3 * 4 * 4, dtype=torch.float32).reshape(6, 3, 4, 4).bfloat16()
    )
    captured = []
    handle = encoder.backbone.embeddings.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].clone())
    )
    hidden, positions, lengths, cu = encoder._embed_images(
        patches, [[1, 2, 2], [1, 1, 2]]
    )
    handle.remove()
    expected = []
    start = 0
    for height, width in ((2, 2), (1, 2)):
        expected.append(torch.zeros(1, 12, dtype=torch.bfloat16))
        for y in range(height * 2):
            for x in range(width * 2):
                block = patches[start + (y // 2) * width + x // 2]
                expected.append(
                    block[
                        :, (y % 2) * 2 : (y % 2 + 1) * 2, (x % 2) * 2 : (x % 2 + 1) * 2
                    ]
                    .permute(1, 2, 0)
                    .reshape(1, 12)
                )
        start += height * width
    torch.testing.assert_close(captured[0], torch.cat(expected), rtol=0, atol=0)
    assert lengths == [17, 9]
    assert cu.tolist() == [0, 17, 26]
    assert positions[[0, 1, 4, 16, 17, 18, 25]].tolist() == [
        [0, 0],
        [1, 1],
        [1, 4],
        [4, 4],
        [0, 0],
        [1, 1],
        [2, 4],
    ]
    torch.testing.assert_close(hidden[0], hidden[17], rtol=0, atol=0)


def test_pool_preserves_spatial_neighborhood_order() -> None:
    hidden = torch.arange(26 * 16).reshape(26, 16)
    pooled = _encoder()._pool(hidden, [[1, 2, 2], [1, 1, 2]], [17, 9])
    groups = [
        [1, 2, 5, 6],
        [3, 4, 7, 8],
        [9, 10, 13, 14],
        [11, 12, 15, 16],
        [18, 19, 22, 23],
        [20, 21, 24, 25],
    ]
    torch.testing.assert_close(
        pooled,
        torch.stack([hidden[group].flatten() for group in groups]),
        rtol=0,
        atol=0,
    )


def test_attention_is_bidirectional_with_separate_image_segments() -> None:
    attention = Inkling2VisionAttention(width=8, heads=2, mm_attention_backend=None)
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.out_proj,
        ):
            projection.weight.copy_(torch.eye(8, dtype=torch.bfloat16))
            projection.bias.zero_()
    hidden = torch.arange(32).reshape(4, 8).bfloat16() / 32
    positions = torch.zeros(4, 2, dtype=torch.int32)
    result = attention(
        hidden, positions, [2, 2], torch.tensor([0, 2, 4], dtype=torch.int32)
    )
    parts = []
    for image in hidden.split(2):
        q = image.reshape(2, 2, 4).transpose(0, 1).unsqueeze(0)
        scores = (q.float() @ q.float().transpose(-1, -2)) * 0.5
        expected = (
            (scores.softmax(-1).to(q.dtype) @ q)
            .squeeze(0)
            .transpose(0, 1)
            .reshape(2, 8)
        )
        parts.append(expected)
    torch.testing.assert_close(result, torch.cat(parts), rtol=0.01, atol=0.004)
    changed = hidden.clone()
    changed[1] *= -2
    changed_result = attention(
        changed, positions, [2, 2], torch.tensor([0, 2, 4], dtype=torch.int32)
    )
    assert not torch.equal(result[0], changed_result[0])
    torch.testing.assert_close(result[2:], changed_result[2:], rtol=0, atol=0)


def test_forward_images_are_independent() -> None:
    encoder = _encoder()
    patches = torch.randn(3, 2, 4, 4, 3, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 1, 1], [1, 1, 2]], dtype=torch.int64)
    result = encoder(patches, grid)
    separate = torch.cat(
        (encoder(patches[:1], grid[:1]), encoder(patches[1:], grid[1:]))
    )
    torch.testing.assert_close(result, separate, rtol=0, atol=0)
    assert result.shape == (3, 24)
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()
    changed = patches.clone()
    changed[1:] += 3
    changed_result = encoder(changed, grid)
    torch.testing.assert_close(result[:1], changed_result[:1], rtol=0, atol=0)
    assert not torch.equal(result[1:], changed_result[1:])


def test_preprocess_temporal_mean_materializes_bfloat16() -> None:
    encoder = _encoder()
    patches = torch.randn(1, 2, 4, 4, 3, dtype=torch.bfloat16)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).bfloat16()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).bfloat16()
    expected = (2 * (patches * std + mean) - 1).mean(1).permute(0, 3, 1, 2)
    torch.testing.assert_close(
        encoder._preprocess_patches(patches), expected, rtol=0, atol=0
    )


def test_registered_vision_backend_preserves_rectangular_image_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.layers.attention import mm_encoder_attention

    def reference_backend(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        bsz: int,
        seq_len: int,
        softmax_scale: float,
        max_seqlen: int,
    ) -> torch.Tensor:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        assert bsz == len(lengths) and seq_len == max_seqlen == max(lengths)
        # Per-image SDPA keeps the reference reduction shapes and BF16 rounding.
        outputs = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            output = torch.nn.functional.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)

    monkeypatch.setitem(
        mm_encoder_attention._BACKENDS, "test_segments", reference_backend
    )
    reference = _encoder()
    encoder = Inkling2VisionEncoder(
        _config(), text_hidden_size=24, mm_attention_backend="test_segments"
    )
    encoder.load_state_dict(reference.state_dict(), strict=True)
    patches = torch.randn(7, 2, 4, 4, 3, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 1, 1], [1, 2, 3]], dtype=torch.int64)
    with torch.inference_mode():
        expected = reference(patches, grid)
        actual = encoder(patches, grid)
        changed = patches.clone()
        changed[1:] += 3
        changed_result = encoder(changed, grid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(changed_result[:1], actual[:1], rtol=0, atol=0)
    assert not torch.equal(changed_result[1:], actual[1:])


@pytest.mark.parametrize("backend", ["missing_backend", "flashinfer_cudnn"])
def test_encoder_rejects_unavailable_backend_contracts(backend: str) -> None:
    with pytest.raises((ValueError, NotImplementedError)):
        Inkling2VisionEncoder(
            _config(), text_hidden_size=24, mm_attention_backend=backend
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_gpu_encoder_matches_cpu_for_multiple_rectangular_images() -> None:
    """Provisional rtol0.03/atol0.02 smoke gate, not pinned GPU bitwise parity."""
    cpu_encoder = _encoder().eval()
    gpu_encoder = (
        Inkling2VisionEncoder(_config(), text_hidden_size=24, mm_attention_backend=None)
        .cuda()
        .eval()
    )
    gpu_encoder.load_state_dict(cpu_encoder.state_dict(), strict=True)
    patches = torch.randn(7, 2, 4, 4, 3, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 1, 1], [1, 2, 3]], dtype=torch.int64)
    with torch.inference_mode():
        expected = cpu_encoder(patches, grid)
        actual = gpu_encoder(patches.cuda(), grid).cpu()
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
        separate = torch.cat(
            (
                gpu_encoder(patches[:1].cuda(), grid[:1]),
                gpu_encoder(patches[1:].cuda(), grid[1:]),
            )
        ).cpu()
        torch.testing.assert_close(actual, separate, rtol=0.03, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_gpu_encoder_does_not_wait_for_the_device() -> None:
    """Serving arms torch's sync-debug mode, and CI rejects any host sync."""
    encoder = _encoder().cuda().eval()
    patches = torch.randn(7, 2, 4, 4, 3, dtype=torch.bfloat16).cuda()
    grid = torch.tensor([[1, 1, 1], [1, 2, 3]], dtype=torch.int64)
    with torch.inference_mode():
        expected = encoder(patches, grid)
        torch.cuda.set_sync_debug_mode("error")
        try:
            actual = encoder(patches, grid)
        finally:
            torch.cuda.set_sync_debug_mode(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("num_hidden_layers", True),
        ("num_attention_heads", 3),
        ("layer_norm_eps", float("nan")),
        ("mlp_ratio", 0.1),
        ("projector_bias", 1),
    ],
)
def test_encoder_rejects_unsupported_geometry(name: str, value: object) -> None:
    config = _config()
    setattr(config, name, value)
    with pytest.raises(ValueError):
        Inkling2VisionEncoder(config, text_hidden_size=24, mm_attention_backend=None)
