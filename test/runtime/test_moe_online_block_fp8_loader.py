"""MoE expert loading with online (bf16 checkpoint) block-scaled FP8 quantization."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.moe.weights.loaders import (
    copy_expert_shard,
    load_w2,
    load_w13,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

BLOCK = (128, 128)


def _dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    block_n, block_k = BLOCK
    out = q.float()
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            rows = slice(i * block_n, min((i + 1) * block_n, q.shape[0]))
            cols = slice(j * block_k, min((j + 1) * block_k, q.shape[1]))
            out[rows, cols] *= scales[i, j]
    return out


def test_copy_expert_shard_quantizes_unquantized_source() -> None:
    n, k = 256, 256
    src = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    dst = torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones(n // 128, k // 128, device="cuda", dtype=torch.float32)

    copy_expert_shard(dst, src, scales, BLOCK)

    assert not torch.allclose(scales, torch.ones_like(scales))
    cosine = torch.nn.functional.cosine_similarity(
        _dequantize(dst, scales).flatten(), src.float().flatten(), dim=0
    )
    assert cosine > 0.99


def test_copy_expert_shard_plain_copy_without_scales() -> None:
    """Without a scale destination the copy must stay byte-for-byte as before."""
    src = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    dst = torch.zeros(8, 16, device="cuda", dtype=torch.bfloat16)

    copy_expert_shard(dst, src)

    torch.testing.assert_close(dst, src)


def test_copy_expert_shard_skips_quantization_for_fp8_source() -> None:
    """A serialized FP8 checkpoint is copied through, leaving scales untouched."""
    src = (torch.randn(256, 256, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    dst = torch.zeros(256, 256, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones(2, 2, device="cuda", dtype=torch.float32)

    copy_expert_shard(dst, src, scales, BLOCK)

    torch.testing.assert_close(dst.float(), src.float())
    torch.testing.assert_close(scales, torch.ones_like(scales))


@pytest.mark.parametrize("shard_id", ["w1", "w3"])
def test_load_w13_quantizes_into_its_half(shard_id: str) -> None:
    """w1 and w3 occupy halves of w13, so each must scale only its own blocks."""
    ispp, hidden = 256, 256
    expert_data = torch.zeros(
        2 * ispp, hidden, device="cuda", dtype=torch.float8_e4m3fn
    )
    expert_scale = torch.ones(
        2 * ispp // 128, hidden // 128, device="cuda", dtype=torch.float32
    )
    loaded = torch.randn(ispp, hidden, device="cuda", dtype=torch.bfloat16) * 0.05

    load_w13(
        expert_data,
        loaded,
        shard_id,
        0,
        tp_rank=0,
        is_bias=False,
        use_presharded_weights=True,
        do_transpose=False,
        expert_scale=expert_scale,
        block_shape=BLOCK,
    )

    start = ispp if shard_id == "w3" else 0
    written = expert_data.narrow(0, start, ispp)
    written_scale = expert_scale.narrow(0, start // 128, ispp // 128)
    cosine = torch.nn.functional.cosine_similarity(
        _dequantize(written, written_scale).flatten(), loaded.float().flatten(), dim=0
    )
    assert cosine > 0.99

    # The other half must be untouched.
    other_start = 0 if shard_id == "w3" else ispp
    assert torch.all(expert_data.narrow(0, other_start, ispp).float() == 0)
    other_scale = expert_scale.narrow(0, other_start // 128, ispp // 128)
    torch.testing.assert_close(other_scale, torch.ones_like(other_scale))


@pytest.mark.parametrize("shard_id", ["w1", "w3"])
def test_load_w13_quantizes_unaligned_half(shard_id: str) -> None:
    """An intermediate size below block_n still scales both halves.

    moe_intermediate_size=512 at tp8 yields ispp=64, so w3 starts at an
    element offset that is not a multiple of block_n while the scale buffer
    still reserves one rounded-up block per half.
    """
    ispp, hidden = 64, 256
    blocks_per_half = 1
    expert_data = torch.zeros(
        2 * ispp, hidden, device="cuda", dtype=torch.float8_e4m3fn
    )
    expert_scale = torch.ones(
        2 * blocks_per_half, hidden // 128, device="cuda", dtype=torch.float32
    )
    loaded = torch.randn(ispp, hidden, device="cuda", dtype=torch.bfloat16) * 0.05

    load_w13(
        expert_data,
        loaded,
        shard_id,
        0,
        tp_rank=0,
        is_bias=False,
        use_presharded_weights=True,
        do_transpose=False,
        expert_scale=expert_scale,
        block_shape=BLOCK,
    )

    half = 1 if shard_id == "w3" else 0
    written = expert_data.narrow(0, half * ispp, ispp)
    written_scale = expert_scale.narrow(0, half * blocks_per_half, blocks_per_half)
    # A plain fp8 cast would leave the scale at one and lose the small values.
    assert not torch.allclose(written_scale, torch.ones_like(written_scale))
    cosine = torch.nn.functional.cosine_similarity(
        _dequantize(written, written_scale).flatten(), loaded.float().flatten(), dim=0
    )
    assert cosine > 0.99

    other = 1 - half
    assert torch.all(expert_data.narrow(0, other * ispp, ispp).float() == 0)
    other_scale = expert_scale.narrow(0, other * blocks_per_half, blocks_per_half)
    torch.testing.assert_close(other_scale, torch.ones_like(other_scale))


def test_load_w2_quantizes_shard() -> None:
    hidden, ispp = 256, 256
    expert_data = torch.zeros(hidden, ispp, device="cuda", dtype=torch.float8_e4m3fn)
    expert_scale = torch.ones(
        hidden // 128, ispp // 128, device="cuda", dtype=torch.float32
    )
    loaded = torch.randn(hidden, ispp, device="cuda", dtype=torch.bfloat16) * 0.05

    load_w2(
        expert_data,
        loaded,
        "w2",
        1,
        tp_rank=0,
        is_bias=False,
        use_presharded_weights=True,
        do_transpose=False,
        expert_scale=expert_scale,
        block_shape=BLOCK,
    )

    cosine = torch.nn.functional.cosine_similarity(
        _dequantize(expert_data, expert_scale).flatten(),
        loaded.float().flatten(),
        dim=0,
    )
    assert cosine > 0.99


@pytest.mark.parametrize(
    ("tp_rank", "real_blocks"),
    [(0, 2), (1, 2), (2, 1), (3, 0)],
)
def test_load_w13_quantizes_after_global_padding_tp_shard(
    tp_rank: int,
    real_blocks: int,
) -> None:
    """Online FP8 quantization uses the same globally padded TP layout."""

    block_n, block_k = BLOCK
    hidden = 2 * block_k
    local_ispp = 2 * block_n
    real_rows = real_blocks * block_n
    loaded = (
        torch.randn(5 * block_n, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    )
    expert_data = torch.zeros(
        2 * local_ispp, hidden, device="cuda", dtype=torch.float8_e4m3fn
    )
    expert_scale = torch.ones(
        2 * local_ispp // block_n,
        hidden // block_k,
        device="cuda",
        dtype=torch.float32,
    )

    load_w13(
        expert_data,
        loaded,
        "w1",
        0,
        tp_rank=tp_rank,
        is_bias=False,
        use_presharded_weights=False,
        do_transpose=False,
        tp_size=4,
        expert_scale=expert_scale,
        block_shape=BLOCK,
    )

    written = expert_data[:local_ispp]
    written_scale = expert_scale[: local_ispp // block_n]
    if real_rows:
        source_start = tp_rank * local_ispp
        reconstructed = _dequantize(
            written[:real_rows],
            written_scale[:real_blocks],
        )
        cosine = torch.nn.functional.cosine_similarity(
            reconstructed.flatten(),
            loaded[source_start : source_start + real_rows].float().flatten(),
            dim=0,
        )
        assert cosine > 0.99
    assert torch.all(written[real_rows:].float() == 0)
    torch.testing.assert_close(
        written_scale[real_blocks:],
        torch.ones_like(written_scale[real_blocks:]),
    )


def test_load_w13_bias_is_not_quantized() -> None:
    """Bias is stored unquantized, so it must take the plain-copy path."""
    expert_data = torch.zeros(4, 512, device="cuda", dtype=torch.bfloat16)
    loaded = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    expert_scale = torch.ones(4, 4, device="cuda", dtype=torch.float32)

    load_w13(
        expert_data,
        loaded,
        "w1",
        0,
        tp_rank=0,
        is_bias=True,
        use_presharded_weights=True,
        do_transpose=False,
        expert_scale=expert_scale,
        block_shape=BLOCK,
    )

    torch.testing.assert_close(expert_data[:, :256], loaded.expand(4, 256))
    torch.testing.assert_close(expert_scale, torch.ones_like(expert_scale))
