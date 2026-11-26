"""Load-time block-FP8 -> MXFP4 expert requantization (--moe-requant-experts).

The recipe must match a PyTorch reference bit for bit, and every TP/EP layout
of GLM-5.3 Flash's expert shapes must hold the bytes of the whole-tensor
requantization at its slice (equal per-block digests), so TP8 MXFP4 and
Gluon Petit EP8 serve identical expert weights.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.quantization.fp8_block_to_mxfp4 import (
    block_digests,
    dequantize_fp8_block,
    requantize_fp8_block_to_mxfp4,
)

BLOCK = (128, 128)
HIDDEN, INTER, EXPERTS = 4096, 2048, 4  # GLM-5.3 Flash shapes, 4 experts
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Triton MXFP4 quantizer needs a GPU"
)


def _fp8_block(experts: int, n: int, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn((experts, n, k), generator=g) * 120).clamp(-448, 448)
    weight = weight.to(torch.float8_e4m3fn)
    # Arbitrary FP32 block scales, as in the checkpoint (amax / 448).
    scale = torch.rand((experts, n // 128, k // 128), generator=g) * 1e-4 + 1.5e-4
    return weight, scale


def _reference(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quark-even MXFP4 in PyTorch: amax rounded at mantissa bit 21, scale
    exponent floor(log2) - 2, E2M1 round-to-nearest-even, saturated at 6."""
    groups = values.reshape(*values.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1)
    bits = (amax.view(torch.int32) + 0x200000) & 0x7F800000
    exponent = ((bits >> 23) - 127 - 2).clamp(-127, 127)
    scaled = groups * torch.exp2(-exponent.to(torch.float32)).unsqueeze(-1)
    magnitude = scaled.abs().clamp(max=6.0)
    # Nearest E2M1 code, ties to the even code index.
    distance = (magnitude.unsqueeze(-1) - _E2M1.to(values.device)).abs()
    best = distance.min(-1, keepdim=True).values
    candidates = distance == best
    index = torch.arange(8, device=values.device)
    tie_even = candidates & (index % 2 == 0)
    code = torch.where(
        candidates.sum(-1) > 1, tie_even.int().argmax(-1), candidates.int().argmax(-1)
    )
    # The quantizer ORs the sign in unconditionally (negative zero is 0x8).
    code = code | torch.signbit(scaled).to(code.dtype) << 3
    code = code.to(torch.uint8).reshape(*values.shape)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed, (exponent + 127).to(torch.uint8)


@requires_gpu
def test_requantization_matches_reference_bitwise() -> None:
    weight, scale = _fp8_block(2, 256, 512, seed=1)
    codes, scales = requantize_fp8_block_to_mxfp4(weight.cuda(), scale.cuda(), BLOCK)
    ref_codes, ref_scales = _reference(dequantize_fp8_block(weight, scale, BLOCK))
    assert torch.equal(scales.cpu(), ref_scales)
    assert torch.equal(codes.cpu(), ref_codes)


@requires_gpu
def test_tp8_and_ep8_slices_hold_the_whole_requantization() -> None:
    w13, s13 = _fp8_block(EXPERTS, 2 * INTER, HIDDEN, seed=2)
    w2, s2 = _fp8_block(EXPERTS, HIDDEN, INTER, seed=3)
    full13 = requantize_fp8_block_to_mxfp4(w13.cuda(), s13.cuda(), BLOCK)
    full2 = requantize_fp8_block_to_mxfp4(w2.cuda(), s2.cuda(), BLOCK)
    ispp = INTER // 8
    ep_rows, tp_rows = [], []
    for rank in range(8):
        # TP8: gate and up rows [r*ispp, (r+1)*ispp) each, down columns likewise.
        rows = torch.cat(
            [
                torch.arange(rank * ispp, (rank + 1) * ispp),
                INTER + torch.arange(rank * ispp, (rank + 1) * ispp),
            ]
        )
        cols = slice(rank * ispp, (rank + 1) * ispp)
        b13 = slice(rank * ispp // 128, (rank + 1) * ispp // 128)
        t13 = requantize_fp8_block_to_mxfp4(
            w13[:, rows].cuda(),
            torch.cat([s13[:, b13], s13[:, INTER // 128 :][:, b13]], 1).cuda(),
            BLOCK,
        )
        t2 = requantize_fp8_block_to_mxfp4(
            w2[:, :, cols].cuda(), s2[:, :, b13].cuda(), BLOCK
        )
        assert torch.equal(t13[0].cpu(), full13[0][:, rows].cpu())
        assert torch.equal(
            t2[0].cpu(), full2[0][:, :, rank * ispp // 2 : (rank + 1) * ispp // 2].cpu()
        )
        tp_rows += block_digests(
            t13[0],
            t13[1],
            t2[0],
            t2[1],
            expert_start=0,
            intermediate_size=INTER,
            tp_rank=rank,
            tp_size=8,
        )
    # EP: whole experts per rank (one expert per rank here).
    for rank in range(EXPERTS):
        e = slice(rank, rank + 1)
        ep_rows += block_digests(
            full13[0][e],
            full13[1][e],
            full2[0][e],
            full2[1][e],
            expert_start=rank,
            intermediate_size=INTER,
            tp_rank=0,
            tp_size=1,
        )
    assert sorted(tp_rows) == sorted(ep_rows)
    assert len(ep_rows) == EXPERTS * 3 * 8


def test_block_digests_name_global_blocks() -> None:
    # TP rank 3 of 8 at intermediate 512 holds block 3 of gate, up and down.
    w13 = torch.zeros((1, 128, 16), dtype=torch.uint8)
    w13[:, 64:] = 1  # up rows differ from gate rows
    s13 = torch.zeros((1, 128, 1), dtype=torch.uint8)
    w2 = torch.zeros((1, 32, 32), dtype=torch.uint8)
    s2 = torch.zeros((1, 32, 2), dtype=torch.uint8)
    rows = block_digests(
        w13, s13, w2, s2, expert_start=5, intermediate_size=512, tp_rank=3, tp_size=8
    )
    assert [(e, p, b) for e, p, b, _ in rows] == [
        (5, "gate", 3),
        (5, "up", 3),
        (5, "down", 3),
    ]
    assert rows[0][3] != rows[1][3]
