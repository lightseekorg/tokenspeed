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

"""KDA merged qkvgb projection, FP8-resident (w8a8) mode.

Covers the FP8 buffer + segment-concatenated scale-grid loading of
``KimiKDAMergedProj`` (codes bitwise, pad rows zero, direct scale placement)
and the w8a8 blockscale GEMM branch of ``kimi3_qkvfab_projection`` against a
dequantized reference, including the flashinfer kernel-path pin. bf16 mode is
asserted structurally unchanged.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for the w8a8 GEMM path"
)

# Scaled-down K3 geometry with the same alignment properties: proj_local=512
# (4 x 128-blocks per q/k/v/g), head_dim=128, 4 local heads -> used rows
# 4*512 + 128 + 4 = 2180, padded to 2304 (18 x 128).
HIDDEN = 256
TP_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 128
PROJ = NUM_HEADS * HEAD_DIM  # 1024 global, 512 per rank
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_per_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    nb, kb = (n + 127) // 128, (k + 127) // 128
    padded = torch.zeros(nb * 128, kb * 128, dtype=torch.float32, device=w.device)
    padded[:n, :k] = w.float()
    blocks = padded.view(nb, 128, kb, 128)
    scales = (blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / _FP8_MAX).contiguous()
    q = (blocks / scales[:, None, :, None]).clamp(-_FP8_MAX, _FP8_MAX)
    return (
        q.view(nb * 128, kb * 128)[:n, :k].to(torch.float8_e4m3fn).contiguous(),
        scales,
    )


def _make_ckpt_segments(generator: torch.Generator) -> dict:
    """Checkpoint-shaped fp8 tensors: q/k/v/g [PROJ,H], f_a [128,H], b [8,H]."""
    ckpt = {}
    for name, rows in (
        ("q", PROJ),
        ("k", PROJ),
        ("v", PROJ),
        ("g", PROJ),
        ("f_a", HEAD_DIM),
        ("b", NUM_HEADS),
    ):
        w = (torch.randn(rows, HIDDEN, generator=generator) * 0.5).cuda()
        ckpt[name] = _quantize_per_block(w)
    return ckpt


def _build_fp8_merged(rank: int):
    from tokenspeed.runtime.models.kimi_k3 import KimiKDAMergedProj

    module = KimiKDAMergedProj(
        hidden_size=HIDDEN,
        proj=PROJ,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        tp_rank=rank,
        tp_size=TP_SIZE,
        fp8_block_quant=True,
    ).cuda()
    return module


def _load(module, ckpt) -> None:
    for name, (codes, scales) in ckpt.items():
        module.weight.weight_loader(module.weight, codes, name)
        module.weight_scale_inv.weight_loader(
            module.weight_scale_inv, scales, name
        )


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n, k = codes.shape
    sf = scales.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    return codes.float() * sf


def test_fp8_merged_loading_codes_scales_and_pad() -> None:
    torch.manual_seed(0)
    ckpt = _make_ckpt_segments(torch.Generator().manual_seed(20260818))
    for rank in range(TP_SIZE):
        module = _build_fp8_merged(rank)
        _load(module, ckpt)
        p = module.proj_local
        assert module.weight.dtype == torch.float8_e4m3fn  # FP8-resident
        assert module.weight.shape[0] % 128 == 0  # padded to the block grid
        # Codes land bitwise per segment (row-sharded except replicated f_a);
        # the scale grid is the direct concatenation of the ckpt grids.
        for name, rows in module._rows.items():
            codes, scales = ckpt[name]
            start = module._offsets[name]
            src = (
                codes
                if name == "f_a"
                else codes.narrow(0, rank * rows, rows)
            )
            assert torch.equal(
                module.weight.data[start : start + rows].view(torch.uint8),
                src.view(torch.uint8),
            ), name
            block_start = start // 128
            if name in ("f_a", "b"):
                expected_scale = scales[:1]
            else:
                nblocks = rows // 128
                expected_scale = scales.narrow(0, rank * nblocks, nblocks)
            assert torch.equal(
                module.weight_scale_inv.data[
                    block_start : block_start + expected_scale.shape[0]
                ],
                expected_scale,
            ), name
        # Pad rows carry zero codes -> exact-zero dequant under b's scale.
        assert torch.all(
            module.weight.data[module.used_rows :].view(torch.uint8) == 0
        )
        # Full-buffer dequant matches the per-segment manual dequant.
        dq = _dequant(module.weight.data, module.weight_scale_inv.data)
        for name, rows in module._rows.items():
            codes, scales = ckpt[name]
            start = module._offsets[name]
            src_codes = (
                codes if name == "f_a" else codes.narrow(0, rank * rows, rows)
            )
            if name in ("f_a", "b"):
                seg_ref = src_codes.float() * scales[:1].repeat_interleave(
                    128, 0
                )[:rows].repeat_interleave(128, 1)[:, :HIDDEN]
            else:
                nblocks = rows // 128
                seg_ref = _dequant(
                    src_codes, scales.narrow(0, rank * nblocks, nblocks)
                )
            assert torch.equal(dq[start : start + rows], seg_ref), name


def test_fp8_merged_rejects_bf16_refit_shard() -> None:
    module = _build_fp8_merged(rank=0)
    with pytest.raises(TypeError, match="bf16 refit"):
        module.weight.weight_loader(
            module.weight,
            torch.zeros(PROJ, HIDDEN, dtype=torch.bfloat16, device="cuda"),
            "q",
        )


def test_bf16_mode_unchanged() -> None:
    """mxfp4/bf16 checkpoints construct exactly the pre-FP8 module."""
    from tokenspeed.runtime.models.kimi_k3 import KimiKDAMergedProj

    module = KimiKDAMergedProj(
        hidden_size=HIDDEN,
        proj=PROJ,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        tp_rank=0,
        tp_size=TP_SIZE,
    )
    assert module.weight.dtype == torch.bfloat16
    used = module.used_rows
    assert module.weight.shape[0] == (used + 15) // 16 * 16  # 16-row align
    assert not hasattr(module, "weight_scale_inv")
    assert module.fp8_block_quant is False


def test_qkvfab_fp8_w8a8_matches_dequant_reference_and_pins_flashinfer() -> None:
    from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_qkvfab_projection

    from tokenspeed.runtime.layers.dense.fp8 import (
        has_flashinfer_fp8_blockscale,
        prepare_flashinfer_fp8_blockscale_weight_scales,
    )

    torch.manual_seed(1)
    ckpt = _make_ckpt_segments(torch.Generator().manual_seed(20260819))
    module = _build_fp8_merged(rank=0)
    _load(module, ckpt)
    n, k = module.weight.shape
    assert n % 128 == 0 and k % 128 == 0

    w_dq = _dequant(module.weight.data, module.weight_scale_inv.data)
    for m in (1, 32):
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.2
        ref32 = x.float() @ w_dq.t()
        out = kimi3_qkvfab_projection(
            x,
            module.weight,
            weight_scale=module.weight_scale_inv,
            enable_pdl=False,
        )
        torch.cuda.synchronize()
        assert out.shape == (m, n) and out.dtype == torch.bfloat16
        rel = ((out.float() - ref32).abs().amax() / ref32.abs().amax()).item()
        assert rel < 5e-2, f"M={m}: {rel=:.3e}"  # w8a8 activation-quant band
        # Pad rows produce exact zeros.
        assert torch.all(out[:, module.used_rows :] == 0)

    if has_flashinfer_fp8_blockscale is None or not has_flashinfer_fp8_blockscale():
        pytest.skip("flashinfer blockscale unavailable for the pin check")
    # The prepacked-scale path pins the flashinfer kernel via override: a
    # successful call IS the selection assertion (the override raises if the
    # kernel cannot serve the shape).
    prepacked = prepare_flashinfer_fp8_blockscale_weight_scales(
        module.weight_scale_inv.data
    )
    x = torch.randn(4, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.2
    out_pinned = kimi3_qkvfab_projection(
        x,
        module.weight,
        weight_scale=module.weight_scale_inv,
        prepacked_scales=prepacked,
        enable_pdl=False,
    )
    torch.cuda.synchronize()
    ref32 = x.float() @ w_dq.t()
    rel = ((out_pinned.float() - ref32).abs().amax() / ref32.abs().amax()).item()
    assert rel < 5e-2, f"pinned flashinfer path: {rel=:.3e}"
