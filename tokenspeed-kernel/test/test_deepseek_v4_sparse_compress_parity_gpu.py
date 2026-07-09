# Copyright (c) 2026 LightSeek Foundation

"""GPU parity for the sparse compress cache insert at 4 vs 16 warps.

The wide launch reorganizes the cross-thread softmax/sum reductions, so the
fp32 accumulation order can differ before quantization. This test runs the
production HCA shape (ratio=128, overlap=False) on identical inputs at both
warp counts and compares every WRITTEN row: scale bytes exactly, FP8 and
RoPE BF16 regions by decoded value (signed-zero encodings can legitimately
differ under reduction reorder; observed on GB200), each region separately. Token classes cover early-return (non-boundary), the first-window boundary
(start == 0), full history, and — via block_table_base_offsets — partially
masked windows crossing the base logical page.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.triton import deepseek_v4 as ops


def _is_sm100() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0) == (10, 0)
    )


# The 16-warp launch is gated to NVIDIA sm100 in production; this parity
# test forces both warp counts directly and is only meaningful (and only
# safe to run) on that target.
pytestmark = pytest.mark.skipif(not _is_sm100(), reason="requires NVIDIA sm100")

HEAD = ops.DEEPSEEK_V4_HEAD_DIM
RATIO = 128
STATE_BLOCK = 64
KV_BLOCK = 64
N_BLOCKS = 2048
TOKEN_STRIDE = ops.DEEPSEEK_V4_SWA_TOKEN_STRIDE
SCALE_DIM = ops.DEEPSEEK_V4_SWA_SCALE_DIM
NOPE = HEAD - ops.DEEPSEEK_V4_ROPE_DIM


def _run(m, num_warps, seed, base_offsets=None):
    dev = "cuda:0"
    torch.manual_seed(seed)
    width = 2 * HEAD
    state = (torch.randn(N_BLOCKS, STATE_BLOCK * width, device=dev) * 0.3).view(
        N_BLOCKS, -1
    )
    t2r = (torch.arange(m, device=dev, dtype=torch.int32) % 4).contiguous()
    # Mix token classes: early-return (non-boundary), boundary with history,
    # and the first full boundary window (window start == 0). Partially masked
    # windows are covered by the dedicated base-offsets test.
    pos = torch.arange(m, device=dev, dtype=torch.int32) + 4096
    pos[0::4] = RATIO - 1  # first full boundary window (start == 0)
    pos[1::4] = 8 * RATIO - 1  # boundary, full history
    slot = torch.arange(m, device=dev, dtype=torch.int32) + 7
    kv_slot = torch.arange(m, device=dev, dtype=torch.int32) + 3
    bt = torch.arange(4 * 256, device=dev, dtype=torch.int32).view(4, 256) % N_BLOCKS
    rms_w = torch.rand(HEAD, device=dev) + 0.5
    cs = torch.randn(65536, ops.DEEPSEEK_V4_ROPE_DIM, device=dev)
    kvc = torch.zeros(
        N_BLOCKS,
        KV_BLOCK * (TOKEN_STRIDE + SCALE_DIM),
        dtype=torch.uint8,
        device=dev,
    )
    ops._deepseek_v4_fused_sparse_compress_cache_kernel[(m,)](
        state,
        state.stride(0),
        width,
        t2r,
        pos,
        slot,
        bt,
        base_offsets,
        bt.stride(0),
        bt.shape[-1],
        STATE_BLOCK,
        rms_w,
        1e-6,
        cs,
        cs.stride(0),
        kvc,
        kv_slot,
        KV_BLOCK,
        HEAD_SIZE=HEAD,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(HEAD),
        STATE_WIDTH=HEAD,
        COMPRESS_RATIO=RATIO,
        OVERLAP=False,
        ROPE_HEAD_DIM=ops.DEEPSEEK_V4_ROPE_DIM,
        FP8_MAX=ops.DEEPSEEK_V4_FP8_MAX,
        QUANT_BLOCK=ops.DEEPSEEK_V4_FP8_QUANT_BLOCK,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=kvc.stride(0),
        num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return kvc


def _written_slots(m):
    # Mirror the kernel's write condition: compress-boundary position with a
    # non-negative state slot and kv slot (matches _run's input construction).
    pos = torch.arange(m, dtype=torch.int64) + 4096
    pos[0::4] = RATIO - 1
    pos[1::4] = 8 * RATIO - 1
    kv_slot = torch.arange(m, dtype=torch.int64) + 3
    boundary = (pos + 1) % RATIO == 0
    return kv_slot[boundary]


def _written_views(kvc, slots):
    blocks = kvc.view(N_BLOCKS, -1)
    rows_v, rows_s, rows_r = [], [], []
    for slot in slots.tolist():
        blk, off = slot // KV_BLOCK, slot % KV_BLOCK
        row = blocks[blk]
        vals = row[: KV_BLOCK * TOKEN_STRIDE].view(KV_BLOCK, TOKEN_STRIDE)[off]
        scale = row[KV_BLOCK * TOKEN_STRIDE :].view(KV_BLOCK, SCALE_DIM)[off]
        rows_v.append(vals[:NOPE])
        rows_r.append(vals[NOPE:TOKEN_STRIDE])
        rows_s.append(scale)
    return (
        torch.stack(rows_v),  # FP8 value bytes
        torch.stack(rows_s),  # encoded scale bytes
        torch.stack(rows_r),  # RoPE BF16 tail bytes
    )


@pytest.mark.parametrize("m", [16, 32, 64, 2048])
def test_sparse_compress_parity_4_vs_16_warps(m):
    slots = _written_slots(m)
    assert slots.numel() > 0, "test must exercise written rows"
    for seed in (0, 1, 2):
        a = _run(m, 4, seed)
        b = _run(m, 16, seed)
        fa, sa, ra = _written_views(a, slots)
        fb, sb, rb = _written_views(b, slots)
        # B200-measured behavior is byte-identical on every written row;
        # compare the three written regions separately so any future
        # divergence is attributable to values, scales, or the RoPE tail.
        assert torch.equal(sa, sb), f"M={m} seed={seed}: scale bytes differ"
        # FP8/BF16 regions: compare decoded VALUES, not raw bytes. Reduction
        # reorder can flip a signed zero (0x00 vs 0x80 encode the same 0.0;
        # observed on GB200 at M=2048), which is byte-different but
        # value-identical.
        da = _dequant_rows(fa, sa)
        db = _dequant_rows(fb, sb)
        if not torch.equal(da, db):
            # Reduction reorder can flip an FP8 rounding tie on isolated
            # values (observed on GB200: 1 value in ~4.6e5 differing by one
            # quantization step, 1.5e-5). Bound the fraction and magnitude
            # instead of requiring bit equality across schedulers.
            frac = (da != db).float().mean().item()
            err = (da - db).abs().max().item()
            rel = ((da - db).abs() / db.abs().clamp_min(1e-6)).max().item()
            assert frac < 1e-4, f"M={m} seed={seed}: mismatch fraction {frac:.2e}"
            assert err < 0.25, f"M={m} seed={seed}: dequant max abs {err:.6f}"
            assert rel < 0.25, f"M={m} seed={seed}: dequant max rel {rel:.4f}"
        va = ra.view(torch.bfloat16).float()
        vb = rb.view(torch.bfloat16).float()
        assert torch.equal(va, vb), f"M={m} seed={seed}: RoPE tail values differ"


def _dequant_rows(fp8_bytes, scale_bytes):
    fp8 = fp8_bytes.view(torch.float8_e4m3fn).float()
    exp = scale_bytes[:, : NOPE // ops.DEEPSEEK_V4_FP8_QUANT_BLOCK].float() - 127.0
    scale = torch.exp2(exp).repeat_interleave(ops.DEEPSEEK_V4_FP8_QUANT_BLOCK, dim=-1)
    return fp8 * scale


@pytest.mark.parametrize("m", [16, 64])
def test_sparse_compress_parity_partial_window_with_base_offsets(m):
    # Production passes state_base_logical_page; when the reduction window
    # crosses the base offset, leading rows resolve to table_idx < 0 and are
    # masked. The masked-softmax reduction is exactly where a warp-count
    # change reorganizes the computation, so compare 4 vs 16 warps here too.
    # pos[1::4] = 1023 boundaries: rows 896..959 fall below base page 15 and
    # are masked, rows 960..1023 stay valid -> a genuine partial window.
    base = torch.full((4,), 15, device="cuda:0", dtype=torch.int32)
    # Only the pos=1023 boundaries have a genuine partial window here; the
    # pos=127 boundaries fall entirely below base page 15 (all-masked window,
    # undefined softmax output) and are excluded from byte comparison.
    pos = torch.arange(m, dtype=torch.int64) + 4096
    pos[0::4] = RATIO - 1
    pos[1::4] = 8 * RATIO - 1
    kv_slot = torch.arange(m, dtype=torch.int64) + 3
    slots = kv_slot[((pos + 1) % RATIO == 0) & (pos == 8 * RATIO - 1)]
    assert slots.numel() > 0
    for seed in (0, 1):
        a = _run(m, 4, seed, base_offsets=base)
        b = _run(m, 16, seed, base_offsets=base)
        fa, sa, ra = _written_views(a, slots)
        fb, sb, rb = _written_views(b, slots)
        assert torch.equal(sa, sb), f"M={m} seed={seed}: scale bytes differ"
        da, db = _dequant_rows(fa, sa), _dequant_rows(fb, sb)
        assert (da != db).float().mean().item() < 1e-4 and (
            da - db
        ).abs().max().item() < 0.25, f"M={m} seed={seed}: FP8 values differ"
        assert torch.equal(
            ra.view(torch.bfloat16).float(), rb.view(torch.bfloat16).float()
        ), f"M={m} seed={seed}: RoPE tail values differ"
