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

"""GPU parity for sparse compress cache insert at 4 versus 16 warps."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.triton import dsv4 as ops


def _is_sm100() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0) == (10, 0)
    )


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="requires NVIDIA sm100")

HEAD = ops.DEEPSEEK_V4_HEAD_DIM
RATIO = 128
STATE_BLOCK = 64
KV_BLOCK = 64
N_BLOCKS = 2048
TOKEN_STRIDE = ops.DEEPSEEK_V4_SWA_TOKEN_STRIDE
SCALE_DIM = ops.DEEPSEEK_V4_SWA_SCALE_DIM
NOPE = HEAD - ops.DEEPSEEK_V4_ROPE_DIM
SENTINEL = 0xA5


def _run(m, num_warps, seed, base_offsets=None):
    dev = "cuda:0"
    torch.manual_seed(seed)
    width = 2 * HEAD
    state = (torch.randn(N_BLOCKS, STATE_BLOCK * width, device=dev) * 0.3).view(
        N_BLOCKS, -1
    )
    t2r = (torch.arange(m, device=dev, dtype=torch.int32) % 4).contiguous()
    pos = torch.arange(m, device=dev, dtype=torch.int32) + 4096
    pos[0::4] = RATIO - 1
    pos[1::4] = 8 * RATIO - 1
    slot = torch.arange(m, device=dev, dtype=torch.int32) + 7
    kv_slot = torch.arange(m, device=dev, dtype=torch.int32) + 3
    bt = torch.arange(4 * 256, device=dev, dtype=torch.int32).view(4, 256) % N_BLOCKS
    rms_w = torch.rand(HEAD, device=dev) + 0.5
    cs = torch.randn(65536, ops.DEEPSEEK_V4_ROPE_DIM, device=dev)
    kvc = torch.full(
        (N_BLOCKS, KV_BLOCK * (TOKEN_STRIDE + SCALE_DIM)),
        SENTINEL,
        dtype=torch.uint8,
        device=dev,
    )
    ops._dsv4_fused_sparse_compress_cache_kernel[(m,)](
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
    pos = torch.arange(m, dtype=torch.int64) + 4096
    pos[0::4] = RATIO - 1
    pos[1::4] = 8 * RATIO - 1
    kv_slot = torch.arange(m, dtype=torch.int64) + 3
    return kv_slot[(pos + 1) % RATIO == 0]


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
    return torch.stack(rows_v), torch.stack(rows_s), torch.stack(rows_r)


def _dequant_rows(fp8_bytes, scale_bytes):
    fp8 = fp8_bytes.view(torch.float8_e4m3fn).float()
    exp = scale_bytes[:, : NOPE // ops.DEEPSEEK_V4_FP8_QUANT_BLOCK].float() - 127.0
    scale = torch.exp2(exp).repeat_interleave(ops.DEEPSEEK_V4_FP8_QUANT_BLOCK, dim=-1)
    return fp8 * scale


def _assert_non_target_cache_bytes_unchanged(kvc, written_slots, label):
    touched_blocks = torch.zeros(N_BLOCKS, dtype=torch.bool, device=kvc.device)
    blocks = torch.div(written_slots, KV_BLOCK, rounding_mode="floor").to(kvc.device)
    offsets = (written_slots % KV_BLOCK).to(kvc.device)
    touched_blocks[blocks] = True
    assert torch.all(
        kvc[~touched_blocks] == SENTINEL
    ), f"{label}: an untargeted cache block changed"

    values = kvc[:, : KV_BLOCK * TOKEN_STRIDE].view(N_BLOCKS, KV_BLOCK, TOKEN_STRIDE)
    scales = kvc[:, KV_BLOCK * TOKEN_STRIDE :].view(N_BLOCKS, KV_BLOCK, SCALE_DIM)
    for block in blocks.unique().tolist():
        targeted_offsets = torch.zeros(KV_BLOCK, dtype=torch.bool, device=kvc.device)
        targeted_offsets[offsets[blocks == block]] = True
        assert torch.all(
            values[block, ~targeted_offsets] == SENTINEL
        ), f"{label}: an untargeted value-cache slot changed in block {block}"
        assert torch.all(
            scales[block, ~targeted_offsets] == SENTINEL
        ), f"{label}: an untargeted scale-cache slot changed in block {block}"


def _assert_written_rows_match(a, b, slots, label):
    fa, sa, ra = _written_views(a, slots)
    fb, sb, rb = _written_views(b, slots)
    assert torch.equal(sa, sb), f"{label}: scale bytes differ"
    da, db = _dequant_rows(fa, sa), _dequant_rows(fb, sb)
    if not torch.equal(da, db):
        frac = (da != db).float().mean().item()
        err = (da - db).abs().max().item()
        rel = ((da - db).abs() / db.abs().clamp_min(1e-6)).max().item()
        assert frac < 1e-4, f"{label}: mismatch fraction {frac:.2e}"
        assert err < 0.25, f"{label}: dequant max abs {err:.6f}"
        assert rel < 0.25, f"{label}: dequant max rel {rel:.4f}"
    assert torch.equal(
        ra.view(torch.bfloat16).float(), rb.view(torch.bfloat16).float()
    ), f"{label}: RoPE tail values differ"


@pytest.mark.parametrize("m", [16, 32, 64, 2048])
def test_sparse_compress_parity_4_vs_16_warps(m):
    slots = _written_slots(m)
    assert slots.numel() > 0
    for seed in (0, 1, 2):
        narrow = _run(m, 4, seed)
        wide = _run(m, 16, seed)
        _assert_non_target_cache_bytes_unchanged(
            narrow, slots, f"M={m} seed={seed} narrow"
        )
        _assert_non_target_cache_bytes_unchanged(wide, slots, f"M={m} seed={seed} wide")
        _assert_written_rows_match(
            narrow,
            wide,
            slots,
            f"M={m} seed={seed}",
        )


@pytest.mark.parametrize("m", [16, 64])
def test_sparse_compress_parity_partial_window_with_base_offsets(m):
    base = torch.full((4,), 15, device="cuda:0", dtype=torch.int32)
    pos = torch.arange(m, dtype=torch.int64) + 4096
    pos[0::4] = RATIO - 1
    pos[1::4] = 8 * RATIO - 1
    kv_slot = torch.arange(m, dtype=torch.int64) + 3
    slots = kv_slot[((pos + 1) % RATIO == 0) & (pos == 8 * RATIO - 1)]
    assert slots.numel() > 0
    all_written_slots = _written_slots(m)
    for seed in (0, 1):
        narrow = _run(m, 4, seed, base_offsets=base)
        wide = _run(m, 16, seed, base_offsets=base)
        _assert_non_target_cache_bytes_unchanged(
            narrow, all_written_slots, f"partial M={m} seed={seed} narrow"
        )
        _assert_non_target_cache_bytes_unchanged(
            wide, all_written_slots, f"partial M={m} seed={seed} wide"
        )
        _assert_written_rows_match(
            narrow,
            wide,
            slots,
            f"partial M={m} seed={seed}",
        )
