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

import importlib.util
import inspect
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.triton import dsv41 as implementation

_LAYOUTS = {
    "global": (512, 16, 256, 288),
    "index": (128, 32, 64, 68),
    "swa": (512, 32, 512, 528),
}


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    return torch.device("cuda:0")


def _reference_quantize(x, fmt):
    """Independent nearest-value oracle; reference inference/kernel.py semantics."""
    dim, group, _, _ = _LAYOUTS[fmt]
    x = x.detach().cpu().float().reshape(-1, dim // group, group)
    amax = x.abs().amax(dim=-1)
    if fmt == "global":
        scale = (amax.clamp_min(6 * 2.0**-9) / 6).to(torch.float8_e4m3fn)
        scale_bytes = scale.view(torch.uint8)
        scale = scale.float()
    else:
        scaled = amax.clamp_min(1e-4 if fmt == "swa" else 6 * 2.0**-126)
        scaled = scaled * (1 / (448 if fmt == "swa" else 6))
        bits = scaled.view(torch.int32)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).int()
        scale = torch.exp2(exponent.float())
        scale_bytes = (exponent + 127).to(torch.uint8)
    normalized = x / scale.unsqueeze(-1)
    if fmt == "swa":
        quant = normalized.clamp(-448, 448).to(torch.float8_e4m3fn)
        values = quant.view(torch.uint8).flatten(1)
        decoded = quant.float()
    else:
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
        # Searching even codes first implements ties-to-even independently of
        # the kernel's seven threshold comparisons.
        order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        distance = (normalized.abs().unsqueeze(-1) - levels[order]).abs()
        code = order[distance.argmin(dim=-1)]
        decoded = levels[code] * torch.where(torch.signbit(normalized), -1.0, 1.0)
        code = (
            (code | (torch.signbit(normalized).int() << 3)).to(torch.uint8).flatten(1)
        )
        values = code[:, ::2] | (code[:, 1::2] << 4)
    packed = torch.cat((values, scale_bytes.flatten(1)), dim=1)
    return packed, (decoded * scale.unsqueeze(-1)).reshape(-1, dim)


def _make_cache(x, fmt):
    width = _LAYOUTS[fmt][3]
    cache = torch.zeros(
        ((x.shape[0] + 63) // 64, 64, width), dtype=torch.uint8, device=x.device
    )
    dsv41.cache_scatter(x, cache, torch.arange(x.shape[0], device=x.device), fmt)
    return cache


def test_public_arguments_are_explicit():
    for name in dsv41.__all__:
        fn = getattr(dsv41, name)
        assert fn.__doc__
        assert all(
            p.default is inspect.Parameter.empty
            for p in inspect.signature(fn).parameters.values()
        )


@pytest.mark.parametrize("fmt", _LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cache_quantization_bytes_and_scales(device, fmt, dtype):
    torch.manual_seed(41)
    dim, group, _, width = _LAYOUTS[fmt]
    rows = torch.randn((37, dim), dtype=torch.float32)
    rows[0].zero_()
    rows[1].fill_(2.0**-20)
    rows[2].fill_(6.375 if fmt == "global" else 6)
    rows[3].fill_(1.875 * (448 if fmt == "swa" else 6))
    rows[4].reshape(-1, group)[:, 0] = -0.0
    rows[5].fill_(6 * 2.0**-126)
    rows[6].fill_(2.0**-127)
    rows[7].fill_(-(2.0**-127))
    rows = rows.to(dtype)
    expected_bytes, expected = _reference_quantize(rows, fmt)
    packed = dsv41.cache_pack(rows.to(device), fmt, None)
    assert packed.shape == (37, width)
    torch.testing.assert_close(packed.cpu(), expected_bytes, rtol=0, atol=0)
    for out_dtype in (torch.float32, torch.bfloat16):
        out = torch.empty((37, dim), dtype=out_dtype, device=device)
        assert dsv41.cache_unpack(packed, fmt, out) is out
        torch.testing.assert_close(out.cpu(), expected.to(out_dtype), rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["global", "index"])
def test_fp4_midpoints_signed_zero_and_saturation(device, fmt):
    dim, group, _, _ = _LAYOUTS[fmt]
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    samples = torch.cat((mids, -mids, torch.tensor([-0.0, 6.0])))
    rows = torch.zeros((3, dim), dtype=torch.float32)
    for i, direction in enumerate((-torch.inf, 0, torch.inf)):
        shifted = (
            samples
            if direction == 0
            else torch.nextafter(samples, torch.full_like(samples, direction))
        )
        rows[i, :16] = shifted
        rows[i].reshape(-1, group)[:, -1] = 6
    expected_bytes, expected = _reference_quantize(rows, fmt)
    packed = dsv41.cache_pack(rows.to(device), fmt, None)
    torch.testing.assert_close(packed.cpu(), expected_bytes, rtol=0, atol=0)
    out = torch.empty_like(rows, device=device)
    dsv41.cache_unpack(packed, fmt, out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert torch.signbit(out[1, 14]).item()
    # Midpoints tie to even codes: 0, 2, 2, 4, 4, 6, 6.
    torch.testing.assert_close(
        out[1, :7].cpu(), torch.tensor([0, 1, 1, 2, 2, 4, 4.0]), rtol=0, atol=0
    )


@pytest.mark.parametrize("fmt", _LAYOUTS)
def test_paged_field_strides_padding_and_invalid_slots(device, fmt):
    torch.manual_seed(42)
    dim, _, _, width = _LAYOUTS[fmt]
    backing = torch.full((4, 67, 2 * width + 19), 173, dtype=torch.uint8, device=device)
    cache = backing[:, :64, 7 : 7 + width * 2 : 2]
    rows = torch.randn((7, dim * 2), dtype=torch.float32, device=device)[:, ::2]
    slot_storage = torch.tensor(
        [191, 9, 63, 9, 64, 9, 0, 9, 255, 9, -1, 9, 256, 9], device=device
    )
    slots = slot_storage[::2]
    dsv41.cache_scatter(rows, cache, slots, fmt)
    packed, decoded = _reference_quantize(rows, fmt)
    expected_backing = torch.full_like(backing.cpu(), 173)
    for i, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < 256:
            expected_backing[slot // 64, slot % 64, 7 : 7 + width * 2 : 2] = packed[i]
    torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)
    requested = slots.unsqueeze(0).expand(2, -1)
    out = torch.empty((2, 7, dim), dtype=torch.float32, device=device)
    assert dsv41.cache_gather(cache, requested, fmt, out) is out
    decoded[-2:].zero_()
    torch.testing.assert_close(
        out.cpu(), decoded.unsqueeze(0).expand(2, -1, -1), rtol=0, atol=0
    )
    # The row-matrix reader must honor strides as well, not flatten the arena.
    direct = dsv41.cache_unpack(cache[2, 63:64], fmt, None)
    torch.testing.assert_close(direct.cpu(), decoded[:1].bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [1, 7, 16])
@pytest.mark.parametrize("with_global", [False, True])
def test_selected_attention_joint_sink_masking_and_chunking(device, heads, with_global):
    torch.manual_seed(43)
    q = torch.randn((5, heads, 512), dtype=torch.bfloat16, device=device) * 2
    original_q = q.clone()
    swa = _make_cache(
        torch.randn((130, 512), dtype=torch.bfloat16, device=device), "swa"
    )
    glob = _make_cache(
        torch.randn((65, 512), dtype=torch.bfloat16, device=device), "global"
    )
    swa_slots = torch.tensor(
        [
            [63, 64, 129, -1],
            [-1, -1, -1, -1],
            [192, -2, 63, 0],
            [0, 0, 1, 2],
            [1, 2, 3, 4],
        ],
        device=device,
    )
    global_slots = torch.tensor(
        [[0, 64, -1], [-1, -1, -1], [128, 0, 1], [0, 1, 2], [2, 3, 4]], device=device
    )
    swa_lens = torch.tensor([3, 4, 3, 4, 0], dtype=torch.int32, device=device)
    global_lens = torch.tensor([2, 3, 2, 3, 0], dtype=torch.int32, device=device)
    sink = torch.linspace(-10, 10, heads, dtype=torch.float32, device=device)
    parts, masks = [], []
    for cache, slots, lens, fmt in [(swa, swa_slots, swa_lens, "swa")] + (
        [(glob, global_slots, global_lens, "global")] if with_global else []
    ):
        parts.append(dsv41.cache_gather(cache, slots, fmt, None).float())
        masks.append(
            (torch.arange(slots.shape[1], device=device) < lens[:, None])
            & (slots >= 0)
            & (slots < cache.shape[0] * 64)
        )
    kv, valid = torch.cat(parts, dim=1), torch.cat(masks, dim=1)
    logits = torch.bmm(q.float(), kv.transpose(1, 2)) * 512**-0.5
    logits.masked_fill_(~valid[:, None], -torch.inf)
    logits = torch.cat((logits, sink[None, :, None].expand(5, -1, 1)), dim=-1)
    expected = torch.bmm(logits.softmax(dim=-1)[..., :-1], kv).bfloat16()
    for chunk in (1, 3, 9):
        out = torch.empty_like(q)
        result = dsv41.selected_attention(
            q,
            swa,
            swa_slots,
            swa_lens,
            glob if with_global else None,
            global_slots if with_global else None,
            global_lens if with_global else None,
            sink,
            512**-0.5,
            out,
            chunk,
        )
        assert result is out
        torch.testing.assert_close(out, expected, rtol=0.008, atol=0.004)
        assert torch.count_nonzero(out[[1, 4]]).item() == 0
    torch.testing.assert_close(q, original_q, rtol=0, atol=0)


def test_selected_attention_full_width_online_stability(device):
    torch.manual_seed(49)
    q = torch.randn((2, 2, 512), dtype=torch.bfloat16, device=device) * 8
    swa_x = torch.randn((128, 512), dtype=torch.bfloat16, device=device)
    global_x = torch.randn((512, 512), dtype=torch.bfloat16, device=device)
    swa, glob = _make_cache(swa_x, "swa"), _make_cache(global_x, "global")
    swa_slots = torch.arange(128, device=device).expand(2, -1)
    global_slots = torch.arange(512, device=device).expand(2, -1)
    sink = torch.tensor([-10000, 10000], dtype=torch.float32, device=device)
    out = dsv41.selected_attention(
        q,
        swa,
        swa_slots,
        torch.full((2,), 128, device=device),
        glob,
        global_slots,
        torch.full((2,), 512, device=device),
        sink,
        512**-0.5,
        None,
        1,
    )
    _, swa_ref = _reference_quantize(swa_x, "swa")
    _, global_ref = _reference_quantize(global_x, "global")
    kv = (
        torch.cat((swa_ref, global_ref)).to(device=device, dtype=torch.bfloat16).float()
    )
    logits = (q.float() @ kv.T) * 512**-0.5
    probabilities = torch.cat(
        (logits, sink[None, :, None].expand(2, -1, 1)), dim=-1
    ).softmax(dim=-1)
    expected = (probabilities[..., :-1] @ kv).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.004)
    assert not out[:, 1].any()


def test_sink_counted_once_and_representations_not_deduplicated(device):
    q = torch.zeros((1, 1, 512), dtype=torch.bfloat16, device=device)
    rows = torch.ones((1, 512), dtype=torch.bfloat16, device=device)
    swa, glob = _make_cache(rows, "swa"), _make_cache(rows, "global")
    slots = torch.zeros((1, 1), dtype=torch.int32, device=device)
    lens = torch.ones((1,), dtype=torch.int32, device=device)
    sink = torch.zeros((1,), device=device)
    out = dsv41.selected_attention(
        q, swa, slots, lens, glob, slots, lens, sink, 512**-0.5, None, 1
    )
    values = (
        dsv41.cache_gather(swa, slots, "swa", None).float()
        + dsv41.cache_gather(glob, slots, "global", None).float()
    )
    torch.testing.assert_close(out, (values / 3).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_index_scores_reference_rounding_per_head_relu_and_weights(
    device, weight_dtype
):
    torch.manual_seed(44)
    q = torch.randn((3, 4, 128), dtype=torch.bfloat16, device=device)
    k = torch.randn((70, 128), dtype=torch.bfloat16, device=device)
    cache = _make_cache(k, "index")
    slots = torch.tensor(
        [[0, 63, 64, -1], [65, 66, 67, 128], [1, 2, 3, 4]], device=device
    )
    weights = torch.tensor(
        [[1, -2, 3, -4], [0.3, -0.2, 0.7, 0.1], [-1, 0, -2, 1]],
        dtype=weight_dtype,
        device=device,
    )
    _, q_ref = _reference_quantize(q, "index")
    _, k_ref = _reference_quantize(k, "index")
    q_ref = q_ref.to(device=device, dtype=torch.bfloat16).reshape(q.shape)
    keys = k_ref.to(device=device, dtype=torch.bfloat16)[slots.clamp(0, 69)]
    dots = torch.einsum("thd,tkd->thk", q_ref, keys)
    expected = (dots.relu() * weights.unsqueeze(-1)).sum(dim=1)
    expected.masked_fill_((slots < 0) | (slots >= 128), -torch.inf)
    out = torch.empty_like(expected)
    assert dsv41.index_score(q, weights, cache, slots, None, out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    quantized = dsv41.index_q_quantize(q, None)
    torch.testing.assert_close(quantized, q_ref, rtol=0, atol=0)


@pytest.mark.parametrize("visible", [0, 1, 7, 8, 9, 511, 512, 513, 16383, 16384, 16385])
def test_full_top512_and_block_candidates_boundaries(device, visible):
    # A mix of magnitudes exercises BF16 scoring ties. Compare selected score
    # multisets rather than insisting on an unspecified tie-breaking rule.
    rows = 16448 if visible > 513 else 576
    k = torch.zeros((rows, 128), dtype=torch.bfloat16, device=device)
    ids = torch.arange(rows, device=device)
    k[:, 0] = (ids // 128).to(torch.bfloat16)
    k[:, 32] = (ids % 128).to(torch.bfloat16) / 128
    # Random keys avoid pretending the quantized monotone construction is tie-free.
    torch.manual_seed(45)
    k += torch.randn_like(k) * 0.02
    q = torch.ones((1, 2, 128), dtype=torch.bfloat16, device=device)
    weights = torch.tensor([[1, -0.125]], dtype=torch.bfloat16, device=device)
    cache = _make_cache(k, "index")
    table = torch.arange(cache.shape[0] - 1, -1, -1, device=device).unsqueeze(0)
    # Populate in reverse physical-page order to exercise request-local IDs.
    logical_slots = table[0, ids // 64] * 64 + ids % 64
    dsv41.cache_scatter(k, cache, logical_slots, "index")
    lens = torch.tensor([visible], device=device)
    out = dsv41.index_topk(
        q, weights, cache, table, lens, None, 512, 2048, 8, 1, 256, None, None
    )
    selected, lengths, candidates, candidate_lens = out
    assert lengths.item() == min(512, visible)
    assert candidate_lens.item() == min(2048, (visible + 7) // 8)
    assert (selected[0, lengths.item() :] == -1).all()
    assert (candidates[0, candidate_lens.item() :] == -1).all()
    if not visible:
        return
    full = dsv41.index_score(
        q, weights, cache, logical_slots[:visible].unsqueeze(0), None, None
    ).float()[0]
    chosen = selected[0, : lengths.item()].long()
    assert (chosen[1:] > chosen[:-1]).all()
    assert chosen.min() >= 0 and chosen.max() < visible
    torch.testing.assert_close(
        full[chosen].sort().values,
        full.topk(min(512, visible)).values.sort().values,
        rtol=0,
        atol=0,
    )
    blocks = (
        torch.nn.functional.pad(full, (0, -visible % 8), value=-torch.inf)
        .view(-1, 8)
        .amax(dim=-1)
    )
    blocks[-1] = torch.inf
    chosen_blocks = candidates[0, : candidate_lens.item()].long()
    assert ((visible - 1) // 8 == chosen_blocks).any()
    torch.testing.assert_close(
        blocks[chosen_blocks].sort().values,
        blocks.topk(min(2048, blocks.numel())).values.sort().values,
        rtol=0,
        atol=0,
    )


def test_source_uses_block_max_not_top512_and_forces_latest(device):
    # 2049 full blocks + one partial latest block; every block has one equally
    # good row, but only 512 row selections. Candidate coverage must be broader.
    rows = 2049 * 8 + 1
    k = torch.zeros((rows, 128), dtype=torch.bfloat16, device=device)
    k[::8, 0] = 6
    k[-1, 0] = -6
    cache = _make_cache(k, "index")
    q = torch.zeros((1, 1, 128), dtype=torch.bfloat16, device=device)
    q[..., 0] = 6
    weights = torch.ones((1, 1), dtype=torch.bfloat16, device=device)
    table = torch.arange(cache.shape[0], device=device).unsqueeze(0)
    result = dsv41.index_topk(
        q,
        weights,
        cache,
        table,
        torch.tensor([rows], device=device),
        None,
        512,
        2048,
        8,
        1,
        512,
        None,
        None,
    )
    top, lens, candidates, candidate_lens = result
    assert lens.item() == 512 and candidate_lens.item() == 2048
    assert (candidates == 2049).any()
    assert torch.unique(candidates).numel() == 2048
    assert torch.unique(top // 8).numel() <= 512


def test_reindex_reads_only_candidate_rows_and_reapplies_causality(device):
    torch.manual_seed(46)
    q = torch.randn((3, 2, 128), dtype=torch.bfloat16, device=device)
    weights = torch.randn((3, 2), dtype=torch.bfloat16, device=device)
    cache = _make_cache(
        torch.randn((320, 128), dtype=torch.bfloat16, device=device), "index"
    )
    table = torch.tensor([[4, 1, 3, 0, 2]], device=device).expand(3, -1)
    visible = torch.tensor([263, 8, 0], device=device)
    candidates = torch.tensor(
        [[32, 0, -1, 17], [0, 1, -1, -1], [1, -1, -1, -1]], device=device
    )
    original_gather = implementation.cache_gather
    reads = []

    def tracked(cache, slots, cache_format, out):
        reads.append(slots.clone())
        return original_gather(cache, slots, cache_format, out)

    with patch.object(implementation, "cache_gather", side_effect=tracked):
        top, lengths, blocks, block_lens = dsv41.index_topk(
            q, weights, cache, table, visible, candidates, 512, 0, 8, 2, 16, None, None
        )
    assert lengths.tolist() == [23, 8, 0]
    assert blocks.shape == (3, 0) and not block_lens.any()
    assert all(slots.shape[0] <= 2 and slots.shape[1] <= 16 for slots in reads)
    assert sum(slots.numel() for slots in reads) == 3 * 4 * 8
    expected0 = torch.tensor(
        list(range(8)) + list(range(136, 144)) + list(range(256, 263)), device=device
    )
    torch.testing.assert_close(top[0, :23].long(), expected0, rtol=0, atol=0)
    assert (top[0, 23:] == -1).all() and (top[2] == -1).all()


def test_candidate_block_max_not_sum(device):
    k = torch.zeros((17, 128), dtype=torch.bfloat16, device=device)
    k[0, 0] = 6
    k[8:16, 0] = 4
    k[16, 0] = -6
    q = torch.zeros((1, 1, 128), dtype=torch.bfloat16, device=device)
    q[..., 0] = 6
    cache = _make_cache(k, "index")
    result = dsv41.index_topk(
        q,
        torch.ones((1, 1), dtype=torch.bfloat16, device=device),
        cache,
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.tensor([17], device=device),
        None,
        512,
        2,
        8,
        1,
        8,
        None,
        None,
    )
    # Block 0 wins by max (36 > 24); block 1 would win by sum (192 > 36).
    # Latest block 2 must still be pinned even with its rectified score zero.
    assert result[2].tolist() == [[0, 2]]
    assert result[3].tolist() == [2]


def test_full_tiles_causal_page_mask_and_output_reuse(device):
    torch.manual_seed(48)
    q = torch.randn((3, 2, 128), dtype=torch.bfloat16, device=device)
    weights = torch.randn((3, 2), dtype=torch.bfloat16, device=device)
    cache = _make_cache(
        torch.randn((192, 128), dtype=torch.bfloat16, device=device), "index"
    )
    table = torch.tensor([[2, 0, 1], [1, -1, 3], [0, 1, 2]], device=device)
    visible = torch.tensor([137, 190, 0], device=device)
    outputs = tuple(
        torch.full(shape, 99, dtype=torch.int32, device=device)
        for shape in ((3, 9), (3,), (3, 2), (3,))
    )
    logical = torch.arange(192, device=device).expand(3, -1)
    pages = table.gather(1, logical // 64)
    slots = pages * 64 + logical % 64
    valid = (logical < visible[:, None]) & (pages >= 0) & (pages < 3)
    slots = slots.masked_fill(~valid, -1)
    expected = dsv41.index_score(q, weights, cache, slots, None, None).float()
    for query_chunk, score_chunk in ((1, 8), (2, 64)):
        with patch.object(
            implementation, "_score", wraps=implementation._score
        ) as score:
            result = dsv41.index_topk(
                q,
                weights,
                cache,
                table,
                visible,
                None,
                9,
                2,
                8,
                query_chunk,
                score_chunk,
                None,
                outputs,
            )
        assert result is outputs
        for call in score.call_args_list:
            tile_q, _, _, tile_slots, _ = call.args
            assert tile_q.shape[0] <= query_chunk
            assert tile_slots.shape[1] <= score_chunk
        assert result[1].tolist() == [9, 9, 0]
        for t in range(2):
            chosen = result[0][t].long()
            torch.testing.assert_close(
                expected[t, chosen].sort().values,
                expected[t].topk(9).values.sort().values,
                rtol=0,
                atol=0,
            )
        assert (result[0][2] == -1).all() and (result[2][2] == -1).all()


def test_reject_unbounded_candidate_input(device):
    q = torch.zeros((1, 1, 128), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((1, 64, 68), dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="at most 2048"):
        dsv41.index_topk(
            q,
            torch.ones((1, 1), dtype=torch.bfloat16, device=device),
            cache,
            torch.zeros((1, 1), dtype=torch.int32, device=device),
            torch.ones((1,), dtype=torch.int32, device=device),
            torch.zeros((1, 2049), dtype=torch.int32, device=device),
            512,
            0,
            8,
            1,
            64,
            None,
            None,
        )


def test_empty_cache_and_zero_width_attention(device):
    for fmt, (dim, _, _, width) in _LAYOUTS.items():
        cache = torch.empty((0, 64, width), dtype=torch.uint8, device=device)
        slots = torch.tensor([[-1, 0, 64]], device=device)
        result = dsv41.cache_gather(cache, slots, fmt, None)
        assert result.shape == (1, 3, dim) and not result.any()
        packed = dsv41.cache_pack(
            torch.empty((0, dim), dtype=torch.bfloat16, device=device), fmt, None
        )
        assert dsv41.cache_unpack(packed, fmt, None).shape == (0, dim)
    q = torch.ones((2, 3, 512), dtype=torch.bfloat16, device=device)
    cache = torch.empty((0, 64, 528), dtype=torch.uint8, device=device)
    slots = torch.empty((2, 0), dtype=torch.int32, device=device)
    lens = torch.zeros((2,), dtype=torch.int32, device=device)
    result = dsv41.selected_attention(
        q,
        cache,
        slots,
        lens,
        None,
        None,
        None,
        torch.zeros(3, device=device),
        512**-0.5,
        None,
        1,
    )
    assert not result.any()


def test_optional_snapshot_quantization_oracle(device):
    """Opt in with DSV41_REFERENCE_DIR; TileLang is not a runtime dependency."""
    directory = os.environ.get("DSV41_REFERENCE_DIR")
    if directory is None:
        pytest.skip("set DSV41_REFERENCE_DIR to the reference inference directory")
    pytest.importorskip("tilelang")
    path = Path(directory) / "kernel.py"
    spec = importlib.util.spec_from_file_location("dsv41_snapshot_kernel", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(47)
    for fmt, (dim, group, _, _) in _LAYOUTS.items():
        x = torch.randn((32, dim), dtype=torch.bfloat16, device=device)
        x[0].zero_()
        if fmt == "swa":
            values, scales = reference.act_quant(
                x,
                block_size=group,
                scale_fmt="ue8m0",
                scale_dtype=torch.float8_e8m0fnu,
                inplace=False,
            )
        else:
            values, scales = reference.fp4_act_quant(
                x,
                block_size=group,
                inplace=False,
                scale_dtype=(
                    torch.float8_e4m3fn if fmt == "global" else torch.float8_e8m0fnu
                ),
            )
        expected = torch.cat(
            (values.view(torch.uint8), scales.view(torch.uint8)), dim=-1
        )
        torch.testing.assert_close(
            dsv41.cache_pack(x, fmt, None), expected, rtol=0, atol=0
        )
