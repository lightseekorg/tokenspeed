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
from tokenspeed_kernel.numerics.torch.attention.dsv41 import (
    torch_dsv41_cache_quantize,
    torch_dsv41_compressor_pairs,
)
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import triton as implementation
from utils import assert_no_triton_compile

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


def _make_cache(x, fmt):
    width = _LAYOUTS[fmt][3]
    cache = torch.zeros(
        ((x.shape[0] + 63) // 64, 64, width), dtype=torch.uint8, device=x.device
    )
    dsv41.cache_scatter(x, cache, torch.arange(x.shape[0], device=x.device), fmt)
    return cache


def test_compressor_tail_scatter_graph_strides_and_replay(device):
    storage = torch.zeros(5, 2, 2, 1024, device=device)
    tail = storage[1:4, :, :, 1::2]
    content = torch.randn(6, 1024, device=device)[:, ::2]
    scores = torch.randn(6, 1024, device=device)[:, 1::2]
    slot_storage = torch.tensor(
        [-1, 99, 1, 99, 3, 99, 5, 99, 6, 99, -2, 99], device=device
    )
    slots = slot_storage[::2]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dsv41.compressor_tail_scatter(content, scores, tail, slots)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            dsv41.compressor_tail_scatter(content, scores, tail, slots)
    torch.cuda.current_stream().wait_stream(stream)
    for values in ([-1, 1, 3, 5, 6, -2], [4, -1, 2, 0, 99, -1], [-1] * 6):
        storage.zero_()
        content.normal_()
        scores.normal_()
        slots.copy_(torch.tensor(values, device=device))
        expected = torch.zeros_like(storage)
        for i, slot in enumerate(values):
            if 0 <= slot < 6:
                expected[1 + slot // 2, slot % 2, 0, 1::2] = content[i]
                expected[1 + slot // 2, slot % 2, 1, 1::2] = scores[i]
        graph.replay()
        torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    dsv41.compressor_tail_scatter(content[:0], scores[:0], tail, slots[:0])


def test_public_argument_defaults():
    for name in dsv41.__all__:
        fn = getattr(dsv41, name)
        if not callable(fn):
            continue
        assert fn.__doc__, name
        for parameter in inspect.signature(fn).parameters.values():
            if name == "index_topk" and parameter.name in ("index_metadata", "slots"):
                assert parameter.default is None
                continue
            assert (
                parameter.default is inspect.Parameter.empty
            ), f"{name}.{parameter.name} must be explicit"


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
    expected_bytes, expected = torch_dsv41_cache_quantize(rows, fmt)
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
    expected_bytes, expected = torch_dsv41_cache_quantize(rows, fmt)
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
    dim, _, values, width = _LAYOUTS[fmt]
    backing = torch.full((4, 67, 2 * width + 19), 173, dtype=torch.uint8, device=device)
    cache = backing[:, :64, 7 : 7 + width * 2 : 2]
    rows = torch.randn((7, dim * 2), dtype=torch.float32, device=device)[:, ::2]
    slot_storage = torch.tensor(
        [191, 9, 63, 9, 64, 9, 0, 9, 255, 9, -1, 9, 256, 9], device=device
    )
    slots = slot_storage[::2]
    dsv41.cache_scatter(rows, cache, slots, fmt)
    packed, decoded = torch_dsv41_cache_quantize(rows, fmt)
    expected_backing = torch.full_like(backing.cpu(), 173)
    for i, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < 256:
            for byte in range(width):
                offset = (
                    (slot % 64) * values + byte
                    if byte < values
                    else 64 * values + (slot % 64) * (width - values) + byte - values
                )
                row, column = divmod(offset, width)
                expected_backing[slot // 64, row, 7 + 2 * column] = packed[i, byte]
    torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)
    requested = slots.unsqueeze(0).expand(2, -1)
    out = torch.empty((2, 7, dim), dtype=torch.float32, device=device)
    assert dsv41.cache_gather(cache, requested, fmt, out) is out
    decoded[-2:].zero_()
    torch.testing.assert_close(
        out.cpu(), decoded.unsqueeze(0).expand(2, -1, -1), rtol=0, atol=0
    )
    # Standalone packed rows retain their row layout; a page-planar field slice
    # is deliberately not a packed row. The row codec still honors byte strides.
    packed_storage = torch.empty((1, width * 2 + 19), dtype=torch.uint8, device=device)
    packed_row = packed_storage[:, 7 : 7 + width * 2 : 2]
    packed_row.copy_(packed[:1].to(device))
    direct = dsv41.cache_unpack(packed_row, fmt, None)
    torch.testing.assert_close(direct.cpu(), decoded[:1].bfloat16(), rtol=0, atol=0)


def test_cache_gather_output_rows_past_int32(device):
    """Gather output rows whose offsets pass 2**31 elements land in place.

    The prefill workspace gathers a batch's whole owner history, more than
    2**31 / 512 rows near a 1M-token context.
    """
    fmt, dim, tail = "global", 512, 3
    rows = 2**31 // dim + tail
    guard = 2**31
    required_bytes = (guard + rows * dim) * torch.bfloat16.itemsize
    if torch.cuda.mem_get_info(device)[0] < required_bytes + (256 << 20):
        pytest.skip("requires 9 GiB of free device memory")
    torch.manual_seed(43)
    values = torch.randn((tail, dim), dtype=torch.float32, device=device)
    cache = _make_cache(values, fmt)
    slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
    slots[-tail:] = torch.arange(tail, device=device)
    # Row 2**31 / dim + k wrapped to 32 bits lands k * dim elements past the
    # storage base, inside the guard, where it would overwrite the sentinel.
    storage = torch.empty(guard + rows * dim, dtype=torch.bfloat16, device=device)
    storage[: tail * dim].fill_(-7.0)
    out = storage[guard:].view(rows, dim)
    out[-tail:].fill_(float("nan"))

    assert dsv41.cache_gather(cache, slots, fmt, out) is out
    _, decoded = torch_dsv41_cache_quantize(values, fmt)
    torch.testing.assert_close(
        out[-tail:].cpu(), decoded.bfloat16().cpu(), rtol=0, atol=0
    )
    sentinel = storage[: tail * dim]
    torch.testing.assert_close(sentinel, torch.full_like(sentinel, -7.0))


def test_cache_scatter_input_rows_past_int32(device):
    """Scatter reads input rows whose offsets pass 2**31 elements.

    The codec's cache side addresses slots in 64 bits; the input rows follow a
    caller-sized buffer, so their offsets do too. Three rows a 2**30 stride
    apart put the last read at 2**31 elements, and a guard before the base
    holds a sentinel a wrapped read would return instead.
    """
    fmt, dim = "global", 512
    stride, guard = 2**30, 2**31
    needed = guard + 2 * stride + dim
    if torch.cuda.mem_get_info(device)[0] < needed * 2 + (256 << 20):
        pytest.skip("requires 9 GiB of free device memory")
    torch.manual_seed(44)
    storage = torch.empty(needed, dtype=torch.bfloat16, device=device)
    storage[:dim].fill_(3.0)
    rows = storage[guard:].as_strided((3, dim), (stride, 1))
    rows.copy_(torch.randn((3, dim), dtype=torch.bfloat16, device=device))
    cache = torch.zeros((1, 64, _LAYOUTS[fmt][3]), dtype=torch.uint8, device=device)
    slots = torch.arange(3, device=device)

    dsv41.cache_scatter(rows, cache, slots, fmt)
    _, decoded = torch_dsv41_cache_quantize(rows.float(), fmt)
    got = dsv41.cache_gather(cache, slots, fmt, None)
    torch.testing.assert_close(got.cpu(), decoded.bfloat16().cpu(), rtol=0, atol=0)
    sentinel = storage[:dim]
    torch.testing.assert_close(sentinel, torch.full_like(sentinel, 3.0))


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
            None,
            None,
            None,
            lse=None,
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
        None,
        None,
        None,
        lse=None,
    )
    _, swa_ref = torch_dsv41_cache_quantize(swa_x, "swa")
    _, global_ref = torch_dsv41_cache_quantize(global_x, "global")
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
        q,
        swa,
        slots,
        lens,
        glob,
        slots,
        lens,
        sink,
        512**-0.5,
        None,
        1,
        None,
        None,
        None,
        lse=None,
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
    _, q_ref = torch_dsv41_cache_quantize(q, "index")
    _, k_ref = torch_dsv41_cache_quantize(k, "index")
    q_ref = q_ref.to(device=device, dtype=torch.bfloat16).reshape(q.shape)
    keys = k_ref.to(device=device, dtype=torch.bfloat16)[slots.clamp(0, 69)]
    dots = torch.einsum("thd,tkd->thk", q_ref, keys)
    expected = (dots.relu() * weights.unsqueeze(-1)).sum(dim=1)
    expected.masked_fill_((slots < 0) | (slots >= 128), -torch.inf)
    out = torch.empty_like(expected)
    assert dsv41.index_score(q, weights, cache, slots, None, out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    quantized = dsv41.index_q_quantize(q, "index", None)
    torch.testing.assert_close(quantized, q_ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "visible", [0, 1, 7, 8, 9, 511, 512, 513, 16383, 16384, 16385, 262153]
)
def test_full_top512_and_block_candidates_boundaries(device, visible):
    # A mix of magnitudes exercises BF16 scoring ties. Compare selected score
    # multisets rather than insisting on an unspecified tie-breaking rule.
    rows = ((visible + 63) // 64) * 64 if visible > 513 else 576
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
        q, weights, cache, table, lens, None, 512, 2048, 8, 1, 256, None, None, None
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
    with patch.object(implementation, "cache_gather", side_effect=AssertionError):
        top, lengths, blocks, block_lens = dsv41.index_topk(
            q,
            weights,
            cache,
            table,
            visible,
            candidates,
            512,
            0,
            8,
            2,
            16,
            None,
            None,
            None,
        )
    assert lengths.tolist() == [23, 8, 0]
    assert blocks.shape == (3, 0) and not block_lens.any()

    expected0 = torch.tensor(
        list(range(8)) + list(range(136, 144)) + list(range(256, 263)), device=device
    )
    torch.testing.assert_close(top[0, :23].long(), expected0, rtol=0, atol=0)
    assert (top[0, 23:] == -1).all() and (top[2] == -1).all()


def test_index_topk_batch_without_any_visible_context(device):
    # A padded warmup batch: 32 replicated heads, every visible length 0, every
    # page null. Native scorers must not fault on a batch with no keys at all.
    q = torch.randn((96, 32, 128), dtype=torch.bfloat16, device=device)
    weights = torch.rand((96, 32), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((2, 64, 132), dtype=torch.uint8, device=device)
    dsv41.cache_scatter(
        torch.randn((128, 128), dtype=torch.bfloat16, device=device),
        cache,
        torch.arange(128, device=device),
        "index_v4",
    )
    table = torch.full((96, 1025), -1, dtype=torch.int32, device=device)
    visible = torch.zeros(96, dtype=torch.int32, device=device)
    top, lengths, blocks, block_lens = dsv41.index_topk(
        q, weights, cache, table, visible, None, 512, 64, 8, 64, 4096, None, None, None
    )
    torch.cuda.synchronize()
    assert not lengths.any() and not block_lens.any()
    assert (top == -1).all() and (blocks == -1).all()


def _index_cache(k, fmt):
    width = implementation._LAYOUTS[fmt][3]
    cache = torch.zeros(
        ((k.shape[0] + 63) // 64, 64, width), dtype=torch.uint8, device=k.device
    )
    dsv41.cache_scatter(k, cache, torch.arange(k.shape[0], device=k.device), fmt)
    return cache


@pytest.mark.parametrize("fmt", ["index", "index_v4"])
@pytest.mark.parametrize("shared_table", [True, False])
def test_index_topk_selects_top_scores_on_every_index_format(device, fmt, shared_table):
    # 32 replicated heads, a paged history in reverse page order, mixed visible
    # lengths: the shape every native scorer serves. A shared (stride-0) table
    # is the chunked-prefill shape, a per-row table the decode shape; native
    # scorers take different paths for the two. Whatever kernel the format
    # routes to, each chosen row must score within rounding of the true top set,
    # and the reindex pass must return exactly the candidate rows it was handed.
    torch.manual_seed(47)
    rows = 64 * 40
    k = torch.randn((rows, 128), dtype=torch.bfloat16, device=device)
    q = torch.randn((4, 32, 128), dtype=torch.bfloat16, device=device)
    weights = torch.rand((4, 32), dtype=torch.bfloat16, device=device)
    cache = _index_cache(k, fmt)
    table = torch.arange(cache.shape[0] - 1, -1, -1, device=device).expand(4, -1)
    if not shared_table:
        table = table.contiguous()
    visible = torch.tensor([rows, 1000, 513, 0], device=device)
    logical = torch.arange(rows, device=device).expand(4, -1)
    slots = table.gather(1, logical // 64) * 64 + logical % 64
    slots = slots.masked_fill(logical >= visible[:, None], -1)
    scores = dsv41.index_score(q, weights, cache, slots, None, None).float()

    top, lengths, blocks, block_lens = dsv41.index_topk(
        q, weights, cache, table, visible, None, 512, 64, 8, 64, 4096, None, None, None
    )
    torch.cuda.synchronize()
    assert lengths.tolist() == [512, 512, 512, 0]
    assert block_lens.tolist() == [64, 64, 64, 0]
    tol = 2**-6 * scores[:, :].abs().amax()
    for r in range(3):
        chosen = top[r, : lengths[r]].long()
        assert chosen.min() >= 0 and chosen.max() < visible[r]
        threshold = scores[r, : visible[r]].topk(512).values.min()
        assert (scores[r, chosen] >= threshold - tol).all()
        block_max = (
            torch.nn.functional.pad(
                scores[r, : visible[r]], (0, -int(visible[r]) % 8), value=-torch.inf
            )
            .view(-1, 8)
            .amax(-1)
        )
        picked = blocks[r, : block_lens[r]].long()
        assert ((visible[r] - 1) // 8 == picked).any()
        assert (block_max[picked] >= block_max.topk(64).values.min() - tol).all()
    assert (top[3] == -1).all() and (blocks[3] == -1).all()

    rerows, relengths, _, _ = dsv41.index_topk(
        q,
        weights,
        cache,
        table,
        visible,
        blocks,
        512,
        0,
        8,
        64,
        4096,
        None,
        None,
        None,
    )
    for r in range(3):
        allowed = (
            blocks[r, : block_lens[r]].long()[:, None] * 8
            + torch.arange(8, device=device)
        ).flatten()
        allowed = allowed[allowed < visible[r]]
        chosen = rerows[r, : relengths[r]].long()
        assert relengths[r] == min(512, allowed.numel())
        assert torch.isin(chosen, allowed).all()
        threshold = scores[r, allowed].topk(int(relengths[r])).values.min()
        assert (scores[r, chosen] >= threshold - tol).all()
    assert relengths[3] == 0


@pytest.mark.parametrize("contiguous", [True, False])
def test_index_query_quantization_matches_the_reference_codec(device, contiguous):
    # The fused quantizer replaces a chain of eager ops, so it has to reproduce
    # them bit for bit: the E4M3 payload decides which rows a pass scores, and
    # the folded weight carries the query scale the logits kernels never apply.
    # A head-major view is the shape a projection hands over before any copy.
    torch.manual_seed(51)
    q = torch.randn((5, 32, 128), dtype=torch.bfloat16, device=device)
    weights = torch.rand((5, 32), dtype=torch.bfloat16, device=device)
    if not contiguous:
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        assert not q.is_contiguous()
    quantized, folded = implementation.quantize_index_queries(q, weights)

    values = q.float()
    scale = values.abs().amax(-1, keepdim=True).clamp_min(1.0e-6) / 448.0
    expected = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        quantized.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        folded, weights.float() * scale.squeeze(-1), rtol=0, atol=0
    )


@pytest.mark.parametrize("fmt", ["index", "index_v4"])
def test_reindex_maps_every_candidate_block_order_back_to_row_ids(device, fmt):
    # A produced pool ascends, but the op accepts any block order with -1
    # padding anywhere, and a scorer that compacts the history down to the pool
    # must still report row ids rather than pool columns. Top-K capacity sits
    # above the pool, so every visible candidate row comes back regardless of
    # score, which pins the mapping itself on each format's scorer.
    torch.manual_seed(46)
    q = torch.randn((3, 32, 128), dtype=torch.bfloat16, device=device)
    weights = torch.rand((3, 32), dtype=torch.bfloat16, device=device)
    k = torch.randn((64 * 6, 128), dtype=torch.bfloat16, device=device)
    cache = _index_cache(k, fmt)
    table = torch.arange(cache.shape[0], device=device).expand(3, -1).contiguous()
    visible = torch.tensor([263, 8, 0], device=device)
    candidates = torch.tensor(
        [[32, 0, -1, 17], [0, 1, -1, -1], [1, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    top, lengths, blocks, block_lens = dsv41.index_topk(
        q,
        weights,
        cache,
        table,
        visible,
        candidates,
        512,
        0,
        8,
        64,
        4096,
        None,
        None,
        None,
    )
    torch.cuda.synchronize()
    assert lengths.tolist() == [23, 8, 0]
    assert blocks.shape == (3, 0) and not block_lens.any()
    expected = torch.tensor(
        list(range(8)) + list(range(136, 144)) + list(range(256, 263)), device=device
    )
    torch.testing.assert_close(top[0, :23].long(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        top[1, :8].long(), torch.arange(8, device=device), rtol=0, atol=0
    )
    assert (top[0, 23:] == -1).all() and (top[1, 8:] == -1).all()
    assert (top[2] == -1).all()


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
            implementation,
            "_index_finish_parts",
            wraps=implementation._index_finish_parts,
        ) as finish:
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
                None,
            )
        assert result is outputs
        for call in finish.call_args_list:
            scores, ids, k, _, _ = call.args
            assert scores.shape[0] <= query_chunk
            assert scores.shape == ids.shape
            assert scores.shape[1] <= 16 * max(
                16, score_chunk, 1 << (k - 1).bit_length()
            )
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
        None,
        None,
        None,
        lse=None,
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


@pytest.mark.parametrize("fmt", ["swa", "swa_v4"])
@pytest.mark.parametrize("normalize", [False, True])
def test_fused_swa_matches_rope_quantization_and_page_bytes(device, fmt, normalize):
    from tokenspeed_kernel.ops.attention.dsv41 import rope_inplace

    torch.manual_seed(419)
    width = implementation._LAYOUTS[fmt][3]
    values = torch.randn(257, 512, device=device, dtype=torch.bfloat16)
    positions = torch.arange(257, device=device, dtype=torch.int64)
    angles = (
        torch.arange(512, device=device)[:, None].float()
        * torch.arange(32, device=device)[None, :]
        / 1000
    )
    rope = torch.cat((angles.cos(), angles.sin()), -1)
    slots = torch.arange(64, 321, device=device, dtype=torch.int32)
    slots[0] = -1
    storage = torch.zeros((6, 64 * width + 512), device=device, dtype=torch.uint8)
    cache = storage[:, : 64 * width].view(6, 64, width)
    expected = storage.clone()[:, : 64 * width].view(6, 64, width)
    weight = (
        torch.randn(512, device=device, dtype=torch.bfloat16) if normalize else None
    )
    normalized = values
    if normalize:
        from tokenspeed_kernel.ops.layernorm import rmsnorm

        normalized = rmsnorm(values, weight, eps=1e-6, residual=None, out=None)
    rotated = rope_inplace(normalized.clone(), positions, rope, None)
    dsv41.cache_scatter(rotated, expected, slots, fmt)
    out = torch.empty_like(values)
    dsv41.swa_rope_scatter(
        values, positions, rope, cache, slots, fmt, out, weight, 1e-6
    )
    reference = dsv41.cache_unpack(dsv41.cache_pack(rotated, fmt, None), fmt, None)
    if normalize:
        error = (out.float() - reference.float()).square().mean().sqrt()
        assert error < 1e-3 * reference.float().square().mean().sqrt()
        stored = dsv41.cache_gather(cache, slots[1:], fmt, None)
        torch.testing.assert_close(stored, out[1:], rtol=0, atol=0)
        assert not cache[0].any()
    else:
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["swa_v4", "global_v4"])
def test_v4_rows_quantize_head_and_carry_rope_tail_verbatim(device, fmt):
    """The V4 row is three planes: E4M3 NoPE, BF16 RoPE, E8M0 scales with pad."""
    dim, group, values, row_bytes, quantized = implementation._LAYOUTS[fmt]
    scale_bytes = implementation._scale_bytes(fmt)
    assert (dim, group, quantized, row_bytes) == (512, 64, 448, 584)
    assert values + (dim - quantized) * 2 + scale_bytes == row_bytes

    torch.manual_seed(41)
    rows = 137
    x = (torch.randn(rows, dim, device=device, dtype=torch.bfloat16) * 0.7).float()
    x[0].zero_()
    x[1, :quantized] = 1e-6
    cache = torch.zeros(
        ((rows + 63) // 64, 64, row_bytes), dtype=torch.uint8, device=device
    )
    slots = torch.arange(rows, device=device, dtype=torch.int32)
    dsv41.cache_scatter(x.to(torch.bfloat16), cache, slots, fmt)
    got = dsv41.cache_gather(cache, slots, fmt, None).float()

    # Independent E8M0-per-group oracle for the head; the tail is not quantized.
    head = x[:, :quantized].reshape(rows, -1, group)
    amax = head.abs().amax(dim=-1).clamp_min(1e-4) / 448.0
    bits = amax.view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).int()
    scale = torch.exp2(exponent.float()).unsqueeze(-1)
    quant = (head / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        got[:, :quantized], (quant.float() * scale).reshape(rows, quantized)
    )
    torch.testing.assert_close(
        got[:, quantized:], x[:, quantized:].to(torch.bfloat16).float(), rtol=0, atol=0
    )

    # Planes are page-planar, so the scale plane starts at a flat page offset.
    flat = cache[0].reshape(-1)
    plane = 64 * (values + (dim - quantized) * 2)
    used = quantized // group
    pad = torch.stack(
        [
            flat[plane + row * scale_bytes + used : plane + (row + 1) * scale_bytes]
            for row in range(64)
        ]
    )
    assert int(pad.ne(0).sum()) == 0


def test_compressor_fused_norm_preserves_pooled_bf16_boundary(device):
    torch.manual_seed(420)
    projection = torch.randn(17, 1024, device=device)
    content, scores = projection[:, :512], projection[:, 512:]
    previous = torch.arange(17, device=device) - 1
    slots = torch.full((17,), 2, device=device, dtype=torch.int32)
    active = torch.arange(17, device=device) % 2 == 1
    tail = torch.randn(5, 2, 2, 512, device=device)
    weight = torch.randn(512, device=device, dtype=torch.bfloat16)
    raw = dsv41.compressor_pool(
        content, scores, previous, tail, slots, active, None, None, 0.0
    )
    maximum = torch.maximum(scores[previous], scores)
    old_exp = torch.exp(scores[previous] - maximum)
    new_exp = torch.exp(scores - maximum)
    expected_raw = (content[previous] * old_exp + content * new_exp) / (
        old_exp + new_exp
    )
    expected_raw[~active] = 0
    torch.testing.assert_close(raw, expected_raw, rtol=1e-6, atol=1e-6)
    rounded = raw.bfloat16().float()
    expected = (
        rounded
        * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-20)
        * weight.float()
    ).bfloat16()
    actual = dsv41.compressor_pool(
        content, scores, previous, tail, slots, active, None, weight, 1e-20
    )
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.004)
    assert not torch.any(actual[~active])


def test_compressor_pool_token_count_and_capacity_do_not_recompile(device):
    """Token counts and tail capacities must not key the compile cache."""

    def run(count, pages):
        tail = torch.empty(pages, 2, 2, 512, device=device)
        tail[-1].normal_()
        projection = torch.randn(count, 1024, device=device)
        content, scores = projection[:, :512], projection[:, 512:]
        previous = torch.arange(count, device=device) - 1
        slots = torch.full((count,), pages * 2 - 1, device=device, dtype=torch.int32)
        active = torch.arange(count, device=device) % 2 == 1
        active[0] = True
        pooled = dsv41.compressor_pool(
            content, scores, previous, tail, slots, active, None, None, 0.0
        )
        # Row 0 pairs with the tail slot; every later active row pairs with
        # its predecessor in this forward.
        prior_content = torch.cat([tail[-1, 1, 0][None], content[:-1]])
        prior_scores = torch.cat([tail[-1, 1, 1][None], scores[:-1]])
        maximum = torch.maximum(prior_scores, scores)
        old_exp = torch.exp(prior_scores - maximum)
        new_exp = torch.exp(scores - maximum)
        expected = (prior_content * old_exp + content * new_exp) / (old_exp + new_exp)
        expected[~active] = 0
        torch.testing.assert_close(pooled, expected, rtol=1e-6, atol=1e-6)

    # Warm at a fixed capacity, then vary both token count and capacity.
    run(32, 5)
    run(33, 5)
    with assert_no_triton_compile(implementation._compressor_pool):
        for count, pages in ((48, 3), (97, 16), (130, 17), (1483, 32), (4096, 33)):
            run(count, pages)

    # The last page starts at 2**31 FP32 elements, beyond int32 offsets.
    pages = (1 << 20) + 1
    required_bytes = pages * 2 * 2 * 512 * torch.float32.itemsize
    if torch.cuda.mem_get_info(device)[0] < required_bytes + (256 << 20):
        pytest.skip("requires 8 GiB of free device memory")
    # AMD may compile a separate variant for buffers larger than 2 GiB.
    run(32, pages)


def _native_available():
    from tokenspeed_kernel.ops.attention.dsv41.flash_mla import (
        is_flash_mla_v41_available,
    )

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
        and is_flash_mla_v41_available()
    )


def _native_formats():
    """Return the (swa, global) formats this target's FlashMLA can read.

    sm90 carries the V4 cache reader only; sm100 adds V4.1 and its FP4 rows.
    """
    if torch.cuda.get_device_capability()[0] == 9:
        return "swa_v4", "global_v4"
    return "swa", "global"


def _native_cache(rows, fmt):
    width = implementation._LAYOUTS[fmt][3]
    backing = torch.zeros(
        (rows // 64 + 1, 64 * width + 512), device="cuda", dtype=torch.uint8
    )
    field = backing[:, : 64 * width].as_strided(
        (rows // 64 + 1, 64, width), (backing.stride(0), width, 1)
    )
    values = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(64, rows + 64, device="cuda", dtype=torch.int32)
    dsv41.cache_scatter(values, field, slots, fmt)
    return backing, field


@pytest.mark.skipif(
    not _native_available(), reason="FlashMLA V4.1 on Hopper or newer required"
)
@pytest.mark.parametrize("batch", [1, 16, 17])
@pytest.mark.parametrize("extra_width", [0, 512, 768])
def test_native_decode_graph_refresh_slots_lengths_and_idle(batch, extra_width):
    torch.manual_seed(741)
    swa_fmt, global_fmt = _native_formats()
    sa, swa = _native_cache(256, swa_fmt)
    ga, glob = _native_cache(1024, global_fmt)
    before_s, before_g = sa.clone(), ga.clone()
    q = torch.randn(batch, 16, 512, dtype=torch.bfloat16, device="cuda") * 0.3
    ss = (
        torch.arange(64, 192, dtype=torch.int32, device="cuda")
        .expand(batch, -1)
        .clone()
    )
    gs = (
        torch.arange(64, 64 + extra_width, dtype=torch.int32, device="cuda")
        .expand(batch, -1)
        .clone()
    )
    sl = torch.zeros(batch, dtype=torch.int32, device="cuda")
    gl = torch.zeros_like(sl)
    sink = torch.linspace(-2, 2, 16, device="cuda")

    def run(schedule):
        return dsv41.selected_attention(
            q,
            swa,
            ss,
            sl,
            glob if extra_width else None,
            gs if extra_width else None,
            gl if extra_width else None,
            sink,
            512**-0.5,
            None,
            256,
            schedule,
            None,
            None,
            lse=None,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(dsv41.new_attention_schedule())
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(dsv41.new_attention_schedule())
    for step in range(4):
        sl.fill_(128 if step != 3 else 0)
        gl.fill_(min(extra_width, (1, 255, 512, 0)[step]))
        ss.copy_(ss.roll(1, dims=1))
        gs.copy_(gs.roll(7, dims=1))
        if batch > 1:
            sl[-1] = 0
            gl[-1] = 0
        graph.replay()
        eager = run(dsv41.new_attention_schedule())
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        parts = [dsv41.cache_gather(swa, ss, swa_fmt, None).float()]
        masks = [torch.arange(128, device="cuda")[None, :] < sl[:, None]]
        if extra_width:
            parts.append(dsv41.cache_gather(glob, gs, global_fmt, None).float())
            masks.append(
                torch.arange(extra_width, device="cuda")[None, :] < gl[:, None]
            )
        kv = torch.cat(parts, 1)
        valid = torch.cat(masks, 1)
        logits = torch.bmm(q.float(), kv.transpose(1, 2)) * 512**-0.5
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        probs = torch.cat(
            (logits, sink[None, :, None].expand(batch, -1, 1)), -1
        ).softmax(-1)[..., :-1]
        expected = torch.bmm(probs, kv).bfloat16()
        torch.testing.assert_close(captured, expected, rtol=0.02, atol=0.004)
    assert torch.equal(sa, before_s) and torch.equal(ga, before_g)


@pytest.mark.skipif(
    not _native_available(), reason="FlashMLA V4.1 on Hopper or newer required"
)
@pytest.mark.parametrize("query_chunk_size", [16, 256])
def test_native_prefill_workspace_matches_joint_softmax(query_chunk_size):
    torch.manual_seed(742)
    q = torch.randn(33, 16, 512, dtype=torch.bfloat16, device="cuda") * 0.3
    kv = torch.randn(1024, 1, 512, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(0, 1024, (33, 640), dtype=torch.int32, device="cuda")
    indices[0] = -1
    indices[1, 127:] = -1
    sink = torch.linspace(-2, 2, 16, device="cuda")
    out = dsv41.selected_attention(
        q,
        None,
        None,
        None,
        None,
        None,
        None,
        sink,
        512**-0.5,
        None,
        query_chunk_size,
        None,
        kv,
        indices,
        lse=None,
    )
    selected = kv[indices.clamp_min(0).long(), 0].float()
    logits = torch.bmm(q.float(), selected.transpose(1, 2)) * 512**-0.5
    logits.masked_fill_(indices[:, None, :] < 0, -torch.inf)
    probs = torch.cat((logits, sink[None, :, None].expand(33, -1, 1)), -1).softmax(-1)[
        ..., :-1
    ]
    expected = torch.bmm(probs, selected).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.004)


def test_attention_head_copy_output_rows_past_int32(device):
    """Copy attention heads into rows whose offsets pass 2**31 elements.

    The destination is the caller's, so its row stride is too. Three rows a
    2**30 stride apart put the last write at 2**31 elements, and a guard before
    the base holds a sentinel a wrapped write would overwrite.
    """
    tokens, heads, dim = 3, 64, 512
    stride, guard = 2**30, 2**31
    needed = guard + 2 * stride + heads * dim
    if torch.cuda.mem_get_info(device)[0] < needed * 2 + (512 << 20):
        pytest.skip("requires 9 GiB of free device memory")
    torch.manual_seed(743)
    storage = torch.empty(needed, dtype=torch.bfloat16, device=device)
    storage[: heads * dim].fill_(-7.0)
    out = storage[guard:].as_strided((tokens, heads, dim), (stride, dim, 1))
    out.fill_(float("nan"))
    src = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device=device)

    implementation._copy_attention_heads(out, src, None, None)
    torch.testing.assert_close(out, src, rtol=0, atol=0)
    sentinel = storage[: heads * dim]
    torch.testing.assert_close(
        sentinel, torch.full_like(sentinel, -7.0), rtol=0, atol=0
    )


@pytest.mark.parametrize("n", [0, 1, 16, 128, 129, 257, 8192, 32768])
@pytest.mark.parametrize("target", ["cpu", "cuda"])
def test_compressor_metadata_consecutive_requests_and_refresh(n, target):
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    indices = torch.arange(n)
    requests = (indices // 257).to(torch.int32)
    positions = indices % 257 + 3 + requests % 2
    # Reorder whole request spans, retaining each request's internal order.
    order = torch.argsort(-requests, stable=True)
    positions, requests = positions[order], requests[order]
    positions[11::31] = -1
    requests[19::43] = -1
    p = torch.empty(n * 2, dtype=torch.int64, device=target)[::2]
    r = torch.empty(n * 3, dtype=torch.int32, device=target)[::3]
    p.copy_(positions)
    r.copy_(requests)
    table = torch.ones(
        (max(1, (n + 256) // 257), 264), dtype=torch.int32, device=target
    )[:, ::2]
    table[:, 13::17] = 0
    table[:, 29::37] = 4  # Out-of-capacity pages must resolve to -1 too.
    out = tuple(
        torch.empty(n, dtype=dtype, device=target)
        for dtype in (
            torch.bool,
            torch.int64,
            torch.int32,
            torch.int64,
            torch.int64,
            torch.int64,
        )
    )

    def prepare():
        dsv41.compressor_metadata(p, r, table, 4, *out)

    prepare()
    for got, want in zip(
        out, torch_dsv41_compressor_pairs(p, r, table, 4), strict=True
    ):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    if target == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare()
    for step in range(3):
        p.add_(1)
        table[:, :4] = 2 if step % 2 else 0
        if step == 2:
            p.fill_(-1)
        for value in out:
            value.fill_(1)
        if target == "cuda":
            graph.replay()
        else:
            prepare()
        for got, want in zip(
            out, torch_dsv41_compressor_pairs(p, r, table, 4), strict=True
        ):
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


def test_compressor_metadata_table_width_does_not_recompile():
    """Chunked prefill widens the tail table every chunk; the kernel must not
    key its compile cache on the table geometry."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")

    def run(rows, width):
        p = torch.arange(rows, dtype=torch.int64, device="cuda")
        r = torch.zeros(rows, dtype=torch.int32, device="cuda")
        table = torch.ones((3, width), dtype=torch.int32, device="cuda")
        table[:, 5::7] = 0
        out = tuple(
            torch.empty(rows, dtype=dtype, device="cuda")
            for dtype in (
                torch.bool,
                torch.int64,
                torch.int32,
                torch.int64,
                torch.int64,
                torch.int64,
            )
        )
        implementation.compressor_metadata(p, r, table, 4, *out)
        for got, want in zip(
            out, torch_dsv41_compressor_pairs(p, r, table, 4), strict=True
        ):
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)

    # Warm both integer-specialization classes (16-divisible and not); Triton
    # still keys on those two properties for runtime scalars.
    run(64, 128)
    run(64, 132)
    with assert_no_triton_compile(implementation._compressor_metadata):
        for rows, width in ((64, 192), (128, 196), (96, 388), (64, 1024)):
            run(rows, width)


@pytest.mark.parametrize("target", ["cpu", "cuda"])
@pytest.mark.parametrize("width", [1, 3, 5, 6, 129])
@pytest.mark.parametrize("bs", [0, 3])
def test_decode_rows_packed_request_bounds_padding_and_replay(target, width, bs):
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    seq = torch.tensor([0, 133], dtype=torch.int32, device=target)
    pools = torch.tensor([19, 7], dtype=torch.int64, device=target)
    storage = [
        torch.full((n + 4,), 777, dtype=dtype, device=target)
        for n, dtype in (
            (bs * width, torch.int64),
            (bs * width, torch.int64),
            (bs, torch.int32),
            (bs, torch.int64),
        )
    ]
    out = tuple(tensor[2:-2] for tensor in storage)

    def prepare(actual, extends):
        dsv41.decode_rows(
            seq, pools, *out, actual, extends, width, query_start_loc=None
        )

    def check(actual, extends):
        lengths = [max(int(seq[i]), width) if i < actual else 0 for i in range(bs)]
        positions, requests = [], []
        for i in range(bs):
            live = extends <= i < actual
            positions.extend(
                range(lengths[i] - width, lengths[i]) if live else [-1] * width
            )
            requests.extend([i if live else -1] * width)
        expected = (
            positions,
            requests,
            lengths,
            [int(pools[i]) if i < actual else -1 for i in range(bs)],
        )
        for got, want, guarded in zip(out, expected, storage, strict=True):
            torch.testing.assert_close(
                got.cpu(), torch.tensor(want, dtype=got.dtype), rtol=0, atol=0
            )
            assert (guarded[:2] == 777).all() and (guarded[-2:] == 777).all()

    for actual, extends in ((2, 0), (1, 0), (0, 0), (2, 1), (2, 2)):
        actual, extends = min(actual, bs), min(extends, bs)
        for tensor in storage:
            tensor.fill_(777)
        prepare(actual, extends)
        check(actual, extends)
    if target == "cuda" and bs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare(2, 0)
        for step in range(3):
            seq.add_(1)
            pools.add_(3)
            for tensor in storage:
                tensor.fill_(777)
            graph.replay()
            check(2, 0)


def test_decode_rows_rejects_request_token_shape_confusion():
    with pytest.raises(ValueError, match="request-sized and packed token"):
        dsv41.decode_rows(
            torch.tensor([5, 10]),
            torch.tensor([19, 7]),
            torch.empty(6, dtype=torch.int64),
            torch.empty(6, dtype=torch.int64),
            torch.empty(6, dtype=torch.int32),
            torch.empty(6, dtype=torch.int64),
            2,
            0,
            3,
            query_start_loc=None,
        )


@pytest.mark.parametrize("target", ["cpu", "cuda"])
def test_decode_window_ragged_addresses_and_replay(target):
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    p = torch.tensor([-1, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 190, 191, -1], device=target)
    r = torch.tensor([-1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, -1], device=target)
    swa = torch.ones((3, 4), dtype=torch.int32, device=target)
    swa[2, 0] = 0  # A null page resolves to -1 rows; residency is not judged here.
    n = p.numel()
    out = (
        torch.empty(n, dtype=torch.int64, device=target),
        torch.empty((n, 128), dtype=torch.int32, device=target),
        torch.empty(n, dtype=torch.int32, device=target),
    )

    def prepare():
        dsv41.decode_window(p, r, *out, swa, 2)

    prepare()
    if target == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare()
    for step in range(3):
        if step == 1:
            swa[2, 0] = 1
        if step == 2:
            p.fill_(-1)
        for tensor in out:
            tensor.fill_(777)
        if target == "cuda":
            graph.replay()
        else:
            prepare()
        table = swa.cpu().tolist()

        def slot(position, request):
            if position < 0 or not 0 <= request < len(table):
                return -1
            column, offset = divmod(position, 64)
            if column >= len(table[request]):
                return -1
            page = table[request][column]
            return page * 64 + offset if 0 < page < 2 else -1

        coordinates = list(zip(p.cpu().tolist(), r.cpu().tolist(), strict=True))
        expected = (
            [slot(position, request) for position, request in coordinates],
            [
                [
                    slot(wanted, request) if wanted <= position else -1
                    for wanted in range(
                        max(0, position - 127), max(0, position - 127) + 128
                    )
                ]
                for position, request in coordinates
            ],
            [min(max(position + 1, 0), 128) for position, _ in coordinates],
        )
        for got, want in zip(out, expected, strict=True):
            torch.testing.assert_close(
                got.cpu(), torch.tensor(want, dtype=got.dtype), rtol=0, atol=0
            )


def _rope_table(device, positions=4096, rotary_dim=64):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim)
    )
    angles = torch.arange(positions, device=device)[:, None].float() * inv_freq
    return torch.cat((angles.cos(), angles.sin()), -1).contiguous()


def test_dspark_rows_matches_norm_rope_round_trip_and_scatter(device):
    """One launch reproduces kv_norm -> RoPE -> SWA codec round trip -> scatter."""
    from tokenspeed_kernel.ops.attention.dsv41 import rope_inplace

    torch.manual_seed(4)
    rows, eps = 12, 1e-6
    merged = torch.randn(rows, 3 * 512, device=device, dtype=torch.bfloat16) * 3
    values = merged[:, 512:1024]
    weight = (torch.rand(512, device=device) + 0.5).to(torch.bfloat16)
    table = _rope_table(device)
    positions = torch.randint(0, 4000, (rows,), device=device)
    slots = torch.arange(64, 64 + 7 * rows, 7, device=device, dtype=torch.int32)
    slots[3], slots[5] = -1, 8 * 64 + 5
    window = torch.zeros(8, 64, 512, device=device, dtype=torch.bfloat16)

    def round_trip(x):
        return dsv41.cache_unpack(dsv41.cache_pack(x, "swa", None), "swa", None)

    normalized = values.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + eps
    )
    normalized = (weight.float() * normalized).to(torch.bfloat16)
    expected = round_trip(rope_inplace(normalized.clone(), positions, table, None))

    out = torch.empty_like(values)
    dsv41.dspark_rows(values, weight, eps, positions, table, window, slots, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    live = (slots >= 0) & (slots < 8 * 64)
    torch.testing.assert_close(
        window[slots[live].long() // 64, slots[live].long() % 64],
        expected[live],
        rtol=0,
        atol=0,
    )
    assert window.reshape(-1, 512).ne(0).any(-1).sum() == int(live.sum())
    # Without the norm the input is used as is; without the table nothing rotates.
    torch.testing.assert_close(
        dsv41.dspark_rows(
            values, None, None, positions, table, None, None, torch.empty_like(values)
        ),
        round_trip(rope_inplace(values.clone(), positions, table, None)),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        dsv41.dspark_rows(
            values, None, None, None, None, None, None, torch.empty_like(values)
        ),
        round_trip(values),
        rtol=0,
        atol=0,
    )
    dsv41.dspark_rows(
        values[:0], weight, eps, positions[:0], table, window, slots[:0], None
    )
    for bad in (
        dict(norm_weight=None),
        dict(norm_eps=None),
        dict(positions=None),
        dict(cos_sin_cache=None),
        dict(slots=None),
        dict(window=window[:, :, :256]),
        dict(window=None, out=None),
        dict(values=values.t()),
    ):
        kwargs = dict(
            values=values,
            norm_weight=weight,
            norm_eps=eps,
            positions=positions,
            cos_sin_cache=table,
            window=window,
            slots=slots,
            out=None,
        )
        kwargs.update(bad)
        with pytest.raises(ValueError):
            dsv41.dspark_rows(**kwargs)


def test_dspark_anchors_pick_last_accepted_verify_rows(device):
    bs, extends, width, spec = 4, 1, 6, 6
    decodes = bs - extends
    tokens = torch.arange(100, 100 + extends + decodes * width, device=device).to(
        torch.int32
    )
    accept = torch.tensor([1, 4, 0, 9], device=device, dtype=torch.int32)
    positions = torch.arange(1000, 1000 + decodes * width, device=device)
    next_tokens = torch.zeros(bs, spec, device=device, dtype=torch.int32)
    start = torch.zeros(decodes, device=device, dtype=torch.int64)
    dsv41.dspark_anchors(
        tokens,
        accept,
        positions,
        extends,
        width,
        next_tokens,
        start,
        num_prefill_outputs=extends,
        decode_query_indptr=None,
    )
    # Extend rows keep their sampled token; decode rows take the token at the
    # last accepted verify row, with accept lengths clamped into 1..width.
    assert next_tokens.tolist() == [[100] * 6, [104] * 6, [107] * 6, [118] * 6]
    assert start.tolist() == [1003, 1006, 1017]
    cpu_next = torch.zeros(bs, spec, dtype=torch.int32)
    cpu_start = torch.zeros(decodes, dtype=torch.int64)
    dsv41.dspark_anchors(
        tokens.cpu(),
        accept.cpu(),
        positions.cpu(),
        extends,
        width,
        cpu_next,
        cpu_start,
        num_prefill_outputs=extends,
        decode_query_indptr=None,
    )
    assert torch.equal(cpu_next, next_tokens.cpu()) and torch.equal(
        cpu_start, start.cpu()
    )
    with pytest.raises(ValueError):
        dsv41.dspark_anchors(
            tokens[:-1],
            accept,
            positions,
            extends,
            width,
            next_tokens,
            start,
            num_prefill_outputs=extends,
            decode_query_indptr=None,
        )


def test_dspark_block_expands_anchors_and_window_addressing(device):
    torch.manual_seed(5)
    n, window, block, hc, rows_per_page, noise = 3, 128, 5, 4, 64, 77
    history = torch.randint(-1, 600, (n, window), device=device, dtype=torch.int32)
    history[0, :100] = -1
    # The drafter hands over the bonus column of its [bs, spec] token table.
    bonus = torch.tensor([[3, 9], [4, 9], [5, 9]], device=device, dtype=torch.int32)[
        :, 0
    ]
    start = torch.tensor([10, 200, 7], device=device)
    outputs = dsv41.dspark_block(bonus, start, history, noise, rows_per_page, hc, block)
    ids, positions, pre_mix, requests, page, row, indices = outputs
    assert [t.dtype for t in outputs] == [
        torch.int32,
        torch.int64,
        torch.float32,
        torch.int64,
        torch.int64,
        torch.int64,
        torch.int32,
    ]
    assert ids.view(n, block)[:, 0].tolist() == [3, 4, 5]
    assert ids.view(n, block)[:, 1:].eq(noise).all()
    assert positions.view(n, block).tolist() == [
        [s + 1 + j for j in range(block)] for s in (10, 200, 7)
    ]
    assert pre_mix.tolist() == [[1.0, 0.0, 0.0, 0.0]] * (n * block)
    assert requests.tolist() == [r for r in range(n) for _ in range(block)]
    clamped = history.clamp_min(0).long()
    assert torch.equal(page, clamped // rows_per_page)
    assert torch.equal(row, clamped % rows_per_page)
    width = window + block
    expected = torch.arange(n * width, device=device, dtype=torch.int32).view(n, width)
    expected[:, :window].masked_fill_(history < 0, -1)
    assert torch.equal(
        indices.view(n, block, width), expected[:, None, :].expand(-1, block, -1)
    )
    cpu = dsv41.dspark_block(
        bonus.cpu(), start.cpu(), history.cpu(), noise, rows_per_page, hc, block
    )
    for got, reference in zip(outputs, cpu, strict=True):
        assert torch.equal(got.cpu(), reference.to(got.dtype))
    with pytest.raises(ValueError):
        dsv41.dspark_block(bonus[:2], start, history, noise, rows_per_page, hc, block)


def test_dspark_sparse_outputs_preserve_inactive_rows_without_recompile(device):
    extends, width = 2, 3
    accept = torch.tensor([1, 0, 2], dtype=torch.int32, device=device)
    positions = torch.tensor([50, 51, 52], dtype=torch.int64, device=device)
    next_tokens = torch.full((3, width), -7, dtype=torch.int32, device=device)
    starts = torch.empty(1, dtype=torch.int64, device=device)

    def run(outputs):
        tokens = torch.arange(
            10, 10 + outputs + width, dtype=torch.int32, device=device
        )
        next_tokens.fill_(-7)
        dsv41.dspark_anchors(
            tokens,
            accept,
            positions,
            extends,
            width,
            next_tokens,
            starts,
            num_prefill_outputs=outputs,
            decode_query_indptr=None,
        )
        assert next_tokens[:outputs, 0].tolist() == list(range(10, 10 + outputs))
        assert (next_tokens[outputs:extends] == -7).all()
        assert next_tokens[extends].tolist() == [10 + outputs + 1] * width
        assert starts.tolist() == [51]

    run(1)
    with assert_no_triton_compile(implementation._dspark_anchors_kernel):
        run(0)
        run(2)


@pytest.mark.parametrize(
    "heads,rows",
    [(32, rows) for rows in (0, 6, 12, 24, 48, 96, 128, 129, 192, 257)]
    + [(heads, 257) for heads in (4, 8, 13, 16)],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_indexer_rope_weights_match_separate_operations(
    device, heads, rows, strided, dtype
):
    torch.manual_seed(731)
    dim = 128
    query_storage = torch.randn(rows, heads * 2, dim, dtype=dtype, device=device)
    weight_storage = torch.randn(rows, heads * 2, dtype=dtype, device=device)
    queries = (
        query_storage[:, ::2] if strided else query_storage[:, :heads].contiguous()
    )
    weights = (
        weight_storage[:, ::2] if strided else weight_storage[:, :heads].contiguous()
    )
    original_q, original_w = queries.clone(), weights.clone()
    query_gaps, weight_gaps = (
        query_storage[:, 1::2].clone(),
        weight_storage[:, 1::2].clone(),
    )
    angles = torch.randn(512, 32, device=device)
    table = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.arange(rows, device=device, dtype=torch.int64) - 1
    scale = dim**-0.5 * heads**-0.5
    expected_q = dsv41.rope_inplace(original_q.clone(), positions, table, "triton")
    expected_w = original_w * scale
    q, w = dsv41.rope_weights_inplace(
        queries, positions, table, weights, scale, "triton"
    )
    assert q is queries and w is weights
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(w, expected_w, rtol=0, atol=0)
    if strided:
        torch.testing.assert_close(query_storage[:, 1::2], query_gaps, rtol=0, atol=0)
        torch.testing.assert_close(weight_storage[:, 1::2], weight_gaps, rtol=0, atol=0)
    # Changing batch rows must reuse the compiled scalar parameters.
    if rows >= 12:
        dsv41.rope_weights_inplace(
            queries[:6], positions[:6], table, weights[:6], scale, "triton"
        )
        with assert_no_triton_compile(implementation._rope_inplace_kernel):
            dsv41.rope_weights_inplace(
                queries[:12], positions[:12], table, weights[:12], scale, "triton"
            )

    if heads == 32 and rows >= 192:
        # Both grid sizes use separate scaling and share the ordinary RoPE JIT.
        with assert_no_triton_compile(implementation._rope_inplace_kernel):
            dsv41.rope_weights_inplace(
                queries[:129], positions[:129], table, weights[:129], scale, "triton"
            )


@pytest.mark.parametrize("rows", [128, 129])
def test_indexer_rope_weight_dispatch_graph_replay(device, rows):
    torch.manual_seed(4096)
    queries = torch.randn(rows, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(rows, 32, dtype=torch.bfloat16, device=device)
    initial_q, initial_w = queries.clone(), weights.clone()
    positions = torch.zeros(rows, dtype=torch.int64, device=device)
    angles = torch.randn(64, 32, device=device)
    table = torch.cat((angles.cos(), angles.sin()), dim=-1)
    scale = 128**-0.5 * 32**-0.5
    dsv41.rope_weights_inplace(queries, positions, table, weights, scale, "triton")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_w = dsv41.rope_weights_inplace(
            queries, positions, table, weights, scale, "triton"
        )
    for position in (-1, 0, 17):
        queries.copy_(initial_q)
        weights.copy_(initial_w)
        positions.fill_(position)
        expected_q = dsv41.rope_inplace(initial_q.clone(), positions, table, "triton")
        graph.replay()
        assert actual_q is queries and actual_w is weights
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_w, initial_w * scale, rtol=0, atol=0)


def _clean_selection_reference(values, lengths, table, pages, capacity, size, align):
    """clean_logits_for_selection on [0, end); NaN marks what it leaves unspecified."""
    width = (capacity + align - 1) // align * align
    out = torch.full((values.shape[0], width), torch.nan, dtype=values.dtype)
    column = torch.arange(capacity)
    page = table[:, column // size]
    valid = (column[None] < lengths.clamp(0, capacity)[:, None]) & (
        (page >= 0) & (page < pages)
    )
    out[:, :capacity] = values[:, :capacity].masked_fill(~valid, -torch.inf)
    return out


def _mask_table_pages_reference(values, lengths, table, pages, size):
    capacity = table.shape[1] * size
    column = torch.arange(capacity)
    page = table[:, column // size]
    hole = (column[None] < lengths.clamp(0, capacity)[:, None]) & (
        (page < 0) | (page >= pages)
    )
    values[:, :capacity].masked_fill_(hole, -torch.inf)
    return values


@pytest.mark.parametrize("alignment_bytes", [256, 2048])
def test_paged_scores_mask_native_rows_in_place(monkeypatch, alignment_bytes):
    """Paged scoring selects from DeepGEMM's rows masked in place.

    The native rows come back at the cleaned copy's padded width, equal to it
    on [0, end). When DeepGEMM's row stride misses DeepSelect's alignment,
    paged scoring keeps the cleaned copy.
    """
    from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as native

    rows, width, pages, capacity = 3, 5, 9, 5 * 64
    table = torch.tensor(
        [[1, -1, 2, 3, 99], [4, 5, 6, 7, 8], [-1, 0, 1, 2, 3]], dtype=torch.int32
    )
    lengths = torch.tensor([200, 400, 0], dtype=torch.int32)
    # DeepGEMM pads each BF16 row to 1024 bytes and returns the capacity slice.
    torch.manual_seed(5)
    storage = torch.randn(rows, 512).to(torch.bfloat16)
    storage[:, capacity:] = torch.nan
    original = storage.clone()
    calls = []

    class Api:
        def get_num_sms(self):
            return 4

        def get_paged_mqa_logits_metadata(self, context_lens, size, sms, indices):
            return torch.zeros(sms + 1, 2, dtype=torch.int32)

        def fp8_fp4_paged_mqa_logits(self, **kwargs):
            return storage[:, : kwargs["max_context_len"]]

    def clean(*args):
        calls.append("clean")
        return _clean_selection_reference(*args)

    def mask(*args):
        calls.append("mask")
        return _mask_table_pages_reference(*args)

    monkeypatch.setattr(native, "_api", lambda queries: Api())
    monkeypatch.setattr(
        native,
        "safe_metadata",
        lambda t, lens, p, cap, size: (
            t.masked_fill((t < 0) | (t >= p), 0),
            lens.clamp(0, cap).to(torch.int32),
        ),
    )
    monkeypatch.setattr(native, "clean_logits_for_selection", clean)
    monkeypatch.setattr(native, "mask_invalid_table_pages", mask)
    monkeypatch.setattr(native, "deep_select_alignment", lambda: (alignment_bytes, 32))
    scores, ends, blocks = native._paged_scores(
        (
            torch.zeros(rows, 32, 64, dtype=torch.uint8),
            torch.zeros(rows, 32, dtype=torch.int32),
        ),
        torch.zeros(pages, 64 * 68, dtype=torch.uint8),
        torch.zeros(rows, 32),
        table,
        lengths,
        capacity,
        64,
        None,
        0,
        rows,
        False,
    )
    alignment = alignment_bytes // 2
    native_rows = 512 * 2 % alignment_bytes == 0
    assert calls == (["mask"] if native_rows else ["clean"])
    assert blocks is None and ends.tolist() == [200, 320, 0]
    expected = _clean_selection_reference(
        original, ends, table, pages, capacity, 64, alignment
    )
    assert scores.shape == expected.shape
    for row, end in enumerate(ends.tolist()):
        torch.testing.assert_close(scores[row, :end], expected[row, :end])
        if end:
            # Same top-k over [0, end) as the cleaned copy.
            k = min(16, end)
            torch.testing.assert_close(
                scores[row, :end].float().topk(k).indices.sort().values,
                expected[row, :end].float().topk(k).indices.sort().values,
            )
    assert torch.isinf(scores[0, 64:128]).all()
    if native_rows:
        assert scores.data_ptr() == storage.data_ptr()
        assert scores.stride(0) == 512
        # Past each row's end the native rows are left as DeepGEMM wrote them.
        for row, end in enumerate(ends.tolist()):
            assert torch.equal(storage[row, end:].isnan(), original[row, end:].isnan())
            torch.testing.assert_close(
                storage[row, end:capacity], original[row, end:capacity]
            )


def test_padded_rows_view_only_within_the_native_row():
    from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as native

    storage = torch.zeros(3, 512, dtype=torch.bfloat16)
    logits = storage[:, :320]
    padded = native._padded_rows(logits, 128)
    assert padded.shape == (3, 384) and padded.stride() == (512, 1)
    assert padded.data_ptr() == storage.data_ptr()
    assert native._padded_rows(logits, 64) is logits
    assert native._padded_rows(logits, 1024) is logits
    # A row stride without room for the padded width keeps the logits.
    narrow = torch.zeros(3, 320, dtype=torch.bfloat16)
    assert native._padded_rows(narrow, 128) is narrow


@pytest.fixture
def triton_device():
    """A GPU, or the CPU when TRITON_INTERPRET=1 runs the Triton kernels there."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    pytest.skip("requires CUDA/ROCm or TRITON_INTERPRET=1")


def _bits(values):
    return values.view(torch.int16 if values.element_size() == 2 else torch.int32)


@pytest.mark.parametrize("table_layout", ["contiguous", "strided", "int64"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mask_invalid_table_pages_chunks_match_the_reference(
    triton_device, dtype, table_layout
):
    """The chunk-parallel paged mask writes exactly the reference's -inf stores.

    1M-token tables (16385 pages, 17 chunks of 1024) with ends from below zero
    to past the table, many not multiples of 64. Holes sit in the first chunk,
    across a chunk boundary, in the last live chunk and on each row's last
    page, plus past each end, where the kernel may not write; one long row has
    no hole at all. The whole buffer, padding included, must match bitwise.
    """
    device = triton_device
    torch.manual_seed(61)
    pages, width = 70, 16385
    capacity = width * 64
    ends = [-5, 0, 1, 63, 64, 65, 1024 * 64, 1024 * 64 + 1, 2048 * 64 - 7]
    ends += [703_001, 703_001, capacity - 1, capacity + 100]
    rows = len(ends)
    table = torch.randint(0, pages, (rows, width), dtype=torch.int32)
    for row, end in enumerate(ends):
        live = (min(max(end, 0), capacity) + 63) // 64
        if row == 10:
            continue  # A long row without holes: nothing to write.
        table[row, [0, 3, 1023, 1024, 2047]] = torch.tensor(
            [pages, -1, -1, -2, 99], dtype=torch.int32
        )
        table[row, max(live - 5, 0)] = pages + 9
        table[row, max(live - 1, 0)] = -1
        table[row, live : live + 4] = -1
        table[row, 1500::977] = -3
    if table_layout == "strided":
        strided = torch.zeros((rows, 2 * width), dtype=torch.int32, device=device)
        strided[:, ::2] = table.to(device)
        device_table = strided[:, ::2]
    else:
        device_table = table.to(
            device, torch.int64 if table_layout == "int64" else None
        )
    lengths = torch.tensor(ends, dtype=torch.int32, device=device)
    storage = torch.randn((rows, capacity + 128), dtype=dtype, device=device)
    storage[:, capacity:] = torch.nan

    reference = storage.to("cpu", copy=True)
    _mask_table_pages_reference(
        reference[:, :capacity], lengths.cpu(), table, pages, 64
    )
    out = implementation.mask_invalid_table_pages(
        storage[:, :capacity], lengths, device_table, pages, 64
    )
    assert out.data_ptr() == storage.data_ptr()
    assert torch.equal(_bits(storage).cpu(), _bits(reference))
    assert torch.isneginf(storage[2, 0]) and not torch.isinf(storage[10]).any()


def _fused_slots(rows, k, ratio, device):
    """Slot inputs covering padding, causality, null pages and page bounds."""
    positions = torch.tensor([-1, 0, 200, 160, 639, 700, 255, 64][:rows], device=device)
    # Request 3 is past the table; -1 is a padded query.
    requests = torch.tensor([-1, 0, 0, 1, 2, 3, 1, 2][:rows], device=device)
    # Page 0 is the null page (first_page 1); pages 8 and 9 are past the bound.
    # Rows 64-127 of request 0 sit on the null page, of request 1 past it.
    storage = torch.tensor(
        [[1, 0, 5, 9, 2, 3], [2, 9, 4, 8, 1, 7], [7, 6, 1, 2, 0, 5]],
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(2, dim=1)
    out = torch.full((rows, k), 4444, dtype=torch.int32, device=device)
    return dsv41.SelectionSlots(out, positions, requests, storage[:, ::2], ratio, 8)


def test_selection_slots_slices_keep_out_a_view():
    """Slicing SelectionSlots keeps out a view of the caller's tensor, and an
    index that would copy it is rejected rather than detaching the writes."""
    out = torch.zeros((4, 2), dtype=torch.int32)
    slots = dsv41.SelectionSlots(
        out,
        torch.arange(4),
        torch.zeros(4, dtype=torch.int64),
        torch.zeros((1, 2), dtype=torch.int32),
        1,
        2,
    )
    part = slots[1:3]
    part.out.fill_(7)
    assert out[1:3].eq(7).all() and out[0].eq(0).all() and out[3].eq(0).all()
    with pytest.raises(TypeError, match="slice indexing"):
        slots[torch.tensor([0, 2])]


def _finish_write_reference(rows, pool, out, slots):
    """The unfused tail: finish_selection, write_selection, then global_slots."""
    indices, scores, ends, candidates, width, block_size = rows
    selected = implementation.finish_selection(
        indices, scores, ends, candidates, width, block_size, True
    )
    chosen = None
    if pool is not None:
        chosen = implementation.finish_selection(
            *pool, None, pool[1].shape[1], 8, False
        )
    implementation.write_selection(selected, chosen, out)
    if slots is not None:
        slots.out.copy_(
            implementation.global_slots(
                out[0],
                slots.positions,
                slots.requests,
                slots.table,
                slots.ratio,
                slots.pages,
            )
        )


def _poisoned_outputs(rows, k, pool_k, device):
    shapes = ((rows, k), (rows,), (rows, pool_k), (rows,))
    return tuple(
        torch.full(shape, 12345, dtype=torch.int32, device=device) for shape in shapes
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["rows", "pool", "candidates"])
@pytest.mark.parametrize("ratio", [None, 1, 2])
def test_fused_selection_matches_finish_write_and_global_slots(
    triton_device, dtype, layout, ratio
):
    """finish_write_selection's one launch writes what the three passes write.

    Row picks (plain or candidate-compacted) and a source pass's block picks
    finish into poisoned outputs, with slots resolved from the same rows.
    """
    device = triton_device
    torch.manual_seed(1931)
    n, k = 8, 16
    width = 96
    capacity = width
    candidates = None
    block_size = 1
    if layout == "candidates":
        # Candidate-compacted columns map through their block; -1, repeated
        # and out-of-history blocks, and rows past the capacity, drop out.
        block_size = 8
        capacity = 100
        candidates = torch.randint(
            -1, 15, (n, width // 8), dtype=torch.int32, device=device
        )
        candidates[:, 0] = -1
    # Strided score rows, -inf holes and a NaN, as native logits arrive.
    scores = torch.randn(n, width + 32, device=device).to(dtype)[:, :width]
    scores[:, ::7] = -torch.inf
    scores[2, 5] = torch.nan
    # Rows 2 and 3 reach rows 64-95 of their requests: null and out-of-bound
    # pages.
    ends = torch.tensor([0, 1, 96, 70, 17, 40, 15, 3], dtype=torch.int32, device=device)
    # Unsorted picks with -1 padding, repeats and columns at or past end.
    indices = torch.randint(
        -1, width + 8, (n, k + 16), dtype=torch.int32, device=device
    )
    indices = indices[:, :k]
    picks = (indices, scores, ends, candidates, capacity, block_size)
    pool = None
    pool_k = 0
    if layout == "pool":
        pool_k = 32
        block_scores = torch.randn(n, 64, device=device)
        block_scores[:, ::5] = -torch.inf
        block_ends = torch.tensor(
            [0, 1, 7, 20, 33, 64, 64, 2], dtype=torch.int32, device=device
        )
        block_picks = torch.randint(
            -1, 70, (n, pool_k), dtype=torch.int32, device=device
        )
        pool = (block_picks, block_scores, block_ends)
    expected = _poisoned_outputs(n, k, pool_k, device)
    expected_slots = _fused_slots(n, k, ratio, device) if ratio else None
    _finish_write_reference(picks, pool, expected, expected_slots)
    actual = _poisoned_outputs(n, k, pool_k, device)
    actual_slots = _fused_slots(n, k, ratio, device) if ratio else None
    implementation.finish_write_selection(picks, pool, actual, actual_slots, False)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)
    assert (expected[1] > 0).any()
    if ratio:
        assert torch.equal(actual_slots.out, expected_slots.out)
        live = expected[0] >= 0
        # Some selected rows resolve; null, out-of-bound and future ones do not.
        assert (actual_slots.out[live] >= 0).any()
        assert (actual_slots.out[live] < 0).any()
    if pool is not None:
        assert (expected[3] > 0).any()


def test_mask_invalid_table_pages_rows_past_int32(triton_device):
    """The paged mask addresses score rows whose offsets pass 2**31 elements.

    Three rows a 2**30 stride apart put the last row at 2**31 elements, and a
    guard before the base holds a sentinel that a wrapped write would hit.
    """
    device = triton_device
    if device.type != "cuda":
        pytest.skip("requires CUDA")
    stride, guard, capacity = 2**30, 2**31, 4 * 64
    needed = guard + 2 * stride + capacity
    if torch.cuda.mem_get_info(device)[0] < needed * 2 + (256 << 20):
        pytest.skip("requires 9 GiB of free device memory")
    torch.manual_seed(62)
    storage = torch.empty(needed, dtype=torch.bfloat16, device=device)
    storage[:capacity].fill_(3.0)
    values = storage[guard:].as_strided((3, capacity), (stride, 1))
    values.copy_(torch.randn((3, capacity), dtype=torch.bfloat16, device=device))
    table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [1, -1, 3, 9]], dtype=torch.int32)
    lengths = torch.full((3,), capacity, dtype=torch.int32, device=device)
    reference = values.to("cpu", copy=True)
    _mask_table_pages_reference(reference, lengths.cpu(), table, 8, 64)

    implementation.mask_invalid_table_pages(values, lengths, table.to(device), 8, 64)
    assert torch.equal(_bits(values).cpu(), _bits(reference))
    assert torch.isneginf(values[2, 64:128]).all()
    sentinel = storage[:capacity]
    torch.testing.assert_close(sentinel, torch.full_like(sentinel, 3.0))


def test_fused_selection_reads_score_rows_past_int32(triton_device):
    """finish_write_selection reads score rows whose offsets pass 2**31 elements.

    Three score rows a 2**30 stride apart put the last read at 2**31 elements;
    a guard before the base holds large sentinel scores that a wrapped read
    would rank first. The fused launch must match the unfused passes run on a
    contiguous copy of the same rows.
    """
    device = triton_device
    if device.type != "cuda":
        pytest.skip("requires CUDA")
    n, k, width = 3, 16, 96
    stride, guard = 2**30, 2**31
    needed = guard + 2 * stride + width
    if torch.cuda.mem_get_info(device)[0] < needed * 2 + (256 << 20):
        pytest.skip("requires 9 GiB of free device memory")
    torch.manual_seed(1932)
    storage = torch.empty(needed, dtype=torch.bfloat16, device=device)
    storage[:width].fill_(100.0)
    scores = storage[guard:].as_strided((n, width), (stride, 1))
    scores.copy_(torch.randn((n, width), dtype=torch.bfloat16, device=device))
    ends = torch.full((n,), width, dtype=torch.int32, device=device)
    indices = torch.randint(-1, width, (n, k), dtype=torch.int32, device=device)

    expected = _poisoned_outputs(n, k, 0, device)
    _finish_write_reference(
        (indices, scores.contiguous(), ends, None, width, 1), None, expected, None
    )
    actual = _poisoned_outputs(n, k, 0, device)
    implementation.finish_write_selection(
        (indices, scores, ends, None, width, 1), None, actual, None, False
    )
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)
    sentinel = storage[:width]
    torch.testing.assert_close(sentinel, torch.full_like(sentinel, 100.0))


def _deep_select_reference(scores, ends, k):
    """DeepSelect's documented picks: a row whose end is at most k returns its
    whole [0, end) first; longer rows return k columns of [0, end) with the
    largest scores. Picks are unsorted and -1 padded."""
    picks = torch.full((scores.shape[0], k), -1, dtype=torch.int32)
    for row, end in enumerate(ends.tolist()):
        if end <= k:
            picks[row, :end] = torch.arange(end - 1, -1, -1)
            continue
        values = scores[row, :end].float().cpu()
        assert not values.isnan().any(), "DeepSelect aborts on NaN"
        order = torch.sort(values, descending=True, stable=True).indices[:k]
        picks[row] = order.flip(0)
    return picks.to(scores.device)


def test_fused_selection_takes_unscored_short_pools_whole(triton_device):
    """Sparse Reindex without _mask_candidate_logits selects what it did.

    Pools that fit k were not scored; the mask wrote 0.0 on their [0, end)
    and finishing took the whole pool. The fused kernel takes it without
    reading those uninitialized logits or the selector's picks. Scored rows
    are read only on [0, end), which the mask merely round-trips.
    """
    from tokenspeed_kernel.ops.attention.dsv41._deep_gemm.indexer import (
        _mask_candidate_logits,
    )

    device = triton_device
    torch.manual_seed(2207)
    n, k, blocks = 8, 16, 16
    width = blocks * 8
    ends = torch.tensor([0, 1, 9, 16, 17, 40, 128, 100], dtype=torch.int32)
    # Normalized pools: ascending request-local blocks, then -1. Blocks 38 and
    # 39 reach past a 300-row capacity, so their last rows drop out.
    normalized = torch.full((n, blocks), -1, dtype=torch.int32)
    for row, end in enumerate(ends.tolist()):
        count = (end + 7) // 8
        normalized[row, :count] = torch.randperm(40)[:count].sort().values
    normalized[6, -1] = 39
    raw = torch.randn(n, width).to(torch.bfloat16)
    raw[:, ::9] = -torch.inf
    short = ends <= k
    past = torch.arange(width)[None, :] >= ends[:, None]
    # Unscored rows and columns past end are uninitialized; NaN stands in.
    raw[short] = torch.nan
    raw[past] = torch.nan
    raw, ends, normalized, short = (
        t.to(device) for t in (raw, ends, normalized, short)
    )
    masked = raw.clone()
    _mask_candidate_logits[(n, (width + 1023) // 1024)](
        masked, ends, masked.stride(0), width, k, 1024
    )
    picks = _deep_select_reference(masked, ends, k)
    slots = _fused_slots(n, k, 1, device)
    expected = _poisoned_outputs(n, k, 0, device)
    _finish_write_reference(
        (picks, masked, ends, normalized, 300, 8), None, expected, slots
    )
    # The fused kernel never reads the picks of an unscored row.
    poisoned = picks.masked_fill(short[:, None], 12345)
    actual = _poisoned_outputs(n, k, 0, device)
    actual_slots = _fused_slots(n, k, 1, device)
    implementation.finish_write_selection(
        (poisoned, raw, ends, normalized, 300, 8), None, actual, actual_slots, True
    )
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)
    assert torch.equal(actual_slots.out, slots.out)
    # Short pools were taken whole: one row per candidate row below end.
    assert actual[1][short].tolist() == ends[short].tolist()
    assert (actual[1][~short] > 0).all()


def test_fused_selection_rejects_outputs_that_disagree_with_picks(triton_device):
    device = triton_device
    picks = (
        torch.full((2, 4), -1, dtype=torch.int32, device=device),
        torch.zeros(2, 8, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
        None,
        8,
        1,
    )
    out = _poisoned_outputs(2, 4, 0, device)
    with pytest.raises(ValueError):
        implementation.finish_write_selection(
            picks, None, (out[0][:, :3], *out[1:]), None, False
        )
    # A pass with candidate blocks needs their picks.
    with pytest.raises(ValueError):
        implementation.finish_write_selection(
            picks, None, _poisoned_outputs(2, 4, 2, device), None, False
        )
    implementation.finish_write_selection(picks, None, out, None, False)
    assert (out[0] == -1).all() and (out[1] == 0).all() and (out[3] == 0).all()


class _FakeDeepGemm:
    """DeepGEMM's packed index scorers over a fixed score table, without a GPU.

    The test numbers query rows through their first head weight; history
    holds each row's logits by request-local history row. Columns a native
    scorer leaves unwritten are NaN.
    """

    def __init__(self, history):
        self.history = history

    def get_num_sms(self):
        return 4

    def get_pdl(self):
        return False

    def get_paged_mqa_logits_metadata(self, context_lens, page_size, sms, indices):
        return torch.zeros(sms + 1, 2, dtype=torch.int32)

    def fp8_fp4_paged_mqa_logits(self, *, weights, context_lens, max_context_len, **_):
        rows = weights[:, 0].long()
        # DeepGEMM pads each BF16 row to 1024 bytes and returns the capacity.
        padded = -(-max_context_len // 512) * 512
        logits = torch.full((rows.numel(), padded), torch.nan, dtype=torch.bfloat16)
        scored = torch.arange(max_context_len)[None, :] < context_lens.view(-1, 1)
        logits[:, :max_context_len] = self.history[rows, :max_context_len].masked_fill(
            ~scored, torch.nan
        )
        return logits[:, :max_context_len]

    def get_paged_sparse_mqa_logits_metadata(
        self, context, table, requests, page_size, sparse, q_dtype, block_kv
    ):
        return context, sparse

    def fp8_fp4_paged_sparse_mqa_logits(
        self, *, weights, metadata, num_max_sparse_blocks, sparse_block_kv, **_
    ):
        context, sparse = metadata
        rows = weights[:, 0].long()
        width = num_max_sparse_blocks * sparse_block_kv
        positions = sparse.long()[:, :, None] * 8 + torch.arange(8)
        positions = positions.reshape(rows.numel(), width)
        values = self.history[
            rows[:, None], positions.clamp(0, self.history.shape[1] - 1)
        ]
        scored = torch.arange(width)[None, :] < context.view(-1, 1)
        return values.masked_fill(~scored, torch.nan)


def _native_selection_inputs(rows, columns, cache_stride, ratio):
    """Decode-shaped index_topk inputs: per-request tables, as selection_table
    builds them, numbered query rows, and the owner's slot geometry."""
    torch.manual_seed(4129 + rows + columns)
    pages = columns * 3 + 2
    storage = torch.zeros(pages, cache_stride, dtype=torch.uint8)
    cache = storage.as_strided((pages, 64, 68), (cache_stride, 68, 1))
    requests = torch.tensor([0, 0, 1, 1, 2, -1, 2, 0][:rows])
    owner = torch.randint(1, pages, (3, columns + 1), dtype=torch.int32)
    owner[0, 1] = 0
    owner[1, 2] = pages + 3
    owner[2, 0] = -1
    lengths = [0, 9, 70, 200, 777, 0, columns * 64, 131]
    positions = torch.tensor(lengths[:rows]) * ratio - 1 + ratio
    positions[requests < 0] = -1
    visible = ((positions + 1) // ratio).clamp_min(0).to(torch.int32)
    table = owner[requests.clamp_min(0), :columns]
    table = table.masked_fill((table <= 0) | (requests[:, None] < 0), -1)
    q = torch.randn(rows, 2, 128).to(torch.bfloat16)
    weights = torch.rand(rows, 2)
    weights[:, 0] = torch.arange(rows)
    history = torch.randn(rows, columns * 64).to(torch.bfloat16)
    history[:, 3::11] = -torch.inf
    return q, weights, cache, table, visible, positions, requests, owner, pages, history


def _native_index_topk(
    native, inputs, candidates, topk, candidate_topk, chunk, slot_ratio
):
    q, weights, cache, table, visible, positions, requests, owner, pages, _ = inputs
    rows = q.shape[0]
    out = tuple(
        torch.full(shape, 12345, dtype=torch.int32)
        for shape in ((rows, topk), (rows,), (rows, candidate_topk), (rows,))
    )
    slots = None
    if slot_ratio:
        slots = dsv41.SelectionSlots(
            torch.full((rows, topk), 4444, dtype=torch.int32),
            positions,
            requests,
            owner,
            slot_ratio,
            pages,
        )
    native.index_topk(
        q,
        weights,
        cache,
        table,
        visible,
        candidates,
        topk,
        candidate_topk,
        8,
        chunk,
        256,
        None,
        out,
        None,
        slots,
    )
    return out + ((slots.out,) if slots is not None else ())


@pytest.mark.parametrize(
    "case",
    [
        "full",  # L2-type: paged Full
        "source",  # L20-type: candidate source, table above 65536 rows
        "source_prepared",  # candidate source with clean-fused block maxima
        "sparse",  # L24-type: native sparse Reindex, short pools unscored
        "reindex",  # Reindex on unaligned pages: candidate gather
    ],
)
@pytest.mark.parametrize("slot_ratio", [None, 2])
def test_native_index_topk_fused_selection_matches_the_separate_passes(
    monkeypatch, case, slot_ratio
):
    """The fused finish leaves every native index_topk output bit-identical.

    The reference run replaces finish_write_selection with finish_selection,
    write_selection and global_slots, and scores sparse Reindex with the
    candidate mask (_sparse_candidate_scores), as before the fusion.
    DeepGEMM, DeepSelect (its documented picks), the paged clean copy and the
    CUDA device checks are replaced; the other Triton kernels run
    (TRITON_INTERPRET=1 on CPU). Two query tiles check that slots follow
    their rows.
    """
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs the Triton kernels on CPU under TRITON_INTERPRET=1")
    from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as native
    from tokenspeed_kernel.ops.attention.dsv41 import deep_select
    from tokenspeed_kernel.ops.attention.dsv41._deep_gemm import indexer

    columns = 1026 if case == "source" else 13
    stride = 4416 if case == "reindex" else 4608
    inputs = _native_selection_inputs(6, columns, stride, 2)
    candidates = None
    candidate_topk = 0
    topk = 16
    if case in ("source", "source_prepared"):
        candidate_topk = 32
    if case in ("sparse", "reindex"):
        visible = inputs[4]
        # Pools at most topk rows long go unscored; repeats, -1 and future
        # blocks drop out, and the newest block stays a candidate.
        candidates = torch.randint(-1, columns * 8 + 2, (6, 24), dtype=torch.int32)
        candidates[:, :4] = torch.tensor([3, 3, -1, 0])
        candidates[:, -1] = ((visible - 1) // 8).clamp_min(0)
        candidates[2, 4:] = -1
    api = _FakeDeepGemm(inputs[-1])

    def deep_select_topk(scores, lengths, k, output):
        output.copy_(_deep_select_reference(scores, lengths, k))
        return output

    monkeypatch.setattr(native, "_api", lambda queries: api)
    monkeypatch.setattr(indexer, "_warmup_sparse_mqa", lambda *args: None)
    # _clean_logits strides its tiles with a loop the interpreter cannot run.
    monkeypatch.setattr(
        native, "clean_logits_for_selection", _clean_selection_reference
    )
    for module in (native, deep_select, indexer):
        monkeypatch.setattr(module, "deep_select_alignment", lambda: (256, 32))
    monkeypatch.setattr(deep_select, "deep_select_topk", deep_select_topk)
    monkeypatch.setattr(deep_select, "eligible", lambda *args: True)
    monkeypatch.setattr(implementation, "_same_device", lambda x, tensors: None)

    def run():
        return _native_index_topk(
            native, inputs, candidates, topk, candidate_topk, 4, slot_ratio
        )

    fused = run()
    with monkeypatch.context() as separate:
        separate.setattr(
            native,
            "finish_write_selection",
            lambda rows, pool, out, slots, short_rows_unscored: (
                _finish_write_reference(rows, pool, out, slots)
            ),
        )
        separate.setattr(
            native, "_sparse_candidate_logits", indexer._sparse_candidate_scores
        )
        expected = run()
    for got, want in zip(fused, expected, strict=True):
        assert torch.equal(got, want)
    rows, lengths, blocks, block_lengths = fused[:4]
    assert (lengths > 0).sum() >= 3 and lengths.max() <= topk
    assert (block_lengths > 0).any() == bool(candidate_topk)
    if slot_ratio:
        q, weights, cache, table, visible, positions, requests, owner, pages, _ = inputs
        torch.testing.assert_close(
            fused[4],
            implementation.global_slots(rows, positions, requests, owner, 2, pages),
            rtol=0,
            atol=0,
        )


def _paged_api(built, seen):
    class Api:
        def get_num_sms(self):
            return 4

        def get_paged_mqa_logits_metadata(self, context_lens, size, sms, indices):
            built.append("schedule")
            return torch.cat((context_lens.flatten(), indices, torch.tensor([sms])))

        def fp8_fp4_paged_mqa_logits(self, **kwargs):
            seen.append(kwargs)
            rows = kwargs["context_lens"].shape[0]
            return torch.zeros(rows, 512, dtype=torch.bfloat16)

    return Api()


def test_prepared_paged_geometry_is_the_scoring_geometry(monkeypatch):
    """prepare_index_topk builds paged Full's tile geometry under the key the
    scorer looks it up by: scoring uses it as built and rebuilds nothing."""
    from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as native

    rows, pages, capacity = 3, 9, 5 * 64
    table = torch.tensor(
        [[1, -1, 2, 3, 99], [4, 5, 6, 7, 8], [-1, 0, 1, 2, 3]], dtype=torch.int32
    )
    lengths = torch.tensor([200, 400, 0], dtype=torch.int32)
    requests = torch.tensor([0, 1, 2], dtype=torch.int32)
    built, seen = [], []
    api = _paged_api(built, seen)

    def safe(t, lens, p, cap, size):
        built.append("table")
        return t.masked_fill((t < 0) | (t >= p), 0), lens.clamp(0, cap).to(torch.int32)

    monkeypatch.setattr(native, "_api", lambda queries: api)
    monkeypatch.setattr(native, "_api_for", lambda heads, device: api)
    monkeypatch.setattr(native, "safe_metadata", safe)
    monkeypatch.setattr(native, "clean_logits_for_selection", lambda *args: args[0])
    monkeypatch.setattr(native, "mask_invalid_table_pages", lambda *args: args[0])
    monkeypatch.setattr(native, "deep_select_alignment", lambda: (1024, 32))
    index_cache = torch.zeros(pages, 64, 68, dtype=torch.uint8)
    cache = index_cache.as_strided((pages, 64 * 68), (index_cache.stride(0), 1))
    queries = (
        torch.zeros(rows, 32, 64, dtype=torch.uint8),
        torch.zeros(rows, 32, dtype=torch.int32),
    )

    def score(metadata):
        seen.clear()
        native._paged_scores(
            queries,
            cache,
            torch.zeros(rows, 32),
            table,
            lengths,
            capacity,
            64,
            metadata,
            0,
            rows,
            False,
        )
        return seen[0]

    plain = score(dsv41.new_index_metadata(table, lengths, requests, history_rows=None))
    assert built == ["table", "schedule"]
    built.clear()
    metadata = dsv41.new_index_metadata(table, lengths, requests, history_rows=None)
    native.prepare_index_topk(
        index_cache, table, lengths, None, 4, 0, 32, rows, metadata
    )
    assert built == ["table", "schedule"]
    (entry,) = metadata.paged.values()
    prepared = score(metadata)
    assert built == ["table", "schedule"]
    table_p, _, context_p, schedule_p, indices_p = entry.tensors
    assert prepared["block_table"] is table_p
    assert prepared["context_lens"] is context_p
    assert prepared["schedule_meta"] is schedule_p
    assert prepared["indices"] is indices_p
    for name in ("block_table", "context_lens", "schedule_meta", "indices"):
        assert torch.equal(prepared[name], plain[name])
    # Dense (history_rows) scoring and empty tables keep no prepared geometry.
    dense = dsv41.new_index_metadata(table, lengths, None, history_rows=(rows,))
    native.prepare_index_topk(index_cache, table, lengths, None, 4, 0, 32, rows, dense)
    empty = table[:, :0]
    blank = dsv41.new_index_metadata(empty, lengths, None, history_rows=None)
    native.prepare_index_topk(index_cache, empty, lengths, None, 4, 0, 32, rows, blank)
    assert not dense.paged and not blank.paged and built == ["table", "schedule"]


def test_prepared_sparse_geometry_is_the_scoring_geometry(monkeypatch):
    """prepare_index_topk builds the native sparse scorer's normalized
    candidates and schedule for the same views; scoring reuses them as built."""
    from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as native
    from tokenspeed_kernel.ops.attention.dsv41._deep_gemm import indexer

    rows, pages = 3, 9
    table = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 0, 0, 0]], dtype=torch.int32)
    visible = torch.tensor([100, 250, 0], dtype=torch.int32)
    candidates = torch.tensor(
        [[3, 0, 5, -1], [1, 2, 31, 7], [0, -1, -1, -1]], dtype=torch.int32
    )
    built, scored = [], []

    class Normalize:
        def __getitem__(self, grid):
            def launch(C, L, T, normalized, sparse, context, end, *args, num_warps):
                built.append("normalize")
                order = C.sort(dim=1).values
                normalized.fill_(-1)[:, : C.shape[1]] = order
                sparse.copy_(normalized + 1)
                context.copy_(L.clamp_min(0) * 2)
                end.copy_(L // 3)

            return launch

    class Mask:
        def __getitem__(self, grid):
            return lambda *args: None

    class Api:
        def get_num_sms(self):
            return 4

        def get_pdl(self):
            return False

        def get_paged_sparse_mqa_logits_metadata(self, *args):
            built.append("schedule")
            context, table, requests, page, sparse = args[:5]
            return torch.cat((context, requests, sparse.flatten(), table.flatten()))

        def fp8_fp4_paged_sparse_mqa_logits(self, **kwargs):
            scored.append(kwargs["metadata"])
            width = kwargs["num_max_sparse_blocks"] * 8
            return torch.zeros(rows, width, dtype=torch.bfloat16)

    api = Api()
    monkeypatch.setattr(indexer, "_normalize_candidate_blocks", Normalize())
    monkeypatch.setattr(indexer, "_mask_candidate_logits", Mask())
    monkeypatch.setattr(indexer, "_warmup_sparse_mqa", lambda *args: None)
    monkeypatch.setattr(indexer, "deep_select_alignment", lambda: (1024, 32))
    monkeypatch.setattr(native, "_api_for", lambda heads, device: api)
    # Native sparse reads need 512-byte page strides.
    index_cache = torch.zeros(pages, 4608, dtype=torch.uint8).as_strided(
        (pages, 64, 68), (4608, 68, 1)
    )
    cache = index_cache.as_strided((pages, 64 * 68), (4608, 1))
    queries = (
        torch.zeros(rows, 32, 64, dtype=torch.uint8),
        torch.zeros(rows, 32, dtype=torch.int32),
    )

    def score(metadata):
        return indexer._sparse_candidate_scores(
            queries,
            torch.zeros(rows, 32, dtype=torch.bfloat16),
            cache,
            table,
            visible,
            candidates,
            api,
            skip_score_topk=4,
            index_metadata=metadata,
        )

    plain = score(dsv41.new_index_metadata(table, visible, None, history_rows=None))
    assert built == ["normalize", "schedule"]
    built.clear()
    metadata = dsv41.new_index_metadata(table, visible, None, history_rows=None)
    native.prepare_index_topk(
        index_cache, table, visible, candidates, 4, 0, 32, rows, metadata
    )
    assert built == ["normalize", "schedule"]
    (entry,) = metadata.sparse.values()
    prepared = score(metadata)
    assert built == ["normalize", "schedule"]
    normalized, ends, schedule = entry.tensors[:3]
    assert scored[-1] is schedule and torch.equal(scored[-1], scored[0])
    assert prepared[1] is ends and prepared[2] is normalized
    for got, want in zip(prepared, plain, strict=True):
        assert torch.equal(got, want)


def test_prepare_index_topk_follows_the_selected_scorer(monkeypatch):
    """The family call prepares through the scorer index_topk would select,
    with the same traits, and reports False when that scorer has no preparer."""
    from types import SimpleNamespace

    from tokenspeed_kernel.selection import NoKernelFoundError

    calls = []
    scorer = {"name": "deep_gemm_dsv41_index_topk"}

    def select(
        family, mode, signature, *, features, platform, traits, solution, override
    ):
        calls.append((mode, solution, traits))
        if mode == "dsv41_index_topk":
            return SimpleNamespace(name=scorer["name"])
        if solution != "deep_gemm":
            raise NoKernelFoundError("no preparer")
        return lambda *args: calls.append(args)

    monkeypatch.setattr(dsv41, "select_kernel", select)
    table = torch.ones(3, 4, dtype=torch.int32)
    lengths = torch.tensor([10, 20, 30], dtype=torch.int32)
    metadata = dsv41.new_index_metadata(table, lengths, None, history_rows=None)
    cache = torch.zeros(9, 64, 68, dtype=torch.uint8)
    args = (cache, table, lengths, None, 4, 0, 32, torch.bfloat16, 3, metadata)
    assert dsv41.prepare_index_topk(*args)
    (scorer_mode, scorer_solution, traits), prepare, prepared = calls
    assert (scorer_mode, scorer_solution) == ("dsv41_index_topk", None)
    assert prepare == ("dsv41_index_topk_prepare", "deep_gemm", traits)
    assert traits["index_heads"] == 32 and traits["index_k_format"] == "mxfp4"
    assert prepared == (cache, table, lengths, None, 4, 0, 32, 3, metadata)
    calls.clear()
    scorer["name"] = "triton_dsv41_index_topk"
    assert not dsv41.prepare_index_topk(*args)
    assert [call[:2] for call in calls] == [
        ("dsv41_index_topk", None),
        ("dsv41_index_topk_prepare", "triton"),
    ]
    with pytest.raises(TypeError, match="new_index_metadata"):
        dsv41.prepare_index_topk(*args[:-1], object())
    other = dsv41.new_index_metadata(table.clone(), lengths, None, history_rows=None)
    with pytest.raises(ValueError, match="different table"):
        dsv41.prepare_index_topk(*args[:-1], other)
