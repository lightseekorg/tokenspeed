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

"""SM107 FP8 sparse prefill: dispatch, numerics and live metadata contracts."""

from dataclasses import replace

import pytest
import torch
from qsa_fp8_prefill_reference import reference as fp8_reference
from tokenspeed_kernel.numerics.torch.attention.qsa import torch_qsa_selected_attention
from tokenspeed_kernel.ops.attention.qsa import (
    QSASparseMetadata,
    QSAWorkspace,
    qsa_sparse_attention,
    select_qsa_sparse_attention_kernel,
)
from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots
from tokenspeed_kernel.platform import ArchVersion, current_platform, pdl_enabled
from tokenspeed_kernel.workspace import capture_session

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia
    or current_platform().arch_version != ArchVersion(10, 7),
    reason="FP8 sparse prefill specialization requires SM107",
)
NAME = "cute_dsl_fp8_qsa_sparse_prefill"


@pytest.fixture(autouse=True)
def fp32_reference_precision(set_matmul_precision):
    set_matmul_precision("ieee")


@pytest.fixture(autouse=True)
def restore_pdl():
    previous = pdl_enabled()
    yield
    torch.cuda.synchronize()
    pdl_enabled(previous)


def _case(prefix, heads, *, rows=11, slot_stride=2051, position_stride=1):
    torch.manual_seed(18)
    page_size = 256
    lengths = torch.tensor(
        [prefix + rows // 2, 0, prefix + rows - rows // 2],
        device="cuda",
        dtype=torch.int32,
    )
    pages = (prefix + rows + page_size - 1) // page_size
    table = (torch.randperm(3 * pages, device="cuda", dtype=torch.int32) + 1).reshape(
        3, pages
    )
    positions = torch.empty(rows * position_stride, device="cuda", dtype=torch.int64)[
        ::position_stride
    ]
    positions.copy_(
        torch.cat(
            (
                torch.arange(prefix, prefix + rows // 2),
                torch.arange(prefix, prefix + rows - rows // 2),
            )
        ).cuda()
    )
    requests = torch.tensor(
        [0] * (rows // 2) + [2] * (rows - rows // 2), device="cuda", dtype=torch.int32
    )
    # Distinct complete blocks, permuted per row, and an implicit partial block.
    blocks = torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
    for row, position in enumerate(positions.cpu().tolist()):
        complete = (position + 1) // 4
        count = min(complete, 512)
        blocks[row, :count] = torch.randperm(
            complete, device="cuda", dtype=torch.int32
        )[:count]
    metadata = QSASparseMetadata(
        blocks,
        table,
        requests,
        positions,
        (0, rows // 2, rows // 2, rows),
        page_size,
        4,
        QSAWorkspace(),
        seq_lens=lengths,
    )
    q = torch.randn((rows, heads * 12, 256), device="cuda", dtype=torch.bfloat16)
    cache_shape = ((3 * pages + 1) * page_size, heads, 256)
    k, v = [
        torch.randn(cache_shape, device="cuda").to(torch.float8_e4m3fn)
        for _ in range(2)
    ]
    k[0].fill_(float("nan"))
    v[0].fill_(float("nan"))
    slots = torch.empty_strided(
        (rows, 2051), (slot_stride, 1), device="cuda", dtype=torch.int32
    )
    slots.copy_(
        qwen4_exp_qsa_selected_slots(
            blocks,
            ((positions + 1) // 4).int(),
            positions,
            requests,
            table,
            page_size,
            4,
            2048,
            enable_pdl=False,
        )
    )
    # Null slots, holes, and an entirely empty row must contribute no mass.
    slots[-1].fill_(-1)
    slots[0, 0] = 0
    return q, k, v, slots, metadata


def _kwargs(
    metadata, *, override=None, query_width=None, solution=None, scales=(1.0, 1.0)
):
    return dict(
        max_seqlen_q=query_width,
        k_scale=scales[0],
        v_scale=scales[1],
        sparse_metadata=metadata,
        override=override,
        solution=solution,
    )


def _call(case, *, scales, override):
    q, k, v, slots, metadata = case
    return qsa_sparse_attention(
        q,
        k,
        v,
        slots,
        scale=1 / 16,
        metadata_capacity_rows=None,
        **_kwargs(metadata, override=override, scales=scales),
    )


def _reference(case, scales):
    q, k, v, slots, _ = case
    return fp8_reference(
        q,
        k,
        v,
        slots,
        case[4].query_positions,
        torch.arange(q.shape[0], device=q.device),
        attention_scale=1 / 16,
        k_scale=scales[0],
        v_scale=scales[1],
    ).to(q.dtype)


# Cover all four head-count/PDL specializations without crossing every prefix
# and scale. Long prefixes saturate the same 512-block sparse budget.
@pytest.mark.parametrize(
    ("heads", "prefix", "enabled", "scales"),
    [
        (1, 0, False, (1.0, 1.0)),
        (1, 196608, True, (0.5, 0.25)),
        (2, 196608, False, (0.5, 0.25)),
        (2, 0, True, (1.0, 1.0)),
    ],
)
def test_fp8_prefill_ragged_prefix_and_scales(heads, prefix, enabled, scales):
    case = _case(prefix, heads)
    pdl_enabled(enabled)
    assert (
        select_qsa_sparse_attention_kernel(*case[:4], **_kwargs(case[4])).name == NAME
    )
    output = _call(case, scales=scales, override=None)
    torch.testing.assert_close(output, _reference(case, scales), atol=0.004, rtol=0.02)
    assert output[-1].count_nonzero() == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_fp8_prefill_graph_observes_producer_and_live_positions(enabled):
    from test_qsa_pdl import _publish

    source = _case(0, 2)
    destination = _case(0, 2)
    pdl_enabled(enabled)
    _call(destination, scales=(1.0, 1.0), override=None)
    graph = torch.cuda.CUDAGraph()
    source_tensors = (*source[:4], source[4].query_positions)
    destination_tensors = (*destination[:4], destination[4].query_positions)
    with capture_session(graph), torch.cuda.graph(graph):
        _publish(source_tensors, destination_tensors)
        output = _call(destination, scales=(1.0, 1.0), override=None)
    for prefix in (65536, 0):
        # Change the row frontier without changing the captured addresses.
        source[0].mul_(0.75)
        source[2].copy_((-source[2].float()).to(source[2].dtype))
        source[4].query_positions.fill_(prefix + 3)
        source[3].fill_(-1)
        source[3][:, :4] = torch.tensor([1, 2, 3, 4], device="cuda")
        source[3][-1].fill_(0)
        graph.replay()
        torch.testing.assert_close(
            output, _reference(source, (1.0, 1.0)), atol=0.004, rtol=0.02
        )
        assert output[-1].count_nonzero() == 0


def test_fp8_prefill_runtime_shapes_do_not_recompile():
    from utils import assert_no_cute_compile

    pdl_enabled(False)
    _call(_case(0, 2, rows=16), scales=(1.0, 1.0), override=None)
    with assert_no_cute_compile():
        for rows, stride, pos_stride in ((1, 2051, 1), (7, 2064, 2), (33, 4096, 3)):
            case = _case(
                8192, 2, rows=rows, slot_stride=stride, position_stride=pos_stride
            )
            output = _call(case, scales=(0.5, 0.25), override=None)
            torch.testing.assert_close(
                output, _reference(case, (0.5, 0.25)), atol=0.004, rtol=0.02
            )


def test_fp8_prefill_cache_offsets_above_int32():
    far_slot = (1 << 31) // 512 + 32
    if torch.cuda.mem_get_info()[0] < 2 * (far_slot + 1) * 512 + (1 << 30):
        pytest.skip("large-offset check needs two FP8 pools above 2 GiB")
    q, _, _, slots, metadata = _case(0, 2, rows=1)
    k, v = [
        torch.empty((far_slot + 1, 2, 256), device="cuda", dtype=torch.float8_e4m3fn)
        for _ in range(2)
    ]
    k[1].zero_()
    v[1].zero_()
    k[far_slot].fill_(0.5)
    v[far_slot].fill_(0.25)
    slots.fill_(-1)
    slots[0, 2048] = far_slot
    output = _call((q, k, v, slots, metadata), scales=(0.5, 0.25), override=NAME)
    torch.testing.assert_close(output, torch.full_like(q, 0.0625), atol=0, rtol=0)


def test_fp8_prefill_dispatch_fallback_and_explicit_rejection():
    case = _case(0, 2)
    for metadata, scales, query_width in (
        (None, (1.0, 1.0), None),
        (None, (1.0, 1.0), 1),
        (case[4], (torch.ones(1, device="cuda"), 1.0), None),
    ):
        assert (
            select_qsa_sparse_attention_kernel(
                *case[:4], **_kwargs(metadata, scales=scales, query_width=query_width)
            ).name
            != NAME
        )
    q, k, v, slots, metadata = case
    strided_q = torch.empty_strided(
        q.shape, (q.stride(0) * 2, 256, 1), dtype=q.dtype, device=q.device
    )
    strided_q.copy_(q)
    bad_case = (strided_q, k, v, slots, metadata)
    assert (
        select_qsa_sparse_attention_kernel(*bad_case[:4], **_kwargs(metadata)).name
        == "flashinfer_fa2_fp8_qsa_sparse_attention"
    )
    with pytest.raises(ValueError, match="dense"):
        _call(bad_case, scales=(1.0, 1.0), override=NAME)


@pytest.mark.parametrize("enabled", [False, True])
def test_fp8_prefill_empty_tile_before_negative_logits(enabled):
    q, k, v, slots, metadata = _case(2048, 2, rows=2)
    pdl_enabled(enabled)
    q.fill_(100)
    k[2].fill_(-0.5)
    v[2].fill_(0.25)
    slots.fill_(-1)
    slots[0, 128] = 2
    # The first tile is empty; a valid logit in the next tile is -800.
    # Its probability is one despite exp(-800) underflowing without a max shift.
    output = _call((q, k, v, slots, metadata), scales=(1.0, 1.0), override=NAME)
    expected = torch.zeros_like(q)
    expected[0].fill_(0.25)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_fp8_prefill_metadata_storage_offsets(enabled):
    q, k, v, slots, metadata = _case(8192, 2, rows=7)
    shifted_slots = torch.empty((7, 2052), device="cuda", dtype=torch.int32)[:, 1:]
    shifted_slots.copy_(slots)
    shifted_positions = torch.empty(22, device="cuda", dtype=torch.int64)[1::3]
    shifted_positions.copy_(metadata.query_positions)
    metadata = replace(metadata, query_positions=shifted_positions)
    case = q, k, v, shifted_slots, metadata
    pdl_enabled(enabled)
    output = _call(case, scales=(0.5, 0.25), override=NAME)
    torch.testing.assert_close(
        output, _reference(case, (0.5, 0.25)), atol=0.004, rtol=0.02
    )


def test_fp8_prefill_rejects_unaligned_vector_reads():
    q, k, v, slots, metadata = _case(0, 2)
    shifted_q = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)[1:].view(
        q.shape
    )
    shifted_q.copy_(q)
    with pytest.raises(ValueError, match="aligned"):
        _call((shifted_q, k, v, slots, metadata), scales=(1.0, 1.0), override=NAME)


def test_fp8_prefill_int32_positions_keep_fallback():
    q, k, v, slots, metadata = _case(0, 2)
    metadata = replace(metadata, query_positions=metadata.query_positions.int())
    case = q, k, v, slots, metadata
    assert (
        select_qsa_sparse_attention_kernel(*case[:4], **_kwargs(metadata)).name != NAME
    )
    output = _call(case, scales=(1.0, 1.0), override=None)
    torch.testing.assert_close(
        output,
        torch_qsa_selected_attention(*case[:4], 1 / 16, 1.0, 1.0),
        atol=0.004,
        rtol=0.01,
    )
