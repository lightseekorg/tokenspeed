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

import gc
import weakref
from dataclasses import replace
from functools import partial

import pytest
import tokenspeed_kernel.ops.attention.qsa as qsa_module
import torch
from tokenspeed_kernel.numerics.torch.attention.qsa import (
    torch_qsa_selected_attention,
)
from tokenspeed_kernel.ops.attention.qsa import (
    QSASparseMetadata,
    QSAWorkspace,
    qsa_sparse_attention,
    select_qsa_sparse_attention_kernel,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.workspace import capture_session

if current_platform().is_nvidia:
    from tokenspeed_kernel.ops.attention.qsa._flashinfer.runner import (
        _FlashInferQSASparseRunner,
        get_flashinfer_qsa_sparse_runner,
    )

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="QSA sparse attention requires CUDA or ROCm"
)


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    return torch_qsa_selected_attention(
        q, k_cache, v_cache, selected_slots, scale, k_scale, v_scale
    )


def _prims_case(*, lengths, prefixes, kv_heads, shared_candidates, block_size, device):
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots

    torch.manual_seed(91)
    page_size, topk, head_dim = 64, 512, 256
    max_pages = (
        max(p + n for p, n in zip(prefixes, lengths)) + page_size - 1
    ) // page_size
    table = (
        (torch.randperm(len(lengths) * max_pages, device=device) + 1)
        .to(torch.int32)
        .view(len(lengths), max_pages)
    )
    slots = (len(lengths) * max_pages + 1) * page_size
    k = torch.randn(slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(
        sum(lengths), kv_heads * 6, head_dim, device=device, dtype=torch.bfloat16
    )
    positions = torch.cat(
        [
            torch.arange(prefix, prefix + length, device=device)
            for prefix, length in zip(prefixes, lengths)
        ]
    )
    requests = torch.repeat_interleave(
        torch.arange(len(lengths), device=device, dtype=torch.int32),
        torch.tensor(lengths, device=device),
    )
    blocks = torch.full((sum(lengths), topk), -1, dtype=torch.int32, device=device)
    for row, position in enumerate(positions.tolist()):
        complete = (position + 1) // block_size
        chosen = (
            torch.arange(min(complete, topk), device=device)
            if shared_candidates
            else torch.randperm(complete, device=device)[:topk]
        )
        blocks[row, : chosen.numel()] = chosen
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    metadata = QSASparseMetadata(
        blocks,
        table,
        requests,
        positions,
        tuple(offsets),
        page_size,
        block_size,
        QSAWorkspace(),
    )
    selected = qwen4_exp_qsa_selected_slots(
        blocks,
        ((positions + 1) // block_size).to(torch.int32),
        positions,
        requests,
        table,
        page_size,
        block_size,
        topk * block_size,
        enable_pdl=False,
    )
    return q, k, v, selected, metadata


@pytest.mark.parametrize("shared_candidates", [False, True])
def test_prims_qsa_prefill_matches_causal_reference(shared_candidates, device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    q, k, v, slots, metadata = _prims_case(
        lengths=(5, 3, 9),
        prefixes=(32768, 29, 0),
        kv_heads=2,
        shared_candidates=shared_candidates,
        block_size=4,
        device=device,
    )
    kwargs = dict(
        scale=1 / 16,
        max_seqlen_q=None,
        metadata_capacity_rows=None,
        k_scale=0.75,
        v_scale=1.25,
        override="flashinfer_prims_qsa_sparse_attention",
        solution=None,
        sparse_metadata=metadata,
    )
    actual = qsa_sparse_attention(q, k, v, slots, **kwargs)
    expected = _reference(q, k, v, slots, 1 / 16, 0.75, 1.25)
    # PrimTS rounds the unnormalized probabilities to BF16 before PV. With
    # only a few keys, cancellation can expose that rounding near zero.
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=1e-2)
    # Reuse the same plan with new candidate values and replacement Q storage.
    metadata.selected_blocks[0].copy_(
        torch.arange(512, 1024, device=device, dtype=torch.int32)
    )
    slots[0, :2048] = (
        metadata.block_table[0, (torch.arange(2048, 4096, device=device) // 64)].long()
        * 64
        + torch.arange(2048, 4096, device=device) % 64
    ).to(torch.int32)
    new_q = q * 0.5
    updated = qsa_sparse_attention(new_q, k, v, slots, **kwargs)
    torch.testing.assert_close(
        updated,
        _reference(new_q, k, v, slots, 1 / 16, 0.75, 1.25),
        atol=4e-3,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "max_rows"),
    [(24, 2, 1), (24, 2, 3), (24, 2, 16), (128, 2, 16), (256, 2, 16)],
)
def test_prims_qsa_warmup_groups_follow_native_limits(q_heads, kv_heads, max_rows):
    decode = pytest.importorskip("flashinfer.attention.prims_ts.decode")
    from tokenspeed_kernel.thirdparty.flashinfer.qsa_prims import _warmup_group_sizes

    offsets = torch.tensor([0, max_rows], dtype=torch.int32, device="cpu")
    expected = []
    for group in range(1, max_rows + 1):
        try:
            decode.validate_q_token_kv_block_sparse_group_size(
                offsets, max_rows, q_heads, kv_heads, group_size=group
            )
        except ValueError:
            continue
        expected.append(group)
    assert set(_warmup_group_sizes(q_heads, kv_heads, max_rows)) == set(expected)


@pytest.mark.parametrize(("q_heads", "kv_heads", "max_rows"), [(3, 2, 16), (24, 2, 0)])
def test_prims_qsa_warmup_rejects_invalid_group_geometry(q_heads, kv_heads, max_rows):
    pytest.importorskip("flashinfer.attention.prims_ts.decode")
    from tokenspeed_kernel.thirdparty.flashinfer.qsa_prims import _warmup_group_sizes

    with pytest.raises(ValueError):
        _warmup_group_sizes(q_heads, kv_heads, max_rows)


def test_prims_qsa_startup_warms_query_groups_and_schedulers(device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    decode = pytest.importorskip("flashinfer.attention.prims_ts.decode")
    pytest.importorskip("flashinfer.autotuner")
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots
    from tokenspeed_kernel.ops.tuning import (
        autotune,
        get_autotune_max_num_tokens,
        kernel_warmup,
    )
    from tokenspeed_kernel.thirdparty.flashinfer.qsa_prims import _get_runner

    _, k, v, _, prototype = _prims_case(
        lengths=(1024,),
        prefixes=(4096,),
        kv_heads=2,
        shared_candidates=True,
        block_size=4,
        device=device,
    )
    k.mul_(0.25)
    v.mul_(0.25)
    workspace = QSAWorkspace()

    def case(lengths):
        rows = sum(lengths)
        q = torch.randn(rows, 24, 256, device=device, dtype=torch.bfloat16)
        positions = torch.cat(
            [torch.arange(4096, 4096 + length, device=device) for length in lengths]
        )
        requests = torch.repeat_interleave(
            torch.arange(len(lengths), device=device, dtype=torch.int32),
            torch.tensor(lengths, device=device),
        )
        table = prototype.block_table.expand(len(lengths), -1).contiguous()
        blocks = prototype.selected_blocks[:rows].clone()
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        metadata = QSASparseMetadata(
            blocks, table, requests, positions, tuple(offsets), 64, 4, workspace
        )
        selected = qwen4_exp_qsa_selected_slots(
            blocks,
            ((positions + 1) // 4).int(),
            positions,
            requests,
            table,
            64,
            4,
            2048,
            enable_pdl=False,
        )
        return q, selected, metadata

    def forward(q, selected, metadata):
        return qsa_sparse_attention(
            q,
            k,
            v,
            selected,
            scale=1 / 16,
            max_seqlen_q=None,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution="flashinfer_prims",
            sparse_metadata=metadata,
        )

    # Clear the compiler lookup cache; live plans retain their own binaries.
    decode._get_compiled_decode.cache_clear()
    original_k, original_v = k.clone(), v.clone()
    with (
        kernel_warmup(enabled=False),
        autotune(tune_mode=True, tuning_buckets=None, round_up=None),
    ):
        forward(*case((1,)))
    torch.testing.assert_close(k, original_k, atol=0, rtol=0)
    torch.testing.assert_close(v, original_v, atol=0, rtol=0)
    misses = decode._get_compiled_decode.cache_info().misses
    # Check every native-legal group independently of the adapter's candidates,
    # including short routes and both native scheduler choices.
    max_rows = get_autotune_max_num_tokens()
    boundaries = torch.tensor([0, max_rows], dtype=torch.int32, device="cpu")
    groups = []
    for group in range(1, max_rows + 1):
        try:
            decode.validate_q_token_kv_block_sparse_group_size(
                boundaries, max_rows, 24, 2, group_size=group
            )
        except ValueError:
            continue
        groups.append(group)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    for routes in (1, min(max_rows, sm_count + 1)):
        q, selected, metadata = case((1,) * routes)
        runner = _get_runner(workspace, q.device)
        row_ids = [0, q.shape[0] - 1]
        expected = _reference(q[row_ids], k, v, selected[row_ids], 1 / 16)
        for group in groups:
            output = runner._run(
                q, k, v, metadata, group_size=group, scale=1 / 16, v_scale=1.0
            )
            assert decode._get_compiled_decode.cache_info().misses == misses
            torch.testing.assert_close(output[row_ids], expected, atol=4e-3, rtol=1e-2)
    for lengths in ((7,), (128,), (1024,), (5, 3), (511, 513), (8,) + (1,) * 120):
        q, selected, metadata = case(lengths)
        eager = forward(q, selected, metadata)
        assert decode._get_compiled_decode.cache_info().misses == misses
        row_ids = [0, q.shape[0] - 1]
        torch.testing.assert_close(
            eager[row_ids],
            _reference(q[row_ids], k, v, selected[row_ids], 1 / 16),
            atol=4e-3,
            rtol=1e-2,
        )
        graph = torch.cuda.CUDAGraph()
        with capture_session(graph), torch.cuda.graph(graph):
            captured = forward(q, selected, metadata)
        q.mul_(0.75)
        graph.replay()
        expected = _reference(q[row_ids], k, v, selected[row_ids], 1 / 16)
        torch.testing.assert_close(captured[row_ids], expected, atol=4e-3, rtol=1e-2)
        assert decode._get_compiled_decode.cache_info().misses == misses


def test_prims_qsa_kernel_warmup_does_not_enable_extra_compilation(device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    decode = pytest.importorskip("flashinfer.attention.prims_ts.decode")
    from tokenspeed_kernel.ops.tuning import autotune, kernel_warmup

    q, k, v, slots, metadata = _prims_case(
        lengths=(1,),
        prefixes=(17,),
        kv_heads=1,
        shared_candidates=True,
        block_size=4,
        device=device,
    )
    k.mul_(0.25)
    v.mul_(0.25)

    def forward():
        return qsa_sparse_attention(
            q,
            k,
            v,
            slots,
            scale=1 / 16,
            max_seqlen_q=None,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution="flashinfer_prims",
            sparse_metadata=metadata,
        )

    decode._get_compiled_decode.cache_clear()
    with autotune(tune_mode=False, tuning_buckets=None, round_up=None):
        with kernel_warmup(enabled=False):
            forward()
        misses = decode._get_compiled_decode.cache_info().misses
        assert misses > 0
        with kernel_warmup(enabled=True):
            actual = forward()
    assert decode._get_compiled_decode.cache_info().misses == misses
    torch.testing.assert_close(
        actual, _reference(q, k, v, slots, 1 / 16), atol=4e-3, rtol=1e-2
    )


@pytest.mark.parametrize("query_width", [1, 4])
@pytest.mark.parametrize("q_heads", [6, 8])
@pytest.mark.parametrize("capture", [False, True])
def test_prims_qsa_decode_matches_reference_and_refreshes_live_metadata(
    query_width: int, q_heads: int, capture: bool, device: str
) -> None:
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots

    q, k, v, slots, metadata = _prims_case(
        lengths=(query_width, query_width),
        prefixes=(4096, 29),
        kv_heads=1,
        shared_candidates=True,
        block_size=4,
        device=device,
    )
    q = torch.randn(q.shape[0], q_heads, 256, device=device, dtype=q.dtype)
    kwargs = dict(
        scale=1 / 16,
        max_seqlen_q=query_width,
        metadata_capacity_rows=None,
        k_scale=None,
        v_scale=None,
        override=None,
        solution="flashinfer_prims",
        sparse_metadata=metadata,
    )
    selected = select_qsa_sparse_attention_kernel(
        q,
        k,
        v,
        slots,
        max_seqlen_q=query_width,
        k_scale=None,
        v_scale=None,
        sparse_metadata=metadata,
        override=None,
        solution=None,
    )
    assert selected.name == (
        "cute_dsl_blackwell_qsa_sparse_attention"
        if q_heads == 6
        else "flashinfer_prims_qsa_sparse_attention"
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = qsa_sparse_attention(q, k, v, slots, **kwargs)
    stream.synchronize()
    torch.testing.assert_close(
        actual, _reference(q, k, v, slots, 1 / 16), atol=4e-3, rtol=1e-2
    )
    graph = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with capture_session(graph), torch.cuda.graph(graph, stream=stream):
            actual = qsa_sparse_attention(q, k, v, slots, **kwargs)
    with torch.cuda.stream(stream):
        metadata.selected_blocks[:query_width].copy_(
            torch.arange(512, 1024, dtype=torch.int32, device=device).expand(
                query_width, -1
            )
        )
        metadata.query_positions.add_(1)
        metadata.selected_blocks[query_width:].fill_(-1)
        for row in range(query_width):
            complete = (31 + row) // 4
            metadata.selected_blocks[query_width + row, :complete] = torch.arange(
                complete, dtype=torch.int32, device=device
            )
        metadata.block_table[:, :2] = metadata.block_table[:, :2].flip(1)
        q.mul_(0.5)
        updated_slots = qwen4_exp_qsa_selected_slots(
            metadata.selected_blocks,
            ((metadata.query_positions + 1) // 4).to(torch.int32),
            metadata.query_positions,
            metadata.token_to_request,
            metadata.block_table,
            64,
            4,
            2048,
            enable_pdl=False,
        )
        if graph is None:
            actual = qsa_sparse_attention(q, k, v, updated_slots, **kwargs)
        else:
            graph.replay()
    stream.synchronize()
    torch.testing.assert_close(
        actual, _reference(q, k, v, updated_slots, 1 / 16), atol=4e-3, rtol=1e-2
    )


def test_qsa_sparse_attention_rejects_ragged_metadata_for_uniform_decode() -> None:
    metadata = QSASparseMetadata(
        torch.zeros((4, 512), dtype=torch.int32),
        torch.ones((2, 1), dtype=torch.int32),
        torch.tensor([0, 1, 1, 1], dtype=torch.int32),
        torch.arange(4),
        (0, 1, 4),
        64,
        4,
        QSAWorkspace(),
    )
    q = torch.empty((4, 8, 256), dtype=torch.bfloat16)
    cache = torch.empty((64, 1, 256), dtype=q.dtype)
    with pytest.raises(ValueError, match="uniform decode width"):
        qsa_sparse_attention(
            q,
            cache,
            cache,
            torch.ones((4, 2051), dtype=torch.int32),
            scale=1 / 16,
            max_seqlen_q=2,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution="flashinfer_prims",
            sparse_metadata=metadata,
        )


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("block_size", [4, 8, 256])
@pytest.mark.parametrize("tensor_scales", [False, True])
def test_prims_qsa_prefill_dispatch(cache_dtype, rows, topk, block_size, tensor_scales):
    platform = current_platform()
    if not platform.is_nvidia or platform.arch_version < ArchVersion(9, 0):
        pytest.skip("QSA BF16/FP8 fallback requires NVIDIA Hopper or newer")
    q = torch.empty((rows, 6, 256), dtype=torch.bfloat16)
    cache = torch.empty((64, 1, 256), dtype=cache_dtype)
    slots = torch.ones((rows, topk * block_size + block_size - 1), dtype=torch.int32)
    metadata = QSASparseMetadata(
        torch.zeros((rows, topk), dtype=torch.int32),
        torch.ones((1, 1), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows),
        (0, rows),
        64,
        block_size,
        QSAWorkspace(),
    )
    expected = (
        "flashinfer_prims_qsa_sparse_attention"
        if platform.arch_version in (ArchVersion(10, 0), ArchVersion(10, 3))
        and cache_dtype == torch.bfloat16
        and topk <= 512
        and block_size in (4, 8)
        and not tensor_scales
        else (
            "flashinfer_fa2_qsa_sparse_attention"
            if cache_dtype == torch.bfloat16
            else "flashinfer_fa2_fp8_qsa_sparse_attention"
        )
    )
    descale = torch.tensor(0.75) if tensor_scales else None
    selected = select_qsa_sparse_attention_kernel(
        q,
        cache,
        cache,
        slots,
        max_seqlen_q=None,
        k_scale=descale,
        v_scale=descale,
        sparse_metadata=metadata,
        override=None,
        solution=None,
    )
    assert selected.name == expected


@pytest.mark.parametrize(
    ("cache_dtype", "block_size"),
    [(torch.bfloat16, 256), (torch.float8_e4m3fn, 4)],
)
def test_prims_qsa_prefill_fa2_fallback_matches_reference(
    cache_dtype: torch.dtype,
    block_size: int,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = current_platform()
    if not platform.is_nvidia or platform.arch_version < ArchVersion(9, 0):
        pytest.skip("QSA BF16/FP8 fallback requires NVIDIA Hopper or newer")
    import tokenspeed_kernel.ops.attention.qsa._flashinfer.runner as runner_module

    # Other tests capture the process runner and pin its metadata shape.
    # Give this fallback geometry a fresh cache of real FA2 runners.
    monkeypatch.setattr(runner_module, "_runners", {})
    q, k, v, slots, metadata = _prims_case(
        lengths=(5, 3),
        prefixes=(385, 29),
        kv_heads=1,
        shared_candidates=False,
        block_size=block_size,
        device=device,
    )
    k, v = (k * 0.25).to(cache_dtype), (v * 0.25).to(cache_dtype)
    descale = 1.0 if cache_dtype == torch.float8_e4m3fn else None
    actual = qsa_sparse_attention(
        q,
        k,
        v,
        slots,
        scale=1 / 16,
        max_seqlen_q=None,
        metadata_capacity_rows=None,
        k_scale=descale,
        v_scale=descale,
        override=None,
        solution=None,
        sparse_metadata=metadata,
    )
    torch.testing.assert_close(
        actual, _reference(q, k, v, slots, 1 / 16), atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize(
    ("offsets", "group_size", "expected"),
    [
        ((0, 5, 5, 8, 17), 4, (0, 4, 5, 8, 12, 16, 17)),
        ((0, 4, 8), 4, (0, 4, 8)),
        ((0, 1, 3), 4, (0, 1, 3)),
        ((0, 0, 0), 4, (0,)),
        ((0, 2, 5), 1, (0, 1, 2, 3, 4, 5)),
    ],
)
def test_prims_query_groups_stop_at_request_boundaries(offsets, group_size, expected):
    # The adapter imports FlashInfer, which is only installed on NVIDIA runners.
    if not current_platform().is_nvidia:
        pytest.skip("FlashInfer PrimTS requires NVIDIA")
    from tokenspeed_kernel.thirdparty.flashinfer.qsa_prims import _group_offsets

    assert _group_offsets(offsets, group_size) == expected


@pytest.mark.parametrize("block_size", [4, 8, 16, 32, 64, 128])
def test_prims_qsa_prefill_supported_block_sizes(block_size, device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    q, k, v, slots, metadata = _prims_case(
        lengths=(5, 3),
        prefixes=(385, 29),
        kv_heads=1,
        shared_candidates=False,
        block_size=block_size,
        device=device,
    )
    actual = qsa_sparse_attention(
        q,
        k,
        v,
        slots,
        scale=1 / 16,
        max_seqlen_q=None,
        metadata_capacity_rows=None,
        k_scale=None,
        v_scale=None,
        override=None,
        solution="flashinfer_prims",
        sparse_metadata=metadata,
    )
    torch.testing.assert_close(
        actual, _reference(q, k, v, slots, 1 / 16), atol=4e-3, rtol=1e-2
    )


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("smaller_first", [False, True])
def test_prims_qsa_interleaves_workspace_geometries(capture, smaller_first, device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    import tokenspeed_kernel.thirdparty.flashinfer.qsa_prims as prims

    large = _prims_case(
        lengths=(257, 3),
        prefixes=(4096, 0),
        kv_heads=1,
        shared_candidates=False,
        block_size=4,
        device=device,
    )
    small = _prims_case(
        lengths=(2, 1),
        prefixes=(17, 0),
        kv_heads=1,
        shared_candidates=True,
        block_size=4,
        device=device,
    )
    first, second = (small, large) if smaller_first else (large, small)
    q, k, v, slots, metadata = first
    expected = _reference(q, k, v, slots, 1 / 16)
    runner = prims._PrimTSQSARunner(q.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner.run(q, k, v, metadata, scale=1 / 16, v_scale=1.0)
    stream.synchronize()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with capture_session(graph), torch.cuda.graph(graph, stream=stream):
            actual = runner.run(q, k, v, metadata, scale=1 / 16, v_scale=1.0)

    # Growing scratch must preserve allocations already used by a graph.
    # Poison the shared scratch too: nonsplit packed runs must regenerate all
    # live workspace values before reading them, including on graph replay.
    for _ in range(2):
        bq, bk, bv, bslots, bmetadata = second
        with torch.cuda.stream(stream):
            other = runner.run(bq, bk, bv, bmetadata, scale=1 / 16, v_scale=1.0)
            runner._workspace.fill_(255)
            if capture:
                graph.replay()
            else:
                actual = runner.run(q, k, v, metadata, scale=1 / 16, v_scale=1.0)
        stream.synchronize()
        torch.testing.assert_close(actual, expected, atol=4e-3, rtol=1e-2)
        torch.testing.assert_close(
            other, _reference(bq, bk, bv, bslots, 1 / 16), atol=4e-3, rtol=1e-2
        )


def test_prims_qsa_keeps_warmed_layer_cache_bindings(device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    import tokenspeed_kernel.thirdparty.flashinfer.qsa_prims as prims

    q, k, v, slots, metadata = _prims_case(
        lengths=(5, 3),
        prefixes=(257, 0),
        kv_heads=1,
        shared_candidates=False,
        block_size=4,
        device=device,
    )
    other_k, other_v = k * 0.5, -v
    runner = prims._PrimTSQSARunner(q.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for keys, values in ((k, v), (other_k, other_v)):
            runner.run(q, keys, values, metadata, scale=1 / 16, v_scale=1.0)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with capture_session(graph), torch.cuda.graph(graph, stream=stream):
        first = runner.run(q, k, v, metadata, scale=1 / 16, v_scale=1.0)
        second = runner.run(q, other_k, other_v, metadata, scale=1 / 16, v_scale=1.0)
    k.mul_(0.75)
    v.mul_(1.25)
    graph.replay()
    for actual, keys, values in ((first, k, v), (second, other_k, other_v)):
        torch.testing.assert_close(
            actual, _reference(q, keys, values, slots, 1 / 16), atol=4e-3, rtol=1e-2
        )


def test_prims_qsa_replays_live_candidates(device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    q, k, v, slots, metadata = _prims_case(
        lengths=(257, 3),
        prefixes=(4096, 0),
        kv_heads=1,
        shared_candidates=True,
        block_size=4,
        device=device,
    )
    kwargs = dict(
        scale=1 / 16,
        max_seqlen_q=None,
        metadata_capacity_rows=None,
        k_scale=None,
        v_scale=None,
        override=None,
        solution="flashinfer_prims",
        sparse_metadata=metadata,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qsa_sparse_attention(q, k, v, slots, **kwargs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with capture_session(graph), torch.cuda.graph(graph, stream=stream):
        actual = qsa_sparse_attention(q, k, v, slots, **kwargs)
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots

    metadata.selected_blocks[:5].add_(512)
    updated_slots = qwen4_exp_qsa_selected_slots(
        metadata.selected_blocks,
        ((metadata.query_positions + 1) // 4).to(torch.int32),
        metadata.query_positions,
        metadata.token_to_request,
        metadata.block_table,
        64,
        4,
        2048,
        enable_pdl=False,
    )
    graph.replay()
    torch.testing.assert_close(
        actual, _reference(q, k, v, updated_slots, 1 / 16), atol=4e-3, rtol=1e-2
    )


@pytest.mark.parametrize("query_width", [None, 1, 4])
def test_prims_qsa_reuses_plan_across_ordered_streams(query_width, device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    q, k, v, slots, metadata = _prims_case(
        lengths=(5, 3) if query_width is None else (query_width, query_width),
        prefixes=(257, 29),
        kv_heads=1,
        shared_candidates=False,
        block_size=4,
        device=device,
    )
    kwargs = dict(
        scale=1 / 16,
        max_seqlen_q=query_width,
        metadata_capacity_rows=None,
        k_scale=None,
        v_scale=None,
        override=None,
        solution="flashinfer_prims",
        sparse_metadata=metadata,
    )
    warmup_stream = torch.cuda.Stream()
    capture_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        qsa_sparse_attention(q, k, v, slots, **kwargs)
    capture_stream.wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    # Capture on another stream without repeating warmup there.
    with capture_session(graph), torch.cuda.graph(graph, stream=capture_stream):
        captured = qsa_sparse_attention(q, k, v, slots, **kwargs)
    warmup_stream.wait_stream(capture_stream)
    with torch.cuda.stream(warmup_stream):
        q.mul_(0.5)
        eager = qsa_sparse_attention(q, k, v, slots, **kwargs)
    capture_stream.wait_stream(warmup_stream)
    with torch.cuda.stream(capture_stream):
        graph.replay()
    torch.cuda.current_stream().wait_stream(capture_stream)
    expected = _reference(q, k, v, slots, 1 / 16)
    torch.testing.assert_close(eager, expected, atol=4e-3, rtol=1e-2)
    torch.testing.assert_close(captured, expected, atol=4e-3, rtol=1e-2)


def test_prims_qsa_captures_many_layers_and_releases_cache_storage(device):
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("PrimTS QSA requires SM100 or SM103")
    q, k, v, slots, metadata = _prims_case(
        lengths=(1,),
        prefixes=(17,),
        kv_heads=1,
        shared_candidates=True,
        block_size=4,
        device=device,
    )
    q = torch.randn((1, 8, 256), device=device, dtype=q.dtype)
    layers = [(k * (1 + i / 100), v * ((i + 1) / 65)) for i in range(65)]
    cache_ref = weakref.ref(layers[0][0])
    kwargs = dict(
        scale=1 / 16,
        max_seqlen_q=1,
        metadata_capacity_rows=None,
        k_scale=None,
        v_scale=None,
        override=None,
        solution="flashinfer_prims",
        sparse_metadata=metadata,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for keys, values in layers:
            qsa_sparse_attention(q, keys, values, slots, **kwargs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with capture_session(graph), torch.cuda.graph(graph, stream=stream):
        outputs = [
            qsa_sparse_attention(q, keys, values, slots, **kwargs)
            for keys, values in layers
        ]
    # Pool rebinding releases eager caches; a live graph still owns its inputs.
    metadata.workspace.clear()
    graph.replay()
    torch.cuda.synchronize()
    for output, (keys, values) in zip(outputs, layers):
        torch.testing.assert_close(
            output, _reference(q, keys, values, slots, 1 / 16), atol=4e-3, rtol=1e-2
        )
    del layers
    gc.collect()
    assert cache_ref() is not None
    del graph
    gc.collect()
    assert cache_ref() is None


@pytest.mark.parametrize(
    "missing_argument",
    [
        "max_seqlen_q",
        "metadata_capacity_rows",
        "k_scale",
        "v_scale",
        "override",
        "solution",
        "sparse_metadata",
    ],
)
def test_qsa_sparse_attention_requires_dispatch_arguments(
    missing_argument: str,
) -> None:
    q = torch.empty((0, 6, 256), dtype=torch.bfloat16)
    cache = torch.zeros((2, 1, 256), dtype=q.dtype)
    slots = torch.empty((0, 2051), dtype=torch.int32)
    kwargs = dict(
        scale=1 / 16,
        max_seqlen_q=None,
        metadata_capacity_rows=None,
        k_scale=None,
        v_scale=None,
        override=None,
        solution=None,
        sparse_metadata=None,
    )
    output = qsa_sparse_attention(q, cache, cache, slots, **kwargs)
    torch.testing.assert_close(output, q)

    del kwargs[missing_argument]
    with pytest.raises(TypeError, match=missing_argument):
        qsa_sparse_attention(q, cache, cache, slots, **kwargs)


def test_qsa_sparse_attention_selects_fa2_fallback(
    a100_platform,
    h100_platform,
) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("FlashInfer FA2 is an NVIDIA fallback")

    traits = {
        "batch_size": 1,
        "q_len": 1,
        "is_decode": True,
        "has_sparse_metadata": False,
        "head_dim": 256,
        "value_head_dim": 256,
        "num_q_heads": 6,
        "num_kv_heads": 1,
        "selected_width": 2051,
    }
    bf16_signature = format_signature(
        q=dense_tensor_format(torch.bfloat16),
        k_cache=dense_tensor_format(torch.bfloat16),
        v_cache=dense_tensor_format(torch.bfloat16),
    )
    fp8_signature = format_signature(
        q=dense_tensor_format(torch.bfloat16),
        k_cache=dense_tensor_format(torch.float8_e4m3fn),
        v_cache=dense_tensor_format(torch.float8_e4m3fn),
    )

    bf16_kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        bf16_signature,
        platform=a100_platform,
        traits=traits,
    )
    assert bf16_kernel.name == "flashinfer_fa2_qsa_sparse_attention"
    fp8_kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        fp8_signature,
        platform=h100_platform,
        traits=traits,
    )
    assert fp8_kernel.name == "flashinfer_fa2_fp8_qsa_sparse_attention"


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("q_len", [1, 4])
def test_qsa_sparse_attention_selects_cute_on_b200_and_b300(
    b200_platform,
    b300_platform,
    cache_dtype: torch.dtype,
    q_len: int,
) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("CuTe DSL QSA requires NVIDIA")
    signature = format_signature(
        q=dense_tensor_format(torch.bfloat16),
        k_cache=dense_tensor_format(cache_dtype),
        v_cache=dense_tensor_format(cache_dtype),
    )
    traits = {
        "batch_size": 1,
        "q_len": q_len,
        "is_decode": True,
        "has_sparse_metadata": False,
        "head_dim": 256,
        "value_head_dim": 256,
        "num_q_heads": 6,
        "num_kv_heads": 1,
        "selected_width": 2051,
    }
    for platform in (b200_platform, b300_platform):
        kernel = select_kernel(
            "attention",
            "qsa_sparse_attention",
            signature,
            features=None,
            platform=platform,
            traits=traits,
            solution=None,
            override=None,
        )
        assert kernel.name == "cute_dsl_blackwell_qsa_sparse_attention"

    kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        signature,
        features=None,
        platform=replace(b300_platform, arch_version=ArchVersion(12, 0)),
        traits=traits,
        solution=None,
        override=None,
    )
    assert kernel.name == (
        "flashinfer_fa2_fp8_qsa_sparse_attention"
        if cache_dtype is torch.float8_e4m3fn
        else "flashinfer_fa2_qsa_sparse_attention"
    )


@pytest.mark.parametrize(
    ("max_seqlen_q", "expected_kernel"),
    [
        (None, "flashinfer_fa2_qsa_sparse_attention"),
        (1, "cute_dsl_blackwell_qsa_sparse_attention"),
        (4, "cute_dsl_blackwell_qsa_sparse_attention"),
    ],
)
def test_qsa_sparse_attention_routes_prefill_and_uniform_decode(
    monkeypatch,
    b200_platform,
    max_seqlen_q,
    expected_kernel,
) -> None:
    if not current_platform().is_blackwell:
        pytest.skip("real CuTe QSA registration requires a Blackwell host")
    rows = 1 if max_seqlen_q is None else max_seqlen_q
    q = torch.empty((rows, 6, 256), dtype=torch.bfloat16, device="cpu")
    cache = torch.empty((16, 1, 256), dtype=torch.bfloat16, device="cpu")
    slots = torch.ones((rows, 2051), dtype=torch.int32, device="cpu")

    def run(kernel, *args, **kwargs):
        assert kernel.name == expected_kernel
        return q

    monkeypatch.setattr(
        qsa_module,
        "select_kernel",
        partial(
            select_kernel,
            features=None,
            platform=b200_platform,
        ),
    )
    monkeypatch.setattr(SelectedKernel, "__call__", run)
    qsa_sparse_attention(
        q,
        cache,
        cache,
        slots,
        scale=1 / 16,
        max_seqlen_q=max_seqlen_q,
        metadata_capacity_rows=None,
        k_scale=None,
        v_scale=None,
        override=None,
        solution=None,
        sparse_metadata=None,
    )


def test_qsa_sparse_attention_validates_uniform_query_length(device: str) -> None:
    q = torch.empty((6, 2, 32), dtype=torch.bfloat16, device=device)
    cache = torch.empty((16, 1, 32), dtype=torch.bfloat16, device=device)
    selected = torch.ones((6, 1), dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="positive"):
        qsa_sparse_attention(
            q,
            cache,
            cache,
            selected,
            scale=1.0,
            max_seqlen_q=0,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution=None,
            sparse_metadata=None,
        )
    with pytest.raises(ValueError, match="divisible"):
        qsa_sparse_attention(
            q,
            cache,
            cache,
            selected,
            scale=1.0,
            max_seqlen_q=4,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
            override=None,
            solution=None,
            sparse_metadata=None,
        )


@pytest.mark.parametrize("solution", ["flashinfer", "cute_dsl"])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("query_width", [1, 4])
def test_qsa_sparse_attention_ignores_nan_in_padding_write_slot(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    solution: str,
    cache_dtype: torch.dtype,
    query_width: int,
) -> None:
    platform = current_platform()
    if not platform.is_nvidia or platform.arch_version < ArchVersion(9, 0):
        pytest.skip("QSA padding test requires NVIDIA Hopper or newer")
    if solution == "cute_dsl" and platform.arch_version not in (
        ArchVersion(10, 0),
        ArchVersion(10, 3),
    ):
        pytest.skip("CuTe DSL QSA requires SM100 or SM103")
    if solution == "flashinfer":
        runner = _FlashInferQSASparseRunner(torch.device(device))
        monkeypatch.setattr(
            "tokenspeed_kernel.ops.attention.qsa.flashinfer.get_flashinfer_qsa_sparse_runner",
            lambda device: runner,
        )

    torch.manual_seed(113)
    rows, live_rows = 4 * query_width, 3 * query_width
    q = torch.randn(rows, 6, 256, device=device, dtype=torch.bfloat16)
    q[live_rows:].fill_(float("nan"))
    k_cache = (torch.randn(512, 1, 256, device=device, dtype=torch.bfloat16) * 0.25).to(
        cache_dtype
    )
    v_cache = (torch.randn(512, 1, 256, device=device, dtype=torch.bfloat16) * 0.25).to(
        cache_dtype
    )
    # The reserved page starts at zero. Padding writes may poison slot 0,
    # while slot 1 remains a finite read target for invalid sparse entries.
    k_cache[:64].zero_()
    v_cache[:64].zero_()
    selected = torch.full((rows, 2051), -1, device=device, dtype=torch.int32)
    selected[:live_rows, :19] = torch.arange(64, 83, device=device)
    selected[:live_rows, 32:40] = 0
    selected[:live_rows, -3] = 128
    kwargs = dict(
        scale=1 / 16,
        max_seqlen_q=query_width,
        metadata_capacity_rows=None,
        k_scale=1.0 if cache_dtype is torch.float8_e4m3fn else None,
        v_scale=1.0 if cache_dtype is torch.float8_e4m3fn else None,
        override=None,
        solution=solution,
        sparse_metadata=None,
    )
    qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)

    # Reproduce a later ReplaySSM forward and refresh the candidate mask.
    k_cache[0].fill_(float("nan"))
    v_cache[0].fill_(float("nan"))
    selected[:live_rows, 1] = -1
    selected[:live_rows, 33] = 129
    expected = _reference(
        q,
        k_cache,
        v_cache,
        selected,
        1 / 16,
    )
    eager = qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    for output in (eager, captured):
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)
        assert torch.count_nonzero(output[live_rows:]) == 0


@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 4, 9])
@pytest.mark.parametrize(
    ("q_heads", "kv_heads"),
    [(6, 1), (12, 2), (24, 4), (12, 1), (24, 1), (24, 2), (6, 2), (12, 4)],
)
def test_qsa_sparse_attention_blackwell_cluster_matches_reference(
    device: str,
    rows: int,
    cache_dtype: torch.dtype,
    q_heads: int,
    kv_heads: int,
) -> None:
    platform = current_platform()
    if platform.arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip(
            "cluster QSA sparse attention is specialized for NVIDIA SM100 or SM103"
        )

    torch.manual_seed(67 + rows)
    cache_slots, head_dim, width = 4096, 256, 2051
    q_storage = torch.randn(
        rows,
        q_heads,
        head_dim * 2,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_storage[..., ::2]
    k_cache = (
        torch.randn(
            cache_slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.25
    ).to(cache_dtype)
    v_cache = (
        torch.randn(
            cache_slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.25
    ).to(cache_dtype)
    slot_storage = torch.full((rows, width * 2), -1, dtype=torch.int32, device=device)
    slots = slot_storage[:, ::2]
    slots[:, :2049] = torch.randint(
        1, cache_slots, (rows, 2049), dtype=torch.int32, device=device
    )
    slots[:, 5::11] = -1
    slots[:, 9::17] = 0
    if rows > 1:
        slots[-1].fill_(-1)

    traits = {
        "batch_size": rows,
        "head_dim": head_dim,
        "value_head_dim": head_dim,
        "num_q_heads": q_heads,
        "num_kv_heads": kv_heads,
        "selected_width": width,
    }
    signature = format_signature(
        q=dense_tensor_format(q.dtype),
        k_cache=dense_tensor_format(k_cache.dtype),
        v_cache=dense_tensor_format(v_cache.dtype),
    )
    selected_kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        signature,
        platform=platform,
        traits=traits,
    )
    assert selected_kernel.name == "cute_dsl_blackwell_qsa_sparse_attention"
    mtp3_kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        signature,
        platform=platform,
        traits={**traits, "batch_size": 32, "q_len": 4},
    )
    assert mtp3_kernel.name == "cute_dsl_blackwell_qsa_sparse_attention"

    scale = head_dim**-0.5
    k_scale, v_scale = (
        (0.5, 0.25) if cache_dtype is torch.float8_e4m3fn else (None, None)
    )
    actual = qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        slots,
        scale=scale,
        max_seqlen_q=(4 if rows == 4 else 1),
        metadata_capacity_rows=None,
        k_scale=k_scale,
        v_scale=v_scale,
        override=None,
        solution=None,
        sparse_metadata=None,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(),
        _reference(
            q,
            k_cache,
            v_cache,
            slots,
            scale,
            k_scale=1.0 if k_scale is None else k_scale,
            v_scale=1.0 if v_scale is None else v_scale,
        ).float(),
        rtol=3.5e-2,
        atol=3.5e-2,
    )


@pytest.mark.parametrize(("q_heads", "kv_heads"), [(1, 1), (8, 1), (16, 2), (24, 8)])
@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_qsa_sparse_attention_blackwell_preserves_other_decode_head_dispatch(
    q_heads: int,
    kv_heads: int,
    cache_dtype: torch.dtype,
) -> None:
    platform = current_platform()
    if platform.arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("cluster QSA sparse attention requires NVIDIA SM100 or SM103")
    selected_kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            k_cache=dense_tensor_format(cache_dtype),
            v_cache=dense_tensor_format(cache_dtype),
        ),
        platform=platform,
        traits={
            "batch_size": 1,
            "q_len": 1,
            "is_decode": True,
            "has_sparse_metadata": False,
            "head_dim": 256,
            "value_head_dim": 256,
            "num_q_heads": q_heads,
            "num_kv_heads": kv_heads,
            "selected_width": 2051,
        },
    )
    assert selected_kernel.name == (
        "flashinfer_fa2_fp8_qsa_sparse_attention"
        if cache_dtype is torch.float8_e4m3fn
        else "flashinfer_fa2_qsa_sparse_attention"
    )


def test_qsa_sparse_attention_blackwell_rejects_single_head_groups(device: str) -> None:
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("cluster QSA sparse attention requires NVIDIA SM100 or SM103")
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import kernel

    q = torch.empty((1, 1, 256), dtype=torch.bfloat16, device=device)
    cache = torch.empty((16, 1, 256), dtype=torch.bfloat16, device=device)
    slots = torch.ones((1, 2051), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="at least two query heads"):
        kernel(
            q,
            cache,
            cache,
            slots,
            scale=256**-0.5,
            max_seqlen_q=1,
            k_scale=None,
            v_scale=None,
            enable_pdl=False,
        )


@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize(("q_heads", "kv_heads"), [(6, 1), (12, 2), (24, 4), (24, 2)])
def test_qsa_sparse_attention_blackwell_cluster_supports_graph_replay(
    device: str,
    cache_dtype: torch.dtype,
    q_heads: int,
    kv_heads: int,
) -> None:
    platform = current_platform()
    if platform.arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip(
            "cluster QSA sparse attention is specialized for NVIDIA SM100 or SM103"
        )
    if (
        KernelRegistry.get().get_by_name("cute_dsl_blackwell_qsa_sparse_attention")
        is None
    ):
        pytest.skip("CuTe DSL QSA sparse attention is unavailable")

    torch.manual_seed(79)
    cache_slots, width = 4096, 2051
    q = torch.randn(1, q_heads, 256, device=device, dtype=torch.bfloat16)
    k_cache = (
        torch.randn(cache_slots, kv_heads, 256, device=device, dtype=torch.bfloat16)
        * 0.25
    ).to(cache_dtype)
    v_cache = (
        torch.randn(cache_slots, kv_heads, 256, device=device, dtype=torch.bfloat16)
        * 0.25
    ).to(cache_dtype)
    selected = torch.randint(
        1, cache_slots, (1, width), device=device, dtype=torch.int32
    )
    scale = 256**-0.5
    k_scale, v_scale = (
        (0.5, 0.25) if cache_dtype is torch.float8_e4m3fn else (None, None)
    )
    kwargs = {
        "scale": scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "max_seqlen_q": 1,
        "metadata_capacity_rows": None,
        "override": "cute_dsl_blackwell_qsa_sparse_attention",
        "solution": None,
        "sparse_metadata": None,
    }

    qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)

    # Alternate head geometry while retaining the captured graph, including
    # equal query-head counts with different KV groups in the compile cache.
    other_q = torch.randn(1, 24, 256, device=device, dtype=torch.bfloat16)
    for other_kv_heads in (1, 2, 4):
        other_k = torch.randn(
            cache_slots, other_kv_heads, 256, device=device, dtype=torch.bfloat16
        ).to(cache_dtype)
        other_v = torch.randn_like(other_k, dtype=torch.bfloat16).to(cache_dtype)
        other_output = qsa_sparse_attention(
            other_q, other_k, other_v, selected, **kwargs
        )
        torch.testing.assert_close(
            other_output,
            _reference(
                other_q,
                other_k,
                other_v,
                selected,
                scale,
                k_scale=1.0 if k_scale is None else k_scale,
                v_scale=1.0 if v_scale is None else v_scale,
            ),
            rtol=3.5e-2,
            atol=3.5e-2,
        )

    replay_query = torch.randn_like(q)
    q.copy_(replay_query)
    selected[:, 1024:].fill_(-1)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output.float(),
        _reference(
            replay_query,
            k_cache,
            v_cache,
            selected,
            scale,
            k_scale=1.0 if k_scale is None else k_scale,
            v_scale=1.0 if v_scale is None else v_scale,
        ).float(),
        rtol=3.5e-2,
        atol=3.5e-2,
    )


@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize(
    ("rows", "q_heads", "kv_heads"),
    [
        (1, 6, 1),
        (4, 6, 1),
        (8, 6, 1),
        (9, 6, 1),
        (1, 12, 2),
        (4, 12, 2),
        (9, 12, 2),
        (1, 24, 4),
        (2, 24, 4),
        (4, 24, 4),
        (1, 12, 1),
        (4, 12, 1),
        (9, 12, 1),
        (1, 24, 1),
        (4, 24, 1),
        (1, 24, 2),
        (4, 24, 2),
    ],
)
def test_qsa_sparse_attention_blackwell_long_context_tail_replay(
    device: str,
    cache_dtype: torch.dtype,
    rows: int,
    q_heads: int,
    kv_heads: int,
) -> None:
    """Exercise every compression phase, empty splits and tail-only attention."""
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("cluster QSA sparse attention requires NVIDIA SM100 or SM103")
    torch.manual_seed(83)
    seq_len, page_size, num_pages = 65539, 256, 1024
    cache_slots = (num_pages + 1) * page_size
    q = torch.randn(rows, q_heads, 256, device=device, dtype=torch.bfloat16)
    k_cache = (
        torch.randn(cache_slots, kv_heads, 256, device=device, dtype=torch.bfloat16)
        * 0.25
    ).to(cache_dtype)
    v_cache = (torch.randn_like(k_cache, dtype=torch.bfloat16) * 0.25).to(cache_dtype)
    # Scatter 512 four-token groups and the three-token remainder over physical
    # pages spanning the full 256K capacity, with page zero reserved.
    pages = torch.randperm(num_pages, device=device, dtype=torch.int32) + 1
    slots = torch.empty((rows, 2051), device=device, dtype=torch.int32)
    offsets = torch.arange(4, device=device, dtype=torch.int32)
    tail = torch.arange(seq_len // 4 * 4, seq_len, device=device, dtype=torch.int32)
    for row in range(rows):
        groups = torch.randperm(seq_len // 4, device=device, dtype=torch.int32)[:512]
        logical = torch.cat(((groups[:, None] * 4 + offsets).flatten(), tail))
        slots[row] = pages[logical // page_size] * page_size + logical % page_size

    k_scale, v_scale = 1.75, 0.25
    kwargs = {
        "scale": 256**-0.5,
        "max_seqlen_q": 4 if rows == 4 else 1,
        "metadata_capacity_rows": None,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "override": "cute_dsl_blackwell_qsa_sparse_attention",
        "solution": None,
        "sparse_metadata": None,
    }
    # Make the tail dominate one head per KV group, forcing rescaling of the full-tile
    # partial rather than merely adding a negligible tail contribution.
    tail_slots = slots[:, -3:].clone()
    for kv_head in range(kv_heads):
        q_head = kv_head * (q_heads // kv_heads)
        k_cache[tail_slots[0, 0].long(), kv_head] = (q[0, q_head] * 8).to(cache_dtype)
    v_cache[0].fill_(1.0e3 if cache_dtype is torch.bfloat16 else 256.0)
    slots[:, 13] = 0
    slots[:, 24] = -1
    slots[:, 33] = slots[:, 34]  # Duplicate slots retain their softmax weight.
    qsa_sparse_attention(q, k_cache, v_cache, slots, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_attention(q, k_cache, v_cache, slots, **kwargs)

    for valid_tail in (0, 1, 2, 3):
        slots[:, -3:].fill_(-1)
        slots[:, 2048 : 2048 + valid_tail] = tail_slots[:, :valid_tail]
        q.mul_(-1)
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(q, k_cache, v_cache, slots, 256**-0.5, k_scale, v_scale)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, rtol=3.5e-2, atol=3.5e-2)

    # All full-tile splits are empty; only the SIMD remainder contributes.
    slots[:, :2048].fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        _reference(q, k_cache, v_cache, slots, 256**-0.5, k_scale, v_scale),
        rtol=3.5e-2,
        atol=3.5e-2,
    )
    slots.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize(
    (
        "cache_dtype",
        "splits",
        "query_rows_per_cta",
        "bf16_smem_slots",
        "bf16_async_ready",
    ),
    [
        (torch.bfloat16, 4, 1, 1, False),
        (torch.bfloat16, 8, 1, 1, False),
        (torch.bfloat16, 16, 1, 3, False),
        (torch.float8_e4m3fn, 4, 1, 1, False),
        (torch.float8_e4m3fn, 8, 1, 1, False),
        (torch.float8_e4m3fn, 16, 1, 3, False),
        (torch.bfloat16, 1, 2, 3, True),
        (torch.bfloat16, 1, 4, 3, True),
        (torch.bfloat16, 2, 1, 3, True),
        (torch.bfloat16, 4, 1, 3, True),
    ],
)
def test_qsa_sparse_attention_blackwell_changed_kv_replay(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    cache_dtype: torch.dtype,
    splits: int,
    query_rows_per_cta: int,
    bf16_smem_slots: int,
    bf16_async_ready: bool,
) -> None:
    """KV staging and partial query tiles must observe changes on graph replay."""
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("cluster QSA sparse attention requires NVIDIA SM100 or SM103")
    import tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse as sparse_module

    if (
        splits == 16
        and sparse_module._wide_cluster_capacity(torch.cuda.current_device()) == 0
    ):
        pytest.skip("the device cannot launch a sixteen-CTA cluster")

    def select_config(
        num_rows: int,
        head_tiles_per_row: int,
        bf16_kv: bool,
        sm_count: int,
        wide_cluster_capacity: int,
    ) -> tuple[int, int, int, bool]:
        del num_rows, head_tiles_per_row, bf16_kv, sm_count, wide_cluster_capacity
        return query_rows_per_cta, splits, bf16_smem_slots, bf16_async_ready

    monkeypatch.setattr(sparse_module, "_select_launch_config", select_config)
    torch.manual_seed(217 + splits)
    rows, q_heads, kv_heads, cache_slots = 3, 24, 2, 4096
    q = torch.randn(rows, q_heads, 512, device=device, dtype=torch.bfloat16)[..., ::2]
    k_cache = (
        torch.randn(cache_slots, kv_heads, 256, device=device, dtype=torch.bfloat16)
        * 0.25
    ).to(cache_dtype)
    v_cache = (torch.randn_like(k_cache, dtype=torch.bfloat16) * 0.25).to(cache_dtype)
    selected = torch.randint(
        1, cache_slots, (rows, 4102), device=device, dtype=torch.int32
    )[:, ::2]
    selected[:, 7::11] = -1
    selected[:, 9::17] = 0
    selected[:, 19] = selected[:, 20]
    selected[-1].fill_(-1)
    v_cache[0].fill_(256)
    kwargs = {
        "scale": 1 / 16,
        "max_seqlen_q": 1,
        "metadata_capacity_rows": None,
        "k_scale": 1.75,
        "v_scale": 0.25,
        "override": "cute_dsl_blackwell_qsa_sparse_attention",
        "solution": None,
        "sparse_metadata": None,
    }
    qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    previous = output.clone()
    for update in range(2):
        # Keep query and indices fixed so the output change must come from KV.
        k_cache.copy_(
            (torch.randn_like(k_cache, dtype=torch.bfloat16) * 0.35).to(cache_dtype)
        )
        v_cache.copy_(
            (
                torch.randn_like(v_cache, dtype=torch.bfloat16) * 0.35
                + 0.5 * (update + 1)
            ).to(cache_dtype)
        )
        v_cache[0].fill_(256)
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(q, k_cache, v_cache, selected, 1 / 16, 1.75, 0.25)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.002)
        assert torch.isfinite(output).all()
        assert not torch.equal(output[:-1], previous[:-1])
        assert torch.count_nonzero(output[-1]).item() == 0
        previous.copy_(output)
    selected.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.parametrize(
    ("num_clusters", "capacity", "splits"),
    [(1, 7, 16), (7, 7, 16), (8, 7, 8), (1, 0, 8), (9, 7, 4)],
)
def test_qsa_sparse_attention_blackwell_cluster_capacity(
    num_clusters: int,
    capacity: int,
    splits: int,
) -> None:
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("cluster QSA sparse attention requires NVIDIA SM100 or SM103")
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import _num_splits

    assert _num_splits(num_clusters, capacity) == splits


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("splits", [1, 4])
def test_qsa_sparse_attention_blackwell_large_kv_offsets(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    cache_dtype: torch.dtype,
    splits: int,
) -> None:
    """Gather, prefetch and tail reads must cross the int32 element boundary."""
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("large-cache QSA requires NVIDIA SM100 or SM103")
    import tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse as sparse_module

    def select_config(
        num_rows: int,
        head_tiles_per_row: int,
        bf16_kv: bool,
        sm_count: int,
        wide_cluster_capacity: int,
    ) -> tuple[int, int, int, bool]:
        del num_rows, head_tiles_per_row, bf16_kv, sm_count, wide_cluster_capacity
        # Two BF16 buffers keep lookahead prefetch active at the large addresses.
        slots = 2 if cache_dtype is torch.bfloat16 else 3
        return 1, splits, slots, cache_dtype is torch.bfloat16

    monkeypatch.setattr(sparse_module, "_select_launch_config", select_config)
    torch.manual_seed(313)
    rows, q_heads, kv_heads, head_dim = 3, 24, 2, 256
    boundary_slot = (1 << 31) // (kv_heads * head_dim)
    cache_slots = boundary_slot + 32
    required_bytes = 2 * cache_slots * kv_heads * head_dim * cache_dtype.itemsize
    if torch.cuda.mem_get_info(device)[0] < required_bytes + (1 << 30):
        pytest.skip("large-offset regression needs room for two 2^31-element caches")
    q = torch.randn(rows, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.empty(cache_slots, kv_heads, head_dim, dtype=cache_dtype, device=device)
    v = torch.empty_like(k)
    # Only initialize the rows that can be read; allocating this cache tests
    # the real pointer arithmetic without filling gigabytes of unused history.
    physical = torch.tensor(
        [2, 3, boundary_slot - 1, boundary_slot, boundary_slot + 1, cache_slots - 1],
        dtype=torch.int32,
        device=device,
    )
    k[1].zero_()
    v[1].zero_()
    k[physical.long()] = (torch.randn(6, kv_heads, head_dim, device=device) * 0.25).to(
        cache_dtype
    )
    v[physical.long()] = (torch.randn(6, kv_heads, head_dim, device=device) * 0.25).to(
        cache_dtype
    )
    selected = physical[torch.arange(2051, device=device) % 6].repeat(rows, 1)
    selected[1, 2048:] = -1
    selected[2].fill_(-1)

    def forward():
        return sparse_module.kernel(
            q,
            k,
            v,
            selected,
            scale=1 / 16,
            max_seqlen_q=1,
            k_scale=None,
            v_scale=None,
            enable_pdl=False,
        )

    expected = _reference(q, k, v, selected, 1 / 16)
    torch.testing.assert_close(forward(), expected, atol=2e-3, rtol=2e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    # Keep K and the slots fixed: changed high-address V must be read again.
    v[physical.long()] = (v[physical.long()].float() + 0.125).to(cache_dtype)
    graph.replay()
    torch.testing.assert_close(
        output, _reference(q, k, v, selected, 1 / 16), atol=2e-3, rtol=2e-2
    )
    assert torch.count_nonzero(output[2]).item() == 0


@pytest.mark.parametrize(
    ("rows", "head_tiles", "bf16", "sm_count", "expected"),
    [
        (32, 1, True, 152, (1, 4, 3, True)),
        (64, 1, True, 152, (1, 2, 3, True)),
        (128, 1, True, 152, (1, 1, 3, True)),
        (256, 1, True, 152, (1, 1, 3, True)),
        (512, 1, True, 152, (1, 1, 3, True)),
        (513, 1, True, 152, (1, 1, 3, True)),
        (128, 4, True, 152, (1, 1, 3, True)),
        (416, 4, True, 152, (1, 1, 3, True)),
        (512, 1, True, 80, (1, 1, 3, True)),
        (512, 1, False, 152, (1, 4, 1, False)),
    ],
)
def test_qsa_sparse_attention_blackwell_output_tiling(
    rows: int,
    head_tiles: int,
    bf16: bool,
    sm_count: int,
    expected: tuple[int, int, int, bool],
) -> None:
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("CTA tiling requires NVIDIA SM100 or SM103")
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import _select_launch_config

    assert _select_launch_config(rows, head_tiles, bf16, sm_count, 7) == expected


@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_qsa_sparse_attention_flashinfer_fa2_matches_reference_and_reuses_plan(
    device: str,
    rows: int,
    cache_dtype: torch.dtype,
    is_prefill: bool,
) -> None:
    platform = current_platform()
    if not platform.is_nvidia or platform.arch_version < ArchVersion(8, 0):
        pytest.skip("FlashInfer FA2 QSA requires NVIDIA Ampere or newer")
    if cache_dtype is torch.float8_e4m3fn and platform.arch_version < ArchVersion(9, 0):
        pytest.skip("FP8 FlashInfer FA2 QSA requires NVIDIA Hopper or newer")
    torch.manual_seed(37 + rows)
    cache_slots, q_heads, kv_heads, head_dim, width = 4096, 6, 1, 256, 2051
    q = torch.randn(rows, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = (
        torch.randn(
            cache_slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.25
    ).to(cache_dtype)
    v_cache = (
        torch.randn(
            cache_slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        * 0.25
    ).to(cache_dtype)
    selected = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        valid = 1424 + row
        selected[row, :valid] = torch.randint(
            1, cache_slots, (valid,), dtype=torch.int32, device=device
        )
        selected[row, width - 3 : width - 1] = torch.randint(
            1, cache_slots, (2,), dtype=torch.int32, device=device
        )
    scale = head_dim**-0.5
    k_scale, v_scale = (
        (0.5, 0.25) if cache_dtype is torch.float8_e4m3fn else (None, None)
    )
    runner = get_flashinfer_qsa_sparse_runner(q.device)
    plan = runner.plan(
        q,
        k_cache,
        v_cache,
        width,
        softmax_scale=scale,
        metadata_capacity_rows=None,
    )

    first = qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected,
        scale=scale,
        max_seqlen_q=None if is_prefill else (4 if rows == 4 else 1),
        metadata_capacity_rows=None,
        k_scale=k_scale,
        v_scale=v_scale,
        override=None,
        solution=None if is_prefill else "flashinfer",
        sparse_metadata=None,
    )
    selected[:, :256] = torch.randint(
        1, cache_slots, (rows, 256), dtype=torch.int32, device=device
    )
    if rows > 1:
        selected[-1].fill_(-1)
    second = qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected,
        scale=scale,
        max_seqlen_q=None if is_prefill else (4 if rows == 4 else 1),
        metadata_capacity_rows=None,
        k_scale=k_scale,
        v_scale=v_scale,
        override=None,
        solution=None if is_prefill else "flashinfer",
        sparse_metadata=None,
    )

    assert (
        runner.plan(
            q,
            k_cache,
            v_cache,
            width,
            softmax_scale=scale,
            metadata_capacity_rows=None,
        )
        is plan
    )
    assert torch.isfinite(first).all()
    torch.testing.assert_close(
        second.float(),
        _reference(
            q,
            k_cache,
            v_cache,
            selected,
            scale,
            k_scale=1.0 if k_scale is None else k_scale,
            v_scale=1.0 if v_scale is None else v_scale,
        ).float(),
        rtol=3.5e-2,
        atol=3.5e-2,
    )


@pytest.mark.parametrize("width", [32, 33, 2051])
def test_flashinfer_qsa_runner_reuses_one_high_watermark_buffer(
    device: str, width: int, monkeypatch
) -> None:
    platform = current_platform()
    if not platform.is_nvidia or platform.arch_version < ArchVersion(8, 0):
        pytest.skip("FlashInfer FA2 QSA requires NVIDIA Ampere or newer")
    import flashinfer.sparse

    def fail_on_rowwise_mask_conversion(*args, **kwargs):
        raise AssertionError("QSA planning must not convert a bool mask row by row")

    monkeypatch.setattr(
        flashinfer.sparse,
        "convert_bsr_mask_layout",
        fail_on_rowwise_mask_conversion,
    )
    rows, head_dim = 4, 64
    q = torch.randn(rows, 4, head_dim, dtype=torch.bfloat16, device=device)
    cache = torch.randn(64, 1, head_dim, dtype=torch.bfloat16, device=device)
    runner = _FlashInferQSASparseRunner(q.device)

    one = runner.plan(
        q[:1],
        cache,
        cache,
        width,
        softmax_scale=0.125,
        metadata_capacity_rows=rows,
    )
    fixed_buffer_ptr = one.indices.data_ptr()
    assert one.indices.untyped_storage().nbytes() >= rows * width * 4
    four = runner.plan(
        q,
        cache,
        cache,
        width,
        softmax_scale=0.125,
        metadata_capacity_rows=None,
    )
    assert four.indices.data_ptr() == fixed_buffer_ptr

    two = runner.plan(
        q[:2],
        cache,
        cache,
        width,
        softmax_scale=0.125,
        metadata_capacity_rows=None,
    )
    assert two.indices.data_ptr() == fixed_buffer_ptr
    torch.testing.assert_close(
        four.wrapper._mask_indptr_buf,
        torch.arange(rows + 1, device=device, dtype=torch.int32) * ((width + 7) // 8),
    )
    assert four.wrapper._int_workspace_buffer.numel() < 8 * 1024 * 1024
    assert four.wrapper._pin_memory_int_workspace_buffer.numel() == 0

    again = runner.plan(
        q[:1],
        cache,
        cache,
        width,
        softmax_scale=0.125,
        metadata_capacity_rows=None,
    )
    assert again is one
    assert again.indices.data_ptr() == fixed_buffer_ptr


@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_qsa_sparse_attention_flashinfer_fa2_supports_graph_replay(
    device: str,
    cache_dtype: torch.dtype,
) -> None:
    platform = current_platform()
    if not platform.is_nvidia or platform.arch_version < ArchVersion(9, 0):
        pytest.skip("graph test requires FlashInfer FA2 on Hopper or newer")
    torch.manual_seed(97)
    cache_slots, width = 4096, 2051
    q = torch.randn(1, 6, 256, device=device, dtype=torch.bfloat16)
    k_cache = (
        torch.randn(cache_slots, 1, 256, device=device, dtype=torch.bfloat16) * 0.25
    ).to(cache_dtype)
    v_cache = (
        torch.randn(cache_slots, 1, 256, device=device, dtype=torch.bfloat16) * 0.25
    ).to(cache_dtype)
    selected = torch.randint(
        1, cache_slots, (1, width), device=device, dtype=torch.int32
    )
    scale = 256**-0.5
    k_scale, v_scale = (
        (0.5, 0.25) if cache_dtype is torch.float8_e4m3fn else (None, None)
    )
    kwargs = {
        "scale": scale,
        "max_seqlen_q": 1,
        "metadata_capacity_rows": 2,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "override": None,
        "solution": "flashinfer",
        "sparse_metadata": None,
    }

    qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = qsa_sparse_attention(q, k_cache, v_cache, selected, **kwargs)

    # Replan the mutable runner for another row count. The directly captured
    # plan must keep its own pointer-stable storage for later replay.
    other_q = torch.randn(2, 6, 256, device=device, dtype=torch.bfloat16)
    other_selected = torch.randint(
        1, cache_slots, (2, width), device=device, dtype=torch.int32
    )
    qsa_sparse_attention(
        other_q,
        k_cache,
        v_cache,
        other_selected,
        **kwargs,
    )

    replay_query = torch.randn_like(q)
    q.copy_(replay_query)
    selected[:, 1024:].fill_(-1)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output.float(),
        _reference(
            replay_query,
            k_cache,
            v_cache,
            selected,
            scale,
            k_scale=1.0 if k_scale is None else k_scale,
            v_scale=1.0 if v_scale is None else v_scale,
        ).float(),
        rtol=3.5e-2,
        atol=3.5e-2,
    )
