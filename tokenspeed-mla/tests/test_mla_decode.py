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


"""MLA decode regression tests; GPU checks are grouped in TestGPU."""

import inspect
import itertools
import math
import multiprocessing
import traceback
from dataclasses import dataclass

import pytest
import torch
from tokenspeed_mla.mla_helpers import compute_q_tile_layout


@dataclass(frozen=True)
class _Case:
    batch: int
    kv_len: int
    heads: int
    q_len: int

    def __post_init__(self):
        if min(self.batch, self.kv_len, self.heads, self.q_len) <= 0:
            raise ValueError("B, K, H and Q must be positive")
        if self.heads > 128 or self.q_len > self.kv_len:
            raise ValueError("Require H <= 128 and Q <= K")

    @property
    def name(self):
        return f"B{self.batch}_K{self.kv_len}_H{self.heads}_Q{self.q_len}"


def _make_inputs(case, dtype_name, variable_kv, device):
    """Return deterministic Q, paged KV, page tables and request-local KV lengths."""
    page_size = 64

    generator = torch.Generator(device=device).manual_seed(42)
    dtype = {
        "fp8": torch.float8_e4m3fn,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]
    source_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    pages = math.ceil(case.kv_len / page_size)
    query = torch.randn(
        case.batch,
        case.q_len,
        case.heads,
        576,
        generator=generator,
        device=device,
        dtype=source_dtype,
    ).to(dtype)
    kv = torch.randn(
        case.batch * pages,
        page_size,
        576,
        generator=generator,
        device=device,
        dtype=source_dtype,
    ).to(dtype)
    # Unique shuffled physical pages avoid accidental cross-request sharing.
    # CuTe TMA page-table rows must have an even number of int32 indices at P64.
    table_width = math.ceil(pages / (128 // page_size)) * (128 // page_size)
    tables = torch.zeros(case.batch, table_width, device=device, dtype=torch.int32)
    tables[:, :pages] = torch.randperm(
        case.batch * pages, generator=generator, device=device, dtype=torch.int32
    ).reshape(case.batch, pages)
    lengths = [
        (
            max(case.q_len, case.kv_len - (b * 37) % max(1, case.kv_len // 2))
            if variable_kv
            else case.kv_len
        )
        for b in range(case.batch)
    ]
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    return query, kv, tables, seq_lens


def _reference_mla(query, kv, tables, seq_lens, request_indices):
    """Return FP32 outputs for selected requests using dequantized inputs.

    Each query token sees K-Q+token+1 keys (bottom-right causal alignment).
    Gather one request at a time to bound long-context reference memory.
    """
    q_len, heads = query.shape[1:3]
    page_size = kv.shape[1]
    outputs = []
    for batch_idx in request_indices:
        length = int(seq_lens[batch_idx])
        page_ids = tables[batch_idx, : math.ceil(length / page_size)].long()
        # CPU advanced indexing does not implement FP8; gather its byte view.
        cache = kv.view(torch.uint8)[page_ids].view(kv.dtype)
        cache = cache.reshape(-1, 576)[:length].float()
        q = query[batch_idx].float().reshape(q_len * heads, 576)
        logits = (q @ cache.T).reshape(q_len, heads, length) * 192**-0.5
        visible = length - q_len + torch.arange(q_len, device=q.device) + 1
        mask = torch.arange(length, device=q.device)[None, :] < visible[:, None]
        logits.masked_fill_(~mask[:, None, :], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        outputs.append(probs @ cache[:, :512])
    return torch.stack(outputs)


def _check_output(actual, expected, dtype):
    """Check output dtype and values against FP32 reference."""
    output_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    if actual.dtype != output_dtype:
        raise AssertionError(f"Expected {output_dtype} output, got {actual.dtype}")
    actual = actual.float()
    # Match PR #4178's FP8 elementwise tolerance. Quantized softmax weights
    # can produce larger absolute errors on rows with cancellation; retain
    # the stricter global relative-RMSE guard below, including at long K.
    tolerance = 0.1 if dtype == "fp8" else 0.01
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    delta = actual - expected
    relative_rmse = (
        delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    ).item()
    # An absolute tolerance alone could accept all-zero outputs at long K.
    if relative_rmse > (0.05 if dtype == "fp8" else 0.01):
        raise AssertionError(f"Relative RMSE too large: {relative_rmse:.6f}")


@pytest.mark.parametrize(
    "batch,q_len,heads,splits,expected",
    [
        (1, 1, 6, 32, 4),
        (1, 1, 48, 32, 2),
        (1, 1, 96, 32, 4),
        (1, 8, 96, 8, 1),
        (4, 1, 48, 32, 1),
        (1, 1, 6, 2, 2),
        (1, 1, 6, 1, 1),
    ],
)
def test_reducer_parallelism_uses_real_rows_and_available_splits(
    batch, q_len, heads, splits, expected
):
    decode = pytest.importorskip("tokenspeed_mla.mla_decode")
    assert decode._get_reducer_d_tiles(batch, q_len, heads, 148, splits) == expected


@pytest.mark.parametrize(
    "kv_len,tile_m,expected_splits",
    [
        (128, 128, 1),
        (384, 128, 3),
        (1024, 128, 8),
        (4096, 128, 32),
        (16512, 64, 43),
    ],
)
def test_fp8_workspace_counts_only_nonempty_partitions(kv_len, tile_m, expected_splits):
    decode = pytest.importorskip("tokenspeed_mla.mla_decode")
    splits, workspace = decode._get_split_kv_and_workspace_size(
        1, 1, 48, 512, 148, kv_len, torch.float8_e4m3fn, (tile_m, 128)
    )
    assert splits == expected_splits
    assert workspace == (0 if splits == 1 else 48 * splits * 513 * 4)
    k_tiles = (kv_len + 127) // 128
    tiles_per_split = (k_tiles + splits - 1) // splits
    assert (splits - 1) * tiles_per_split < k_tiles <= splits * tiles_per_split


def test_bf16_split_selection_is_unchanged():
    decode = pytest.importorskip("tokenspeed_mla.mla_decode")
    splits, workspace = decode._get_split_kv_and_workspace_size(
        1, 1, 96, 512, 148, 128, torch.bfloat16, (128, 128)
    )
    assert splits == 32
    assert workspace == 96 * 32 * 513 * 4


@pytest.mark.parametrize("dtype", ["fp8", "bf16"])
def test_input_pages_are_unique_and_tma_aligned(dtype):
    case = _Case(4, 129, 12, 8)
    q, kv, tables, lengths = _make_inputs(case, dtype, True, "cpu")
    q_again, kv_again, _, _ = _make_inputs(case, dtype, True, "cpu")
    assert torch.equal(q.view(torch.uint8), q_again.view(torch.uint8))
    assert torch.equal(kv.view(torch.uint8), kv_again.view(torch.uint8))
    assert tables.shape == (4, 4)  # Three used pages and one padded table index.
    assert torch.unique(tables[:, :3]).numel() == 12
    assert lengths.max() == 129
    assert lengths.min() >= case.q_len
    assert torch.unique(lengths).numel() > 1


def test_reference_uses_page_table_lengths_and_bottom_right_causality():
    # Zero logits make attention a simple prefix mean. Logical tokens are
    # [1, 3, 7] and [9, 11], with deliberately permuted physical pages.
    q = torch.zeros(2, 2, 1, 576)
    kv = torch.full((4, 2, 576), 1000.0)
    kv[2, 0].fill_(1)
    kv[2, 1].fill_(3)
    kv[0, 0].fill_(7)
    kv[3, 0].fill_(9)
    kv[3, 1].fill_(11)
    tables = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    lengths = torch.tensor([3, 2], dtype=torch.int32)
    output = _reference_mla(q, kv, tables, lengths, [0, 1])
    expected = torch.tensor([[2, 11 / 3], [9, 10]])[:, :, None, None].expand(
        2, 2, 1, 512
    )
    torch.testing.assert_close(output, expected)


def test_reference_dequantizes_fp8_before_arithmetic():
    q, kv, tables, lengths = _make_inputs(_Case(2, 67, 6, 4), "fp8", False, "cpu")
    output = _reference_mla(q, kv, tables, lengths, [0, 1])
    expected = _reference_mla(q.float(), kv.float(), tables, lengths, [0, 1])
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("dtype", ["fp8", "fp16", "bf16"])
def test_input_and_output_dtype_contract(dtype):
    q, kv, tables, lengths = _make_inputs(_Case(1, 67, 6, 4), dtype, False, "cpu")
    expected_dtype = {
        "fp8": torch.float8_e4m3fn,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype]
    assert q.dtype == kv.dtype == expected_dtype
    expected = _reference_mla(q, kv, tables, lengths, [0])
    output_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    _check_output(expected.to(output_dtype), expected, dtype)
    wrong_dtype = torch.bfloat16 if dtype == "fp16" else torch.float16
    with pytest.raises(AssertionError, match="output"):
        _check_output(expected.to(wrong_dtype), expected, dtype)


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_accuracy_check_rejects_empty_graph_and_zero_long_context_outputs(value):
    expected = torch.full((1, 1, 1, 512), 0.001)
    actual = torch.full(expected.shape, value, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        _check_output(actual, expected, "fp8")


@pytest.mark.parametrize(
    "heads,queries,expected",
    [
        (96, 4, (384, 3, 128)),
        (96, 8, (768, 6, 128)),
        (96, 3, (288, 3, 32)),
        (48, 3, (144, 2, 16)),
        (12, 11, (132, 2, 4)),
        (128, 3, (384, 3, 128)),
    ],
)
def test_packed_tile_geometry(heads, queries, expected):
    assert compute_q_tile_layout(heads, queries, 128) == expected


@pytest.mark.parametrize("args", [(0, 4, 128), (96, 0, 128), (129, 4, 128), (96, 4, 0)])
def test_invalid_packed_geometry(args):
    with pytest.raises(ValueError):
        compute_q_tile_layout(*args)


@pytest.mark.parametrize("batch,expected_splits", [(1, 24), (4, 6)])
def test_packed_workspace_matches_launch_geometry(batch, expected_splits):
    import tokenspeed_mla.mla_decode as decode

    _, tiles, _ = compute_q_tile_layout(96, 4, 128)
    splits, workspace = decode._get_split_kv_and_workspace_size(
        batch, tiles, 128, 512, 148, 65536, torch.float8_e4m3fn, (128, 128)
    )
    assert splits == expected_splits
    assert workspace == batch * 128 * tiles * splits * 513 * 4


def test_packed_q_is_opt_in():
    import tokenspeed_mla.mla_decode as decode
    from tokenspeed_mla.mla_decode_fp16 import (
        BlackwellMultiHeadLatentAttentionForwardFP16,
    )

    assert (
        inspect.signature(decode.tokenspeed_mla_decode)
        .parameters["enable_packed_q"]
        .default
        is False
    )
    assert (
        inspect.signature(BlackwellMultiHeadLatentAttentionForwardFP16)
        .parameters["pack_q"]
        .default
        is False
    )
    assert (
        inspect.signature(decode._get_compiled_mla_kernel).parameters["pack_q"].default
        is False
    )


def _check_fp8_gpu(case, variable_kv):
    from tokenspeed_mla import tokenspeed_mla_decode

    q, kv, tables, lengths = _make_inputs(case, "fp8", variable_kv, "cuda")
    workspace = torch.zeros(256 * 1024**2, dtype=torch.int8, device="cuda")
    out = torch.empty(
        case.batch,
        case.q_len,
        case.heads,
        512,
        dtype=torch.bfloat16,
        device="cuda",
    )
    indices = (
        list(range(case.batch))
        if case.batch <= 4
        else sorted({0, case.batch // 2, case.batch - 1})
    )
    expected = _reference_mla(q, kv, tables, lengths, indices)
    kwargs = dict(
        query=q,
        kv_cache=kv,
        workspace_buffer=workspace,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=tables,
        seq_lens=lengths,
        max_seq_len=case.kv_len,
        softmax_scale=192**-0.5,
        output_scale=1.0,
        out=out,
        is_var_seq=variable_kv,
        causal_mask=True,
        window_left=-1,
        enable_pdl=False,
        return_lse=False,
        causal_seqs=None,
        cp_world=1,
        cp_rank=0,
        enable_packed_q=False,
    )
    tokenspeed_mla_decode(**kwargs)
    torch.cuda.synchronize()
    _check_output(out[indices], expected, "fp8")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        tokenspeed_mla_decode(**kwargs)
    out.fill_(float("nan"))
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    _check_output(out[indices], expected, "fp8")


def _check_packed_gpu():
    import tokenspeed_mla.mla_decode as decode

    torch.backends.cuda.matmul.allow_tf32 = False
    workspace = torch.zeros(32 * 1024**2, dtype=torch.int8, device="cuda")
    original_compile = decode._get_compiled_mla_kernel
    selections = []

    def record_compile(*args, **kwargs):
        selections.append(kwargs["pack_q"])
        return original_compile(*args, **kwargs)

    decode._get_compiled_mla_kernel = record_compile
    cases = [
        # Cross-query masks straddle a K128 boundary, with split-KV.
        (_Case(1, 129, 96, 4), "fp8", False, False, False, True),
        (_Case(4, 257, 96, 8), "fp8", True, True, False, True),
        # Partial final rows, no reducer, and multiple persistent work items.
        (_Case(32, 128, 96, 3), "fp8", False, False, False, True),
        (_Case(1, 257, 48, 3), "fp8", False, True, False, True),
        # FP16/BF16 share packing, including tails, folding and PDL.
        (_Case(1, 129, 96, 4), "bf16", False, False, False, True),
        (_Case(1, 129, 96, 4), "fp16", False, False, False, True),
        (_Case(4, 257, 96, 8), "fp16", True, True, False, True),
        (_Case(32, 128, 96, 3), "fp16", False, False, False, True),
        (_Case(1, 129, 48, 4), "fp16", False, False, False, True),
        # Existing M64 and token-gapped M128 paths stay selected.
        (_Case(1, 129, 12, 4), "fp8", False, False, False, False),
        (_Case(1, 129, 96, 4), "fp8", False, False, True, False),
        (_Case(1, 129, 96, 4), "fp16", False, False, True, False),
    ]
    for case, dtype, variable_kv, pdl, token_gap, expected_packed in cases:
        q, kv, tables, lengths = _make_inputs(case, dtype, variable_kv, "cuda")
        if token_gap:
            storage = torch.empty(
                (case.batch, case.q_len * 2, case.heads, 576),
                device="cuda",
                dtype=q.dtype,
            )
            view = storage[:, ::2]
            view.copy_(q)
            q = view
        indices = list(range(case.batch))
        expected = _reference_mla(q, kv, tables, lengths, indices)
        # Guard the final output allocation against tail-row writes.
        storage = torch.full(
            (case.batch * case.q_len * case.heads * 512 + 1024,),
            123.0,
            dtype=torch.float16 if dtype == "fp16" else torch.bfloat16,
            device="cuda",
        )
        out = storage[512:-512].view(case.batch, case.q_len, case.heads, 512)
        kwargs = dict(
            query=q,
            kv_cache=kv,
            workspace_buffer=workspace,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=tables,
            seq_lens=lengths,
            max_seq_len=case.kv_len,
            softmax_scale=192**-0.5,
            is_var_seq=variable_kv,
            enable_pdl=pdl,
            out=out,
            return_lse=True,
        )
        default_out, default_lse = decode.tokenspeed_mla_decode(**kwargs)
        assert selections[-1] is False
        default_out = default_out.clone()
        default_lse = default_lse.clone()
        _check_output(default_out, expected, dtype)
        disabled_out, disabled_lse = decode.tokenspeed_mla_decode(
            **kwargs, enable_packed_q=False
        )
        torch.testing.assert_close(disabled_out, default_out, atol=0, rtol=0)
        torch.testing.assert_close(disabled_lse, default_lse, atol=0, rtol=0)
        result, lse = decode.tokenspeed_mla_decode(**kwargs, enable_packed_q=True)
        assert selections[-1] is expected_packed
        torch.cuda.synchronize()
        _check_output(result, expected, dtype)
        if dtype != "fp8":
            torch.testing.assert_close(lse, default_lse, atol=2e-5, rtol=1e-5)
        if not expected_packed:
            torch.testing.assert_close(result, default_out, atol=0, rtol=0)
        # Uniform attention exposes off-by-one causal boundaries in LSE.
        q.zero_()
        expected = _reference_mla(q, kv, tables, lengths, indices)
        visible = (
            lengths[:, None]
            - case.q_len
            + torch.arange(1, case.q_len + 1, device="cuda")[None, :]
        )
        expected_lse = visible.float().log2()[:, :, None].expand(-1, -1, case.heads)
        result, lse = decode.tokenspeed_mla_decode(**kwargs, enable_packed_q=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=1e-5)
        _check_output(result, expected, dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result, lse = decode.tokenspeed_mla_decode(**kwargs, enable_packed_q=True)
        result.fill_(float("nan"))
        lse.fill_(float("nan"))
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        _check_output(result, expected, dtype)
        torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=1e-5)
        assert torch.all(storage[:512] == 123) and torch.all(storage[-512:] == 123)
        print(
            f"Packed Q verified: {case.name} dtype={dtype} packed={expected_packed}",
            flush=True,
        )


def _check_packed_masks_gpu():
    from tokenspeed_mla import tokenspeed_mla_decode

    torch.backends.cuda.matmul.allow_tf32 = False
    workspace = torch.zeros(32 * 1024**2, dtype=torch.int8, device="cuda")
    # Narrow and multi-tile windows exercise the unsplit epilogue and reducer.
    # H96/Q3 also puts both the window boundary and output in a partial tile.
    case = _Case(2, 769, 96, 3)
    for dtype in ("fp8", "fp16", "bf16"):
        q, kv, tables, lengths = _make_inputs(case, dtype, True, "cuda")
        for window in (2, 256):
            for causal in (False, True):
                expected = torch.empty(
                    case.batch, case.q_len, case.heads, 512, device="cuda"
                )
                expected_lse = torch.empty(
                    case.batch, case.q_len, case.heads, device="cuda"
                )
                visible = torch.empty_like(expected_lse)
                for batch in range(case.batch):
                    length = int(lengths[batch])
                    keys = kv.float()[tables[batch].long()].reshape(-1, 576)
                    for row in range(case.q_len):
                        low = max(0, length - case.q_len - window + row)
                        high = length - case.q_len + row + 1 if causal else length
                        logits = q[batch, row].float() @ keys[low:high].T
                        logits *= 192**-0.5
                        expected[batch, row] = logits.softmax(-1) @ keys[low:high, :512]
                        expected_lse[batch, row] = logits.logsumexp(-1) / torch.log(
                            torch.tensor(2.0, device="cuda")
                        )
                        visible[batch, row] = high - low
                kwargs = dict(
                    query=q,
                    kv_cache=kv,
                    workspace_buffer=workspace,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    block_tables=tables,
                    seq_lens=lengths,
                    max_seq_len=case.kv_len,
                    softmax_scale=192**-0.5,
                    is_var_seq=True,
                    causal_mask=causal,
                    window_left=window,
                    enable_pdl=True,
                    return_lse=True,
                )
                for packed in (False, True):
                    result, lse = tokenspeed_mla_decode(
                        **kwargs, enable_packed_q=packed
                    )
                    _check_output(result, expected, dtype)
                    lse_tol = 0.05 if dtype == "fp8" else 0.002
                    torch.testing.assert_close(
                        lse, expected_lse, atol=lse_tol, rtol=0.001
                    )
                # A uniform query makes each window's exact key count observable.
                zero_kwargs = dict(kwargs, query=torch.zeros_like(q))
                for packed in (False, True):
                    _, lse = tokenspeed_mla_decode(
                        **zero_kwargs, enable_packed_q=packed
                    )
                    torch.testing.assert_close(
                        lse, visible.log2(), atol=2e-5, rtol=1e-5
                    )
                print(
                    f"Window verified: dtype={dtype} window={window} causal={causal}",
                    flush=True,
                )

    # Strided DCP keys use global positions for the causal upper boundary.
    # Check against an explicit reference rather than only against the old kernel.
    torch.manual_seed(19)
    length, world, heads, queries = 515, 2, 96, 3
    query = torch.randn(1, queries, heads, 576, device="cuda").to(torch.float8_e4m3fn)
    keys = torch.randn(length, 576, device="cuda").to(torch.float8_e4m3fn)
    for rank in range(world):
        local = keys[rank::world]
        pages = (local.shape[0] + 63) // 64
        cache = torch.zeros(pages, 64, 576, dtype=keys.dtype, device="cuda")
        cache.view(-1, 576)[: local.shape[0]] = local
        table = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
        positions = torch.arange(rank, length, world, device="cuda")
        expected = torch.empty(1, queries, heads, 512, device="cuda")
        visible = torch.empty(1, queries, heads, device="cuda")
        for row in range(queries):
            count = int((positions < length - queries + row + 1).sum())
            logits = query[0, row].float() @ local[:count].float().T
            expected[0, row] = (logits * 192**-0.5).softmax(-1) @ local[
                :count, :512
            ].float()
            visible[0, row] = count
        kwargs = dict(
            query=query,
            kv_cache=cache,
            workspace_buffer=workspace,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=table,
            seq_lens=torch.tensor([local.shape[0]], dtype=torch.int32, device="cuda"),
            max_seq_len=local.shape[0],
            softmax_scale=192**-0.5,
            is_var_seq=False,
            causal_mask=True,
            window_left=-1,
            cp_world=world,
            cp_rank=rank,
            causal_seqs=torch.tensor([length], dtype=torch.int32, device="cuda"),
            return_lse=True,
        )
        for packed in (False, True):
            output, _ = tokenspeed_mla_decode(**kwargs, enable_packed_q=packed)
            _check_output(output, expected, "fp8")
            _, lse = tokenspeed_mla_decode(
                **dict(kwargs, query=torch.zeros_like(query)),
                enable_packed_q=packed,
            )
            torch.testing.assert_close(lse, visible.log2(), atol=2e-5, rtol=1e-5)
        print(f"DCP verified: rank={rank} world={world}", flush=True)


def _check_reducer_variants():
    import tokenspeed_mla.mla_decode as decode

    torch.backends.cuda.matmul.allow_tf32 = False
    original_compile = decode._get_compiled_mla_kernel
    workspace = torch.zeros(256 * 1024**2, dtype=torch.int8, device="cuda")
    for case, capacity in [(_Case(1, 8192, 12, 4), 64), (_Case(1, 8192, 96, 1), 32)]:
        q, kv, tables, lengths = _make_inputs(case, "fp8", False, "cuda")
        q.zero_()  # Uniform attention has a closed-form base-2 LSE.
        expected_out = _reference_mla(q, kv, tables, lengths, [0])
        visible = (
            case.kv_len - case.q_len + torch.arange(1, case.q_len + 1, device="cuda")
        )
        expected_lse = (
            visible.float().log2()[None, :, None].expand(1, case.q_len, case.heads)
        )
        for pdl in (False, True):
            reference_out = reference_lse = None
            for bands, max_splits in [
                (1, 256),
                (1, capacity),
                (2, capacity),
                (4, capacity),
            ]:
                # Exercise the original full-capacity reducer and all new
                # topologies through the same wrapper/cache in one process.
                def compile_variant(*args, **kwargs):
                    kwargs.update(reducer_d_tiles=bands, reducer_max_splits=max_splits)
                    return original_compile(*args, **kwargs)

                decode._get_compiled_mla_kernel = compile_variant
                output, lse = decode.tokenspeed_mla_decode(
                    query=q,
                    kv_cache=kv,
                    workspace_buffer=workspace,
                    kv_lora_rank=512,
                    qk_rope_head_dim=64,
                    block_tables=tables,
                    seq_lens=lengths,
                    max_seq_len=case.kv_len,
                    softmax_scale=192**-0.5,
                    return_lse=True,
                    is_var_seq=False,
                    enable_pdl=pdl,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=1e-5)
                torch.testing.assert_close(
                    output.float(), expected_out, atol=0.001, rtol=0.05
                )
                if reference_out is None:
                    reference_out, reference_lse = output.clone(), lse.clone()
                else:
                    torch.testing.assert_close(output, reference_out, atol=0, rtol=0)
                    torch.testing.assert_close(lse, reference_lse, atol=0, rtol=0)


def _gpu_worker(check, arguments, send):
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        check(*arguments)
        send.send(None)
    except Exception:
        send.send(traceback.format_exc())
    finally:
        send.close()


def _run_gpu_check(check, arguments, timeout):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("Requires Blackwell SM100/SM103")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_gpu_worker, args=(check, arguments, send))
    process.start()
    send.close()
    try:
        assert receive.poll(timeout), f"{check.__name__} timed out after {timeout}s"
        assert (error := receive.recv()) is None, error
    finally:
        receive.close()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()


class TestGPU:
    @pytest.mark.parametrize(
        "case,variable_kv",
        [
            (_Case(1, 1024, heads, q_len), False)
            for heads, q_len in itertools.product((6, 12, 24, 48, 96, 128), (1, 4, 8))
        ]
        + [
            (_Case(4, 1024, 12, 8), False),
            (_Case(1, 1024, 128, 2), False),
            (_Case(10, 385, 96, 8), True),
            (_Case(128, 384, 96, 1), False),
            # Exercise all reducer bands and both static capacities (32 and 64).
            (_Case(1, 16384, 48, 1), False),
            (_Case(1, 16384, 96, 1), False),
            (_Case(1, 16512, 6, 1), False),
            # Normalizing to one split must take the workspace-free kernel path.
            (_Case(1, 128, 96, 1), False),
        ],
        ids=lambda value: value.name if isinstance(value, _Case) else None,
    )
    def test_fp8_decode_accuracy_and_cuda_graph(self, case, variable_kv):
        _run_gpu_check(_check_fp8_gpu, (case, variable_kv), 180)

    def test_packed_q_outputs_lse_tails_and_legacy_paths(self):
        _run_gpu_check(_check_packed_gpu, (), 600)

    def test_packed_q_preserves_sliding_window_and_dcp_masks(self):
        _run_gpu_check(_check_packed_masks_gpu, (), 600)

    def test_reducer_bands_preserve_output_lse_and_pdl(self):
        _run_gpu_check(_check_reducer_variants, (), 240)
