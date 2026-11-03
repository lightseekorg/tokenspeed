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

"""Rubin GDN numerics, arena padding and graph replay through the registry."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tokenspeed_kernel.benchmark.generators.gdn import prepare_gdn_decode_step
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
)
from tokenspeed_kernel.numerics.torch.attention.gdn import (
    gdn_gate,
    gdn_l2norm,
    gdn_l2norm_clamped,
    torch_gdn_recurrence,
)
from tokenspeed_kernel.ops.attention.gdn import (
    GdnStateCheckpoints,
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
    gdn_decode_supports,
)
from tokenspeed_kernel.ops.attention.gdn.flashinfer import (
    is_rubin_decode_available as is_available,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.selection import NoKernelFoundError

pytestmark = pytest.mark.skipif(
    not is_available(), reason="SM107 and FlashInfer PR6604 are required"
)


def test_rubin_decode_benchmark_prepares_and_invokes():
    request = BenchmarkRequest(
        family="attention",
        mode="gdn_decode_step",
        parameters={
            "model_profile": "qwen38_flash_tp2",
            "batch": 4,
            "dtype": "bfloat16",
            "state_dtype": "bfloat16",
        },
        solution="flashinfer_rubin",
        registration=None,
        cold_cache=True,
        seed=6604,
    )
    candidate = prepare_gdn_decode_step(request, current_platform())
    reference = prepare_gdn_decode_step(
        replace(request, solution="triton"), current_platform()
    )
    # Both preparations generate identical inputs. Repeated calls also read
    # the state written by the preceding invocation.
    for _ in range(2):
        actual = candidate.invocation.invoke()
        expected = reference.invocation.invoke()
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
    assert candidate.parameters["state_dtype"] == "bfloat16"


def test_rubin_decode_benchmark_rejects_fp32_before_invocation():
    request = BenchmarkRequest(
        family="attention",
        mode="gdn_decode_step",
        parameters={
            "model_profile": "qwen38_flash_tp2",
            "batch": 4,
            "dtype": "bfloat16",
            "state_dtype": "float32",
        },
        solution="flashinfer_rubin",
        registration=None,
        cold_cache=True,
        seed=6604,
    )
    with pytest.raises(BenchmarkCaseError) as error:
        prepare_gdn_decode_step(request, current_platform())
    assert error.value.status == BenchmarkStatus.NOT_APPLICABLE


@pytest.fixture(params=[False, True])
def pdl(request):
    previous = pdl_enabled()
    pdl_enabled(request.param)
    try:
        yield request.param
    finally:
        torch.cuda.synchronize()
        pdl_enabled(previous)


def _inputs(batch: int, heads: int, value_heads: int):
    torch.manual_seed(6604)
    dim = 128
    # Views into a wider projected row, including row-strided a/b.
    widths = [heads * dim, heads * dim, value_heads * dim, value_heads, value_heads]
    projection = torch.randn(
        batch, sum(widths) + 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, v, a, b = projection[:, : sum(widths)].split(widths, dim=-1)
    q, k = (x.view(batch, 1, heads, dim) for x in (q, k))
    v = v.view(batch, 1, value_heads, dim)
    a, b = (x.view(batch, 1, value_heads) for x in (a, b))
    # Interleaved layer views exercise the cache arena's noncontiguous page stride.
    storage = (
        torch.randn(
            2 * batch + 4,
            2,
            value_heads,
            dim,
            dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    pool = storage[:, 0]
    pool[0].zero_()
    parameters = dict(
        A_log=torch.randn(value_heads, device="cuda"),
        dt_bias=torch.randn(value_heads, device="cuda"),
        a=a,
        b=b,
        scale=dim**-0.5,
        use_qk_l2norm=True,
    )
    return (q, k, v), parameters, storage, pool


def _reference(qkv, parameters, state):
    q, k, v = qkv
    q, k = (gdn_l2norm(x.double(), 1e-6) for x in (q, k))
    gates = gdn_gate(
        parameters["a"].double(),
        parameters["dt_bias"].double(),
        parameters["A_log"].double(),
    )
    return torch_gdn_recurrence(
        q,
        k,
        v.double(),
        gates,
        parameters["b"].double().sigmoid(),
        state,
        scale=parameters["scale"],
        parents=None,
        compute_dtype=torch.float64,
    )


def test_named_decode_solution_rejects_unsupported_state():
    assert not gdn_decode_supports(
        "gdn_decode_step",
        torch.bfloat16,
        head_dim=128,
        value_head_dim=128,
        state_dtype=torch.float32,
        trait="strided_qkv",
        solution="flashinfer_rubin",
    )
    qkv, parameters, _, pool = _inputs(1, 4, 12)
    rows = torch.ones(1, device="cuda", dtype=torch.int32)
    with pytest.raises(NoKernelFoundError):
        gdn_decode_step(
            *qkv,
            **parameters,
            initial_state=pool.float(),
            initial_state_indices=rows,
            output_state_indices=rows,
            solution="flashinfer_rubin",
        )


@pytest.mark.parametrize(
    "batch,heads,value_heads", [(1, 16, 48), (8, 16, 48), (32, 4, 12)]
)
def test_decode_numerics_and_split_pages(batch, heads, value_heads, pdl):
    qkv, params, storage, pool = _inputs(batch, heads, value_heads)
    before = storage.clone()
    read = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    write = read + batch
    if batch > 1:
        read[-1] = -1
        write[-2] = -1
    valid = (read >= 0) & (write >= 0)
    ref_params = {**params, "a": params["a"][valid], "b": params["b"][valid]}
    reference, states = _reference(
        tuple(x[valid] for x in qkv), ref_params, pool[read[valid].long()]
    )
    # Use automatic selection: runtime and its capability probes take this path.
    out = gdn_decode_step(
        *qkv,
        **params,
        initial_state=pool,
        initial_state_indices=read,
        output_state_indices=write,
    )
    torch.testing.assert_close(out[valid].double(), reference, atol=0.002, rtol=0.02)
    torch.testing.assert_close(
        pool[write[valid].long()].double(), states[:, 0], atol=0.002, rtol=0.02
    )
    unchanged = torch.ones(pool.shape[0], device="cuda", dtype=torch.bool)
    unchanged[write[valid].long()] = False
    torch.testing.assert_close(pool[unchanged], before[:, 0][unchanged], atol=0, rtol=0)
    torch.testing.assert_close(storage[:, 1], before[:, 1], atol=0, rtol=0)


def test_decode_graph_refresh_and_padding(pdl):
    qkv, params, storage, pool = _inputs(8, 4, 12)
    original = storage.clone()
    read = torch.arange(1, 9, device="cuda", dtype=torch.int32)
    write = read + 8

    def call():
        return gdn_decode_step(
            *qkv,
            **params,
            initial_state=pool,
            initial_state_indices=read,
            output_state_indices=write,
            solution="flashinfer_rubin",
        )

    call()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for live in (3, 0, 8):
        read.copy_(torch.arange(1, 9, device="cuda", dtype=torch.int32).flip(0))
        write.copy_(torch.arange(9, 17, device="cuda", dtype=torch.int32))
        read[live:] = -1
        write[live:] = -1
        qkv[0].mul_(0.75)
        qkv[2].add_(0.125)
        storage.copy_(original)
        expected = call().clone()
        expected_storage = storage.clone()
        storage.copy_(original)
        graph.replay()
        torch.testing.assert_close(captured[:live], expected[:live], atol=0, rtol=0)
        torch.testing.assert_close(storage, expected_storage, atol=0, rtol=0)
        assert torch.count_nonzero(pool[0]).item() == 0


def test_bf16_verify_auto_keeps_null_state(pdl):
    qkv, params, _, source_pool = _inputs(4, 4, 12)
    # The unchanged portable verify path writes a dense scratch pool.
    pool = source_pool.clone()
    read = torch.tensor([1, -1, 3, -1], device="cuda", dtype=torch.int32)
    before = pool.clone()
    out = gdn_decode_mtp(
        *(x.expand(-1, 2, -1, -1).contiguous() for x in qkv),
        **{
            **params,
            "a": params["a"].expand(-1, 2, -1).contiguous(),
            "b": params["b"].expand(-1, 2, -1).contiguous(),
        },
        initial_state=pool,
        initial_state_indices=read,
        disable_state_update=False,
        tree_ancestors=None,
    )
    assert torch.isfinite(out[[0, 2]]).all()
    untouched = [i for i in range(pool.shape[0]) if i not in (1, 3)]
    torch.testing.assert_close(pool[untouched], before[untouched], atol=0, rtol=0)


def test_prefill_varlen_checkpoints_and_graph(pdl):
    torch.manual_seed(107)
    heads, value_heads, dim, tokens = 4, 12, 128, 196
    q, k = [
        torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    v = (
        torch.randn(1, tokens, value_heads, dim, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    g = -torch.rand(1, tokens, value_heads, device="cuda") * 0.1
    beta = torch.rand_like(g).to(torch.bfloat16)
    initial = torch.randn(2, value_heads, dim, dim, device="cuda") * 0.1
    bounds = torch.tensor([0, 67, tokens], device="cuda", dtype=torch.int64)
    checkpoint_offsets = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int64)
    checkpoints = torch.empty(3, value_heads, dim, dim, device="cuda")
    spec = GdnStateCheckpoints(
        states=checkpoints, cu_starts=checkpoint_offsets, every_n_tokens=64
    )

    def call():
        return gdn_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=dim**-0.5,
            initial_state=initial,
            cu_seqlens=bounds,
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
            state_checkpoints=spec,
            solution="flashinfer",
        )

    call()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for split in (67, 128):
        bounds[1] = split
        checkpoint_offsets[1] = split // 64
        v.add_(0.01)
        initial.mul_(0.9)
        expected = call()
        eager_out, eager_state = expected.out.clone(), expected.final_state.clone()
        eager_checkpoints = checkpoints.clone()
        graph.replay()
        torch.testing.assert_close(captured.out, eager_out, atol=0, rtol=0)
        torch.testing.assert_close(captured.final_state, eager_state, atol=0, rtol=0)
        torch.testing.assert_close(checkpoints, eager_checkpoints, atol=0, rtol=0)
        ckpt_start = 0
        for seq, (start, end) in enumerate(((0, split), (split, tokens))):
            # Prefill's public contract normalizes into BF16 before the scan.
            ref_out, ref_states = torch_gdn_recurrence(
                gdn_l2norm_clamped(q[:, start:end], 1e-6),
                gdn_l2norm_clamped(k[:, start:end], 1e-6),
                v[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                initial[seq : seq + 1],
                scale=dim**-0.5,
                parents=None,
                compute_dtype=torch.float64,
            )
            torch.testing.assert_close(
                captured.out[:, start:end].double(), ref_out, atol=0.002, rtol=0.02
            )
            torch.testing.assert_close(
                captured.final_state[seq].double(),
                ref_states[0, -1],
                atol=0.002,
                rtol=0.02,
            )
            for offset in range(63, end - start, 64):
                torch.testing.assert_close(
                    checkpoints[ckpt_start].double(),
                    ref_states[0, offset],
                    atol=0.002,
                    rtol=0.02,
                )
                ckpt_start += 1
