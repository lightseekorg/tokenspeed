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

"""Exercise GDN dependency visibility and capture-time PDL selection."""

from __future__ import annotations

import os
import sys

import pytest
import torch
import triton
import triton.language as tl
from tokenspeed_kernel.ops.attention.gdn import flashinfer as flashinfer_gdn
from tokenspeed_kernel.ops.attention.gdn import (
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
)
from tokenspeed_kernel.ops.attention.gdn.triton import fused_qkv_split_gdn_prefill
from tokenspeed_kernel.platform import current_platform, pdl_enabled

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_update,
)
from tokenspeed.runtime.layers.attention.linear.layernorm_gated import rmsnorm_fn
from tokenspeed.runtime.models.qwen3_5 import fused_qkvzba_split_reshape_cat_contiguous

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper_plus, reason="PDL requires NVIDIA SM90+"
)


@triton.jit
def _delayed_projection(source, output, N: tl.constexpr, BLOCK: tl.constexpr):
    # Launch the consumer while these CTAs are still busy. Missing waits can
    # otherwise hide in tests whose tiny producers finish before the next launch.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    now = start
    while now - start < 100000:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %clock64;",
            constraints="=l",
            args=[],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        output + offsets, tl.load(source + offsets, offsets < N, other=0), offsets < N
    )


def _graph_kernel_edges(graph):
    from cuda.bindings import driver as cu

    def checked(result):
        assert result[0] == cu.CUresult.CUDA_SUCCESS, result
        return result[1:]

    raw = cu.CUgraph(graph.raw_cuda_graph())
    _, count = checked(cu.cuGraphGetNodes(raw, 0))
    nodes, _ = checked(cu.cuGraphGetNodes(raw, count))
    names = {}
    for node in nodes:
        (kind,) = checked(cu.cuGraphNodeGetType(node))
        if kind == cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            (params,) = checked(cu.cuGraphKernelNodeGetParams(node))
            (name,) = checked(cu.cuFuncGetName(params.func))
            names[int(node)] = name.decode()
    _, _, _, count = checked(cu.cuGraphGetEdges(raw, 0))
    sources, targets, edges, _ = checked(cu.cuGraphGetEdges(raw, count))
    return names, [
        (names.get(int(source), ""), names.get(int(target), ""), int(edge.type))
        for source, target, edge in zip(sources, targets, edges)
    ]


@pytest.fixture
def restore_pdl():
    previous = pdl_enabled()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        pdl_enabled(previous)


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("steps", [1, 4])
@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_gdn_chain_pdl_toggle(batch, steps, solution, state_dtype, restore_pdl):
    if solution == "flashinfer" and not flashinfer_gdn.is_decode_available():
        pytest.skip("FlashInfer GDN unavailable")
    torch.manual_seed(31)
    # TP4 Qwen3.8 uses 4 QK / 12 V heads; MTP with 3 draft steps verifies T=4.
    heads, value_heads, dim = 4, 12, 128
    rows = batch * steps
    qkv_width = (2 * heads + value_heads) * dim
    qkvz_width = qkv_width + value_heads * dim
    source = torch.randn(
        rows, qkvz_width + 2 * value_heads, device="cuda", dtype=torch.bfloat16
    )
    projection = torch.empty_like(source)
    pool_size = 1 + batch + rows
    conv = torch.randn(pool_size, qkv_width, 3, device="cuda", dtype=source.dtype)
    state = torch.randn(
        pool_size, value_heads, dim, dim, device="cuda", dtype=state_dtype
    )
    weights = torch.randn(qkv_width, 4, device="cuda", dtype=source.dtype)
    norm_weight = torch.randn(dim, device="cuda", dtype=source.dtype)
    A_log = torch.randn(value_heads, device="cuda")
    dt_bias = torch.randn(value_heads, device="cuda")
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    writes = torch.arange(batch + 1, pool_size, device="cuda", dtype=torch.int32).view(
        batch, steps
    )

    def forward():
        _delayed_projection[(triton.cdiv(source.numel(), 1024),)](
            source,
            projection,
            N=source.numel(),
            BLOCK=1024,
            launch_pdl=True,
        )
        qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
            projection[:, :qkvz_width],
            projection[:, qkvz_width:],
            heads,
            value_heads,
            dim,
            dim,
        )
        causal_conv1d_update(
            qkv.view(batch, steps, qkv_width).transpose(1, 2),
            conv,
            weights,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            conv_state_indices=reads,
            num_accepted_tokens=None,
            intermediate_conv_window=None,
            output_state_indices=writes,
            pad_slot_id=-1,
            validate_data=False,
        )
        if steps == 1:
            # Actual decode uses packed, strided Q/K/V views.
            q, k, v = qkv.split([heads * dim, heads * dim, value_heads * dim], dim=-1)
        else:
            q, k, v = fused_qkv_split_gdn_prefill(
                qkv,
                heads,
                heads,
                value_heads,
                dim,
                dim,
                dim,
                fuse_l2norm=False,
                replay=None,
            )
        kwargs = dict(
            q=q.view(batch, steps, heads, dim),
            k=k.view(batch, steps, heads, dim),
            v=v.view(batch, steps, value_heads, dim),
            a=a.view(batch, steps, value_heads),
            b=b.view(batch, steps, value_heads),
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=state,
            initial_state_indices=reads,
            scale=dim**-0.5,
            use_qk_l2norm=True,
            solution=solution,
            override=None,
        )
        if steps == 1:
            out = gdn_decode_step(**kwargs, output_state_indices=writes[:, 0])
        else:
            out = gdn_decode_mtp(
                **kwargs,
                disable_state_update=False,
                output_state_indices=writes,
                intermediate_states_buffer=None,
            )
        return rmsnorm_fn(
            out.reshape(-1, dim),
            norm_weight,
            z=z.reshape(-1, dim),
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            sigmoid_gate=False,
        )

    # Return to false after true to exercise both upstream and private caches.
    for enabled in (False, True, False):
        pdl_enabled(enabled)
        forward()
        forward()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            actual = forward()
        names, edges = _graph_kernel_edges(graph)
        stages = ("fused_qkvzba", "causal_conv1d", "gdn", "rms_norm")
        for stage in stages:
            assert any(stage in name for name in names.values()), names
        if steps > 1:
            assert any("fused_qkv_split" in name for name in names.values()), names
        for _, target, edge_type in edges:
            if any(stage in target for stage in stages) or "fused_qkv_split" in target:
                assert edge_type == int(enabled), (enabled, target, edges)
        # Inputs change after capture; replay must retain capture-time PDL even
        # when the global switch changes in between captures.
        for _ in range(2):
            source.normal_()
            pdl_enabled(False)
            expected = forward().clone()
            expected_conv, expected_state = conv.clone(), state.clone()
            pdl_enabled(not enabled)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_prefill_pdl_toggle(solution, restore_pdl):
    if solution == "flashinfer" and not flashinfer_gdn.is_supported(
        128, torch.bfloat16, 4, 8
    ):
        pytest.skip("FlashInfer SM100 prefill unavailable")
    torch.manual_seed(41)
    q, k = [
        torch.randn(1, 130, 4, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    v = torch.randn(1, 130, 8, 128, device="cuda", dtype=q.dtype)
    gate = -torch.rand(1, 130, 8, device="cuda")
    beta = torch.rand(1, 130, 8, device="cuda", dtype=q.dtype)
    state = torch.randn(2, 8, 128, 128, device="cuda")
    cu = torch.tensor([0, 65, 130], device="cuda", dtype=torch.int32)

    def forward():
        return gdn_chunk_prefill(
            q,
            k,
            v,
            gate,
            beta,
            scale=128**-0.5,
            initial_state=state,
            cu_seqlens=cu,
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
            solution=solution,
            override=None,
        )

    pdl_enabled(False)
    reference = forward()
    for enabled in (True, False):
        pdl_enabled(enabled)
        result = forward()
        torch.testing.assert_close(result.out, reference.out, rtol=0, atol=0)
        torch.testing.assert_close(
            result.final_state, reference.final_state, rtol=0, atol=0
        )
        # Warmed varlen metadata is pointer-stable for capture.
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            result = forward()
        names, edges = _graph_kernel_edges(graph)
        custom = (
            "l2norm",
            "chunk_",
            "solve_tril",
            "merge_",
            "recompute_w_u",
            "_gdn_pdl_kernel",
            "kernel_cutlass",
        )
        checked = 0
        for _, target, edge_type in edges:
            if any(stage in target for stage in custom):
                assert edge_type == int(enabled), (target, edges)
                checked += 1
        assert checked >= 2, names
        q.normal_()
        k.normal_()
        v.normal_()
        pdl_enabled(False)
        reference = forward()
        graph.replay()
        torch.testing.assert_close(result.out, reference.out, rtol=0, atol=0)
        torch.testing.assert_close(
            result.final_state, reference.final_state, rtol=0, atol=0
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
