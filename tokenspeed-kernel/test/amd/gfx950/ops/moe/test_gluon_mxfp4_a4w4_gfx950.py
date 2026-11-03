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

import pytest
import torch
from tokenspeed_kernel.numerics.torch.activation import eager_silu_and_mul
from tokenspeed_kernel.numerics.torch.moe import torch_moe_experts
from tokenspeed_kernel.numerics.torch.quantization import (
    dequantize_mxfp4,
    fake_quantize_mxfp4_even,
)
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the Gluon A4W4 MXFP4 MoE tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.moe import moe_process_weights  # noqa: E402
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import mxfp4_gemm  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (  # noqa: E402
    preprocess_n16_mxfp4_weights,
)

# A 640-wide expert sharded over TP2 gives I=320: its 160-byte down-projection
# rows end in a half MFMA K chunk.
EXPERTS = 64
HIDDEN = 1024
INTER = 320
TOP_K = 8


def _weights(seed: int = 0) -> torch.nn.Module:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def codes(*shape):
        return torch.randint(
            0, 256, shape, device="cuda", dtype=torch.uint8, generator=g
        )

    def scales(*shape):
        return torch.randint(
            118, 124, shape, device="cuda", dtype=torch.uint8, generator=g
        )

    w = torch.nn.Module()
    w.w13_weight = torch.nn.Parameter(
        codes(EXPERTS, 2 * INTER, HIDDEN // 2), requires_grad=False
    )
    w.w13_weight_scale = torch.nn.Parameter(
        scales(EXPERTS, 2 * INTER, HIDDEN // 32), requires_grad=False
    )
    w.w2_weight = torch.nn.Parameter(
        codes(EXPERTS, HIDDEN, INTER // 2), requires_grad=False
    )
    w.w2_weight_scale = torch.nn.Parameter(
        scales(EXPERTS, HIDDEN, INTER // 32), requires_grad=False
    )
    w.top_k = TOP_K
    w.w13_input_layout = "concatenated"
    return w


def _plan(solution: str | None) -> dict:
    return moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ispp=INTER,
        internal_activation_dtype="mxfp4",
        solution=solution,
        hidden=HIDDEN,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=False,
        combine_order="rank",
        features=frozenset(),
    )


def _routes(m: int, seed: int = 1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    ids = torch.rand(m, EXPERTS, device="cuda", generator=g).topk(TOP_K, dim=-1)
    weights = torch.softmax(torch.randn(m, TOP_K, device="cuda", generator=g), dim=-1)
    return x, weights, ids.indices.to(torch.int32)


def _torch_reference(x, w, topk_weights, topk_ids):
    """FP32-GEMM reference of the dynamic MXFP4 SiLU MoE semantics."""
    return torch_moe_experts(
        x,
        dequantize_mxfp4(w.w13_weight, w.w13_weight_scale),
        dequantize_mxfp4(w.w2_weight, w.w2_weight_scale),
        topk_ids,
        topk_weights,
        # Gate, up and silu(gate) each round to BF16.
        activation=lambda gate_up: eager_silu_and_mul(gate_up.bfloat16()),
        w13_bias=None,
        w2_bias=None,
        quantize_input=fake_quantize_mxfp4_even,
        quantize_intermediate=fake_quantize_mxfp4_even,
        round_expert_output=torch.Tensor.bfloat16,
        first_expert=0,
    )


# 1-64 rows run the decode blocks; 300 and 1100 the LDS-staged larger blocks.
@pytest.mark.parametrize("m", [1, 5, 64, 300, 1100])
def test_matches_reference(m: int) -> None:
    w = _weights()
    reference_w = _weights()
    plan = _plan(None)
    moe_process_weights(plan, w)
    x, topk_weights, topk_ids = _routes(m)

    actual = moe_apply(
        plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
    ).float()
    expected = _torch_reference(x, reference_w, topk_weights, topk_ids)
    # FP32 accumulation order may flip a BF16 rounding before requantization.
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)

    # The portable Triton path implements the same roundings.
    triton_w = _weights()
    triton_plan = _plan("triton")
    moe_process_weights(triton_plan, triton_w)
    portable = moe_apply(
        triton_plan, x, triton_w, None, topk_weights=topk_weights, topk_ids=topk_ids
    ).float()
    torch.testing.assert_close(actual, portable, atol=3e-2, rtol=2e-2)
    assert (actual == portable).float().mean().item() > 0.99


# Decode route sets whose padded expert blocks span one to several 256-row
# padding stores of the fused sort.
@pytest.mark.parametrize("m", [1, 16, 64, 128])
def test_fused_sort_matches_the_standalone_sort(m: int, monkeypatch) -> None:
    w = _weights()
    moe_process_weights(_plan(None), w)
    x, topk_weights, topk_ids = _routes(m, seed=m)

    def run():
        return mxfp4_gemm.launch_gluon_mxfp4_a4w4_moe_apply_gfx950(
            x,
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
            topk_weights,
            topk_ids,
        )

    fused = run()
    # Every route set takes gluon_moe_sorting instead.
    monkeypatch.setattr(mxfp4_gemm, "_FUSED_SORT_MAX_ROUTES", 0)
    # Routes are computed per row and combined in slot order, so the order of
    # routes within an expert block does not change the bits.
    torch.testing.assert_close(fused, run(), atol=0, rtol=0)


def test_row_strided_activations_match_contiguous() -> None:
    w = _weights()
    plan = _plan(None)
    moe_process_weights(plan, w)
    x, topk_weights, topk_ids = _routes(5)
    # A row slice of a wider buffer is consumed in place.
    wide = torch.zeros((5, HIDDEN + 512), device="cuda", dtype=torch.bfloat16)
    wide[:, :HIDDEN] = x
    strided = moe_apply(
        plan, wide[:, :HIDDEN], w, None, topk_weights=topk_weights, topk_ids=topk_ids
    )
    dense = moe_apply(plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids)
    torch.testing.assert_close(strided, dense, atol=0, rtol=0)


def test_out_of_range_expert_contributes_zero() -> None:
    w = _weights()
    moe_process_weights(_plan(None), w)
    x, topk_weights, topk_ids = _routes(4)
    # int64 ids are read directly.
    dropped = topk_ids.clone().long()
    dropped[:, -1] = EXPERTS
    actual = mxfp4_gemm.launch_gluon_mxfp4_a4w4_moe_apply_gfx950(
        x,
        w.w13_weight,
        w.w13_weight_scale,
        w.w2_weight,
        w.w2_weight_scale,
        topk_weights,
        dropped,
    )
    kept_weights = topk_weights.clone()
    kept_weights[:, -1] = 0
    expected = mxfp4_gemm.launch_gluon_mxfp4_a4w4_moe_apply_gfx950(
        x,
        w.w13_weight,
        w.w13_weight_scale,
        w.w2_weight,
        w.w2_weight_scale,
        kept_weights,
        topk_ids,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_n16_bank_loader_round_trips_half_chunk_rows() -> None:
    """I = 320 ends each W2 row in a half 64-byte chunk and a partial scale
    cell; the N16 loader still sees and updates the linear expert."""
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    w = _weights()
    linear = {name: getattr(w, name).detach().clone() for name in names}
    seen = {}

    def loader(param, loaded_weight, shard_id, local_expert_id):
        seen[shard_id] = param.data[local_expert_id].clone()
        param.data[local_expert_id].copy_(loaded_weight)

    for name in names:
        getattr(w, name).weight_loader = loader
    preprocess_n16_mxfp4_weights(w)
    assert w.w2_weight.shape == (EXPERTS, HIDDEN // 16, 8 * INTER)
    with pytest.raises(ValueError, match="already prepared"):
        preprocess_n16_mxfp4_weights(w)

    for name in names:
        param = getattr(w, name)
        update = torch.randint_like(linear[name][3], 0, 256)
        param.weight_loader(param, update, shard_id="update", local_expert_id=3)
        torch.testing.assert_close(seen["update"], linear[name][3], atol=0, rtol=0)
        param.weight_loader(param, update, shard_id="read", local_expert_id=3)
        torch.testing.assert_close(seen["read"], update, atol=0, rtol=0)
        param.weight_loader(param, linear[name][2], shard_id="read", local_expert_id=2)
        torch.testing.assert_close(seen["read"], linear[name][2], atol=0, rtol=0)


def test_batch_sizes_do_not_recompile() -> None:
    w = _weights()
    moe_process_weights(_plan(None), w)
    kernels = (
        mxfp4_gemm.gluon_mxfp4_a4w4_moe_apply_prep_gfx950,
        mxfp4_gemm.gluon_mxfp4_a4w4_moe_apply_stage1_gfx950,
        mxfp4_gemm.gluon_mxfp4_a4w4_moe_apply_stage2_gfx950,
        mxfp4_gemm.gluon_mxfp4_a4w4_moe_apply_combine_gfx950,
    )

    def run(m: int) -> None:
        x, topk_weights, topk_ids = _routes(m, seed=m)
        mxfp4_gemm.launch_gluon_mxfp4_a4w4_moe_apply_gfx950(
            x,
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
            topk_weights,
            topk_ids,
        )

    # Every decode width (routes <= 4 per expert, sorted in the prep launch
    # with a power-of-two route bound) shares one specialization, and so does
    # every width of the 64-row prefill blocks; the row count is a runtime
    # argument in every kernel.
    run(1)
    run(400)
    with assert_no_triton_compile(*kernels):
        for m in (2, 3, 7, 16, 17, 32, 401, 517, 1000):
            run(m)
    torch.cuda.synchronize()


def test_cuda_graph_replay_matches_eager() -> None:
    w = _weights()
    moe_process_weights(_plan(None), w)
    x, topk_weights, topk_ids = _routes(8)

    def run():
        return mxfp4_gemm.launch_gluon_mxfp4_a4w4_moe_apply_gfx950(
            x,
            w.w13_weight,
            w.w13_weight_scale,
            w.w2_weight,
            w.w2_weight_scale,
            topk_weights,
            topk_ids,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    new_x, new_weights, new_ids = _routes(8, seed=7)
    x.copy_(new_x)
    topk_weights.copy_(new_weights)
    topk_ids.copy_(new_ids)
    graph.replay()
    torch.testing.assert_close(out, run(), atol=0, rtol=0)
