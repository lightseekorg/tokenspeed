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
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.thirdparty.cuda import moe_finalize_fuse_shared

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia
    or current_platform().arch_version < ArchVersion(9, 0),
    reason="moe_finalize_fuse_shared needs an NVIDIA SM>=90 GPU",
)


def _make_inputs(num_tokens, hidden, top_k, num_experts, ew_dtype, num_shared):
    """Random permuted gemm2 rows + a valid expanded->permuted map."""
    torch.manual_seed(num_tokens * 31 + top_k)
    total_padded = num_tokens * top_k + 16
    gemm2_out = torch.randn(total_padded, hidden, dtype=torch.bfloat16, device="cuda")
    perm = torch.randperm(total_padded, device="cuda", dtype=torch.int32)
    expanded_idx = perm[: num_tokens * top_k].contiguous()
    # Sprinkle dropped slots (-1), as EP produces for remote experts.
    drop = torch.rand(num_tokens * top_k, device="cuda") < 0.1
    expanded_idx = torch.where(drop, torch.full_like(expanded_idx, -1), expanded_idx)
    weights = torch.rand(num_tokens, top_k + num_shared, dtype=ew_dtype, device="cuda")
    return gemm2_out, expanded_idx, weights


def _reference(gemm2_out, expanded_idx, weights, shared, top_k):
    num_tokens = weights.shape[0]
    idx = expanded_idx.view(num_tokens, top_k).long()
    valid = idx >= 0
    rows = gemm2_out.float()[idx.clamp(min=0)]  # [T, K, H]
    rows = rows * valid.unsqueeze(-1)
    out = (rows * weights.float()[:, :top_k, None]).sum(dim=1)
    if shared is not None:
        if shared.dim() == 2:
            out = out + shared.float()
        else:
            gammas = weights.float()[:, top_k:]  # [T, S]
            out = out + torch.einsum("sth,ts->th", shared.float(), gammas)
    return out.to(torch.bfloat16)


# num_tokens 4 exercises the general kernel; 4096 the vectorized one
# (dispatch flips at numBlocksX * numBlocksY >= 1184).
@pytest.mark.parametrize("num_tokens", [4, 4096])
@pytest.mark.parametrize("ew_dtype", [torch.float32, torch.bfloat16])
def test_finalize_verbatim_shared_add(num_tokens, ew_dtype):
    """numShared == 0 regression: pre-combined [T, H] residual, added verbatim."""
    hidden, top_k = 512, 8
    gemm2_out, expanded_idx, weights = _make_inputs(
        num_tokens, hidden, top_k, 32, ew_dtype, num_shared=0
    )
    shared = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")

    out = moe_finalize_fuse_shared(gemm2_out, expanded_idx, weights, shared, top_k)
    ref = _reference(gemm2_out, expanded_idx, weights, shared, top_k)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [4, 4096])
@pytest.mark.parametrize("ew_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_shared", [1, 2])
@pytest.mark.parametrize("top_k", [6, 8])
def test_finalize_weighted_shared_sink(num_tokens, ew_dtype, num_shared, top_k):
    """[T, K+S] weights: tail columns weight the [S, T, H] shared outputs."""
    hidden = 512
    gemm2_out, expanded_idx, weights = _make_inputs(
        num_tokens, hidden, top_k, 32, ew_dtype, num_shared=num_shared
    )
    shared = torch.randn(
        num_shared, num_tokens, hidden, dtype=torch.bfloat16, device="cuda"
    )

    out = moe_finalize_fuse_shared(gemm2_out, expanded_idx, weights, shared, top_k)
    ref = _reference(gemm2_out, expanded_idx, weights, shared, top_k)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_finalize_no_shared():
    """shared_output=None still works (plain weighted finalize)."""
    gemm2_out, expanded_idx, weights = _make_inputs(64, 512, 8, 32, torch.float32, 0)
    out = moe_finalize_fuse_shared(gemm2_out, expanded_idx, weights, None, 8)
    ref = _reference(gemm2_out, expanded_idx, weights, None, 8)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not current_platform().is_nvidia
    or not current_platform().is_blackwell
    or current_platform().arch_version != ArchVersion(10, 0),
    reason="routed trtllm MoE kernels need SM100",
)
def test_routed_deferred_finalize_matches_finalized():
    """trtllm routed unquant MoE: do_finalize=False + our fused finalize must
    reproduce do_finalize=True. Also proves the deferred gemm2 rows are
    un-weighted (the finalize applies the only weighting)."""
    import tokenspeed_kernel

    torch.manual_seed(0)
    num_experts, top_k, hidden, inter = 16, 6, 256, 256
    num_tokens, num_shared = 33, 2

    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ispp=inter,
        internal_activation_dtype="input",
        solution="flashinfer_trtllm",
    )
    assert plan["apply_kernel_name"] == "flashinfer_trtllm_unquant_routed_moe_apply"
    assert plan["supports_deferred_finalize"] is True

    w = torch.nn.Module()
    w.w13_weight = torch.nn.Parameter(
        torch.randn(num_experts, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda")
        * 0.05,
        requires_grad=False,
    )
    w.w2_weight = torch.nn.Parameter(
        torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16, device="cuda")
        * 0.05,
        requires_grad=False,
    )
    w.num_experts = num_experts
    w.top_k = top_k
    w.intermediate_size = inter
    w.tp_size = 1
    w.ep_rank = 0
    w.num_local_experts = num_experts
    tokenspeed_kernel.moe_process_weights(plan, w)

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
    router_logits = torch.randn(num_tokens, num_experts, device="cuda")
    full_weights = torch.rand(
        num_tokens, top_k + num_shared, dtype=torch.float32, device="cuda"
    )
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)

    def apply(do_finalize):
        return tokenspeed_kernel.moe_apply(
            plan,
            x,
            w,
            router_logits,
            topk_weights=full_weights[:, :top_k],
            topk_ids=topk_ids,
            do_finalize=do_finalize,
        )

    finalized = apply(do_finalize=True)
    gemm2_out, _, expanded_idx = apply(do_finalize=False)

    shared = torch.randn(
        num_shared, num_tokens, hidden, dtype=torch.bfloat16, device="cuda"
    )
    fused = moe_finalize_fuse_shared(
        gemm2_out, expanded_idx, full_weights, shared, top_k
    )
    gammas = full_weights[:, top_k:]
    expected = finalized.float() + torch.einsum(
        "sth,ts->th", shared.float(), gammas.float()
    )
    # The finalized path applies bf16 routed weights in-kernel while ours
    # applies f32; tolerate the rounding delta.
    torch.testing.assert_close(fused.float(), expected, atol=5e-2, rtol=5e-2)
