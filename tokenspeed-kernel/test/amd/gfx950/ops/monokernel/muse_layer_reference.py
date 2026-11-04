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

"""The regular Muse Spark layer's math for the fused-layer tests, and one
fused launch on the builder's inputs (``muse_layer_builder``).

``regular_reference`` composes the regular path's steps on the same inputs:
today's decode kernel over the FP8 cache (with the forward's split schedule
on full layers), the gated o-norm, the o_proj all-reduce, the attention
boundary, the FP32 router with sigmoid top-8 (checked untied), the pooler
with its norm and MXFP4 quantization, today's direct-decode MoE kernels per
rank and their route sum, the routed norm, the post-expert projection, the
FFN boundary, and the next layer's QKV projection and ``gqa_prologue``.
Collectives run over the default process group.
"""

import os

import torch
import torch.distributed as dist
from muse_layer_builder import (
    EPS,
    HEAD_DIM,
    INTER,
    KV_HEADS,
    KV_WIDTH,
    POOLED,
    Q_HEADS,
    Q_WIDTH,
    TOPK,
    WIDTH,
    fresh_next_prologue,
)
from tokenspeed_kernel.numerics.torch.residual import (
    torch_post_norm_residual_pre_norm,
)
from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache
from tokenspeed_kernel.ops.attention.prologue import gqa_prologue
from tokenspeed_kernel.ops.layernorm.fused_norm import (
    weightless_rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage1 import (
    invoke_stage1_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage2 import (
    invoke_stage2_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _quantize_mxfp4_activation,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    empty_swizzled_cdna4_mxfp4_scale,
)

SWIGLU = dict(swiglu_alpha=1.0, swiglu_limit=0.0, swiglu_beta=0.0)
# 8th vs 9th biased routing score: far above the logit error of a reordered
# FP32 sum over 8192 products (~2**-20 relative).
ROUTING_GAP = 1e-3
# Seeds a test tries, in order, until the inputs have no routing tie.
SEEDS = range(16)


def failure_records(record_dir, world):
    """Every rank's mirrored failure record (site, logical workgroup, detail,
    epoch, label), one line per rank: a fail_stop trap ends the ranks with a
    queue error, and the record names the wait that failed."""
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent.mirror import (
        FailureMirror,
    )

    lines = []
    for rank in range(world):
        path = os.path.join(record_dir, f"rank{rank}.bin")
        record = FailureMirror.read(path) if os.path.exists(path) else "no file"
        lines.append(f"rank {rank}: {record}")
    return "\n".join(lines)


class RoutingTie(AssertionError):
    """The inputs' 8th and 9th routing scores are within ``ROUTING_GAP``:
    a correct kernel may pick either expert. Every rank raises it together
    (the scores follow the all-reduce), so a test can draw the next seed."""


def run_fused(fused, key, inputs, residual):
    """One fused launch that updates ``residual`` (rows of the step's
    residual) in place; the next layer's caches are fresh copies, so a launch
    can be repeated."""
    common = dict(positions=inputs["positions"], cache=inputs["cache"])
    if key.next_kind == "regular":
        normed, residual = fused.forward_last(
            inputs["prepared_query"], residual, next_norm=inputs["next_norm"], **common
        )
        return dict(normed=normed, residual=residual)
    target = fresh_next_prologue(inputs["next_prologue"])
    (q, gate), residual = fused.forward(
        inputs["prepared_query"], residual, next_prologue=target, **common
    )
    return dict(q=q.reshape(q.shape[0], -1), gate=gate, residual=residual,
                next_k=target.cache.k_cache, next_v=target.cache.v_cache)  # fmt: skip


def attention_reference(inputs):
    """Today's decode of the step's rows over the FP8 cache (the registered
    kernel the MHA backend calls, with the forward's split schedule on full
    layers): [rows, Q_WIDTH] BF16."""
    q, _ = inputs["prepared_query"]
    cache = inputs["cache"]
    rows = q.shape[0]
    out = mha_decode_with_kvcache(
        q=q.view(rows, Q_HEADS, HEAD_DIM), k_cache=cache.k_cache,
        v_cache=cache.v_cache, page_table=cache.page_table,
        cache_seqlens=cache.seq_lens, max_seqlen_k=cache.max_seqlen_k,
        max_seqlen_q=WIDTH, window_left=cache.window_left,
        schedule=cache.split_schedule,
    )  # fmt: skip
    return out.reshape(rows, Q_WIDTH)


def _direct_decode(xq, xs, experts, ids):
    """Today's MXFP4 decode stages on this rank's expert slices."""
    rows = ids.shape[0]
    w13 = _extract_gluon_raw_w(experts.w13_weight_triton_tensor)
    w13_scale = _extract_gluon_raw_s(experts.w13_precision_config.b_mx_scale)
    w2 = _extract_gluon_raw_w(experts.w2_weight_triton_tensor)
    w2_scale = _extract_gluon_raw_s(experts.w2_precision_config.b_mx_scale)
    mids = torch.empty((rows * TOPK, INTER // 2), dtype=torch.uint8, device="cuda")
    mids_scale = empty_swizzled_cdna4_mxfp4_scale(
        rows * TOPK, INTER // 32, device=torch.device("cuda")
    )
    invoke_stage1_mxfp4_mfma_decode_gluon(
        xq, xs, w13, w13_scale, ids, mids, TOPK, 0, mids_scale,
        prefetch_tiles=8, BLOCK_N=64, **SWIGLU,
    )  # fmt: skip
    out = torch.empty((rows * TOPK, POOLED), dtype=torch.bfloat16, device="cuda")
    # Per-route outputs (the caller applies the route weights): unit weights,
    # and the route index buffer stage 2 needs without finalize is scratch.
    ones = torch.ones((rows, TOPK), dtype=torch.float32, device="cuda")
    route_index = torch.empty((rows * TOPK,), dtype=torch.int32, device="cuda")
    invoke_stage2_mxfp4_mfma_decode_gluon(
        mids, mids_scale, w2, w2_scale, ids, ones, out, TOPK, 0, False, route_index
    )
    return out


def _rmsnorm(x, weight, eps, offset):
    x = x.double()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x * (weight.float() + offset).to(torch.bfloat16).double()


def regular_reference(key, weights, inputs):
    """The regular layer's steps on the same inputs (collectives over the
    world group)."""
    q, gate = inputs["prepared_query"]
    gated = weightless_rmsnorm_gated_sigmoid(
        attention_reference(inputs), gate, Q_HEADS, HEAD_DIM, EPS, False
    )
    attention = gated.float() @ weights["o_proj"].float().t()
    dist.all_reduce(attention)
    post_w, post_eps, post_offset = weights["post_attention_norm"]
    pre_w, pre_eps, pre_offset = weights["pre_feedforward_norm"]
    stream, pre, pre_bf16 = torch_post_norm_residual_pre_norm(
        inputs["residual"], attention.to(torch.bfloat16), post_w, post_eps,
        post_offset, *weights["post_attention_gate"], pre_w, pre_eps, pre_offset,
        False, None, 1,
    )  # fmt: skip
    phases = dict(attention=attention, hprime=stream.float())

    # Router: FP32 logits scaled, sigmoid scores, the bias picks only.
    logits = (pre.float() @ weights["router"]) * weights["router_multiplier"]
    scores = logits.sigmoid()
    biased = scores + weights["router_bias"]
    ranked = biased.sort(dim=-1, descending=True).values
    gap = (ranked[:, TOPK - 1] - ranked[:, TOPK]).min().item()
    if gap <= ROUTING_GAP:
        raise RoutingTie(f"routing ties within {gap}")
    ids = biased.topk(TOPK, dim=-1).indices
    phases.update(logits=logits, ids=ids.sort(-1).values)
    route_weights = scores.gather(1, ids)
    route_weights = route_weights / route_weights.sum(-1, keepdim=True)
    route_weights = route_weights * weights["routed_scaling_factor"]

    # Pooler, its norm and the MXFP4 activation the experts read.
    pooled = (pre_bf16.float() @ weights["pooler"].float().t()).to(torch.bfloat16)
    normed = _rmsnorm(pooled, *weights["pooled_norm"]).to(torch.bfloat16)
    xq, xs = _quantize_mxfp4_activation(normed)
    routes = _direct_decode(xq, xs, weights["experts"], ids.to(torch.int32)).float()
    dist.all_reduce(routes)
    routes = routes.view(-1, TOPK, POOLED)
    combined = (
        _rmsnorm(routes, *weights["post_expert_norm"])
        * route_weights.double()[..., None]
    ).sum(1)
    phases.update(pooled=pooled, combined=combined)
    mlp = combined.to(torch.bfloat16).float() @ weights["post_expert_proj"].float().t()
    post2_w, post2_eps, post2_offset = weights["post_feedforward_norm"]
    next_w, next_eps, next_offset = weights["next_input_norm"]
    stream, _, next_input = torch_post_norm_residual_pre_norm(
        stream.float(), mlp.to(torch.bfloat16), post2_w, post2_eps, post2_offset,
        *weights["post_feedforward_gate"], next_w, next_eps, next_offset,
        False, None, 1,
    )  # fmt: skip
    phases.update(residual=stream.float())
    if key.next_kind == "regular":
        return dict(normed=next_input, residual=stream.float(), phases=phases)

    qkvg = (next_input.float() @ weights["next_qkv_proj"].float().t()).to(
        torch.bfloat16
    )
    q, k, v, next_gate = qkvg.split((Q_WIDTH, KV_WIDTH, KV_WIDTH, Q_WIDTH), dim=-1)
    target = fresh_next_prologue(inputs["next_prologue"])
    prepared = gqa_prologue(
        q, k, v, norm=target.norm, rotary=target.rotary, cache=target.cache,
        query_dtype=target.query_dtype, return_kv=False, solution=None, override=None,
    )  # fmt: skip
    return dict(q=prepared.q, gate=next_gate, residual=stream.float(),
                next_k=target.cache.k_cache, next_v=target.cache.v_cache,
                phases=phases)  # fmt: skip


def _relative_rms(actual, expected):
    actual, expected = actual.double(), expected.double()
    return (
        ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
    )


# One BF16 rounding is 2**-9 relative RMS; each reordered sum on the path
# (attention all-reduce, router and pooler K parts, route sum, QKV K parts)
# adds at most one, and a reordered value that crosses an MXFP4 rounding
# boundary moves that element one E2M1 step (rare: the bound leaves room).
_BF16_BOUND = 2**-6
# The FP8 E4M3 query and K/V: one rounding is 2**-4 relative.
_FP8_BOUND = 2**-3


def phase_errors(state, weights, expected, rows):
    """Relative RMS error of the fused launch's intermediates that stay in
    its state against the reference's, to locate the first phase that
    differs: h' (the residual after attention), the router logits (the sum
    of the K parts), the pooled rows and the summed experts (S)."""
    phases = expected["phases"]
    errors = {}
    hprime = state.hprime[:rows]
    errors["hprime"] = _relative_rms(hprime, phases["hprime"])
    words = state.logits_parts[:, :rows]
    parts = (words & 0xFFFFFFFF).to(torch.int32).view(torch.float32).sum(0)
    # The parts hold x . w; the multiplier applies after their sum.
    errors["logits"] = _relative_rms(
        parts * weights["router_multiplier"], phases["logits"]
    )
    errors["combined"] = _relative_rms(state.s_rows[:rows].float(), phases["combined"])
    return errors


def assert_matches(key, actual, expected):
    for name in ("residual", "normed", "gate"):
        if name in expected:
            error = _relative_rms(actual[name], expected[name])
            assert error < _BF16_BOUND, (key, name, error)
    if "q" in expected:
        error = _relative_rms(actual["q"].float(), expected["q"].float())
        assert error < _FP8_BOUND, (key, "q", error)
        for name in ("next_k", "next_v"):
            got = actual[name].view(-1, KV_HEADS, HEAD_DIM)
            want = expected[name].view(-1, KV_HEADS, HEAD_DIM)
            error = _relative_rms(got.float(), want.float())
            assert error < _FP8_BOUND, (key, name, error)
            # Only the step's slots are written.
            written = torch.zeros(got.shape[0], dtype=torch.bool, device="cuda")
            written[expected["slots"]] = True
            assert not got[~written].float().any(), (key, name)
