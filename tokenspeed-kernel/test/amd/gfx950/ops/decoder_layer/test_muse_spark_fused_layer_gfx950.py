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

"""One fused Muse Spark layer launch against the regular layer's math, on 8
gfx950 ranks.

Each rank prepares the ``muse_spark`` leaf over random weights in the
production layouts (EP1 expert slices from ``make_experts``) and runs one
launch per plan key. The reference composes the regular path's steps on the
same inputs: today's decode kernel over the FP8 cache (with the forward's
split schedule on full layers, ~141k context), the gated o-norm, the
o_proj all-reduce, the attention boundary, the FP32 router with sigmoid
top-8, the pooler with its norm and MXFP4 quantization, today's direct-decode
MoE kernels per rank and their route sum, the routed norm, the post-expert
projection, the FFN boundary, and the next layer's QKV projection and
``gqa_prologue``. Routing is checked untied on the reference logits, so both
sides pick the same experts.

The fused sums run in a different order (deterministic, not bitwise), and the
pooled rows feed an MXFP4 quantization, where an order difference can move a
value across a rounding boundary. So outputs are compared by relative RMS
error (one BF16 rounding per reordered step, with headroom for those rare
MXFP4 steps), and bitwise only where both sides compute alike: a full
layer's gated attention rows, across ranks, across repeated launches, and
between replay and eager.
"""

import socket
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

WORLD = 8
if not (
    torch.cuda.is_available()
    and torch.cuda.device_count() >= WORLD
    and current_platform().is_cdna4
):
    pytest.skip("needs 8 gfx950 GPUs", allow_module_level=True)

# make_experts lives with the native MoE tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "moe"))

from native_moe_helpers import make_experts  # noqa: E402
from tokenspeed_kernel.numerics.torch.residual import (  # noqa: E402
    torch_post_norm_residual_pre_norm,
)
from tokenspeed_kernel.ops.attention.mha import (  # noqa: E402
    mha_decode_schedule,
    mha_decode_with_kvcache,
)
from tokenspeed_kernel.ops.attention.prologue import (  # noqa: E402
    HeadKVCache,
    HeadNorm,
    RopeStyle,
    Rotary,
    gqa_prologue,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.fused_layer import (  # noqa: E402
    BOX_WORDS,
)
from tokenspeed_kernel.ops.decoder_layer import (  # noqa: E402
    PlanKey,
    prepare_decoder_layer,
    prepare_decoder_layer_state,
)
from tokenspeed_kernel.ops.layernorm.fused_norm import (  # noqa: E402
    weightless_rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage1 import (  # noqa: E402
    invoke_stage1_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_stage2 import (  # noqa: E402
    invoke_stage2_mxfp4_mfma_decode_gluon,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    _extract_gluon_raw_s,
    _extract_gluon_raw_w,
    _quantize_mxfp4_activation,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (  # noqa: E402
    empty_swizzled_cdna4_mxfp4_scale,
)

# Muse Spark per rank at TP 8 / EP1.
HIDDEN, POOLED, EXPERTS, TOPK, INTER = 8192, 4096, 256, 8, 512
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 2, 64
Q_WIDTH, KV_WIDTH = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
WIDTH, PAGE, WINDOW_LEFT = 8, 64, 2047
# Cached tokens per request before the verify rows. Full layers read the
# whole context: one request at ~141k (the served context), and two, whose
# split schedule then gives workgroups two items.
CONTEXT = {"sliding": (1500, 2600), "full": (141_000, 141_000)}
MAX_POSITION = 1 << 18
EPS, POST_EPS = 1e-5, 1e-8
TRAITS = dict(verify_width=WIDTH, tp_size=WORLD, ep_size=1, kv_cache_dtype="fp8_e4m3")
SWIGLU = dict(swiglu_alpha=1.0, swiglu_limit=0.0, swiglu_beta=0.0)
# 8th vs 9th biased routing score: far above the logit error of a reordered
# FP32 sum over 8192 products (~2**-20 relative).
ROUTING_GAP = 1e-3


@dataclass(frozen=True)
class _DecodeCache:
    """The fields of the runtime's DecodeKVView the launch reads."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen_k: int
    split_schedule: torch.Tensor | None
    window_left: int


@dataclass(frozen=True)
class _PrologueTarget:
    """The fields of the runtime's GQAPrologueTarget the launch reads."""

    cache: HeadKVCache
    norm: HeadNorm
    rotary: Rotary | None
    query_dtype: torch.dtype


def _randn(generator, *shape, scale=1.0, dtype=torch.bfloat16):
    values = torch.randn(shape, device="cuda", generator=generator) * scale
    return values.to(dtype)


def _norm(generator, width, eps):
    # Zero-centered gains (offset 1.0), as Muse's norms store them.
    return _randn(generator, width, scale=0.1), eps, 1.0


def _weights(rank):
    """One layer's weights as ``MuseSparkDecoderLayer.fused_weights`` hands
    them over: replicated tensors equal on every rank, slices per rank."""
    shared = torch.Generator(device="cuda").manual_seed(0)
    local = torch.Generator(device="cuda").manual_seed(100 + rank)
    gates = [
        torch.rand(HIDDEN, device="cuda", generator=shared) * 0.2 + low
        for low in (0.8, 0.1, 0.8, 0.1)
    ]
    weights = {
        "attention": dict(scale=HEAD_DIM**-0.5, window_left=WINDOW_LEFT, gate_eps=EPS),
        "o_proj": _randn(local, HIDDEN, Q_WIDTH, scale=Q_WIDTH**-0.5),
        # Muse's post-attention norm has no weight.
        "post_attention_norm": (None, POST_EPS, 1.0),
        "post_attention_gate": (gates[0], gates[1]),
        "pre_feedforward_norm": _norm(shared, HIDDEN, EPS),
        "router": _randn(shared, HIDDEN, EXPERTS, dtype=torch.float32),
        "router_multiplier": 0.125,
        "router_bias": torch.randn(EXPERTS, device="cuda", generator=shared) * 0.05,
        "routed_scaling_factor": 2.5,
        "pooler": _randn(shared, POOLED, HIDDEN, scale=HIDDEN**-0.5),
        "pooler_replicated": True,
        "pooled_norm": _norm(shared, POOLED, EPS),
        "experts": make_experts(EXPERTS, "cuda", 1000 + rank)[0],
        "post_expert_proj": _randn(shared, HIDDEN, POOLED, scale=POOLED**-0.5),
        "post_expert_replicated": True,
        "post_expert_norm": _norm(shared, POOLED, POST_EPS),
        "post_feedforward_norm": _norm(shared, HIDDEN, POST_EPS),
        "post_feedforward_gate": (gates[2], gates[3]),
        "next_input_norm": _norm(shared, HIDDEN, EPS),
        "next_qkv_proj": _randn(
            local, Q_WIDTH * 2 + KV_WIDTH * 2, HIDDEN, scale=HIDDEN**-0.5
        ),
    }
    return weights


def _cos_sin():
    inv = 10000.0 ** (-torch.arange(0, HEAD_DIM, 2, device="cuda") / HEAD_DIM)
    angles = torch.arange(MAX_POSITION, device="cuda")[:, None] * inv[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float().contiguous()


def _inputs(key):
    """One verify step of ``key.requests`` requests: the prepared query, the
    FP8 caches of this layer (holding every key the step reads) and the next,
    the page table and the write slots. The same on every rank, which reads
    them as its own heads."""
    generator = torch.Generator(device="cuda").manual_seed(7 + key.requests)
    requests = key.requests
    rows = requests * WIDTH
    lengths = CONTEXT[key.layer_kind][:requests]
    per_request = -(-(max(lengths) + WIDTH) // PAGE)
    total = requests * per_request
    pages = torch.randperm(total, device="cuda", generator=generator).to(torch.int32)
    page_table = pages.view(requests, per_request).contiguous()
    context = torch.tensor(lengths, device="cuda")
    seq_lens = (context + WIDTH).to(torch.int32)
    positions = (context[:, None] + torch.arange(WIDTH, device="cuda")).flatten()
    page_of = page_table.long().gather(1, (positions.view(requests, WIDTH) // PAGE))
    slots = (page_of * PAGE + positions.view(requests, WIDTH) % PAGE).flatten()

    def cache(scale):
        values = torch.randn(
            (total, PAGE, KV_HEADS, HEAD_DIM), device="cuda", generator=generator
        )
        return (values * scale).to(torch.float8_e4m3fn)

    q = (_randn(generator, rows, Q_WIDTH, dtype=torch.float32) * 2).to(
        torch.float8_e4m3fn
    )
    gate = _randn(generator, rows, Q_WIDTH)
    residual = _randn(generator, rows, HIDDEN, dtype=torch.float32)
    next_k = torch.zeros((total, PAGE, KV_HEADS, HEAD_DIM), device="cuda").to(
        torch.float8_e4m3fn
    )
    norm_generator = torch.Generator(device="cuda").manual_seed(3)
    head_norm = HeadNorm(
        _randn(norm_generator, HEAD_DIM, scale=0.1),
        _randn(norm_generator, HEAD_DIM, scale=0.1),
        1.0,
        EPS,
    )
    rotary = (
        None
        if key.next_kind == "full"
        else Rotary(_cos_sin(), positions, RopeStyle.GPTJ, None)
    )
    k_cache, v_cache = cache(1.0), cache(1.0)
    window_left = WINDOW_LEFT if key.layer_kind == "sliding" else -1
    schedule = None
    if key.layer_kind == "full":
        # The backend's call: the forward's balanced schedule, in place.
        schedule = mha_decode_schedule(
            q=q.view(rows, Q_HEADS, HEAD_DIM), k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=seq_lens, max_seqlen_q=WIDTH,
            out=torch.zeros((requests + 1, 2), dtype=torch.int32, device="cuda"),
        )  # fmt: skip
        assert schedule is not None
    return dict(
        prepared_query=(q, gate),
        residual=residual,
        positions=positions,
        cache=_DecodeCache(
            k_cache, v_cache, page_table, seq_lens, MAX_POSITION, schedule, window_left
        ),
        next_k=next_k,
        next_v=next_k.clone(),
        slots=slots,
        head_norm=head_norm,
        rotary=rotary,
    )


def _next_prologue(inputs, k_cache, v_cache):
    flat = (-1, KV_HEADS, HEAD_DIM)
    return _PrologueTarget(
        HeadKVCache(k_cache.view(flat), v_cache.view(flat), None, inputs["slots"]),
        inputs["head_norm"],
        inputs["rotary"],
        torch.float8_e4m3fn,
    )


def _run(fused, key, inputs):
    """One fused launch; the residual and the next caches it updates are
    copies, so a launch can be repeated."""
    residual = inputs["residual"].clone()
    common = dict(positions=inputs["positions"], cache=inputs["cache"])
    if key.next_kind == "regular":
        normed, residual = fused.forward_last(
            inputs["prepared_query"], residual, next_norm=inputs["next_norm"], **common
        )
        return dict(normed=normed, residual=residual)
    next_k, next_v = inputs["next_k"].clone(), inputs["next_v"].clone()
    (q, gate), residual = fused.forward(
        inputs["prepared_query"],
        residual,
        next_prologue=_next_prologue(inputs, next_k, next_v),
        **common,
    )
    return dict(q=q.reshape(q.shape[0], -1), gate=gate, residual=residual,
                next_k=next_k, next_v=next_v)  # fmt: skip


def _attention(inputs):
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
    invoke_stage2_mxfp4_mfma_decode_gluon(
        mids, mids_scale, w2, w2_scale, ids, None, out, TOPK, 0, False, None
    )
    return out


def _rmsnorm(x, weight, eps, offset):
    x = x.double()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x * (weight.float() + offset).to(torch.bfloat16).double()


def _reference(key, weights, inputs):
    """The regular layer's steps on the same inputs (collectives over the
    world group)."""
    q, gate = inputs["prepared_query"]
    gated = weightless_rmsnorm_gated_sigmoid(
        _attention(inputs), gate, Q_HEADS, HEAD_DIM, EPS, False
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

    # Router: FP32 logits scaled, sigmoid scores, the bias picks only.
    logits = (pre.float() @ weights["router"]) * weights["router_multiplier"]
    scores = logits.sigmoid()
    biased = scores + weights["router_bias"]
    ranked = biased.sort(dim=-1, descending=True).values
    gap = (ranked[:, TOPK - 1] - ranked[:, TOPK]).min().item()
    assert gap > ROUTING_GAP, f"routing ties within {gap}; draw other inputs"
    ids = biased.topk(TOPK, dim=-1).indices
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
    mlp = combined.to(torch.bfloat16).float() @ weights["post_expert_proj"].float().t()
    post2_w, post2_eps, post2_offset = weights["post_feedforward_norm"]
    next_w, next_eps, next_offset = weights["next_input_norm"]
    stream, _, next_input = torch_post_norm_residual_pre_norm(
        stream.float(), mlp.to(torch.bfloat16), post2_w, post2_eps, post2_offset,
        *weights["post_feedforward_gate"], next_w, next_eps, next_offset,
        False, None, 1,
    )  # fmt: skip
    if key.next_kind == "regular":
        return dict(normed=next_input, residual=stream.float())

    qkvg = (next_input.float() @ weights["next_qkv_proj"].float().t()).to(
        torch.bfloat16
    )
    q, k, v, next_gate = qkvg.split((Q_WIDTH, KV_WIDTH, KV_WIDTH, Q_WIDTH), dim=-1)
    next_k, next_v = inputs["next_k"].clone(), inputs["next_v"].clone()
    target = _next_prologue(inputs, next_k, next_v)
    prepared = gqa_prologue(
        q, k, v, norm=target.norm, rotary=target.rotary, cache=target.cache,
        query_dtype=target.query_dtype, return_kv=False, solution=None, override=None,
    )  # fmt: skip
    return dict(q=prepared.q, gate=next_gate, residual=stream.float(),
                next_k=next_k, next_v=next_v)  # fmt: skip


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


def _compare(key, actual, expected):
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


PLAN_KEYS = [
    PlanKey(requests, kind, next_kind)
    for kind in ("sliding", "full")
    for requests in (1, 2)
    for next_kind in ("sliding", "full", "regular")
]


def _check_attention_box(state, inputs):
    """The launch's attention box holds the gated-normed attention rows; for
    a full layer, the split-schedule attention followed by the gated norm
    must equal today's decode kernel and gated-norm kernel bit for bit (the
    box stores -0.0 as +0.0: its empty word is the -0.0 pair)."""
    q, gate = inputs["prepared_query"]
    rows = q.shape[0]
    expected = weightless_rmsnorm_gated_sigmoid(
        _attention(inputs), gate, Q_HEADS, HEAD_DIM, EPS, False
    )
    # This launch's copy of the box (epoch & 1), as BF16 rows.
    epoch = int(state.epochs[0])
    box = state.iris.box.view(2, BOX_WORDS)[epoch & 1]
    actual = box.view(torch.bfloat16).view(-1, Q_WIDTH)[:rows]

    def positive_zero(values):
        return torch.where(values == 0, torch.zeros_like(values), values)

    assert torch.equal(
        positive_zero(actual).view(torch.int16),
        positive_zero(expected).view(torch.int16),
    )


def _worker(rank, port, record_dir, errors):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=WORLD
        )
        state = prepare_decoder_layer_state(
            recipe="muse_spark",
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_requests=2,
            failure_record_path=f"{record_dir}/rank{rank}.bin",
            device=torch.device("cuda", rank),
        )
        base = _weights(rank)
        for key in PLAN_KEYS:
            window_left = WINDOW_LEFT if key.layer_kind == "sliding" else -1
            weights = dict(
                base, attention=dict(base["attention"], window_left=window_left)
            )
            fused = prepare_decoder_layer(
                recipe="muse_spark", plan_keys=[key], traits=TRAITS,
                weights=weights, state=state,
            )  # fmt: skip
            assert fused is not None
            assert fused.decline_reasons(key.requests) == [], key
            inputs = _inputs(key)
            inputs["next_norm"] = weights["next_input_norm"]
            expected = _reference(key, weights, inputs)
            expected["slots"] = inputs["slots"]
            actual = _run(fused, key, inputs)
            if key.layer_kind == "full":
                _check_attention_box(state, inputs)
            _compare(key, actual, expected)

            # Deterministic: the same launch again, at the next epoch and
            # the other copy of every two-copy area, gives the same bits.
            again = _run(fused, key, inputs)
            for name, value in actual.items():
                assert torch.equal(
                    value.view(torch.uint8), again[name].view(torch.uint8)
                ), (key, name)

            # Every rank holds the same residual.
            gathered = [torch.empty_like(actual["residual"]) for _ in range(WORLD)]
            dist.all_gather(gathered, actual["residual"])
            assert all(torch.equal(r, actual["residual"]) for r in gathered), key

            # Graph replay equals eager.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _run(fused, key, inputs)
            graph.replay()
            torch.cuda.synchronize()
            for name, value in actual.items():
                assert torch.equal(
                    value.view(torch.uint8), captured[name].view(torch.uint8)
                ), (key, name)
    except Exception:
        errors[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_fused_layer_matches_the_regular_layer(tmp_path):
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(port, str(tmp_path), errors), nprocs=WORLD, join=True)
    assert not errors, "\n".join(f"rank {r}: {e}" for r, e in errors.items())
