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

"""Random Muse Spark layers in the production layouts, for the fused-layer
test and benchmark on 8 gfx950 ranks.

``build_layer_inputs`` gives one rank's weights (as
``MuseSparkDecoderLayer.fused_weights`` hands them to the leaf: replicated
tensors equal on every rank, this rank's slices and EP1 expert slices) and
one verify step's inputs (the prepared query, the residual, the positions,
this layer's FP8 cache view, and the next layer's prologue target or input
norm).
"""

from dataclasses import dataclass, replace

import torch
from moe_stream_helpers import make_experts
from tokenspeed_kernel.ops.attention.mha import mha_decode_schedule
from tokenspeed_kernel.ops.attention.prologue import (
    HeadKVCache,
    HeadNorm,
    RopeStyle,
    Rotary,
)
from tokenspeed_kernel.ops.monokernel import PlanKey

# Muse Spark per rank at TP 8 / EP1.
WORLD = 8
HIDDEN, POOLED, EXPERTS, TOPK, INTER = 8192, 4096, 256, 8, 512
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 2, 64
Q_WIDTH, KV_WIDTH = Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
WIDTH, PAGE, WINDOW_LEFT = 8, 64, 2047
MAX_POSITION = 1 << 18
EPS, POST_EPS = 1e-5, 1e-8
TRAITS = dict(verify_width=WIDTH, tp_size=WORLD, ep_size=1, kv_cache_dtype="fp8_e4m3")


@dataclass(frozen=True)
class DecodeCache:
    """The fields of the runtime's DecodeKVView the launch reads, and the
    window the reference decode applies."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen_k: int
    split_schedule: torch.Tensor | None
    window_left: int


@dataclass(frozen=True)
class PrologueTarget:
    """The fields of the runtime's GQAPrologueTarget the launch reads."""

    cache: HeadKVCache
    norm: HeadNorm
    rotary: Rotary | None
    query_dtype: torch.dtype


def _randn(generator, device, *shape, scale=1.0, dtype=torch.bfloat16):
    values = torch.randn(shape, device=device, generator=generator) * scale
    return values.to(dtype)


def _norm(generator, device, width, eps):
    # Zero-centered gains (offset 1.0), as Muse's norms store them.
    return _randn(generator, device, width, scale=0.1), eps, 1.0


def _weights(plan_key, rank, seed, device):
    shared = torch.Generator(device=device).manual_seed(seed)
    local = torch.Generator(device=device).manual_seed(seed + 100 + rank)
    gates = [
        torch.rand(HIDDEN, device=device, generator=shared) * 0.2 + low
        for low in (0.8, 0.1, 0.8, 0.1)
    ]
    window_left = WINDOW_LEFT if plan_key.layer_kind == "sliding" else -1
    return {
        "attention": dict(scale=HEAD_DIM**-0.5, window_left=window_left, gate_eps=EPS),
        "o_proj": _randn(local, device, HIDDEN, Q_WIDTH, scale=Q_WIDTH**-0.5),
        # Muse's post-attention norm has no weight.
        "post_attention_norm": (None, POST_EPS, 1.0),
        "post_attention_gate": (gates[0], gates[1]),
        "pre_feedforward_norm": _norm(shared, device, HIDDEN, EPS),
        "router": _randn(shared, device, HIDDEN, EXPERTS, dtype=torch.float32),
        "router_multiplier": 0.125,
        "router_bias": torch.randn(EXPERTS, device=device, generator=shared) * 0.05,
        "routed_scaling_factor": 2.5,
        "pooler": _randn(shared, device, POOLED, HIDDEN, scale=HIDDEN**-0.5),
        "pooler_replicated": True,
        "pooled_norm": _norm(shared, device, POOLED, EPS),
        "experts": make_experts(EXPERTS, device, seed + 1000 + rank)[0],
        "post_expert_proj": _randn(shared, device, HIDDEN, POOLED, scale=POOLED**-0.5),
        "post_expert_replicated": True,
        "post_expert_norm": _norm(shared, device, POOLED, POST_EPS),
        "post_feedforward_norm": _norm(shared, device, HIDDEN, POST_EPS),
        "post_feedforward_gate": (gates[2], gates[3]),
        "next_input_norm": _norm(shared, device, HIDDEN, EPS),
        "next_qkv_proj": _randn(
            local, device, Q_WIDTH * 2 + KV_WIDTH * 2, HIDDEN, scale=HIDDEN**-0.5
        ),
    }


def _cos_sin(device):
    inv = 10000.0 ** (-torch.arange(0, HEAD_DIM, 2, device=device) / HEAD_DIM)
    angles = torch.arange(MAX_POSITION, device=device)[:, None] * inv[None, :]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float().contiguous()


def build_layer_inputs(plan_key: PlanKey, rank: int, seed: int, device, context):
    """One rank's random layer and one verify step for ``plan_key``.

    Args:
        plan_key: The launch's plan key (requests, layer kind, next kind).
        rank: This rank in the group of ``WORLD``.
        seed: Seeds the weights and the inputs.
        device: This rank's GPU.
        context: Cached tokens of each request before its verify rows (one
            per request); a full layer reads all of them.

    Returns:
        ``(weights, inputs)``: the leaf's weights dict, and a dict with the
        launch's ``prepared_query`` (FP8 q, BF16 gate), ``residual`` (FP32),
        ``positions``, ``cache`` (:class:`DecodeCache`; full layers carry the
        backend's split schedule) and either ``next_prologue``
        (:class:`PrologueTarget` over zeroed next-layer caches) or
        ``next_norm``; plus ``slots``, the rows' K/V write slots.
    """
    if len(context) != plan_key.requests:
        raise ValueError(f"{plan_key}: one context length per request, got {context}")
    weights = _weights(plan_key, rank, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 7)
    requests = plan_key.requests
    rows = requests * WIDTH
    per_request = -(-(max(context) + WIDTH) // PAGE)
    total = requests * per_request
    pages = torch.randperm(total, device=device, generator=generator).to(torch.int32)
    page_table = pages.view(requests, per_request).contiguous()
    lengths = torch.tensor(context, device=device)
    seq_lens = (lengths + WIDTH).to(torch.int32)
    positions = (lengths[:, None] + torch.arange(WIDTH, device=device)).flatten()
    page_of = page_table.long().gather(1, positions.view(requests, WIDTH) // PAGE)
    slots = (page_of * PAGE + positions.view(requests, WIDTH) % PAGE).flatten()

    def cache():
        values = torch.randn(
            (total, PAGE, KV_HEADS, HEAD_DIM), device=device, generator=generator
        )
        return values.to(torch.float8_e4m3fn)

    q = (_randn(generator, device, rows, Q_WIDTH, dtype=torch.float32) * 2).to(
        torch.float8_e4m3fn
    )
    gate = _randn(generator, device, rows, Q_WIDTH)
    residual = _randn(generator, device, rows, HIDDEN, dtype=torch.float32)
    k_cache, v_cache = cache(), cache()
    schedule = None
    window_left = weights["attention"]["window_left"]
    if plan_key.layer_kind == "full":
        # The backend's call: the forward's balanced schedule, in place.
        schedule = mha_decode_schedule(
            q=q.view(rows, Q_HEADS, HEAD_DIM), k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=seq_lens, max_seqlen_q=WIDTH,
            out=torch.zeros((requests + 1, 2), dtype=torch.int32, device=device),
        )  # fmt: skip
        assert schedule is not None
    inputs = dict(
        prepared_query=(q, gate),
        residual=residual,
        positions=positions,
        cache=DecodeCache(
            k_cache, v_cache, page_table, seq_lens, MAX_POSITION, schedule, window_left
        ),
        slots=slots,
    )
    if plan_key.next_kind == "regular":
        inputs["next_norm"] = weights["next_input_norm"]
        return weights, inputs
    norm_generator = torch.Generator(device=device).manual_seed(seed + 3)
    next_cache = torch.zeros(
        (total, PAGE, KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
    )
    flat = (-1, KV_HEADS, HEAD_DIM)
    inputs["next_prologue"] = PrologueTarget(
        # The runtime's types: FP32 norm weights, int32 write slots.
        HeadKVCache(
            next_cache.view(flat),
            next_cache.clone().view(flat),
            None,
            slots.to(torch.int32),
        ),
        HeadNorm(
            _randn(norm_generator, device, HEAD_DIM, scale=0.1, dtype=torch.float32),
            _randn(norm_generator, device, HEAD_DIM, scale=0.1, dtype=torch.float32),
            1.0,
            EPS,
        ),
        # Muse's full layers use no position embedding (NoPE).
        (
            None
            if plan_key.next_kind == "full"
            else Rotary(_cos_sin(device), positions, RopeStyle.GPTJ, None)
        ),
        torch.float8_e4m3fn,
    )
    return weights, inputs


def fresh_next_prologue(target: PrologueTarget) -> PrologueTarget:
    """``target`` over copies of its caches, so a launch can be repeated from
    the same starting caches."""
    cache = replace(
        target.cache,
        k_cache=target.cache.k_cache.clone(),
        v_cache=target.cache.v_cache.clone(),
    )
    return replace(target, cache=cache)
