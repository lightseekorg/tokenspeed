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

"""Verify Kimi-K3's fused MoE layer tail against the regular decode ops."""

import dataclasses
import socket
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.numerics.torch.activation import torch_situ_and_mul
from tokenspeed_kernel.numerics.torch.moe import DequantizedExperts
from tokenspeed_kernel.numerics.torch.quantization import dequantize_mxfp4
from tokenspeed_kernel.ops.activation.triton import attnres_combine, attnres_partial
from tokenspeed_kernel.ops.attention.kda import KdaOutputNorm, apply_kda_output_norm
from tokenspeed_kernel.ops.gemm import (
    kimi3_latent_projection_add3,
    kimi3_shared_down_projection,
    mm,
)
from tokenspeed_kernel.ops.model.kimi_k3 import (
    KimiK3MoeTailLayer,
    KimiK3MoeTailShape,
    KimiK3NextMix,
    kimi_k3_moe_tail,
    kimi_k3_moe_tail_expert_layout,
    pack_kimi_k3_moe_tail_layer,
    prepare_kimi_k3_moe_tail,
)
from tokenspeed_kernel.ops.moe import (
    latent_moe_input_projections,
    moe_plan,
    moe_process_weights,
    moe_topk,
)
from tokenspeed_kernel.ops.residual import attn_res_fwd
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile, make_mxfp4_moe_weights

# Kimi-K3.1 at TP8, one request's speculative verify step; a case sets its own
# rows, one request's decode width.
SHAPE = KimiK3MoeTailShape(
    rows=8,
    hidden_size=7168,
    tp_size=8,
    num_heads=96,
    head_dim=128,
    num_experts=896,
    top_k=16,
    latent_size=3584,
    intermediate_size=3072,
    shared_intermediate_size=6144,
    weight_dtype="mxfp4",
    num_expert_groups=1,
    topk_groups=1,
    scoring_func="sigmoid",
    routed_scaling_factor=1.0,
    renormalize=True,
    situ_beta=4.0,
    situ_linear_beta=25.0,
)
# Kimi-K3.1's norm epsilon.
EPS = 1e-5
# Kimi-K3.1's AttnRes snapshot bank: 93 layers in blocks of 12.
BANK_BLOCKS = 8
# Kimi-K3.1's merged KDA input projection on one rank, whose fourth 1536
# columns are the attention output's gate logits.
KDA_PROJECTION = 6288

# The router logits of the two paths differ only by FP32 summation order, so
# a row whose 16th and 17th biased scores lie closer than this may still route
# differently; such rows are left out of the comparison.
ROUTE_MARGIN = 2**-20
# Per compared row: the error's norm against the norm of the row's change, and
# its largest element against the change's root mean square. Each bound is twice
# the largest value measured over every case, both steps and all ranks, rounded
# up to a power of two.
ROW_ERROR = 2**-7
PEAK_ERROR = 2**-3
# A swapped route must move every compared row at least this many times the
# row bound, so that the bound tells a wrong expert from rounding.
SEPARATION = 4

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4 or torch.cuda.device_count() < SHAPE.tp_size,
    reason="the tail runs one rank on each of eight CDNA4 GPUs",
)


def _weights(
    rank: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """This rank's dense weights by the tail's role names, and its routed
    experts' MXFP4 bytes and scales as the checkpoint stores them. Replicated
    weights are equal on every rank; the attention output projection and the
    shared and routed experts are this rank's shards."""
    hidden, latent, tp = SHAPE.hidden_size, SHAPE.latent_size, SHAPE.tp_size
    attention_width = SHAPE.num_heads // tp * SHAPE.head_dim
    shared_width = SHAPE.shared_intermediate_size // tp
    replicated = torch.Generator(device=device).manual_seed(7)
    local = torch.Generator(device=device).manual_seed(100 + rank)

    def normal(generator, *shape, std):
        return (torch.randn(shape, device=device, generator=generator) * std).bfloat16()

    def norm(width):
        return (torch.rand(width, device=device, generator=replicated) + 0.5).bfloat16()

    weights = {
        "o_proj": normal(
            local, hidden, attention_width, std=(attention_width * tp) ** -0.5
        ),
        "mlp_res_norm": norm(hidden),
        "mlp_res_proj": normal(replicated, hidden, std=1 / 64),
        "post_attention_norm": norm(hidden),
        "router": normal(replicated, SHAPE.num_experts, hidden, std=hidden**-0.5),
        "router_bias": torch.randn(
            SHAPE.num_experts, device=device, generator=replicated
        )
        * 0.02,
        "latent_down": normal(replicated, latent, hidden, std=hidden**-0.5),
        "latent_norm": norm(latent),
        "latent_up": normal(replicated, hidden, latent, std=latent**-0.5),
        "shared_gate_up": normal(local, 2 * shared_width, hidden, std=hidden**-0.5),
        "shared_down": normal(
            local, hidden, shared_width, std=SHAPE.shared_intermediate_size**-0.5
        ),
    }
    experts = make_mxfp4_moe_weights(
        SHAPE.num_experts,
        latent,
        SHAPE.intermediate_size // tp,
        local,
        device=device,
        scale_range=(119, 122),
    )
    return weights, experts


def _inputs(
    rank: int,
    device: torch.device,
    step: int,
    rows: int,
    num_blocks: int,
    restarts_prefix: bool,
    gated_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, KdaOutputNorm | None]:
    """One step's inputs: this rank's gated attention output, and the incoming
    stream and the snapshot bank, equal on every rank. A layer that restarts
    the prefix has stored its incoming stream as the newest snapshot. With
    ``gated_norm`` the attention output is a KDA layer's raw recurrence output
    instead, and the step also has the layer's gated output norm."""
    replicated = torch.Generator(device=device).manual_seed(3000 + step)
    local = torch.Generator(device=device).manual_seed(4000 + 16 * step + rank)
    hidden = SHAPE.hidden_size
    attention_width = SHAPE.num_heads // SHAPE.tp_size * SHAPE.head_dim
    attn_output = torch.randn(
        (rows, attention_width), device=device, generator=local
    ).bfloat16()
    stream = torch.randn((rows, hidden), device=device, generator=replicated).bfloat16()
    blocks = torch.randn(
        (BANK_BLOCKS, rows, hidden), device=device, generator=replicated
    ).bfloat16()
    if restarts_prefix:
        blocks[num_blocks - 1] = stream
    attn_norm = None
    if gated_norm:
        # The tail compiles in the gate's row stride.
        projection = torch.randn(
            (rows, KDA_PROJECTION), device=device, generator=local
        ).bfloat16()
        weight = torch.rand(SHAPE.head_dim, device=device, generator=replicated)
        attn_norm = KdaOutputNorm(
            projection[:, 3 * attention_width : 4 * attention_width],
            (weight + 0.5).bfloat16(),
            EPS,
        )
    return attn_output, stream, blocks, attn_norm


def _tail_mlp_input(layer: KimiK3MoeTailLayer) -> torch.Tensor:
    """The rows the tail's last launch fed its experts (the MLP-side mix after
    the post-attention norm), which it keeps in its scratch."""
    from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail.contract import (
        scratch_layout,
    )

    offset, size = scratch_layout(layer.options.build)["xrow"]
    rows = layer.tail.workspace.scratch[offset : offset + size].view(torch.bfloat16)
    return rows.view(layer.tail.shape.rows, SHAPE.hidden_size).clone()


def _fake_quantize_mxfp4(values: torch.Tensor) -> torch.Tensor:
    """MXFP4 round trip as the tail quantizes its expert inputs: a 32-value
    group's scale is ``2 ** ceil(log2(amax / 6))``, and each value rounds to
    the nearest E2M1 value, a tie to the even code."""
    groups = values.float().unflatten(-1, (-1, 32))
    bits = (groups.abs().amax(dim=-1, keepdim=True) * (1.0 / 6.0)).view(torch.int32)
    scale = (((bits + 0x7FFFFF) >> 23).clamp(max=254) << 23).view(torch.float32)
    scaled = groups.abs() / torch.where(scale == 0, 1.0, scale)
    # Even codes first, so that argmin settles a tie on the even code.
    levels = scaled.new_tensor((0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0))
    nearest = levels[(scaled.unsqueeze(-1) - levels).abs().argmin(dim=-1)]
    return (nearest * groups.sign() * scale).flatten(-2)


def _routed_experts(
    latent: torch.Tensor,
    experts: dict[str, torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """This rank's routed experts with the tail's arithmetic: MXFP4 inputs and
    SiTU outputs, FP32 products, each route's output times its weight rounded
    to BF16, and a row's routes added in descending score order with a BF16
    rounding after each add. No regular SiTU MoE quantizes its activations to
    MXFP4 (the regular decode uses FP8), so this is PyTorch."""
    w13 = DequantizedExperts(
        dequantize_mxfp4, experts["w13_weight"], experts["w13_scale"]
    )
    w2 = DequantizedExperts(dequantize_mxfp4, experts["w2_weight"], experts["w2_scale"])
    inputs = _fake_quantize_mxfp4(latent)
    order = scores.gather(-1, topk_ids.long()).argsort(dim=-1, descending=True)
    out = torch.zeros_like(latent)
    for row in range(latent.shape[0]):
        for slot in order[row].tolist():
            expert = int(topk_ids[row, slot])
            gate_up = w13[expert] @ inputs[row]
            activation = torch_situ_and_mul(
                gate_up, SHAPE.situ_beta, SHAPE.situ_linear_beta
            )
            term = (w2[expert] @ _fake_quantize_mxfp4(activation)) * topk_weights[
                row, slot
            ]
            out[row] = (out[row].float() + term.bfloat16().float()).bfloat16()
    return out


def _regular_seam(
    weights: dict[str, torch.Tensor],
    attn_output: torch.Tensor,
    stream: torch.Tensor | None,
    blocks: torch.Tensor,
    num_blocks: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The new prefix and the MLP-side input rows, from the regular decode ops:
    the o-projection, its all-reduce and residual add, then the AttnRes mix
    and the post-attention norm."""
    attention = mm(attn_output, weights["o_proj"]).float()
    dist.all_reduce(attention, group=group)
    attention = attention.bfloat16()
    prefix = (
        attention if stream is None else (stream.float() + attention.float()).bfloat16()
    )
    hidden = attn_res_fwd(
        prefix,
        blocks[:num_blocks],
        weights["mlp_res_proj"],
        weights["mlp_res_norm"],
        EPS,
        out_norm_weight=weights["post_attention_norm"],
        out_norm_eps=EPS,
        num_valid_blocks=num_blocks,
    )
    return prefix, hidden


def _regular_moe(
    weights: dict[str, torch.Tensor],
    experts: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    prefix: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    swap_last_route: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The layer output from the MLP-side input rows on, and the router
    logits. ``swap_last_route`` is the control: each row's lowest-scoring
    route goes to its 17th choice instead."""
    logits, latent, shared_input = latent_moe_input_projections(
        hidden,
        weights["router"],
        weights["latent_down"],
        weights["shared_gate_up"],
        gate_clamp=SHAPE.situ_beta,
        up_clamp=SHAPE.situ_linear_beta,
    )
    topk_weights, topk_ids = moe_topk(
        logits,
        SHAPE.top_k,
        SHAPE.scoring_func,
        "topk",
        renormalize=SHAPE.renormalize,
        routed_scaling_factor=SHAPE.routed_scaling_factor,
        correction_bias=weights["router_bias"],
    )
    scores = logits.sigmoid() + weights["router_bias"]
    if swap_last_route:
        ranked = scores.topk(SHAPE.top_k + 1, dim=-1).indices.to(topk_ids.dtype)
        topk_ids = torch.where(topk_ids == ranked[:, -2:-1], ranked[:, -1:], topk_ids)
    routed = _routed_experts(latent, experts, topk_weights, topk_ids, scores)
    shared = kimi3_shared_down_projection(shared_input, weights["shared_down"])
    partials = torch.cat((routed, shared), dim=-1).float()
    dist.all_reduce(partials, group=group)
    routed, shared = partials.bfloat16().split(
        (SHAPE.latent_size, SHAPE.hidden_size), dim=-1
    )
    out = kimi3_latent_projection_add3(
        routed.contiguous(),
        weights["latent_up"],
        prefix,
        shared.contiguous(),
        norm_weight=weights["latent_norm"],
        eps=EPS,
    )
    return out, logits


def _clear_rows(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The rows whose 16th and 17th biased router scores are apart."""
    scores = (logits.sigmoid() + bias).topk(SHAPE.top_k + 1, dim=-1).values
    return scores[:, -2] - scores[:, -1] > ROUTE_MARGIN


def _row_errors(
    actual: torch.Tensor, expected: torch.Tensor, base: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's error norm against the norm of its change from ``base``,
    and its largest error element against the change's root mean square."""
    change = expected.float() - base.float()
    error = actual.float() - expected.float()
    size = change.norm(dim=-1)
    return (
        error.norm(dim=-1) / size,
        error.abs().amax(dim=-1) / (size / change.shape[-1] ** 0.5),
    )


def _assert_rows_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    base: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    relative, peak = (error[rows] for error in _row_errors(actual, expected, base))
    assert relative.max() <= ROW_ERROR and peak.max() <= PEAK_ERROR, (
        relative.tolist(),
        peak.tolist(),
    )


def _assert_separated(
    swapped: torch.Tensor,
    expected: torch.Tensor,
    base: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    relative = _row_errors(swapped, expected, base)[0][rows]
    assert relative.min() >= SEPARATION * ROW_ERROR, relative.tolist()


def _check(
    rank: int,
    device: torch.device,
    group: dist.ProcessGroup,
    layer_index: int,
    num_blocks: int,
    restarts_prefix: bool,
    writes_next_mix: bool,
    rows: int,
    gated_norm: bool,
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.kimi_k3.moe_tail import (
        gluon_kimi_k3_moe_tail_gfx950,
    )

    shape = dataclasses.replace(SHAPE, rows=rows)

    weights, experts = _weights(rank, device)
    # The MoE layer stores its banks in the layout the tail reads.
    moe = torch.nn.Module()
    moe.w13_weight = torch.nn.Parameter(experts["w13_weight"], requires_grad=False)
    moe.w13_weight_scale = torch.nn.Parameter(experts["w13_scale"], requires_grad=False)
    moe.w2_weight = torch.nn.Parameter(experts["w2_weight"], requires_grad=False)
    moe.w2_weight_scale = torch.nn.Parameter(experts["w2_scale"], requires_grad=False)
    plan = moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=SHAPE.intermediate_size // SHAPE.tp_size,
        internal_activation_dtype="input",
        expert_layout=kimi_k3_moe_tail_expert_layout(shape),
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )
    moe_process_weights(plan, moe)

    tail = prepare_kimi_k3_moe_tail(
        shape, rank=rank, cpu_group=dist.group.WORLD, device=device
    )
    # Start one step before the step count wraps, so the two steps run across
    # the wrap.
    tail.workspace.steps[0] = (1 << 22) - 1
    eps = {"mlp_res": EPS, "post_attention": EPS, "latent": EPS}
    if writes_next_mix:
        eps["next_mix"] = EPS
    # The same layer as two model layers, one a step; the second must not
    # compile again for its own slot or a new step.
    layers = [
        pack_kimi_k3_moe_tail_layer(
            tail,
            layer_index=index,
            starts_step=True,
            num_blocks=num_blocks,
            restarts_prefix=restarts_prefix,
            eps=eps,
            tensors={
                **weights,
                "w13": moe.w13_weight,
                "w13_scale": moe.w13_weight_scale,
                "w2": moe.w2_weight,
                "w2_scale": moe.w2_weight_scale,
            },
            writes_next_mix=writes_next_mix,
        )
        for index in (layer_index, layer_index + 1)
    ]
    # The next layer's attention-side AttnRes weights.
    generator = torch.Generator(device=device).manual_seed(5000)
    hidden = SHAPE.hidden_size
    score_weight = (
        (torch.rand(hidden, device=device, generator=generator) + 0.5)
        * torch.randn(hidden, device=device, generator=generator)
        / 64
    ).bfloat16()
    input_norm = (
        torch.rand(hidden, device=device, generator=generator) + 0.5
    ).bfloat16()

    for step, layer in enumerate(layers):
        attn_output, stream, blocks, attn_norm = _inputs(
            rank, device, step, rows, num_blocks, restarts_prefix, gated_norm
        )
        next_mix = None
        if writes_next_mix:
            partial = (
                torch.empty(rows, device=device),
                torch.empty(rows, device=device),
                torch.empty((rows, hidden), device=device),
            )
            # Of the snapshots, as the next layer would mix them.
            attnres_partial(blocks[:num_blocks], score_weight, EPS, partial)
            next_mix = KimiK3NextMix(
                score_weight, input_norm, partial, torch.empty_like(stream)
            )
        incoming = None if restarts_prefix else stream
        with (
            assert_no_triton_compile(gluon_kimi_k3_moe_tail_gfx950)
            if step
            else nullcontext()
        ):
            actual = kimi_k3_moe_tail(
                layer,
                attn_output,
                incoming,
                blocks,
                torch.empty_like(stream),
                next_mix=next_mix,
                attn_norm=attn_norm,
            )
        if attn_norm is not None:
            # The regular decode applies the norm in a launch of its own; the
            # tail on its rows, here a step later, must give the same bits.
            # The other cases check that tail against the regular ops.
            regular = kimi_k3_moe_tail(
                layer,
                apply_kda_output_norm(attn_output, attn_norm),
                incoming,
                blocks,
                torch.empty_like(stream),
                next_mix=next_mix,
            )
            torch.testing.assert_close(
                actual.view(torch.int16), regular.view(torch.int16), atol=0, rtol=0
            )
            continue
        tail_hidden = _tail_mlp_input(layer)
        prefix, hidden_rows = _regular_seam(
            weights, attn_output, incoming, blocks, num_blocks, group
        )
        zeros = torch.zeros_like(stream)
        every_row = torch.ones(rows, dtype=torch.bool, device=device)
        _assert_rows_close(tail_hidden, hidden_rows, zeros, every_row)
        # The routed experts quantize their inputs to four bits, so a rounding
        # step in one input row flips quantized values and moves the row's
        # output by percents: the MoE half runs on the tail's own input rows,
        # which the line above checked.
        expected, logits = _regular_moe(
            weights, experts, tail_hidden, prefix, group, swap_last_route=False
        )
        swapped, _ = _regular_moe(
            weights, experts, tail_hidden, prefix, group, swap_last_route=True
        )
        clear = _clear_rows(logits, weights["router_bias"])
        assert clear.any()
        base = zeros if incoming is None else incoming
        _assert_rows_close(actual, expected, base, clear)
        _assert_separated(swapped, expected, base, clear)
        if next_mix is not None:
            expected_next = attnres_combine(
                actual, score_weight, input_norm, EPS, next_mix.partial, zeros.clone()
            )
            _assert_rows_close(next_mix.out, expected_next, zeros, every_row)
    assert int(tail.workspace.steps[0]) == (3 if gated_norm else 1)
    # Peers read this rank's memory until their last launch ends.
    torch.cuda.synchronize()
    dist.barrier()


def _worker(
    rank: int,
    port: int,
    layer_index: int,
    num_blocks: int,
    restarts_prefix: bool,
    writes_next_mix: bool,
    rows: int,
    gated_norm: bool,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Gloo exchanges the tail's buffer addresses; RCCL runs the reference's
    # all-reduces.
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=SHAPE.tp_size,
        timeout=timedelta(seconds=180),
    )
    try:
        group = dist.new_group(backend="nccl", timeout=timedelta(seconds=180))
        dist.barrier(group=group, device_ids=[rank])
        _check(
            rank,
            device,
            group,
            layer_index,
            num_blocks,
            restarts_prefix,
            writes_next_mix,
            rows,
            gated_norm,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    (
        "rows",
        "layer_index",
        "num_blocks",
        "restarts_prefix",
        "writes_next_mix",
        "gated_norm",
    ),
    [
        # Mixes four snapshots and the incoming stream, and writes the next
        # layer's attention input: at one request's decode (1 row) and
        # speculative verify widths (4 and 8 rows).
        (1, 37, 4, False, True, False),
        (4, 37, 4, False, True, False),
        (8, 37, 4, False, True, False),
        # Starts snapshot 4: mixes five snapshots and a prefix that restarts
        # from zero.
        (8, 48, 5, True, False, False),
        # Also applies the KDA layer's gated output norm, which the tail takes
        # at 8 rows only.
        (8, 37, 4, False, True, True),
    ],
    ids=[
        "plain-rows1",
        "plain-rows4",
        "plain-rows8",
        "block_write-rows8",
        "gated_norm-rows8",
    ],
)
def test_kimi_k3_moe_tail_matches_regular_ops(
    rows: int,
    layer_index: int,
    num_blocks: int,
    restarts_prefix: bool,
    writes_next_mix: bool,
    gated_norm: bool,
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _worker,
        args=(
            port,
            layer_index,
            num_blocks,
            restarts_prefix,
            writes_next_mix,
            rows,
            gated_norm,
        ),
        nprocs=SHAPE.tp_size,
        join=True,
    )
