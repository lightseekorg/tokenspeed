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

"""Host side of the fused Muse Spark layer on gfx950: the group's launch
state, a layer's weights read in place, compile-time checks, and launches.

:class:`MuseSparkFusedDecoderLayer` is the ``("decoder_layer", "prepare")``
leaf for the ``muse_spark`` recipe (registered in ``decoder_layer/gluon.py``).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import gluon, triton
from tokenspeed_kernel.ops.attention.prologue.types import RopeStyle
from tokenspeed_kernel.ops.communication._iris import (
    HEAP_HEADROOM_BYTES,
    MIN_HEAP_BYTES,
    IrisGroup,
    get_or_create_context,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.fused_layer import (
    MAX_ROWS,
    MuseNativeIrisState,
)
from tokenspeed_kernel.ops.decoder_layer import PlanKey
from tokenspeed_kernel.ops.decoder_layer._gluon.muse_spark.layer import (
    ATT_BLOCK_M,
    EXCHANGE_WORDS,
    FULL_PROGRAMS,
    HIDDEN,
    KV_HEADS,
    NUM_WARPS,
    PART_ROWS,
    POOLED,
    POOLED_LOCAL,
    Q_HEADS,
    Q_WIDTH,
    QKV_PARTS,
    QKV_WIDTH,
    RECORD_WORDS,
    SLIDING_SPLITS,
    STAMP_STRIDE,
    WORLD_SIZE,
    XQ_ROW_BYTES,
    XS_MBLOCK_STRIDE,
    ModelConstants,
    gluon_decoder_layer_muse_spark_gfx950,
    kernel_constexprs,
    offline_attrs,
    offline_signature,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.native_stream import (
    INTER,
    TAG_WORDS,
    native_moe_weights,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent import (
    ENTRY_WORDS,
    GRID,
    persistent_compile_options,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent.arena import UncachedBuffer
from tokenspeed_kernel_amd.ops.gfx950.persistent.mirror import FailureMirror

__all__ = ["MuseSparkDecoderLayerState", "MuseSparkFusedDecoderLayer"]

# LDS each workgroup reserves (more than half of the CU's 160 KB: one
# workgroup per CU).
_LDS_RESERVE_BYTES = 96 * 1024
_CU_LDS_BYTES = 160 * 1024
# Registers per lane of a wave when 8 waves share a CU (2 per SIMD).
_MAX_VGPRS = 256
# Clock ticks (10 ns) a wait may last before the launch fail-stops: 60 s,
# so a rank stalled between launches (a first compile, a host pause) is not
# taken for a failure.
DEADLINE_TICKS = 6_000_000_000
# Plan fields of the default variant (benchmarks sweep them one at a time).
# The MoE ring is 16 chunks per wave: the kernel's register maximum (set by
# other phases) does not move, and the down units keep >= 15 loads in flight
# per wave instead of draining to 2 at unit ends.
DEFAULT_PLAN_FIELDS = dict(
    ring=16,
    quad_map=0,
    dn_order=0,
    dn_lines=0,
    addr_soffset=0,
    tail_num=1,
    tail_den=2,
    num_stages=2,
    # Two 16 KB QKV tiles per CU in flight: the 128 workgroups streaming
    # theirs while the tail exchanges run stay near 3.3 TB/s, below the
    # ~4.5 TB/s where polls slow down (four tiles would be ~6).
    qkv_window=2,
    # Who merges a sliding block's splits: split 0 ("designated") or every
    # split its share of the rows ("spread").
    att_merge="designated",
)


class MuseSparkDecoderLayerState:
    """The launch state every fused layer of one group shares.

    Holds the entry counters (epochs, role claims), the Iris areas and
    error words (``MuseNativeIrisState``), the uncached exchange (ids, xq
    rows and tags), the host mirror of the failure record, and the
    launch-local scratch (attention partials and records, K-part partials,
    the MoE mids, S rows, h'). One state serves every layer: a layer's
    launch is the group's next epoch, and consecutive launches alternate
    the two-copy areas. The Iris heap must already hold
    :meth:`heap_bytes` spare (``decoder_layer_state_heap_bytes``).

    Args:
        group: The device process group of the 8 ranks.
        rank_in_group: This rank in it.
        max_requests: Requests per launch the state is sized for (1 or 2).
        failure_record_path: The host file mirroring the failure record.
        device: This rank's GPU.
    """

    def __init__(
        self,
        *,
        group,
        rank_in_group: int,
        max_requests: int,
        failure_record_path: str,
        device: torch.device,
    ) -> None:
        if not 1 <= max_requests <= 2:
            raise ValueError(
                f"fused Muse layers serve 1-2 requests, not {max_requests}"
            )
        rows = max_requests * 8
        self.device = device
        self.rank = rank_in_group
        self.max_requests = max_requests
        ctx = get_or_create_context(
            heap_size=max(
                MIN_HEAP_BYTES,
                self.heap_bytes(world_size=WORLD_SIZE, max_requests=max_requests)
                + HEAP_HEADROOM_BYTES,
            )
        )
        peers = IrisGroup(ctx, group, rank_in_group, device)
        if peers.world_size != WORLD_SIZE:
            raise ValueError(f"fused Muse layers need {WORLD_SIZE} ranks")
        self.iris = MuseNativeIrisState(peers)
        self.epochs = torch.zeros(ENTRY_WORDS, dtype=torch.int64, device=device)
        self.claims = torch.zeros(ENTRY_WORDS, dtype=torch.int64, device=device)
        self.arena = torch.zeros(ENTRY_WORDS, dtype=torch.int64, device=device)
        self.mirror = FailureMirror(self.iris.errors, failure_record_path)
        self.exchange = UncachedBuffer(EXCHANGE_WORDS, torch.int32, device)
        self.xq = UncachedBuffer(MAX_ROWS * XQ_ROW_BYTES, torch.uint8, device)
        self.xs = UncachedBuffer(XS_MBLOCK_STRIDE, torch.uint8, device)
        # Attention partials and per-wave int32 records, sized for the
        # larger of the two layouts: a sliding layer's [tokens, heads, 8
        # splits, head dim] and a full layer's balanced items [items,
        # BLOCK_M rows, head dim].
        partial_rows = max(rows * Q_HEADS * SLIDING_SPLITS, FULL_PROGRAMS * ATT_BLOCK_M)
        # Two copies by launch parity (each launch zeroes the other).
        self.att_records = torch.zeros(
            2 * RECORD_WORDS, dtype=torch.int32, device=device
        )
        self.mid_o = torch.empty((partial_rows, 64), dtype=torch.float32, device=device)
        self.mid_lse = torch.empty(partial_rows, dtype=torch.float32, device=device)
        # Tagged K-part partials (tag high): router, pooler, QKV; every
        # launch rewrites all PART_ROWS token slots of each part.
        self.logits_parts = torch.zeros(
            (8, PART_ROWS, 256), dtype=torch.int64, device=device
        )
        self.pooled_parts = torch.zeros(
            (8, PART_ROWS, POOLED_LOCAL), dtype=torch.int64, device=device
        )
        self.qkv_parts = torch.zeros(
            (QKV_PARTS, PART_ROWS, QKV_WIDTH), dtype=torch.int64, device=device
        )
        # Typed stand-ins for operands a variant does not read.
        self.dummies = {
            name: torch.zeros(64, dtype=dtype, device=device)
            for name, dtype in (
                ("bf16", torch.bfloat16),
                ("fp8", torch.float8_e4m3fn),
                ("fp32", torch.float32),
                ("i64", torch.int64),
            )
        }
        routes = rows * 8
        self.mids = torch.zeros(routes * INTER // 2, dtype=torch.uint8, device=device)
        self.mids_scale = torch.zeros(
            routes * INTER // 32 // 4, dtype=torch.int32, device=device
        )
        # The MoE's readiness tags: written on one XCD, polled with scalar
        # loads on any other, so uncached (scalar polls of cached device
        # memory are only good for words written on the same XCD).
        self.moe_tags = UncachedBuffer(TAG_WORDS, torch.int32, device)
        self.s_rows = torch.zeros(
            (MAX_ROWS, POOLED), dtype=torch.bfloat16, device=device
        )
        self.hprime = torch.zeros(
            (MAX_ROWS, HIDDEN), dtype=torch.float32, device=device
        )
        # Clock stamps of the last launch made with stamps (benchmarks).
        self.stamps = torch.zeros(GRID * STAMP_STRIDE, dtype=torch.int64, device=device)
        peers.finish_prepare()

    @staticmethod
    def heap_bytes(*, world_size: int, max_requests: int) -> int:
        """Iris heap bytes the state allocates (the symmetric areas, sized
        for the most rows any launch carries)."""
        if world_size != WORLD_SIZE or not 1 <= max_requests <= 2:
            return 0
        return MuseNativeIrisState.heap_bytes()


def _router_hi_lo(router: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The FP32 router ``[hidden, experts]`` as BF16 ``hi + lo`` rows
    ``[experts, hidden]``: ``hi`` rounds each weight, ``lo`` rounds what
    ``hi`` missed, so ``hi + lo`` keeps ~16 of its 24 significand bits."""
    hi = router.t().contiguous().to(torch.bfloat16)
    lo = (router.t() - hi.float()).to(torch.bfloat16).contiguous()
    return hi, lo


def _bf16_with_offset(weight: torch.Tensor | None, offset: float, width: int, device):
    """``bf16(fp32(weight) + offset)``: the norm multiplier the regular
    kernels form, rounded as they round it; ones for a weightless norm."""
    if weight is None:
        return torch.ones(width, dtype=torch.bfloat16, device=device)
    return (weight.float() + offset).to(torch.bfloat16).contiguous()


def _require(name: str, tensor: torch.Tensor, shape: tuple, dtype: torch.dtype):
    if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.stride(-1) != 1:
        raise ValueError(
            f"fused Muse layer: {name} must be {dtype} {list(shape)} with unit inner "
            f"stride, got {tensor.dtype} {list(tensor.shape)}"
        )
    return tensor


@dataclass(frozen=True)
class _LayerTensors:
    """One layer's weights in the layouts the kernel reads (in place, except
    the router's BF16 hi/lo and the norm multipliers)."""

    o_proj: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    post_attn_norm: torch.Tensor
    post_attn_bf16: torch.Tensor
    pre_ffn_bf16: torch.Tensor
    router_hi: torch.Tensor
    router_lo: torch.Tensor
    router_multiplier: float
    router_bias: torch.Tensor
    pooler: torch.Tensor
    pooled_norm: torch.Tensor
    route_norm: torch.Tensor
    post_expert: torch.Tensor
    alpha2: torch.Tensor
    beta2: torch.Tensor
    post2_bf16: torch.Tensor
    next_pre_bf16: torch.Tensor
    next_qkv: torch.Tensor | None


@functools.cache
def _compile_problems(constexpr_items: tuple) -> tuple[str, ...]:
    """Compile one variant offline (no launch) and check that it keeps one
    resident workgroup per CU; memoized, since every layer of a model
    compiles the same variants."""
    constexprs = dict(constexpr_items)
    fn = gluon_decoder_layer_muse_spark_gfx950
    signature = offline_signature(constexprs)
    attrs = offline_attrs(signature)
    full = {n: signature[n] for n in fn.arg_names if n in signature}
    full.update({n: "constexpr" for n in constexprs})
    compiled = triton.compile(
        gluon._runtime.GluonASTSource(fn, full, constexprs=constexprs, attrs=attrs),
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": NUM_WARPS, **persistent_compile_options()},
    )
    return tuple(check_compiled(compiled))


def check_compiled(compiled) -> list[str]:
    """Why a compiled variant cannot keep one resident workgroup per CU, or
    none: LDS above half the CU's, no scratch, VGPRs + AGPRs within the
    8-wave budget, no device-library calls."""
    reasons = []
    shared = compiled.metadata.shared
    if not _LDS_RESERVE_BYTES <= shared <= _CU_LDS_BYTES:
        reasons.append(f"uses {shared} B of LDS")
    assembly = compiled.asm["amdgcn"]
    scratch = re.findall(r"\.private_segment_fixed_size:\s+(\d+)", assembly)
    if any(int(value) for value in scratch):
        reasons.append("spills to scratch")
    vgprs = re.search(r"\.vgpr_count:\s+(\d+)", assembly)
    agprs = re.search(r"\.agpr_count:\s+(\d+)", assembly)
    total = int(vgprs.group(1)) + (int(agprs.group(1)) if agprs else 0)
    if total > _MAX_VGPRS:
        reasons.append(f"uses {total} VGPRs + AGPRs")
    if re.search(r"\.uses_dynamic_stack:\s+true", assembly):
        reasons.append("calls a function")
    return reasons


class MuseSparkFusedDecoderLayer:
    """One Muse Spark decoder layer (and the next layer's QKV projection and
    prologue) as one persistent gfx950 launch per decode step.

    Args:
        plan_keys: The plan variants this layer serves; all compile here.
        weights: The layer's tensors (``MuseSparkDecoderLayer.fused_weights``),
            read in place: ``attention`` (dict: ``scale``, ``window_left``,
            ``gate_eps``), ``o_proj``, the norms as ``(weight, eps,
            weight_offset)`` (``post_attention_norm``,
            ``pre_feedforward_norm``, ``pooled_norm``, ``post_expert_norm``,
            ``post_feedforward_norm``, ``next_input_norm``), the residual
            gates as ``(alpha, beta)`` (``post_attention_gate``,
            ``post_feedforward_gate``), ``router`` (FP32 ``[hidden,
            experts]``), ``router_multiplier``, ``router_bias``,
            ``routed_scaling_factor``, ``pooler`` and ``post_expert_proj``
            (replicated, with ``pooler_replicated`` / ``post_expert_replicated``),
            ``experts`` (the MoE layer), ``next_qkv_proj``.
        state: The group's :class:`MuseSparkDecoderLayerState`.
    """

    def __init__(
        self,
        *,
        plan_keys,
        weights: dict,
        state: MuseSparkDecoderLayerState,
        plan_fields: dict,
        stamps: bool,
        deadline_ticks: int,
    ):
        """``plan_fields``: :data:`DEFAULT_PLAN_FIELDS`' keys (serving passes
        the defaults; benchmarks vary them); ``stamps``: store per-workgroup
        clock stamps into ``state.stamps``; ``deadline_ticks``: how long a
        wait may last, in 10 ns clock ticks (:data:`DEADLINE_TICKS` when
        serving; a test stops a withheld launch sooner)."""
        if set(plan_fields) != set(DEFAULT_PLAN_FIELDS):
            raise ValueError(f"plan fields must be {sorted(DEFAULT_PLAN_FIELDS)}")
        if deadline_ticks <= 0:
            raise ValueError(f"deadline_ticks must be positive, got {deadline_ticks}")
        self.plan_fields = dict(plan_fields)
        self.stamps = stamps
        self.deadline_ticks = deadline_ticks
        self.state = state
        self.plan_keys = tuple(plan_keys)
        kinds = {(key.layer_kind, key.next_kind) for key in self.plan_keys}
        if len(kinds) != 1:
            raise ValueError("a fused layer's plan keys share its layer and next kinds")
        self.layer_kind, self.next_kind = kinds.pop()
        self.declines: dict[int, list[str]] = {}
        # Variants whose launched binary passed the residency check, and the
        # page-table stride every launch must keep (a compile-time value).
        self.checked: set[tuple] = set()
        self.page_table_stride: int | None = None
        self.tensors, self.model, problems = self._read_weights(weights, state.rank)
        # The lane-major gate/up scales cost a 32 MiB copy per layer: off
        # until a benchmark shows the win is worth the memory.
        self.moe = native_moe_weights(weights["experts"], scale_lanes=False)
        for key in self.plan_keys:
            reasons = problems + self._compile(key)
            if reasons:
                self.declines.setdefault(key.requests, []).extend(reasons)

    def _read_weights(self, weights: dict, rank: int):
        def norm(name):
            weight, eps, offset = weights[name]
            return weight, float(eps), float(offset)

        problems = []
        if not weights["pooler_replicated"]:
            problems.append("pre_expert_proj is not replicated on this rank")
        if not weights["post_expert_replicated"]:
            problems.append("post_expert_proj is not replicated on this rank")
        attention = weights["attention"]
        device = weights["o_proj"].device
        post_attn_w, post_attn_eps, post_attn_offset = norm("post_attention_norm")
        pre_ffn_w, pre_ffn_eps, pre_ffn_offset = norm("pre_feedforward_norm")
        pooled_w, pooled_eps, pooled_offset = norm("pooled_norm")
        route_w, route_eps, route_offset = norm("post_expert_norm")
        post2_w, post2_eps, post2_offset = norm("post_feedforward_norm")
        next_w, next_eps, next_offset = norm("next_input_norm")
        alpha, beta = weights["post_attention_gate"]
        alpha2, beta2 = weights["post_feedforward_gate"]
        # The router as BF16 hi + lo rows [experts, hidden] (a one-time copy).
        router = _require("router", weights["router"], (HIDDEN, 256), torch.float32)
        router_hi, router_lo = _router_hi_lo(router)
        pooler = _require(
            "pooler", weights["pooler"], (POOLED, HIDDEN), torch.bfloat16
        )[rank * POOLED_LOCAL : (rank + 1) * POOLED_LOCAL]
        next_qkv = weights["next_qkv_proj"]
        if self.next_kind != "regular":
            _require("next_qkv_proj", next_qkv, (QKV_WIDTH, HIDDEN), torch.bfloat16)
        tensors = _LayerTensors(
            o_proj=_require("o_proj", weights["o_proj"], (HIDDEN, Q_WIDTH), torch.bfloat16),
            alpha=_require("alpha", alpha, (HIDDEN,), torch.float32),
            beta=_require("beta", beta, (HIDDEN,), torch.float32),
            # A weightless post-attention norm (Muse) passes the multiplier
            # vector of ones and no weight to OWN1.
            # Without a weight (Muse), OWN1 reads none: the BF16 multiplier
            # vector of ones stands in for the pointer.
            post_attn_norm=(
                _bf16_with_offset(None, 0.0, HIDDEN, device)
                if post_attn_w is None
                else _require("post_attention_norm", post_attn_w, (HIDDEN,), torch.bfloat16)
            ),
            post_attn_bf16=_bf16_with_offset(post_attn_w, post_attn_offset, HIDDEN, device),
            pre_ffn_bf16=_bf16_with_offset(pre_ffn_w, pre_ffn_offset, HIDDEN, device),
            router_hi=router_hi,
            router_lo=router_lo,
            router_multiplier=float(weights["router_multiplier"]),
            router_bias=_require("router_bias", weights["router_bias"], (256,), torch.float32),
            pooler=pooler,
            pooled_norm=_require("pooled_norm", pooled_w, (POOLED,), torch.bfloat16),
            route_norm=_require("post_expert_norm", route_w, (POOLED,), torch.bfloat16),
            post_expert=_require(
                "post_expert_proj", weights["post_expert_proj"], (HIDDEN, POOLED),
                torch.bfloat16,
            ),
            alpha2=_require("ffn alpha", alpha2, (HIDDEN,), torch.float32),
            beta2=_require("ffn beta", beta2, (HIDDEN,), torch.float32),
            post2_bf16=_bf16_with_offset(post2_w, post2_offset, HIDDEN, device),
            next_pre_bf16=_bf16_with_offset(next_w, next_offset, HIDDEN, device),
            next_qkv=next_qkv,
        )  # fmt: skip
        model = dict(
            sm_scale=float(attention["scale"]),
            window_left=int(attention["window_left"]),
            gate_eps=float(attention["gate_eps"]),
            post_attn_eps=post_attn_eps,
            post_attn_offset=post_attn_offset,
            post_attn_has_weight=post_attn_w is not None,
            pre_ffn_eps=pre_ffn_eps,
            pooled_eps=pooled_eps,
            pooled_offset=pooled_offset,
            route_eps=route_eps,
            route_offset=route_offset,
            routed_scaling_factor=float(weights["routed_scaling_factor"]),
            post2_eps=post2_eps,
            pre2_eps=next_eps,
        )
        return tensors, model, problems

    def _constexprs(
        self,
        key,
        *,
        page_size,
        page_table_stride,
        qk_eps,
        qk_offset,
        kv_scaled,
        next_rotary,
    ):
        model = ModelConstants(
            page_size=page_size,
            page_table_stride=page_table_stride,
            qk_eps=qk_eps,
            qk_offset=qk_offset,
            swiglu_alpha=1.0,
            swiglu_limit=0.0,
            swiglu_beta=0.0,
            **self.model,
        )
        return kernel_constexprs(
            key, model, rank=self.state.rank, peer_cache=".wt",
            scale_lanes=int(self.moe.s13_lanes is not None), n_phys1=self.moe.n_phys1,
            n_phys2=self.moe.n_phys2, kv_scaled=kv_scaled, next_rotary=next_rotary,
            stamps=self.stamps, **self.plan_fields,
        )  # fmt: skip

    def _compile(self, key: PlanKey) -> list[str]:
        """Compile ``key`` offline now and check its residency; the reasons
        this rank cannot serve it, or none."""
        properties = torch.cuda.get_device_properties(self.state.device)
        if properties.multi_processor_count != GRID:
            return [
                f"the device has {properties.multi_processor_count} CUs; a fused "
                f"launch needs exactly {GRID}"
            ]
        # An early check with the usual page geometry, which the first
        # launch specializes; that launch checks its own binary again.
        constexprs = self._constexprs(
            key, page_size=64, page_table_stride=4096, qk_eps=1e-6, qk_offset=0.0,
            kv_scaled=False, next_rotary=key.next_kind == "sliding",
        )  # fmt: skip
        return [
            f"{key}: {reason}"
            for reason in _compile_problems(tuple(sorted(constexprs.items())))
        ]

    def decline_reasons(self, requests: int) -> list[str]:
        if not any(key.requests == requests for key in self.plan_keys):
            return [f"no plan variant for {requests} requests"]
        return list(self.declines.get(requests, []))

    def _key(self, rows: int) -> PlanKey:
        if rows % 8:
            raise ValueError(f"fused Muse layer: {rows} rows is not whole requests")
        key = PlanKey(rows // 8, self.layer_kind, self.next_kind)
        if key not in self.plan_keys:
            raise ValueError(f"fused Muse layer: {key} was not prepared")
        return key

    def _launch(self, key, q, gate, residual, positions, cache, prologue, normed_out):
        state, t, moe = self.state, self.tensors, self.moe
        rows = q.shape[0]
        if key.layer_kind == "full" and cache.split_schedule is None:
            raise ValueError("a full layer's decode view carries no split schedule")
        if (
            prologue is not None
            and prologue.rotary is not None
            and (
                prologue.rotary.style != RopeStyle.GPTJ
                or prologue.rotary.mrope is not None
            )
        ):
            raise ValueError(
                "the fused prologue rotates interleaved (GPT-J) pairs only, without mrope"
            )
        scales = None if prologue is None else prologue.cache.scales
        # The page-table width is fixed once the decode metadata exists (the
        # graph buffers); the kernel specializes on its stride.
        stride = int(cache.page_table.stride(0))
        if self.page_table_stride is None:
            self.page_table_stride = stride
        elif stride != self.page_table_stride:
            raise ValueError(
                f"page table stride changed from {self.page_table_stride} to {stride}"
            )
        constexprs = self._constexprs(
            key,
            page_size=int(cache.k_cache.shape[1]),
            page_table_stride=stride,
            qk_eps=0.0 if prologue is None else float(prologue.norm.eps),
            qk_offset=0.0 if prologue is None else float(prologue.norm.weight_offset),
            kv_scaled=scales is not None,
            next_rotary=prologue is not None and prologue.rotary is not None,
        )
        if prologue is not None:
            next_q = torch.empty(
                (rows, Q_HEADS, 64), dtype=prologue.query_dtype, device=q.device
            )
            next_gate = torch.empty(
                (rows, Q_WIDTH), dtype=torch.bfloat16, device=q.device
            )
            rotary = prologue.rotary
            next_args = (
                t.next_qkv, state.qkv_parts, next_q, next_gate, prologue.norm.q_weight,
                prologue.norm.k_weight,
                residual if rotary is None else rotary.cos_sin_cache,
                0 if rotary is None else rotary.cos_sin_cache.stride(0),
                positions if rotary is None else rotary.positions,
                prologue.cache.k_cache, prologue.cache.v_cache, prologue.cache.slots,
                prologue.cache.slots.shape[0],
                1.0 if scales is None else 1.0 / scales.k,
                1.0 if scales is None else 1.0 / scales.v,
                residual,
            )  # fmt: skip
        else:
            next_q = next_gate = None
            # The next layer is regular: the QKV and prologue operands are
            # unused stand-ins of their types.
            d = state.dummies
            next_args = (
                d["bf16"], state.qkv_parts, d["fp8"], d["bf16"], d["bf16"], d["bf16"],
                d["fp32"], 64, positions, d["fp8"], d["fp8"], d["i64"], 0, 1.0, 1.0,
                normed_out,
            )  # fmt: skip
        compiled = gluon_decoder_layer_muse_spark_gfx950[(GRID,)](
            state.epochs, state.claims, state.iris.errors, state.arena, state.exchange,
            state.xq, state.xs, self.deadline_ticks, 0,
            *state.iris.view_args(),
            q, gate, gate.stride(0), cache.k_cache, cache.v_cache, cache.page_table,
            cache.seq_lens,
            cache.page_table if key.layer_kind == "sliding" else cache.split_schedule,
            state.mid_o, state.mid_lse, state.att_records,
            t.o_proj, residual, t.alpha, t.beta, t.post_attn_norm, t.post_attn_bf16,
            t.pre_ffn_bf16, t.router_hi, t.router_lo, t.router_multiplier,
            t.router_bias, state.logits_parts, t.pooler, state.pooled_parts,
            t.pooled_norm,
            moe.w13, moe.s13, moe.s13 if moe.s13_lanes is None else moe.s13_lanes,
            moe.w2, moe.s2, state.mids, state.mids_scale, state.moe_tags,
            moe.stride_w13, moe.stride_s13, moe.stride_s13lin, moe.stride_s13nb,
            moe.stride_w2, moe.stride_s2, moe.stride_s2lin, moe.stride_s2nb,
            t.route_norm, state.s_rows, state.hprime, t.post_expert, t.alpha2,
            t.beta2, t.post2_bf16, t.next_pre_bf16,
            *next_args,
            state.stamps,
            **constexprs,
            num_warps=NUM_WARPS,
            **persistent_compile_options(),
        )  # fmt: skip
        variant = tuple(sorted(constexprs.items()))
        if variant not in self.checked:
            # The first launch of a variant compiled its binary (graph
            # capture's warm-up, before serving): check that one too.
            problems = check_compiled(compiled)
            if problems:
                raise RuntimeError(f"fused Muse layer {key}: {'; '.join(problems)}")
            self.checked.add(variant)
        return next_q, next_gate

    def forward(self, prepared_query, residual, *, positions, cache, next_prologue):
        q, gate = prepared_query
        key = self._key(q.shape[0])
        if key.next_kind == "regular":
            raise ValueError("this layer's next layer is regular: use forward_last")
        next_q, next_gate = self._launch(
            key, q, gate, residual, positions, cache, next_prologue, None
        )
        return (next_q, next_gate), residual

    def forward_last(self, prepared_query, residual, *, positions, cache, next_norm):
        q, gate = prepared_query
        key = self._key(q.shape[0])
        if key.next_kind != "regular":
            raise ValueError("this layer's next layer is fused: use forward")
        weight, eps, offset = next_norm
        if float(eps) != self.model["pre2_eps"] or not torch.equal(
            _bf16_with_offset(weight, float(offset), HIDDEN, q.device),
            self.tensors.next_pre_bf16,
        ):
            raise ValueError("next_norm differs from the prepared next input norm")
        normed = torch.empty(
            (q.shape[0], HIDDEN), dtype=torch.bfloat16, device=q.device
        )
        self._launch(key, q, gate, residual, positions, cache, None, normed)
        return normed, residual
