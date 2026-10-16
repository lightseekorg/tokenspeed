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

"""Beam's grouped sigmoid route in one Triton launch (``triton_router_route_lanepar``).

One program per row, on one warp, loads the row's FP32 logits (eight per lane) and
the FP32 correction bias, computes ``s = div.rn.f32(1, add.rn.f32(1,
__nv_expf(-logit)))`` -- Torch's FP32 ``sigmoid_kernel_cuda`` recipe: libdevice
``exp`` plus inline PTX ``.rn`` add and div, no FTZ (:data:`RECIPE`; the helpers
:func:`_sigmoid_rn`, :func:`_add_rn_f32`, :func:`_div_rn_f32`) -- stores ``s`` to
``scores_ptr`` and adds the bias with one ``add.rn.f32`` (Torch's add). The radix key
is :func:`_ordered_key_tl`; the jit helpers, constexpr constants, the launch options
and the admission rule ``grouped_select_rejection`` of
:mod:`tokenspeed_kernel.ops.moe._triton.grouped_select` are that module's objects,
imported by identity, never copied. The tie contract is the pinned Torch
dispatch's: no stable tie-break is added; :data:`PINNED_TORCH` is the same
recorded, not enforced, Torch.

The Torch pin (empirical, FP32 CUDA): on the pinned Torch the recipe reproduces
``torch.sigmoid`` bit for bit -- a dense binade-stratified sweep of 35,127,296
values and the enumerated edge corpus of 2,278,482 values (``+-0``, subnormals,
``+-inf``, NaN payloads, the two exhaustive ``expf`` overflow windows,
``+-k ln 2``) showed 0 mismatches under the NaN rule (equal ``isnan`` masks,
byte-equal non-NaN elements). The GPU test
``test/nvidia/ops/moe/test_router_select_cuda.py`` re-runs the edge corpus through
:func:`_sigmoid_rn` and fails on a mismatch -- it never skips -- so a Torch upgrade
must re-run it before the route is served.

Selection (``_router_select_lanepar_kernel``, launched under
``_LANEPAR_LAUNCH_OPTIONS``: the product launch options with ``num_warps=1``, and
as a programmatic dependent launch where ``pdl_enabled()``: the program waits on
its predecessor before its first load and releases its dependents): the
group score takes the two largest keys of each group with multiplicity in one tuple
reduction (:func:`_top2_combine`) and adds their values, recovered exactly by
:func:`_key_value` (the inverse of the radix map), with the one FP32 add. Each k-th
key and its fill set come from :func:`_extract_top`: ``K`` rounds that remove the
largest remaining key at its lowest index, so the ``K``-th round's key is the K-th
largest key and the removed indices are exactly the gatherTopK fill's (every key
above the k-th, then the lowest indices equal to it); :func:`_fill_rank` puts them
in the fill's slots with a 32-lane rank permutation. The kept groups are the group
extraction's removed indices. Both sorting networks are :func:`_sort_desc_trim`:
the ``_swap_desc`` stages of the 15-stage network (:data:`NETWORK`) with a stride
below ``K`` (9 of 15 for the four groups, 12 of 15 for the eight experts); every
dropped stage is the identity on the fill pattern (lanes ``[0, K)`` valid, the
others the identical invalid triple), so the exchange decisions and the output
order are the full network's for every input, NaN and ties included: the
selection of ``torch_router_select``.

The route (``triton_router_route_lanepar``): the same kernel under ``ROUTE``
gathers the eight selected unbiased scores from its registers and performs the
route epilogue's operations on them in Torch's order and rounding (the Torch CUDA
K8 tree in ``add.rn.f32``, ``+ 1e-20``, ``div.rn.f32``, one ``mul.rn.f32`` by the
scale's FP32 rounding; :mod:`tokenspeed_kernel.ops.moe._triton.route_epilogue`),
storing the FP32 weights and int32 ids. The outputs are ``torch_grouped_route``'s
(:mod:`tokenspeed_kernel.ops.moe.grouped_route`) byte for byte, which the CUDA
tests check. Its host guard :func:`router_route_rejection` admits Beam's routing
shape only -- 256 experts in 8 groups, group score over the top 2, 4 kept groups,
top-8, a contiguous FP32 ``[rows, 256]`` CUDA tensor with a positive row count and
a contiguous FP32 ``[256]`` bias on the same device -- and a scale the kernel can
hold as an FP32 constant; Beam calls the kernel where the guard admits the call on
NVIDIA SM100 to SM103 and runs the Torch statements otherwise.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.ops.moe._triton.grouped_select import (
    _EXPERT_CHUNK,
    _GROUP_CHUNK,
    _GROUP_SIZE,
    _INT32_MIN,
    _KEY_NAN,
    _KEY_NEG_INF,
    _LAUNCH_OPTIONS,
    _NUM_EXPERTS,
    _NUM_GROUPS,
    _TOPK,
    _TOPK_GROUPS,
    _ordered_key_tl,
    _swap_desc,
    grouped_select_rejection,
)
from tokenspeed_kernel.ops.moe._triton.route_epilogue import (
    _add_rn,
    _mul_rn,
    route_scale_fp32,
    route_scale_rejection,
)
from tokenspeed_kernel.platform import pdl_enabled

__all__ = [
    "RECIPE",
    "router_route_rejection",
    "router_select_rejection",
    "triton_router_route_lanepar",
]

# The recipe of the fold: libdevice ``exp`` and the inline ``.rn`` forms, FTZ off
# (the product launch options). Torch's ``sigmoid_kernel_cuda`` computes
# ``1 / (1 + expf(-x))`` with IEEE add and div; Torch's add is one ``add.rn.f32``.
RECIPE = "s = div.rn.f32(1, add.rn.f32(1, __nv_expf(-x))); x = add.rn.f32(s, bias)"


# ---------------------------------------------------------------------------------------
# The recipe helpers (the only new arithmetic of the fold)
# ---------------------------------------------------------------------------------------


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sigmoid_rn(t):
    # Torch's FP32 sigmoid_kernel_cuda: 1 / (1 + expf(-t)), IEEE .rn add and div, libdevice
    # expf, no FTZ.
    one = tl.full(t.shape, 1.0, tl.float32)
    return _div_rn_f32(one, _add_rn_f32(one, libdevice.exp(-t)))


# ---------------------------------------------------------------------------------------
# Host: admission and launch
# ---------------------------------------------------------------------------------------


def router_select_rejection(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_groups: int,
    group_topk: int,
    topk_groups: int,
    topk: int,
) -> str:
    """Why the lane-parallel selection cannot serve this call; empty when admitted.

    Metadata only (no device read), the first failing clause in this order:
    the bias FP32, of shape ``(256,)``, contiguous (the kernel reads
    ``bias_ptr + lane``); the logits by the product rule
    :func:`grouped_select_rejection` (2-D with a positive row count, FP32, 256
    experts, the four routing parameters equal to the kernel's constexprs,
    contiguous, a CUDA tensor -- its device clause last, so shape and layout
    findings are reported on any device; the reason is prefixed ``logits:``);
    the bias on the logits device. The bias metadata clauses come first so a
    CPU test reaches them.

    Args:
        gating_output: FP32 router logits the kernel scores and selects from.
        correction_bias: FP32 selection bias added to the sigmoid.
        num_groups: Requested group count.
        group_topk: Requested group-score term count.
        topk_groups: Requested kept-group count.
        topk: Requested expert count.

    Returns:
        The rejection reason, or ``""`` when the kernel admits the call.
    """
    if correction_bias.dtype is not torch.float32:
        return f"bias must be {torch.float32}, got {correction_bias.dtype}"
    if tuple(correction_bias.shape) != (_NUM_EXPERTS,):
        return (
            f"bias must have shape ({_NUM_EXPERTS},), got "
            f"{tuple(correction_bias.shape)}"
        )
    if not correction_bias.is_contiguous():
        return (
            f"bias must be contiguous (the kernel reads bias_ptr + lane), got strides "
            f"{tuple(correction_bias.stride())}"
        )
    reason = grouped_select_rejection(
        gating_output, num_groups, group_topk, topk_groups, topk
    )
    if reason:
        return f"logits: {reason}"
    if correction_bias.device != gating_output.device:
        return (
            f"bias must be on the logits device {gating_output.device}, got "
            f"{correction_bias.device}"
        )
    return ""


# ---------------------------------------------------------------------------------------
# The lane-parallel kernel: one row per program on one warp
# ---------------------------------------------------------------------------------------

# The product launch options with one warp per program: the row's 256 keys sit eight
# per lane of one warp, so no CTA barrier or shared-memory layout conversion is needed.
# ``launch_pdl`` follows ``pdl_enabled()`` at launch (the kernel waits on its
# predecessor before its first load).
_LANEPAR_LAUNCH_OPTIONS = {**_LAUNCH_OPTIONS, "num_warps": 1}


@triton.jit
def _key_value(key):
    # The inverse of _ordered_key_tl (the map bits ^ ((bits >> 31) & 0x7FFFFFFF) is an
    # involution): the exact FP32 bits of every non-NaN key; KEY_NAN gives 0x7FFFFFFF, a NaN.
    return (key ^ ((key >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)


@triton.jit
def _top2_combine(a1, a2, b1, b2):
    # The two largest keys, with multiplicity, of the union of two multisets given as
    # (largest, second largest); associative and commutative.
    return tl.maximum(a1, b1), tl.maximum(tl.minimum(a1, b1), tl.maximum(a2, b2))


@triton.jit
def _extract_top(keys, idx, K: tl.constexpr, N: tl.constexpr, INT32_MIN: tl.constexpr):
    # K rounds: the largest remaining key, its lowest index, that element removed. Lane r
    # of (mv, iv) holds round r's (key, index); kth is round K-1's key; taken marks the K
    # removed indices. INT32_MIN is below every radix key, so a removed element never
    # returns.
    lane32 = tl.arange(0, 32)
    mv = tl.zeros((32,), dtype=tl.int32)
    iv = tl.zeros((32,), dtype=tl.int32)
    taken = idx < 0
    work = keys
    kth = tl.max(work, axis=0)
    for r in tl.static_range(K):
        kth = tl.max(work, axis=0)
        first = tl.min(tl.where(work == kth, idx, N), axis=0)
        hit = idx == first
        work = tl.where(hit, INT32_MIN, work)
        taken = taken | hit
        mv = tl.where(lane32 == r, kth, mv)
        iv = tl.where(lane32 == r, first, iv)
    return mv, iv, kth, taken


@triton.jit
def _fill_rank(mv, iv, kth, K: tl.constexpr, N: tl.constexpr):
    # The gatherTopK fill of the K extracted elements into the 32-lane sort input: fill
    # priority q = index (key above the k-th) or N + index (key equal to it); slot = rank
    # of q among the K; lane s pulls the element of rank s. Values are the keys' FP32
    # values; slots K..31 are the product's invalid lanes (0.0, order 0, valid False).
    lane32 = tl.arange(0, 32)
    valid = lane32 < K
    q = tl.where(valid, tl.where(mv == kth, N + iv, iv), 2 * N)
    rank = tl.zeros((32,), dtype=tl.int32)
    for r in tl.static_range(K):
        q_r = tl.gather(q, tl.full((32,), r, tl.int32), axis=0)
        rank += (q_r < q).to(tl.int32)
    src = tl.zeros((32,), dtype=tl.int32)
    for r in tl.static_range(K):
        rank_r = tl.gather(rank, tl.full((32,), r, tl.int32), axis=0)
        src = tl.where(rank_r == lane32, r, src)
    order = tl.where(valid, tl.gather(iv, src, axis=0), 0)
    values = tl.where(valid, _key_value(tl.gather(mv, src, axis=0)), 0.0)
    return values, order, valid


@triton.jit
def _sort_desc_trim(keys, order, valid, K: tl.constexpr):
    # _sort_desc_tl restricted to the NETWORK stages with STRIDE < K, in NETWORK order;
    # each stage is the product _swap_desc. The dropped stages are identities on the fill
    # pattern (lanes [0, K) valid, lanes [K, 32) the identical invalid triple): a pair
    # inside an aligned K-block is valid-valid or invalid-invalid, and a pair across it has
    # its valid lane low with direction False, so it never exchanges.
    tl.static_assert((K == 4) or (K == 8))
    keys, order, valid = _swap_desc(keys, order, valid, 1, 2)
    keys, order, valid = _swap_desc(keys, order, valid, 2, 4)
    keys, order, valid = _swap_desc(keys, order, valid, 1, 4)
    if K > 4:
        keys, order, valid = _swap_desc(keys, order, valid, 4, 8)
    keys, order, valid = _swap_desc(keys, order, valid, 2, 8)
    keys, order, valid = _swap_desc(keys, order, valid, 1, 8)
    if K > 4:
        keys, order, valid = _swap_desc(keys, order, valid, 4, 16)
    keys, order, valid = _swap_desc(keys, order, valid, 2, 16)
    keys, order, valid = _swap_desc(keys, order, valid, 1, 16)
    if K > 4:
        keys, order, valid = _swap_desc(keys, order, valid, 4, 0)
    keys, order, valid = _swap_desc(keys, order, valid, 2, 0)
    keys, order, valid = _swap_desc(keys, order, valid, 1, 0)
    return keys, order, valid


@triton.jit
def _router_select_lanepar_kernel(
    group_ids_ptr,
    ids_ptr,
    NUM_EXPERTS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOPK_GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    GROUP_CHUNK: tl.constexpr,
    EXPERT_CHUNK: tl.constexpr,
    INT32_MIN: tl.constexpr,
    KEY_NEG_INF: tl.constexpr,
    KEY_NAN: tl.constexpr,
    logits_ptr,
    bias_ptr,
    scores_ptr,
    ENABLE_PDL: tl.constexpr,
    weights_ptr,
    route_ids_ptr,
    ROUTE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    SCALE: tl.constexpr,
):
    # One program per row on one warp: torch_router_select's selection, lane-parallel;
    # under ROUTE the route epilogue's outputs (see the route block).
    if ENABLE_PDL:
        # The router GEMV's logits are the predecessor's stores; the route
        # epilogue after this kernel waits on it the same way.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    lane = tl.arange(0, NUM_EXPERTS)
    group = tl.arange(0, NUM_GROUPS)
    # fold: the recipe helpers; the bias is loaded after the scores store
    logit = tl.load(logits_ptr + row * NUM_EXPERTS + lane)
    s = _sigmoid_rn(logit)
    if not ROUTE:
        tl.store(scores_ptr + row * NUM_EXPERTS + lane, s)
    x = _add_rn_f32(s, tl.load(bias_ptr + lane))
    # A. radix keys of the choices
    key = _ordered_key_tl(x, KEY_NAN)
    # B. group score: the two largest keys of each group (with multiplicity), one FP32 add
    # of their values
    kg = tl.reshape(key, (NUM_GROUPS, GROUP_SIZE))
    m1, m2 = tl.reduce((kg, tl.full(kg.shape, INT32_MIN, tl.int32)), 1, _top2_combine)
    g = _key_value(m1) + _key_value(m2)
    gkey = _ordered_key_tl(g, KEY_NAN)
    # E. group stage: extraction, fill, trimmed network -> group_ids; the kept flag per group
    gmv, giv, gkth, kept = _extract_top(gkey, group, TOPK_GROUPS, NUM_GROUPS, INT32_MIN)
    gkeys, gorder, gvalid = _fill_rank(gmv, giv, gkth, TOPK_GROUPS, NUM_GROUPS)
    gkeys, gorder, gvalid = _sort_desc_trim(gkeys, gorder, gvalid, TOPK_GROUPS)
    group_ids = tl.gather(gorder, tl.arange(0, TOPK_GROUPS), axis=0)
    # F. mask: experts of the kept groups keep their key, the others KEY_NEG_INF
    mkey = tl.reshape(tl.where(kept[:, None], kg, KEY_NEG_INF), (NUM_EXPERTS,))
    # G. expert stage: extraction, fill, trimmed network -> ids
    emv, eiv, ekth, _ = _extract_top(mkey, lane, TOPK, NUM_EXPERTS, INT32_MIN)
    ekeys, eorder, evalid = _fill_rank(emv, eiv, ekth, TOPK, NUM_EXPERTS)
    ekeys, eorder, evalid = _sort_desc_trim(ekeys, eorder, evalid, TOPK)
    ids = tl.gather(eorder, tl.arange(0, TOPK), axis=0)
    if ROUTE:
        # H. the route epilogue on the row's registers, torch_route_epilogue's
        # arithmetic: the eight selected unbiased scores in slot order; under NORMALIZE
        # divided by their pinned Torch CUDA K8 sum ((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))
        # plus 1e-20 (add.rn, div.rn); with a SCALE one mul.rn by it; the FP32 weights
        # and the int32 ids.
        slots = tl.arange(0, TOPK)
        weights = tl.gather(s, ids, axis=0)
        if NORMALIZE:
            tl.static_assert(TOPK == 8)
            # Shuffle-down offsets 4, 2, 1: pair slot i with slot i + 4, then i + 2,
            # then i + 1 (exact layout moves, one add.rn per pair).
            lo, hi = tl.split(tl.trans(tl.reshape(weights, (2, 4))))
            quad = _add_rn(lo, hi)
            lo, hi = tl.split(tl.trans(tl.reshape(quad, (2, 2))))
            even, odd = tl.split(_add_rn(lo, hi))
            denominator = _add_rn(_add_rn(even, odd), 1e-20)
            weights = tl.div_rn(weights, denominator)
        if SCALE is not None:
            weights = _mul_rn(weights, SCALE)
        tl.store(weights_ptr + row * TOPK + slots, weights)
        tl.store(route_ids_ptr + row * TOPK + slots, ids)
    else:
        tl.store(
            group_ids_ptr + row * TOPK_GROUPS + tl.arange(0, TOPK_GROUPS),
            group_ids.to(tl.int64),
        )
        tl.store(ids_ptr + row * TOPK + tl.arange(0, TOPK), ids.to(tl.int64))


# ---------------------------------------------------------------------------------------
# The route in one launch: the lane-parallel selection with the route epilogue
# ---------------------------------------------------------------------------------------


def router_route_rejection(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_groups: int,
    group_topk: int,
    topk_groups: int,
    topk: int,
    renormalize: bool,
    scale: float | None = None,
) -> str:
    """Why ``triton_router_route_lanepar`` cannot serve this call; empty when admitted.

    The epilogue's own clauses first --
    ``renormalize`` a ``bool`` (it selects the compiled variant), ``scale``
    ``None`` or a real number whose FP32 rounding is finite and zero or normal
    (``route_scale_rejection``) -- then :func:`router_select_rejection` (the
    selection's envelope, zero rows included; its device clauses last).
    Metadata only (no device read).
    """
    if not isinstance(renormalize, bool):
        return f"renormalize must be a bool, got {type(renormalize).__name__}"
    reason = route_scale_rejection(scale)
    if reason:
        return reason
    return router_select_rejection(
        gating_output, correction_bias, num_groups, group_topk, topk_groups, topk
    )


def triton_router_route_lanepar(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    num_groups: int,
    group_topk: int,
    topk_groups: int,
    topk: int,
    renormalize: bool,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Beam's route in one launch: the lane-parallel selection, then the route
    epilogue on the row's registers.

    The selection is ``_router_select_lanepar_kernel``'s; under ``ROUTE`` the
    program gathers the selected unbiased scores from its registers and runs the
    epilogue's operations in Torch's order and rounding (the K8 tree of
    ``add.rn.f32``, ``+ 1e-20``, ``div.rn.f32``, one ``mul.rn.f32`` by the scale's
    FP32 rounding). So the outputs are ``torch_grouped_route``'s bytes, NaN
    payloads included.

    Args:
        gating_output: ``[rows, 256]`` contiguous FP32 CUDA router logits.
        correction_bias: ``[256]`` contiguous FP32 bias on the same device.
        num_groups: Must be 8 (the kernel's constexpr).
        group_topk: Must be 2 (the kernel's constexpr).
        topk_groups: Must be 4 (the kernel's constexpr).
        topk: Must be 8 (the kernel's constexpr).
        renormalize: ``True`` divides the gathered scores by their pinned K8 sum
            plus ``1e-20``.
        scale: ``None``, or the routed scale the weights are then multiplied by.

    Returns:
        Fresh contiguous ``[rows, 8]`` tensors: the FP32 weights and the int32 ids.

    Raises:
        ValueError: :func:`router_route_rejection` rejects the call. There is no
            fallback.
    """
    rejection = router_route_rejection(
        gating_output,
        correction_bias,
        num_groups,
        group_topk,
        topk_groups,
        topk,
        renormalize,
        scale,
    )
    if rejection:
        raise ValueError(
            f"triton_router_route_lanepar cannot serve this call: {rejection}"
        )
    rows = int(gating_output.shape[0])
    weights = torch.empty(
        (rows, _TOPK), dtype=torch.float32, device=gating_output.device
    )
    ids = torch.empty((rows, _TOPK), dtype=torch.int32, device=gating_output.device)
    enable_pdl = pdl_enabled()
    _router_select_lanepar_kernel[(rows,)](
        # Not written under ROUTE.
        ids,
        ids,
        NUM_EXPERTS=_NUM_EXPERTS,
        NUM_GROUPS=_NUM_GROUPS,
        GROUP_SIZE=_GROUP_SIZE,
        TOPK_GROUPS=_TOPK_GROUPS,
        TOPK=_TOPK,
        GROUP_CHUNK=_GROUP_CHUNK,
        EXPERT_CHUNK=_EXPERT_CHUNK,
        INT32_MIN=_INT32_MIN,
        KEY_NEG_INF=_KEY_NEG_INF,
        KEY_NAN=_KEY_NAN,
        logits_ptr=gating_output,
        bias_ptr=correction_bias,
        scores_ptr=weights,
        ENABLE_PDL=enable_pdl,
        weights_ptr=weights,
        route_ids_ptr=ids,
        ROUTE=True,
        NORMALIZE=renormalize,
        SCALE=None if scale is None else route_scale_fp32(scale),
        **{**_LANEPAR_LAUNCH_OPTIONS, "launch_pdl": enable_pdl},
    )
    return weights, ids
