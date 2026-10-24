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

"""Host packing of Qwen3.8-Flash-Next weights for the decode megakernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import (
    gdn,
    hc,
    moe,
    moe_bf16,
    ple,
    qsa,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import (
    bf16_tiles,
    mxfp4_scales,
    mxfp4_tiles,
    n32_scales,
)


@dataclass(frozen=True)
class MixerWeights:
    """One hyper-connection mixer, in the layouts ``hc.py`` reads."""

    norm_w: torch.Tensor  # BF16 [4 * HIDDEN], raw Gemma weight
    w_down: torch.Tensor  # BF16 [LOWRANK, 4 * HIDDEN], W_down / 4 (bf16_tiles)
    w_inj: torch.Tensor  # BF16 [4, 4 * HIDDEN], W_inj / 4
    w_up: torch.Tensor  # BF16 [4 * HIDDEN, LOWRANK] (bf16_tiles)

    def kernel_args(self) -> tuple:
        """The kernel operand of ``hc.Mixer``."""
        return (self.norm_w, self.w_down, self.w_inj, self.w_up)


def pack_mixer(
    norm_w: torch.Tensor,
    projection: torch.Tensor,
    up: torch.Tensor,
) -> MixerWeights:
    """Pack TokenSpeed's ``GatedResidualSimple`` weights for the megakernel.

    The down and up projections are copied into the GEMV's tile order
    (``bf16_tiles``).

    Args:
        norm_w: ``hc_norm.weight`` ``[4 * HIDDEN]``.
        projection: ``mix_inject_proj.weight``. Rows ``[0, 320)`` are the
            down projection and rows ``[320, 324)`` are the inject projection.
            Both have the ``1 / 4`` scale folded in. This function ignores
            storage padding rows. A mixer without injection (the model's final
            mixer) has only the 320 down rows. Its inject weights are then
            zero, and nothing reads its inject logits.
        up: ``input_mix_weight_up.weight`` ``[4 * HIDDEN, 320]``.

    Raises:
        ValueError: A weight is not contiguous BF16 of the mixer's shape.
    """
    lowrank, width = int(hc.LOWRANK), int(hc.WIDTH)
    rows = projection.shape[0]
    if rows == lowrank:
        inject = projection.new_zeros(4, width)
    elif rows >= lowrank + 4:
        inject = projection[lowrank : lowrank + 4]
    else:
        raise ValueError(
            f"projection must have {lowrank} or at least {lowrank + 4} rows, got {rows}"
        )
    _check_shapes(
        {
            "norm_w": (norm_w, (width,)),
            "projection": (projection, (rows, width)),
            "up": (up, (width, lowrank)),
        }
    )
    for name, t in (("projection", projection), ("up", up)):
        if t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous BF16, got {t.dtype}")
    return MixerWeights(
        norm_w=norm_w.contiguous(),
        w_down=bf16_tiles(projection[:lowrank]),
        w_inj=inject.contiguous(),
        w_up=bf16_tiles(up),
    )


def pack_dense(
    weight: torch.Tensor, scale: torch.Tensor, k: int, tiled: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """N16 tiles and ``k4`` scale dwords of an MXFP4 ``[N, K]`` projection.

    Args:
        weight: Packed E2M1 ``[N, K/2]``. When ``tiled``, it is already in N16
            tile order (TokenSpeed's MFMA linear layout), and this function
            uses it in place.
        scale: UE8M0 ``[N, K/32]`` in checkpoint order.
        k: K, a multiple of 128.
        tiled: Whether ``weight`` is already in N16 tile order.
    """
    tiles = weight if tiled else mxfp4_tiles(weight, k)
    return tiles, mxfp4_scales(scale, k)


def alternate_gate_up(rows: torch.Tensor) -> torch.Tensor:
    """``[gate; up]`` rows (a weight, its N16 tiles or its scales) in the
    routed MoE banks' order: 16-row gate and up blocks alternate."""
    n = rows.shape[0]
    if n % 32:
        raise ValueError(f"gate / up rows must be 16-row blocks, got {n}")
    return rows.view(2, n // 32, 16, -1).transpose(0, 1).reshape(rows.shape)


@dataclass(frozen=True)
class MoeWeights:
    """The sparse MoE block (router, routed and shared experts) of a layer."""

    router_w: torch.Tensor  # BF16 [EXPERTS, HIDDEN]
    shared_gate_w: torch.Tensor  # BF16 [HIDDEN]
    w13: torch.Tensor  # TokenSpeed's N16 expert banks
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    # Shared expert gate / up: N16 tiles and n32 cells, gate and up blocks
    # alternating as in the banks (alternate_gate_up).
    shared_gu_tiles: torch.Tensor
    shared_gu_scales: torch.Tensor
    shared_dn_tiles: torch.Tensor  # shared expert down, N16 tiles
    shared_dn_scales: torch.Tensor  # n32 cells (batched with the banks)

    def kernel_args(self) -> tuple:
        """The kernel operand of ``moe.Weights``."""
        return (
            self.router_w,
            self.shared_gate_w,
            self.w13,
            self.s13,
            self.shared_gu_tiles,
            self.shared_gu_scales,
            self.w2,
            self.s2,
            self.shared_dn_tiles,
            self.shared_dn_scales,
        )


def pack_moe(
    *,
    router_w: torch.Tensor,
    shared_gate_w: torch.Tensor,
    banks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    shared_gu: tuple[torch.Tensor, torch.Tensor],
    shared_dn: tuple[torch.Tensor, torch.Tensor],
    dense_tiled: bool,
) -> MoeWeights:
    """Pack a layer's MoE block.

    Args:
        router_w, shared_gate_w: Router ``[EXPERTS, HIDDEN]`` and shared
            expert gate ``[1, HIDDEN]`` weights.
        banks: ``(w13, s13, w2, s2)`` of the routed experts as
            ``preprocess_n16_mxfp4_weights`` lays them out (TokenSpeed's W4A4
            MoE pipeline).
        shared_gu, shared_dn: ``(weight, scale)`` of the shared expert's
            ``gate_up_proj`` (gate rows over up rows) and ``down_proj``
            (MXFP4, linear scales ``[N, K/32]``).
        dense_tiled: See :func:`pack_dense`.

    Raises:
        ValueError: A tensor has an unexpected shape.
    """
    hidden, inter = int(moe.HIDDEN), int(moe.INTER)
    _check_shapes(
        {
            "router_w": (router_w, (int(moe.EXPERTS), hidden)),
            "shared_gate_w": (shared_gate_w.reshape(-1), (hidden,)),
        }
    )
    w13, s13, w2, s2 = banks
    experts = int(moe.EXPERTS)
    _check_shapes(
        {
            "w13": (w13, (experts, 2 * inter // 16, hidden // 128, 4, 16, 16)),
            "s13": (s13, (experts, 2 * inter // 32, hidden // 256, 4, 16, 4)),
            "w2": (w2, (experts, hidden // 16, 8 * inter)),
            "s2": (s2, (experts, hidden // 32, int(moe.W2_CELLS), 4, 16, 4)),
        }
    )
    gu_weight, gu_scale = shared_gu
    dn_weight, dn_scale = shared_dn
    return MoeWeights(
        router_w=router_w.to(torch.bfloat16).contiguous(),
        shared_gate_w=shared_gate_w.reshape(-1).to(torch.bfloat16).contiguous(),
        w13=w13,
        s13=s13,
        w2=w2,
        s2=s2,
        shared_gu_tiles=alternate_gate_up(
            gu_weight if dense_tiled else mxfp4_tiles(gu_weight, hidden)
        ),
        shared_gu_scales=n32_scales(alternate_gate_up(gu_scale), hidden),
        shared_dn_tiles=dn_weight if dense_tiled else mxfp4_tiles(dn_weight, inter),
        shared_dn_scales=n32_scales(dn_scale, inter),
    )


@dataclass(frozen=True)
class MoeBf16Weights:
    """A layer's MoE block with BF16 experts, read in place (``moe_bf16.py``)."""

    router_w: torch.Tensor  # BF16 [EXPERTS, HIDDEN]
    shared_gate_w: torch.Tensor  # BF16 [HIDDEN]
    w13: torch.Tensor  # BF16 [EXPERTS, 2 INTER, HIDDEN]: gate rows, then up
    w2: torch.Tensor  # BF16 [EXPERTS, HIDDEN, INTER]
    shared_gu: torch.Tensor  # BF16 [2 INTER, HIDDEN]: gate rows, then up
    shared_dn: torch.Tensor  # BF16 [HIDDEN, INTER]

    def kernel_args(self) -> tuple:
        """The kernel operand of ``moe_bf16.Weights``."""
        return (
            self.router_w,
            self.shared_gate_w,
            self.w13,
            self.w2,
            self.shared_gu,
            self.shared_dn,
        )


def pack_moe_bf16(
    *,
    router_w: torch.Tensor,
    shared_gate_w: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    shared_gu: torch.Tensor,
    shared_dn: torch.Tensor,
) -> MoeBf16Weights:
    """Pack a layer's MoE block whose experts are BF16 (TokenSpeed's
    unquantized MoE weights and BF16 shared expert). The kernel reads the
    expert weights in place.

    Args:
        router_w, shared_gate_w: Router ``[EXPERTS, HIDDEN]`` and shared
            expert gate ``[1, HIDDEN]`` weights.
        w13, w2: The routed experts' ``w13_weight`` ``[EXPERTS, 2 INTER,
            HIDDEN]`` (gate rows, then up rows) and ``w2_weight``
            ``[EXPERTS, HIDDEN, INTER]``.
        shared_gu, shared_dn: The shared expert's ``gate_up_proj``
            ``[2 INTER, HIDDEN]`` (gate rows, then up rows) and
            ``down_proj`` ``[HIDDEN, INTER]``.

    Raises:
        ValueError: A tensor has an unexpected shape, or a weight is not
            contiguous BF16.
    """
    hidden, inter, experts = int(moe.HIDDEN), int(moe.INTER), int(moe.EXPERTS)
    expected = {
        "router_w": (router_w, (experts, hidden)),
        "shared_gate_w": (shared_gate_w.reshape(-1), (hidden,)),
        "w13": (w13, (experts, 2 * inter, hidden)),
        "w2": (w2, (experts, hidden, inter)),
        "shared_gu": (shared_gu, (2 * inter, hidden)),
        "shared_dn": (shared_dn, (hidden, inter)),
    }
    _check_shapes(expected)
    for name in ("w13", "w2", "shared_gu", "shared_dn"):
        t = expected[name][0]
        if t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous BF16, got {t.dtype}")
    if int(moe_bf16.INTER) != inter:
        raise ValueError("moe_bf16 and moe disagree on INTER")
    return MoeBf16Weights(
        router_w=router_w.to(torch.bfloat16).contiguous(),
        shared_gate_w=shared_gate_w.reshape(-1).to(torch.bfloat16).contiguous(),
        w13=w13,
        w2=w2,
        shared_gu=shared_gu,
        shared_dn=shared_dn,
    )


def _check_shapes(expected: dict) -> None:
    for name, (tensor, shape) in expected.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")


def _bf16_weight(
    name: str, weight: torch.Tensor, shape: tuple[int, int]
) -> torch.Tensor:
    """A copy of the BF16 projection ``weight`` in the GEMV's tile order
    (``bf16_tiles``)."""
    _check_shapes({name: (weight, shape)})
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError(f"{name} must be contiguous BF16, got {weight.dtype}")
    return bf16_tiles(weight)


@dataclass(frozen=True)
class PleWeights:
    """The PLE block of a decoder layer, in the layouts ``ple.py`` reads."""

    kv_w: torch.Tensor  # kv_proj, BF16 [key 4 * HIDDEN | value HIDDEN, HIDDEN]
    norm_key: torch.Tensor  # BF16 [4 * HIDDEN], 1 + w rounded to BF16
    norm_query: torch.Tensor
    norm_conv: torch.Tensor
    conv_w: torch.Tensor  # BF16 [4 * HIDDEN, 4], depthwise conv
    table: torch.Tensor  # BF16 [rows, 160] n-gram embedding table
    multipliers: torch.Tensor  # int64 [3], the n-gram hash multipliers
    sizes: torch.Tensor  # int64 [16], per-head vocabulary sizes
    offsets: torch.Tensor  # int64 [16], per-head table offsets
    eos: int  # the EOS token id (blocks older context)

    def kernel_args(self) -> tuple:
        """The kernel operand of ``ple.Weights``."""
        return (
            self.kv_w,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            self.conv_w,
            self.table,
            self.multipliers,
            self.sizes,
            self.offsets,
            self.eos,
        )


def pack_ple(
    *,
    kv: torch.Tensor,
    norm_key: torch.Tensor,
    norm_query: torch.Tensor,
    norm_conv: torch.Tensor,
    conv_w: torch.Tensor,
    table: torch.Tensor,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos: int,
) -> PleWeights:
    """Pack a PLE block.

    Args:
        kv: The BF16 ``kv_proj`` weight (key rows, then value). The kernel
            reads a tiled copy (``bf16_tiles``).
        norm_key, norm_query, norm_conv: The three Gemma norms' BF16 ``1 + w``
            ``[4 * HIDDEN]`` (``gemma_weight``).
        conv_w: Depthwise conv weight ``[4 * HIDDEN, 1, 4]`` or ``[.., 4]``.
        table: The n-gram embedding table, BF16 ``[rows, 160]`` (device,
            unsharded).
        multipliers, sizes, offsets: The n-gram hash parameters (int64 ``[3]``,
            ``[16]``, ``[16]``).
        eos: The EOS token id.
    """
    width = int(ple.WIDTH)
    _check_shapes(
        {
            "norm_key": (norm_key, (width,)),
            "norm_query": (norm_query, (width,)),
            "norm_conv": (norm_conv, (width,)),
            "conv_w": (conv_w.reshape(width, -1), (width, 4)),
            "multipliers": (multipliers, (3,)),
            "sizes": (sizes, (int(ple.NGRAM_HEADS),)),
            "offsets": (offsets, (int(ple.NGRAM_HEADS),)),
        }
    )
    if (
        table.dtype != torch.bfloat16
        or table.shape[-1] != int(ple.NGRAM_DIM)
        or (not table.is_contiguous())
    ):
        raise ValueError(
            f"n-gram table must be dense BF16 [rows, 160], got {table.dtype} {tuple(table.shape)}"
        )

    def bf16(t):
        return t.to(torch.bfloat16).contiguous()

    return PleWeights(
        kv_w=_bf16_weight("kv", kv, (int(ple.KV_N), int(ple.HIDDEN))),
        norm_key=bf16(norm_key),
        norm_query=bf16(norm_query),
        norm_conv=bf16(norm_conv),
        conv_w=bf16(conv_w.reshape(width, 4)),
        table=table,
        multipliers=multipliers.to(torch.int64).contiguous(),
        sizes=sizes.to(torch.int64).contiguous(),
        offsets=offsets.to(torch.int64).contiguous(),
        eos=int(eos),
    )


@dataclass(frozen=True)
class DeltaLayerWeights:
    """One Gated DeltaNet decoder layer, in the layouts ``layer.py`` reads."""

    attn_mixer: MixerWeights
    # in_proj_qkvzba, BF16 [QKV + VALUE + 2 * VALUE_HEADS, HIDDEN]
    qkvzba: torch.Tensor
    conv_w: torch.Tensor  # BF16 [QKV, 4]
    a_log: torch.Tensor  # FP32 [VALUE_HEADS]
    dt_bias: torch.Tensor  # FP32 [VALUE_HEADS]
    norm_w: torch.Tensor  # BF16 [HEAD], gated RMS norm
    out_w: torch.Tensor  # out_proj, BF16 [HIDDEN, VALUE]
    mlp_mixer: MixerWeights
    moe: MoeWeights | MoeBf16Weights
    ple: PleWeights | None  # the PLE block, for the layer that has one

    def gdn_args(self) -> tuple:
        """The kernel operand of ``gdn.Weights``."""
        return (
            self.qkvzba,
            self.conv_w,
            self.a_log,
            self.dt_bias,
            self.norm_w,
        )


def pack_delta_layer(
    *,
    attn_mixer: MixerWeights,
    qkvzba: torch.Tensor,
    conv_w: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    out: torch.Tensor,
    mlp_mixer: MixerWeights,
    moe: MoeWeights | MoeBf16Weights,
    ple: PleWeights | None,
) -> DeltaLayerWeights:
    """Pack a Gated DeltaNet layer's TokenSpeed tensors for ``layer.py``.

    Args:
        attn_mixer, mlp_mixer: The two hyper-connection mixers (``pack_mixer``).
        qkvzba, out: The BF16 ``in_proj_qkvzba`` (``q | k | v``, ``z``, then
            ``b``, ``a``) and ``out_proj`` weights. The kernel reads tiled
            copies (``bf16_tiles``).
        conv_w: Conv weight ``[QKV, 4]``.
        a_log, dt_bias: Decay parameters ``[VALUE_HEADS]``.
        norm_w: Gated RMS norm weight ``[HEAD]``.
        moe: The MoE block (:func:`pack_moe` or :func:`pack_moe_bf16`).
        ple: The PLE block (:func:`pack_ple`), or None for a layer without.

    Raises:
        ValueError: A tensor has an unexpected shape.
    """
    hidden, value = int(gdn.HIDDEN), int(gdn.VALUE)
    rows = int(gdn.QKV + gdn.VALUE + 2 * gdn.VALUE_HEADS)
    _check_shapes({"conv_w": (conv_w, (int(gdn.QKV), int(gdn.CONV)))})
    return DeltaLayerWeights(
        attn_mixer=attn_mixer,
        qkvzba=_bf16_weight("qkvzba", qkvzba, (rows, hidden)),
        conv_w=conv_w.to(torch.bfloat16).contiguous(),
        a_log=a_log.float().contiguous(),
        dt_bias=dt_bias.float().contiguous(),
        norm_w=norm_w.to(torch.bfloat16).contiguous(),
        out_w=_bf16_weight("out", out, (hidden, value)),
        mlp_mixer=mlp_mixer,
        moe=moe,
        ple=ple,
    )


@dataclass(frozen=True)
class QsaLayerWeights:
    """One QSA full-attention decoder layer, in the layouts ``layer.py`` reads."""

    attn_mixer: MixerWeights
    qkv_w: torch.Tensor  # qkv_proj ([q | gate] per head, k, v), BF16
    idx_w: torch.Tensor  # index_qk_proj, BF16
    q_norm_w: torch.Tensor  # BF16 [HEAD], raw Gemma weight
    k_norm_w: torch.Tensor  # BF16 [HEAD], raw Gemma weight
    iq_norm_w: torch.Tensor  # BF16 [IDX_DIM], 1 + w rounded to BF16
    ik_norm_w: torch.Tensor  # BF16 [IDX_DIM], 1 + w rounded to BF16
    out_w: torch.Tensor  # o_proj, BF16 [HIDDEN, HEADS * HEAD]
    mlp_mixer: MixerWeights
    moe: MoeWeights | MoeBf16Weights
    # The model's final mixer when this is the last layer (it then runs in
    # the layer's launch), else None.
    final_mixer: MixerWeights | None

    def qsa_args(self) -> tuple:
        """The kernel operand of ``qsa.Weights``."""
        return (
            self.qkv_w,
            self.idx_w,
            self.q_norm_w,
            self.k_norm_w,
            self.iq_norm_w,
            self.ik_norm_w,
        )


def pack_qsa_layer(
    *,
    attn_mixer: MixerWeights,
    qkv: torch.Tensor,
    index_qk: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    iq_norm_w: torch.Tensor,
    ik_norm_w: torch.Tensor,
    out: torch.Tensor,
    mlp_mixer: MixerWeights,
    moe: MoeWeights | MoeBf16Weights,
    final_mixer: MixerWeights | None,
) -> QsaLayerWeights:
    """Pack a QSA layer's TokenSpeed tensors for ``layer.py``.

    Args:
        attn_mixer, mlp_mixer: The two hyper-connection mixers (``pack_mixer``).
        qkv, index_qk, out: The BF16 ``qkv_proj``, ``index_qk_proj`` and
            ``o_proj`` weights. The kernel reads tiled copies
            (``bf16_tiles``).
        q_norm_w, k_norm_w: ``q_norm`` / ``k_norm`` weights ``[HEAD]`` (raw
            Gemma ``w``). The kernel forms ``1 + w`` in FP32 as TokenSpeed's
            attention prologue does.
        iq_norm_w, ik_norm_w: The indexer's ``q_layernorm`` / ``k_layernorm``
            weights (raw ``w``). This function packs them as the BF16 ``1 + w``
            that the indexer's kernels read.
        moe: The MoE block (:func:`pack_moe` or :func:`pack_moe_bf16`).
        final_mixer: The model's final mixer (``pack_mixer``) when this is
            the last layer, else ``None``.

    Raises:
        ValueError: A tensor has an unexpected shape.
    """
    hidden = int(qsa.HIDDEN)
    _check_shapes(
        {
            "q_norm_w": (q_norm_w, (int(qsa.HEAD),)),
            "k_norm_w": (k_norm_w, (int(qsa.HEAD),)),
            "iq_norm_w": (iq_norm_w, (int(qsa.IDX_DIM),)),
            "ik_norm_w": (ik_norm_w, (int(qsa.IDX_DIM),)),
        }
    )

    def plus_one(w):
        return (w.to(torch.bfloat16) + 1.0).to(torch.bfloat16).contiguous()

    return QsaLayerWeights(
        attn_mixer=attn_mixer,
        qkv_w=_bf16_weight("qkv", qkv, (int(qsa.QKV_N), hidden)),
        idx_w=_bf16_weight("index_qk", index_qk, (int(qsa.PROJ_N - qsa.QKV_N), hidden)),
        q_norm_w=q_norm_w.to(torch.bfloat16).contiguous(),
        k_norm_w=k_norm_w.to(torch.bfloat16).contiguous(),
        iq_norm_w=plus_one(iq_norm_w),
        ik_norm_w=plus_one(ik_norm_w),
        out_w=_bf16_weight("out", out, (hidden, int(qsa.HEADS) * int(qsa.HEAD))),
        mlp_mixer=mlp_mixer,
        moe=moe,
        final_mixer=final_mixer,
    )


__all__ = [
    "DeltaLayerWeights",
    "MixerWeights",
    "MoeBf16Weights",
    "MoeWeights",
    "PleWeights",
    "QsaLayerWeights",
    "alternate_gate_up",
    "pack_delta_layer",
    "pack_dense",
    "pack_mixer",
    "pack_moe",
    "pack_moe_bf16",
    "pack_ple",
    "pack_qsa_layer",
]
