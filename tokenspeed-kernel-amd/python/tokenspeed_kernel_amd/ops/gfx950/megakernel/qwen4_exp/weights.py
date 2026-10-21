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
    ple,
    qsa,
)
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import (
    mxfp4_scales,
    mxfp4_tiles,
    n32_scales,
    quantize_mxfp4_exact,
)


@dataclass(frozen=True)
class MixerWeights:
    """One hyper-connection mixer, in the layouts ``hc.py`` reads."""

    norm_w: torch.Tensor  # BF16 [4 * HIDDEN], raw Gemma weight
    w_down: torch.Tensor  # N16 tiles of W_down / 4
    s_down: torch.Tensor  # int32 scale dwords
    w_inj: torch.Tensor  # BF16 [4, 4 * HIDDEN], W_inj / 4
    w_up: torch.Tensor  # N16 tiles of W_up, K padded to UP_K
    s_up: torch.Tensor

    def kernel_args(self) -> tuple:
        """The kernel operand of ``hc.Mixer``."""
        return (self.norm_w, self.w_down, self.s_down, self.w_inj, self.w_up, self.s_up)


def pack_mixer(
    norm_w: torch.Tensor,
    projection: torch.Tensor,
    up: torch.Tensor,
) -> MixerWeights:
    """Pack TokenSpeed's ``GatedResidualSimple`` weights for the megakernel.

    Args:
        norm_w: ``hc_norm.weight`` ``[4 * HIDDEN]``.
        projection: ``mix_inject_proj.weight``. Rows ``[0, 320)`` are the
            down projection and rows ``[320, 324)`` are the inject projection.
            Both have the ``1 / 4`` scale folded in. This function ignores
            storage padding rows. A mixer without injection (the model's final
            mixer) has only the 320 down rows. Its inject weights are then
            zero, and nothing reads its inject logits.
        up: ``input_mix_weight_up.weight`` ``[4 * HIDDEN, 320]``.

    The checkpoint stores the mix weights rounded to MXFP4 values, so the
    MXFP4 encoding here is exact (``quantize_mxfp4_exact`` checks it).
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
    packed, scales = quantize_mxfp4_exact(projection[:lowrank])
    up_packed, up_scales = quantize_mxfp4_exact(up)
    return MixerWeights(
        norm_w=norm_w.contiguous(),
        w_down=mxfp4_tiles(packed, width),
        s_down=mxfp4_scales(scales, width),
        w_inj=inject.contiguous(),
        w_up=mxfp4_tiles(up_packed, int(hc.UP_K)),
        s_up=mxfp4_scales(up_scales, int(hc.UP_K)),
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
    w13: torch.Tensor  # TokenSpeed N16 expert banks (intermediate padded)
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
        banks: ``(w13, s13, w2, s2)`` of the routed experts as the MXFP8 N16
            MoE pipeline preprocesses them (intermediate padded to 768).
        shared_gu, shared_dn: ``(weight, scale)`` of the shared expert's
            ``gate_up_proj`` and ``down_proj`` (MXFP4).
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
    pad = int(moe.INTER_PAD)
    if w13.numel() != int(moe.EXPERTS) * 2 * pad * hidden // 2 or w2.numel() != (
        int(moe.EXPERTS) * hidden * pad // 2
    ):
        raise ValueError(
            f"expert banks must be padded to intermediate {pad}: "
            f"w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)}"
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


def _check_shapes(expected: dict) -> None:
    for name, (tensor, shape) in expected.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")


@dataclass(frozen=True)
class PleWeights:
    """The PLE block of a decoder layer, in the layouts ``ple.py`` reads."""

    kv_tiles: torch.Tensor  # kv_proj [key 4*HIDDEN | value HIDDEN], N16 tiles
    kv_scales: torch.Tensor  # k4 dwords
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
            self.kv_tiles,
            self.kv_scales,
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
    kv: tuple[torch.Tensor, torch.Tensor],
    norm_key: torch.Tensor,
    norm_query: torch.Tensor,
    norm_conv: torch.Tensor,
    conv_w: torch.Tensor,
    table: torch.Tensor,
    multipliers: torch.Tensor,
    sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos: int,
    dense_tiled: bool,
) -> PleWeights:
    """Pack a PLE block.

    Args:
        kv: ``(weight, scale)`` of the MXFP4 ``kv_proj`` (key rows, then value).
        norm_key, norm_query, norm_conv: The three Gemma norms' BF16 ``1 + w``
            ``[4 * HIDDEN]`` (``gemma_weight``).
        conv_w: Depthwise conv weight ``[4 * HIDDEN, 1, 4]`` or ``[.., 4]``.
        table: The n-gram embedding table, BF16 ``[rows, 160]`` (device,
            unsharded).
        multipliers, sizes, offsets: The n-gram hash parameters (int64 ``[3]``,
            ``[16]``, ``[16]``).
        eos: The EOS token id.
        dense_tiled: See :func:`pack_dense`.
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
    kv_tiles, kv_scales = pack_dense(*kv, int(ple.HIDDEN), dense_tiled)
    if kv_tiles.shape[0] != int(ple.KV_N):
        raise ValueError(f"kv_proj rows: {kv_tiles.shape[0]}")

    def bf16(t):
        return t.to(torch.bfloat16).contiguous()

    return PleWeights(
        kv_tiles=kv_tiles,
        kv_scales=kv_scales,
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
    qkvz_tiles: torch.Tensor  # in_proj_qkvz, N16 tiles
    qkvz_scales: torch.Tensor  # k4 dwords
    conv_w: torch.Tensor  # BF16 [QKV, 4]
    ba_w: torch.Tensor  # BF16 [in_proj_b | in_proj_a]
    a_log: torch.Tensor  # FP32 [VALUE_HEADS]
    dt_bias: torch.Tensor  # FP32 [VALUE_HEADS]
    norm_w: torch.Tensor  # BF16 [HEAD], gated RMS norm
    out_tiles: torch.Tensor  # out_proj, N16 tiles
    out_scales: torch.Tensor  # k4 dwords
    mlp_mixer: MixerWeights
    moe: MoeWeights
    ple: PleWeights | None  # the PLE block, for the layer that has one

    def gdn_args(self) -> tuple:
        """The kernel operand of ``gdn.Weights``."""
        return (
            self.qkvz_tiles,
            self.qkvz_scales,
            self.conv_w,
            self.ba_w,
            self.a_log,
            self.dt_bias,
            self.norm_w,
        )


def pack_delta_layer(
    *,
    attn_mixer: MixerWeights,
    qkvz: tuple[torch.Tensor, torch.Tensor],
    conv_w: torch.Tensor,
    ba_w: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor],
    mlp_mixer: MixerWeights,
    moe: MoeWeights,
    ple: PleWeights | None,
    dense_tiled: bool,
) -> DeltaLayerWeights:
    """Pack a Gated DeltaNet layer's TokenSpeed tensors for ``layer.py``.

    Args:
        attn_mixer, mlp_mixer: The two hyper-connection mixers (``pack_mixer``).
        qkvz, out: ``(weight, scale)`` of the MXFP4 ``in_proj_qkvz`` and
            ``out_proj``.
        conv_w: Conv weight ``[QKV, 4]``.
        ba_w: ``in_proj_ba`` weight ``[2 * VALUE_HEADS, HIDDEN]``.
        a_log, dt_bias: Decay parameters ``[VALUE_HEADS]``.
        norm_w: Gated RMS norm weight ``[HEAD]``.
        moe: The MoE block (:func:`pack_moe`).
        ple: The PLE block (:func:`pack_ple`), or None for a layer without.
        dense_tiled: See :func:`pack_dense`.

    Raises:
        ValueError: A tensor has an unexpected shape.
    """
    hidden, value = int(gdn.HIDDEN), int(gdn.VALUE)
    _check_shapes(
        {
            "conv_w": (conv_w, (int(gdn.QKV), int(gdn.CONV))),
            "ba_w": (ba_w, (2 * int(gdn.VALUE_HEADS), hidden)),
        }
    )
    qkvz_tiles, qkvz_scales = pack_dense(*qkvz, hidden, dense_tiled)
    out_tiles, out_scales = pack_dense(*out, value, dense_tiled)
    return DeltaLayerWeights(
        attn_mixer=attn_mixer,
        qkvz_tiles=qkvz_tiles,
        qkvz_scales=qkvz_scales,
        conv_w=conv_w.to(torch.bfloat16).contiguous(),
        ba_w=ba_w.to(torch.bfloat16).contiguous(),
        a_log=a_log.float().contiguous(),
        dt_bias=dt_bias.float().contiguous(),
        norm_w=norm_w.to(torch.bfloat16).contiguous(),
        out_tiles=out_tiles,
        out_scales=out_scales,
        mlp_mixer=mlp_mixer,
        moe=moe,
        ple=ple,
    )


@dataclass(frozen=True)
class QsaLayerWeights:
    """One QSA full-attention decoder layer, in the layouts ``layer.py`` reads."""

    attn_mixer: MixerWeights
    qkv_tiles: torch.Tensor  # qkv_proj ([q | gate] per head, k, v), N16 tiles
    qkv_scales: torch.Tensor  # k4 dwords
    idx_tiles: torch.Tensor  # index_qk_proj, N16 tiles
    idx_scales: torch.Tensor  # k4 dwords
    q_norm_w: torch.Tensor  # BF16 [HEAD], raw Gemma weight
    k_norm_w: torch.Tensor  # BF16 [HEAD], raw Gemma weight
    iq_norm_w: torch.Tensor  # BF16 [IDX_DIM], 1 + w rounded to BF16
    ik_norm_w: torch.Tensor  # BF16 [IDX_DIM], 1 + w rounded to BF16
    out_tiles: torch.Tensor  # o_proj, N16 tiles
    out_scales: torch.Tensor  # k4 dwords
    mlp_mixer: MixerWeights
    moe: MoeWeights
    # The model's final mixer when this is the last layer (it then runs in
    # the layer's launch), else None.
    final_mixer: MixerWeights | None

    def qsa_args(self) -> tuple:
        """The kernel operand of ``qsa.Weights``."""
        return (
            self.qkv_tiles,
            self.qkv_scales,
            self.idx_tiles,
            self.idx_scales,
            self.q_norm_w,
            self.k_norm_w,
            self.iq_norm_w,
            self.ik_norm_w,
        )


def pack_qsa_layer(
    *,
    attn_mixer: MixerWeights,
    qkv: tuple[torch.Tensor, torch.Tensor],
    index_qk: tuple[torch.Tensor, torch.Tensor],
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    iq_norm_w: torch.Tensor,
    ik_norm_w: torch.Tensor,
    out: tuple[torch.Tensor, torch.Tensor],
    mlp_mixer: MixerWeights,
    moe: MoeWeights,
    final_mixer: MixerWeights | None,
    dense_tiled: bool,
) -> QsaLayerWeights:
    """Pack a QSA layer's TokenSpeed tensors for ``layer.py``.

    Args:
        attn_mixer, mlp_mixer: The two hyper-connection mixers (``pack_mixer``).
        qkv, index_qk, out: ``(weight, scale)`` of the MXFP4 ``qkv_proj``,
            ``index_qk_proj`` and ``o_proj``.
        q_norm_w, k_norm_w: ``q_norm`` / ``k_norm`` weights ``[HEAD]`` (raw
            Gemma ``w``). The kernel forms ``1 + w`` in FP32 as TokenSpeed's
            attention prologue does.
        iq_norm_w, ik_norm_w: The indexer's ``q_layernorm`` / ``k_layernorm``
            weights (raw ``w``). This function packs them as the BF16 ``1 + w``
            that the indexer's kernels read.
        moe: The MoE block (:func:`pack_moe`).
        final_mixer: The model's final mixer (``pack_mixer``) when this is
            the last layer, else ``None``.
        dense_tiled: See :func:`pack_dense`.

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
    qkv_tiles, qkv_scales = pack_dense(*qkv, hidden, dense_tiled)
    idx_tiles, idx_scales = pack_dense(*index_qk, hidden, dense_tiled)
    out_tiles, out_scales = pack_dense(
        *out, int(qsa.HEADS) * int(qsa.HEAD), dense_tiled
    )
    if qkv_tiles.shape[0] != int(qsa.QKV_N) or idx_tiles.shape[0] != int(
        qsa.PROJ_N - qsa.QKV_N
    ):
        raise ValueError(
            f"qkv_proj / index_qk_proj rows: {qkv_tiles.shape[0]}, {idx_tiles.shape[0]}"
        )

    def plus_one(w):
        return (w.to(torch.bfloat16) + 1.0).to(torch.bfloat16).contiguous()

    return QsaLayerWeights(
        attn_mixer=attn_mixer,
        qkv_tiles=qkv_tiles,
        qkv_scales=qkv_scales,
        idx_tiles=idx_tiles,
        idx_scales=idx_scales,
        q_norm_w=q_norm_w.to(torch.bfloat16).contiguous(),
        k_norm_w=k_norm_w.to(torch.bfloat16).contiguous(),
        iq_norm_w=plus_one(iq_norm_w),
        ik_norm_w=plus_one(ik_norm_w),
        out_tiles=out_tiles,
        out_scales=out_scales,
        mlp_mixer=mlp_mixer,
        moe=moe,
        final_mixer=final_mixer,
    )


__all__ = [
    "DeltaLayerWeights",
    "MixerWeights",
    "MoeWeights",
    "PleWeights",
    "QsaLayerWeights",
    "alternate_gate_up",
    "pack_delta_layer",
    "pack_dense",
    "pack_mixer",
    "pack_moe",
    "pack_ple",
    "pack_qsa_layer",
]
