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

"""Weight packing of one Muse Spark TP8 rank's layer launch.

The packer takes the regular path's tensors of one rank and returns the
operands that :func:`layer._launch_layer` reads:

* BF16 N16 tiles (``bf16_tiles``) of the QKVG, o_proj, pooler and
  post-expert weights. The pooler and post-expert weights are replicated on
  every rank; a rank tiles only the rows it computes.
* The MXFP4 experts, read in place in the layout of the regular path's
  gluon MoE (``preprocess_gluon_mxfp4_gfx950_moe_weights``).
* The router columns of the rank's owned hidden columns, the norm gains and
  the residual gate coefficients. The coefficients stay views: the regular
  path re-derives them in place when weights reload, and the launch reads
  the same storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import dims
from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import bf16_tiles

# The kernels' dimensions as host integers.
WORLD = int(dims.WORLD)
HIDDEN = int(dims.HIDDEN)
OWNED = int(dims.OWNED)  # hidden columns a rank owns
QKVG = int(dims.QKVG)  # q | k | v | gate rows of a rank
Q = int(dims.Q)  # q (and attention output) width of a rank
POOLED = int(dims.POOLED)
POOL_OWNED = POOLED // WORLD  # pooled columns a rank computes
EXPERTS = int(dims.EXPERTS)


@dataclass(frozen=True)
class LayerWeights:
    """The packed weights of one layer launch of one rank.

    Gains are FP32 multipliers (``bf16(w + offset)``). ``alpha*`` and
    ``beta*`` are views of the regular path's residual gate coefficients.
    """

    rank: int
    full: bool  # a full-attention (NoPE) layer
    qkvg: torch.Tensor  # BF16 tiles [QKVG, HIDDEN]
    o: torch.Tensor  # BF16 tiles [HIDDEN, Q]
    alpha1: torch.Tensor  # FP32 [OWNED]
    beta1: torch.Tensor
    pre_gain: torch.Tensor  # FP32 [OWNED], pre-FFN norm
    router: torch.Tensor  # FP32 [EXPERTS, OWNED]
    router_bias: torch.Tensor  # FP32 [EXPERTS]
    pool: torch.Tensor  # BF16 tiles [POOL_OWNED, HIDDEN]
    # The gluon MXFP4 MoE's preprocessed expert tensors and their scales.
    w13: torch.Tensor
    w13s: torch.Tensor
    w2: torch.Tensor
    w2s: torch.Tensor
    pre_exp_gain: torch.Tensor  # FP32 [POOLED]
    pen_gain: torch.Tensor  # FP32 [POOL_OWNED], post-expert norm
    post: torch.Tensor  # BF16 tiles [OWNED, POOLED]
    post_gain: torch.Tensor  # FP32 [HIDDEN], post-FFN norm: out reads it all
    alpha2: torch.Tensor  # FP32 [OWNED]
    beta2: torch.Tensor  # FP32 [HIDDEN]: every rank's out stage reads it all
    in_gain: torch.Tensor  # FP32 [HIDDEN], the next layer's input norm

    def kernel_args(
        self,
        *,
        h: torch.Tensor,
        a: torch.Tensor,
        seq_len: torch.Tensor,
        slot: torch.Tensor,
        page_table: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cos_sin: torch.Tensor,
        h_out: torch.Tensor,
        a_out: torch.Tensor,
    ) -> tuple[tuple, int, tuple]:
        """Return ``(attn_ops, table_stride, moe_ops)`` for this step (see
        ``gluon_muse_spark_layer_gfx950``). The page-table row stride is a
        top-level runtime argument, outside the operand tuples."""
        attn_ops = (
            self.qkvg,
            a,
            seq_len,
            slot,
            page_table,
            k_cache,
            v_cache,
            cos_sin,
            self.o,
            h,
            self.alpha1,
            self.beta1,
            self.pre_gain,
        )
        w13s, s13 = self.w13s, self.w13s.stride()
        w2s, s2 = self.w2s, self.w2s.stride()
        moe_ops = (
            self.router,
            self.router_bias,
            self.pool,
            self.w13,
            w13s,
            self.w13.stride(0),
            *s13,
            self.w2,
            w2s,
            self.w2.stride(0),
            *s2,
            self.pre_exp_gain,
            self.pen_gain,
            self.post,
            self.post_gain,
            self.alpha2,
            self.beta2,
            self.in_gain,
            h_out,
            a_out,
        )
        return attn_ops, page_table.stride(0), moe_ops


def _mismatch(name: str, t: torch.Tensor, shape: tuple, dtype: torch.dtype) -> list:
    """``[message]`` when ``t`` is not ``dtype`` of ``shape``, else ``[]``."""
    if tuple(t.shape) == shape and t.dtype == dtype:
        return []
    return [f"{name}: expected {dtype} {shape}, got {t.dtype} {tuple(t.shape)}"]


def pack_layer(
    *,
    rank: int,
    full: bool,
    qkvg: torch.Tensor,
    o: torch.Tensor,
    alpha1: torch.Tensor,
    beta1: torch.Tensor,
    pre_gain: torch.Tensor,
    router: torch.Tensor,
    router_bias: torch.Tensor,
    pool: torch.Tensor,
    experts: tuple[torch.Tensor, ...],
    pre_exp_gain: torch.Tensor,
    pen_gain: torch.Tensor,
    post: torch.Tensor,
    post_gain: torch.Tensor,
    alpha2: torch.Tensor,
    beta2: torch.Tensor,
    in_gain: torch.Tensor,
) -> LayerWeights:
    """Pack one rank's layer.

    Args:
        rank: The rank in the TP8 group: it owns hidden columns
            ``1024 rank ..`` and pooled columns ``512 rank ..``.
        full: Whether the layer is a full-attention (NoPE) layer.
        qkvg: The rank's BF16 fused projection ``[2304, 8192]`` (q | k | v |
            gate rows).
        o: The rank's BF16 o_proj ``[8192, 1024]``.
        alpha1, beta1, alpha2, beta2: FP32 ``[8192]`` residual gate
            coefficients after attention and after the FFN.
        pre_gain, post_gain, in_gain: FP32 ``[8192]`` gains of the pre-FFN,
            post-FFN and next layer's input norms.
        router: FP32 ``[8192, 256]`` router weight (logits are ``x @
            router``); ``router_bias`` FP32 ``[256]``.
        pool: BF16 pooler (pre-expert projection) ``[4096, 8192]``.
        experts: ``(w13, w13 scales, w2, w2 scales)`` of the MXFP4 experts
            as the regular path's gluon MoE prepared them (raw tensors of
            the triton tensors).
        pre_exp_gain, pen_gain: FP32 ``[4096]`` gains of the pre-expert and
            post-expert norms.
        post: BF16 post-expert projection ``[8192, 4096]``.

    Raises:
        ValueError: A tensor has an unexpected shape or dtype.
    """
    if not 0 <= rank < WORLD:
        raise ValueError(f"rank {rank} is not in a group of {WORLD}")
    f32, bf16 = torch.float32, torch.bfloat16
    errors = []
    for name, t in (
        ("alpha1", alpha1),
        ("beta1", beta1),
        ("alpha2", alpha2),
        ("beta2", beta2),
        ("pre_gain", pre_gain),
        ("post_gain", post_gain),
        ("in_gain", in_gain),
    ):
        errors += _mismatch(name, t, (HIDDEN,), f32)
    errors += _mismatch("qkvg", qkvg, (QKVG, HIDDEN), bf16)
    errors += _mismatch("o", o, (HIDDEN, Q), bf16)
    errors += _mismatch("router", router, (HIDDEN, EXPERTS), f32)
    errors += _mismatch("router_bias", router_bias, (EXPERTS,), f32)
    errors += _mismatch("pool", pool, (POOLED, HIDDEN), bf16)
    errors += _mismatch("post", post, (HIDDEN, POOLED), bf16)
    errors += _mismatch("pre_exp_gain", pre_exp_gain, (POOLED,), f32)
    errors += _mismatch("pen_gain", pen_gain, (POOLED,), f32)
    w13, w13s, w2, w2s = experts
    if w13.shape[0] != EXPERTS or w2.shape[0] != EXPERTS:
        errors.append(
            f"experts: expected {EXPERTS} experts, got {w13.shape[0]} and {w2.shape[0]}"
        )
    # The expert stages read these with buffer loads, whose byte offset (the
    # expert's offset plus the in-expert one) is 32-bit: every byte of each
    # tensor must lie below 2^31.
    for name, t in (("w13", w13), ("w13 scales", w13s), ("w2", w2), ("w2 scales", w2s)):
        end = 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))
        if end * t.element_size() > 2**31:
            errors.append(
                f"experts: {name} spans past 2^31 bytes, past a buffer load's offset"
            )
    if errors:
        raise ValueError("; ".join(errors))
    own = slice(rank * OWNED, (rank + 1) * OWNED)
    pool_own = slice(rank * POOL_OWNED, (rank + 1) * POOL_OWNED)
    return LayerWeights(
        rank=rank,
        full=full,
        qkvg=bf16_tiles(qkvg),
        o=bf16_tiles(o),
        alpha1=alpha1[own],
        beta1=beta1[own],
        pre_gain=pre_gain[own].contiguous(),
        router=router[own].t().contiguous(),
        router_bias=router_bias.contiguous(),
        pool=bf16_tiles(pool[pool_own]),
        w13=w13,
        w13s=w13s,
        w2=w2,
        w2s=w2s,
        pre_exp_gain=pre_exp_gain.contiguous(),
        pen_gain=pen_gain[pool_own].contiguous(),
        post=bf16_tiles(post[own]),
        post_gain=post_gain.contiguous(),
        alpha2=alpha2[own],
        beta2=beta2,
        in_gain=in_gain.contiguous(),
    )


__all__ = ["LayerWeights", "pack_layer"]
