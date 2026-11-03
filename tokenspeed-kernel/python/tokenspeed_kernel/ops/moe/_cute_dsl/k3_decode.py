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

"""Kimi-K3 TP8 decode chain over Lamport mailboxes.

Per MoE layer: one GEMM publishes this rank's router logits and latent-down
block to every rank, one kernel routes top-16 and gathers the latent, the
persistent experts compute and combine their rows and publish the latent
partial, and one tail kernel all-reduces it, normalizes, up-projects, adds the
shared expert, and publishes the output row. The shared expert runs on a side
stream. Either a ring kernel adds that row to the residual, or the next layer's
AttnRes entry reads it: per decoder layer, the entry mixes the residual into
the attention input, a split-K GEMM runs the input projection, and the
o-projection exit all-reduces the partials through a mailbox and applies the
post-attention AttnRes. Every rank must run the chain for the same row counts,
since the mailboxes are collective.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass.cute.runtime import make_fake_stream
from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode_support import (
    BATCHES,
    EXPERTS,
    HIDDEN,
    KDA_ROWS,
    LATENT,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.attn_res.prepared import (
    PreparedAttnRes,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.host import (
    OprojMailbox,
    fake,
    stream,
    to_cute,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.mailbox import poll, rearm
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.primitives import (
    at,
    launch,
    load_words,
    pack_pair,
    store_words,
    unpack_bf16,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.twoshot import TwoShot
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.gemm.projection import (
    PreparedProjection,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe.experts_runtime import (
    PersistentExperts,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe.front import SharedFront
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe.prepared import (
    MoEScratch,
    PreparedMoE,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe.tail_runtime import (
    MoeTailBox,
    PreparedTail,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.oproj.prepared import (
    PreparedOproj,
)


class RingResidual:
    """residual'[t] = BF16(residual[t] + row[t]) for the tail's published rows.

    Polls each 16-byte fragment of the output row until no word holds the
    sentinel, then re-arms it for the next MoE layer.
    """

    def __init__(self, batch: int) -> None:
        self.batch = batch

    @cute.jit
    def __call__(self, ring, residual, out, stream):
        launch(self.kernel, (ring, residual, out, stream), (7, self.batch), 128)

    @cute.kernel
    def kernel(self, args):

        ring, residual, out = args
        tid, _, _ = cute.arch.thread_idx()
        block, token, _ = cute.arch.block_idx()
        col = (block * 128 + tid) * 8
        base = ring.iterator + token * HIDDEN + col
        got = poll(base, HIDDEN, 1)
        rearm(base)
        cute.arch.griddepcontrol_wait()
        old = load_words(at(residual, token, col), count=4, mutable=True)
        words = cute.make_rmem_tensor(4, cutlass.Uint32)
        for j in cutlass.range_constexpr(4):
            dlo, dhi = unpack_bf16(got[0, j])
            plo, phi = unpack_bf16(old[j])
            words[j] = pack_pair(plo + dlo, phi + dhi, policy=None)
        cute.arch.griddepcontrol_launch_dependents()
        store_words(at(out, token, col), tuple(words[j] for j in range(4)))


@functools.cache
def _compile_ring(batch: int, device: int):
    args = (fake((1, 64, HIDDEN)), fake((batch, HIDDEN)), fake((batch, HIDDEN)))
    with torch.accelerator.device_index(device):
        return cute.compile(RingResidual(batch), *args, make_fake_stream())


class KimiK3DecodeMoE:
    """Model-owned mailboxes and scratch shared by every MoE layer's chain.

    Construct collectively on every rank of the MoE TP8 group, after the
    expert weights are in their serving layout and before graph capture.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        batches: tuple[int, ...],
        verify_width: int,
        moe_layers: int,
        beta: float,
        linear_beta: float,
    ) -> None:
        """Allocate the mailboxes and scratch and compile the residual ring.

        Args:
            group: The MoE TP8 process group the mailboxes span.
            batches: The token row counts the chain serves (from ``BATCHES``).
            verify_width: Rows per request of every decode forward (1 without
                speculative decoding); picks the shared expert's splits.
            moe_layers: The model's MoE layer count; the tail rotates its
                mailbox parities over it.
            beta: SiTU's gate cap.
            linear_beta: SiTU's linear cap.
        """
        tp = dist.get_world_size(group)
        if tp != 8:
            raise ValueError(f"the K3 decode MoE chain serves TP8, got TP{tp}")
        unsupported = set(batches) - set(BATCHES)
        if unsupported:
            raise ValueError(
                f"no K3 decode MoE chain for batches {sorted(unsupported)}"
            )
        self.tp, self.rank = tp, dist.get_rank(group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.batches = tuple(sorted(batches))
        self.verify_width = verify_width
        self.moe_layers = moe_layers
        self.beta, self.linear_beta = beta, linear_beta
        owner = (group, self.device, tp, self.rank)
        self.state = MoEScratch(tp, self.device, self.batches)
        self.front_box = OprojMailbox(*owner, (1, 64, LATENT + 2 * EXPERTS))
        self.tail_box = MoeTailBox(*owner)
        self.rings = {b: _compile_ring(b, self.device.index) for b in self.batches}

    def layer(
        self,
        *,
        ordinal: int,
        layer_idx: int,
        router: torch.Tensor,
        correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        renormalize: bool,
        down_shard: torch.Tensor,
        shared_gate_up: torch.Tensor,
        shared_down: torch.Tensor,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        up_shard: torch.Tensor,
    ) -> "KimiK3DecodeMoELayer":
        """One MoE layer's chain.

        Args:
            ordinal: The layer's position among the MoE layers, from zero.
            layer_idx: The decoder layer's model index; it keys the chain's
                next-attention prefetch, which this integration does not set.
            router: [896, 7168] BF16 router weight (every expert).
            correction_bias: [896] FP32 routing bias.
            routed_scaling_factor: Scale of the routed expert weights.
            renormalize: Whether the top-16 weights are renormalized to sum to 1.
            down_shard: [448, 7168] BF16 rows of the latent-down weight this
                rank owns.
            shared_gate_up: [1536, 7168] BF16 gate then up rows of this rank's
                shared-expert shard.
            shared_down: [7168, 768] BF16 shared-expert down shard.
            w13, w13_scale, w2, w2_scale: routed MXFP4 experts in the
                TRT-LLM-gen serving layout.
            norm_weight: [3584] BF16 latent RMSNorm weight.
            norm_eps: The latent RMSNorm epsilon.
            up_shard: [896, 3584] BF16 rows of the up-projection this rank owns.
        """
        return KimiK3DecodeMoELayer(
            self,
            ordinal=ordinal,
            layer_idx=layer_idx,
            router=router,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
            down_shard=down_shard,
            shared_gate_up=shared_gate_up,
            shared_down=shared_down,
            w13=w13,
            w13_scale=w13_scale,
            w2=w2,
            w2_scale=w2_scale,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            up_shard=up_shard,
        )


class KimiK3DecodeMoELayer:
    """One MoE layer's prepared chain; see :class:`KimiK3DecodeMoE`."""

    def __init__(
        self, owner: KimiK3DecodeMoE, *, ordinal: int, layer_idx: int, **w
    ) -> None:
        tp, rank = owner.tp, owner.rank
        rows = EXPERTS // tp
        if w["down_shard"].shape != (LATENT // tp, HIDDEN):
            raise ValueError(f"latent-down shard {tuple(w['down_shard'].shape)}")
        if w["up_shard"].shape != (HIDDEN // tp, LATENT):
            raise ValueError(f"up-projection shard {tuple(w['up_shard'].shape)}")
        # This rank's router rows, padded to 16, then its latent-down rows.
        router = w["router"][rank * rows : (rank + 1) * rows]
        pad = router.new_zeros((-rows % 16, HIDDEN))
        self.packed = torch.cat((router, pad, w["down_shard"])).contiguous()
        moe = SimpleNamespace(
            shared_experts=SimpleNamespace(
                gate_up_proj=SimpleNamespace(weight=w["shared_gate_up"]),
                down_proj=SimpleNamespace(weight=w["shared_down"]),
            ),
            routed_scaling_factor=w["routed_scaling_factor"],
            moe_renormalize=w["renormalize"],
            gate=SimpleNamespace(e_score_correction_bias=w["correction_bias"]),
        )
        state, batches = owner.state, owner.batches
        front = SharedFront(
            moe,
            self.packed,
            owner.front_box,
            state.front,
            batches,
            owner.beta,
            owner.linear_beta,
            plan=state.plan,
            counters=state.shared,
            q=owner.verify_width,
        )
        experts = PersistentExperts(
            w["w13"].view(torch.uint8),
            w["w13_scale"].view(torch.uint8),
            w["w2"].view(torch.uint8),
            w["w2_scale"].view(torch.uint8),
            beta=owner.beta,
            linear_beta=owner.linear_beta,
        )
        placement = (owner.tail_box, ordinal, owner.moe_layers)
        tail = PreparedTail(
            tp,
            rank,
            w["norm_eps"],
            w["norm_weight"],
            w["up_shard"],
            batches,
            placement,
        )
        self.chain = PreparedMoE(front, experts, tail, state, layer_idx=layer_idx)
        self.owner = owner

    def run(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Return ``residual`` plus this layer's MoE output for decode rows ``x``.

        Args:
            x: [B, 7168] BF16 MoE input, replicated on every rank.
            residual: [B, 7168] BF16 residual the output accumulates onto.
        """
        batch = x.shape[0]
        out = torch.empty_like(residual)
        ring = self.owner.rings[batch]

        def consume(row):
            ring(to_cute(row), to_cute(residual), to_cute(out), stream(x.device))
            return row

        self.chain.run(x, self.owner.tail_box.out.buffer, consume=consume)
        return out

    def publish(self, x: torch.Tensor) -> None:
        """Run the chain for ``x``, leaving its output row in the tail's ring for
        the next :class:`KimiK3DecodeAttentionLayer` or final AttnRes to consume."""
        self.chain.run(x, self.owner.tail_box.out.buffer, consume=lambda row: row)


def _projection_tactics(rows: int, batches) -> dict[int, tuple[int, int, int, int]]:
    """Split-K (M tile, N tile, split, stages) per batch, by its next tested bin;
    GB300 TP8 sweeps of the vendored kernels (KDA 6288 rows, MLA 3648)."""
    tactics = {}
    for batch in batches:
        n = next(b for b in (8, 16, 32, 64) if batch <= b)
        if rows == KDA_ROWS:
            tactics[batch] = (
                (64, 64, 1, 6) if n == 64 else (128, n, 2, 5 if n == 32 else 6)
            )
        else:
            stages = {8: 8, 16: 10, 32: 8, 64: 5}[n]
            tactics[batch] = (64, n, 4 if n == 8 else 2, stages)
    return tactics


class KimiK3DecodeAttention:
    """Model-owned o-projection mailbox and the MoE ring the AttnRes kernels read.

    Construct collectively on every rank of the attention TP8 group, after
    :class:`KimiK3DecodeMoE`, whose output ring the AttnRes kernels consume.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        batches: tuple[int, ...],
        moe: KimiK3DecodeMoE,
    ) -> None:
        """Allocate the o-projection mailbox and the two-shot buffers.

        Args:
            group: The attention TP8 process group the mailboxes span.
            batches: The token row counts the chain serves (the MoE chain's).
            moe: The MoE chain, whose output ring the AttnRes entries read.
        """
        tp = dist.get_world_size(group)
        if tp != 8:
            raise ValueError(f"the K3 decode attention chain serves TP8, got TP{tp}")
        self.batches, self.moe = tuple(sorted(batches)), moe
        self.device = torch.device("cuda", torch.cuda.current_device())
        owner = (group, self.device, tp, dist.get_rank(group))
        self.mailbox = OprojMailbox(*owner)
        # Batches above 32 reduce the o-projection in two shots.
        self.twoshot = TwoShot(*owner)

    def layer(
        self,
        decoder,
        *,
        layer_idx: int,
        mla: bool,
        first: bool,
        projection_weight: torch.Tensor,
    ) -> "KimiK3DecodeAttentionLayer":
        """One decoder layer's AttnRes entry, input projection and o-projection exit.

        Args:
            decoder: The decoder layer; its AttnRes norms and projections,
                ``self_attn.o_proj`` (this rank's [7168, K] BF16 shard) and its
                block-write position are read.
            layer_idx: The layer's model index (selects the mailbox parity).
            mla: Whether the attention is MLA (else KDA).
            first: Whether this is the first layer, whose entry has no prefix.
            projection_weight: The attention's packed BF16 input projection
                [rows, 7168] (row pitch kept).
        """
        return KimiK3DecodeAttentionLayer(
            self, decoder, layer_idx, mla, first, projection_weight
        )

    def final(self, *, res_norm, res_proj, out_norm, num_blocks: int):
        """The model's closing AttnRes, which also applies the final norm.

        Args:
            res_norm: The closing AttnRes RMSNorm (``weight``, ``variance_epsilon``).
            res_proj: The closing AttnRes scorer projection (``weight``).
            out_norm: The model's final RMSNorm (``weight``, ``variance_epsilon``).
            num_blocks: The number of block snapshots the mix reads.
        """
        weights = (res_norm.weight, res_proj.weight, out_norm.weight)
        eps = (res_norm.variance_epsilon, out_norm.variance_epsilon)
        entry = PreparedAttnRes(weights, (num_blocks, True, -1, True), eps)
        entry.bind_ring(self.moe.tail_box.out.buffer)
        return entry


class KimiK3DecodeAttentionLayer:
    """See :meth:`KimiK3DecodeAttention.layer`."""

    def __init__(self, owner, decoder, layer_idx, mla, first, weight) -> None:
        rows = weight.shape[0]
        self.projection = PreparedProjection(
            weight, _projection_tactics(rows, owner.batches)
        ).prepare()
        # The owner's mailbox and two-shot buffers back this layer's exit.
        self.owner, self.rows = owner, rows
        self.entry = PreparedAttnRes.for_target(decoder, has_delta=not first)
        self.entry.bind_ring(owner.moe.tail_box.out.buffer)
        target = SimpleNamespace(
            self_attn=decoder.self_attn,
            mlp_res_norm=decoder.mlp_res_norm,
            mlp_res_proj=decoder.mlp_res_proj,
            post_attention_layernorm=decoder.post_attention_layernorm,
            layer_idx=layer_idx,
            prev_valid_blocks=decoder.prev_valid_blocks,
            is_block_write_layer=decoder.is_block_write_layer,
        )
        self.exit = PreparedOproj(
            target, owner.mailbox, owner.batches, mla, twoshot=owner.twoshot
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """The packed input projection [B, rows] of the attention input ``x``."""
        return self.projection.run(x, x.new_empty((x.shape[0], self.rows)))

    def enter(self, delta, prefix, blocks, out, *, ring: bool) -> torch.Tensor:
        """AttnRes into ``out``: add ``delta`` (or the MoE ring row) to ``prefix``.

        ``blocks`` is token-major [B, num_blocks, 7168].
        """
        if ring:
            return self.entry.run_ring(prefix, blocks, out)
        return self.entry.run(delta, prefix, blocks, out)

    def leave(self, local, prefix, blocks, out) -> torch.Tensor:
        """O-projection of ``local`` [B, K], its all-reduce into ``prefix`` and the
        post-attention AttnRes into ``out``, the feed-forward input."""
        self.exit.run(local, prefix, blocks, out)
        return out
