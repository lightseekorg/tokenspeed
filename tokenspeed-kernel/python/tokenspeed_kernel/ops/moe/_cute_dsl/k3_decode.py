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

"""Kimi-K3 TP8 decode MoE chain over Lamport mailboxes.

Per MoE layer: one GEMM publishes this rank's router logits and latent-down
block to every rank, one kernel routes top-16 and gathers the latent, the
persistent experts compute and combine their rows and publish the latent
partial, and one tail kernel all-reduces it, normalizes, up-projects, adds the
shared expert and publishes the output row. A ring kernel adds that row to the
residual. The shared expert runs on a side stream. Every rank must run the
chain for the same batches, since the mailboxes are collective.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass.cute.runtime import make_fake_stream
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

HIDDEN, LATENT, EXPERTS = 7168, 3584, 896
# Decode batches the chain is compiled for; others take the ordinary MoE.
BATCHES = (*range(1, 9), 16, 32, 64)


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
        moe_layers: int,
        beta: float,
        linear_beta: float,
    ) -> None:
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
        """One MoE layer's chain; ``ordinal`` counts MoE layers from zero and
        ``layer_idx`` is the decoder layer's model index.

        Args:
            router: [896, 7168] BF16 router weight (every expert).
            correction_bias: [896] FP32 routing bias.
            down_shard: [448, 7168] BF16 rows of the latent-down weight this
                rank owns.
            shared_gate_up: [1536, 7168] BF16 gate then up rows of this rank's
                shared-expert shard.
            shared_down: [7168, 768] BF16 shared-expert down shard.
            w13, w13_scale, w2, w2_scale: routed MXFP4 experts in the
                TRT-LLM-gen serving layout.
            norm_weight: [3584] BF16 latent RMSNorm weight.
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
