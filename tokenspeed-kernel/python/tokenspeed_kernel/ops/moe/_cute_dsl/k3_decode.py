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
stream, which the forward rejoins at its end (:meth:`KimiK3DecodeMoE.join`).
Either a ring kernel adds that row to the residual, or the next layer's
AttnRes entry reads it: per decoder layer, the entry mixes the residual into
the attention input, a split-K GEMM runs the input projection, and the
o-projection exit all-reduces the partials through a mailbox and applies the
post-attention AttnRes. At a few rows a side stream prefetches the weights the
next kernels read into L2 (``k3_decode_prefetch``). Every rank must run the
chain for the same row counts, since the mailboxes are collective.
"""

from __future__ import annotations

import ctypes
import functools
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass.cute.runtime import make_fake_stream
from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode_prefetch import (
    ahead_table,
    attention_moe_batches,
    experts_ahead_caps,
    next_attention_groups,
    oproj_down_batches,
    weight_ranges,
)
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
    _checked,
    fake,
    resolve_kernel,
    stream,
    to_cute,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.mailbox import poll, rearm
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.prefetch import (
    compile_kernel,
)
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
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.moe.experts import PREFETCH_ROWS
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


class _WeightPrefetch:
    """One prefetch launch.

    It holds its weights (kept alive), their (address, bytes) device table and
    the compiled kernel. A launch with a shared-memory carveout or the greatest
    stream priority goes through the driver, since CuTe's launcher drops both
    attributes.
    """

    def __init__(self, weights, ranges, *, grid, carveout, priority) -> None:
        self.weights = tuple(weights)
        device = self.weights[0].device
        self.table = torch.tensor(ranges, dtype=torch.int64, device=device)
        self.compiled = compile_kernel(len(ranges), grid, carveout, device.index)
        self.view = to_cute(self.table, 16)
        self.executor, self.function = resolve_kernel(self.compiled, device.index)
        # The driver launch's only kernel parameter: the table's address.
        self.pointer = ctypes.c_void_p(self.table.data_ptr())
        self.arguments = (ctypes.c_void_p * 1)(ctypes.addressof(self.pointer))
        ids, self.attributes = cuda.CUlaunchAttributeID, []
        if priority:
            top = cuda.CUlaunchAttribute()
            top.id = ids.CU_LAUNCH_ATTRIBUTE_PRIORITY
            top.value.priority = _checked(cuda.cuCtxGetStreamPriorityRange())[1]
            self.attributes.append(top)
        if carveout is not None:
            smem = cuda.CUlaunchAttribute()
            smem.id = ids.CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            smem.value.sharedMemCarveout = carveout
            self.attributes.append(smem)
        self.config: cuda.CUlaunchConfig | None = None
        if self.attributes:
            config = self.config = cuda.CUlaunchConfig()
            config.gridDimX, config.gridDimY, config.gridDimZ = grid, 1, 1
            config.blockDimX, config.blockDimY, config.blockDimZ = 32, 1, 1
            config.sharedMemBytes, config.attrs = 0, self.attributes
            config.numAttrs = len(self.attributes)

    def launch(self, side: torch.cuda.Stream) -> None:
        handle = cuda.CUstream(side.cuda_stream)
        if self.config is None:
            self.compiled(self.view, handle)
            return
        self.config.hStream = handle
        address = ctypes.addressof(self.arguments)
        _checked(cuda.cuLaunchKernelEx(self.config, self.function, address, 0))


class KimiK3DecodePrefetch:
    """The decode chain's L2 prefetches of weight ranges.

    Launches on one side stream, by (point, decoder layer, row count), and the
    routed experts' in-kernel prefetch tables.

    The points: ``oproj_down`` after the o-projection exit, ``attention_moe``
    after the KDA input projection or the MLA core, ``next_attention`` once the
    routed experts launch (``k3_decode_prefetch`` has the row counts). A fork
    starts after the main stream's work so far, and :meth:`join` rejoins the
    side stream. The tables hold weight addresses: the weights must keep them
    while the chain is bound.
    """

    def __init__(self, device: torch.device) -> None:
        with torch.accelerator.device_index(device.index):
            self.stream = torch.cuda.Stream(device=device)
        self.device = device
        self.forked = torch.cuda.Event()
        self.pending = False  # side-stream work the main stream has not joined
        self.launches: dict[tuple[str, int, int], _WeightPrefetch] = {}
        self.tables: dict[tuple[int, int], torch.Tensor] = {}
        self.table_weights: list[tuple[torch.Tensor, ...]] = []

    def add(
        self,
        point: str,
        layer_idx: int,
        batches: tuple[int, ...],
        weights: tuple[torch.Tensor, ...],
        *,
        byte_cap: int,
        grid: int,
        carveout: int | None,
        priority: bool,
    ) -> None:
        """Prefetch ``weights`` at ``point`` of layer ``layer_idx``.

        The launch covers ``byte_cap`` bytes (0 for all) for each of ``batches``
        rows, on ``grid`` one-warp CTAs; ``priority`` launches at the greatest
        stream priority.
        """
        ranges = weight_ranges(weights, byte_cap)
        if not batches or not ranges:
            return
        launch = _WeightPrefetch(
            weights, ranges, grid=grid, carveout=carveout, priority=priority
        )
        for batch in batches:
            self.launches[point, layer_idx, batch] = launch

    def add_ahead(
        self,
        layer_idx: int,
        caps: dict[int, int],
        first: tuple[torch.Tensor, ...],
        weights: tuple[torch.Tensor, ...],
    ) -> None:
        """The routed experts' tables of layer ``layer_idx``.

        Each covers ``first`` whole, then ``weights`` up to the byte cap of its
        row count in ``caps``.
        """
        for batch, cap in caps.items():
            table = ahead_table(first, weights, cap, PREFETCH_ROWS, self.device)
            self.tables[layer_idx, batch] = table
        self.table_weights.append((*first, *weights))

    def ahead(self, layer_idx: int, batch: int) -> torch.Tensor | None:
        """The experts' [PREFETCH_ROWS, 2] table of (layer, rows), or None."""
        return self.tables.get((layer_idx, batch))

    def fork(self, point: str, layer_idx: int, batch: int) -> None:
        """Launch the prefetch of (``point``, layer, rows), if any.

        It runs on the side stream after the current stream's work so far.
        """
        launch = self.launches.get((point, layer_idx, batch))
        if launch is None:
            return
        self.forked.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(self.forked)
        self.pending = True
        launch.launch(self.stream)

    def join(self) -> None:
        """Make the current stream wait for the side stream's prefetches."""
        if self.pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.pending = False


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
        self.prefetch = KimiK3DecodePrefetch(self.device)

    def join(self) -> None:
        """Make the current stream wait for the chain's side-stream work.

        Call once at the end of every forward that ran the chain: the L2
        prefetches run on a side stream, and so does the shared expert at the
        row counts whose routing releases the experts early, which the tail
        never joins; a CUDA graph capture fails while a forked stream is not
        joined.
        """
        self.prefetch.join()
        self.state.join()

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
        next_attention: tuple[torch.Tensor, ...],
    ) -> "KimiK3DecodeMoELayer":
        """One MoE layer's chain.

        Args:
            ordinal: The layer's position among the MoE layers, from zero.
            layer_idx: The decoder layer's model index; it keys the layer's
                L2 prefetches.
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
            next_attention: The next decoder layer's attention weights in the
                order a decode step reads them, prefetched into L2 while this
                layer's experts run; empty for the last layer. Each must
                start 16-byte aligned and keep its address while bound.
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
            next_attention=next_attention,
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
        # The weights the front GEMM, shared expert and tail read, in that order.
        self.non_expert_weights = (self.packed, w["shared_gate_up"], w["shared_down"])
        self.non_expert_weights += (w["up_shard"],)
        prefetch, following = owner.prefetch, w["next_attention"]
        if following:
            for (cap, carveout), rows in next_attention_groups(batches).items():
                prefetch.add(
                    "next_attention",
                    layer_idx,
                    rows,
                    following,
                    byte_cap=cap,
                    grid=32,
                    carveout=carveout,
                    priority=False,
                )
            # The tail's up-projection shard, which it reads right after the experts.
            caps = experts_ahead_caps(batches)
            prefetch.add_ahead(layer_idx, caps, (w["up_shard"],), following)
        self.chain = PreparedMoE(
            front, experts, tail, state, prefetch=prefetch, layer_idx=layer_idx
        )
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
        moe: KimiK3DecodeMoELayer | None,
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
            moe: The layer's MoE chain, whose weights (and the o-projection
                shard) are prefetched into L2 while the attention runs; None
                for a dense layer.
        """
        return KimiK3DecodeAttentionLayer(
            self, decoder, layer_idx, mla, first, projection_weight, moe
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

    def __init__(self, owner, decoder, layer_idx, mla, first, weight, moe) -> None:
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
        self.prefetch, self.layer_idx, self.mla = owner.moe.prefetch, layer_idx, mla
        if moe is not None:  # both on the critical path: urgent launches
            urgent = dict(byte_cap=0, carveout=100, priority=True)
            rows = oproj_down_batches(owner.batches)
            down = (moe.packed,)
            self.prefetch.add("oproj_down", layer_idx, rows, down, grid=8, **urgent)
            rows = attention_moe_batches(mla, owner.batches)
            weights = (decoder.self_attn.o_proj.weight, *moe.non_expert_weights)
            self.prefetch.add(
                "attention_moe", layer_idx, rows, weights, grid=32, **urgent
            )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """The packed input projection [B, rows] of the attention input ``x``."""
        projected = self.projection.run(x, x.new_empty((x.shape[0], self.rows)))
        if not self.mla:  # beside the KDA recurrence
            self.prefetch.fork("attention_moe", self.layer_idx, x.shape[0])
        return projected

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
        if self.mla:  # the MLA core is done
            self.prefetch.fork("attention_moe", self.layer_idx, local.shape[0])
        self.exit.run(local, prefix, blocks, out)
        self.prefetch.fork("oproj_down", self.layer_idx, local.shape[0])
        return out
