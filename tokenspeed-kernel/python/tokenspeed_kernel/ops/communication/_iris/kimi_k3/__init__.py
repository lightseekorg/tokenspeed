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

"""Kimi-K3's prepared Iris fusions, shared by sequential decoder layers."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication.iris import IrisAllReduce


# Decode rows served by the fused replicated paths: the latent input gather,
# the Lamport two-shot reduce and the gathered MoE tail. Above it, rows shard.
# 192 covers 48 requests' 4-token verify; the model's gather-tail and SiTU
# warp-decode bounds and the comm's sharded-prefix floor follow it.
K3_FUSED_DECODE_MAX_ROWS = 192
# Decode rows with a gathered column-parallel MoE up-projection; GEMV senders to 4 rows.
_K3_GATHER_MAX_ROWS = K3_FUSED_DECODE_MAX_ROWS
_K3_GATHER_GEMV_MAX_ROWS = 4
_K3_GATHER_LATENT = 3584
_K3_GATHER_LOCAL_WIDTH = 896
_K3_GATHER_NUM_WARPS = 8
_K3_GATHER_WAVES_PER_OUTPUT = 2
_K3_GATHER_BLOCK_K = 512
_K3_GATHER_MFMA_CHUNK = 64
# The decode MoE input projection: [router; routed; shared gate; shared up].
_K3_HIDDEN = 7168
_K3_ROUTER = 896
_K3_LATENT = 3584
_K3_SHARED = 768
_K3_PACKED_INPUT_ROWS = _K3_ROUTER + _K3_LATENT + 2 * _K3_SHARED
_K3_LATENT_GEMM_NUM_WARPS = 2
# One-shot Lamport window (it also sums MoE top-k slots) and latent gather row limits.
_K3_LAMPORT_MAX_ROWS = 7
_K3_LATENT_GATHER_MAX_ROWS = K3_FUSED_DECODE_MAX_ROWS
# The FP8 scale rendezvous spans its slots with gl.arange, so a power of two.
_K3_LATENT_SYNC_ROWS = 1 << (_K3_LATENT_GATHER_MAX_ROWS - 1).bit_length()
# From 8 rows the Lamport two-shot all-reduce wins; 1024-element tiles need even rows.
_K3_TWO_SHOT_MIN_ROWS = 8
_K3_TWO_SHOT_MAX_ROWS = K3_FUSED_DECODE_MAX_ROWS
_K3_TWO_SHOT_ALIGN = 8 * 128


def _k3_lamport_numel(producer_direct_max_numel: int) -> int:
    """Elements of K3's Lamport mailbox: the routed/shared all-reduce's
    window, and the decode MoE input gather's 1024 elements a row through
    its rows where the producer buffer holds them."""
    rows = min(producer_direct_max_numel // (3584 + 7168), _K3_LAMPORT_MAX_ROWS)
    if not rows:
        return 0
    gather = max(
        _K3_LATENT_GATHER_MAX_ROWS * 1024,
        -(-_K3_GATHER_MAX_ROWS // 16) * 16 * _K3_GATHER_LOCAL_WIDTH,
    )
    return max(
        rows * (3584 + 7168),
        min(gather, producer_direct_max_numel // 512 * 512),
    )


def _k3_two_shot_numel(producer_direct_max_numel: int) -> int:
    """Elements of the MoE all-reduce's Lamport two-shot capacity, or 0."""
    numel = min(
        _K3_TWO_SHOT_MAX_ROWS * (3584 + 7168),
        producer_direct_max_numel // _K3_TWO_SHOT_ALIGN * _K3_TWO_SHOT_ALIGN,
    )
    return numel if numel >= _K3_TWO_SHOT_MIN_ROWS * (3584 + 7168) else 0


def _k3_two_shot_heap_bytes(producer_direct_max_numel: int) -> int:
    from tokenspeed_kernel.ops.communication._iris import allocation_bytes

    numel = _k3_two_shot_numel(producer_direct_max_numel)
    return allocation_bytes((3, 2, 8, numel // 8), torch.bfloat16) if numel else 0


def allreduce_residual_attnres_max_tokens(world_size: int) -> int:
    """Return K3's measured fused AttnRes window for this group size."""
    return 16 if current_platform().is_cdna4 and world_size == 8 else 0


class K3IrisState:
    """Retain one group's all-reduce and its optional K3 fusion storage.

    Producers, attention mixing, and the MoE tail use these buffers in stream
    order. A sharded activation is a view of state storage overwritten by
    the next fusion; its local residual shard remains caller-owned.
    """

    def __init__(
        self,
        all_reduce: "IrisAllReduce",
        *,
        attnres_max_numel: int,
        attnres_max_rows: int,
        enable_lamport: bool,
        sharded_max_rows: int,
        pull_exchange_max_bytes: int = 0,
    ):
        from tokenspeed_kernel.ops.communication._iris.all_reduce import ProducerLamport
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.allreduce_sharded import (
            IrisAllReduceShardedState,
        )
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.attnres import (
            IrisAttnResState,
        )
        from tokenspeed_kernel.ops.communication._iris.pull_exchange import (
            IrisPullExchange,
        )

        self.all_reduce = all_reduce
        # K3.1's KVP decode gathers and records exchange, pulled in place.
        self.pull_exchange = (
            IrisPullExchange(all_reduce.peers, pull_exchange_max_bytes)
            if pull_exchange_max_bytes
            else None
        )
        self.enable_lamport = enable_lamport
        self.attnres = (
            IrisAttnResState(
                all_reduce.peers, attnres_max_numel, attnres_max_rows, all_reduce.dtype
            )
            if attnres_max_numel
            else None
        )
        # Only the K3 TP8 routed/shared pair uses this measured Lamport window.
        self.lamport_max_rows = (
            min(
                all_reduce.producer_direct_max_numel // (3584 + 7168),
                _K3_LAMPORT_MAX_ROWS,
            )
            if enable_lamport
            and ProducerLamport.supported(all_reduce.world_size, all_reduce.dtype)
            else 0
        )
        self.lamport = (
            ProducerLamport(
                all_reduce.producer,
                max_numel=_k3_lamport_numel(all_reduce.producer_direct_max_numel),
            )
            if self.lamport_max_rows
            else None
        )
        self.sharded = (
            IrisAllReduceShardedState(
                all_reduce.producer_pull, max_rows=sharded_max_rows
            )
            if sharded_max_rows
            else None
        )
        # [stage][scatter, gather][source][shard] mailboxes, one counter per program.
        two_shot = (
            _k3_two_shot_numel(all_reduce.producer_direct_max_numel)
            if self.lamport is not None and all_reduce.world_size == 8
            else 0
        )
        self.two_shot_max_numel = two_shot
        self.two_shot_region = None
        self.two_shot_epochs = None
        self.two_shot_peer_addresses = None
        if two_shot:
            peers = all_reduce.peers
            self.two_shot_region = peers.ctx.empty(
                (3, 2, 8, two_shot // 8), dtype=torch.bfloat16
            )
            self.two_shot_region.view(torch.int32).fill_(-2147483648)
            self.two_shot_epochs = torch.zeros(
                two_shot // _K3_TWO_SHOT_ALIGN, dtype=torch.int32, device=peers.device
            )
            offset = self.two_shot_region.data_ptr() - peers.addresses[peers.rank]
            self.two_shot_peer_addresses = tuple(
                base + offset for base in peers.addresses[:8]
            )
        # FP8 scale rendezvous: -1 row maxima, zero counter; every launch restores this.
        self._latent_fp8_sync = (
            torch.cat(
                (
                    torch.full(
                        (_K3_LATENT_SYNC_ROWS,),
                        -1,
                        dtype=torch.int32,
                        device=all_reduce.device,
                    ),
                    torch.zeros(1, dtype=torch.int32, device=all_reduce.device),
                )
            )
            if self.lamport is not None
            else None
        )

    def _lamport_moe_shapes(self, shapes: tuple[tuple[int, ...], ...]) -> bool:
        return (
            self.lamport is not None
            and len(shapes) == 2
            and len(shapes[0]) == len(shapes[1]) == 2
            and 0 < shapes[0][0] == shapes[1][0] <= self.lamport_max_rows
            and (shapes[0][1], shapes[1][1]) in ((3584, 7168), (7168, 3584))
        )

    def _two_shot_moe_shapes(self, shapes: tuple[tuple[int, ...], ...]) -> bool:
        if (
            self.two_shot_region is None
            or len(shapes) != 2
            or len(shapes[0]) != 2
            or len(shapes[1]) != 2
            or shapes[0][0] != shapes[1][0]
            or (shapes[0][1], shapes[1][1]) not in ((3584, 7168), (7168, 3584))
        ):
            return False
        rows = shapes[0][0]
        numel = rows * (3584 + 7168)
        return (
            _K3_TWO_SHOT_MIN_ROWS <= rows
            and numel <= self.two_shot_max_numel
            and numel % _K3_TWO_SHOT_ALIGN == 0
        )

    def _run_two_shot(
        self, numel: int, source: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Lamport two-shot all-reduce of the producer's first ``numel``
        elements, or of the local contiguous ``source``."""
        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            iris_arch_policy,
            iris_lamport_twoshot_bf16_allreduce,
        )

        output = torch.empty(
            numel, dtype=self.all_reduce.dtype, device=self.all_reduce.device
        )
        iris_lamport_twoshot_bf16_allreduce[(numel // _K3_TWO_SHOT_ALIGN,)](
            self.all_reduce.producer.data if source is None else source,
            self.two_shot_region,
            output,
            self.two_shot_epochs,
            self.two_shot_peer_addresses,
            RANK=self.all_reduce.rank_in_group,
            WORLD_SIZE=8,
            total_elements=numel,
            MAX_SHARD_ELEMENTS=self.two_shot_max_numel // 8,
            NUM_STAGES=3,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            num_warps=1,
        )
        return output

    def reduce_attention(self, partial: torch.Tensor) -> torch.Tensor | None:
        """Lamport all-reduce of a contiguous ``[rows, 7168]`` attention
        partial into new storage, read where it lies (one-shot below 8 rows,
        two-shot from 8), or None to use the ordinary one."""
        if partial.ndim != 2 or not partial.is_contiguous():
            return None
        return self._reduce_attention_lamport(partial)

    def can_reduce_moe_topk_partials(self, shapes: tuple[tuple[int, ...], ...]) -> bool:
        """Whether ``reduce_moe`` takes routed top-k partials for these outputs."""
        return (
            self._lamport_moe_shapes(shapes) or self._two_shot_moe_shapes(shapes)
        ) and shapes[0][1] == 3584

    def reduce_moe(
        self,
        tensors: tuple[torch.Tensor, ...],
        *,
        routed_partials: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Reduce owned producer outputs using K3's existing shape policy.

        With ``routed_partials``, contiguous BF16 ``[rows * top_k, routed]``
        rows of the MoE's top-k slots, the routed output (``tensors[0]``) is
        their slot-order sum, as the MoE's own top-k reduce would have
        written it; ``tensors[0]`` itself is not read.
        """
        assert self.all_reduce.owns_outputs(tensors)
        shapes = tuple(tuple(tensor.shape) for tensor in tensors)
        if routed_partials is not None:
            rows, width = shapes[0]
            if not (
                self.can_reduce_moe_topk_partials(shapes)
                and routed_partials.dtype == self.all_reduce.dtype
                and routed_partials.is_contiguous()
                and routed_partials.ndim == 2
                and routed_partials.shape[1] == width
                and routed_partials.shape[0] % rows == 0
            ):
                raise ValueError("routed top-k partials need the Lamport MoE shapes")
        if self._lamport_moe_shapes(shapes):
            output = torch.empty(
                sum(t.numel() for t in tensors),
                dtype=self.all_reduce.dtype,
                device=self.all_reduce.device,
            )
            if routed_partials is None:
                self.lamport.run(output)
            else:
                self._run_lamport_topk(output, routed_partials, rows, width)
            return self.all_reduce.producer.views(output, shapes)
        if self._two_shot_moe_shapes(shapes):
            numel = sum(t.numel() for t in tensors)
            if routed_partials is None:
                output = self._run_two_shot(numel)
            else:
                output = self._run_two_shot_topk(numel, routed_partials, rows, width)
            return self.all_reduce.producer.views(output, shapes)
        return self.all_reduce.all_reduce_symmetric(tensors)

    def _run_two_shot_topk(
        self, numel: int, partials: torch.Tensor, rows: int, width: int
    ) -> torch.Tensor:
        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            iris_arch_policy,
        )
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.moe_reduce import (
            iris_k3moe_lamport_twoshot_topk,
        )

        output = torch.empty(
            numel, dtype=self.all_reduce.dtype, device=self.all_reduce.device
        )
        iris_k3moe_lamport_twoshot_topk[(numel // _K3_TWO_SHOT_ALIGN,)](
            partials,
            self.all_reduce.producer.data,
            self.two_shot_region,
            output,
            self.two_shot_epochs,
            self.two_shot_peer_addresses,
            RANK=self.all_reduce.rank_in_group,
            WORLD_SIZE=8,
            TOTAL_ELEMENTS=numel,
            ROUTED_ELEMENTS=rows * width,
            ROUTED_WIDTH=width,
            TOPK=partials.shape[0] // rows,
            MAX_SHARD_ELEMENTS=self.two_shot_max_numel // 8,
            NUM_STAGES=3,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            num_warps=1,
        )
        return output

    def _run_lamport_topk(
        self, output: torch.Tensor, partials: torch.Tensor, rows: int, width: int
    ) -> None:
        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            iris_arch_policy,
        )
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.moe_reduce import (
            iris_k3moe_lamport_allreduce_topk,
        )

        lamport = self.lamport
        if (
            rows < 1
            or width % lamport.BLOCK_ELEMENTS
            or output.numel() % lamport.BLOCK_ELEMENTS
            or not rows * width < output.numel() <= lamport.max_numel
        ):
            raise ValueError(
                "routed Lamport output must fit its mailbox in complete tiles"
            )
        iris_k3moe_lamport_allreduce_topk[(output.numel() // lamport.BLOCK_ELEMENTS,)](
            partials,
            lamport.producer.data,
            lamport.mailbox,
            output,
            lamport.epochs,
            lamport.peer_mailboxes,
            RANK=lamport.producer.peers.rank,
            WORLD_SIZE=lamport.producer.peers.world_size,
            TOTAL_ELEMENTS=output.numel(),
            ROUTED_ELEMENTS=rows * width,
            ROUTED_WIDTH=width,
            TOPK=partials.shape[0] // rows,
            MAX_ELEMENTS=lamport.max_numel,
            NUM_STAGES=lamport.NUM_STAGES,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            num_warps=lamport.NUM_SUBGROUPS,
        )

    def can_moe_gather_tail(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
        prefix: torch.Tensor,
        projection_weight: torch.Tensor,
        norm_weight: torch.Tensor | None,
    ) -> bool:
        """Whether ``moe_gather_tail`` serves these reduced decode rows."""
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.moetail_gather import (
            K3_GATHER_ROW_STRIDE,
        )

        rows = routed.shape[0] if routed.ndim == 2 else 0
        width = self.all_reduce.world_size * _K3_GATHER_LOCAL_WIDTH
        return (
            self.lamport is not None
            and self.all_reduce.world_size == 8
            and 1 <= rows <= _K3_GATHER_MAX_ROWS
            and (
                (1 << (rows - 1).bit_length()) * K3_GATHER_ROW_STRIDE
                if rows <= _K3_GATHER_GEMV_MAX_ROWS
                else -(-rows // 16) * 16 * _K3_GATHER_LOCAL_WIDTH
            )
            <= self.lamport.max_numel
            and routed.shape == (rows, _K3_GATHER_LATENT)
            and shared.shape == prefix.shape == (rows, width)
            and projection_weight.shape == (width, _K3_GATHER_LATENT)
            and projection_weight.is_contiguous()
            and all(
                t.dtype == torch.bfloat16
                and t.device == self.all_reduce.device
                and t.stride(-1) == 1
                # The kernel keeps 16-byte row vectors on this promise.
                and t.stride(0) % 8 == 0
                for t in (routed, shared, prefix, projection_weight)
            )
            and (
                norm_weight is None
                or (
                    norm_weight.shape == (_K3_GATHER_LATENT,)
                    and norm_weight.dtype == torch.bfloat16
                    and norm_weight.is_contiguous()
                )
            )
        )

    def moe_gather_tail(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
        prefix: torch.Tensor,
        projection_weight: torch.Tensor,
        *,
        norm_weight: torch.Tensor | None,
        eps: float | None,
    ) -> torch.Tensor:
        """Return ``prefix + norm(routed) @ projection_weight.T + shared`` for decode rows.

        ``routed`` and ``shared`` are the reduced latent and shared-expert rows,
        replicated on every rank. Each rank projects its column block of the
        replicated weight and the blocks are gathered through the Lamport
        mailbox, so the call is collective. Rows match RMSNorm followed by
        the fused MFMA add3 epilogue up to FP32 summation order (the norm's
        sum of squares and the projection).
        """
        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            iris_arch_policy,
        )
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.moetail_gather import (
            K3_GATHER_ROW_STRIDE,
            iris_k3moe_gemv_lamport_gather,
            iris_k3moe_mfma_lamport_gather,
        )

        assert self.can_moe_gather_tail(
            routed, shared, prefix, projection_weight, norm_weight
        )
        if norm_weight is not None and (eps is None or eps <= 0.0):
            raise ValueError("the fused RMSNorm needs a positive epsilon")
        rows = routed.shape[0]
        world_size = self.all_reduce.world_size
        rank = self.all_reduce.rank_in_group
        local_width = _K3_GATHER_LOCAL_WIDTH
        out = torch.empty(
            (rows, world_size * local_width), dtype=routed.dtype, device=routed.device
        )
        lamport = self.lamport
        args = (
            routed,
            norm_weight if norm_weight is not None else routed,
            projection_weight[rank * local_width : (rank + 1) * local_width],
            prefix,
            shared,
            out,
            lamport.mailbox,
            lamport.epochs,
            lamport.peer_mailboxes,
            rows,
            routed.stride(0),
            prefix.stride(0),
            shared.stride(0),
            float(eps) if norm_weight is not None else 0.0,
        )
        constants = dict(
            RANK=rank,
            WORLD_SIZE=world_size,
            K=_K3_GATHER_LATENT,
            LOCAL_WIDTH=local_width,
            MAX_ELEMENTS=lamport.max_numel,
            HAS_NORM=norm_weight is not None,
            NUM_WARPS=_K3_GATHER_NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            num_warps=_K3_GATHER_NUM_WARPS,
        )
        if rows <= _K3_GATHER_GEMV_MAX_ROWS:
            outputs = _K3_GATHER_NUM_WARPS // _K3_GATHER_WAVES_PER_OUTPUT
            iris_k3moe_gemv_lamport_gather[(local_width // outputs + rows,)](
                *args,
                ROW_STRIDE=K3_GATHER_ROW_STRIDE,
                BLOCK_M=1 << (rows - 1).bit_length(),
                BLOCK_K=_K3_GATHER_BLOCK_K,
                WAVES_PER_OUTPUT=_K3_GATHER_WAVES_PER_OUTPUT,
                **constants,
            )
        else:
            # Per 16-row tile: a sender per 16 columns, a reducer per 1024 slot items.
            iris_k3moe_mfma_lamport_gather[
                (-(-rows // 16) * (local_width // 16 + 16 * local_width // 1024),)
            ](*args, CHUNK=_K3_GATHER_MFMA_CHUNK, **constants)
        return out

    def can_latent_input_gather(
        self, hidden_states: torch.Tensor, packed_weight: torch.Tensor
    ) -> bool:
        """Whether ``latent_input_gather`` serves these decode rows."""
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.latent_gather import (
            K3_LATENT_GATHER_ROW_WORDS,
        )

        rows = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
        return (
            self.lamport is not None
            and current_platform().is_cdna4
            and self.all_reduce.world_size == 8
            and 2
            <= rows
            <= min(
                _K3_LATENT_GATHER_MAX_ROWS,
                self.lamport.max_numel // (2 * K3_LATENT_GATHER_ROW_WORDS),
            )
            and hidden_states.shape[1] == _K3_HIDDEN
            and packed_weight.shape == (_K3_PACKED_INPUT_ROWS, _K3_HIDDEN)
            and hidden_states.is_contiguous()
            and packed_weight.is_contiguous()
            and hidden_states.dtype == packed_weight.dtype == torch.bfloat16
            and hidden_states.device == packed_weight.device == self.all_reduce.device
        )

    def latent_input_gather(
        self,
        hidden_states: torch.Tensor,
        packed_weight: torch.Tensor,
        *,
        beta: float,
        linear_beta: float | None,
        quantize: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """K3's router, routed-latent and shared inputs with sharded columns.

        Returns what ``launch_gluon_latent_input_small_batch_gfx950`` returns
        for the packed ``[router; routed; shared gate; shared up]`` weight, bit
        for bit except that a negative-zero router logit arrives as zero: each
        rank projects its eighth of the router and routed columns with the same
        split-K GEMM, and the blocks are gathered through the Lamport mailbox,
        so the call is collective. ``quantize`` appends the routed latent's
        ``(fp8, scale)`` as the gfx950 MoE's dynamic per-tensor quantizer
        computes them.
        """
        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            iris_arch_policy,
        )
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.latent_gather import (
            iris_k3moe_latent_input_gather,
        )
        from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import (
            latent_input_small_batch as small_batch,
        )
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.quantize import (
            _DYNAMIC_FP8_E4M3_MAX,
            _DYNAMIC_FP8_MIN_AMAX,
        )

        assert self.can_latent_input_gather(hidden_states, packed_weight)
        if beta <= 0.0 or (linear_beta is not None and linear_beta <= 0.0):
            raise ValueError("SiTU beta values must be positive")
        rows = hidden_states.shape[0]
        world_size = self.all_reduce.world_size
        rank = self.all_reduce.rank_in_group
        device = hidden_states.device
        block = small_batch._BLOCK_N
        router_local = _K3_ROUTER // world_size
        latent_local = _K3_LATENT // world_size
        latent_tiles = -(-latent_local // block)
        shared_tiles = 2 * _K3_SHARED // block
        # The replicated launch's tiles and split, so every column keeps its bits.
        block_m, warps_m = (
            (small_batch._BLOCK_M, 1)
            if rows <= small_batch._BLOCK_M
            else (small_batch._LARGE_BLOCK_M, 2)
        )
        split_k = small_batch._split_k(rows, _K3_PACKED_INPUT_ROWS, _K3_HIDDEN, block_m)
        partials = torch.empty(
            (split_k, rows, _K3_PACKED_INPUT_ROWS), dtype=torch.float32, device=device
        )
        split_stride = rows * _K3_PACKED_INPUT_ROWS
        # Narrow tiles spread a rank's columns; split-K sums do not depend on the tile.
        gemm_block = 32 if rows <= small_batch._BLOCK_M else 64
        gemm_router_tiles = -(-router_local // gemm_block)
        gemm_latent_tiles = -(-latent_local // gemm_block)
        gemm_shared_tiles = 2 * _K3_SHARED // gemm_block
        gemm_tiles = gemm_router_tiles + gemm_latent_tiles + gemm_shared_tiles
        small_batch.gluon_latent_input_small_batch_gfx950[
            (-(-rows // block_m) * split_k * gemm_tiles,)
        ](
            hidden_states,
            packed_weight,
            partials,
            rows,
            split_stride,
            HIDDEN=_K3_HIDDEN,
            TOTAL_N=_K3_PACKED_INPUT_ROWS,
            BLOCK_M=block_m,
            BLOCK_N=gemm_block,
            BLOCK_K=small_batch._BLOCK_K,
            SPLIT_K=split_k,
            NUM_WARPS=_K3_LATENT_GEMM_NUM_WARPS,
            WARPS_M=warps_m,
            REGION0_START=rank * router_local,
            REGION0_TILES=gemm_router_tiles,
            REGION1_START=_K3_ROUTER + rank * latent_local,
            REGION1_TILES=gemm_latent_tiles,
            REGION2_START=_K3_ROUTER + _K3_LATENT,
            REGION2_TILES=gemm_shared_tiles,
            num_warps=_K3_LATENT_GEMM_NUM_WARPS,
        )
        router_out = torch.empty((rows, _K3_ROUTER), dtype=torch.float32, device=device)
        routed_out = torch.empty(
            (rows, _K3_LATENT), dtype=hidden_states.dtype, device=device
        )
        shared_out = torch.empty(
            (rows, _K3_SHARED), dtype=hidden_states.dtype, device=device
        )
        if quantize:
            routed_fp8 = torch.empty(
                (rows, _K3_LATENT), dtype=torch.float8_e4m3fn, device=device
            )
            routed_scale = torch.empty(1, dtype=torch.float32, device=device)
        else:
            routed_fp8, routed_scale = routed_out, router_out
        lamport = self.lamport
        iris_k3moe_latent_input_gather[
            (rows * (1 + latent_tiles + shared_tiles // 2) + rows,)
        ](
            partials,
            shared_out,
            router_out,
            routed_out,
            routed_fp8,
            self._latent_fp8_sync,
            routed_scale,
            lamport.mailbox,
            lamport.epochs,
            lamport.peer_mailboxes,
            float(beta),
            1.0 if linear_beta is None else float(linear_beta),
            rows,
            split_stride,
            RANK=rank,
            WORLD_SIZE=world_size,
            SPLIT_K=split_k,
            TOTAL_N=_K3_PACKED_INPUT_ROWS,
            ROUTER_N=_K3_ROUTER,
            LATENT_N=_K3_LATENT,
            SHARED_N=_K3_SHARED,
            HAS_LINEAR_BETA=linear_beta is not None,
            MAX_ELEMENTS=lamport.max_numel,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            NUM_WARPS=4,
            QUANTIZE=quantize,
            MIN_AMAX=_DYNAMIC_FP8_MIN_AMAX,
            FP8_MAX=_DYNAMIC_FP8_E4M3_MAX,
            MAX_ROWS=_K3_LATENT_SYNC_ROWS,
            num_warps=4,
        )
        if quantize:
            return router_out, routed_out, shared_out, (routed_fp8, routed_scale)
        return router_out, routed_out, shared_out

    def moe_row_sharded_inputs(
        self,
        hidden_states: torch.Tensor,
        packed_weight: torch.Tensor,
        *,
        beta: float,
        linear_beta: float | None,
        route,
        attention_gather,
    ):
        """Project, route and quantize replicated MoE rows once per row, then gather.

        See ``IrisAllReduceShardedState.moe_row_sharded_inputs``; ``None`` when
        the sharded state is absent or declines the shape. ``attention_gather``
        is ``attention_mix``'s pending gather of ``hidden_states``, or None.
        """
        if self.sharded is None:
            return None
        return self.sharded.moe_row_sharded_inputs(
            hidden_states,
            packed_weight,
            beta=beta,
            linear_beta=linear_beta,
            route=route,
            attention_gather=attention_gather,
        )

    def join_gather(self, pending) -> torch.Tensor:
        """Order the calling stream after ``attention_mix``'s deferred gather."""
        return self.sharded.join_gather(pending)

    def gather_prefill_topk(
        self, selection: torch.Tensor, lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """See ``IrisAllReduceShardedState.gather_prefill_topk``; ``None`` without it."""
        if self.sharded is None:
            return None
        return self.sharded.gather_prefill_topk(selection, lens)

    def moe_tail(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix: torch.Tensor,
        projection_weight: torch.Tensor,
        *,
        prefix_is_sharded: bool,
        norm_weight: torch.Tensor | None,
        eps: float | None,
        shared_reduced,
    ) -> torch.Tensor | None:
        """Reduce, normalize, and project owned row shards, then gather the output.

        ``shared_reduced`` is ``project_moe_shared``'s record, or None.
        """
        if self.sharded is None or not self.all_reduce.owns_outputs(
            (routed_partial, shared_partial)
        ):
            self.join_moe_shared(shared_reduced, shared_partial)
            return None
        return self.sharded.moe_tail(
            routed_partial,
            shared_partial,
            prefix,
            projection_weight,
            prefix_is_sharded=prefix_is_sharded,
            norm_weight=norm_weight,
            eps=eps,
            shared_reduced=shared_reduced,
        )

    def moe_tail_mix(
        self,
        routed_partial: torch.Tensor,
        shared_partial: torch.Tensor,
        prefix: torch.Tensor,
        projection_weight: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Finish owned MoE residual rows; gather the next AttnRes mix instead."""
        if self.sharded is None or not self.all_reduce.owns_outputs(
            (routed_partial, shared_partial)
        ):
            self.join_moe_shared(kwargs["shared_reduced"], shared_partial)
            return None
        return self.sharded.moe_tail_mix(
            routed_partial, shared_partial, prefix, projection_weight, **kwargs
        )

    def attention_mix(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor | None,
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        *,
        eps: float,
        out_norm_weight: torch.Tensor,
        out_norm_eps: float,
        num_valid_blocks: int,
        residual_is_sharded: bool,
        published,
        defer_gather: bool,
    ) -> tuple | None:
        """Mix owned attention row shards and gather the normalized activation.

        ``published`` is ``take_published``'s record of this partial, or None.
        With ``defer_gather`` a third value is the pending gather to join.
        """
        if self.sharded is None or not self.all_reduce.owns_outputs((partial,)):
            return None
        return self.sharded.attn_mix(
            partial,
            residual,
            block_residual,
            res_weight,
            rms_weight,
            eps=eps,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            num_valid_blocks=num_valid_blocks,
            residual_is_sharded=residual_is_sharded,
            published=published,
            defer_gather=defer_gather,
        )

    def project_attention(
        self, activation: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
    ) -> bool:
        """Write ``activation @ weight.T`` into ``out``, publishing row tiles.

        Returns False, launching nothing, unless ``out`` is the borrowed producer
        input and ``attention_mix`` can overlap its scatter with this GEMM.
        """
        if self.sharded is None or not self.all_reduce.owns_outputs((out,)):
            return False
        return self.sharded.project_attention(activation, weight, out)

    def project_moe_shared(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        outputs: tuple[torch.Tensor, torch.Tensor],
    ):
        """Write the MoE shared partial into ``outputs[1]`` and reduce it on the side.

        ``outputs`` are the borrowed ``(routed, shared)`` producer inputs. Returns
        a record for ``moe_tail``, ``moe_tail_mix`` or ``join_moe_shared``, or
        None, launching nothing, when the overlap does not apply.
        """
        if self.sharded is None or not self.all_reduce.owns_outputs(outputs):
            return None
        return self.sharded.project_moe_shared(activation, weight, *outputs)

    def join_moe_shared(self, shared_reduced, shared_partial: torch.Tensor) -> None:
        """Order the calling stream after ``project_moe_shared``'s reduction."""
        if shared_reduced is not None:
            self.sharded.join_moe_shared(shared_reduced, shared_partial)

    def take_published(self, partial: torch.Tensor):
        """The last ``project_attention`` record if it wrote ``partial``, else None."""
        if self.sharded is None:
            return None
        return self.sharded.take_published(partial)

    def attnres_supported(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        """Check the prepared fused AttnRes contract without allocating."""
        from tokenspeed_kernel.ops.communication._iris.kimi_k3.attnres import (
            ATTNRES_EPILOGUE_MAX_ROWS,
            attnres_supported,
        )

        if self.attnres is None or partial.ndim != 2:
            return False
        rows = partial.shape[0]
        # Rows past the push one-shot's inbox reduce through the two-shot.
        pushed = (
            partial.numel() <= self.attnres.max_numel and rows <= self.attnres.max_rows
        )
        reduced = (
            _K3_TWO_SHOT_MIN_ROWS <= rows <= ATTNRES_EPILOGUE_MAX_ROWS
            and self.two_shot_region is not None
            and partial.numel() <= self.two_shot_max_numel
            and partial.is_contiguous()
        )
        return (pushed or reduced) and attnres_supported(
            partial,
            residual,
            score_weight,
            output_weight,
            scratch,
            world_size=self.all_reduce.world_size,
            device=self.all_reduce.device,
            max_rows=ATTNRES_EPILOGUE_MAX_ROWS,
        )

    def attnres_combine(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized AttnRes output and the accumulated residual.

        The partial reduces through the Lamport one-shot (below 8 rows) or
        two-shot mailboxes, in the fused kernel's ascending rank order, and
        a row epilogue finishes it; the fused push one-shot serves the rest.
        """
        assert self.attnres is not None
        reduced = (
            self._reduce_attention_lamport(partial) if partial.is_contiguous() else None
        )
        if reduced is not None:
            return self.attnres.epilogue(
                reduced, residual, score_weight, output_weight, scratch, eps
            )
        return self.attnres.run(
            partial, residual, score_weight, output_weight, scratch, eps
        )

    def _reduce_attention_lamport(self, partial: torch.Tensor) -> torch.Tensor | None:
        from tokenspeed_kernel.ops.communication._iris.all_reduce import (
            iris_arch_policy,
            iris_lamport_allreduce,
        )

        rows = partial.shape[0]
        numel = partial.numel()
        if (
            partial.ndim != 2
            or partial.shape[1] != _K3_HIDDEN
            or partial.dtype != self.all_reduce.dtype
        ):
            return None
        if (
            rows >= _K3_TWO_SHOT_MIN_ROWS
            and self.two_shot_region is not None
            and numel <= self.two_shot_max_numel
        ):
            return self._run_two_shot(numel, partial).view(rows, _K3_HIDDEN)
        lamport = self.lamport
        if lamport is None or numel > lamport.max_numel:
            return None
        output = torch.empty_like(partial)
        iris_lamport_allreduce[(numel // lamport.BLOCK_ELEMENTS,)](
            partial,
            lamport.mailbox,
            output,
            lamport.epochs,
            *lamport.peer_mailboxes,
            RANK=lamport.producer.peers.rank,
            WORLD_SIZE=lamport.producer.peers.world_size,
            TOTAL_ELEMENTS=numel,
            MAX_ELEMENTS=lamport.max_numel,
            NUM_STAGES=lamport.NUM_STAGES,
            PEER_SIGNAL_STORE_CACHE=iris_arch_policy().peer_signal_store,
            num_warps=lamport.NUM_SUBGROUPS,
        )
        return output


def prepare_k3_communication(
    *,
    group: dist.ProcessGroup,
    rank_in_group: int,
    staged_max_numel: int,
    producer_direct_max_numel: int,
    attnres_max_numel: int,
    attnres_max_rows: int,
    enable_lamport: bool,
    sharded_max_rows: int,
    dtype: torch.dtype,
    device: torch.device,
    all_reduce: "IrisAllReduce | None",
    pull_exchange_max_bytes: int = 0,
) -> K3IrisState:
    """Prepare a group's combined K3 buffers before graph capture.

    Capacities are elements except the explicitly named row and byte
    capacities. Zero omits that path. ``all_reduce`` shares an already
    prepared generic operation, or None creates it. The returned state owns
    all model-specific storage.
    """
    from tokenspeed_kernel.ops.communication._iris import (
        HEAP_HEADROOM_BYTES,
        MIN_HEAP_BYTES,
        IrisGroup,
        get_or_create_context,
    )
    from tokenspeed_kernel.ops.communication._iris.all_reduce import ProducerLamport
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.allreduce_sharded import (
        IrisAllReduceShardedState,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3.attnres import (
        IrisAttnResState,
    )
    from tokenspeed_kernel.ops.communication._iris.pull_exchange import (
        IrisPullExchange,
    )
    from tokenspeed_kernel.ops.communication.iris import IrisAllReduce

    world_size = group.size()
    if (
        min(
            staged_max_numel,
            producer_direct_max_numel,
            attnres_max_numel,
            attnres_max_rows,
            sharded_max_rows,
            pull_exchange_max_bytes,
        )
        < 0
    ):
        raise ValueError("Iris communication capacities must be non-negative")
    if bool(attnres_max_numel) != bool(attnres_max_rows):
        raise ValueError(
            "AttnRes element and row capacities must both be zero or non-zero"
        )
    if all_reduce is not None:
        if (
            all_reduce.group is not group
            or all_reduce.dtype != dtype
            or all_reduce.rank_in_group != rank_in_group
            or all_reduce.device != device
            or all_reduce.staged_max_numel < staged_max_numel
            or all_reduce.producer_direct_max_numel < producer_direct_max_numel
        ):
            raise RuntimeError(
                "Prepare K3 communication before the group's first all-reduce"
            )
        producer_direct_max_numel = all_reduce.producer_direct_max_numel
    if sharded_max_rows:
        IrisAllReduceShardedState.validate_capacity(
            sharded_max_rows, world_size, dtype, producer_direct_max_numel
        )
    total_bytes = (
        IrisAllReduce.heap_bytes(
            staged_max_numel, producer_direct_max_numel, world_size, dtype
        )
        if all_reduce is None
        else 0
    )
    if attnres_max_numel:
        total_bytes += IrisAttnResState.heap_bytes(
            attnres_max_numel, attnres_max_rows, world_size, dtype
        )
    if enable_lamport and ProducerLamport.supported(world_size, dtype):
        lamport_numel = _k3_lamport_numel(producer_direct_max_numel)
        if lamport_numel:
            total_bytes += ProducerLamport.heap_bytes(lamport_numel)
            if world_size == 8:
                total_bytes += _k3_two_shot_heap_bytes(producer_direct_max_numel)
    sharded_bytes = (
        IrisAllReduceShardedState.heap_bytes(sharded_max_rows, world_size, dtype)
        if sharded_max_rows
        else 0
    )
    pull_exchange_bytes = (
        IrisPullExchange.heap_bytes(pull_exchange_max_bytes, world_size)
        if pull_exchange_max_bytes
        else 0
    )
    # Preserve the base heap's headroom for other collective operations.
    ctx = get_or_create_context(
        heap_size=max(MIN_HEAP_BYTES, total_bytes + HEAP_HEADROOM_BYTES)
        + sharded_bytes
        + pull_exchange_bytes,
    )
    if all_reduce is None:
        all_reduce = IrisAllReduce(
            IrisGroup(ctx, group, rank_in_group, device),
            staged_max_numel,
            producer_direct_max_numel,
            dtype,
        )
    state = K3IrisState(
        all_reduce,
        attnres_max_numel=attnres_max_numel,
        attnres_max_rows=attnres_max_rows,
        enable_lamport=enable_lamport,
        sharded_max_rows=sharded_max_rows,
        pull_exchange_max_bytes=pull_exchange_max_bytes,
    )
    all_reduce.peers.finish_prepare()
    return state
