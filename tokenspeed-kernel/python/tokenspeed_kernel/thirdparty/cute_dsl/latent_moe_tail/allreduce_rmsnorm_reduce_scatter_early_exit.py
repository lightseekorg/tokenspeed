# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Specialized from FlashInfer's oneshotAllreduceFusionKernel.

"""Routed AllReduce/RMSNorm with CTA-specialized ReduceScatter early exit."""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from .primitives import (
    NUM_LAMPORT_BUFFERS,
    PACKED_BYTES,
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    block_sum_specialized,
    fragment_is_dirty,
    load_global_bf16_as_f32,
    load_global_s32,
    load_global_u32x4,
    load_volatile_u32,
    map_shared_to_peer,
    packed_u32x4_to_bf16x8,
    pdl_enabled,
    red_async_release_gpu_add_u32,
    sanitize_negative_zero,
    store_global_u32x4,
    store_lamport_sentinel_128,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic_m,
)


def _mapping(
    tp_size: int,
    latent_dim: int,
    hidden_dim: int,
) -> tuple[int, int, int, int]:
    shard_dim = hidden_dim // tp_size
    cluster_ctas = latent_dim // shard_dim
    threads = shard_dim // VEC_BF16
    shared_roles = tp_size // cluster_ctas
    return shard_dim, cluster_ctas, threads, shared_roles


def validate_shape(*, tp_size: int, latent_dim: int, hidden_dim: int) -> None:
    """Validate constraints imposed by the fused collective mapping."""

    if tp_size <= 0 or latent_dim <= 0 or hidden_dim <= 0:
        raise ValueError("collective dimensions must be positive")
    if hidden_dim % tp_size:
        raise ValueError("hidden_dim must be divisible by tp_size")
    shard, cluster, threads, _ = _mapping(tp_size, latent_dim, hidden_dim)
    if shard % VEC_BF16:
        raise ValueError("hidden_dim / tp_size must be divisible by 8")
    if latent_dim % shard:
        raise ValueError("latent_dim must be an integer multiple of shard_dim")
    if cluster > 16 or cluster & (cluster - 1):
        raise ValueError("collective cluster width must be a power of two <= 16")
    if tp_size % cluster:
        raise ValueError("tp_size must be divisible by collective cluster width")
    if not 32 <= threads <= 1024:
        raise ValueError("collective threads per CTA must be in [32, 1024]")
    if threads < cluster:
        raise ValueError("collective threads must cover every cluster CTA")


def _select_routed_schedule(
    tp_size: int,
    latent_dim: int,
    hidden_dim: int,
    max_m: int,
) -> tuple[int, int]:
    """Match upstream MNNVL's one-cluster-per-token occupancy policy."""

    _, cluster_ctas, threads, _ = _mapping(tp_size, latent_dim, hidden_dim)
    sm_count = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).multi_processor_count
    while max_m * cluster_ctas > sm_count and cluster_ctas > 1 and threads <= 512:
        cluster_ctas //= 2
        threads *= 2
    return threads, cluster_ctas


class AllReduceRMSNormWithReduceScatterEarlyExit:
    """One routed role plus one ReduceScatter role per destination group."""

    def __init__(
        self,
        *,
        rank: int,
        tp_size: int,
        latent_dim: int,
        hidden_dim: int,
        max_m: int,
        max_token_ctas: int,
        fp32_internal: bool = False,
        include_reduce_scatter: bool = True,
        include_routed: bool = True,
        use_pdl: bool | None = None,
        finalize_top_k: int | None = None,
    ):
        # Bind in host Python; the JIT resolves names from self, not module globals.
        if use_pdl is None:
            self.use_pdl = pdl_enabled()
        else:
            self.use_pdl = use_pdl
        validate_shape(
            tp_size=tp_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        if not 0 <= rank < tp_size:
            raise ValueError(f"rank must be in [0,{tp_size}), got {rank}")
        if not include_routed and not include_reduce_scatter:
            raise ValueError("at least one collective role must be enabled")
        if finalize_top_k is not None and not 1 <= finalize_top_k <= 64:
            raise ValueError(f"finalize_top_k must be in [1, 64], got {finalize_top_k}")
        # Deferred-finalize input mode: the routed publish phase consumes the
        # MoE kernel's (gemm2 permuted rows, expert weights, expanded->permuted
        # index) triple instead of a materialized [M, latent] partial. The
        # finalize math is semantically equivalent to moe_finalize_fuse_shared
        # (FP32 accumulate in ascending fixed k order — rank-deterministic —
        # skip idx == -1 slots for EP non-local / padding, one final round to
        # BF16); bitwise identity is not guaranteed across compilers.
        self.finalize_input = finalize_top_k is not None
        self.finalize_top_k = finalize_top_k if finalize_top_k is not None else 0
        self.rank = rank
        self.tp_size = tp_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        (
            self.shard_dim,
            mapped_cluster,
            mapped_threads,
            shared_roles,
        ) = _mapping(tp_size, latent_dim, hidden_dim)
        if include_reduce_scatter:
            self.cluster_ctas = mapped_cluster
            self.threads = mapped_threads
        else:
            self.threads, self.cluster_ctas = _select_routed_schedule(
                tp_size, latent_dim, hidden_dim, max_m
            )
        # Compile-time diagnostic specialization: a one-role grid runs only the routed path, leaving the production fused path unchanged.
        if include_routed and include_reduce_scatter:
            self.roles = 1 + shared_roles
        elif include_routed:
            self.roles = 1
        else:
            self.roles = shared_roles
        self.warps = (self.threads + 31) // 32
        self.last_warp_lanes = self.threads - (self.warps - 1) * 32
        self.last_warp_mask = (1 << self.last_warp_lanes) - 1
        # Keep one cluster per token in the routed-only diagnostic: reusing a cluster across token waves lets the Lamport generation metadata change between waves.
        self.token_ctas = (
            min(max_m, max_token_ctas) if include_reduce_scatter else max_m
        )
        self.fp32_internal = fp32_internal
        self.include_reduce_scatter = include_reduce_scatter
        self.include_routed = include_routed

    @cute.jit
    def __call__(
        self,
        latent_source: cute.Tensor,
        gamma: cute.Tensor,
        latent_output: cute.Tensor,
        routed_workspace: cute.Tensor,
        latent_flags: cute.Tensor,
        latent_multicast_ptr: Int64,
        shared_source: cute.Tensor,
        shared_output: cute.Tensor,
        shared_workspace: cute.Tensor,
        shared_flags: cute.Tensor,
        shared_peer_ptrs: cute.Tensor,
        finalize_weights: cute.Tensor,
        finalize_idx: cute.Tensor,
        m: Int32,
        epsilon: Float32,
        stream: cuda.CUstream,
    ):
        # In finalize-input mode ``latent_source`` carries the gemm2 permuted
        # rows [total_padded_rows, latent_dim]; ``finalize_weights`` /
        # ``finalize_idx`` are the flat [m * top_k] expert scales and
        # expanded->permuted map. Otherwise the last two are unused dummies.
        self.kernel(
            latent_source,
            gamma,
            latent_output,
            routed_workspace,
            latent_flags,
            latent_multicast_ptr,
            shared_source,
            shared_output,
            shared_workspace,
            shared_flags,
            shared_peer_ptrs,
            finalize_weights,
            finalize_idx,
            m,
            epsilon,
        ).launch(
            grid=(self.token_ctas, self.cluster_ctas, self.roles),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_ctas, 1),
            smem=(self.warps + self.cluster_ctas) * 4,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        latent_source: cute.Tensor,
        gamma: cute.Tensor,
        latent_output: cute.Tensor,
        routed_workspace: cute.Tensor,
        latent_flags: cute.Tensor,
        latent_multicast_ptr: Int64,
        shared_source: cute.Tensor,
        shared_output: cute.Tensor,
        shared_workspace: cute.Tensor,
        shared_flags: cute.Tensor,
        shared_peer_ptrs: cute.Tensor,
        finalize_weights: cute.Tensor,
        finalize_idx: cute.Tensor,
        m: Int32,
        epsilon: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_cta, cta_y, role = cute.arch.block_idx()
        logical_role = role
        if cutlass.const_expr(not self.include_routed):
            # A shared-only grid starts at z=0; _token_device reserves logical role 0 for the routed collective.
            logical_role = role + Int32(1)
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())

        cute.arch.griddepcontrol_wait()
        token = token_cta
        while token < m:
            self._token_device(
                latent_source,
                gamma,
                latent_output,
                routed_workspace,
                latent_flags,
                latent_multicast_ptr,
                shared_source,
                shared_output,
                shared_workspace,
                shared_flags,
                shared_peer_ptrs,
                finalize_weights,
                finalize_idx,
                m,
                epsilon,
                token,
                token_cta,
                cta_y,
                logical_role,
                cluster_rank,
                tidx,
            )
            token = token + self.token_ctas
        # Trigger dependents only after every CTA stores its final token wave.
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _token_device(
        self,
        latent_source: cute.Tensor,
        gamma: cute.Tensor,
        latent_output: cute.Tensor,
        routed_workspace: cute.Tensor,
        latent_flags: cute.Tensor,
        latent_multicast_ptr: Int64,
        shared_source: cute.Tensor,
        shared_output: cute.Tensor,
        shared_workspace: cute.Tensor,
        shared_flags: cute.Tensor,
        shared_peer_ptrs: cute.Tensor,
        finalize_weights: cute.Tensor,
        finalize_idx: cute.Tensor,
        m: Int32,
        epsilon: Float32,
        token: Int32,
        token_cta: Int32,
        cta_y: Int32,
        role: Int32,
        cluster_rank: Int32,
        tidx: Int32,
    ):
        if role == 0:
            packed_idx = cluster_rank * self.threads + tidx
            element_offset = (
                Int64(token) * self.latent_dim + Int64(packed_idx) * VEC_BF16
            )
            current_index = cute.arch.load((latent_flags.iterator + 0).llvm_ptr, Uint32)
            dirty_index = cute.arch.load((latent_flags.iterator + 1).llvm_ptr, Uint32)
            bytes_per_buffer = cute.arch.load(
                (latent_flags.iterator + 2).llvm_ptr, Uint32
            )
            dirty_num_stages = cute.arch.load(
                (latent_flags.iterator + 3).llvm_ptr, Uint32
            )
            bytes_to_clear = cute.arch.load(
                (latent_flags.iterator + 4).llvm_ptr, Uint32
            )
            current_elements = Int64(current_index) * (
                Int64(bytes_per_buffer) // Int64(2)
            )
            dirty_elements = Int64(dirty_index) * (Int64(bytes_per_buffer) // Int64(2))

            if cutlass.const_expr(self.finalize_input):
                # Inline MoE finalize: gather this fragment from every routed
                # expert's gemm2 row and combine with the expert scales.
                # Semantically equivalent to the standalone finalize kernel
                # (FP32 ascending-k accumulate, single BF16 round); bitwise
                # identity is not guaranteed across compilers. idx == -1
                # slots are EP non-local / padded entries: they contribute
                # nothing.
                finalize_accum = cute.make_rmem_tensor(
                    cute.make_layout((VEC_BF16,)), Float32
                )
                for element in cutlass.range_constexpr(VEC_BF16):
                    finalize_accum[element] = Float32(0.0)
                for k in cutlass.range_constexpr(self.finalize_top_k):
                    slot = Int64(token) * self.finalize_top_k + k
                    permuted_row = load_global_s32(finalize_idx.iterator + slot)
                    if permuted_row >= Int32(0):
                        scale = load_global_bf16_as_f32(
                            finalize_weights.iterator + slot
                        )
                        row_ptr = cute.make_ptr(
                            BFloat16,
                            (
                                latent_source.iterator
                                + Int64(permuted_row) * self.latent_dim
                                + Int64(packed_idx) * VEC_BF16
                            ).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        expert_values = packed_u32x4_to_bf16x8(
                            load_global_u32x4(row_ptr, volatile=False)
                        ).to(Float32)
                        for element in cutlass.range_constexpr(VEC_BF16):
                            finalize_accum[element] = (
                                finalize_accum[element] + scale * expert_values[element]
                            )
                local_packed = sanitize_negative_zero(
                    bf16x8_to_packed_u32x4(finalize_accum.load().to(BFloat16))
                )
            else:
                local_ptr = cute.make_ptr(
                    BFloat16,
                    (latent_source.iterator + element_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                local_packed = sanitize_negative_zero(
                    load_global_u32x4(local_ptr, volatile=False)
                )
            multicast_offset = (
                Int64(current_index) * Int64(bytes_per_buffer)
                + (
                    (Int64(token) * self.tp_size + self.rank) * self.latent_dim
                    + Int64(packed_idx) * VEC_BF16
                )
                * 2
            )
            store_global_u32x4(
                latent_multicast_ptr + multicast_offset,
                local_packed,
                volatile=False,
            )

            cute.arch.cluster_arrive()
            # All CTAs must complete this handshake before the st.shared::cluster exchange: a partial wait skews barrier phases and the pre-DSM wait can match a stale phase, losing the peer-entered guarantee.
            cute.arch.cluster_wait()
            if cluster_rank == 0 and tidx == 0:
                red_async_release_gpu_add_u32(latent_flags.iterator + 8, Uint32(1))

            global_tid = (
                Int64(token) * self.cluster_ctas + Int64(cta_y)
            ) * self.threads + Int64(tidx)
            total_threads = Int64(m) * self.cluster_ctas * self.threads
            clear_fragments = (Int64(bytes_to_clear) + PACKED_BYTES - 1) // PACKED_BYTES
            clear_idx = global_tid
            if dirty_num_stages > Uint32(0):
                while clear_idx < clear_fragments:
                    clear_ptr = cute.make_ptr(
                        BFloat16,
                        (
                            routed_workspace.iterator
                            + dirty_elements
                            + clear_idx * VEC_BF16
                        ).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    store_lamport_sentinel_128(clear_ptr)
                    clear_idx = clear_idx + total_threads

            rank_words = cute.make_rmem_tensor(
                cute.make_layout((self.tp_size, 4), stride=(4, 1)), Uint32
            )
            for word in cutlass.range_constexpr(4):
                rank_words[self.rank, word] = local_packed[word]
            valid = False
            while not valid:
                valid = True
                for source_rank in cutlass.range_constexpr(self.tp_size):
                    if cutlass.const_expr(source_rank != self.rank):
                        remote_element = current_elements + (
                            (Int64(token) * self.tp_size + source_rank)
                            * self.latent_dim
                            + Int64(packed_idx) * VEC_BF16
                        )
                        remote_ptr = cute.make_ptr(
                            BFloat16,
                            (routed_workspace.iterator + remote_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        remote = load_global_u32x4(remote_ptr, volatile=True)
                        for word in cutlass.range_constexpr(4):
                            rank_words[source_rank, word] = remote[word]
                        valid = valid & (not fragment_is_dirty(remote))

            accum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            for element in cutlass.range_constexpr(VEC_BF16):
                accum[element] = Float32(0.0)
            for source_rank in cutlass.range_constexpr(self.tp_size):
                values = packed_u32x4_to_bf16x8(
                    rank_words[source_rank, None].load()
                ).to(Float32)
                for element in cutlass.range_constexpr(VEC_BF16):
                    accum[element] = accum[element] + values[element]

            if cutlass.const_expr(self.fp32_internal):
                # High-precision fused mode: keep the rank reduction in FP32 through the RMS square.
                norm_input = accum.load()
                norm_square = norm_input * norm_input
            else:
                norm_input_bf16 = accum.load().to(BFloat16)
                norm_input = norm_input_bf16.to(Float32)
                # Upstream-compatible mode: BF16 * BF16 first, then promote the rounded square to FP32.
                norm_square = (norm_input_bf16 * norm_input_bf16).to(Float32)
            thread_sum = norm_square.reduce(
                cute.ReductionOp.ADD,
                init_val=Float32(0.0),
                reduction_profile=0,
            )
            smem = cutlass.utils.SmemAllocator()
            warp_sums = smem.allocate_tensor(
                Float32, cute.make_layout((self.warps,)), byte_alignment=4
            )
            cluster_sums = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.cluster_ctas,)),
                byte_alignment=4,
            )
            block_sum = block_sum_specialized(
                thread_sum,
                warp_sums,
                tidx,
                self.warps,
                self.last_warp_lanes,
                self.last_warp_mask,
            )
            if tidx < self.cluster_ctas:
                local_slot = cluster_sums.iterator + cluster_rank
                remote_slot = map_shared_to_peer(local_slot, Int32(tidx))
                store_shared_cluster_f32(remote_slot, block_sum)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            full_sum = Float32(0.0)
            for peer in cutlass.range_constexpr(self.cluster_ctas):
                full_sum = full_sum + cluster_sums[peer]
            inv_rms = cute.math.rsqrt(
                full_sum / Float32(self.latent_dim) + epsilon, fastmath=True
            )
            gamma_ptr = cute.make_ptr(
                BFloat16,
                (gamma.iterator + Int64(packed_idx) * VEC_BF16).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            gamma_values = packed_u32x4_to_bf16x8(
                load_global_u32x4(gamma_ptr, volatile=False)
            )
            result = (norm_input * inv_rms * gamma_values.to(Float32)).to(BFloat16)
            store_global_u32x4(
                Int64((latent_output.iterator + element_offset).toint()),
                bf16x8_to_packed_u32x4(result),
                volatile=False,
            )

            # The x=0 CTA rotates only on its final token wave: waiting for all M arrivals guarantees every token-wave CTA loaded the current generation before the metadata advances.
            if (
                token_cta == 0
                and token + self.token_ctas >= m
                and cta_y == 0
                and tidx == 0
            ):
                access_counter = latent_flags.iterator + 8
                arrived = load_volatile_u32(access_counter)
                while arrived < Uint32(m):
                    arrived = load_volatile_u32(access_counter)
                next_index = (current_index + Uint32(1)) % Uint32(NUM_LAMPORT_BUFFERS)
                actual_bytes = Uint32(m) * Uint32(self.tp_size * self.latent_dim * 2)
                cute.arch.store((latent_flags.iterator + 0).llvm_ptr, next_index)
                cute.arch.store((latent_flags.iterator + 1).llvm_ptr, current_index)
                cute.arch.store(
                    (latent_flags.iterator + 2).llvm_ptr,
                    bytes_per_buffer,
                )
                cute.arch.store((latent_flags.iterator + 3).llvm_ptr, Uint32(1))
                cute.arch.store((latent_flags.iterator + 4).llvm_ptr, actual_bytes)
                for index in cutlass.range_constexpr(5, 8):
                    cute.arch.store(
                        (latent_flags.iterator + index).llvm_ptr,
                        Uint32(0),
                    )
                cute.arch.store(access_counter.llvm_ptr, Uint32(0))

        else:
            shared_group = role - 1
            destination = shared_group * self.cluster_ctas + cluster_rank
            current_index = cute.arch.load((shared_flags.iterator + 0).llvm_ptr, Uint32)
            dirty_index = cute.arch.load((shared_flags.iterator + 1).llvm_ptr, Uint32)
            bytes_per_buffer = cute.arch.load(
                (shared_flags.iterator + 2).llvm_ptr, Uint32
            )
            dirty_num_stages = cute.arch.load(
                (shared_flags.iterator + 3).llvm_ptr, Uint32
            )
            bytes_to_clear = cute.arch.load(
                (shared_flags.iterator + 4).llvm_ptr, Uint32
            )
            current_elements = Int64(current_index) * (
                Int64(bytes_per_buffer) // Int64(2)
            )
            dirty_elements = Int64(dirty_index) * (Int64(bytes_per_buffer) // Int64(2))

            source_element = (
                Int64(token) * self.hidden_dim
                + Int64(destination) * self.shard_dim
                + Int64(tidx) * VEC_BF16
            )
            source_ptr = cute.make_ptr(
                BFloat16,
                (shared_source.iterator + source_element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            local_packed = sanitize_negative_zero(
                load_global_u32x4(source_ptr, volatile=False)
            )
            peer_base = cute.arch.load(
                (shared_peer_ptrs.iterator + destination).llvm_ptr,
                Int64,
            )
            destination_element = current_elements + (
                (Int64(token) * self.tp_size + self.rank) * self.shard_dim
                + Int64(tidx) * VEC_BF16
            )
            store_global_u32x4(
                peer_base + destination_element * 2,
                local_packed,
                volatile=False,
            )

            # The wait must be unconditional: every per-token-loop barrier use must be arrive/wait-symmetric on every CTA, or the phase counters skew across iterations and the next pre-DSM wait matches a stale phase.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if cluster_rank == 0 and tidx == 0:
                red_async_release_gpu_add_u32(shared_flags.iterator + 8, Uint32(1))

            global_tid = (
                Int64(token) * self.tp_size + Int64(destination)
            ) * self.threads + Int64(tidx)
            total_threads = Int64(m) * self.tp_size * self.threads
            clear_fragments = (Int64(bytes_to_clear) + PACKED_BYTES - 1) // PACKED_BYTES
            clear_idx = global_tid
            if dirty_num_stages > Uint32(0):
                while clear_idx < clear_fragments:
                    clear_ptr = cute.make_ptr(
                        BFloat16,
                        (
                            shared_workspace.iterator
                            + dirty_elements
                            + clear_idx * VEC_BF16
                        ).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    store_lamport_sentinel_128(clear_ptr)
                    clear_idx = clear_idx + total_threads

            if destination == self.rank:
                rank_words = cute.make_rmem_tensor(
                    cute.make_layout((self.tp_size, 4), stride=(4, 1)), Uint32
                )
                valid = False
                while not valid:
                    valid = True
                    for source_rank in cutlass.range_constexpr(self.tp_size):
                        remote_element = current_elements + (
                            (Int64(token) * self.tp_size + source_rank) * self.shard_dim
                            + Int64(tidx) * VEC_BF16
                        )
                        remote_ptr = cute.make_ptr(
                            BFloat16,
                            (shared_workspace.iterator + remote_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        remote = load_global_u32x4(remote_ptr, volatile=True)
                        for word in cutlass.range_constexpr(4):
                            rank_words[source_rank, word] = remote[word]
                        valid = valid & (not fragment_is_dirty(remote))

                accum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                for element in cutlass.range_constexpr(VEC_BF16):
                    accum[element] = Float32(0.0)
                for source_rank in cutlass.range_constexpr(self.tp_size):
                    values = packed_u32x4_to_bf16x8(
                        rank_words[source_rank, None].load()
                    ).to(Float32)
                    for element in cutlass.range_constexpr(VEC_BF16):
                        accum[element] = accum[element] + values[element]
                result = accum.load().to(BFloat16)
                output_element = (
                    Int64(token) * self.hidden_dim
                    + self.rank * self.shard_dim
                    + Int64(tidx) * VEC_BF16
                )
                store_global_u32x4(
                    Int64((shared_output.iterator + output_element).toint()),
                    bf16x8_to_packed_u32x4(result),
                    volatile=False,
                )

            if destination == self.rank:
                cute.arch.barrier()

            if (
                token_cta == 0
                and token + self.token_ctas >= m
                and shared_group == 0
                and cluster_rank == 0
                and tidx == 0
            ):
                access_counter = shared_flags.iterator + 8
                arrived = load_volatile_u32(access_counter)
                target = Uint32(m) * Uint32(self.tp_size // self.cluster_ctas)
                while arrived < target:
                    arrived = load_volatile_u32(access_counter)
                next_index = (current_index + Uint32(1)) % Uint32(NUM_LAMPORT_BUFFERS)
                actual_bytes = Uint32(m) * Uint32(self.tp_size * self.shard_dim * 2)
                cute.arch.store((shared_flags.iterator + 0).llvm_ptr, next_index)
                cute.arch.store((shared_flags.iterator + 1).llvm_ptr, current_index)
                cute.arch.store(
                    (shared_flags.iterator + 2).llvm_ptr,
                    bytes_per_buffer,
                )
                cute.arch.store((shared_flags.iterator + 3).llvm_ptr, Uint32(1))
                cute.arch.store((shared_flags.iterator + 4).llvm_ptr, actual_bytes)
                for index in cutlass.range_constexpr(5, 8):
                    cute.arch.store(
                        (shared_flags.iterator + index).llvm_ptr,
                        Uint32(0),
                    )
                cute.arch.store(access_counter.llvm_ptr, Uint32(0))


_COMPILED: dict[tuple[object, ...], Any] = {}


def _routed_workspace_cute(workspace: torch.Tensor):
    return to_cute(workspace.view(torch.bfloat16), 16)


def _compile_key(
    rank: int,
    tp_size: int,
    latent_dim: int,
    hidden_dim: int,
    max_m: int,
    max_token_ctas: int,
    fp32_internal: bool,
    include_reduce_scatter: bool,
    include_routed: bool,
    finalize_top_k: int | None = None,
):
    return (
        torch.accelerator.current_device_index(),
        rank,
        tp_size,
        latent_dim,
        hidden_dim,
        max_m,
        max_token_ctas,
        fp32_internal,
        include_reduce_scatter,
        include_routed,
        finalize_top_k,
    )


def _runtime_args(
    latent_source: torch.Tensor,
    gamma: torch.Tensor,
    latent_output: torch.Tensor,
    routed_workspace: torch.Tensor,
    routed_flags: torch.Tensor,
    routed_multicast_ptr: int,
    shared_source: torch.Tensor,
    shared_output: torch.Tensor,
    shared_workspace: torch.Tensor,
    shared_flags: torch.Tensor,
    shared_peer_ptrs: torch.Tensor,
    finalize_weights: torch.Tensor,
    finalize_idx: torch.Tensor,
    num_tokens: int,
    rms_eps: float,
):
    # ``latent_source`` is [M, latent] in the standard mode and the gemm2
    # permuted rows [P, latent] in finalize-input mode; both are dynamic in
    # mode 0 so one compiled kernel serves every token count. The finalize
    # scale/index tensors are flat [M * top_k] (dummies otherwise) and only
    # ever scalar-loaded, so 4-byte alignment suffices.
    return (
        to_cute_dynamic_m(latent_source, mode=0, assumed_align=16),
        to_cute(gamma, 16),
        to_cute(latent_output, 16),
        _routed_workspace_cute(routed_workspace),
        to_cute(routed_flags, 16),
        Int64(routed_multicast_ptr),
        to_cute_dynamic_m(shared_source, mode=0, assumed_align=16),
        to_cute(shared_output, 16),
        to_cute(shared_workspace, 16),
        to_cute(shared_flags, 16),
        to_cute(shared_peer_ptrs, 16),
        to_cute_dynamic_m(finalize_weights, mode=0, assumed_align=4),
        to_cute_dynamic_m(finalize_idx, mode=0, assumed_align=4),
        Int32(num_tokens),
        Float32(rms_eps),
        cuda.CUstream(torch.cuda.current_stream(latent_source.device).cuda_stream),
    )


def compile_kernel(
    *,
    rank: int,
    tp_size: int,
    latent_dim: int,
    hidden_dim: int,
    max_m: int,
    max_token_ctas: int,
    latent_output: torch.Tensor,
    routed_workspace: torch.Tensor,
    routed_flags: torch.Tensor,
    routed_multicast_ptr: int,
    shared_output: torch.Tensor,
    shared_workspace: torch.Tensor,
    shared_flags: torch.Tensor,
    shared_peer_ptrs: torch.Tensor,
    rms_eps: float,
    fp32_internal: bool,
    include_reduce_scatter: bool = True,
    include_routed: bool = True,
    finalize_top_k: int | None = None,
) -> None:
    """Compile the rank/M specialization without retaining caller tensors."""

    key = _compile_key(
        rank,
        tp_size,
        latent_dim,
        hidden_dim,
        max_m,
        max_token_ctas,
        fp32_internal,
        include_reduce_scatter,
        include_routed,
        finalize_top_k,
    )
    if key in _COMPILED:
        return
    device = latent_output.device
    latent_rows = max_m if finalize_top_k is None else max_m * finalize_top_k
    finalize_slots = 8 if finalize_top_k is None else max_m * finalize_top_k
    latent = torch.empty((latent_rows, latent_dim), dtype=torch.bfloat16, device=device)
    gamma = torch.empty((latent_dim,), dtype=torch.bfloat16, device=device)
    shared = torch.empty((max_m, hidden_dim), dtype=torch.bfloat16, device=device)
    finalize_weights = torch.zeros(
        (finalize_slots,), dtype=torch.bfloat16, device=device
    )
    finalize_idx = torch.full((finalize_slots,), -1, dtype=torch.int32, device=device)
    kernel = AllReduceRMSNormWithReduceScatterEarlyExit(
        rank=rank,
        tp_size=tp_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        max_m=max_m,
        max_token_ctas=max_token_ctas,
        fp32_internal=fp32_internal,
        include_reduce_scatter=include_reduce_scatter,
        include_routed=include_routed,
        finalize_top_k=finalize_top_k,
    )
    _COMPILED[key] = cute.compile(
        kernel,
        *_runtime_args(
            latent,
            gamma,
            latent_output,
            routed_workspace,
            routed_flags,
            routed_multicast_ptr,
            shared,
            shared_output,
            shared_workspace,
            shared_flags,
            shared_peer_ptrs,
            finalize_weights,
            finalize_idx,
            max_m,
            rms_eps,
        ),
    )


def launch(
    latent_source: torch.Tensor,
    gamma: torch.Tensor,
    latent_output: torch.Tensor,
    routed_workspace: torch.Tensor,
    routed_flags: torch.Tensor,
    routed_multicast_ptr: int,
    shared_source: torch.Tensor,
    shared_output: torch.Tensor,
    shared_workspace: torch.Tensor,
    shared_flags: torch.Tensor,
    shared_peer_ptrs: torch.Tensor,
    finalize_weights: torch.Tensor,
    finalize_idx: torch.Tensor,
    num_tokens: int,
    rms_eps: float,
    *,
    rank: int,
    tp_size: int,
    latent_dim: int,
    hidden_dim: int,
    max_m: int,
    max_token_ctas: int,
    fp32_internal: bool,
    include_reduce_scatter: bool = True,
    include_routed: bool = True,
    finalize_top_k: int | None = None,
) -> None:
    compile_kernel(
        rank=rank,
        tp_size=tp_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        max_m=max_m,
        max_token_ctas=max_token_ctas,
        latent_output=latent_output,
        routed_workspace=routed_workspace,
        routed_flags=routed_flags,
        routed_multicast_ptr=routed_multicast_ptr,
        shared_output=shared_output,
        shared_workspace=shared_workspace,
        shared_flags=shared_flags,
        shared_peer_ptrs=shared_peer_ptrs,
        rms_eps=rms_eps,
        fp32_internal=fp32_internal,
        include_reduce_scatter=include_reduce_scatter,
        include_routed=include_routed,
        finalize_top_k=finalize_top_k,
    )
    _COMPILED[
        _compile_key(
            rank,
            tp_size,
            latent_dim,
            hidden_dim,
            max_m,
            max_token_ctas,
            fp32_internal,
            include_reduce_scatter,
            include_routed,
            finalize_top_k,
        )
    ](
        *_runtime_args(
            latent_source,
            gamma,
            latent_output,
            routed_workspace,
            routed_flags,
            routed_multicast_ptr,
            shared_source,
            shared_output,
            shared_workspace,
            shared_flags,
            shared_peer_ptrs,
            finalize_weights,
            finalize_idx,
            num_tokens,
            rms_eps,
        )
    )


class CollectiveKernel:
    """Own and launch the routed AllReduce/RMSNorm plus shared ReduceScatter."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        rank: int,
        tp_size: int,
        latent_dim: int,
        hidden_dim: int,
        max_m: int,
        max_token_ctas: int,
        rms_eps: float,
        fp32_internal: bool,
        scratch_allocator=None,
        finalize_top_k: int | None = None,
        precompile_split: bool = False,
    ) -> None:
        validate_shape(
            tp_size=tp_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        if finalize_top_k is not None and not 1 <= finalize_top_k <= 64:
            raise ValueError(f"finalize_top_k must be in [1, 64], got {finalize_top_k}")
        self.rank = rank
        self.tp_size = tp_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.shard_dim = hidden_dim // tp_size
        self.max_m = max_m
        self.max_token_ctas = max_token_ctas
        self.rms_eps = float(rms_eps)
        self.fp32_internal = fp32_internal
        self.precompile_split = precompile_split
        # When set, ``call_deferred`` additionally accepts the MoE kernel's
        # deferred-finalize triple; the standard materialized-input mode
        # stays available on the same instance.
        self.finalize_top_k = finalize_top_k
        device = torch.device("cuda", torch.accelerator.current_device_index())
        # Placeholders for the finalize operand slots of the standard-mode
        # kernel signature (never dereferenced when finalize is compiled out).
        self._finalize_dummy_weights = torch.zeros(
            (8,), dtype=torch.bfloat16, device=device
        )
        self._finalize_dummy_idx = torch.full(
            (8,), -1, dtype=torch.int32, device=device
        )

        bytes_per_routed_buffer = max_m * tp_size * latent_dim * 2
        routed_bytes = NUM_LAMPORT_BUFFERS * bytes_per_routed_buffer
        self._routed_workspace = symm_mem.empty(
            routed_bytes // 4,
            dtype=torch.float32,
            device=device,
        )
        self._routed_symm_mem = symm_mem.rendezvous(self._routed_workspace, group)
        self._routed_workspace.fill_(-0.0)
        actual_bytes_per_buffer = (
            self._routed_symm_mem.buffer_size // NUM_LAMPORT_BUFFERS // 16 * 16
        )
        if actual_bytes_per_buffer < bytes_per_routed_buffer:
            raise RuntimeError("routed symmetric workspace is too small")
        self._routed_flags = torch.tensor(
            [0, 2, actual_bytes_per_buffer, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint32,
            device=device,
        )
        routed_multicast_ptr = self._routed_symm_mem.multicast_ptr
        if routed_multicast_ptr is None or routed_multicast_ptr == 0:
            raise RuntimeError("routed NVLS multicast mapping is unavailable")
        self._routed_multicast_ptr = int(routed_multicast_ptr)

        # Shared staging views must be reacquired per call and warmed before graph capture.
        self._scratch_allocator = scratch_allocator
        self._scratch_specs = (
            ((max_m, latent_dim), torch.bfloat16),
            ((max_m, hidden_dim), torch.bfloat16),
        )
        if scratch_allocator is None:
            self._latent_output = torch.empty(
                (max_m, latent_dim), dtype=torch.bfloat16, device=device
            )
            self._shared_output = torch.empty(
                (max_m, hidden_dim), dtype=torch.bfloat16, device=device
            )
        else:
            # Grow the pool to peak before capture freezes its address.
            self._latent_output, self._shared_output = scratch_allocator(
                *self._scratch_specs
            )
        self._shared_workspace = symm_mem.empty(
            (NUM_LAMPORT_BUFFERS, max_m, tp_size, self.shard_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self._shared_symm_mem = symm_mem.rendezvous(self._shared_workspace, group)
        self._shared_workspace.view(torch.int32).fill_(-0x80000000)
        self._shared_flags = torch.zeros(12, dtype=torch.int32, device=device)
        self._shared_flags[1] = 1
        self._shared_flags[2] = max_m * tp_size * self.shard_dim * 2
        peer_ptrs = [
            self._shared_symm_mem.get_buffer(
                peer,
                self._shared_workspace.shape,
                torch.bfloat16,
            ).data_ptr()
            for peer in range(tp_size)
        ]
        if any(pointer == 0 for pointer in peer_ptrs):
            raise RuntimeError("shared LSA peer mapping is unavailable")
        self._shared_peer_ptrs = torch.tensor(
            peer_ptrs, dtype=torch.int64, device=device
        )

        torch.accelerator.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])
        for owner in range(tp_size):
            if rank == owner:
                variants = [(True, True, None)]
                if finalize_top_k is not None:
                    variants.append((True, True, finalize_top_k))
                if precompile_split:
                    variants.extend(
                        [
                            (True, False, None),
                            (False, True, None),
                        ]
                    )
                    if finalize_top_k is not None:
                        variants.append((False, True, finalize_top_k))
                for include_reduce_scatter, include_routed, variant_top_k in variants:
                    compile_kernel(
                        rank=rank,
                        tp_size=tp_size,
                        latent_dim=latent_dim,
                        hidden_dim=hidden_dim,
                        max_m=max_m,
                        max_token_ctas=max_token_ctas,
                        latent_output=self._latent_output,
                        routed_workspace=self._routed_workspace,
                        routed_flags=self._routed_flags,
                        routed_multicast_ptr=self._routed_multicast_ptr,
                        shared_output=self._shared_output,
                        shared_workspace=self._shared_workspace,
                        shared_flags=self._shared_flags,
                        shared_peer_ptrs=self._shared_peer_ptrs,
                        rms_eps=self.rms_eps,
                        fp32_internal=fp32_internal,
                        include_reduce_scatter=include_reduce_scatter,
                        include_routed=include_routed,
                        finalize_top_k=variant_top_k,
                    )
            dist.barrier(group=group, device_ids=[device.index])

    def __call__(
        self,
        latent_source: torch.Tensor,
        shared_source: torch.Tensor,
        gamma: torch.Tensor,
        *,
        include_reduce_scatter: bool = True,
        include_routed: bool = True,
        shared_output_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not include_reduce_scatter and not include_routed:
            raise ValueError("at least one collective role must be enabled")
        if include_reduce_scatter != include_routed and not self.precompile_split:
            raise RuntimeError("split collective roles require precompile_split=True")
        if latent_source.ndim != 2 or shared_source.ndim != 2:
            raise ValueError("latent_source and shared_source must be rank-2")
        m = latent_source.shape[0]
        device = self._routed_workspace.device
        expected = (
            (latent_source, (m, self.latent_dim), "latent_source"),
            (shared_source, (m, self.hidden_dim), "shared_source"),
            (gamma, (self.latent_dim,), "gamma"),
        )
        for tensor, shape, name in expected:
            if (
                tensor.shape != shape
                or tensor.dtype != torch.bfloat16
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous CUDA BF16 {list(shape)}")
        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")
        return self._dispatch(
            latent_source,
            shared_source,
            gamma,
            m,
            finalize_weights=self._finalize_dummy_weights,
            finalize_idx=self._finalize_dummy_idx,
            finalize_top_k=None,
            include_reduce_scatter=include_reduce_scatter,
            include_routed=include_routed,
            shared_output_override=shared_output_override,
        )

    def call_deferred(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_source: torch.Tensor,
        gamma: torch.Tensor,
        num_tokens: int,
        *,
        include_reduce_scatter: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused collective over the MoE kernel's deferred-finalize triple.

        Args:
            gemm2_output: Permuted expert rows ``[total_padded_rows,
                latent_dim]`` (contiguous BF16); rows not referenced by the
                index map are never read.
            expert_weights: Flat per-slot scales ``[num_tokens * top_k]``
                (contiguous BF16).
            expanded_idx_to_permuted_idx: Flat expanded->permuted map
                ``[num_tokens * top_k]`` (contiguous int32); ``-1`` drops the
                slot (EP non-local expert or padding).
            shared_source: This rank's shared partial ``[num_tokens, hidden]``.
            gamma: Latent RMSNorm weight ``[latent_dim]``.
            num_tokens: Token count M for this launch.

        Returns:
            ``(latent_output[:M], shared_shard)`` exactly like ``__call__``.
        """
        if self.finalize_top_k is None:
            raise RuntimeError(
                "CollectiveKernel was constructed without finalize_top_k; "
                "the deferred-finalize variant is not compiled"
            )
        if not include_reduce_scatter and not self.precompile_split:
            raise RuntimeError("split collective roles require precompile_split=True")
        m = int(num_tokens)
        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")
        device = self._routed_workspace.device
        slots = m * self.finalize_top_k
        if (
            gemm2_output.ndim != 2
            or gemm2_output.shape[1] != self.latent_dim
            or gemm2_output.dtype != torch.bfloat16
            or gemm2_output.device != device
            or not gemm2_output.is_contiguous()
        ):
            raise ValueError(
                f"gemm2_output must be contiguous CUDA BF16 [*, {self.latent_dim}]"
            )
        for tensor, dtype, name in (
            (expert_weights, torch.bfloat16, "expert_weights"),
            (expanded_idx_to_permuted_idx, torch.int32, "expanded_idx_to_permuted_idx"),
        ):
            if (
                tensor.shape != (slots,)
                or tensor.dtype != dtype
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous CUDA {dtype} [{slots}]")
        if (
            shared_source.shape != (m, self.hidden_dim)
            or shared_source.dtype != torch.bfloat16
            or shared_source.device != device
            or not shared_source.is_contiguous()
        ):
            raise ValueError(
                f"shared_source must be contiguous CUDA BF16 [{m}, {self.hidden_dim}]"
            )
        if (
            gamma.shape != (self.latent_dim,)
            or gamma.dtype != torch.bfloat16
            or gamma.device != device
            or not gamma.is_contiguous()
        ):
            raise ValueError(f"gamma must be contiguous CUDA BF16 [{self.latent_dim}]")
        return self._dispatch(
            gemm2_output,
            shared_source,
            gamma,
            m,
            finalize_weights=expert_weights,
            finalize_idx=expanded_idx_to_permuted_idx,
            finalize_top_k=self.finalize_top_k,
            include_reduce_scatter=include_reduce_scatter,
            include_routed=True,
            shared_output_override=None,
        )

    def _dispatch(
        self,
        latent_source: torch.Tensor,
        shared_source: torch.Tensor,
        gamma: torch.Tensor,
        m: int,
        *,
        finalize_weights: torch.Tensor,
        finalize_idx: torch.Tensor,
        finalize_top_k: int | None,
        include_reduce_scatter: bool,
        include_routed: bool,
        shared_output_override: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._routed_workspace.device
        if self._scratch_allocator is not None:
            # Pool views cannot survive another allocation, which may move the block.
            self._latent_output, self._shared_output = self._scratch_allocator(
                *self._scratch_specs
            )

        shared_output = (
            self._shared_output
            if shared_output_override is None
            else shared_output_override
        )
        if (
            shared_output.shape != (self.max_m, self.hidden_dim)
            or shared_output.dtype != torch.bfloat16
            or shared_output.device != device
            or not shared_output.is_contiguous()
        ):
            raise ValueError(
                "shared_output_override must be contiguous CUDA BF16 "
                f"[{self.max_m}, {self.hidden_dim}]"
            )
        shard_start = self.rank * self.shard_dim
        shared_shard = shared_output[:, shard_start : shard_start + self.shard_dim]

        with torch.accelerator.device_index(device.index):
            launch(
                latent_source,
                gamma,
                self._latent_output,
                self._routed_workspace,
                self._routed_flags,
                self._routed_multicast_ptr,
                shared_source,
                shared_output,
                self._shared_workspace,
                self._shared_flags,
                self._shared_peer_ptrs,
                finalize_weights,
                finalize_idx,
                m,
                self.rms_eps,
                rank=self.rank,
                tp_size=self.tp_size,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                max_m=self.max_m,
                max_token_ctas=self.max_token_ctas,
                fp32_internal=self.fp32_internal,
                include_reduce_scatter=include_reduce_scatter,
                include_routed=include_routed,
                finalize_top_k=finalize_top_k,
            )
        return (
            self._latent_output[:m],
            shared_shard,
        )

    @property
    def latent_output(self) -> torch.Tensor:
        if self._scratch_allocator is not None:
            raise RuntimeError(
                "staging comes from the shared workspace pool; holding a view "
                "outside a call violates the pool contract -- use the tensors "
                "returned by __call__"
            )
        return self._latent_output

    @property
    def shared_output(self) -> torch.Tensor:
        if self._scratch_allocator is not None:
            raise RuntimeError(
                "staging comes from the shared workspace pool; holding a view "
                "outside a call violates the pool contract -- use the tensors "
                "returned by __call__"
            )
        return self._shared_output
