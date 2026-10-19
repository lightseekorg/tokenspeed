# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Address, copy and launch helpers shared by Blackwell QSA kernels."""

import math
from functools import lru_cache, partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
cpy_dice = (None,) + mma_dice  # (CPY, #CPY_MMA, #CPY_M, #CPY_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
_HEAD_DIM = 256
_GROUPED_HEAD_TILE = 8
_CONVERT_WARPGROUPS = 2
_SELECTED_WIDTH = 2051
_PADDING_READ_SLOT = 1
_SMALL_MAX_CLUSTERS = 8
_SMALL_NUM_SPLITS = 8
_WIDE_NUM_SPLITS = 16
_LARGE_NUM_SPLITS = 4
log2_e = math.log2(math.e)
fadd2 = cute.arch.add_packed_f32x2
fmul2 = cute.arch.mul_packed_f32x2
ffma2 = cute.arch.fma_packed_f32x2
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")


@dsl_user_op
def store_smem_prefix_block(destination, first_slot, *, loc=None, ip=None):
    """Store one validated, aligned four-slot block in temporary shared memory."""
    _llvm.inline_asm(
        None,
        [
            destination.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            first_slot.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 b,c,d; add.u32 b,$1,1; add.u32 c,$1,2; "
        "add.u32 d,$1,3; st.shared.v4.u32 [$0], {$1,b,c,d}; }",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@lru_cache(maxsize=None)
def _wide_cluster_capacity(device_index: int) -> int:
    """Number of resident 16-CTA clusters, or zero when unsupported.

    The CuTe occupancy probe reserves one SM's shared memory per CTA, matching
    this kernel's single-CTA residency. Query under the target device context
    once, before compilation/capture, and retain portable eight-CTA launches
    on devices or partitions that cannot accommodate the larger cluster.
    """
    with torch.cuda.device(device_index):
        hardware = utils.HardwareInfo(device_id=device_index)
        try:
            return hardware.get_max_active_clusters(
                cluster_size=_WIDE_NUM_SPLITS,
                stream=cuda.CUstream(
                    torch.cuda.current_stream(device_index).cuda_stream
                ),
            )
        except RuntimeError:
            return 0


def _num_splits(num_clusters: int, wide_cluster_capacity: int) -> int:
    if num_clusters <= _SMALL_MAX_CLUSTERS:
        return (
            _WIDE_NUM_SPLITS
            if num_clusters <= wide_cluster_capacity
            else _SMALL_NUM_SPLITS
        )
    return _LARGE_NUM_SPLITS


@lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    with torch.cuda.device(device_index):
        status, count = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            cuda.CUdevice(device_index),
        )
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cannot query QSA device SM count: {status}")
        return count


def _select_launch_config(
    num_rows: int,
    head_tiles_per_row: int,
    bf16_kv: bool,
    sm_count: int,
    wide_cluster_capacity: int,
) -> tuple[int, int, int, bool]:
    """Return query rows/CTA, KV splits, BF16 slots and async publication.

    These thresholds come from joint tile/stage measurements. They are not
    occupancy guarantees: query the compiled launch's occupancy separately.
    """
    output_tiles = num_rows * head_tiles_per_row
    if bf16_kv and output_tiles * 16 >= sm_count * 3:
        # Sufficient independent output work avoids split-KV. Compare one
        # query per CTA with the parent's serial two/four-query scheduling;
        # the useful MMA tile and operand pipeline remain the same.
        query_rows_per_cta = 1
        splits = 1
        while splits < 4 and output_tiles * splits * 4 < sm_count * 3:
            splits *= 2
        # Avoid another resident CTA wave for a small grid remainder.
        while splits > 1 and output_tiles <= sm_count < output_tiles * splits:
            splits //= 2
        # Compare fine-grained query scheduling against serial query tiles.
        # Each CTA retains the same N16 head tile and KV copy pipeline.
        # Three shared K/V buffers let copies run ahead of the QK/PV consumers.
        return query_rows_per_cta, splits, 3, True
    splits = _num_splits(output_tiles, wide_cluster_capacity)
    return 1, splits, 3 if splits == _WIDE_NUM_SPLITS else 1, False


def _can_group_six_queries(output_tiles: int, sm_count: int) -> bool:
    """Require one full SM wave and at least 60% useful N72 leader fill."""
    shared_ctas = output_tiles // 6
    waves = (shared_ctas + sm_count - 1) // sm_count
    # Retain the small-grid floor; measure native72 on less-filled second waves.
    return shared_ctas >= sm_count and shared_ctas * 5 >= waves * sm_count * 3


def _select_head_tile(heads_per_kv: int, bf16_kv: bool, query_group: int):
    return (
        72
        if bf16_kv and query_group == 6 and (heads_per_kv == 72)
        else (
            96
            if bf16_kv and query_group == 8 and (heads_per_kv == 96)
            else (
                48
                if bf16_kv and query_group in (3, 4) and (heads_per_kv == 48)
                else (
                    32
                    if query_group in (3, 4)
                    or (query_group == 2 and heads_per_kv == 24)
                    else 16 if bf16_kv and heads_per_kv == 12 else _GROUPED_HEAD_TILE
                )
            )
        )
    )


@dsl_user_op
def set_block_rank(smem_ptr, peer_rank, *, loc=None, ip=None):
    dsmem_ptr = cute.arch.map_dsmem_ptr(smem_ptr, peer_rank, loc=loc, ip=ip)
    return cutlass.Int32(dsmem_ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def ld_shared_remote_f32(remote_addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _llvm.inline_asm(
            T.f32(),
            [remote_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def copy_gmem_to_smem_u128(smem_ptr, gmem_ptr, *, loc=None, ip=None):
    """Issue one aligned 16-byte FP8 asynchronous global-to-shared copy."""
    _llvm.inline_asm(
        None,
        [
            smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 smem_addr;\n"
        "cvt.u32.u64 smem_addr, $0;\n"
        "cp.async.cg.shared.global [smem_addr], [$1], 16;\n"
        "}\n",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_gmem_l2(gmem_ptr, *, loc, ip):
    _llvm.inline_asm(
        None,
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def invalidate_mbarrier(barrier, *, loc, ip):
    _nvvm.mbarrier_inval(barrier.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)


@cute.jit
def _prefetch_private_kv_payload(
    k_iter: cute.Pointer,
    v_iter: cute.Pointer,
    selected_slots: cute.Tensor,
    row_idx: cutlass.Int32,
    staged_key: cute.Tensor,
    staged_value: cute.Tensor,
    query_group: cutlass.Constexpr[int],
    num_kv_heads: cutlass.Constexpr[int],
    prefetch_warp: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
    if warp_idx == prefetch_warp:
        for private_index in cutlass.range(query_group * 3, unroll=1):
            query = private_index // 3
            tail = private_index % 3
            raw_slot = selected_slots[row_idx * query_group + query, 2048 + tail]
            slot = cutlass.Int64(raw_slot if raw_slot > 0 else _PADDING_READ_SLOT)
            dim = lane_idx * 8
            offset = slot * num_kv_heads * _HEAD_DIM + dim
            key_destination = cute.domain_offset(
                (dim, tail, query), staged_key
            ).iterator
            value_destination = cute.domain_offset(
                (dim, tail, query), staged_value
            ).iterator
            copy_gmem_to_smem_u128(key_destination, k_iter + offset)
            copy_gmem_to_smem_u128(value_destination, v_iter + offset)
            cute.arch.cp_async_commit_group()
            # Bound pending groups while keeping the newest four in flight.
            if (private_index & 3) == 3:
                cute.arch.cp_async_wait_group(4)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()


def _to_tvm_meta(tensor: torch.Tensor, assumed_align: int) -> cute.Tensor:
    """Create one compile-time tensor descriptor for direct Torch TVM FFI."""

    storage = tensor
    fp8 = tensor.dtype is torch.float8_e4m3fn
    if fp8:
        storage = tensor.view(torch.uint8)
    result = from_dlpack(
        storage.detach(),
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    )
    if fp8:
        result.element_type = cutlass.Float8E4M3FN
    return result


@dsl_user_op
def prepare_cta_all(value, thread_count: cutlass.Constexpr[int], *, loc=None, ip=None):
    # The explicit launch-sized full team reaches each reduction; its uniform
    # first result controls whether that same team reaches the second one.
    result = _nvvm.barrier_cta_red(
        cutlass.Boolean(value).ir_value(loc=loc, ip=ip),
        cutlass.Int32(1).ir_value(loc=loc, ip=ip),
        _nvvm.BarrierReduxKind.AND,
        thread_count=cutlass.Int32(thread_count).ir_value(loc=loc, ip=ip),
        aligned=True,
        results=[T.i32()],
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(result) != 0


def _scalar(value: float | torch.Tensor | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("QSA FP8 descales must be scalar")
        return float(value.item())
    return float(value)


class _ScratchArena:
    """Create aligned views inside a caller-owned shared-memory allocation."""

    def __init__(self, base, capacity):
        self.base = base
        self.capacity = capacity
        self.offset = 0

    @dsl_user_op
    def allocate_array(
        self, element_type, num_elems=1, *, byte_alignment=1, loc=None, ip=None
    ):
        alignment = max(byte_alignment, element_type.width // 8)
        self.offset = ((self.offset + alignment - 1) // alignment) * alignment
        result = cute.recast_ptr(
            self.base + self.offset, dtype=element_type, loc=loc, ip=ip
        )
        self.offset += num_elems * element_type.width // 8
        assert self.offset <= self.capacity
        return result

    @dsl_user_op
    def allocate_tensor(
        self, element_type, layout, byte_alignment, swizzle=None, *, loc=None, ip=None
    ):
        assert cute.is_static(layout)
        pointer = self.allocate_array(
            element_type,
            cute.cosize(layout),
            byte_alignment=byte_alignment,
            loc=loc,
            ip=ip,
        )
        return cute.make_tensor(
            cute.recast_ptr(pointer, swizzle, dtype=element_type, loc=loc, ip=ip),
            layout,
            loc=loc,
            ip=ip,
        )
