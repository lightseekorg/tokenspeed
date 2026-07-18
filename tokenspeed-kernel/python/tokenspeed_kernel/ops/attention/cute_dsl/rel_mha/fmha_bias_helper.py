"""Compatibility layer between the rel_mha kernels and installed FA4.

The runtime package is distributed as ``tokenspeed-fa4`` and exposes Python
modules under ``flash_attn.cute``. This module supplies the cutlass-dsl
compatibility shims plus the APIs missing from the installed FA4 version;
the kernels import everything FA4-related through it.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib.util
import inspect as _inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

for _compat_name in ("ThrMma", "ThrCopy"):
    if not hasattr(cute.core, _compat_name):
        setattr(cute.core, _compat_name, getattr(cute, _compat_name))
if not hasattr(cute, "make_fragment"):
    cute.make_fragment = cute.make_rmem_tensor

if importlib.util.find_spec("flash_attn.cute") is None:
    raise ImportError("tokenspeed-fa4 is required for the rel_mha kernels")

from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute import utils
from flash_attn.cute.block_sparse_utils import (
    get_total_block_count,
    handle_block_sparse_empty_tile_correction_sm100,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
)
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.copy_utils import tiled_copy_2d
from flash_attn.cute.fa_logging import fa_log, fa_printf
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.pack_gqa import (
    PackGQA,
    make_packgqa_tiled_tma_atom,
    pack_gqa_layout,
)
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.seqlen_info import SeqlenInfoQK, SeqlenInfoQKNewK
from flash_attn.cute.softmax import SoftmaxSm100 as _Fa4SoftmaxSm100
from flash_attn.cute.softmax import apply_score_mod_inner
from flash_attn.cute.tile_scheduler import (
    ClcState,
    SchedulingMode,
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    StaticPersistentTileScheduler,
)
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments as _Fa4TileSchedulerArguments,
)
from flash_attn.cute.tile_scheduler import (
    TileSchedulerProtocol,
)
from flash_attn.cute.utils import get_batch_from_cu_tensor


def assume_strides_aligned(tensor, align: int = 16):
    """Return strides annotated with the requested byte alignment."""
    elements_per_alignment = align * 8 // tensor.element_type.width
    return (
        *(
            (
                stride
                if isinstance(stride, int)
                else cute.assume(stride, divby=elements_per_alignment)
            )
            for stride in tensor.stride[:-1]
        ),
        tensor.stride[-1],
    )


def assume_tensor_aligned(tensor, align: int = 16):
    """FA4 alignment helper extended with the ``align`` argument."""
    if tensor is None:
        return None
    layout = cute.make_layout(
        tensor.shape, stride=assume_strides_aligned(tensor, align)
    )
    return cute.make_tensor(tensor.iterator, layout)


class NamedBarrierFwdSm100(enum.IntEnum):
    """Named barriers used by the newer bias/MXFP8 forward kernel."""

    Epilogue = enum.auto()
    TmemPtr = enum.auto()
    SoftmaxStatsW0 = enum.auto()
    SoftmaxStatsW1 = enum.auto()
    SoftmaxStatsW2 = enum.auto()
    SoftmaxStatsW3 = enum.auto()
    SoftmaxStatsW4 = enum.auto()
    SoftmaxStatsW5 = enum.auto()
    SoftmaxStatsW6 = enum.auto()
    SoftmaxStatsW7 = enum.auto()
    Softmax = enum.auto()
    Correction = enum.auto()


@dataclass(frozen=True)
class BlockInfo:
    """Block-range helper required by relative bias and dynamic split-KV."""

    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    num_splits: Int32 = 1
    num_splits_dynamic_ptr: Optional[cute.Tensor] = None
    num_n_blocks_per_split: Optional[cutlass.Constexpr[Int32]] = None

    @cute.jit
    def get_n_idx_left_right(
        self, seqlen_info: SeqlenInfoQK, m_idx: Int32
    ) -> Tuple[Int32, Int32]:
        m_idx_actual = m_idx // self.qhead_per_kvhead_packgqa
        if const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            n_idx_right = m_idx_actual + 1 + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            if const_expr(self.window_size_right is not None):
                n_idx_right += self.window_size_right
        else:
            n_idx_right = seqlen_info.seqlen_k
        n_idx_left = 0
        if const_expr(self.is_local and self.window_size_left is not None):
            n_idx_left = cutlass.max(
                m_idx_actual
                + seqlen_info.seqlen_k
                - seqlen_info.seqlen_q
                - self.window_size_left,
                0,
            )
        return n_idx_left, n_idx_right

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        batch_idx: Int32 = 0,
        half_tile_m: bool = False,
        absolute: bool = False,
        half_tile_n: bool = False,
    ) -> Tuple[Int32, Int32]:
        tile_m = self.tile_m // 2 if const_expr(half_tile_m) else self.tile_m
        tile_n = self.tile_n // 2 if const_expr(half_tile_n) else self.tile_n
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, tile_n)
        if const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            m_idx_max = (m_block + 1) * tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = (
                n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
            )
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, tile_n))
        n_block_min = 0
        if const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // tile_n, 0)
        if const_expr(self.is_split_kv and not absolute):
            num_splits = (
                self.num_splits_dynamic_ptr[batch_idx]
                if const_expr(self.num_splits_dynamic_ptr is not None)
                else self.num_splits
            )
            if const_expr(self.num_n_blocks_per_split is not None):
                num_n_blocks_per_split = self.num_n_blocks_per_split
            else:
                num_n_blocks_per_split = (
                    Int32(0)
                    if n_block_max <= n_block_min
                    else (n_block_max - n_block_min + num_splits - 1) // num_splits
                )
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
        return n_block_min, n_block_max

    @cute.jit
    def get_m_block_min_max(
        self, seqlen_info: SeqlenInfoQK, n_block: Int32
    ) -> Tuple[Int32, Int32]:
        seqlen_q = seqlen_info.seqlen_q * self.qhead_per_kvhead_packgqa
        m_block_max = cute.ceil_div(seqlen_q, self.tile_m)
        m_block_min = 0
        if const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            n_idx_min = n_block * self.tile_n
            m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_right = (
                m_idx if const_expr(self.is_causal) else m_idx - self.window_size_right
            )
            m_block_min = max(
                m_block_min,
                m_idx_right * self.qhead_per_kvhead_packgqa // self.tile_m,
            )
        if const_expr(self.is_local and self.window_size_left is not None):
            n_idx_max = (n_block + 1) * self.tile_n
            m_idx = n_idx_max + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_left = m_idx + self.window_size_left
            m_block_max = min(
                m_block_max,
                cute.ceil_div(m_idx_left * self.qhead_per_kvhead_packgqa, self.tile_m),
            )
        return m_block_min, m_block_max

    @cute.jit
    def get_n_block_k_new_min_max(
        self,
        seqlen_info: SeqlenInfoQKNewK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        n_block_min, n_block_max = self.get_n_block_min_max(
            seqlen_info, m_block, split_idx, num_splits
        )
        idx_k_new_min = cutlass.max(
            n_block_min * self.tile_n - seqlen_info.seqlen_k_og, 0
        )
        idx_k_new_max = cutlass.min(
            n_block_max * self.tile_n - seqlen_info.seqlen_k_og,
            seqlen_info.seqlen_k_new,
        )
        n_block_new_min = idx_k_new_min // self.tile_n
        n_block_new_max = (
            cute.ceil_div(idx_k_new_max, self.tile_n)
            if idx_k_new_max > idx_k_new_min
            else n_block_new_min
        )
        return n_block_new_min, n_block_new_max

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        m_idx_min = m_block * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if const_expr(not self.is_local or self.window_size_right is None)
            else n_idx + self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.tile_n)

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        if const_expr(not self.is_local or self.window_size_left is None):
            return n_block_min
        m_idx_max = (m_block + 1) * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
        n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_left = n_idx - self.window_size_left
        return cutlass.max(n_block_min, cute.ceil_div(n_idx_left, self.tile_n))

    @cute.jit
    def get_n_block_max_for_m_block(
        self, seqlen_info: SeqlenInfoQK, m_block: Int32
    ) -> Int32:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(self.is_causal or self.window_size_right is not None):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx_right = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            if const_expr(self.window_size_right is not None):
                n_idx_right += self.window_size_right
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        return n_block_max


@dataclass
class SoftmaxSm100(_Fa4SoftmaxSm100):
    """FA4 softmax extended with optional unscaled row-max tracking."""

    row_max_true: Optional[cute.Tensor] = None

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
        store_row_max: cutlass.Constexpr[bool] = False,
    ) -> "SoftmaxSm100":
        num_rows = 1
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        row_max_true = (
            cute.make_rmem_tensor(num_rows, Float32)
            if const_expr(store_row_max)
            else None
        )
        kwargs = dict(rescale_threshold=rescale_threshold, row_max_true=row_max_true)
        if any(f.name == "max_offset" for f in dataclasses.fields(SoftmaxSm100)):
            # Wheel-lineage base has max_offset, fork-lineage doesn't; pass the default 0 only when accepted.
            kwargs["max_offset"] = 0
        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            100,
            softmax_scale,
            **kwargs,
        )

    @cute.jit
    def reset(self) -> None:
        if const_expr(self.row_max_true is not None):
            self.row_max_true.fill(-Float32.inf)
        return super().reset()

    @cute.jit
    def update_row_max(
        self, acc_S_row: cute.TensorSSA, is_first: int
    ) -> Tuple[Float32, Float32]:
        if const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -Float32.inf else 0.0
            acc_scale = 0.0
            if const_expr(self.row_max_true is not None):
                self.row_max_true[0] = row_max_new
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            if const_expr(self.row_max_true is not None):
                self.row_max_true[0] = max(row_max_new, self.row_max_true[0])
            row_max_safe = row_max_new if row_max_new != -Float32.inf else 0.0
            acc_scale_log2 = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_log2, fastmath=True)
            if const_expr(self.rescale_threshold > 0.0):
                if acc_scale_log2 >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def scale_subtract_rowmax(self, acc_S_row: cute.Tensor, row_max: Float32) -> None:
        assert cute.size(acc_S_row.shape) % 2 == 0
        row_max_scaled = row_max * self.scale_log2
        for index in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[index], acc_S_row[index + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[index], acc_S_row[index + 1]),
                (self.scale_log2, self.scale_log2),
                (-row_max_scaled, -row_max_scaled),
            )


if any(f.name == "row_max_true" for f in dataclasses.fields(_Fa4SoftmaxSm100)):
    # Fork FA4 already carries row_max_true/reset; a second override loops DSL super() resolution.
    SoftmaxSm100 = _Fa4SoftmaxSm100  # noqa: F811


@dataclass
class TileSchedulerArguments(_Fa4TileSchedulerArguments):
    """Newer scheduler arguments accepted by the standalone kernel."""

    qhead_per_kvhead: cutlass.Constexpr[int] = 1
    num_splits_dynamic_ptr: Optional[cute.Tensor] = None
    num_m_blocks_ptr: Optional[cute.Tensor] = None
    varlen_batch_idx_ptr: Optional[cute.Tensor] = None
    num_nheads_in_l2_ptr: Optional[cute.Tensor] = None
    tile_count_semaphore: Optional[cute.Pointer] = None
    persistent_cta_multiplier: cutlass.Constexpr[int] = 1
    causal: cutlass.Constexpr[bool] = False
    local: cutlass.Constexpr[bool] = False
    disable_swizzle: cutlass.Constexpr[bool] = False


class DynamicPersistentVarlenScheduler:
    """Fail clearly for an optional scheduler absent from this FA4 build."""

    @staticmethod
    def to_underlying_arguments(*args, **kwargs):
        raise NotImplementedError(
            "Dynamic persistent varlen requires a newer tokenspeed-fa4 build"
        )


@cute.jit
def gemm_blockscaled(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCtSFA: cute.Tensor,
    tCtSFB: cute.Tensor,
    zero_init: bool | Boolean = True,
    sB: Optional[cute.Tensor] = None,
    **kwargs,
) -> None:
    """Issue block-scaled MMA with per-K-block TMEM scale addresses."""
    num_kblocks = cute.size(tCrA.shape[2])
    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
        scale_coord = (None, None, kblock_idx)
        tiled_mma.set(cute.nvgpu.tcgen05.Field.SFA, tCtSFA[scale_coord].iterator)
        tiled_mma.set(cute.nvgpu.tcgen05.Field.SFB, tCtSFB[scale_coord].iterator)
        tiled_mma.set(
            cute.nvgpu.tcgen05.Field.ACCUMULATE,
            not zero_init or kblock_idx != 0,
        )
        cute.gemm(
            tiled_mma,
            acc,
            tCrA[None, None, kblock_idx],
            tCrB[None, None, kblock_idx],
            acc,
        )


sm100_utils.gemm_blockscaled = gemm_blockscaled


@dsl_user_op
def _fmax_compat(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32 | None = None,
    *,
    loc=None,
    ip=None,
) -> Float32:
    return Float32(
        nvvm.fmax(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


utils.fmax = _fmax_compat


@dsl_user_op
def cvt_bf16x2_ue8m0x2(
    packed_scales: cutlass.Int16, *, loc=None, ip=None
) -> cutlass.Int32:
    """Convert packed UE8M0x2 scales to packed BF16x2."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int16(packed_scales).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.ue8m0x2 $0, $1;",
            "=r,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cvt_tensor_ue8m0_to_bf16(scales: cute.Tensor, scales_out: cute.Tensor) -> None:
    count = cute.size(scales)
    assert count % 2 == 0
    assert cute.size(scales_out) == count
    assert scales.element_type.width == 8
    assert scales_out.element_type.width == 16
    scales_x2 = cute.recast_tensor(scales, dtype=cutlass.Int16)
    scales_out_x2 = cute.recast_tensor(scales_out, dtype=cutlass.Int32)
    for index in cutlass.range_constexpr(count // 2):
        scales_out_x2[index] = cvt_bf16x2_ue8m0x2(scales_x2[index])


smid = utils.smid


# post20260706 wheels renamed apply_mask_sm100's aux_tensors to aux_data.
MASK_TAKES_AUX_TENSORS = (
    "aux_tensors" in _inspect.signature(AttentionMask.apply_mask_sm100).parameters
)
