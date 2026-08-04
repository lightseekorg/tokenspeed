# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026, Colfax International. (modifications)

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# - split-kv
#
# Colfax modifications:
# - relative bias
# - MXFP8 dtype
#
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_ROOT = os.path.join(_SCRIPT_DIR, ".cache", "flash_fwd_sm100_bias")
# Never touch TMPDIR here: torch's symmetric-memory rendezvous binds a
# unix socket under it, and this package-relative path exceeds the 108-char
# sun_path limit -> "Failed to bind socket: Invalid argument" on every rank.
for _cache_name, _cache_path in (
    ("CUTE_DSL_CACHE_DIR", os.path.join(_CACHE_ROOT, "cute_dsl")),
    ("TRITON_CACHE_DIR", os.path.join(_CACHE_ROOT, "triton")),
    ("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE_ROOT, "torchinductor")),
):
    os.environ.setdefault(_cache_name, _cache_path)
    os.makedirs(os.environ[_cache_name], exist_ok=True)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

# quack-kernels 0.4.1 still annotates these classes through cute.core.
# CuTe DSL 4.6 exports them from the public cute namespace instead.
for _compat_name in ("ThrMma", "ThrCopy"):
    if not hasattr(cute.core, _compat_name):
        setattr(cute.core, _compat_name, getattr(cute, _compat_name))
if not hasattr(cute, "make_fragment"):
    cute.make_fragment = cute.make_rmem_tensor

import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as cutlass_pipeline
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Boolean, Float32, Int32, Int64, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase

from .fmha_bias_helper import (
    MASK_TAKES_AUX_TENSORS,
    AttentionMask,
    BlockInfo,
    BlockSparseTensors,
    ClcState,
    DynamicPersistentVarlenScheduler,
    NamedBarrierFwdSm100,
    PackGQA,
    PagedKVManager,
    SchedulingMode,
    SeqlenInfoQK,
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    SoftmaxSm100,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
    apply_score_mod_inner,
    assume_tensor_aligned,
    cvt_tensor_ue8m0_to_bf16,
    fa_log,
    fa_printf,
    get_total_block_count,
    handle_block_sparse_empty_tile_correction_sm100,
    make_packgqa_tiled_tma_atom,
    pack_gqa_layout,
    pipeline_custom,
    produce_block_sparse_loads_sm100,
    sm100_desc,
    sm100_utils,
    smid,
    softmax_block_sparse_sm100,
    utils,
)


@dataclass
class SfS2TCopies:
    tCtSFQ: cute.Tensor
    tCtSFK: cute.Tensor
    tiled_copy_sfq: cute.TiledCopy
    tiled_copy_sfk: cute.TiledCopy
    tCsSFQ_s2t: cute.Tensor
    tCtSFQ_s2t: cute.Tensor
    tCsSFK_s2t: cute.Tensor
    tCtSFK_s2t: cute.Tensor


# === TUNING KNOBS (agent-editable) ===
# Keys: (use_2cta_instrs: bool, is_causal: bool, head_dim_padded: int, is_sm103: bool)
# Values:
#   ex2_emu_freq: int — how often to use emulated exp2 (0=all hardware exp2, higher=more emulation).
#                        SM103 has fast native exp2, so set freq=0 there.
#   ex2_emu_res: int — (hd256 only) number of fragment-pairs per freq period to emulate.
#   ex2_emu_start_frg: int — fragment index to start emulation from
#   num_regs_softmax: int — register count for softmax warps (multiple of 8)
#   num_regs_correction: int — register count for correction warps (multiple of 8)
#   num_regs_other is derived: 512 - num_regs_softmax * 2 - num_regs_correction
#                  (hd256 exception: num_regs_other is fixed at 32, not derived)
_TUNING_CONFIG = {
    (True, False, 128, False): {
        "ex2_emu_freq": 10,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 176,
        "num_regs_correction": 88,
    },
    (False, True, 128, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 192,
        "num_regs_correction": 72,
    },
    (True, False, 192, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 184,
        "num_regs_correction": 80,
    },
    (False, True, 192, False): {
        "ex2_emu_freq": 32,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 192,
        "num_regs_correction": 72,
    },
    (True, False, 128, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 80,
    },
    (False, True, 128, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 64,
    },
    (True, False, 192, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 64,
    },
    (False, True, 192, True): {
        "ex2_emu_freq": 0,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 176,
        "num_regs_correction": 72,
    },
    (True, False, 256, False): {
        "ex2_emu_freq": 14,
        "ex2_emu_res": 6,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 256,
        "num_regs_correction": 160,
    },
    (True, True, 256, False): {
        "ex2_emu_freq": 14,
        "ex2_emu_res": 6,
        "ex2_emu_start_frg": 0,
        "num_regs_softmax": 256,
        "num_regs_correction": 160,
    },
}
# === END TUNING KNOBS ===


class FlashAttentionForwardSm100:

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        bias_block_size: int = 128,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        is_dynamic_persistent_varlen: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        use_clc_scheduler: bool = False,
        has_bias: bool = False,
        rel_extent_padded: int = 128,
        has_scheduler_metadata: bool = False,
        seqlen_k_per_split: Optional[int] = None,
        qk_blockscaled: bool = False,
        v_dequant: bool = False,
        q_sf_interleaved: bool = False,
        kv_sf_interleaved: bool = False,
        use_pdl: bool = True,
        paged_kv_blocks_per_page: int = 1,
    ):
        # KV tiles (n_block_size rows) per physical page: page_size ==
        # paged_kv_blocks_per_page * n_block_size. 1 keeps the historical
        # page == n_block TMA indexing; >1 divides the page-table index and
        # selects the in-page tile (TMA path only; the non-TMA manager
        # already divmods per row).
        assert paged_kv_blocks_per_page >= 1
        self.paged_kv_blocks_per_page = paged_kv_blocks_per_page
        self.use_pdl = use_pdl
        self.has_bias = has_bias
        self.rel_extent_padded = rel_extent_padded
        assert rel_extent_padded % n_block_size == 0
        self.bias_n_max = rel_extent_padded // n_block_size if has_bias else 0
        self.qk_blockscaled = qk_blockscaled
        self.v_dequant = v_dequant
        self.q_sf_interleaved = q_sf_interleaved
        self.kv_sf_interleaved = kv_sf_interleaved
        self.use_cpasync_to_load_sfq = self.qk_blockscaled and not self.q_sf_interleaved
        self.use_tma_KV = not paged_kv_non_tma
        assert not (
            qk_blockscaled and self.q_sf_interleaved and is_varlen_q
        ), "if varlen q, can't have SFQ interleaved"
        assert not (
            qk_blockscaled and not self.kv_sf_interleaved and self.use_tma_KV
        ), "if scale KV not interleaved, can't use tma KV"
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.bias_block_size = bias_block_size
        self.q_stage = q_stage
        assert self.q_stage in [1, 2]
        self.bias_stage = (
            2 if (self.q_stage == 2 or (self.q_stage == 1 and not is_split_kv)) else 1
        )
        # self.bias_stage = self.q_stage
        assert self.bias_stage >= self.q_stage
        self.use_2cta_instrs = use_2cta_instrs and False
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        # assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, f"Only SM 10.x and 11.x are supported but got {self.arch}"
        assert seqlen_k_per_split is None or seqlen_k_per_split % n_block_size == 0
        self.num_n_blocks_per_split = (
            seqlen_k_per_split // n_block_size
            if seqlen_k_per_split is not None
            else None
        )

        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        # cta_tiler M includes only 1 CTA, the scheduler will take into account the cluster shape
        self.cta_tiler = (
            self.q_stage * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        # With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
        # Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
        self.mma_tiler_qk = (
            self.cta_group_size * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.cta_group_size * m_block_size,
            self.head_dim_v_padded,
            n_block_size,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)

        self.dynamic_persistent = is_dynamic_persistent_varlen
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        self.use_tma_O = (
            not (self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0)
            # and not is_varlen_q
        )
        self.ragged_O = self.use_tma_O and self.is_varlen_q
        self.use_correction_warps_for_epi = not self.use_tma_O
        self.q_subtile_factor = q_subtile_factor
        assert not (
            self.is_split_kv and self.head_dim_v_padded >= 192
        ), "SplitKV is not supported for hdim >= 192"
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        self.has_scheduler_metadata = has_scheduler_metadata
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        self.is_sm103 = is_sm103
        # enable_ex2_emu is derived: True if tuning config has freq > 0, else fallback to default logic
        _default_enable_ex2_emu = (
            self.head_dim_padded <= 128
            or (
                self.head_dim_padded == 192
                and self.use_2cta_instrs
                and not self.is_causal
                and not self.is_local
            )
        ) and not is_sm103
        self.enable_ex2_emu = _default_enable_ex2_emu
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = (
            self.head_dim_padded == 192 and self.head_dim_v_padded >= 64
        ) or (
            self.head_dim_v_padded >= 128
            and (self.is_split_kv or (has_bias and self.q_stage == 2))
        )
        if self.q_stage == 2 and self.v_dequant:
            self.overlap_sO_sQ = True

        # useful prints
        # print("is causal = ", self.is_causal)
        # print("is local = ", self.is_local)
        # print("has_bias = ", has_bias)
        # print("rel_extent_padded = ", rel_extent_padded)
        # print("bias_n_max = ", self.bias_n_max)
        # print("q_stage = ", q_stage)
        # print("bias_stage (could recalc) = ", self.bias_stage)
        # print("bias block size = ", self.bias_block_size)
        # print("is_split_kv = ", self.is_split_kv)
        # print("pack_gqa = ", self.pack_gqa)
        # print("overlap_sO_sQ = ", self.overlap_sO_sQ)
        # print("use_tma_O = ", self.use_tma_O)
        # print("ragged_O = ", self.ragged_O)

        assert self.use_tma_KV or not (
            self.check_hdim_oob or self.check_hdim_v_oob
        ), "Paged KV does not support irregular head dim"

        # ClC does not compose with these other features, so disable even if requested
        self.use_clc_scheduler = (
            use_clc_scheduler and not is_dynamic_persistent_varlen and self.use_tma_KV
        )
        # print("use_clc_scheduler (provided) = ", use_clc_scheduler)
        # print("use_clc_scheduler (chosen) = ", self.use_clc_scheduler)
        self.sched_stages = 1
        if self.use_clc_scheduler:
            assert (
                self.cluster_shape_mn[1] == 1
            ), f"CLC requires cluster N == 1: {self.cluster_shape_mn}"
            assert self.cluster_shape_mn[0] in (
                1,
                2,
            ), f"bad CLC cluster M: {self.cluster_shape_mn}"
            assert (
                self.cluster_shape_mn[0] == self.cta_group_size
            ), f"CLC cluster M != cta_group_size: {self.cluster_shape_mn}, {self.cta_group_size}"

        self.is_persistent = (
            is_persistent or self.dynamic_persistent or self.use_clc_scheduler
        )

        if self.dynamic_persistent:
            self.scheduling_mode = SchedulingMode.DYNAMIC
        elif self.use_clc_scheduler:
            self.scheduling_mode = SchedulingMode.CLC
        else:
            self.scheduling_mode = SchedulingMode.STATIC

        self.use_varlen_scheduler = False
        if self.is_varlen_q:
            if self.dynamic_persistent:
                self.use_varlen_scheduler = True
                self.TileScheduler = DynamicPersistentVarlenScheduler
            elif is_persistent:
                self.TileScheduler = (
                    StaticPersistentTileScheduler
                    if not self.use_clc_scheduler
                    else SingleTileLPTScheduler
                )
            else:
                self.use_varlen_scheduler = True
                self.TileScheduler = SingleTileVarlenScheduler
        elif self.is_causal or self.is_local or self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler

        self.static_persistent = (
            is_persistent and not self.use_clc_scheduler and not self.dynamic_persistent
        )

        fa_log(
            1,
            f"TileScheduler={self.TileScheduler.__name__}, scheduling_mode={self.scheduling_mode.name}, USE_2CTA={self.use_2cta_instrs}",
        )

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        self.use_tma_Q = not (
            self.pack_gqa and self.m_block_size % self.qhead_per_kvhead != 0
        )

        if self.q_stage == 1:
            if not self.use_tma_KV or not self.use_tma_Q:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            else:
                self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()

        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids

        self.dynamic_scheduler_warp_id = self.load_warp_ids[0]

        non_empty_warps_ids = set(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
            )
        )
        self.num_non_empty_warps = len(non_empty_warps_ids)

        self.clc_scheduler_warp_id = (
            self.empty_warp_ids[0] if self.use_clc_scheduler else None
        )

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # vec buffer for row_max & row_sum (unused)
        self.tmem_vec_offset = self.tmem_s_offset

        # Look up tuning config for register counts and ex2_emu params
        _tune_key = (
            self.use_2cta_instrs,
            self.is_causal,
            self.head_dim_padded,
            self.is_sm103,
        )
        self._tune = _TUNING_CONFIG.get(_tune_key, {})
        if "ex2_emu_freq" in self._tune:
            self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0

        if self.v_dequant:
            if self.q_stage == 2:
                bias_reg_factor = 8 if self.has_bias else 0
                self.num_regs_softmax = 176 + bias_reg_factor
                self.num_regs_correction = (
                    112 - 2 * bias_reg_factor if not paged_kv_non_tma else 80
                )
                self.num_regs_other = (
                    48 if not paged_kv_non_tma else 80 - bias_reg_factor * 2
                )
                self.num_regs_load = self.num_regs_other
            else:
                self.num_regs_softmax = 184
                self.num_regs_correction = 112
                self.num_regs_other = 88
                self.num_regs_load = 88
        elif self.head_dim_padded < 96:
            self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
            self.num_regs_correction = 64
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
            self.num_regs_load = self.num_regs_other
        else:
            if not paged_kv_non_tma and "num_regs_softmax" in self._tune:
                self.num_regs_softmax = self._tune["num_regs_softmax"]
                self.num_regs_correction = self._tune["num_regs_correction"]
            elif not paged_kv_non_tma:
                self.num_regs_softmax = 192
                self.num_regs_correction = 80
            else:
                self.num_regs_softmax = 184
                self.num_regs_correction = 64
            self.num_regs_other = (
                512 - self.num_regs_softmax * 2 - self.num_regs_correction
            )
            self.num_regs_load = self.num_regs_other

        assert (
            self.num_regs_correction
            + self.num_regs_softmax * self.q_stage
            + self.num_regs_other
            <= 512
        )
        assert (
            self.num_regs_correction
            + self.num_regs_softmax
            + self.num_regs_other
            + self.num_regs_load
            <= 512
        )

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        smem_size_q = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_padded
            * self.q_dtype.width
            // 8
        )
        if self.qk_blockscaled:
            smem_size_q += (
                self.q_stage
                * self.m_block_size
                * self.head_dim_padded
                // self.qk_sf_vec_size
                * self.sfq_dtype.width
                // 8
            )
        smem_size_bias = (
            self.bias_stage
            * self.bias_block_size
            * self.n_block_size
            * self.bias_dtype.width
            // 8
            if self.has_bias
            else 0
        )
        smem_size_o = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_v_padded
            * self.o_dtype.width
            // 8
        )
        smem_size_q_o = (
            smem_size_q + smem_size_o
            if not self.overlap_sO_sQ
            else max(smem_size_q, smem_size_o)
        )
        smem_size_q_o_bias = (
            smem_size_q + smem_size_o + smem_size_bias
            if not self.overlap_sO_sQ
            else max(smem_size_q + smem_size_bias, smem_size_o)
        )
        smem_size_k_per_stage = (
            self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        )
        if self.qk_blockscaled:
            smem_size_k_per_stage += (
                self.n_block_size
                * self.head_dim_padded
                // self.qk_sf_vec_size
                * self.sfk_dtype.width
                // 8
            )
        smem_size_v_per_stage = (
            self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        )
        if self.v_dequant:
            smem_size_v_per_stage += (
                self.n_block_size
                * self.head_dim_v_padded
                // self.v_sf_vec_size
                * self.sfv_dtype.width
                // 8
            )
            smem_size_v_per_stage += (
                self.n_block_size * self.head_dim_v_padded * self.v_mma_dtype.width // 8
            )
        if self.v_dequant:
            # separate kv pipelines for mxf8
            smem_size_kv_per_stage = smem_size_k_per_stage + smem_size_v_per_stage
        else:
            smem_size_kv_per_stage = (
                max(smem_size_k_per_stage, smem_size_v_per_stage) // self.cta_group_size
            )
        kv_stage = (224 * 1024 - smem_size_q_o_bias) // smem_size_kv_per_stage
        # print("kv stage = ", kv_stage)
        # failsafe, recalculate
        if kv_stage <= 1 and self.q_stage == 1 and self.bias_stage == 2:
            self.bias_stage = 1
            smem_size_bias = (
                self.bias_stage
                * self.bias_block_size
                * self.n_block_size
                * self.bias_dtype.width
                // 8
                if self.has_bias
                else 0
            )
            smem_size_q_o_bias = (
                smem_size_q + smem_size_o + smem_size_bias
                if not self.overlap_sO_sQ
                else max(smem_size_q + smem_size_bias, smem_size_o)
            )
            kv_stage = (224 * 1024 - smem_size_q_o_bias) // smem_size_kv_per_stage
            # print(f"recalc {kv_stage=}, set bias_stage=1")
        if (
            self.head_dim_padded == 192
            and self.head_dim_v_padded == 128
            and kv_stage == 2
        ):
            # For hdim 192,128, we can fit 3 stages if we use uneven_kv_smem
            kv_stage = 3
        v_mma_stage = kv_stage
        # TODO: revisit hard-coded stage counts for mxfp8
        if self.v_dtype.width == 8:
            kv_stage = 2
            v_mma_stage = 1 if self.has_bias and self.q_stage == 2 else kv_stage
        self.kv_stage = kv_stage
        self.v_mma_stage = v_mma_stage
        # print("kv_stage", self.kv_stage)
        # print("v_mma_stage", self.v_mma_stage)
        self.s_stage = 2
        assert self.s_stage >= self.q_stage
        # For hdim 192,128 1CTA, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            self.head_dim_padded == 192
            and self.head_dim_v_padded == 128
            and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.n_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        softmax_scale: Float32,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        qk_sf_vec_size: cutlass.Constexpr[Optional[int]] = None,
        v_sf_vec_size: cutlass.Constexpr[Optional[int]] = None,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        mBias: Optional[
            cute.Tensor
        ] = None,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        tile_count_semaphore: Optional[cute.Tensor] = None,
        max_seqlen_q: Int32 | int | None = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        self.kv_size_ratio = self.v_dtype.width // self.k_dtype.width
        # scale factor setup for blockscaled
        self.sfq_dtype = mSFQ.element_type if const_expr(mSFQ is not None) else None
        self.sfk_dtype = mSFK.element_type if const_expr(mSFK is not None) else None
        self.sfv_dtype = mSFV.element_type if const_expr(mSFV is not None) else None
        self.qk_sf_vec_size = (
            qk_sf_vec_size if const_expr(qk_sf_vec_size is not None) else 1
        )
        self.v_sf_vec_size = (
            v_sf_vec_size if const_expr(v_sf_vec_size is not None) else 1
        )

        if const_expr(self.v_dequant):
            self.v_mma_dtype = cutlass.BFloat16
            self.kv_size_ratio = 1
        else:
            self.v_mma_dtype = self.v_dtype
        # bias setup
        self.bias_dtype = (
            mBias.element_type if const_expr(mBias is not None) else self.q_dtype
        )
        mQ, mK, mV, mO, mBias = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mO, mBias)
        ]
        mSFQ, mSFK, mSFV = [
            assume_tensor_aligned(t, align=4) for t in (mSFQ, mSFK, mSFV)
        ]
        Q_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        # (s_q, static, nheads, batch) or (total_q, static, nheads)
        mQ, mBias = [
            (
                cute.make_tensor(
                    t.iterator, cute.select(t.layout, mode=Q_layout_transpose)
                )
                if const_expr(t is not None)
                else None
            )
            for t in (mQ, mBias)
        ]
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        if const_expr(self.is_split_kv):
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = (
                [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            )
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = (
                [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            )
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
            num_splits = Int32(1)
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        mLSE, mRowMax = (
            (
                cute.make_tensor(
                    t.iterator, cute.select(t.layout, mode=LSE_layout_transpose)
                )
                if const_expr(t is not None)
                else None
            )
            for t in (mLSE, mRowMax)
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = (
            [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        )
        mV = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )

        # Broadcast SF tensors to blockscaled layout
        if const_expr(self.qk_blockscaled):
            # ((32,4),(32,4)):((16,4),(0,1))
            sf_atom = blockscaled_utils.BlockScaledBasicChunk(
                self.qk_sf_vec_size
            ).layout
            sfk_layout = cute.tile_to_shape(sf_atom, mK.shape, (2, 1, 3, 4))
            # print("sf_atom = ", sf_atom)
            # print("sfk_layout = ", sfk_layout)
            if const_expr(self.q_sf_interleaved):
                # (s_q, d, h, b)
                sfq_layout = cute.tile_to_shape(sf_atom, mQ.shape, (2, 1, 3, 4))
                mSFQ = cute.make_tensor(mSFQ.iterator, sfq_layout)
            else:
                # (s_q, d, h, b) or (total_q, d, h)
                mSFQ = cute.make_tensor(
                    mSFQ.iterator, cute.select(mSFQ.layout, mode=Q_layout_transpose)
                )

            if const_expr(self.kv_sf_interleaved):
                # (s_k, d, h_k, b)
                mSFK = cute.make_tensor(mSFK.iterator, sfk_layout)
            else:
                assert (
                    not self.use_tma_KV
                ), "can't use TMA to load SFK if not interleaved in gmem"
                # (num_pages, page_size, h_k, d//32) -> (page_size, d//32, h_k, num_pages)
                mSFK = cute.make_tensor(
                    mSFK.iterator, cute.select(mSFK.layout, mode=KV_layout_transpose)
                )

        if const_expr(self.v_dequant):
            sfv_atom = blockscaled_utils.BlockScaledBasicChunk(
                self.v_sf_vec_size
            ).layout
            sfv_layout = cute.tile_to_shape(sfv_atom, mV.shape, (2, 1, 3, 4))
            # print("sfv_atom = ", sfv_atom)
            # print("sfv_layout = ", sfv_layout)

            if const_expr(self.kv_sf_interleaved):
                # (dv, s_k, h_k, b)
                mSFV = cute.make_tensor(mSFV.iterator, sfv_layout)
            else:
                assert (
                    not self.use_tma_KV
                ), "can't use TMA to load SFV if not interleaved in gmem"
                # (num_pages, page_size, h_k, d//32) -> (page_size, d//32, h_k, num_pages)
                mSFV = cute.make_tensor(
                    mSFV.iterator, cute.select(mSFV.layout, mode=KV_layout_transpose)
                )
                # (page_size, d//32, h_k, num_pages) -> (d//32, page_size, h_k, num_pages)
                mSFV = cute.make_tensor(
                    mSFV.iterator, cute.select(mSFV.layout, mode=V_layout_transpose)
                )

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(
            not self.qk_blockscaled
            and not self.v_dequant
            and self.q_dtype != self.v_dtype
        ):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        if const_expr(mBias is not None and self.bias_dtype != self.v_mma_dtype):
            raise TypeError(f"Type mismatch: {self.v_mma_dtype} != {self.bias_dtype}")
        if const_expr(
            self.qk_blockscaled
            and (self.sfk_dtype is None or self.qk_sf_vec_size is None)
        ):
            raise TypeError(
                "SFK dtype and qk_sf_vec_size must be provided when qk_blockscaled."
            )
        if const_expr(self.qk_blockscaled and self.sfq_dtype != self.sfk_dtype):
            raise TypeError(f"Type mismatch: {self.sfq_dtype} != {self.sfk_dtype}")

        self._setup_attributes()

        # Compute SF TMEM offsets for blockscaled
        if const_expr(self.qk_blockscaled):
            self.num_sfq_tmem_cols = 4 if self.qk_sf_vec_size == 32 else 8
            self.num_sfk_tmem_cols = self.num_sfq_tmem_cols
            # SF for stage i overlaps with tmem_s_offset[i ^ 1]:
            #   stage 0 SFs start at col 128 (tmem_s_offset[1])
            #   stage 1 SFs start at col 0   (tmem_s_offset[0])
            # No additional TMEM needed — SFs fit within the S region of the other stage.
            self.tmem_sfq_offset = [
                self.tmem_s_offset[1 - i] for i in range(self.q_stage)
            ]
            self.tmem_sfk_offset = [
                self.tmem_sfq_offset[i] + self.num_sfq_tmem_cols
                for i in range(self.q_stage)
            ]

        # This can be tuned
        # This is currently very ad-hoc, we should tune it systematically
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 1)
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = self._tune.get("ex2_emu_freq", 16)
            if const_expr(
                self.pack_gqa
                and self.head_dim_padded > 64
                and not self.is_causal
                and not self.is_local
            ):
                self.ex2_emu_freq = (
                    32
                    if mCuSeqlensQ is not None or mSeqUsedQ is not None
                    else self._tune.get("ex2_emu_freq", 10)
                )

        self.store_row_max = mRowMax is not None

        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        if const_expr(not self.qk_blockscaled):
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                q_major_mode,
                k_major_mode,
                self.qk_acc_dtype,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                q_major_mode,
                k_major_mode,
                self.sfq_dtype,
                self.qk_sf_vec_size,
                cta_group,
                self.mma_tiler_qk[:2],
            )
            mma_inst_shape_mn_sfk = (
                self.mma_tiler_qk[0] // (2 if self.use_2cta_instrs else 1),
                cute.round_up(self.mma_tiler_qk[1], 128),
            )
            self.mma_tiler_qk_sfk = (*mma_inst_shape_mn_sfk, self.mma_tiler_qk[2])
            self.tiled_mma_qk_sfk = (
                sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                    self.q_dtype,
                    q_major_mode,
                    k_major_mode,
                    self.sfq_dtype,
                    self.qk_sf_vec_size,
                    tcgen05.CtaGroup.ONE,
                    mma_inst_shape_mn_sfk,
                )
            )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_mma_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        if const_expr(self.qk_blockscaled):
            cta_layout_sfk_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (self.tiled_mma_qk_sfk.thr_id.shape,),
            )

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage * self.kv_size_ratio,
        )
        if const_expr(self.qk_blockscaled):
            sSFQ_layout = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.qk_sf_vec_size,
                self.q_stage,
            )
            sSFK_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_qk_sfk,
                self.mma_tiler_qk_sfk,
                self.qk_sf_vec_size,
                self.kv_stage,
            )
        else:
            sSFQ_layout = None
            sSFK_layout = None
        if const_expr(self.v_dequant):
            # Blockscaled tiled_mma for SFV layout computation only (PV GEMM stays non-blockscaled)
            mma_inst_shape_mn_sfv = (
                self.mma_tiler_pv[0] // (2 if self.use_2cta_instrs else 1),
                cute.round_up(self.mma_tiler_pv[1], 128),
            )
            mma_tiler_pv_sfv = (*mma_inst_shape_mn_sfv, self.mma_tiler_pv[2])
            self.tiled_mma_pv_sfv = (
                sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                    self.v_dtype,
                    p_major_mode,
                    v_major_mode,
                    self.sfv_dtype,
                    self.v_sf_vec_size,
                    tcgen05.CtaGroup.ONE,
                    mma_inst_shape_mn_sfv,
                )
            )
            cta_layout_sfv_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (self.tiled_mma_pv_sfv.thr_id.shape,),
            )
            sSFV_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_pv_sfv,
                mma_tiler_pv_sfv,
                self.v_sf_vec_size,
                self.kv_stage,
            )
        else:
            sSFV_layout = None
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.v_mma_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_mma_dtype, self.v_mma_stage
        )
        if const_expr(self.v_dequant):
            tiled_mma_pv_vq = sm100_utils_basic.make_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                v_major_mode,
                self.pv_acc_dtype,
                cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )
            sVq_layout = sm100_utils_basic.make_smem_layout_b(
                tiled_mma_pv_vq, self.mma_tiler_pv, self.v_dtype, self.kv_stage
            )
        else:
            tiled_mma_pv_vq = None
            sVq_layout = None
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )
        if const_expr(not self.v_dequant and not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up.
            # When K and V have different dtypes (e.g., FP8 K and BF16 V), the same element-count stride
            # maps to different byte offsets. We compute the stage stride in bytes and convert back.
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            if const_expr(not self.uneven_kv_smem):
                # Compute in bytes for proper alignment when dtypes differ
                stride_sK_bytes = const_expr(stride_sK * self.k_dtype.width // 8)
                stride_sV_bytes = const_expr(stride_sV * self.v_dtype.width // 8)
                stage_stride_bytes = const_expr(max(stride_sK_bytes, stride_sV_bytes))
                sK_stage_stride = const_expr(
                    stage_stride_bytes * 8 // self.k_dtype.width
                )
                sV_stage_stride = const_expr(
                    stage_stride_bytes * 8 // self.v_dtype.width
                )
            else:
                sK_stage_stride = const_expr((stride_sK + stride_sV) // 2)
                sV_stage_stride = sK_stage_stride
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], sK_stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.v_mma_stage),
                    stride=(*sV_layout.outer.stride[:-1], sV_stage_stride),
                ),
            )

        mO_og = mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )
            if const_expr(mRowMax is not None):
                mRowMax = pack_gqa_layout(
                    mRowMax, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )
            if const_expr(mBias is not None):
                mBias = pack_gqa_layout(
                    mBias, self.qhead_per_kvhead, nheads_kv, head_idx=2
                )
            if const_expr(self.qk_blockscaled):
                mSFQ = pack_gqa_layout(
                    mSFQ, self.qhead_per_kvhead, nheads_kv, head_idx=2
                )

        self.tma_copy_bytes = {
            name: (
                cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
                if const_expr(mX is not None)
                else 0
            )
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout if const_expr(not self.v_dequant) else sVq_layout),
                ("SFQ", mSFQ, sSFQ_layout),
                ("SFK", mSFK, sSFK_layout),
                ("SFV", mSFV, sSFV_layout),
            ]
        }
        for name in ("Q", "K", "V", "SFQ", "SFK", "SFV"):
            self.tma_copy_bytes[name] *= self.cta_group_size
        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        if const_expr(self.use_tma_Q):
            tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mQ,
                cute.select(sQ_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            gmem_tiled_copy_Q = None
        else:
            tma_atom_Q = None
            async_copy_elems = 128 // self.q_dtype.width
            num_load_threads = cute.arch.WARP_SIZE * len(self.load_warp_ids)
            threads_per_row = math.gcd(
                self.head_dim_padded // async_copy_elems, num_load_threads
            )
            gmem_tiled_copy_Q = copy_utils.tiled_copy_2d(
                self.q_dtype,
                threads_per_row,
                num_load_threads,
                async_copy_elems,
                is_async=True,
            )

        tma_atom_K = None
        tma_atom_V = None
        tma_atom_SFQ = None
        tma_atom_SFK = None
        tma_atom_SFV = None
        tma_atom_O = None
        tma_atom_bias = None
        gmem_tiled_copy_SFQ = None
        gmem_tiled_copy_O = None

        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            # TMA load for V (uses sVq_layout for v_dequant)
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                (
                    cute.select(sVq_layout, mode=[0, 1, 2])
                    if const_expr(self.v_dequant)
                    else cute.select(sV_layout, mode=[0, 1, 2])
                ),
                self.mma_tiler_pv,
                tiled_mma_pv_vq if const_expr(self.v_dequant) else tiled_mma_pv,
                cta_layout_vmnk.shape,
            )

        if const_expr(self.qk_blockscaled):
            sfq_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma_qk.thr_id
            )
            if const_expr(not self.use_cpasync_to_load_sfq):
                tma_atom_SFQ, mSFQ = cute.nvgpu.make_tiled_tma_atom_A(
                    sfq_tma_op,
                    mSFQ,
                    cute.select(sSFQ_layout, mode=[0, 1, 2]),
                    self.mma_tiler_qk,
                    tiled_mma_qk,
                    cta_layout_vmnk.shape,
                    internal_type=cutlass.Int16,
                )
            # cpasync fallback for load SFQ
            atom_async_copy_SFQ = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                self.sfq_dtype,
                num_bits_per_copy=32,
            )
            num_load_threads = cute.arch.WARP_SIZE * len(self.load_warp_ids)
            thr_layout_SFQ = cute.make_ordered_layout(
                (num_load_threads, 1),
                order=(0, 1),
            )
            val_layout_SFQ = cute.make_layout((1, 4))
            gmem_tiled_copy_SFQ = cute.make_tiled_copy_tv(
                atom_async_copy_SFQ,
                thr_layout_SFQ,
                val_layout_SFQ,
            )
            if const_expr(self.use_tma_KV):
                assert (
                    self.kv_sf_interleaved
                ), "SFK must be interleaved in gmem to use TMA"
                sfk_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                    self.cluster_shape_mn, tiled_mma_qk.thr_id
                )
                tma_atom_SFK, mSFK = cute.nvgpu.make_tiled_tma_atom_B(
                    sfk_tma_op,
                    mSFK,
                    cute.select(sSFK_layout, mode=[0, 1, 2]),
                    self.mma_tiler_qk_sfk,
                    self.tiled_mma_qk_sfk,
                    cta_layout_sfk_vmnk.shape,
                    internal_type=cutlass.Int16,
                )
        if const_expr(self.v_dequant and self.use_tma_KV):
            assert self.kv_sf_interleaved, "SFV must be interleaved in gmem to use TMA"
            sfv_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma_pv.thr_id
            )
            tma_atom_SFV, mSFV = cute.nvgpu.make_tiled_tma_atom_B(
                sfv_tma_op,
                mSFV,
                cute.select(sSFV_layout, mode=[0, 1, 2]),
                mma_tiler_pv_sfv,
                self.tiled_mma_pv_sfv,
                cta_layout_sfv_vmnk.shape,
                internal_type=cutlass.Int16,
            )

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        make_tiled_tma_atom_fn = (
            partial(
                make_packgqa_tiled_tma_atom,
                qhead_per_kvhead=self.qhead_per_kvhead,
                head_idx=2,
            )
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.ragged_O):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, mO = make_tiled_tma_atom_fn(
                tma_store_op,
                mO_tma,
                cute.select(sO_layout, mode=[0, 1]),
                self.epi_tile,
            )
        else:
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(
                atom_universal_copy, tO_layout, vO_layout
            )

        if const_expr(mBias is not None):
            bias_layout_enum = cutlass.utils.LayoutEnum.from_tensor(mBias)
            self.bias_major_mode = bias_layout_enum.mma_major_mode()
            if const_expr(self.bias_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mBias is wrong")
            # (m_block_size, n_block_size, q_stage)
            sBias_layout = sm100_utils_basic.make_smem_layout_epi(
                self.bias_dtype,
                bias_layout_enum,
                (self.bias_block_size, self.n_block_size),
                self.bias_stage,
            )
            sBias_size = cute.cosize(sBias_layout)
            self.tma_copy_bytes["bias"] = cute.size_in_bytes(
                self.bias_dtype, cute.select(sBias_layout, mode=[0, 1])
            )
            tma_atom_bias, mBias = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mBias,
                cute.select(sBias_layout, mode=[0, 1]),
                (self.bias_block_size, self.n_block_size),
                1,  # no mcast
            )
            bias_s2r_thr_layout = cute.make_ordered_layout(
                (self.bias_block_size, 1), order=(1, 0)
            )
            bias_s2r_val_layout = cute.make_ordered_layout(
                (1, 128 // self.bias_dtype.width), order=(1, 0)
            )
            bias_s2r_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.bias_dtype,
                num_bits_per_copy=128,
            )
            bias_s2r_tiled_copy = cute.make_tiled_copy_tv(
                bias_s2r_copy_atom, bias_s2r_thr_layout, bias_s2r_val_layout
            )
        else:
            sBias_layout = None
            sBias_size = 0
            bias_s2r_tiled_copy = None

        TileScheduler = self.TileScheduler
        _num_block_divisor = self.cta_tiler[0] * (
            self.cta_group_size
            if not self.static_persistent and self.cta_group_size > 1
            else 1
        )
        if const_expr(max_seqlen_q is None):
            eff_seqlen_q = cute.size(mQ.shape[0])
        else:
            eff_seqlen_q = (
                max_seqlen_q
                if const_expr(not self.pack_gqa)
                else max_seqlen_q * self.qhead_per_kvhead
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(eff_seqlen_q, _num_block_divisor),
            cute.size(mQ.shape[2]),
            (
                cute.size(mQ.shape[3])
                if const_expr(mCuSeqlensQ is None)
                else cute.size(mCuSeqlensQ.shape[0] - 1)
            ),
            num_splits,
            (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mQ.shape[1],
            mV.shape[
                0
            ],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=(
                cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=not self.static_persistent and self.cta_group_size > 1,
            num_splits_dynamic_ptr=num_splits_dynamic_ptr,
            tile_count_semaphore=(
                tile_count_semaphore.iterator
                if tile_count_semaphore is not None
                else None
            ),
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        # cute.printf("grid dim = {}", grid_dim)

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        # if Q is fp8 and O is fp32, rely on overlapping O with all of QKV
        # O stage=0 overlaps with Q, O stage=1 overlaps with KV
        QO_dilation_factor = min(self.o_dtype.width // self.q_dtype.width, 2)
        # sQ_size = (
        #     cute.cosize(sQ_layout) if const_expr(not self.overlap_sO_sQ or self.has_bias or self.v_dequant) else
        #     cutlass.max(cute.cosize(sQ_layout), cute.cosize(sO_layout) * QO_dilation_factor)
        # )
        sQ_size = cute.cosize(sQ_layout)
        sSFQ_size = cute.cosize(sSFQ_layout) if const_expr(self.qk_blockscaled) else 0
        sSFK_size = cute.cosize(sSFK_layout) if const_expr(self.qk_blockscaled) else 0
        sSFV_size = cute.cosize(sSFV_layout) if const_expr(self.v_dequant) else 0
        sVq_size = cute.cosize(sVq_layout) if const_expr(self.v_dequant) else 0
        sV_dequant_size = cute.cosize(sV_layout) if const_expr(self.v_dequant) else 0
        use_sf_mbar = 1 if const_expr(self.qk_blockscaled and self.q_stage == 2) else 0
        use_vq_mbar = 1 if const_expr(self.v_dequant) else 0
        use_sfq_mbar = 1 if const_expr(self.qk_blockscaled) else 0

        clc_response_size = self.sched_stages * 4 if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        sO_size_bytes = cute.cosize(sO_layout) * self.o_dtype.width // 8
        sQ_size_bytes = sQ_size * self.q_dtype.width // 8
        sK_size_bytes = cute.cosize(sK_layout) * self.k_dtype.width // 8
        sVq_size_bytes = (
            sVq_size * self.v_dtype.width // 8 if const_expr(self.v_dequant) else 0
        )
        sVdeq_size_bytes = (
            sV_dequant_size * self.v_mma_dtype.width // 8
            if const_expr(self.v_dequant)
            else 0
        )
        sBias_size_bytes = (
            sBias_size * self.bias_dtype.width // 8 if const_expr(self.has_bias) else 0
        )
        sO_size_buffer = (
            sQ_size_bytes
            + sBias_size_bytes
            + sK_size_bytes
            + sVq_size_bytes
            + sVdeq_size_bytes
        )
        # print(f"{sO_size_buffer=} and {sO_size_bytes=}")
        if const_expr(self.overlap_sO_sQ and (sO_size_bytes > sO_size_buffer)):
            sO_size_buffer -= sQ_size_bytes
            sQ_size = cutlass.max(
                cute.cosize(sQ_layout), cute.cosize(sO_layout) * QO_dilation_factor
            )
            sO_size_buffer += sQ_size
            # print(f"expanding sO size buffer to {sO_size_buffer=}")

        assert (
            not self.overlap_sO_sQ or sO_size_bytes <= sO_size_buffer
        ), "error: smem for O overfills buffer"

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
            # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar_ptr: Int64
            # Scheduler m_barrier and work info for dynamic persistent
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            work_info: cute.struct.MemRange[Int32, 4]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Scale factor
            mbar_s0_s1_empty_for_sf: cute.struct.MemRange[
                Int64, self.q_stage * 2 * use_sf_mbar
            ]
            # Bias
            mbar_load_bias: cute.struct.MemRange[Int64, self.bias_stage * 2]
            # Q blockscaled pipeline (cp.async load SFQ separate from Q)
            mbar_load_SFQ: cute.struct.MemRange[Int64, self.q_stage * 2 * use_sfq_mbar]
            # V blockscaled pipelines
            mbar_load_Vq: cute.struct.MemRange[Int64, self.kv_stage * 2 * use_vq_mbar]
            mbar_v_upcast: cute.struct.MemRange[Int64, self.kv_stage * 2 * use_vq_mbar]
            # Load-Epi m_barrier for persistent and smem overlapping
            mbar_load_epi: cute.struct.MemRange[Int64, 2]
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            # CLC buffers placed here to utilize padding before sO's 1024-byte alignment.
            # This avoids adding bytes at the end when we're at the smem limit.
            # PipelineClcFetchAsync expects 2 * sched_stages mbarriers (full + empty).
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            # CLC response storage (16 bytes per stage, stored as 4 Int32s).
            clc_response: cute.struct.MemRange[Int32, clc_response_size]
            # Large TMA buffers with 1024-byte alignment
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                # In non-v_dequant, K and V share this SMEM (V reinterprets sK's pointer)
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sVq: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.v_dtype
                        if const_expr(self.v_dequant)
                        else cutlass.Float8E4M3FN
                    ),
                    sVq_size,
                ],
                self.buffer_align_bytes,
            ]
            sV_dequant: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.v_mma_dtype
                        if const_expr(self.v_dequant)
                        else cutlass.BFloat16
                    ),
                    sV_dequant_size,
                ],
                self.buffer_align_bytes,
            ]
            sBias: cute.struct.Align[
                cute.struct.MemRange[self.bias_dtype, sBias_size],
                self.buffer_align_bytes,
            ]
            # scale factor tensors
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.sfq_dtype
                        if const_expr(self.sfq_dtype is not None)
                        else cutlass.Float8E8M0FNU
                    ),  # dummy dtype for sfq
                    sSFQ_size,
                ],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.sfk_dtype
                        if const_expr(self.sfk_dtype is not None)
                        else cutlass.Float8E8M0FNU
                    ),  # dummy dtype for sfk
                    sSFK_size,
                ],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[
                    (
                        self.sfv_dtype
                        if const_expr(self.sfv_dtype is not None)
                        else cutlass.Float8E8M0FNU
                    ),  # dummy dtype for sfv
                    sSFV_size,
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        softmax_scale_true = softmax_scale
        inv_softmax_scale = 1 / softmax_scale
        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softmax_scale = None
        else:
            # NB: If a users passes in a score mod, we want to apply the score-mod in the sm_scaled qk
            # But in the original base 10. We hijack softmax_scale_log2 to just be the change of base
            # and correctly apply the softmax_scale prior to score_mod in the softmax step
            softmax_scale_log2 = LOG2_E
            softmax_scale = softmax_scale

        window_size_left = (
            Int32(window_size_left) if window_size_left is not None else None
        )
        window_size_right = (
            Int32(window_size_right) if window_size_right is not None else None
        )

        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, mPageTable
        )

        head_divmod = None
        if cutlass.const_expr(self.pack_gqa):
            head_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError(
                "Block sparsity + paged KV not supported on SM100"
            )
        if cutlass.const_expr(self.use_block_sparsity and self.is_varlen_q):
            assert const_expr(
                blocksparse_tensors.cu_total_m_blocks is not None
            ), "blocksparse_tensors.cu_total_m_blocks must be provided for varlen blocksparsity"

        smem_size_bytes = self.shared_storage.size_in_bytes()
        # print("smem size = ", smem_size_bytes)
        assert (
            smem_size_bytes <= 227 * 1024
        ), f"insufficient smem, requested {smem_size_bytes}"
        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mSFQ,
            mSFK,
            mSFV,
            mLSE,
            mRowMax,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            mBias,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tma_atom_SFQ,
            tma_atom_SFK,
            tma_atom_SFV,
            tma_atom_bias,
            softmax_scale_log2,
            softmax_scale,
            softmax_scale_true,
            inv_softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sVq_layout,
            sO_layout,
            sSFQ_layout,
            sSFK_layout,
            sSFV_layout,
            sBias_layout,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_SFQ,
            gmem_tiled_copy_O,
            bias_s2r_tiled_copy,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_pv_vq if cutlass.const_expr(self.v_dequant) else None,
            self.tiled_mma_qk_sfk if cutlass.const_expr(self.qk_blockscaled) else None,
            self.tiled_mma_pv_sfv if cutlass.const_expr(self.v_dequant) else None,
            tile_sched_params,
            num_splits,
            num_splits_dynamic_ptr,
            tile_count_semaphore,
            aux_tensors,
            fastdiv_mods,
            head_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=(
                self.cluster_shape_mnk
                if cute.size(self.cluster_shape_mnk) > 1
                else None
            ),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mSFQ: Optional[cute.Tensor],  # scale factor tensor TODO: write layout
        mSFK: Optional[cute.Tensor],  # scale factor tensor TODO: write layout
        mSFV: Optional[cute.Tensor],  # scale factor tensor for V
        mLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        mBias: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        tma_atom_SFQ: Optional[cute.CopyAtom],
        tma_atom_SFK: Optional[cute.CopyAtom],
        tma_atom_SFV: Optional[cute.CopyAtom],
        tma_atom_bias: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        softmax_scale_true: Float32,
        inv_softmax_scale: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sVq_layout: Optional[cute.ComposedLayout],
        sO_layout: cute.ComposedLayout,
        sSFQ_layout: Optional[cute.Layout],
        sSFK_layout: Optional[cute.Layout],
        sSFV_layout: Optional[cute.Layout],
        sBias_layout: Optional[cute.ComposedLayout],
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_SFQ: Optional[cute.TiledCopy],
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        bias_s2r_tiled_copy: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_vq: Optional[cute.TiledMma],
        tiled_mma_qk_sfk: Optional[cute.TiledMma],
        tiled_mma_pv_sfv: Optional[cute.TiledMma],
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,
        tile_count_semaphore: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
                tma_atom_bias,
                tma_atom_SFQ,
                tma_atom_SFK,
                tma_atom_SFV,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        load_warps = ThreadCooperativeGroup(len(self.load_warp_ids))
        load_threads = ThreadCooperativeGroup(
            len(self.load_warp_ids) * cute.arch.WARP_SIZE
        )
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        correction_warps = ThreadCooperativeGroup(len(self.correction_warp_ids))
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_warps = ThreadCooperativeGroup(len(self.epilogue_warp_ids))
        epilogue_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE
            * len(self.softmax0_warp_ids + self.correction_warp_ids)
            * self.cta_group_size
        )

        pipeline_sfq = None
        pipeline_sf_overlap = None
        pipeline_vq = None
        pipeline_v_mma = None
        pipeline_s0_s1_sequence = None
        pipeline_o_epi = None
        pipeline_bias = None
        pipeline_load_epi = None

        if const_expr(self.use_tma_Q):
            pipeline_q_tx_count = (
                self.tma_copy_bytes["Q"] + self.tma_copy_bytes["SFQ"]
                if const_expr(self.qk_blockscaled and not self.use_cpasync_to_load_sfq)
                else self.tma_copy_bytes["Q"]
            )
            pipeline_q = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_Q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=pipeline_q_tx_count,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_Q.data_ptr(),
                num_stages=self.q_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )

        if const_expr(self.use_cpasync_to_load_sfq):
            pipeline_sfq = pipeline_custom.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_SFQ.data_ptr(),
                num_stages=self.q_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )

        if const_expr(self.use_tma_KV):
            pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_kv = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=load_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        if const_expr(self.qk_blockscaled and self.q_stage == 2):
            pipeline_sf_overlap = pipeline_custom.PipelineUmmaAsync.create(
                barrier_storage=storage.mbar_s0_s1_empty_for_sf.data_ptr(),
                num_stages=self.q_stage,
                producer_group=mma_warp,
                consumer_group=softmax_threads,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        if const_expr(self.v_dequant):
            if const_expr(self.use_tma_KV):
                pipeline_vq = pipeline_custom.PipelineTmaAsync.create(
                    barrier_storage=storage.mbar_load_Vq.data_ptr(),
                    num_stages=self.kv_stage,
                    producer_group=tma_warp,
                    consumer_group=correction_warps,
                    tx_count=self.tma_copy_bytes["V"] + self.tma_copy_bytes["SFV"],
                    defer_sync=True,
                )
            else:
                pipeline_vq = pipeline_custom.PipelineAsync.create(
                    barrier_storage=storage.mbar_load_Vq.data_ptr(),
                    num_stages=self.kv_stage,
                    producer_group=load_threads,
                    consumer_group=correction_warps,
                    defer_sync=True,
                )
            pipeline_v_mma = pipeline_custom.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_v_upcast.data_ptr(),
                num_stages=self.v_mma_stage,
                producer_group=correction_threads,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        if const_expr(self.s0_s1_barrier and self.q_stage > 1):
            # This is not a typical producer-consumer pipeline. We will directly use
            # pipeline_s0_s1_sequence.sync_object_full and will not use
            # pipeline_s0_s1_sequence.sync_object_empty.
            pipeline_s0_s1_sequence = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_s0_s1_sequence.data_ptr(),
                num_stages=2,
                producer_group=softmax_threads,
                consumer_group=softmax_threads,
                defer_sync=True,
            )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_O_epi.data_ptr(),
                num_stages=self.q_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )
        if const_expr(tma_atom_bias is not None):
            pipeline_bias = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_load_bias.data_ptr(),
                num_stages=self.bias_stage,
                producer_group=tma_warp,
                consumer_group=softmax_warps,
                tx_count=self.tma_copy_bytes["bias"],
                defer_sync=True,
            )
        if const_expr(self.overlap_sO_sQ and self.is_persistent):
            pipeline_load_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_load_epi.data_ptr(),
                num_stages=1,
                producer_group=epilogue_warps,
                consumer_group=load_warps,
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sSFQ = None
        sSFK = None
        sVq = None
        sSFV = None
        if const_expr(self.v_dequant):
            # (MMA, MMA_K, MMA_D, PIPE)
            sV = storage.sV_dequant.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            sVq = storage.sVq.get_tensor(sVq_layout.outer, swizzle=sVq_layout.inner)
            sSFV = storage.sSFV.get_tensor(sSFV_layout)
        else:
            # Strip swizzle info to reuse smem
            sV = cute.make_tensor(
                cute.recast_ptr(sK.iterator, sV_layout.inner, self.v_mma_dtype),
                sV_layout.outer,
            )
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype),
                sO_layout.outer,
            )

        if const_expr(self.qk_blockscaled):
            sSFQ = storage.sSFQ.get_tensor(
                sSFQ_layout
            )  # .outer, swizzle=sSFQ_layout.inner)
            sSFK = storage.sSFK.get_tensor(
                sSFK_layout
            )  # .outer, swizzle=sSFK_layout.inner)

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.q_stage * self.m_block_size * 2)
        )
        if const_expr(self.has_bias):
            sBias = storage.sBias.get_tensor(
                sBias_layout.outer, swizzle=sBias_layout.inner
            )
        else:
            sBias = sO

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_qk_sfk = None
        thr_mma_pv_sfv = None
        if const_expr(self.qk_blockscaled):
            thr_mma_qk_sfk = tiled_mma_qk_sfk.get_slice(mma_tile_coord_v)
        if const_expr(self.v_dequant):
            thr_mma_pv_sfv = tiled_mma_pv_sfv.get_slice(mma_tile_coord_v)
            thr_mma_pv_vq = tiled_mma_pv_vq.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_mma_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_mma_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
            num_splits=num_splits,
            num_splits_dynamic_ptr=num_splits_dynamic_ptr,
            num_n_blocks_per_split=self.num_n_blocks_per_split,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=(
                mK.shape[0]
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks
                if blocksparse_tensors is not None
                else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()

            clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_clc_consumer_warps_per_cta = self.threads_per_cta // cute.arch.WARP_SIZE
            # NB on CTA0 warp15 == scheduler on CTA1 == empty but still both consume
            num_clc_consumer_warps = (
                num_clc_consumer_warps_per_cta * self.cta_group_size
            )
            clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread,
                cute.arch.WARP_SIZE * num_clc_consumer_warps,
            )

            block_idx = cute.arch.block_idx()
            clc = ClcState.create(
                hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                    self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                    block_idx,
                    cute.arch.grid_dim(),
                    clc_response_ptr,
                ),
                pipeline=cutlass_pipeline.PipelineClcFetchAsync.create(
                    barrier_storage=clc_mbar_ptr,
                    num_stages=self.sched_stages,
                    producer_group=clc_pipeline_producer_group,
                    consumer_group=clc_pipeline_consumer_group,
                    tx_count=16,
                    cta_layout_vmnk=cta_layout_vmnk,
                ),
                consumer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Consumer, self.sched_stages
                ),
                producer_state=cutlass_pipeline.make_pipeline_state(
                    cutlass_pipeline.PipelineUserType.Producer, self.sched_stages
                ),
            )
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
        elif const_expr(self.dynamic_persistent):
            assert tile_count_semaphore is not None
            work_info = storage.work_info.get_tensor((4,))
            sched_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            )
            # empty warps don't consume for non-clc dynamic persistent scheduler
            sched_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_non_empty_warps
            )
            sched_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.sched_pipeline_array_ptr.data_ptr(),
                num_stages=1,
                producer_group=sched_pipeline_producer_group,
                consumer_group=sched_pipeline_consumer_group,
            )
            tile_scheduler = self.tile_scheduler_cls.create(
                tile_sched_params, work_info, sched_pipeline
            )
        else:
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        # assert isinstance(tile_scheduler, TileSchedulerProtocol), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / CLC SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(self.use_clc_scheduler):
            if warp_idx == self.clc_scheduler_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                if is_leader_cta:
                    self.clc_scheduler_warp(tile_scheduler)
                else:
                    self.empty_warp(tile_scheduler)
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if (
                    warp_idx == self.empty_warp_ids[i]
                    and warp_idx != self.clc_scheduler_warp_id
                ):
                    cute.arch.setmaxregister_decrease(self.num_regs_other)
                    self.empty_warp(tile_scheduler)
        else:
            for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
                if warp_idx == self.empty_warp_ids[i]:
                    cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            if const_expr(self.num_regs_load < 128):
                cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_qk,
                thr_mma_pv if const_expr(not self.v_dequant) else thr_mma_pv_vq,
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV if const_expr(not self.v_dequant) else sVq,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                gmem_tiled_copy_Q,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
                mSFQ=mSFQ,
                mSFK=mSFK,
                mSFV=mSFV,
                sSFQ=sSFQ,
                sSFK=sSFK,
                sSFV=sSFV,
                mBias=mBias,
                sBias=sBias,
                tma_atom_SFQ=tma_atom_SFQ,
                tma_atom_SFK=tma_atom_SFK,
                tma_atom_SFV=tma_atom_SFV,
                thr_mma_qk_sfk=thr_mma_qk_sfk,
                thr_mma_pv_sfv=thr_mma_pv_sfv,
                tma_atom_bias=tma_atom_bias,
                pipeline_bias=pipeline_bias,
                pipeline_load_epi=pipeline_load_epi,
                pipeline_sfq=pipeline_sfq,
                pipeline_vq=pipeline_vq,
                gmem_tiled_copy_SFQ=gmem_tiled_copy_SFQ,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tStS,
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
                tmem_ptr=tmem_ptr,
                sSFQ=sSFQ,
                sSFK=sSFK,
                sSFQ_layout=sSFQ_layout,
                sSFK_layout=sSFK_layout,
                pipeline_sfq=pipeline_sfq,
                pipeline_sf_overlap=pipeline_sf_overlap,
                pipeline_v_mma=pipeline_v_mma,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if (
                warp_idx >= self.epilogue_warp_ids[0]
                and warp_idx <= self.epilogue_warp_ids[-1]
            ):
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    mma_tile_coord_v,
                    tile_scheduler=tile_scheduler,
                    pipeline_load_epi=pipeline_load_epi,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]
        ) or (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1]):
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                softmax_scale_true=softmax_scale_true,
                inv_softmax_scale=inv_softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                learnable_sink=learnable_sink,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                blocksparse_tensors=blocksparse_tensors,
                tile_scheduler=tile_scheduler,
                sBias=sBias,
                bias_s2r_tiled_copy=bias_s2r_tiled_copy,
                pipeline_bias=pipeline_bias,
                mRowMax=mRowMax,
                pipeline_sf_overlap=pipeline_sf_overlap,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(
                    0
                    if const_expr(self.q_stage == 1)
                    or warp_idx < self.softmax1_warp_ids[0]
                    else 1
                )
                softmax_loop(stage=stage, tStS=tStS)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    softmax_loop(stage=0, tStS=tStS)
                if (
                    warp_idx < self.correction_warp_ids[0]
                    and warp_idx >= self.softmax1_warp_ids[0]
                ):
                    softmax_loop(stage=1, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.correction_warp_ids[0]
            and warp_idx <= self.correction_warp_ids[-1]
        ):
            if const_expr(self.num_regs_correction < 128):
                cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                learnable_sink,
                gmem_tiled_copy_O,
                tma_atom_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
                pipeline_load_epi=pipeline_load_epi,
                pipeline_vq=pipeline_vq,
                pipeline_v_mma=pipeline_v_mma,
                sVq=sVq,
                sSFV=sSFV,
                sV=sV,
            )
            tmem_alloc_barrier.arrive()

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler: TileSchedulerProtocol,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        sSFQ: Optional[cute.Tensor] = None,
        sSFK: Optional[cute.Tensor] = None,
        sSFV: Optional[cute.Tensor] = None,
        mBias: Optional[cute.Tensor] = None,
        sBias: Optional[cute.Tensor] = None,
        tma_atom_SFQ: Optional[cute.CopyAtom] = None,
        tma_atom_SFK: Optional[cute.CopyAtom] = None,
        tma_atom_SFV: Optional[cute.CopyAtom] = None,
        thr_mma_qk_sfk: Optional[cute.ThrMma] = None,
        thr_mma_pv_sfv: Optional[cute.ThrMma] = None,
        tma_atom_bias: Optional[cute.CopyAtom] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
        pipeline_load_epi: Optional[pipeline.PipelineAsync] = None,
        pipeline_sfq: Optional[pipeline.PipelineAsync] = None,
        pipeline_vq: Optional[pipeline.PipelineAsync] = None,
        gmem_tiled_copy_SFQ: Optional[cute.TiledCopy] = None,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        issue_kv_for_this_warp = (
            const_expr(not self.use_tma_KV or len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        )
        # reused for bias
        issue_q_for_this_warp = (
            const_expr(not self.use_tma_Q or len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        )
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        if const_expr(self.v_dequant):
            vq_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kv_stage
            )
        bias0_producer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.bias_stage // self.q_stage
        )
        bias1_producer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.bias_stage // self.q_stage
        )
        load_epi_consumer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_scheduler_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, 1
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            head_idx_kv = (
                head_idx // self.qhead_per_kvhead
                if const_expr(not self.pack_gqa)
                else head_idx
            )
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)

            load_SFQ_fn = None
            if const_expr(self.qk_blockscaled):
                tiler_gQ_sf = (
                    (self.mma_tiler_qk[0] * self.q_stage),
                    self.head_dim_padded // 32,
                )
                mSFQ_cur = seqlen.offset_batch_Q(mSFQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                gSFQ = cute.local_tile(mSFQ_cur, tiler_gQ, (m_block, 0))
                gSFQ = layout_utils.select(
                    cute.flat_divide(gSFQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
                )
                # cpasync tiler expects compact layouts
                gSFQ_sf = cute.local_tile(mSFQ_cur, tiler_gQ_sf, (m_block, 0))
                gSFQ_sf = layout_utils.select(
                    cute.flat_divide(gSFQ_sf, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
                )
                if const_expr(self.use_cpasync_to_load_sfq):
                    cpasync_load_SFQ = partial(
                        self.cpasync_load_SFQ,
                        gSFQ_sf,
                        sSFQ,
                        gmem_tiled_copy_SFQ,
                        pipeline_sfq,
                        tidx=tidx,
                        phase=q_producer_phase,
                        m_block=m_block,
                        seqlen_q=seqlen.seqlen_q
                        * (self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1),
                    )
                else:
                    tSgSFQ = thr_mma_qk.partition_A(gSFQ)
                    load_SFQ_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_SFQ,
                        0,
                        cute.make_layout(1),
                        tSgSFQ,
                        sSFQ,
                        filter_zeros=True,
                    )

            if const_expr(self.use_tma_Q):
                gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (128 * 2, 128)
                gQ = layout_utils.select(
                    cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
                )  # (128, 128, 2)
                tSgQ = thr_mma_qk.partition_A(gQ)
                load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
                )
                load_Q = partial(
                    self.load_Q,
                    load_Q_fn,
                    pipeline_q=pipeline_q,
                    phase=q_producer_phase,
                    load_SFQ_fn=load_SFQ_fn,
                )
            else:
                assert gmem_tiled_copy_Q is not None
                load_Q = partial(
                    self.load_Q_non_tma,
                    mQ_cur,
                    sQ,
                    gmem_tiled_copy_Q,
                    pipeline_q,
                    tidx,
                    seqlen.seqlen_q,
                    m_block,
                    phase=q_producer_phase,
                )

            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [
                        t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)
                    ]
                    if const_expr(self.qk_blockscaled):
                        mSFK_cur = mSFK[None, None, head_idx_kv, batch_idx]
                    if const_expr(self.v_dequant):
                        mSFV_cur = mSFV[None, None, head_idx_kv, batch_idx]
                else:
                    mK_cur = cute.domain_offset(
                        (seqlen.offset_k, 0), mK[None, None, head_idx_kv]
                    )
                    mV_cur = cute.domain_offset(
                        (0, seqlen.offset_k), mV[None, None, head_idx_kv]
                    )
                    if const_expr(self.qk_blockscaled):
                        mSFK_cur = cute.domain_offset(
                            (seqlen.offset_k, 0), mSFK[None, None, head_idx_kv]
                        )
                    if const_expr(self.v_dequant):
                        mSFV_cur = cute.domain_offset(
                            (0, seqlen.offset_k), mSFV[None, None, head_idx_kv]
                        )
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
                )
                if const_expr(self.qk_blockscaled):
                    gSFK = cute.local_tile(
                        mSFK_cur,
                        cute.select(self.mma_tiler_qk_sfk, mode=[1, 2]),
                        (None, 0),
                    )
                if const_expr(self.v_dequant):
                    gSFV = cute.local_tile(
                        mSFV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)
                    )
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                # Ditto with scale factor tensors
                if const_expr(self.qk_blockscaled):
                    mSFK_cur = mSFK[None, None, head_idx_kv, None]
                if const_expr(self.v_dequant):
                    mSFV_cur = mSFV[None, None, head_idx_kv, None]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                gV = cute.local_tile(
                    mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                )
                if const_expr(self.qk_blockscaled):
                    gSFK = cute.local_tile(
                        mSFK_cur,
                        cute.select(self.mma_tiler_qk_sfk, mode=[1, 2]),
                        (None, 0, None),
                    )
                if const_expr(self.v_dequant):
                    gSFV = cute.local_tile(
                        mSFV_cur,
                        cute.select(self.mma_tiler_pv, mode=[1, 2]),
                        (0, None, None),
                    )
            tSgK = thr_mma_qk.partition_B(gK)
            tOgV = thr_mma_pv.partition_B(gV)

            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV, 0, 3),
                )
                if const_expr(self.qk_blockscaled):
                    tSgSFK = thr_mma_qk_sfk.partition_B(gSFK)
                    tKsSFK, tKgSFK = cpasync.tma_partition(
                        tma_atom_SFK,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sSFK, 0, 3),
                        cute.group_modes(tSgSFK, 0, 3),
                    )
                    tKsSFK = cute.filter_zeros(tKsSFK)
                    tKgSFK = cute.filter_zeros(tKgSFK)
                if const_expr(self.v_dequant):
                    tOgSFV = thr_mma_pv_sfv.partition_B(gSFV)
                    tVsSFV, tVgSFV = cpasync.tma_partition(
                        tma_atom_SFV,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sSFV, 0, 3),
                        cute.group_modes(tOgSFV, 0, 3),
                    )
                    tVsSFV = cute.filter_zeros(tVsSFV)
                    tVgSFV = cute.filter_zeros(tVgSFV)
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmodDivisor(page_size),
                    batch_idx,
                    head_idx_kv,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                    mSFK,
                    mSFV,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None
                tKsSFK, tKgSFK = None, None
                tVsSFV, tVgSFV = None, None

            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
                tma_atom_sf=tma_atom_SFK if const_expr(self.qk_blockscaled) else None,
                tXgSFX=tKgSFK if const_expr(self.qk_blockscaled) else None,
                tXsSFX=tKsSFK if const_expr(self.qk_blockscaled) else None,
                sSFX=sSFK if const_expr(self.qk_blockscaled) else None,
                stage_dilation=self.kv_size_ratio,
            )
            load_V = partial(
                self.load_KV,
                tma_atom_V,
                tVgV,
                tVsV,
                paged_kv_manager,
                sV,
                pipeline_kv=pipeline_vq if const_expr(self.v_dequant) else pipeline_kv,
                K_or_V="V",
                tma_atom_sf=tma_atom_SFV if const_expr(self.v_dequant) else None,
                tXgSFX=tVgSFV if const_expr(self.v_dequant) else None,
                tXsSFX=tVsSFV if const_expr(self.v_dequant) else None,
                sSFX=sSFV if const_expr(self.v_dequant) else None,
            )
            if const_expr(tma_atom_bias is not None):
                # (seqlen, rel_extent_padded)
                mBias_cur = seqlen.offset_batch_Q(mBias, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                # (TILE_M, TILE_N, rest_m, rest_n)
                gBias = cute.local_tile(
                    mBias_cur, (self.bias_block_size, self.n_block_size), (None, None)
                )
                # (TMA, STAGE) and (TMA, rest_m, rest_n)
                tBsBias, tBgBias = cpasync.tma_partition(
                    tma_atom_bias,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sBias, 0, 2),
                    cute.group_modes(gBias, 0, 2),
                )
                assert cute.rank(tBsBias) == 2
                assert cute.rank(tBgBias) == 3

                load_bias = partial(
                    self.load_bias,
                    tma_atom_bias,
                    tBgBias,
                    tBsBias,
                    pipeline_bias=pipeline_bias,
                )

            if const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    m_block,
                    split_idx,
                    batch_idx,
                )
                _, n_block_max_abs = block_info.get_n_block_min_max(
                    seqlen,
                    m_block,
                    split_idx,
                    batch_idx,
                    absolute=True,
                )
                _, n_block_max_abs0 = block_info.get_n_block_min_max(
                    seqlen,
                    self.q_stage * m_block,
                    split_idx,
                    batch_idx,
                    half_tile_m=self.q_stage > 1,
                    absolute=True,
                )
                # if tidx == 0:
                #     cute.printf("n_block_max_abs = {}, n_block_max_abs0 = {}", n_block_max_abs, n_block_max_abs0)
                # get number of n_blocks from this split's last n_block to absolute max n_block
                # note: bias_max_idx0 >= bias_max_idx1 under assumptions
                bias_idx_offset1 = n_block_max_abs - n_block_max
                bias_max_idx1 = self.bias_n_max - 1 - bias_idx_offset1
                bias_idx_offset0 = n_block_max_abs0 - n_block_max
                bias_max_idx0 = self.bias_n_max - 1 - max(bias_idx_offset0, 0)
                # if tidx == 0:
                #     cute.printf(
                #         "bias_idx_offset1 = {}, bias_max_idx1 = {}, bias_idx_offset0 = {}, bias_max_idx0 = {}",
                #         bias_idx_offset1, bias_max_idx1, bias_idx_offset0, bias_max_idx0
                #     )

                # Example of possible behavior:
                # ... O X X X X | X X O
                # ... O O X X X | X X X
                dummy_first_bias_load = bias_idx_offset0 < 0 and const_expr(
                    self.q_stage == 2
                )
                staggered_bias_loads = (
                    dummy_first_bias_load or (bias_max_idx0 > bias_max_idx1)
                ) and const_expr(self.q_stage == 2)
                if self.process_work_tile(seqlen, n_block_min, n_block_max):
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    page_idx = (
                        mPageTable[
                            batch_idx, n_block_first // self.paged_kv_blocks_per_page
                        ]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)
                    # load V0 first if dequant V
                    if const_expr(self.v_dequant):
                        if issue_kv_for_this_warp:
                            load_V(
                                block=n_block_max - 1,
                                producer_state=vq_producer_state,
                                page_idx=page_idx,
                            )
                            vq_producer_state.advance()
                    if issue_kv_for_this_warp:
                        load_K(
                            block=n_block_max - 1,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )  # K0
                        kv_producer_state.advance()
                    if issue_q_for_this_warp:
                        load_Q(block=0, stage=0)
                    if const_expr(self.use_cpasync_to_load_sfq):
                        cpasync_load_SFQ(block=0, stage=0)
                    if const_expr(tma_atom_bias is not None):
                        if issue_q_for_this_warp and (
                            const_expr(not self.is_split_kv) or bias_max_idx0 >= 0
                        ):
                            bias0_producer_state = load_bias(
                                m_block=self.q_stage * m_block + 0,
                                n_block=bias_max_idx0,
                                bias_producer_state=bias0_producer_state,
                                q_stage=0,
                            )  # Bias0
                    if const_expr(self.q_stage == 2):
                        if const_expr(self.use_cpasync_to_load_sfq):
                            cpasync_load_SFQ(block=1, stage=1)
                        if issue_q_for_this_warp:
                            load_Q(block=1, stage=1)
                            if const_expr(tma_atom_bias is not None):
                                if (
                                    const_expr(not self.is_split_kv)
                                    or bias_max_idx1 >= 0
                                ):
                                    bias1_producer_state = load_bias(
                                        m_block=self.q_stage * m_block + 1,
                                        n_block=bias_max_idx1,
                                        bias_producer_state=bias1_producer_state,
                                        q_stage=1,
                                    )  # Bias1
                    q_producer_phase ^= 1
                    if const_expr(not self.v_dequant):
                        if issue_kv_for_this_warp:
                            load_V(
                                block=n_block_max - 1,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )  # V0
                            kv_producer_state.advance()
                    # prologue loads involving bias
                    if const_expr(tma_atom_bias is not None):
                        prologue_loads = min(
                            bias_max_idx1, n_block_max - 1 - n_block_min
                        )
                        tail_bias_load = (
                            staggered_bias_loads
                            and prologue_loads < n_block_max - 1 - n_block_min
                            and prologue_loads >= 0
                            # and bias_max_idx0 - 1 - prologue_loads + dummy_first_bias_load >= 0
                        )
                        prologue_loads = max(prologue_loads, 0)
                        # if tidx == 0:
                        #     cute.printf("prologue_loads = {} for n_block_max, n_block_min = {}, {} and m_block = {}, split = {}, bias_max_idx0 = {}, bias_max_idx1 = {}, dummy_first_bias_load = {}, tail = {}",
                        #                 prologue_loads, n_block_max, n_block_min, m_block, split_idx, bias_max_idx0, bias_max_idx1, dummy_first_bias_load, tail_bias_load)
                        for i in cutlass.range(prologue_loads, unroll=1):
                            n_block = n_block_max - 2 - i
                            page_idx = (
                                mPageTable[
                                    batch_idx, n_block // self.paged_kv_blocks_per_page
                                ]
                                if const_expr(
                                    mPageTable is not None and self.use_tma_KV
                                )
                                else None
                            )
                            if const_expr(not self.use_tma_KV):
                                paged_kv_manager.load_page_table(n_block)
                            if issue_kv_for_this_warp:
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )  # Ki
                                kv_producer_state.advance()
                            if const_expr(tma_atom_bias is not None):
                                if issue_q_for_this_warp:
                                    bias0_producer_state = load_bias(
                                        m_block=self.q_stage * m_block + 0,
                                        n_block=bias_max_idx0
                                        - 1
                                        - i
                                        + dummy_first_bias_load,
                                        bias_producer_state=bias0_producer_state,
                                        q_stage=0,
                                    )  # Bias0
                                    if const_expr(self.q_stage == 2):
                                        bias1_producer_state = load_bias(
                                            m_block=self.q_stage * m_block + 1,
                                            n_block=bias_max_idx1 - 1 - i,
                                            bias_producer_state=bias1_producer_state,
                                            q_stage=1,
                                        )  # Bias1
                            if issue_kv_for_this_warp:
                                if const_expr(self.v_dequant):
                                    load_V(
                                        block=n_block,
                                        producer_state=vq_producer_state,
                                        page_idx=page_idx,
                                    )  # Vi
                                    vq_producer_state.advance()
                                else:
                                    load_V(
                                        block=n_block,
                                        producer_state=kv_producer_state,
                                        page_idx=page_idx,
                                    )  # Vi
                                    kv_producer_state.advance()

                        if issue_q_for_this_warp and tail_bias_load:
                            bias0_producer_state = load_bias(
                                m_block=self.q_stage * m_block + 0,
                                n_block=bias_max_idx0
                                - 1
                                - prologue_loads
                                + dummy_first_bias_load,
                                bias_producer_state=bias0_producer_state,
                                q_stage=0,
                            )  # Bias0
                    else:
                        prologue_loads = 0

                    for i in cutlass.range(
                        prologue_loads, n_block_max - 1 - n_block_min, unroll=1
                    ):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            mPageTable[
                                batch_idx, n_block // self.paged_kv_blocks_per_page
                            ]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        if issue_kv_for_this_warp:
                            load_K(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )  # Ki
                            kv_producer_state.advance()
                            if const_expr(self.v_dequant):
                                load_V(
                                    block=n_block,
                                    producer_state=vq_producer_state,
                                    page_idx=page_idx,
                                )  # Vi
                                vq_producer_state.advance()
                            else:
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )  # Vi
                                kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )

            if const_expr(self.dynamic_persistent):
                if warp_idx == self.dynamic_scheduler_warp_id:
                    tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()

            if const_expr(self.overlap_sO_sQ and self.is_persistent):
                pipeline_load_epi.consumer_wait(load_epi_consumer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.consumer_release(load_epi_consumer_state)
                load_epi_consumer_state.advance()
            # End of persistent scheduler loop

        if issue_kv_for_this_warp:
            if const_expr(self.v_dequant):
                pipeline_vq.producer_tail(vq_producer_state)
            pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail
        if issue_q_for_this_warp:
            pipeline_q.producer_acquire_w_index_phase(
                self.q_stage - 1, q_producer_phase
            )
        if const_expr(self.use_cpasync_to_load_sfq):
            pipeline_sfq.producer_acquire_w_index_phase(
                self.q_stage - 1, q_producer_phase
            )

        if const_expr(self.dynamic_persistent):
            if warp_idx == self.dynamic_scheduler_warp_id:
                tile_scheduler.producer_tail()

    @cute.jit
    def load_bias(
        self,
        tma_atom_bias: cute.CopyAtom,
        tBgBias: cute.Tensor,
        tBsBias: cute.Tensor,
        pipeline_bias: pipeline.PipelineAsync,
        m_block: Int32,
        n_block: Int32,
        bias_producer_state: pipeline.PipelineState,
        q_stage: Int32,
    ):
        stage = bias_producer_state.index + q_stage
        phase = bias_producer_state.phase
        mbar_ptr = pipeline_bias.sync_object_full.get_barrier(stage)
        pipeline_bias.producer_acquire_w_index_phase(stage, phase)
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        cute.copy(
            tma_atom_bias,
            tBgBias[None, m_block, n_block],
            tBsBias[None, stage],
            tma_bar_ptr=mbar_ptr,
        )
        bias_producer_state.advance()
        return bias_producer_state

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
        tile_scheduler=None,
        tmem_ptr=None,
        # Blockscaled SF args
        sSFQ: Optional[cute.Tensor] = None,
        sSFK: Optional[cute.Tensor] = None,
        sSFQ_layout=None,
        sSFK_layout=None,
        pipeline_sfq: Optional[pipeline.PipelineAsync] = None,
        pipeline_sf_overlap: Optional[pipeline.PipelineAsync] = None,
        pipeline_v_mma: Optional[pipeline.PipelineAsync] = None,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        qk_mma_idesc, pv_mma_idesc = sm100_desc.mma_op_to_idesc(
            qk_mma_op
        ), sm100_desc.mma_op_to_idesc(pv_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        v_smem_base = sm100_desc.smem_desc_base_from_tensor(sV, sm100_desc.Major.MN)
        q_smem_start = [
            sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator)
            for stage in range(self.q_stage)
        ]

        sm100_utils.declare_ptx_smem_desc(
            q_smem_start[self.q_stage - 1],
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="fa_fwd_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        if const_expr(self.qk_blockscaled):
            sf_copies = [
                self.make_sf_tmem_copies(
                    tmem_ptr, stage, tiled_mma_qk, sSFQ, sSFK, sSFQ_layout, sSFK_layout
                )
                for stage in range(self.q_stage)
            ]

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_stage == 1):
            sQ_stage_stride = 0
        if const_expr(not self.qk_blockscaled):
            gemm_Si = [
                partial(
                    # sm100_utils.gemm_ptx_precomputed,
                    # self.tmem_s_offset[stage],
                    # smem_desc_start_a=q_smem_start[stage],
                    # idesc=qk_mma_idesc,
                    # smem_desc_base_a=q_smem_base,
                    # smem_desc_base_b=k_smem_base,
                    # tCrA_layout=tSrQ[None, None, None, 0].layout,
                    sm100_utils.gemm_ptx_precomputed_varname,
                    self.tmem_s_offset[stage],
                    # idesc=qk_mma_idesc,
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK[None, None, None, 0].layout,
                    smem_var_name_prefix=f"fa_fwd_q_smem_desc",
                    idesc_var_name=f"fa_fwd_qk_mma_idesc",
                    smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.q_stage)
            ]
            # gemm_Si = [
            #     partial(
            #         sm100_utils.gemm,
            #         tiled_mma_qk,
            #         tStS[None, None, None, stage],
            #         tCrA=tSrQ[None, None, None, stage],
            #         zero_init=True,
            #     )
            #     for stage in range(self.q_stage)
            # ]
        else:
            gemm_Si = [
                partial(
                    sm100_utils.gemm_blockscaled,
                    tiled_mma_qk,
                    tStS[None, None, None, stage],
                    tSrQs[stage],
                    tCtSFA=sf_copies[stage].tCtSFQ,
                    tCtSFB=sf_copies[stage].tCtSFK,
                    zero_init=True,
                )
                for stage in range(self.q_stage)
            ]

        gemm_Pi = [
            partial(
                # sm100_utils.gemm_ptx_precomputed,
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=None,
                split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
                # smem_desc_start_a=tOrP[None, None, None, stage].iterator.toint(),
                # smem_desc_start_a=self.tmem_p_offset[stage],
                # idesc=pv_mma_idesc,
                # smem_desc_base_a=None,
                # smem_desc_base_b=v_smem_base,
                # tCrA_layout=tOrP[None, None, None, 0].layout,
                # tCrB_layout=tOrV[None, None, None, 0].layout
                tA_addr=(
                    self.tmem_p_offset[stage]
                    if const_expr(self.qk_blockscaled or self.v_dequant)
                    else None
                ),
                cta_group=self.cta_group_size,
            )
            for stage in range(self.q_stage)
        ]
        # gemm_Pi = [
        #     partial(
        #         sm100_utils.gemm, tOtO[None, None, None, stage], tCrA=tOrP[None, None, None, stage]
        #     )
        #     for stage in range(self.q_stage)
        # ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        if const_expr(self.v_dequant):
            v_dequant_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.v_mma_stage
            )
        P_full_O_rescaled_phase = Int32(0)
        if const_expr(self.qk_blockscaled):
            sf_overlap_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen,
                    m_block,
                    split_idx,
                    batch_idx,
                )
                block_iter_count = n_block_max - n_block_min
                process_tile = self.process_work_tile(seqlen, n_block_min, n_block_max)

            if process_tile and is_leader_cta:
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                    if const_expr(self.use_cpasync_to_load_sfq):
                        pipeline_sfq.consumer_wait_w_index_phase(
                            stage, mma_q_consumer_phase
                        )
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    # S2T copy SFQ
                    if const_expr(self.qk_blockscaled):
                        if const_expr(self.q_stage == 2):
                            pipeline_sf_overlap.producer_acquire_w_index_phase(
                                stage, sf_overlap_phase ^ stage
                            )
                        sfq_s2t_stage_coord = (None, None, None, None, stage)
                        cute.copy(
                            sf_copies[stage].tiled_copy_sfq,
                            sf_copies[stage].tCsSFQ_s2t[sfq_s2t_stage_coord],
                            sf_copies[stage].tCtSFQ_s2t,
                        )
                    # S2T copy SFK (each stage needs its own SFK in TMEM)
                    if const_expr(self.qk_blockscaled):
                        sfk_s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            mma_kv_consumer_state.index,
                        )
                        cute.copy(
                            sf_copies[stage].tiled_copy_sfk,
                            sf_copies[stage].tCsSFK_s2t[sfk_s2t_stage_coord],
                            sf_copies[stage].tCtSFK_s2t,
                        )
                    Ki_index, Ki_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    Ki_index *= self.kv_size_ratio
                    tSrKi = tSrK[None, None, None, Ki_index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, Ki_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                    if const_expr(self.qk_blockscaled):
                        gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    else:
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                sK_cur.iterator
                            )
                        )
                    # gemm_Si[stage](tCrB=tSrKi)
                    # 4. release S0 / S1
                    pipeline_s_p_o.producer_commit_w_index(stage)
                if const_expr(self.qk_blockscaled and self.q_stage == 2):
                    sf_overlap_phase ^= 1
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                O_should_accumulate = False
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    if const_expr(self.v_dequant):
                        pipeline_v_mma.consumer_wait(v_dequant_consumer_state)
                        Vi_index, Vi_phase = (
                            v_dequant_consumer_state.index,
                            v_dequant_consumer_state.phase,
                        )
                    else:
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index, Vi_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                    tOrVi = tOrV[None, None, None, Vi_index]
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # 2. acquire corrected O0/O1_partial and P0 / P1
                        # For the first iteration in this work tile, waiting for O0/O1_partial
                        # means that the correction warps has finished reading tO during
                        # the last iteration of the previous work tile.
                        pipeline_s_p_o.producer_acquire_w_index_phase(
                            stage, P_full_O_rescaled_phase
                        )
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                        # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                            zero_init=not O_should_accumulate,
                            mbar_ptr=(
                                pipeline_p_lastsplit.sync_object_full.get_barrier(stage)
                                if self.split_P_arrive > 0
                                else None
                            ),
                            mbar_phase=P_full_O_rescaled_phase,
                        )
                        # Don't need to signal O_full to the correction warps since the
                        # correction warps wait for the softmax warps anyway. By the time the softmax
                        # warps finished, S_i for the next iteration must have been done, so O_i-1
                        # must have been done as well.
                        # pipeline_o_acc.producer_commit_w_index(stage)
                        # 4. release V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            if const_expr(self.v_dequant):
                                pipeline_v_mma.consumer_release(
                                    v_dequant_consumer_state
                                )
                                v_dequant_consumer_state.advance()
                            else:
                                pipeline_kv.consumer_release(mma_kv_release_state)
                                mma_kv_release_state.advance()
                        # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        # 1. wait for Ki (advance so consumer index points to K, not V)
                        if const_expr(stage == 0):
                            if const_expr(not self.v_dequant):
                                mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        Ki_index *= self.kv_size_ratio

                        # 1a. Scale factor S2T copies.
                        if const_expr(self.qk_blockscaled and self.q_stage == 2):
                            pipeline_sf_overlap.producer_acquire_w_index_phase(
                                stage, sf_overlap_phase ^ stage
                            )
                        if const_expr(self.qk_blockscaled):
                            sfk_s2t_stage_coord = (
                                None,
                                None,
                                None,
                                None,
                                mma_kv_consumer_state.index,
                            )
                            cute.copy(
                                sf_copies[stage].tiled_copy_sfk,
                                sf_copies[stage].tCsSFK_s2t[sfk_s2t_stage_coord],
                                sf_copies[stage].tCtSFK_s2t,
                            )
                        # S2T copy SFQ (re-copy since S GEMM overwrote the SF TMEM region)
                        if const_expr(self.qk_blockscaled and self.q_stage == 2):
                            sfq_s2t_stage_coord = (None, None, None, None, stage)
                            cute.copy(
                                sf_copies[stage].tiled_copy_sfq,
                                sf_copies[stage].tCsSFQ_s2t[sfq_s2t_stage_coord],
                                sf_copies[stage].tCtSFQ_s2t,
                            )
                        # 2. gemm
                        # Don't need to wait for the softmax warp to have finished reading the previous
                        # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                        # has been read and Pi has been written.
                        # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrK[None, None, None, Ki_index], zero_init=True)
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        if const_expr(self.qk_blockscaled):
                            gemm_Si[stage](
                                tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur
                            )
                        else:
                            gemm_Si[stage](
                                smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                    sK_cur.iterator
                                )
                            )
                        # gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index])
                        # 3. release S0 / S1
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        # End of GEMM_QK0i (Q0 * Ki -> S0)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    P_full_O_rescaled_phase ^= 1
                    if const_expr(self.qk_blockscaled and self.q_stage == 2):
                        sf_overlap_phase ^= 1
                    O_should_accumulate = True
                # End of seqlen_kv loop

                # release Q0 & Q1
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)
                    if const_expr(self.use_cpasync_to_load_sfq):
                        pipeline_sfq.consumer_release_w_index(stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                if const_expr(self.v_dequant):
                    pipeline_v_mma.consumer_wait(v_dequant_consumer_state)
                    Vi_index, Vi_phase = (
                        v_dequant_consumer_state.index,
                        v_dequant_consumer_state.phase,
                    )
                else:
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(self.q_stage):
                    # 2. acquire corrected Oi_partial and Pi
                    pipeline_s_p_o.producer_acquire_w_index_phase(
                        stage, P_full_O_rescaled_phase
                    )
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    gemm_Pi[stage](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        # smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sV_cur.iterator),
                        zero_init=not O_should_accumulate,
                        mbar_ptr=(
                            pipeline_p_lastsplit.sync_object_full.get_barrier(stage)
                            if self.split_P_arrive > 0
                            else None
                        ),
                        mbar_phase=P_full_O_rescaled_phase,
                    )
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warps, the softmax warp has just finished
                    # computing the row sum of the current tile. It does not guarantee that the 1st
                    # tile of the next work tile has been computed yet.
                    if const_expr(not self.overlap_sO_sQ):
                        pipeline_o_acc.producer_commit_w_index(stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_O_rescaled_phase ^= 1
                # 5. release Vi_end
                if const_expr(self.v_dequant):
                    pipeline_v_mma.consumer_release(v_dequant_consumer_state)
                    v_dequant_consumer_state.advance()
                else:
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

                # only signal completion after releasing all operands
                if const_expr(self.overlap_sO_sQ):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_o_acc.producer_commit_w_index(stage)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # pipeline_s_p_o.producer_acquire_w_index_phase(self.q_stage - 1, P_full_O_rescaled_phase)
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        softmax_scale_true: Float32,
        inv_softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
        sBias: Optional[cute.Tensor] = None,
        bias_s2r_tiled_copy: Optional[cute.TiledCopy] = None,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
        mRowMax: Optional[cute.Tensor] = None,
        pipeline_sf_overlap: Optional[pipeline.PipelineAsync] = None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] * self.v_mma_dtype.width // Float32.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_scale_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
        )
        thr_tmem_store_scale = tcgen05.make_tmem_copy(
            tmem_store_scale_atom, tStScale
        ).get_slice(tidx)
        tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)
        bias_si_consumer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.bias_stage // self.q_stage
        )

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if const_expr(bias_s2r_tiled_copy is not None):
            bias_s2r_thr_copy = bias_s2r_tiled_copy.get_slice(tidx)
            tS2RsBias = bias_s2r_thr_copy.partition_S(sBias)
            # print("SMEM: tS2RsBias = ", tS2RsBias)
        else:
            bias_s2r_thr_copy = None
            tS2RsBias = None

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                m_block,
                split_idx,
                batch_idx,
            )
            _, n_block_max_abs_local = block_info.get_n_block_min_max(
                seqlen,
                self.q_stage * m_block + stage,
                split_idx,
                batch_idx,
                half_tile_m=self.q_stage > 1,
                absolute=True,
            )
            bias_idx_offset = n_block_max_abs_local - n_block_max
            num_bias_loads = min(
                self.bias_n_max - bias_idx_offset, n_block_max - n_block_min
            )
            # if tidx == 0:
            #     cute.printf(
            #         "n_block_max = {}, n_block_max_abs_local = {}, bias_idx_offset = {}, num_bias_loads = {}",
            #         n_block_max, n_block_max_abs_local, bias_idx_offset, num_bias_loads
            #     )

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=head_idx,
            )
            if const_expr(MASK_TAKES_AUX_TENSORS):
                shared_mask_kwargs["aux_tensors"] = aux_tensors

            # Recompute fastdiv_mods if necessary
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    (
                        seqlen_q_divmod
                        if not recompute_fastdiv_mods_q
                        else FastDivmodDivisor(seqlen.seqlen_q)
                    ),
                    (
                        seqlen_k_divmod
                        if not recompute_fastdiv_mods_k
                        else FastDivmodDivisor(seqlen.seqlen_k)
                    ),
                )

            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
                store_row_max=self.store_row_max,
            )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen,
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = self.process_work_tile(seqlen, n_block_min, n_block_max)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                inv_softmax_scale=inv_softmax_scale,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                tS2RsBias=tS2RsBias,
                bias_s2r_thr_copy=bias_s2r_thr_copy,
                pipeline_bias=pipeline_bias,
                pipeline_sf_overlap=pipeline_sf_overlap,
            )

            if const_expr(self.use_block_sparsity) or has_work:
                pipeline_sm_stats.producer_acquire_w_index_phase(
                    stage, sm_stats_producer_phase
                )
                sm_stats_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                # When aux_tensors exist, Q indices beyond seqlen_q must be wrapped to avoid
                # OOB aux_tensor access. Only edge tiles (where m_tile_end > seqlen_q) need this.
                if const_expr(aux_tensors is not None):
                    m_tile_end = (
                        (self.q_stage * m_block + stage + 1) * self.cta_group_size
                    ) * self.m_block_size
                    check_m_boundary = m_tile_end > seqlen.seqlen_q
                else:
                    check_m_boundary = False
                (
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    self.q_stage,
                    Int32(stage),
                    check_m_boundary,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                if not empty_tile:
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    # if tidx == 0:
                    #     cute.printf("softmax row sum stage %d: %f, row_max = %f\n", stage, softmax.row_sum[0], softmax.row_max[0])
                    # See block_sparse_utils.py NOTE [SM100 block-sparse empty tiles: mbarrier contract].
                    # pipeline_sm_stats.producer_commit_w_index(stage)
                    sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
                    # if tidx == 0: cute.printf("softmax row sum stage %d: %f\n", stage, softmax.row_sum[0])
            else:
                if self.process_work_tile(seqlen, n_block_min, n_block_max):
                    if const_expr(self.has_bias) and (
                        const_expr(not self.is_split_kv) or num_bias_loads > 0
                    ):
                        (
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            s0_s1_sequence_phase,
                            bias_si_consumer_state,
                        ) = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            s0_s1_sequence_phase,
                            n_block_max - 1,
                            is_first=True,
                            mask_fn=partial(mask_fn, mask_seqlen=True),
                            apply_bias=True,
                            bias_si_consumer_state=bias_si_consumer_state,
                        )
                    else:
                        (
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            s0_s1_sequence_phase,
                        ) = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            s0_s1_sequence_phase,
                            n_block_max - 1,
                            is_first=True,
                            mask_fn=partial(mask_fn, mask_seqlen=True),
                        )
                    n_block_max -= 1

                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local or self.has_bias):
                        if const_expr(self.has_bias):
                            n_block_min_causal_local_mask = max(
                                n_block_max + 1 - num_bias_loads, n_block_min
                            )
                        else:
                            n_block_min_causal_local_mask = (
                                block_info.get_n_block_min_causal_local_mask(
                                    seqlen, m_block, n_block_min
                                )
                            )
                        # if tidx == 0:
                        #     cute.printf("n_block_max = {}, n_block_min_causal_local_mask = {}", n_block_max, n_block_min_causal_local_mask)
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                bias_si_consumer_state,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=(
                                    partial(mask_fn, mask_seqlen=False)
                                    if const_expr(
                                        not self.has_bias
                                        # The bias tile only encodes causal (-inf
                                        # for future keys); it stores 0.0, not
                                        # -inf, for keys older than the extent. So
                                        # on the local path the left sliding-window
                                        # edge must still be masked per-element
                                        # here, else it is enforced only at n_block
                                        # granularity and leaks up to tile_n-1
                                        # tokens (catastrophic for prediction>1,
                                        # where the earliest row's left bound can
                                        # fall a whole block early).
                                        or (
                                            self.is_local
                                            and block_info.window_size_left is not None
                                        )
                                    )
                                    else None
                                ),
                                apply_bias=True,
                                bias_si_consumer_state=bias_si_consumer_state,
                            )
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )
                    # The remaining iterations have no masking (but may still need mask_mod)
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    # if tidx == 0:
                    #     cute.printf("n_block_max = {}, n_block_min_before_local_mask = {}", n_block_max, n_block_min_causal_local_mask)
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        n_block = n_block_max - n_tile - 1
                        if const_expr(self.mask_mod is not None):
                            (
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        else:
                            (
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                            )
                    # Separate iterations with local masking on the left
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            0, n_block_max - n_block_min, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                            ) = softmax_step(
                                mma_si_consumer_phase,
                                sm_stats_producer_phase,
                                s0_s1_sequence_phase,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    # Dense path always writes scale / signals
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    # pipeline_sm_stats.producer_commit_w_index(stage)
                    sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + cute.math.log2(softmax.row_sum[0], fastmath=True)) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Write row max to gmem directly
            if const_expr(self.store_row_max):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mRowMax_cur = (
                            mRowMax[None, head_idx, batch_idx, split_idx]
                            if const_expr(mRowMax is not None)
                            else None
                        )
                    else:
                        mRowMax_cur = (
                            mRowMax[None, head_idx, batch_idx]
                            if const_expr(mRowMax is not None)
                            else None
                        )
                else:
                    offset = (
                        seqlen.offset_q
                        if const_expr(not self.pack_gqa)
                        else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mRowMax_cur = (
                            cute.domain_offset(
                                (offset,), mRowMax[None, head_idx, split_idx]
                            )
                            if const_expr(mRowMax is not None)
                            else None
                        )
                    else:
                        mRowMax_cur = (
                            cute.domain_offset((offset,), mRowMax[None, head_idx])
                            if const_expr(mRowMax is not None)
                            else None
                        )
                seqlen_q = (
                    seqlen.seqlen_q
                    if const_expr(not self.pack_gqa)
                    else seqlen.seqlen_q * self.qhead_per_kvhead
                )
                mma_tile_coord_v = thr_mma_qk.thr_idx
                m_tile_idx = (
                    m_block * self.q_stage + stage
                ) * self.cta_group_size + mma_tile_coord_v
                gRowMax = cute.local_tile(
                    mRowMax_cur, (self.m_block_size,), (m_tile_idx,)
                )
                row_max_true = softmax.row_max_true[0]
                acc_O_mn_row_is_nan = row_max_true != row_max_true
                if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                    gRowMax[tidx] = (
                        row_max_true * softmax_scale_true
                        if not acc_O_mn_row_is_nan
                        else -Float32.inf
                    )

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        # This is equivalent to pipeline_s0_s1.producer_tail
        if const_expr(self.s0_s1_barrier):
            if stage == 0:
                pipeline_s0_s1_sequence.sync_object_full.wait(
                    stage, s0_s1_sequence_phase
                )

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.core.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        inv_softmax_scale: Float32,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        tS2RsBias: Optional[cute.Tensor] = None,
        bias_s2r_thr_copy: Optional[cute.CopyAtom] = None,
        apply_bias: bool = False,
        pipeline_bias: Optional[pipeline.PipelineAsync] = None,
        bias_si_consumer_state: Optional[pipeline.PipelineState] = None,
        pipeline_sf_overlap: Optional[pipeline.PipelineAsync] = None,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] * self.v_mma_dtype.width // Float32.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        # tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(
            thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype
        )
        # tSrS_t2r = copy_utils.load_t2r(thr_tmem_load, tScS_shape, tStS_t2r)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        # for i in cutlass.range_constexpr(cute.size(tStS_t2r.shape[2])):
        #     cute.copy(thr_tmem_load, tStS_t2r[None, None, i], tSrS_t2r[None, None, i])

        if const_expr(self.qk_blockscaled and self.q_stage == 2):
            cute.arch.fence_view_async_tmem_load()
            pipeline_sf_overlap.consumer_release_w_index(1 - stage)

        if const_expr(self.has_bias and apply_bias):
            bias_si_phase = bias_si_consumer_state.phase
            bias_si_stage = bias_si_consumer_state.index + stage
            pipeline_bias.consumer_wait_w_index_phase(bias_si_stage, bias_si_phase)
            tBrS = cute.make_tensor(
                tSrS_t2r.iterator,
                cute.make_fragment_like(tS2RsBias[None, None, None, 0].layout),
            )
            if const_expr(self.bias_block_size == 128) or tidx < self.bias_block_size:
                for i in cutlass.range_constexpr(cute.size(tS2RsBias.shape[2])):
                    tBrS_cur = tBrS[None, 0, i]
                    tS2RsBias_cur = tS2RsBias[None, 0, i, bias_si_stage]
                    tS2RrBias_cur = cute.make_fragment_like(
                        tS2RsBias[None, 0, 0, 0]
                    )  # create fragment inside loop
                    cute.copy(bias_s2r_thr_copy, tS2RsBias_cur, tS2RrBias_cur)
                    for j in cutlass.range_constexpr(0, cute.size(tBrS_cur.shape), 2):
                        tBrS_cur[j], tBrS_cur[j + 1] = cute.arch.fma_packed_f32x2(
                            (
                                tS2RrBias_cur[j].to(self.qk_acc_dtype),
                                tS2RrBias_cur[j + 1].to(self.qk_acc_dtype),
                            ),
                            (inv_softmax_scale, inv_softmax_scale),
                            (tBrS_cur[j], tBrS_cur[j + 1]),
                        )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Softmax) + stage,
                number_of_threads=128,
            )
            pipeline_bias.consumer_release_w_index(bias_si_stage)

        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
            )

        # may need to mask out dummy bias add
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            # tSrScale_r2t = cute.make_fragment(thr_tmem_store_scale.partition_S(tScScale).shape, Float32)
            # tSrScale_r2t[0] = acc_scale
            # cute.copy(thr_tmem_store_scale, tSrScale_r2t, tStScale_r2t)
            # cute.arch.fence_view_async_tmem_store()
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
            # if thread_idx == 0: cute.printf("softmax acc_scale stage %d: %f, row_max = %f\n", stage, acc_scale, row_max)
        # Notify correction wg that row_max is ready
        # pipeline_sm_stats.producer_commit_w_index(stage)
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        # if thread_idx == 0 and stage == 0: cute.print_tensor(tSrS_t2r)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        # Sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_mma_dtype),
            tSrS_t2r.layout,
        )
        # softmax.scale_apply_exp2_convert(tSrS_t2r, row_max, tSrP_r2t)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=(
                self.ex2_emu_freq
                if const_expr(mask_fn is None and not apply_bias)
                else 0
            ),
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        # acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
        if const_expr(bias_si_consumer_state is not None):
            if const_expr(self.has_bias and apply_bias):
                bias_si_consumer_state.advance()
            return (
                mma_si_consumer_phase ^ 1,
                sm_stats_producer_phase ^ 1,
                s0_s1_sequence_phase ^ 1,
                bias_si_consumer_state,
            )
        else:
            return (
                mma_si_consumer_phase ^ 1,
                sm_stats_producer_phase ^ 1,
                s0_s1_sequence_phase ^ 1,
            )

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        tile_scheduler=None,
        pipeline_load_epi: Optional[pipeline.PipelineAsync] = None,
        pipeline_vq: Optional[pipeline.PipelineTmaAsync] = None,
        pipeline_v_mma: Optional[pipeline.PipelineAsyncUmma] = None,
        sVq: Optional[cute.Tensor] = None,
        sSFV: Optional[cute.Tensor] = None,
        sV: Optional[cute.Tensor] = None,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        # tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        # tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
        # tStScales = tuple(
        #     cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
        #     for stage in range(self.q_stage)
        # )
        # tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        # tmem_load_v_atom = cute.make_copy_atom(
        #     tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype
        # )
        # thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

        # tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
        # tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.q_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)
        if const_expr(self.v_dequant):
            vq_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kv_stage
            )
            v_dequant_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.v_mma_stage
            )
        load_epi_producer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, 1
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                m_block,
                split_idx,
                batch_idx,
            )

            if const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(
                    mO, batch_idx, dim=3, ragged=self.ragged_O
                )[None, None, head_idx, split_idx]
            else:
                mO_cur = seqlen.offset_batch_Q(
                    mO, batch_idx, dim=3, ragged=self.ragged_O
                )[None, None, head_idx]
            gO = None
            if const_expr(self.use_tma_O or not self.pack_gqa):
                tiler_gO = (
                    (self.mma_tiler_pv[0] * self.q_stage),
                    self.head_dim_v_padded,
                )
                gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128 * 2, 128)
                gO = layout_utils.select(
                    cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                )  # (128, 128, 2)
                gO = cute.flat_divide(
                    gO, (self.mma_tiler_pv[0] // self.cta_group_size,)
                )[None, mma_tile_coord_v, None, None]

            # Default LSE to -inf for invalid split_idx tiles
            stats = [
                (
                    0.0,
                    (
                        -Float32.inf
                        if const_expr(mLSE is not None or learnable_sink is not None)
                        else None
                    ),
                    True,
                )
            ] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                    seqlen_info=seqlen,
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = self.process_work_tile(seqlen, n_block_min, n_block_max)

            if const_expr(self.v_dequant):
                dequant_v_fn = partial(
                    self.dequant_v,
                    sVq=sVq,
                    sSFV=sSFV,
                    sV=sV,
                    pipeline_vq=pipeline_vq,
                    pipeline_v_mma=pipeline_v_mma,
                    seqlen_k=seqlen.seqlen_k,
                )

            if has_work:
                # V0 dequant FIRST — overlaps with QK0 gemm + softmax
                if const_expr(self.v_dequant):
                    # Tiles arrive in the softmax/load order: the boundary
                    # tile (n_block_max - 1) first, then descending. Block-
                    # sparse orders are opaque here -> -1 disables the
                    # seqused zeroing for them.
                    if const_expr(self.use_block_sparsity):
                        dq_n_block = Int32(-1)
                    else:
                        dq_n_block = n_block_max - 1
                    vq_consumer_state, v_dequant_producer_state = dequant_v_fn(
                        vq_consumer_state=vq_consumer_state,
                        v_dequant_producer_state=v_dequant_producer_state,
                        tidx=tidx,
                        n_block=dq_n_block,
                    )

                # Now wait for first sm_stats (no correction needed for first iter)
                # pipeline_sm_stats.consumer_wait_w_index_phase(0, sm_stats_consumer_phase)
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                if const_expr(self.q_stage == 2):
                    # pipeline_sm_stats.consumer_wait_w_index_phase(1, sm_stats_consumer_phase)
                    sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                # tSrScale_t2r = cute.make_fragment(tSrScale_t2r_shape, Float32)
                for i in cutlass.range(total_block_count - 1, unroll=1):
                    if const_expr(self.v_dequant):
                        if const_expr(self.use_block_sparsity):
                            dq_n_block = Int32(-1)
                        else:
                            dq_n_block = n_block_max - 2 - i
                        vq_consumer_state, v_dequant_producer_state = dequant_v_fn(
                            vq_consumer_state=vq_consumer_state,
                            v_dequant_producer_state=v_dequant_producer_state,
                            tidx=tidx,
                            n_block=dq_n_block,
                        )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait for S0 / S1
                        # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * 4 + warp_idx
                        )
                        # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                        # cute.arch.fence_view_async_tmem_load()
                        # scale = tSrScale_t2r[0]
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        # should_rescale = True
                        # if tidx == 0: cute.printf("Correction scale i = %d, for stage %d: %f, should_rescale = %d\n", i, stage, scale, should_rescale)
                        # Don't need O_full anymore, since by the time softmax has signaled the correction
                        # warps, S_i must have been done, so O_i-1 must have been done as well.
                        # pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv, tOtO[None, None, None, stage], tidx, scale
                            )
                        # Notify mma warp that O has been rescaled
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(
                            self.q_stage - 1 - stage
                        )
                    sm_stats_consumer_phase ^= 1
                    # o_corr_consumer_phase ^= 1
                if const_expr(self.q_stage == 2):
                    pipeline_sm_stats.consumer_release_w_index(1)
                # End of seqlen_corr_loop_steps

                # Even in the case of self.overlap_sO_sQ, we can write to stage 0 of sO without
                # additional sync because the MMA in the top half must have been done.
                # Similarly we can write to stage 1 of sO without additional sync.
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:  # Each thread might have a different sink value due to different q_head
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                (
                                    (
                                        (m_block * self.q_stage + stage)
                                        * self.cta_group_size
                                        + mma_tile_coord_v
                                    )
                                    * self.m_block_size
                                    + tidx
                                )
                                % self.qhead_per_kvhead
                                + head_idx * self.qhead_per_kvhead
                            )
                            learnable_sink_val[stage] = Float32(
                                learnable_sink[q_head_idx]
                            )
                for stage in cutlass.range_constexpr(self.q_stage):
                    # pipeline_sm_stats.consumer_wait_w_index_phase(stage, sm_stats_consumer_phase)
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    # cute.copy(tiled_tmem_load_vec, tStScales_t2r[stage], tSrScale_t2r)
                    # cute.arch.fence_view_async_tmem_load()
                    # scale = tSrScale_t2r[0]
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                # It's possible to have an empty row with splitKV.
                                row_max = sink_val * (LOG2_E / softmax_scale_log2)
                                row_sum = Float32(1.0)
                            else:
                                row_sum += cute.math.exp2(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2,
                                    fastmath=True,
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(
                        row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0
                    )
                    # if tidx == 0: cute.printf("Epilogue stage %d: row_sum=%f, scale=%f, is_zero_or_nan=%d\n", stage, row_sum, scale, acc_O_mn_row_is_zero_or_nan)
                    # Wait for the last O to be ready from the MMA warp
                    pipeline_o_acc.consumer_wait_w_index_phase(
                        stage, o_corr_consumer_phase
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_acquire_w_index_phase(
                            stage, corr_epi_producer_phase
                        )
                    gO_stage = (
                        gO[None, None, stage] if const_expr(gO is not None) else None
                    )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO_stage,
                        gmem_tiled_copy_O,
                    )
                    # Signal for the next work tile that O buffers in tmem are already read, so
                    # mma warp can write to them
                    pipeline_s_p_o.consumer_release_w_index(stage)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_commit_w_index(stage)
                    # if tidx == 0: cute.printf("Correction final scale for stage %d: %f\n", stage, scale)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                if const_expr(self.use_block_sparsity):
                    (
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtO,
                        sO,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        pipeline_o_epi,
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            # signal smem is free for next load in persistent work loop
            if const_expr(
                self.overlap_sO_sQ
                and self.is_persistent
                and self.use_correction_warps_for_epi
            ):
                pipeline_load_epi.producer_acquire(load_epi_producer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.producer_commit(load_epi_producer_state)
                load_epi_producer_state.advance()

            if const_expr(mLSE is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen.offset_q
                        if const_expr(not self.pack_gqa)
                        else (0, seqlen.offset_q)
                    )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset(
                            (offset,), mLSE[None, head_idx, split_idx]
                        )
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = (
                        m_block * self.q_stage + stage
                    ) * self.cta_group_size + mma_tile_coord_v
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    # if tidx == 0 and stage <= 1:
                    #     cute.printf("row_sum = {}, row_max = {}, acc_O_mn_row_is_zero_or_nan = {}\n", row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    LN2 = math.log(2.0)
                    lse = (
                        (
                            row_max * softmax_scale_log2
                            + cute.math.log2(row_sum, fastmath=True)
                        )
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen.seqlen_q * self.qhead_per_kvhead
                    )
                    if const_expr(
                        not self.pack_gqa
                        or self.m_block_size % self.qhead_per_kvhead == 0
                    ):
                        gLSE = cute.local_tile(
                            mLSE_cur, (self.m_block_size,), (m_tile_idx,)
                        )
                        if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                            # This actually just works with PackGQA too
                            gLSE[tidx] = lse
                    else:
                        idx = m_tile_idx * self.m_block_size + tidx
                        if idx < seqlen_q:
                            m_idx = idx // self.qhead_per_kvhead
                            h_idx = idx - m_idx * self.qhead_per_kvhead
                            lse_ptr_i64 = utils.elem_pointer(
                                mLSE_cur, ((h_idx, m_idx),)
                            ).toint()
                            lse_gmem_ptr = cute.make_ptr(
                                mLSE_cur.element_type,
                                lse_ptr_i64,
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            cute.make_tensor(lse_gmem_ptr, (1,))[0] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop
        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(
                self.q_stage - 1, corr_epi_producer_phase
            )

    @cute.jit
    def dequant_v(
        self,
        sVq: cute.Tensor,
        sSFV: cute.Tensor,
        sV: cute.Tensor,
        pipeline_vq: pipeline.PipelineTmaAsync,
        pipeline_v_mma: pipeline.PipelineAsyncUmma,
        vq_consumer_state: pipeline.PipelineState,
        v_dequant_producer_state: pipeline.PipelineState,
        tidx: Int32,
        n_block: Int32,
        seqlen_k: Int32,
    ):
        """``n_block``/``seqlen_k``: rows at or beyond ``seqlen_k`` dequantize
        to exact 0.0 — unwritten fp8 pool bytes may decode to NaN, and the
        P@V MMA computes 0 * NaN = NaN for masked columns, so a value-side
        select is the only way to keep OOB data at zero effect (the S-side
        masks cannot help). ``n_block < 0`` disables the bound (callers that
        cannot name the tile, e.g. block-sparse iteration orders)."""
        num_threads = cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        num_load_elems = 128 // self.v_dtype.width // 2
        num_store_elems = 128 // self.v_mma_dtype.width
        assert num_load_elems == num_store_elems

        sVq_layout_dnp = cute.make_ordered_layout(
            (*cute.select(self.mma_tiler_pv, mode=[1, 2]), self.kv_stage),
            order=(0, 1, 2),
        )
        # (tile_n, hdim_v, kv_stage), row-major
        sVq_layout_ndp = cute.make_ordered_layout(
            (*cute.select(self.mma_tiler_pv, mode=[1, 2]), self.kv_stage),
            order=(1, 0, 2),
        )
        # (tile_n, hdim_v, v_mma_stage), row-major
        sV_layout_ndp = cute.make_ordered_layout(
            (*cute.select(self.mma_tiler_pv, mode=[1, 2]), self.v_mma_stage),
            order=(1, 0, 2),
        )

        vq_stage = vq_consumer_state.index
        v_upcast_stage = v_dequant_producer_state.index

        sV_ = cute.composition(sV, sV_layout_ndp)
        sVq_ = cute.composition(sVq, sVq_layout_ndp)
        # passing col major to composition is correct for scale
        sSFV_ = cute.composition(
            sSFV, sVq_layout_dnp
        )  # ((32,4),(32,4),2):((16,4),(0,1),512)
        sSFV_cur = sSFV_[None, None, vq_stage]

        tiled_copy_s2r = copy_utils.tiled_copy_2d(
            self.v_dtype,
            # threads_per_row=self.head_dim_v_padded//num_load_elems,
            threads_per_row=4,
            num_threads=num_threads,
            num_copy_elems=num_load_elems,
        )
        tiled_copy_r2s = copy_utils.tiled_copy_2d(
            self.v_mma_dtype,
            # threads_per_row=self.head_dim_v_padded//num_store_elems,
            threads_per_row=4,
            num_threads=num_threads,
            num_copy_elems=num_store_elems,
        )
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        # (V, M, N)
        tVsV_f8 = thr_copy_s2r.partition_S(sVq_)[None, None, None, vq_stage]
        tVrV_f8 = cute.make_fragment_like(tVsV_f8)
        tVsV_f16 = thr_copy_r2s.partition_S(sV_)[None, None, None, v_upcast_stage]

        cV = cute.make_identity_tensor(cute.select(self.mma_tiler_pv, mode=[1, 2]))
        tVcV = thr_copy_s2r.partition_S(cV)

        num_rows = cute.size(tVrV_f8.shape[1])
        num_cols = cute.size(tVrV_f8.shape[2])

        delay_v_mma_acquire = self.v_mma_stage == 1

        pipeline_vq.consumer_wait_w_index_phase(
            vq_consumer_state.index, vq_consumer_state.phase
        )
        if const_expr(not delay_v_mma_acquire):
            pipeline_v_mma.producer_acquire_w_index_phase(
                v_dequant_producer_state.index, v_dequant_producer_state.phase
            )

        cute.copy(tiled_copy_s2r, tVsV_f8, tVrV_f8)

        if const_expr(delay_v_mma_acquire):
            # (1) load scales, fence and sync
            # (2) release vq
            # (3) acquire v_mma
            scales_ue8m0_all = cute.make_rmem_tensor(
                (num_cols, num_rows), dtype=sSFV.element_type
            )
            for i in cutlass.range_constexpr(num_rows):
                row_idx = tVcV[0, i, 0][0]
                sSFV_cur_row = sSFV_cur[row_idx, (0, None)]  # 4 values
                cute.autovec_copy(sSFV_cur_row, scales_ue8m0_all[None, i])

            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Correction),
                number_of_threads=len(self.correction_warp_ids) * cute.arch.WARP_SIZE,
            )

            if const_expr(self.use_tma_KV):
                pipeline_vq.consumer_release_w_index(vq_consumer_state.index)
            else:
                if cute.arch.lane_idx() == 0:
                    pipeline_vq.consumer_release_w_index(vq_consumer_state.index)

            pipeline_v_mma.producer_acquire_w_index_phase(
                v_dequant_producer_state.index, v_dequant_producer_state.phase
            )

        # TODO: used fused f8 -> bf16 + ue8m0 scale cvt instruction
        # note: counting on src zfill to avoid predication
        # tVrV_f16 = cute.make_fragment_like(tVsV_f16)

        for i in cutlass.range_constexpr(num_rows):
            row_idx = tVcV[0, i, 0][0]
            tVsV_f8_frg = tVsV_f8[None, i, None]
            tVrV_f8_frg = tVrV_f8[None, i, None]
            tVsV_f16_frg = tVsV_f16[None, i, None]
            # tVrV_f16_frg = tVrV_f16[None, i, None]
            tVrV_f16_frg = cute.make_fragment_like(tVsV_f16_frg)
            if const_expr(not delay_v_mma_acquire):
                scales_ue8m0 = cute.make_rmem_tensor(
                    (num_cols,), dtype=sSFV.element_type
                )
            else:
                scales_ue8m0 = scales_ue8m0_all[None, i]
            scales_bf16 = cute.make_rmem_tensor((num_cols,), dtype=cutlass.BFloat16)
            sSFV_cur_row = sSFV_cur[row_idx, (0, None)]  # 4 values

            # step 1: load 4 scales for row and cvt to bfloat16
            if const_expr(not delay_v_mma_acquire):
                cute.autovec_copy(sSFV_cur_row, scales_ue8m0)
            cvt_tensor_ue8m0_to_bf16(scales_ue8m0, scales_bf16)

            # step 2: upcast vals to bf16
            # no direct cvt f8 -> bf16 with tensor ssa, go through f32
            tVrV_f16_frg.store(
                tVrV_f8_frg.load().to(cutlass.Float32).to(self.v_mma_dtype)
            )

            # step 3: scale
            for j in cutlass.range_constexpr(num_cols):
                tVrV_f16_r8 = tVrV_f16_frg[None, j]
                tVrV_f16_r8.store(tVrV_f16_r8.load() * scales_bf16[j])

            # step 4: rows past seqused carry unwritten pool bytes; overwrite
            # with exact zeros (a select — multiplication cannot kill NaN).
            if n_block >= 0:
                if n_block * self.n_block_size + row_idx >= seqlen_k:
                    tVrV_f16_frg.fill(0.0)

            cute.copy(tiled_copy_r2s, tVrV_f16_frg, tVsV_f16_frg)

        # cute.copy(tiled_copy_r2s, tVrV_f16, tVsV_f16)

        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwdSm100.Correction),
            number_of_threads=len(self.correction_warp_ids) * cute.arch.WARP_SIZE,
        )

        # deferred vq release
        if const_expr(not delay_v_mma_acquire):
            if const_expr(self.use_tma_KV):
                pipeline_vq.consumer_release_w_index(vq_consumer_state.index)
            else:
                if cute.arch.lane_idx() == 0:
                    pipeline_vq.consumer_release_w_index(vq_consumer_state.index)

        pipeline_v_mma.producer_commit_w_index(v_dequant_producer_state.index)
        vq_consumer_state.advance()
        v_dequant_producer_state.advance()
        return vq_consumer_state, v_dequant_producer_state

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.composition(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_fragment((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_fragment(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: cutlass.Constexpr[Int32],
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        corr_tile_size = 8 * 32 // self.o_dtype.width
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
        )

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(
            thr_tmem_load, tOsO_i[(None, None), None]
        )
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(
            self.head_dim_v_padded // corr_tile_size, unroll_full=True
        ):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_fragment(
                tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            cute.arch.fence_view_async_tmem_load()
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

        if const_expr(self.use_correction_warps_for_epi):
            assert not self.use_tma_O
            assert gmem_tiled_copy_O is not None
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE,
            )
            mma_tile_coord_v = thr_mma.thr_idx
            m_tile_idx = (
                m_block * self.q_stage + stage
            ) * self.cta_group_size + mma_tile_coord_v
            self._store_O_to_gmem(
                sO, gO, mO_cur, gmem_tiled_copy_O, tidx, seqlen_q, m_tile_idx
            )

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: Optional[cute.Tensor],
        mO_cur: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        seqlen_q: Int32,
        m_tile_idx: Int32,
    ):
        """Copy a single stage of O from smem to gmem via registers."""
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO_stage)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
        tOpO = copy_utils.predicate_k(tOcO, limit=mO_cur.shape[1])
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_v_padded,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
        )

        # load acc O from smem to rmem for wider vectorization
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)
        # copy acc O from rmem to gmem
        if const_expr(not self.pack_gqa):
            assert gO is not None
            tOgO = gmem_thr_copy_O.partition_D(gO)
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0]
                    < seqlen_q - m_tile_idx * self.m_block_size - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=(
                            tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None
                        ),
                    )
        else:
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_tile_idx, seqlen_q
            )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        mma_tile_coord_v: Int32 = 0,
        tile_scheduler=None,
        pipeline_load_epi: Optional[pipeline.PipelineAsync] = None,
    ):
        epi_consumer_phase = Int32(0)
        load_epi_producer_state = pipeline_custom.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, 1
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen,
                m_block,
                split_idx,
                batch_idx,
            )

            if self.process_work_tile(seqlen, n_block_min, n_block_max):
                if const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(
                        mO, batch_idx, dim=3, ragged=self.ragged_O
                    )[None, None, head_idx, split_idx]
                else:
                    mO_cur = seqlen.offset_batch_Q(
                        mO, batch_idx, dim=3, ragged=self.ragged_O
                    )[None, None, head_idx]
                gO = None
                if const_expr(self.use_tma_O or not self.pack_gqa):
                    tiler_gO = (
                        (self.mma_tiler_pv[0] * self.q_stage),
                        self.head_dim_v_padded,
                    )
                    gO = cute.local_tile(
                        mO_cur, tiler_gO, (m_block, 0)
                    )  # (128 * 2, 128)
                    gO = layout_utils.select(
                        cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                    )  # (128, 128, 2)
                    gO = cute.flat_divide(
                        gO, (self.mma_tiler_pv[0] // self.cta_group_size,)
                    )[None, mma_tile_coord_v, None, None]

                if const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range(self.q_stage, unroll_full=True):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        pipeline_o_epi.consumer_wait_w_index_phase(
                            stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(
                            self.q_stage - 1 - stage, read=True
                        )
                        pipeline_o_epi.consumer_release_w_index(stage)
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        pipeline_o_epi.consumer_wait_w_index_phase(
                            stage, epi_consumer_phase
                        )
                        # 2. copy O0 / O1 to gmem
                        m_tile_idx = (
                            m_block * self.q_stage + stage
                        ) * self.cta_group_size + mma_tile_coord_v
                        gO_stage = (
                            gO[None, None, stage]
                            if const_expr(gO is not None)
                            else None
                        )
                        self._store_O_to_gmem(
                            sO[None, None, stage],
                            gO_stage,
                            mO_cur,
                            gmem_tiled_copy_O,
                            tidx,
                            seqlen.seqlen_q,
                            m_tile_idx,
                        )
                        pipeline_o_epi.consumer_release_w_index(stage)

                epi_consumer_phase ^= 1

            # signal smem is free for next load in persistent work loop
            if const_expr(
                self.overlap_sO_sQ
                and self.is_persistent
                and not self.use_correction_warps_for_epi
            ):
                pipeline_load_epi.producer_acquire(load_epi_producer_state)
                with cute.arch.elect_one():
                    pipeline_load_epi.producer_commit(load_epi_producer_state)
                load_epi_producer_state.advance()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            if (
                cute.arch.thread_idx()[0]
                == self.clc_scheduler_warp_id * cute.arch.WARP_SIZE
            ):
                fa_printf(
                    3,
                    "[CLC] query sm={} cta={} (m_blk={},h={},b={},s={}) valid={}\n",
                    smid(),
                    cute.arch.block_idx()[0],
                    work_tile.tile_idx[0],
                    work_tile.tile_idx[1],
                    work_tile.tile_idx[2],
                    work_tile.tile_idx[3],
                    work_tile.is_valid_tile,
                )
        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_tile = tile_scheduler.advance_to_next_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
        load_SFQ_fn: Optional[Callable] = None,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(
            src_idx=block,
            dst_idx=stage,
            tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage),
        )
        if const_expr(load_SFQ_fn is not None):
            load_SFQ_fn(
                src_idx=block,
                dst_idx=stage,
                tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage),
            )

    @cute.jit
    def cpasync_load_SFQ(
        self,
        gSFQ: cute.Tensor,
        sSFQ: cute.Tensor,
        gmem_tiled_copy_SFQ: cute.TiledCopy,
        pipeline_sfq: pipeline.PipelineAsync,
        tidx: Int32,
        phase: Int32,
        block: Int32,  # local idx, matching load_Q
        stage: Int32,  # smem stage
        m_block: Int32,  # global m_block
        seqlen_q: Int32,
    ):
        gmem_thr_copy_SFQ = gmem_tiled_copy_SFQ.get_slice(tidx)
        gSFQ_cpt = cute.filter_zeros(gSFQ)
        sSFQ_cpt = cute.filter_zeros(sSFQ)
        sSFQ_cpt_shape_mdp = (
            self.m_block_size,
            self.head_dim_padded // 32,
            self.q_stage,
        )
        sSFQ_cpt_layout_mdp = cute.make_ordered_layout(
            sSFQ_cpt_shape_mdp,
            order=(0, 1, 2),
        )
        # (tile_m, 4)
        sSFQ_cpt_mdp = cute.composition(sSFQ_cpt, sSFQ_cpt_layout_mdp)
        # (V=(4,1), M=128//num_load_threads, N=1, P=q_stage)
        tSFQgSFQ = gmem_thr_copy_SFQ.partition_S(gSFQ_cpt)
        tSFQsSFQ = gmem_thr_copy_SFQ.partition_S(sSFQ_cpt_mdp)

        cSFQ = cute.make_identity_tensor(
            (self.m_block_size, self.head_dim_padded // 32),
        )
        tSFQcSFQ = gmem_thr_copy_SFQ.partition_S(cSFQ)
        tSFQc0SFQ = gmem_thr_copy_SFQ.get_slice(0).partition_S(cSFQ)

        seqlen_q_row_limit = (
            seqlen_q
            - m_block * self.cta_tiler[0]
            - block * self.m_block_size
            - tSFQcSFQ[0][0]
            if m_block >= 0
            else 0
        )

        # print("tSFQgSFQ: ", tSFQgSFQ)
        # print("tSFQsSFQ: ", tSFQsSFQ)
        # print("tSFQcSFQ: ", tSFQcSFQ)

        pipeline_sfq.producer_acquire_w_index_phase(stage, phase)

        # ((4,1),m,1)
        tSFQgSFQ_cur = tSFQgSFQ[None, None, None, block]
        tSFQsSFQ_cur = tSFQsSFQ[None, None, None, stage]

        for m in cutlass.range_constexpr(cute.size(tSFQsSFQ_cur, mode=[1])):
            row_valid = tSFQc0SFQ[0, m, 0][0] < seqlen_q_row_limit
            should_load = cute.make_fragment_like(
                tSFQsSFQ_cur[(0, None), m, None], cute.Boolean
            )
            should_load.fill(row_valid)

            cute.copy(
                gmem_thr_copy_SFQ,
                tSFQgSFQ_cur[None, m, None],
                tSFQsSFQ_cur[None, m, None],
                pred=should_load,
            )

        cute.arch.cp_async_commit_group()
        pipeline_sfq.sync_object_full.arrive_cp_async_mbarrier(stage)

    def load_Q_non_tma(
        self,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        gmem_tiled_copy_Q: cute.TiledCopy,
        pipeline_q: pipeline.PipelineAsync,
        tidx: Int32,
        seqlen_q: Int32,
        m_block: Int32,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        assert self.cta_group_size == 1, "cta_group_size must be 1 for non-tma Q load"
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_padded,
            self.check_hdim_oob,
            self.qhead_per_kvhead,
        )
        sQ_stage = sQ[None, None, None, stage]
        sQ_pi = cute.make_tensor(
            sQ_stage.iterator,
            cute.make_layout(
                (sQ_stage.shape[0][0], (sQ_stage.shape[0][1], sQ_stage.shape[2])),
                stride=(
                    sQ_stage.stride[0][0],
                    (sQ_stage.stride[0][1], sQ_stage.stride[2]),
                ),
            ),
        )
        pack_gqa.load_Q(
            mQ, sQ_pi, gmem_tiled_copy_Q, tidx, m_block * self.q_stage + block, seqlen_q
        )
        cute.arch.cp_async_commit_group()
        pipeline_q.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        tma_atom_sf: Optional[cute.CopyAtom] = None,
        tXgSFX: Optional[cute.Tensor] = None,
        tXsSFX: Optional[cute.Tensor] = None,
        sSFX: Optional[cute.Tensor] = None,
        stage_dilation: cutlass.Constexpr[int] = 1,
    ):
        assert K_or_V in ("K", "V")
        blockscaled: cutlass.Constexpr[bool] = all(
            [t is not None for t in [tXgSFX, tXsSFX, sSFX]]
        )
        stage, phase = producer_state.index, producer_state.phase
        sf_stage = stage
        stage *= stage_dilation
        sf_key = "SFK" if const_expr(K_or_V == "K") else "SFV"
        if const_expr(K_or_V == "V" and self.v_dequant):
            extra_tx_count = 0
        else:
            extra_tx_count = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
            if const_expr(K_or_V == "K"):
                extra_tx_count += self.tma_copy_bytes[sf_key]
        extra_kwargs = (
            {"extra_tx_count": extra_tx_count} if const_expr(self.use_tma_KV) else {}
        )
        pipeline_kv.producer_acquire(producer_state, **extra_kwargs)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                pipeline_kv.sync_object_empty.wait(1, phase)

        if const_expr(self.use_tma_KV):
            assert tXgX is not None and tXsX is not None and tma_atom is not None
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            # Paged: page_size == paged_kv_blocks_per_page * n_block_size; the
            # in-page tile is block % blocks_per_page (tile mode extent
            # matches), page_idx was read at block // blocks_per_page.
            tXgX_cur = (
                tXgX[None, block]
                if const_expr(page_idx is None)
                else tXgX[None, block % self.paged_kv_blocks_per_page, page_idx]
            )
            cute.copy(
                tma_atom,
                tXgX_cur,
                tXsX_cur,
                tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
            )
            if const_expr(blockscaled):
                tXsSFX_cur = tXsSFX[None, sf_stage]
                # Same indexing as tXgX_cur
                tXgSFX_cur = (
                    tXgSFX[None, block]
                    if const_expr(page_idx is None)
                    else tXgSFX[None, block % self.paged_kv_blocks_per_page, page_idx]
                )
                cute.copy(
                    tma_atom_sf,
                    tXgSFX_cur,
                    tXsSFX_cur,
                    tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
                )
        else:
            assert paged_kv_manager is not None
            sX_cur = sX[None, None, None, stage]
            if const_expr(self.uneven_kv_smem):
                sX_cur = self.offset_kv_smem(sX_cur, stage, phase ^ 1)
            paged_kv_manager.load_KV(block, sX_cur, K_or_V)
            if const_expr(sSFX is not None):
                paged_kv_manager.load_sf_KV(
                    block, sSFX[None, None, None, stage], K_or_V
                )
            cute.arch.cp_async_commit_group()
            pipeline_kv.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            # Hint that the offset is 128-bit aligned so that
            # ptr + offset preserves the alignment needed by cp.async.
            offset = cute.assume(offset, divby=128 // self.k_dtype.width)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def make_sf_tmem_copies(
        self,
        tmem_ptr,
        stage,
        tiled_mma_qk,
        sSFQ,
        sSFK,
        sSFQ_layout,
        sSFK_layout,
    ) -> SfS2TCopies:

        # Create SF TMEM tensors and S2T copy partitions for blockscaled
        if const_expr(self.qk_blockscaled):
            sfq_tmem_ptr = cute.recast_ptr(
                tmem_ptr + self.tmem_sfq_offset[stage],
                dtype=self.sfq_dtype,
            )
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.qk_sf_vec_size,
                cute.slice_(sSFQ_layout, (None, None, None, 0)),
            )
            tCtSFQ = cute.make_tensor(sfq_tmem_ptr, tCtSFQ_layout)

            sfk_tmem_ptr = cute.recast_ptr(
                tmem_ptr + self.tmem_sfk_offset[stage],
                dtype=self.sfk_dtype,
            )
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.qk_sf_vec_size,
                cute.slice_(sSFK_layout, (None, None, None, 0)),
            )
            tCtSFK = cute.make_tensor(sfk_tmem_ptr, tCtSFK_layout)

            # S2T copy setup for SFQ
            tCtSFQ_compact = cute.filter_zeros(tCtSFQ)
            copy_atom_s2t_sfq = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
                self.sfq_dtype,
            )
            tiled_copy_s2t_sfq = tcgen05.make_s2t_copy(
                copy_atom_s2t_sfq, tCtSFQ_compact
            )
            thr_copy_s2t_sfq = tiled_copy_s2t_sfq.get_slice(0)
            tCsSFQ_compact = cute.filter_zeros(sSFQ)
            tCsSFQ_compact_s2t_ = thr_copy_s2t_sfq.partition_S(tCsSFQ_compact)
            tCsSFQ_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfq, tCsSFQ_compact_s2t_
            )
            tCtSFQ_compact_s2t = thr_copy_s2t_sfq.partition_D(tCtSFQ_compact)

            # S2T copy setup for SFK
            tCtSFK_compact = cute.filter_zeros(tCtSFK)
            copy_atom_s2t_sfk = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
                self.sfk_dtype,
            )
            tiled_copy_s2t_sfk = tcgen05.make_s2t_copy(
                copy_atom_s2t_sfk, tCtSFK_compact
            )
            thr_copy_s2t_sfk = tiled_copy_s2t_sfk.get_slice(0)
            tCsSFK_compact = cute.filter_zeros(sSFK)
            tCsSFK_compact_s2t_ = thr_copy_s2t_sfk.partition_S(tCsSFK_compact)
            tCsSFK_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfk, tCsSFK_compact_s2t_
            )
            tCtSFK_compact_s2t = thr_copy_s2t_sfk.partition_D(tCtSFK_compact)
        else:
            tCtSFQ, tCtSFK = None, None
            tiled_copy_s2t_sfq, tCsSFQ_compact_s2t, tCtSFQ_compact_s2t = (
                None,
                None,
                None,
            )
            tiled_copy_s2t_sfk, tCsSFK_compact_s2t, tCtSFK_compact_s2t = (
                None,
                None,
                None,
            )

        return SfS2TCopies(
            tCtSFQ=tCtSFQ,
            tCtSFK=tCtSFK,
            tiled_copy_sfq=tiled_copy_s2t_sfq,
            tiled_copy_sfk=tiled_copy_s2t_sfk,
            tCsSFQ_s2t=tCsSFQ_compact_s2t,
            tCtSFQ_s2t=tCtSFQ_compact_s2t,
            tCsSFK_s2t=tCsSFK_compact_s2t,
            tCtSFK_s2t=tCtSFK_compact_s2t,
        )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset(
            (m_block * self.m_block_size, n_block * self.n_block_size), cS
        )
        tScS = thr_mma_qk.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa):
            assert head_divmod is not None
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical, head_offset = divmod(q_physical, head_divmod)
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = divmod(q_idx_logical, seqlen_q_divmod)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=(
                self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa) else 1
            ),
        )

    @cute.jit
    def process_work_tile(
        self,
        seqlen_info: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
    ):
        is_varlen_q = seqlen_info.has_cu_seqlens_q or seqlen_info.has_seqused_q
        process_work_tile_k = (
            const_expr(not self.is_split_kv) or n_block_min < n_block_max
        )
        if const_expr(is_varlen_q and not self.use_varlen_scheduler):
            process_work_tile_q = seqlen_info.seqlen_q > 0
        else:
            process_work_tile_q = True
        return process_work_tile_k and process_work_tile_q


def _to_cute_tensor(tensor: torch.Tensor, assumed_align: int = 16) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=assumed_align).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


def _validate_cuda_int32_vector(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _make_cu_seqlens(lengths: tuple[int, ...]) -> torch.Tensor:
    cumulative = [0]
    for length in lengths:
        if length < 0:
            raise ValueError("sequence lengths must be nonnegative")
        cumulative.append(cumulative[-1] + length)
    return torch.tensor(cumulative, dtype=torch.int32, device="cuda")


def _shear_relative_bias_one(
    raw_bias: torch.Tensor, seqlen_q: int, seqlen_k: int
) -> torch.Tensor:
    heads, rel_extent = raw_bias.shape[1:]
    padded_q = (seqlen_q + 127) // 128 * 128
    padded_extent = rel_extent + 256
    raw_padded = torch.zeros(
        (padded_q, heads, rel_extent),
        dtype=raw_bias.dtype,
        device=raw_bias.device,
    )
    raw_padded[:seqlen_q] = raw_bias[:seqlen_q]
    query = torch.arange(padded_q, device=raw_bias.device)[:, None]
    physical = torch.arange(padded_extent, device=raw_bias.device)[None, :]
    m_block = query // 128
    max_n_block = min((seqlen_k + 127) // 128, 1 << 30)
    n_block = torch.minimum(
        torch.full_like(m_block, max_n_block),
        ((m_block + 1) * 128 + seqlen_k - seqlen_q + 127) // 128,
    )
    logical_k = physical - padded_extent + n_block * 128
    relative = query + seqlen_k - seqlen_q - logical_k
    gather = relative.clamp(0, rel_extent - 1)
    gather = gather[:, None, :].expand(-1, heads, -1)
    values = raw_padded.gather(-1, gather)
    values = values.masked_fill((relative < 0)[:, None, :], -torch.inf)
    return values.masked_fill((relative >= rel_extent)[:, None, :], 0.0)


def _prepare_relative_bias(
    raw_bias: torch.Tensor,
    shape: "_ForwardShapeInfo",
    cu_seqlens_q: torch.Tensor | None,
) -> torch.Tensor:
    if raw_bias.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("relative bias must have dtype BF16 or FP16")
    if not raw_bias.is_cuda or not raw_bias.is_contiguous():
        raise ValueError("relative bias must be contiguous on CUDA")
    rel_extent = raw_bias.shape[-1]
    if rel_extent < 128 or rel_extent % 128 != 0:
        raise ValueError("relative-bias extent must be a positive multiple of 128")
    q_offsets = [0]
    if cu_seqlens_q is not None:
        q_offsets = _metadata_values(cu_seqlens_q) or [0]
    if cu_seqlens_q is None:
        expected = (shape.batch, shape.seqlen_q, shape.num_heads_q, rel_extent)
        if raw_bias.shape != expected:
            raise ValueError(f"padded relative bias must have shape {expected}")
        padded_q = (shape.seqlen_q + 127) // 128 * 128
        output = torch.zeros(
            (shape.batch, padded_q, shape.num_heads_q, rel_extent + 256),
            dtype=raw_bias.dtype,
            device=raw_bias.device,
        )
    else:
        expected = (shape.total_q, shape.num_heads_q, rel_extent)
        if raw_bias.shape != expected:
            raise ValueError(f"packed relative bias must have shape {expected}")
        output = torch.zeros(
            (shape.total_q + 128, shape.num_heads_q, rel_extent + 256),
            dtype=raw_bias.dtype,
            device=raw_bias.device,
        )
    for batch_idx, (seqlen_q, seqlen_k) in enumerate(
        zip(shape.q_lengths, shape.k_lengths)
    ):
        q_offset = q_offsets[batch_idx] if cu_seqlens_q is not None else 0
        raw_one = (
            raw_bias[q_offset : q_offset + seqlen_q]
            if cu_seqlens_q is not None
            else raw_bias[batch_idx, :seqlen_q]
        )
        sheared = _shear_relative_bias_one(raw_one, seqlen_q, seqlen_k)
        if cu_seqlens_q is not None:
            output[q_offset : q_offset + seqlen_q] = sheared[:seqlen_q]
        else:
            output[batch_idx, :seqlen_q] = sheared[:seqlen_q]
    return output


def _metadata_values(tensor: torch.Tensor | None) -> list[int] | None:
    if tensor is None:
        return None
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _validate_cu_seqlens(
    tensor: torch.Tensor | None, name: str, batch: int, total: int
) -> list[int] | None:
    if tensor is None:
        return None
    _validate_cuda_int32_vector(tensor, name)
    values = _metadata_values(tensor)
    assert values is not None
    if len(values) != batch + 1 or values[0] != 0 or values[-1] != total:
        raise ValueError(f"{name} must be a complete prefix sum of the packed axis")
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    return values


def _validate_seqused(
    tensor: torch.Tensor | None,
    name: str,
    batch: int,
    capacities: list[int],
) -> list[int] | None:
    if tensor is None:
        return None
    _validate_cuda_int32_vector(tensor, name)
    values = _metadata_values(tensor)
    assert values is not None
    if len(values) != batch:
        raise ValueError(f"{name} must have shape ({batch},)")
    if any(
        length < 0 or length > capacity for length, capacity in zip(values, capacities)
    ):
        raise ValueError(f"{name} contains a length outside its storage capacity")
    return values


@dataclass(frozen=True)
class _ForwardShapeInfo:
    batch: int
    seqlen_q: int
    seqlen_k: int
    total_q: int
    total_k: int
    num_heads_q: int
    num_heads_k: int
    head_dim: int
    head_dim_v: int
    q_lengths: tuple[int, ...]
    k_lengths: tuple[int, ...]


def _query_storage_shape(
    q: torch.Tensor, cu_seqlens_q: torch.Tensor | None
) -> tuple[int, int, int, int, int]:
    if cu_seqlens_q is None:
        if q.ndim != 4:
            raise ValueError("padded Q must have shape (batch, seqlen, heads, dim)")
        batch, seqlen_q, num_heads_q, head_dim = q.shape
        return batch, seqlen_q, batch * seqlen_q, num_heads_q, head_dim
    if q.ndim != 3:
        raise ValueError("packed Q must have shape (total_q, heads, dim)")
    _validate_cuda_int32_vector(cu_seqlens_q, "cu_seqlens_q")
    total_q, num_heads_q, head_dim = q.shape
    return cu_seqlens_q.numel() - 1, total_q, total_q, num_heads_q, head_dim


def _kv_storage_shape(
    k: torch.Tensor, v: torch.Tensor, cu_seqlens_k: torch.Tensor | None
) -> tuple[int, int, int, int, int, int]:
    expected_rank = 4 if cu_seqlens_k is None else 3
    if k.ndim != expected_rank or v.ndim != expected_rank:
        mode = "padded" if cu_seqlens_k is None else "packed"
        raise ValueError(f"{mode} K and V must both have rank {expected_rank}")
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError("K and V must match in sequence and head dimensions")
    if cu_seqlens_k is not None:
        _validate_cuda_int32_vector(cu_seqlens_k, "cu_seqlens_k")
    batch = k.shape[0] if cu_seqlens_k is None else cu_seqlens_k.numel() - 1
    seqlen_k = k.shape[-3]
    total_k = batch * seqlen_k if cu_seqlens_k is None else seqlen_k
    return batch, seqlen_k, total_k, k.shape[-2], k.shape[-1], v.shape[-1]


def _normalize_forward_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_k: torch.Tensor | None,
) -> _ForwardShapeInfo:
    if not all(tensor.is_cuda for tensor in (q, k, v)):
        raise ValueError("Q, K, and V must be on CUDA")
    if not all(tensor.is_contiguous() for tensor in (q, k, v)):
        raise ValueError("Q, K, and V must be contiguous")
    q_shape = _query_storage_shape(q, cu_seqlens_q)
    kv_shape = _kv_storage_shape(k, v, cu_seqlens_k)
    batch, seqlen_q, total_q, num_heads_q, head_dim = q_shape
    batch_k, seqlen_k, total_k, num_heads_k, head_dim_k, head_dim_v = kv_shape
    if batch != batch_k:
        raise ValueError("Q and K must describe the same batch size")
    if head_dim != head_dim_k:
        raise ValueError("Q and K must have the same head dimension")
    if num_heads_q % num_heads_k != 0:
        raise ValueError("the number of Q heads must be divisible by KV heads")
    q_prefix = _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", batch, total_q)
    k_prefix = _validate_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", batch, total_k)
    q_capacities = (
        [seqlen_q] * batch
        if q_prefix is None
        else [b - a for a, b in zip(q_prefix, q_prefix[1:])]
    )
    k_capacities = (
        [seqlen_k] * batch
        if k_prefix is None
        else [b - a for a, b in zip(k_prefix, k_prefix[1:])]
    )
    q_used = _validate_seqused(seqused_q, "seqused_q", batch, q_capacities)
    k_used = _validate_seqused(seqused_k, "seqused_k", batch, k_capacities)
    q_lengths = tuple(q_capacities if q_used is None else q_used)
    k_lengths = tuple(k_capacities if k_used is None else k_used)
    return _ForwardShapeInfo(
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        total_q=total_q,
        total_k=total_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        q_lengths=q_lengths,
        k_lengths=k_lengths,
    )


def _validate_forward_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sfq: torch.Tensor | None,
    sfk: torch.Tensor | None,
    sfv: torch.Tensor | None,
    v_dequant: bool,
) -> tuple[bool, torch.dtype]:
    qk_blockscaled = sfq is not None
    if qk_blockscaled != (sfk is not None):
        raise ValueError("SFQ and SFK must either both be present or both be absent")
    if qk_blockscaled:
        if q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype:
            raise TypeError("block-scaled Q and K must both be float8_e4m3fn")
        if sfq.dtype != torch.float8_e8m0fnu or sfk.dtype != sfq.dtype:
            raise TypeError("SFQ and SFK must both be float8_e8m0fnu")
        if (
            not sfq.is_cuda
            or not sfk.is_cuda
            or not sfq.is_contiguous()
            or not sfk.is_contiguous()
        ):
            raise ValueError("SFQ and SFK must be contiguous CUDA tensors")
    elif q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        raise TypeError("Q and K must both be BF16 or FP16")
    if v_dequant:
        if v.dtype != torch.float8_e4m3fn:
            raise TypeError("SFV requires float8_e4m3fn V")
        if sfv is None or sfv.dtype != torch.float8_e8m0fnu:
            raise TypeError("FP8 V requires float8_e8m0fnu SFV")
        if not sfv.is_cuda or not sfv.is_contiguous():
            raise ValueError("SFV must be a contiguous CUDA tensor")
        output_dtype = torch.bfloat16
    else:
        if sfv is not None:
            raise ValueError("SFV unexpectedly selected the dense-V path")
        if v.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("V must be BF16/FP16 unless SFV selects FP8 storage")
        if not qk_blockscaled and v.dtype != q.dtype:
            raise TypeError("dense Q, K, and V must have the same dtype")
        output_dtype = v.dtype if qk_blockscaled else q.dtype
    return qk_blockscaled, output_dtype


def _torch_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_bias: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Compute causal relative-bias attention on CUDA with PyTorch."""
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    qf = q.float().permute(0, 2, 1, 3)
    kf = k.float().permute(0, 2, 1, 3)
    vf = v.float().permute(0, 2, 1, 3)
    if qf.shape[1] != kf.shape[1]:
        repeats = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(repeats, dim=1)
        vf = vf.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * softmax_scale

    q_idx = torch.arange(seqlen_q, device=q.device)[:, None]
    k_idx = torch.arange(seqlen_k, device=q.device)[None, :]
    rel_idx = q_idx + (seqlen_k - seqlen_q) - k_idx
    rel_extent = raw_bias.shape[-1]
    bias_idx = rel_idx.clamp(0, rel_extent - 1)[None, None, :, :]
    bias_idx = bias_idx.expand(q.shape[0], q.shape[2], -1, -1)
    bias = raw_bias.permute(0, 2, 1, 3).gather(dim=-1, index=bias_idx)
    valid_bias = (rel_idx >= 0) & (rel_idx < rel_extent)
    scores = scores + bias.masked_fill(~valid_bias[None, None, :, :], 0.0)

    causal_mask = k_idx > q_idx + (seqlen_k - seqlen_q)
    scores = scores.masked_fill(causal_mask[None, None, :, :], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, vf).permute(0, 2, 1, 3)


def _runner_reference_rows(
    inputs: dict[str, object],
    output: torch.Tensor,
    softmax_scale: float,
    reference_fn: Callable = _torch_reference_attention,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    raw_bias = inputs["rel_bias"]
    q_lengths = inputs["q_lengths"]
    k_lengths = inputs["k_lengths"]
    packed_q = inputs["packed_q"]
    packed_k = inputs["packed_k"]
    actual_rows, reference_rows = [], []
    q_offset = k_offset = 0
    for batch_idx, (seqlen_q, seqlen_k) in enumerate(zip(q_lengths, k_lengths)):
        q_one = (
            q[q_offset : q_offset + seqlen_q][None]
            if packed_q
            else q[batch_idx : batch_idx + 1, :seqlen_q]
        )
        k_one = (
            k[k_offset : k_offset + seqlen_k][None]
            if packed_k
            else k[batch_idx : batch_idx + 1, :seqlen_k]
        )
        v_one = (
            v[k_offset : k_offset + seqlen_k][None]
            if packed_k
            else v[batch_idx : batch_idx + 1, :seqlen_k]
        )
        bias_one = (
            raw_bias[q_offset : q_offset + seqlen_q][None]
            if packed_q
            else raw_bias[batch_idx : batch_idx + 1, :seqlen_q]
        )
        reference_rows.append(
            reference_fn(q_one, k_one, v_one, bias_one, softmax_scale)[0].clone()
        )
        actual_rows.append(
            output[q_offset : q_offset + seqlen_q]
            if packed_q
            else output[batch_idx, :seqlen_q]
        )
        q_offset += seqlen_q if packed_q else 0
        k_offset += seqlen_k if packed_k else 0
    return torch.cat(actual_rows), torch.cat(reference_rows)


_STANDALONE_FORWARD_CACHE: dict[tuple, Callable] = {}


def _allocate_forward_outputs(
    shape: _ForwardShapeInfo,
    packed_q: bool,
    output_dtype: torch.dtype,
    device: torch.device,
    return_lse: bool,
    return_logits_max: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    q_shape = (shape.total_q,) if packed_q else (shape.batch, shape.seqlen_q)
    output = torch.empty(
        (*q_shape, shape.num_heads_q, shape.head_dim_v),
        dtype=output_dtype,
        device=device,
    )
    lse_shape = (
        (shape.num_heads_q, shape.total_q)
        if packed_q
        else (shape.batch, shape.num_heads_q, shape.seqlen_q)
    )
    lse = (
        torch.empty(lse_shape, dtype=torch.float32, device=device)
        if return_lse
        else None
    )
    logits_max = (
        torch.empty(lse_shape, dtype=torch.float32, device=device)
        if return_logits_max
        else None
    )
    return output, lse, logits_max


def _resolve_source_window(
    causal: bool,
    window_size_left: int | None,
    window_size_right: int | None,
) -> tuple[bool, bool, int | None, int | None]:
    """Canonicalize causal/local settings like the source interface."""
    if causal:
        window_size_right = 0
    if (
        window_size_left is not None
        and window_size_right is not None
        and window_size_left + window_size_right < 0
    ):
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, is_local = True, False
            window_size_right = None
        else:
            causal, is_local = False, True
    else:
        is_local = False
    if window_size_left is not None and window_size_left < 0:
        raise ValueError("window_size_left must be nonnegative or part of (-1, -1)")
    if window_size_right is not None and window_size_right < 0:
        raise ValueError("window_size_right must be nonnegative or part of (-1, -1)")
    return causal, is_local, window_size_left, window_size_right


def _flash_attn_fwd_standalone(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rel_bias: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    pack_gqa: bool | None = None,
    num_splits: int = 1,
    return_lse: bool = False,
    return_logits_max: bool = False,
    zfill_padded_output: bool = True,
    out: torch.Tensor | None = None,
    sfq: torch.Tensor | None = None,
    sfk: torch.Tensor | None = None,
    sfv: torch.Tensor | None = None,
    v_dequant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run the unchanged SM100 kernel with source-interface varlen semantics."""
    if num_splits != 1:
        raise NotImplementedError(
            "the standalone wrapper currently supports num_splits=1"
        )
    if return_logits_max and not return_lse:
        raise ValueError("return_logits_max=True requires return_lse=True")
    shape = _normalize_forward_shapes(
        q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k
    )
    qk_blockscaled, output_dtype = _validate_forward_dtypes(
        q, k, v, sfq, sfk, sfv, v_dequant
    )
    if max_seqlen_q is None:
        max_seqlen_q = max(shape.q_lengths, default=0)
    if max_seqlen_k is None:
        max_seqlen_k = max(shape.k_lengths, default=0)
    if max_seqlen_q < max(shape.q_lengths, default=0):
        raise ValueError("max_seqlen_q is smaller than an active Q sequence")
    if max_seqlen_k < max(shape.k_lengths, default=0):
        raise ValueError("max_seqlen_k is smaller than an active K sequence")
    if max_seqlen_q < 1 or max_seqlen_k < 1:
        raise ValueError("at least one active Q and K token is required")
    if cu_seqlens_k is not None and (sfk is not None or sfv is not None):
        raise NotImplementedError(
            "packed K/V with SFK or SFV is unsupported by the unchanged kernel; "
            "use padded K/V with seqused_k"
        )
    head_ratio = shape.num_heads_q // shape.num_heads_k
    if pack_gqa is None:
        pack_gqa = head_ratio > 1 and 128 % head_ratio == 0
    if pack_gqa and 128 % head_ratio != 0:
        raise ValueError("pack_gqa requires the Q/KV head ratio to divide 128")
    causal, is_local, window_size_left, window_size_right = _resolve_source_window(
        causal, window_size_left, window_size_right
    )
    if rel_bias is not None and not causal:
        raise NotImplementedError(
            "the standalone host relative-bias shear supports full causal "
            "attention only; local relative-bias windows are unsupported"
        )
    is_varlen_q = cu_seqlens_q is not None or seqused_q is not None
    q_sf_interleaved = not is_varlen_q and not pack_gqa
    effective_q = max_seqlen_q * (head_ratio if pack_gqa else 1)
    q_stage = 1 if qk_blockscaled else (2 if effective_q > 128 else 1)
    bias_block_size = (effective_q + 7) // 8 * 8 if effective_q < 128 else 128
    softmax_scale = shape.head_dim**-0.5 if softmax_scale is None else softmax_scale
    allocated_out, lse, logits_max = _allocate_forward_outputs(
        shape,
        cu_seqlens_q is not None,
        output_dtype,
        q.device,
        return_lse,
        return_logits_max,
    )
    if out is None:
        out = allocated_out
    elif out.shape != allocated_out.shape or out.dtype != output_dtype:
        raise ValueError(
            f"out must have shape {tuple(allocated_out.shape)} and dtype {output_dtype}"
        )
    elif not out.is_cuda or not out.is_contiguous():
        raise ValueError("out must be a contiguous CUDA tensor")
    if zfill_padded_output and cu_seqlens_q is None and seqused_q is not None:
        out.zero_()
    bias = (
        _prepare_relative_bias(rel_bias, shape, cu_seqlens_q)
        if rel_bias is not None
        else None
    )
    if bias is not None and bias.dtype != output_dtype:
        raise TypeError("relative bias dtype must match the PV/output dtype")
    rel_extent_padded = bias.shape[-1] if bias is not None else 0
    m_q, m_k, m_v, m_o = map(_to_cute_tensor, (q, k, v, out))
    m_lse = _to_cute_tensor(lse, assumed_align=4) if lse is not None else None
    m_logits_max = (
        _to_cute_tensor(logits_max, assumed_align=4) if logits_max is not None else None
    )
    m_sfq = _to_cute_tensor(sfq, assumed_align=4) if sfq is not None else None
    m_sfk = _to_cute_tensor(sfk, assumed_align=4) if sfk is not None else None
    m_sfv = _to_cute_tensor(sfv, assumed_align=4) if sfv is not None else None
    metadata = (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    m_cu_q, m_cu_k, m_used_q, m_used_k = (
        _to_cute_tensor(tensor, assumed_align=4) if tensor is not None else None
        for tensor in metadata
    )
    m_bias = _to_cute_tensor(bias) if bias is not None else None
    stream = cutlass_torch.default_stream()
    kernel = FlashAttentionForwardSm100(
        head_dim=shape.head_dim,
        head_dim_v=shape.head_dim_v,
        qhead_per_kvhead=head_ratio,
        is_causal=causal,
        is_local=is_local,
        is_split_kv=False,
        pack_gqa=pack_gqa,
        m_block_size=128,
        n_block_size=128,
        bias_block_size=bias_block_size,
        q_stage=q_stage,
        is_persistent=False,
        is_dynamic_persistent_varlen=False,
        is_varlen_q=is_varlen_q,
        use_clc_scheduler=False,
        has_bias=bias is not None,
        rel_extent_padded=rel_extent_padded,
        qk_blockscaled=qk_blockscaled,
        v_dequant=v_dequant,
        q_sf_interleaved=q_sf_interleaved,
        kv_sf_interleaved=True,
    )
    compile_args = (
        m_q,
        m_k,
        m_v,
        m_o,
        m_lse,
        m_logits_max,
        softmax_scale,
        m_sfq,
        m_sfk,
        m_sfv,
        32 if qk_blockscaled else None,
        32 if v_dequant else None,
        m_cu_q,
        m_cu_k,
        m_used_q,
        m_used_k,
        None,
        window_size_left,
        window_size_right,
        None,
        None,
        None,
        m_bias,
        None,
        None,
        max_seqlen_q,
        stream,
    )
    cache_key = (
        q.dtype,
        k.dtype,
        v.dtype,
        output_dtype,
        q.ndim,
        k.ndim,
        shape.head_dim,
        shape.head_dim_v,
        head_ratio,
        causal,
        is_local,
        pack_gqa,
        q_stage,
        bias_block_size,
        return_lse,
        return_logits_max,
        qk_blockscaled,
        v_dequant,
        q_sf_interleaved,
        cu_seqlens_q is not None,
        cu_seqlens_k is not None,
        seqused_q is not None,
        seqused_k is not None,
        window_size_left is not None,
        window_size_right is not None,
        bias is not None,
        rel_extent_padded,
        sfq.ndim if sfq is not None else None,
        sfk.ndim if sfk is not None else None,
        sfv.ndim if sfv is not None else None,
    )
    compiled = _STANDALONE_FORWARD_CACHE.get(cache_key)
    if compiled is None:
        compiled = cute.compile(kernel, *compile_args)
        _STANDALONE_FORWARD_CACHE[cache_key] = compiled
    runtime_args = compile_args[:10] + compile_args[12:]
    compiled(*runtime_args)
    return out, lse, logits_max


def _reject_unsupported_wrapper_options(**options: object) -> None:
    unsupported = [name for name, value in options.items() if value is not None]
    if unsupported:
        names = ", ".join(unsupported)
        raise NotImplementedError(
            f"the standalone caller does not support these source options: {names}"
        )


def _validate_source_wrapper_options(
    *,
    window_size: tuple[int | None, int | None],
    softcap: float | None,
) -> None:
    if len(window_size) != 2:
        raise ValueError("window_size must contain (left, right)")
    if softcap not in (None, 0.0):
        raise NotImplementedError("the standalone caller does not support softcap")


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_bias: torch.Tensor | None = None,
    qv: torch.Tensor | None = None,
    gather_kv_indices: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: torch.Tensor | None = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    score_mod: Callable | None = None,
    score_mod_bwd: Callable | None = None,
    mask_mod: Callable | None = None,
    aux_tensors: list[torch.Tensor] | None = None,
    block_sparse_tensors: object | None = None,
    block_sparse_tensors_bwd: object | None = None,
    return_lse: bool = False,
    return_logits_max: bool = False,
    sfq: torch.Tensor | None = None,
    sfk: torch.Tensor | None = None,
    sfv: torch.Tensor | None = None,
):
    """Run fixed-length attention through the unchanged SM100 Prefill kernel.

    The argument order mirrors ``flash_attn.cute.interface.flash_attn_func``.
    This standalone caller implements dense Q/K/V, compact relative bias,
    causal/local masking, GQA packing, optional Q/K scale factors, and the
    source SFV-driven FP8-V path. Source options that need additional kernels
    or helper infrastructure are accepted for interface compatibility and fail
    clearly when selected.

    Args:
        q: Q tensor with shape ``(B, Sq, Hq, D)``.
        k: K tensor with shape ``(B, Sk, Hkv, D)``.
        v: V tensor with shape ``(B, Sk, Hkv, Dv)``. It must be BF16/FP16
            unless ``sfv`` is provided, in which case it must be FP8 E4M3.
        rel_bias: Optional compact relative bias ``(B, Sq, Hq, R)``. The
            standalone host shear supports it with full causal attention.
        qv: Optional source QV tensor; unsupported by this standalone caller.
        gather_kv_indices: Optional gathered-KV indices; unsupported here.
        softmax_scale: Score scale, defaulting to ``1 / sqrt(D)``.
        causal: Apply bottom-right causal masking when true.
        window_size: Optional ``(left, right)`` local-attention window.
        learnable_sink: Optional attention sink; unsupported here.
        softcap: Logit soft cap; only disabled values ``0.0`` and ``None`` are
            supported.
        num_splits: KV split count; currently must be one.
        pack_gqa: Pack grouped-query heads, or infer it when ``None``.
        deterministic: Source compatibility flag; one unsplit launch is
            deterministic with respect to split reduction.
        score_mod: Optional score modifier; unsupported here.
        score_mod_bwd: Backward-only source option, ignored by forward.
        mask_mod: Optional mask modifier; unsupported here.
        aux_tensors: Auxiliary score-mod tensors; unsupported here.
        block_sparse_tensors: Block-sparse metadata; unsupported here.
        block_sparse_tensors_bwd: Backward-only source option, ignored.
        return_lse: Allocate and return FP32 log-sum-exp values when true.
        return_logits_max: Return row-wise logit maxima as a third result.
        sfq: Optional UE8M0 Q scale-factor tensor.
        sfk: Optional UE8M0 K scale-factor tensor; required with ``sfq``.
        sfv: Optional UE8M0 V scale-factor tensor. Its presence selects FP8
            E4M3 V storage and BF16 dequantization before the BF16 PV MMA.

    Returns:
        ``(out, lse)`` normally, or ``(out, lse, logits_max)`` when
        ``return_logits_max=True``. ``lse`` is ``None`` unless requested.
    """
    del deterministic, score_mod_bwd, block_sparse_tensors_bwd
    _validate_source_wrapper_options(
        window_size=window_size,
        softcap=softcap,
    )
    _reject_unsupported_wrapper_options(
        qv=qv,
        gather_kv_indices=gather_kv_indices,
        learnable_sink=learnable_sink,
        score_mod=score_mod,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors or None,
        block_sparse_tensors=block_sparse_tensors,
    )
    out, lse, logits_max = _flash_attn_fwd_standalone(
        q,
        k,
        v,
        rel_bias=rel_bias,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        pack_gqa=pack_gqa,
        num_splits=num_splits,
        return_lse=return_lse,
        return_logits_max=return_logits_max,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        v_dequant=sfv is not None,
    )
    return (out, lse, logits_max) if return_logits_max else (out, lse)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_bias: torch.Tensor | None = None,
    qv: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    min_seqlen_k: int | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    gather_kv_indices: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: torch.Tensor | None = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    score_mod: Callable | None = None,
    score_mod_bwd: Callable | None = None,
    mask_mod: Callable | None = None,
    block_sparse_tensors: object | None = None,
    aux_tensors: list[torch.Tensor] | None = None,
    return_lse: bool = False,
    return_logits_max: bool = False,
    scheduler_metadata: object | None = None,
    seqlen_k_per_split: int | None = None,
    disable_scheduler_metadata: bool = False,
    zfill_padded_output: bool = True,
    sfq: torch.Tensor | None = None,
    sfk: torch.Tensor | None = None,
    sfv: torch.Tensor | None = None,
):
    """Run packed or padded variable-length SM100 Prefill attention.

    The argument order mirrors
    ``flash_attn.cute.interface.flash_attn_varlen_func``. Packed storage uses
    rank-3 tensors plus cumulative lengths; padded storage uses rank-4 tensors
    plus ``seqused_q/k``.

    Args:
        q: Packed ``(total_q, Hq, D)`` or padded ``(B, Sq, Hq, D)`` Q.
        k: Packed ``(total_k, Hkv, D)`` or padded ``(B, Sk, Hkv, D)`` K.
        v: K-shaped V with final dimension ``Dv``. FP8 requires ``sfv``.
        rel_bias: Packed ``(total_q, Hq, R)`` or padded
            ``(B, Sq, Hq, R)`` compact relative bias, supported with full
            causal attention.
        qv: Optional source QV tensor; unsupported here.
        cu_seqlens_q: CUDA int32 prefix sums of packed Q lengths.
        cu_seqlens_k: CUDA int32 prefix sums of packed K/V lengths.
        max_seqlen_q: Maximum active Q length, inferred when omitted.
        max_seqlen_k: Maximum active K length, inferred when omitted.
        min_seqlen_k: Optional source scheduler hint. It is validated but does
            not alter the static unsplit schedule.
        seqused_q: CUDA int32 active Q lengths for padded storage.
        seqused_k: CUDA int32 active K/V lengths for padded storage.
        gather_kv_indices: Optional gathered-KV indices; unsupported here.
        page_table: Optional paged-KV mapping; unsupported by this caller.
        softmax_scale: Score scale, defaulting to ``1 / sqrt(D)``.
        causal: Apply bottom-right causal masking when true.
        window_size: Optional ``(left, right)`` local-attention window.
        learnable_sink: Optional attention sink; unsupported here.
        softcap: Logit soft cap; only ``0.0`` is supported.
        num_splits: KV split count; currently must be one.
        pack_gqa: Pack grouped-query heads, or infer it when ``None``.
        deterministic: Source compatibility flag; forward ignores it.
        score_mod: Optional score modifier; unsupported here.
        score_mod_bwd: Backward-only source option, ignored by forward.
        mask_mod: Optional mask modifier; unsupported here.
        block_sparse_tensors: Block-sparse metadata; unsupported here.
        aux_tensors: Auxiliary score-mod tensors; unsupported here.
        return_lse: Allocate and return FP32 log-sum-exp values when true.
        return_logits_max: Return row-wise logit maxima as a third result.
        scheduler_metadata: Dynamic varlen scheduler metadata; unsupported.
        seqlen_k_per_split: Dynamic split scheduler hint; unsupported.
        disable_scheduler_metadata: Accepted source flag; the standalone
            wrapper always uses the static unsplit scheduler.
        zfill_padded_output: Zero inactive padded-Q rows before launch.
        sfq: Optional UE8M0 Q scale-factor tensor.
        sfk: Optional UE8M0 K scale-factor tensor; required with ``sfq``.
        sfv: Optional UE8M0 V scale-factor tensor. Its presence selects FP8 V
            storage, BF16 dequantization, and SFV application before PV.

    Returns:
        ``(out, lse)`` normally, or ``(out, lse, logits_max)`` when requested.
        Packed Q statistics have shape ``(Hq, total_q)``; padded statistics
        have shape ``(B, Hq, Sq)``.
    """
    del deterministic, score_mod_bwd, disable_scheduler_metadata
    _validate_source_wrapper_options(
        window_size=window_size,
        softcap=softcap,
    )
    if min_seqlen_k is not None and min_seqlen_k < 0:
        raise ValueError("min_seqlen_k must be nonnegative or None")
    _reject_unsupported_wrapper_options(
        qv=qv,
        gather_kv_indices=gather_kv_indices,
        page_table=page_table,
        learnable_sink=learnable_sink,
        score_mod=score_mod,
        mask_mod=mask_mod,
        block_sparse_tensors=block_sparse_tensors,
        aux_tensors=aux_tensors or None,
        scheduler_metadata=scheduler_metadata,
        seqlen_k_per_split=seqlen_k_per_split,
    )
    out, lse, logits_max = _flash_attn_fwd_standalone(
        q,
        k,
        v,
        rel_bias=rel_bias,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        pack_gqa=pack_gqa,
        num_splits=num_splits,
        return_lse=return_lse,
        return_logits_max=return_logits_max,
        zfill_padded_output=zfill_padded_output,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        v_dequant=sfv is not None,
    )
    return (out, lse, logits_max) if return_logits_max else (out, lse)


def _normalize_runner_shape(
    shape: tuple[int, int | tuple[int, ...], int, int], name: str
) -> tuple[int, tuple[int, ...], int, int, bool]:
    if len(shape) != 4:
        raise ValueError(f"{name} must have four entries: batch, sequence, heads, dim")
    batch, sequence, heads, head_dim = shape
    if batch < 1 or heads < 1 or head_dim < 1:
        raise ValueError(f"{name} batch, heads, and head dimension must be positive")
    is_varlen = isinstance(sequence, tuple)
    lengths = sequence if is_varlen else (sequence,) * batch
    if len(lengths) != batch:
        raise ValueError(f"{name} sequence tuple must contain exactly {batch} lengths")
    if any(length < 1 for length in lengths):
        raise ValueError(f"{name} sequence lengths must be positive")
    return batch, tuple(lengths), heads, head_dim, is_varlen


def _make_runner_inputs(
    q_shape: tuple[int, int | tuple[int, ...], int, int],
    k_shape: tuple[int, int | tuple[int, ...], int, int],
    rel_bias_extent: int,
    dtype: torch.dtype,
    seed: int,
    varlen_storage: str,
) -> dict[str, object]:
    q_info = _normalize_runner_shape(q_shape, "q_shape")
    k_info = _normalize_runner_shape(k_shape, "k_shape")
    batch, q_lengths, heads_q, head_dim, q_varlen = q_info
    batch_k, k_lengths, heads_k, head_dim_k, k_varlen = k_info
    if batch != batch_k or head_dim != head_dim_k or heads_q % heads_k != 0:
        raise ValueError(
            "Q/K batch and dimension must match, and Q heads must divide by KV heads"
        )
    if varlen_storage not in ("packed", "padded"):
        raise ValueError("varlen_storage must be 'packed' or 'padded'")
    packed_q = q_varlen and varlen_storage == "packed"
    packed_k = k_varlen and varlen_storage == "packed"
    q_physical = (
        (sum(q_lengths), heads_q, head_dim)
        if packed_q
        else (batch, max(q_lengths), heads_q, head_dim)
    )
    kv_physical = (
        (sum(k_lengths), heads_k, head_dim)
        if packed_k
        else (batch, max(k_lengths), heads_k, head_dim)
    )
    cu_q = _make_cu_seqlens(q_lengths) if packed_q else None
    cu_k = _make_cu_seqlens(k_lengths) if packed_k else None
    used_q = (
        torch.tensor(q_lengths, dtype=torch.int32, device="cuda")
        if q_varlen and not packed_q
        else None
    )
    used_k = (
        torch.tensor(k_lengths, dtype=torch.int32, device="cuda")
        if k_varlen and not packed_k
        else None
    )
    torch.manual_seed(seed)
    q = torch.randint(-2, 3, q_physical, device="cuda").to(dtype)
    k = torch.randint(-2, 3, kv_physical, device="cuda").to(dtype)
    v = torch.randint(-2, 3, kv_physical, device="cuda").to(dtype)
    bias_shape = (
        (sum(q_lengths), heads_q, rel_bias_extent)
        if packed_q
        else (batch, max(q_lengths), heads_q, rel_bias_extent)
    )
    rel_bias = torch.randint(-2, 3, bias_shape, device="cuda").to(dtype) * 0.25
    return {
        "q": q,
        "k": k,
        "v": v,
        "rel_bias": rel_bias,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "used_q": used_q,
        "used_k": used_k,
        "q_lengths": q_lengths,
        "k_lengths": k_lengths,
        "packed_q": packed_q,
        "packed_k": packed_k,
        "heads_q": heads_q,
        "heads_k": heads_k,
        "head_dim": head_dim,
    }


def run_standalone(
    batch: int = 1,
    seqlen: int = 128,
    heads: int = 1,
    head_dim: int = 128,
    rel_bias_extent: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 1111,
    atol: float = 0.1,
    rtol: float = 1e-5,
    check_reference: bool = True,
    *,
    q_shape: tuple[int, int | tuple[int, ...], int, int] | None = None,
    k_shape: tuple[int, int | tuple[int, ...], int, int] | None = None,
    varlen_storage: str = "packed",
) -> float:
    """Compile, run, and verify fixed or variable-length causal attention.

    Args:
        batch: Batch size.
        seqlen: Shared query/key sequence length.
        heads: Number of query, key, and value heads.
        head_dim: Per-head Q/K/V dimension; supported values are 64, 96, and 128.
        rel_bias_extent: Compact relative-bias extent, in multiples of 128.
        dtype: Input, bias, and output dtype (`torch.bfloat16` or `torch.float16`).
        seed: CUDA random seed used for deterministic inputs.
        atol: Absolute tolerance for output verification.
        rtol: Relative tolerance for output verification.
        check_reference: Run and compare against the PyTorch reference when true.
        q_shape: Optional `(B, S_q, H_q, D)` shape; `S_q` may be a length tuple.
        k_shape: Optional `(B, S_k, H_kv, D)` shape; `S_k` may be a length tuple.
        varlen_storage: Use `packed` cumulative lengths or padded `seqused` storage.

    Returns:
        The maximum absolute output error, or NaN when reference checking is disabled.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] not in (10, 11):
        raise RuntimeError(f"SM100/SM110 GPU required, found capability {capability}")
    if q_shape is None:
        q_shape = (batch, seqlen, heads, head_dim)
    if k_shape is None:
        k_shape = q_shape
    q_shape_info = _normalize_runner_shape(q_shape, "q_shape")
    k_shape_info = _normalize_runner_shape(k_shape, "k_shape")
    if q_shape_info[3] not in (64, 96, 128):
        raise ValueError("head dimension must be one of 64, 96, or 128")
    if rel_bias_extent < 128 or rel_bias_extent % 128 != 0:
        raise ValueError("rel_bias_extent must be a positive multiple of 128")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("dtype must be torch.bfloat16 or torch.float16")

    inputs = _make_runner_inputs(
        q_shape,
        k_shape,
        rel_bias_extent,
        dtype,
        seed,
        varlen_storage,
    )
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    raw_bias = inputs["rel_bias"]
    any_varlen = q_shape_info[4] or k_shape_info[4]
    softmax_scale = 1.0 / math.sqrt(q_shape_info[3])
    print("Compiling and running the standalone FlashAttention wrapper ...")
    compile_start = time.perf_counter()
    if any_varlen:
        output, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            rel_bias=raw_bias,
            cu_seqlens_q=inputs["cu_q"],
            cu_seqlens_k=inputs["cu_k"],
            seqused_q=inputs["used_q"],
            seqused_k=inputs["used_k"],
            max_seqlen_q=max(inputs["q_lengths"]),
            max_seqlen_k=max(inputs["k_lengths"]),
            softmax_scale=softmax_scale,
            causal=True,
            pack_gqa=inputs["heads_q"] != inputs["heads_k"],
        )
    else:
        output, _ = flash_attn_func(
            q,
            k,
            v,
            rel_bias=raw_bias,
            softmax_scale=softmax_scale,
            causal=True,
            pack_gqa=inputs["heads_q"] != inputs["heads_k"],
        )
    compilation_seconds = time.perf_counter() - compile_start
    torch.cuda.synchronize()
    print(f"Compilation time: {compilation_seconds:.3f} s")

    if not check_reference:
        print("PASS (reference check skipped)")
        return float("nan")

    import torch._inductor.config as inductor_config

    with inductor_config.patch(
        {
            "triton.cudagraphs": False,
            "max_autotune_gemm_backends": "ATEN",
        }
    ):
        reference_fn = torch.compile(
            _torch_reference_attention,
            mode="max-autotune",
            dynamic=True,
        )
        actual, reference = _runner_reference_rows(
            inputs, output, softmax_scale, reference_fn
        )
    torch.cuda.synchronize()
    output_fp32 = actual.float()
    max_error = (output_fp32 - reference).abs().max().item()
    torch.testing.assert_close(
        output_fp32,
        reference,
        atol=atol,
        rtol=rtol,
    )
    print(f"PASS: output matches PyTorch reference (max abs error {max_error:.6f})")
    return max_error


def _parse_nested_shape(value: str):
    """Parse `B,S,H,D` or `B,(s0,...,sB-1),H,D`."""
    try:
        value = value.strip()
        if "(" not in value:
            return tuple(int(item.strip()) for item in value.split(","))
        start, end = value.find("("), value.find(")")
        if start < 0 or end < start:
            raise ValueError("mismatched parentheses")
        before = value[:start].strip().rstrip(",")
        middle = value[start + 1 : end].strip()
        after = value[end + 1 :].strip().lstrip(",")
        result = [int(item.strip()) for item in before.split(",") if item.strip()]
        result.append(tuple(int(item.strip()) for item in middle.split(",")))
        result.extend(int(item.strip()) for item in after.split(",") if item.strip())
        return tuple(result)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid attention shape: {value}") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and verify the standalone SM100 causal relative-bias kernel."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument(
        "--head-dim",
        type=int,
        choices=(64, 96, 128),
        default=128,
    )
    parser.add_argument(
        "--q-shape",
        "--q_shape",
        type=_parse_nested_shape,
        default=None,
        help="Q shape B,S,H,D or varlen B,(s0,...),H,D",
    )
    parser.add_argument(
        "--k-shape",
        "--k_shape",
        type=_parse_nested_shape,
        default=None,
        help="K/V shape B,S,Hkv,D or varlen B,(s0,...),Hkv,D",
    )
    parser.add_argument(
        "--varlen-storage",
        choices=("packed", "padded"),
        default="packed",
        help="Represent tuple sequence lengths with cu_seqlens or seqused",
    )
    parser.add_argument("--rel-bias-extent", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--atol", type=float, default=0.1)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Compile and launch once without running the PyTorch reference.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    run_standalone(
        batch=args.batch,
        seqlen=args.seqlen,
        heads=args.heads,
        head_dim=args.head_dim,
        q_shape=args.q_shape,
        k_shape=args.k_shape,
        varlen_storage=args.varlen_storage,
        rel_bias_extent=args.rel_bias_extent,
        dtype=dtype,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        check_reference=not args.skip_reference,
    )


if __name__ == "__main__":
    _main()
