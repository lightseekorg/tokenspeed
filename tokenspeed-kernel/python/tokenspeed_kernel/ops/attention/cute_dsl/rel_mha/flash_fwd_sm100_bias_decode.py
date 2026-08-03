# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Standalone SM100 dense or paged GQA decode with relative-position bias.

This decode dataflow is derived from CUTLASS's Blackwell ``gqa_decode.py``:
the first GEMM is K @ Q, the query/prediction dimension is packed with grouped
query heads, and long KV sequences can be split across CTAs.  Relative bias is
fused into the online softmax in log2 space, before the per-split max/sum are
reduced.  The bias uses the same bottom-right causal convention as the prefill
runner in ``flash_fwd_sm100_bias.py``.  An optional left sliding window culls
inactive KV tiles and applies exact causal/window masks at tile boundaries.
Paged KV maps logical sequence pages to a contiguous physical-page pool and
uses page-sized TMA loads to reconstruct each MMA tile in shared memory.
"""

import argparse
import math
from functools import partial
from typing import NamedTuple, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import testing
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
)
from cutlass.pipeline import NamedBarrier as nbar

for _compat_name in ("ThrMma", "ThrCopy"):
    if not hasattr(cute.core, _compat_name):
        setattr(cute.core, _compat_name, getattr(cute, _compat_name))
if not hasattr(cute, "make_fragment"):
    cute.make_fragment = cute.make_rmem_tensor

from cutlass.cute.typing import (
    BFloat16,
    Float8E4M3FN,
    Float8E5M2,
    Float8E8M0FNU,
    Float16,
    Float32,
    Int32,
    Int64,
    Literal,
    Optional,
)
from quack import copy_utils

# Kernel invariants
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)  # (MMA, #MMA_M, #MMA_K)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
max_reduction_iters = 4  # log2(16)

# Math helpers
log2_e = math.log2(math.e)  # change exponential base
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")


class SwaDecodePlan(NamedTuple):
    """Host-side launch plan for fixed-length sliding-window Decode."""

    kv_splits: int
    reduction: str
    base_ctas: int
    active_tiles_per_query_tile: Tuple[int, ...]
    direct_tile_limit: int
    candidates: Tuple[int, ...]


def choose_swa_prediction_tile(
    prediction: int,
    grouped_head_tile: int,
    window_size_left: int,
    bf16_eager: bool = True,
) -> int:
    """Choose query packing before planning a fixed-length SWA launch.

    Dense BF16 W512 sweeps show that packing 32 output rows into one CTA can
    serialize multi-token decode badly enough that split-K cannot recover the
    lost parallelism.  Keep roughly eight query/head rows per CTA for this
    short-window eager path.  Other dtypes and larger windows retain the
    established maximum packing.

    Args:
        prediction: Query sequence length.
        grouped_head_tile: Rounded query heads per KV head in one CTA.
        window_size_left: Inclusive causal window left extent.
        bf16_eager: Apply the BF16 eager W512 packing model when true.

    Returns:
        A power-of-two query tile supported by the kernel.
    """
    if prediction <= 0 or grouped_head_tile <= 0:
        raise ValueError("prediction and grouped_head_tile must be positive")
    if grouped_head_tile & (grouped_head_tile - 1):
        raise ValueError("grouped_head_tile must be a power of two")
    if grouped_head_tile > 32:
        raise ValueError("grouped_head_tile must not exceed 32")
    if window_size_left < 0:
        raise ValueError("window_size_left must be nonnegative")

    prediction_tile = 1 << (prediction - 1).bit_length()
    prediction_tile = min(32 // grouped_head_tile, prediction_tile)
    if bf16_eager and window_size_left < 512:
        target_output_rows = 8
        prediction_tile = min(
            prediction_tile,
            max(1, target_output_rows // grouped_head_tile),
        )
    return prediction_tile


def plan_swa_decode(
    batches: int,
    prediction: int,
    seqlen: int,
    heads_q: int,
    heads_k: int,
    grouped_head_tile: int,
    prediction_tile: int,
    sequence_tile: int,
    head_dim: int,
    window_size_left: int,
    sm_count: int,
    bf16_eager: bool = True,
) -> SwaDecodePlan:
    """Choose a shape-aware split count for fixed-length SWA Decode.

    The planner models each packed query tile separately, caps the split count
    at the number of local KV tiles that can do useful work, and accounts for
    both CTA waves and the eager-call cost of a second deterministic launch.
    It deliberately applies only to SWA; full attention retains its existing
    split policy and reduction path.

    Args:
        batches: Batch size.
        prediction: Query sequence length.
        seqlen: KV sequence length.
        heads_q: Number of query heads.
        heads_k: Number of KV heads.
        grouped_head_tile: Query heads per KV head packed into one CTA.
        prediction_tile: Query tokens packed into one CTA.
        sequence_tile: KV tokens consumed by one loop iteration.
        head_dim: Q/K/V head dimension.
        window_size_left: Inclusive causal window left extent.
        sm_count: Number of streaming multiprocessors on the target GPU.
        bf16_eager: Use the BF16 direct-call crossover model when true.

    Returns:
        A :class:`SwaDecodePlan`. ``candidates`` is a small safe set for an
        optional empirical autotuner; ``kv_splits`` is the static model choice.
    """
    if (
        min(
            batches,
            prediction,
            seqlen,
            heads_q,
            heads_k,
            grouped_head_tile,
            prediction_tile,
            sequence_tile,
            head_dim,
            sm_count,
        )
        <= 0
    ):
        raise ValueError("SWA Decode dimensions, tiles, and SM count must be positive")
    if seqlen < prediction:
        raise ValueError("KV sequence length must be at least the query length")
    if heads_q % heads_k != 0:
        raise ValueError("heads_q must be divisible by heads_k")
    if grouped_head_tile & (grouped_head_tile - 1):
        raise ValueError("grouped_head_tile must be a power of two")
    if prediction_tile & (prediction_tile - 1):
        raise ValueError("prediction_tile must be a power of two")
    if grouped_head_tile * prediction_tile > 32:
        raise ValueError("grouped_head_tile * prediction_tile must not exceed 32")
    if head_dim % 64:
        raise ValueError("head_dim must be a multiple of 64")
    if window_size_left < 0:
        raise ValueError("window_size_left must be nonnegative")

    grouped_heads = heads_q // heads_k
    grouped_head_tiles = math.ceil(grouped_heads / grouped_head_tile)
    query_tiles = math.ceil(prediction / prediction_tile)
    base_ctas = batches * heads_k * grouped_head_tiles * query_tiles

    active_tiles = []
    for query_tile in range(query_tiles):
        query_begin = query_tile * prediction_tile
        query_end = min(query_begin + prediction_tile, prediction)
        key_begin = max(
            seqlen - prediction + query_begin - window_size_left,
            0,
        )
        key_end = min(seqlen - prediction + query_end, seqlen)
        tile_begin = key_begin // sequence_tile
        tile_end = math.ceil(key_end / sequence_tile)
        active_tiles.append(max(1, tile_end - tile_begin))

    max_active_tiles = max(active_tiles)
    max_parallel_splits = min(max_active_tiles, max(1, sm_count // base_ctas))

    # Direct-call B200 sweeps show that the one-launch crossover scales with D,
    # packed GQA width, and prediction tiling.  Larger grouped-head tiles remain
    # efficient, so their penalty grows only every two powers of two.  W512
    # query packing is also efficient after capping each CTA near eight output
    # rows, so its prediction penalty grows at the same sublinear rate.
    direct_tile_limit = 4
    if bf16_eager:
        grouped_tile_log2 = grouped_head_tile.bit_length() - 1
        grouped_tile_penalty = 1 << max(0, (grouped_tile_log2 - 1) // 2)
        prediction_tile_penalty = prediction_tile
        if window_size_left < 512:
            prediction_tile_log2 = prediction_tile.bit_length() - 1
            prediction_tile_penalty = 1 << max(0, (prediction_tile_log2 - 1) // 2)
        direct_limit_raw = (16 * 128) // (
            head_dim * prediction_tile_penalty * grouped_tile_penalty
        )
        direct_tile_limit = max(1, min(16, direct_limit_raw))
        direct_tile_limit = 1 << (direct_tile_limit.bit_length() - 1)
        if window_size_left < 512:
            # W512 spans at most six 128-token tiles after packed-query and
            # alignment expansion.  Once s1 writes final output directly,
            # measured s2/s4 gains are below the 5%
            # tuning threshold even for the widest D256 MQA CTA; the second
            # deterministic launch otherwise regresses common shapes.
            direct_tile_limit = max(8, direct_tile_limit)
    if max_active_tiles <= direct_tile_limit:
        modeled_splits = 1
    else:
        useful_split_cap = min(
            max_active_tiles,
            max_parallel_splits,
            max(8, math.ceil(max_active_tiles / 2)),
        )
        modeled_splits = 1 << (useful_split_cap.bit_length() - 1)

    candidate_set = {1, modeled_splits, max_parallel_splits}
    candidate_set.update(active_tiles)
    power_of_two = 1
    while power_of_two <= max_parallel_splits:
        candidate_set.add(power_of_two)
        power_of_two *= 2
    for neighbor in (modeled_splits - 1, modeled_splits + 1):
        if 1 <= neighbor <= max_parallel_splits:
            candidate_set.add(neighbor)
    candidates = tuple(
        sorted(split for split in candidate_set if split <= max_active_tiles)
    )
    return SwaDecodePlan(
        modeled_splits,
        "direct" if modeled_splits == 1 else "kernel",
        base_ctas,
        tuple(active_tiles),
        direct_tile_limit,
        candidates,
    )


@dsl_user_op
def cvt_bf16x2_ue8m0x2(
    packed_scales: cutlass.Int16, *, loc=None, ip=None
) -> cutlass.Int32:
    """Convert two packed UE8M0 scale factors to packed BF16."""
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
def cvt_tensor_ue8m0_to_bf16(scales: cute.Tensor, output: cute.Tensor):
    """Convert an even-sized UE8M0 register fragment to BF16."""
    count = cute.size(scales)
    assert count % 2 == 0 and cute.size(output) == count
    scales_x2 = cute.recast_tensor(scales, dtype=cutlass.Int16)
    output_x2 = cute.recast_tensor(output, dtype=cutlass.Int32)
    for index in cutlass.range_constexpr(count // 2):
        output_x2[index] = cvt_bf16x2_ue8m0x2(scales_x2[index])


def create_mxfp8_scale_factor_tensor(
    mn, k, l, sf_vec_size=32, device="cuda", pattern="random", phase=0
):
    """Create UE8M0 blocked storage plus an elementwise FP32 reference view."""
    atom_m, atom_k = 128, 4
    m_atoms = math.ceil(mn / atom_m)
    sf_k = math.ceil(k / sf_vec_size)
    sf_k_atoms = math.ceil(sf_k / atom_k)
    mn_padded = m_atoms * atom_m
    sf_k_padded = sf_k_atoms * atom_k
    choices = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32, device=device)
    if pattern == "random":
        indices = torch.randint(
            0, len(choices), (mn_padded, sf_k_padded, l), device=device
        )
    elif pattern == "unit":
        indices = torch.ones(
            (mn_padded, sf_k_padded, l), dtype=torch.long, device=device
        )
    elif pattern == "row":
        m_idx = torch.arange(mn_padded, device=device)[:, None, None]
        indices = m_idx.expand(-1, sf_k_padded, l) % len(choices)
    elif pattern == "column":
        k_idx = torch.arange(sf_k_padded, device=device)[None, :, None]
        indices = k_idx.expand(mn_padded, -1, l) % len(choices)
    else:
        m_idx = torch.arange(mn_padded, device=device)[:, None, None]
        k_idx = torch.arange(sf_k_padded, device=device)[None, :, None]
        l_idx = torch.arange(l, device=device)[None, None, :]
        indices = (m_idx + 2 * k_idx + (phase + 1) * l_idx + phase) % len(choices)
    logical_padded = choices[indices]
    storage = (
        logical_padded.reshape(m_atoms, 4, 32, sf_k_atoms, atom_k, l)
        .permute(5, 0, 3, 2, 1, 4)
        .contiguous()
    )
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
        storage,
        Float8E8M0FNU,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    logical = logical_padded[:mn, :sf_k]
    elementwise = (
        logical.unsqueeze(2)
        .expand(-1, -1, sf_vec_size, -1)
        .reshape(mn, sf_k * sf_vec_size, l)[:, :k]
        .contiguous()
    )
    return elementwise, cute_tensor, torch_tensor


def _paged_cache_plan(batches, seqlen, page_size, seed):
    """Return the fixed-length logical-to-physical page mapping used by the CLI."""
    pages_per_batch = math.ceil(seqlen / page_size)
    physical_pages = batches * pages_per_batch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    page_table = torch.randperm(physical_pages, generator=generator).tolist()
    if physical_pages > 1 and all(
        logical == physical for logical, physical in enumerate(page_table)
    ):
        page_table = page_table[1:] + page_table[:1]
    return {
        "pages_per_batch": pages_per_batch,
        "padded_seqlen": pages_per_batch * page_size,
        "physical_pages": physical_pages,
        "page_table": tuple(page_table),
        "table_offsets": tuple(batch * pages_per_batch for batch in range(batches)),
    }


def _scatter_paged_pages(logical_pages, page_table, table_offsets, fill_value=31.0):
    """Scatter ``[batch, logical_page, ...]`` storage into a physical page pool."""
    physical = torch.full(
        (len(page_table), *logical_pages.shape[2:]),
        fill_value,
        dtype=logical_pages.dtype,
        device=logical_pages.device,
    )
    for batch, table_offset in enumerate(table_offsets):
        for logical_page in range(logical_pages.shape[1]):
            physical_page = page_table[table_offset + logical_page]
            physical[physical_page].copy_(logical_pages[batch, logical_page])
    return physical


def _paged_scale_values(shape, pattern, phase, device, operand, page_size):
    """Create page-local FP32 scale values before conversion to UE8M0 bytes."""
    choices = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32, device=device)
    if pattern == "random":
        indices = torch.randint(0, len(choices), shape, device=device)
    elif pattern == "unit":
        indices = torch.ones(shape, dtype=torch.long, device=device)
    else:
        coordinates = []
        for axis, extent in enumerate(shape):
            view_shape = [1] * len(shape)
            view_shape[axis] = extent
            coordinates.append(torch.arange(extent, device=device).view(view_shape))
        batch, page, vector, head, feature = coordinates
        layer = batch * shape[3] + head
        if operand == "k":
            row = page * page_size + vector
            column = feature
        elif operand == "v":
            row = feature
            column = (page * page_size + vector * 32) // 32
        else:
            raise ValueError("operand must be 'k' or 'v'")
        if pattern == "row":
            indices = row
        elif pattern == "column":
            indices = column
        else:
            indices = row + 2 * column + (phase + 1) * layer + phase
        indices = indices.expand(shape) % len(choices)
    return choices[indices.long()]


def _to_cute_host_tensor(tensor, assumed_align=16):
    """Wrap a CUDA tensor while preserving its physical strides."""
    return from_dlpack(tensor, assumed_align=assumed_align).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


class FlashAttentionDecodeSm100Bias:
    """SM100 Decode kernel with dense, MXFP8-QK, and paged-KV modes.

    Args:
        headdim: Q/K/V head dimension.
        grouped_head_tile: Rounded number of query heads per KV head in a CTA.
        prediction_tile: Rounded query-token tile size.
        sequence_tile: Logical KV-token tile size.
        reduction_mode: Direct, deterministic workspace, or atomic reduction.
        rel_bias_layout: ``compact`` for (B, Sq, Hq, R), or ``sheared`` for
            Prefill ShearingBias output (B, round_up(Sq, 128), Hq, R + 256).
        qk_blockscaled: Store Q/K as FP8 plus UE8M0 and use block-scaled GEMM1.
        qk_sf_vec_size: Q/K scale vector width; currently 32.
        v_dequant: Static V-storage gate. When false, V must be BF16/FP16 and
            v_sf must be absent. When true, V must be FP8 with UE8M0 v_sf and
            is dequantized to v_mma_dtype before GEMM2.
        v_sf_vec_size: V scale vector width; currently 32.
        v_mma_dtype: BF16 or FP16 dtype used by V/P GEMM2 after dequantization.
        window_size_left: Optional causal sliding-window left extent.
        page_size: Optional physical KV page size (8, 16, 32, 64, 128, or 256).
            Page sizes 128 and 256 are supported for dense BF16/FP16 Q/K/V only.
    """

    def __init__(
        self,
        headdim,
        grouped_head_tile,  # Grouped heads per threadblock (GQA packing factor)
        prediction_tile=1,  # Predicted tokens per threadblock
        sequence_tile=256,  # KV tokens per threadblock per loop iteration
        reduction_mode: Literal[  # split-K reduction algorithm
            "direct",  # Single split, normalized output written directly
            "kernel",  # Deterministic kernel reduction with partial result workspace
            "atomic",  # Cluster reduction with atomic adds, no workspace
        ] = "kernel",
        *,
        rel_bias_layout: Literal["compact", "sheared"] = "compact",
        qk_blockscaled=False,
        qk_sf_vec_size=32,
        v_dequant=False,
        v_sf_vec_size=32,
        v_mma_dtype=BFloat16,
        window_size_left=None,
        page_size=None,
        use_pdl: bool = True,
    ):
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.prediction_tile = prediction_tile
        self.sequence_tile = sequence_tile
        self.do_direct_red = reduction_mode == "direct"
        self.do_kernel_red = reduction_mode == "kernel"
        self.do_atomic_red = reduction_mode == "atomic"
        self.do_cluster_red = self.do_direct_red or self.do_atomic_red
        self.rel_bias_layout = rel_bias_layout
        self.bias_is_sheared = rel_bias_layout == "sheared"
        self.qk_blockscaled = qk_blockscaled
        self.qk_sf_vec_size = qk_sf_vec_size
        self.v_dequant = v_dequant
        self.v_sf_vec_size = v_sf_vec_size
        self.v_mma_dtype = v_mma_dtype
        self.window_size_left = window_size_left
        self.page_size = page_size
        self.use_pdl = use_pdl
        self.is_paged_kv = page_size is not None
        self.threads_per_cta = 4 * warpgroup_threads

        assert headdim > 0 and headdim % 64 == 0
        assert grouped_head_tile * prediction_tile in (1, 2, 4, 8, 16, 32)
        assert sequence_tile > 0 and sequence_tile % 128 == 0
        assert sum((self.do_direct_red, self.do_kernel_red, self.do_atomic_red)) == 1
        assert rel_bias_layout in ("compact", "sheared")
        assert not qk_blockscaled or qk_sf_vec_size == 32
        assert not v_dequant or v_sf_vec_size == 32
        assert not v_dequant or v_mma_dtype in (BFloat16, Float16)
        assert window_size_left is None or window_size_left >= 0
        assert page_size is None or page_size in (8, 16, 32, 64, 128, 256)
        assert page_size not in (128, 256) or not (qk_blockscaled or v_dequant)
        assert page_size is None or sequence_tile % page_size == 0
        assert page_size is None or 128 % page_size == 0 or page_size % 128 == 0

    ##############################
    # Launch helpers
    ##############################
    # Runtime implementable check
    def can_implement(
        self,
        kv_splits,
        qo_shape_bshd,
        kv_shape_bshd,
        bias_shape_bshr,
        qk_dtype,
        v_dtype,
        o_dtype,
        sf_dtype=None,
        v_sf_dtype=None,
        v_shape_bshd=None,
        v_sf_shape=None,
        k_sf_shape=None,
        k_sf_stride=None,
    ):
        """Validate a proposed Decode launch configuration.

        Args:
            kv_splits: Number of sequence splits in the launch grid.
            qo_shape_bshd: Logical Q/O shape (B, Sq, Hq, D).
            kv_shape_bshd: Logical dense K shape or physical paged-K shape.
            bias_shape_bshr: Optional compact shape (B, Sq, Hq, R), or
                Prefill-sheared shape (B, round_up(Sq, 128), Hq, R + 256).
                ``None`` selects the static no-relative-bias specialization.
            qk_dtype: Q/K storage dtype.
            v_dtype: V storage dtype.
            o_dtype: Output storage dtype.
            sf_dtype: Optional Q/K scale-factor dtype.
            v_sf_dtype: Optional V scale-factor dtype.
            v_shape_bshd: Optional V shape, checked against K when supplied.
            v_sf_shape: Optional paged SFV shape. FP8 paged V uses
                (physical_pages, ceil(page_size / 32), Hkv, D).
            k_sf_shape: Optional compact paged SFK shape. MXFP8 paged K uses
                (physical_pages, page_size, Hkv, ceil(D / 32)).
            k_sf_stride: Element strides for compact paged SFK. Head planes
                must be contiguous and 16-byte aligned for one-dimensional TMA.

        Returns:
            None. Invalid configurations raise TypeError or ValueError.
        """
        if v_shape_bshd is not None and tuple(v_shape_bshd) != tuple(kv_shape_bshd):
            raise ValueError(
                f"K shape {kv_shape_bshd} and V shape {v_shape_bshd} must match"
            )
        b_q, s_q, h_q, d_q = qo_shape_bshd
        if self.is_paged_kv:
            physical_pages, page_size, h_k, d_k = kv_shape_bshd
            if physical_pages <= 0:
                raise ValueError("paged KV requires at least one physical page")
            if page_size != self.page_size:
                raise ValueError(
                    f"K/V page dimension({page_size}) must match page_size({self.page_size})"
                )
            s_k = None
        else:
            b_k, s_k, h_k, d_k = kv_shape_bshd

        if qk_dtype is cutlass.Float8E4M3:
            raise TypeError("use Float8E4M3FN instead of Float8E4M3")
        if not self.v_dequant and v_dtype not in (BFloat16, Float16):
            raise TypeError(
                "v_dequant=False requires BF16 or FP16 V; "
                "set v_dequant=True for FP8 V plus SFV"
            )
        if not self.v_dequant and v_sf_dtype is not None:
            raise TypeError("v_dequant=False requires v_sf=None")

        if self.qk_blockscaled:
            if self.headdim != 128:
                raise ValueError("MXFP8 decode currently requires headdim=128")
            if qk_dtype not in (Float8E4M3FN, Float8E5M2):
                raise TypeError("MXFP8 Q/K must use Float8E4M3FN or Float8E5M2")
            if sf_dtype is not Float8E8M0FNU:
                raise TypeError("MXFP8 scale factors must use Float8E8M0FNU")
            if not self.v_dequant and v_dtype not in (BFloat16, Float16):
                raise TypeError("MXFP8 decode requires BF16 or FP16 V/P")
            grouped_heads = h_q // h_k
            if grouped_heads > self.grouped_head_tile or s_q > self.prediction_tile:
                raise ValueError(
                    "MXFP8 currently supports one packed query tile per CTA"
                )
        elif not self.v_dequant and qk_dtype != v_dtype:
            raise TypeError("dense Q/K/V dtypes must match")

        if self.v_dequant:
            if v_dtype not in (Float8E4M3FN, Float8E5M2):
                raise TypeError("FP8 V must use Float8E4M3FN or Float8E5M2")
            if v_sf_dtype is not Float8E8M0FNU:
                raise TypeError("FP8 V scale factors must use Float8E8M0FNU")

        if self.is_paged_kv:
            if not self.qk_blockscaled and qk_dtype not in (BFloat16, Float16):
                raise TypeError("paged dense K must use BF16 or FP16")
            if self.qk_blockscaled and k_sf_shape is None:
                raise ValueError("paged MXFP8 K requires compact page-local SFK")
            if not self.qk_blockscaled and k_sf_shape is not None:
                raise ValueError("paged BF16/FP16 K requires k_sf=None")
            if self.qk_blockscaled:
                expected_k_sf_shape = (
                    physical_pages,
                    self.page_size,
                    h_k,
                    math.ceil(d_k / self.qk_sf_vec_size),
                )
                if tuple(k_sf_shape) != expected_k_sf_shape:
                    raise ValueError(
                        "paged SFK shape must be "
                        f"{expected_k_sf_shape}, got {k_sf_shape}"
                    )
                groups = expected_k_sf_shape[-1]
                expected_k_sf_stride = (
                    self.page_size * h_k * groups,
                    groups,
                    self.page_size * groups,
                    1,
                )
                if k_sf_stride is None or tuple(k_sf_stride) != expected_k_sf_stride:
                    raise ValueError(
                        "paged SFK must use compact head-major backing with strides "
                        f"{expected_k_sf_stride}, got {k_sf_stride}"
                    )
            if not self.qk_blockscaled and not self.v_dequant and v_dtype != qk_dtype:
                raise TypeError("paged dense V must match the Q/K dtype")
            if self.v_dequant and v_sf_shape is not None:
                expected_v_sf_shape = (
                    physical_pages,
                    math.ceil(self.page_size / self.v_sf_vec_size),
                    h_k,
                    d_k,
                )
                if tuple(v_sf_shape) != expected_v_sf_shape:
                    raise ValueError(
                        "paged SFV shape must be "
                        f"{expected_v_sf_shape}, got {v_sf_shape}"
                    )
            pv_dtype = self.v_mma_dtype if self.v_dequant else v_dtype
            pv_mma_tile_k = 128 * 8 // pv_dtype.width
            if (
                pv_mma_tile_k % self.page_size != 0
                and self.page_size % pv_mma_tile_k != 0
            ):
                raise ValueError(
                    f"page_size({self.page_size}) and the V MMA K tile "
                    f"({pv_mma_tile_k}) must evenly tile one another"
                )

        if not (d_q == d_k == self.headdim):
            raise ValueError(
                f"headdim_q({d_q}), headdim_k({d_k}) must be {self.headdim}"
            )

        if h_q % h_k != 0:
            raise ValueError(f"heads_q({h_q}) must be a multiple of heads_k({h_k})")

        if s_k is not None and 0 < s_k < s_q:
            raise ValueError(
                f"non-zero seqlen({s_k}) must be at least prediction({s_q})"
            )

        # Not sure why this case fails {$nv-internal-release}
        if s_k is not None and 0 < s_k < 8:
            raise ValueError(
                f"non-zero seqlen({s_k}) with TMA masking must be at least 8"
            )

        if not self.is_paged_kv and b_k != b_q:
            raise ValueError(f"batches_k({b_k}) and batches_q({b_q}) mismatch")

        if bias_shape_bshr is None:
            rel_extent = None
        elif self.bias_is_sheared:
            b_bias, s_bias, h_bias, bias_storage_extent = bias_shape_bshr
            expected_q = math.ceil(s_q / 128) * 128
            rel_extent = bias_storage_extent - 256
            expected_prefix = (b_q, expected_q, h_q)
            if (b_bias, s_bias, h_bias) != expected_prefix:
                raise ValueError(
                    "sheared bias shape must be "
                    "(batch, round_up(prediction, 128), heads_q, rel_extent + 256); "
                    f"got {bias_shape_bshr} for Q shape {qo_shape_bshd}"
                )
            grouped_heads = h_q // h_k
            if grouped_heads <= 0 or 128 % grouped_heads != 0:
                raise ValueError(
                    "sheared Pack-GQA bias requires heads_q/heads_k to divide 128"
                )
            if rel_extent <= 0 or rel_extent % 128 != 0:
                raise ValueError(
                    "sheared bias requires positive 128-aligned rel_extent"
                )
        else:
            b_bias, s_bias, h_bias, bias_storage_extent = bias_shape_bshr
            rel_extent = bias_storage_extent
            if (b_bias, s_bias, h_bias) != (b_q, s_q, h_q):
                raise ValueError(
                    "compact bias shape must be (batch, prediction, heads_q, relative_extent); "
                    f"got {bias_shape_bshr} for Q shape {qo_shape_bshd}"
                )
            if rel_extent <= 0:
                raise ValueError("relative bias extent must be positive")

        if self.do_direct_red and kv_splits != 1:
            raise ValueError(f"direct reduction requires kv_splits=1, got {kv_splits}")

        if self.do_atomic_red:
            if kv_splits not in (1, 2, 4, 8, 16):
                raise ValueError(
                    f"atomic reduction requires kv_splits po2 <= 16, got {kv_splits}"
                )

            if o_dtype not in (Float32, BFloat16, Float16):
                raise TypeError(
                    f"atomic reduction requires (Float32, BFloat16, Float16) o_dtype, got {o_dtype}"
                )

    # Pack grouped heads with predicted tokens (s_q)
    @staticmethod
    def gqa_pack(t_bshd: cute.Tensor, h_k: int):
        d, h_q, s_q, b = tuple(reversed(t_bshd.shape))[:4]
        stride_d, stride_h, stride_s, stride_b = tuple(reversed(t_bshd.stride))[:4]
        # Batch + partial stride must be coalescible
        # to get 5 independent TMA modes (TMA limitation)
        has_partial = cute.rank(t_bshd) == 5
        b_partial = b * t_bshd.shape[0] if has_partial else b
        h_g = h_q // h_k  # grouped heads
        gqa_shape = (b_partial, (h_g, s_q), h_k, d)
        gqa_stride = (stride_b, (stride_h, stride_s), stride_h * h_g, stride_d)
        gqa_layout = cute.make_layout(gqa_shape, stride=gqa_stride)
        return cute.make_tensor(t_bshd.iterator, gqa_layout)

    @staticmethod
    def packed_gqa_view(t_thd: cute.Tensor, h_k: int, s_first: bool):
        total, h_q, d = t_thd.shape
        stride_t, stride_h, stride_d = t_thd.stride
        h_g = h_q // h_k
        if s_first:
            shape = ((h_g, total), d, h_k)
            stride = ((stride_h, stride_t), stride_d, stride_h * h_g)
        else:
            shape = (d, (h_g, total), h_k)
            stride = (stride_d, (stride_h, stride_t), stride_h * h_g)
        return cute.make_tensor(t_thd.iterator, cute.make_layout(shape, stride=stride))

    @staticmethod
    def packed_kv_view(t_thd: cute.Tensor, s_first: bool):
        if s_first:
            modes = (0, 2, 1)
        else:
            modes = (2, 0, 1)
        layout = cute.select(t_thd.layout, modes)
        return cute.make_tensor(t_thd.iterator, layout)

    @staticmethod
    def packed_ragged_o_view(t_thd: cute.Tensor, h_k: int):
        ragged = copy_utils.create_ragged_tensor_for_tma(
            t_thd, ragged_dim=0, ptr_shift=True
        )
        big, h_q, d, extra = ragged.shape
        stride_big, stride_h, stride_d, stride_extra = ragged.stride
        h_g = h_q // h_k
        shape = (d, (h_g, big), (h_k, extra))
        stride = (
            stride_d,
            (stride_h, stride_big),
            (stride_h * h_g, stride_extra),
        )
        return cute.make_tensor(ragged.iterator, cute.make_layout(shape, stride=stride))

    # Reorder and group modes for GEMM
    @staticmethod
    def gemm_view(t_bshd: cute.Tensor, s_first: bool):
        sdhb = (1, 3, 2, 0)  # GEMM1 MKL
        dshb = (3, 1, 2, 0)  # GEMM2 MKL
        reorder = sdhb if s_first else dshb
        mT_layout = cute.select(t_bshd.layout, reorder)
        mT_layout = cute.group_modes(mT_layout, 2, 4)
        return cute.make_tensor(t_bshd.iterator, mT_layout)

    @staticmethod
    def paged_ksf_view(t_pshg: cute.Tensor):
        pages, page_size, heads, groups = t_pshg.shape
        stride_p, _, stride_h, _ = t_pshg.stride
        layout = cute.make_layout(
            (page_size * groups, (heads, pages)),
            stride=(1, (stride_h, stride_p)),
        )
        return cute.make_tensor(t_pshg.iterator, layout)

    # Pack, reorder, and group modes for GEMM workspace
    @staticmethod
    def gemm_view_bsh(t_bsh: cute.Tensor, h_k: int):
        h_q, s_q, b = tuple(reversed(t_bsh.shape))[:3]
        stride_h, stride_s, stride_b = tuple(reversed(t_bsh.stride))[:3]
        h_g = h_q // h_k
        mT_shape = ((h_g, s_q), (h_k, b))
        mT_stride = ((stride_h, stride_s), (stride_h * h_g, stride_b))
        has_partial = cute.rank(t_bsh) == 4
        mT_shape += (t_bsh.shape[0],) if has_partial else ()
        mT_stride += (t_bsh.stride[0],) if has_partial else ()
        mT_layout = cute.make_layout(mT_shape, stride=mT_stride)
        return cute.make_tensor(t_bsh.iterator, mT_layout)

    ##############################
    # Decode Kernel launch
    ##############################
    @cute.jit
    def dense_bf16_bias(
        self,
        kv_splits: Int32,
        q_bshd: cute.Tensor,
        k_bshd: cute.Tensor,
        v_bshd: cute.Tensor,
        o_bshd: cute.Tensor,
        bias_bshr: cute.Tensor,
        o_partial_bshd: cute.Tensor,
        m_partial_bsh: cute.Tensor,
        l_partial_bsh: cute.Tensor,
        scale_s: Float32,
        stream: cuda.CUstream,
    ):
        """Launch dense BF16 relative-bias attention with deterministic splits.

        Args:
            kv_splits: Number of KV sequence splits.
            q_bshd, k_bshd, v_bshd: Dense BF16 Q/K/V tensors.
            o_bshd: BF16 output tensor.
            bias_bshr: Compact or statically selected Prefill-sheared BF16 bias.
            o_partial_bshd: FP32 partial outputs by split.
            m_partial_bsh, l_partial_bsh: FP32 split max/sum workspaces.
            scale_s: Softmax scale.
            stream: CUDA stream for both launches.

        Returns:
            None. The shared main and reduction kernels write o_bshd.
        """
        assert not self.is_paged_kv and self.do_kernel_red
        assert not self.qk_blockscaled and not self.v_dequant
        assert q_bshd.dtype == k_bshd.dtype == v_bshd.dtype == BFloat16
        assert o_bshd.dtype == bias_bshr.dtype == BFloat16
        return self.__call__(
            kv_splits,
            q_bshd,
            k_bshd,
            v_bshd,
            None,
            None,
            None,
            o_bshd,
            bias_bshr,
            o_partial_bshd,
            m_partial_bsh,
            l_partial_bsh,
            scale_s,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            stream,
        )

    @cute.jit
    def dense_bf16_bias_direct(
        self,
        q_bshd: cute.Tensor,
        k_bshd: cute.Tensor,
        v_bshd: cute.Tensor,
        o_bshd: cute.Tensor,
        bias_bshr: cute.Tensor,
        scale_s: Float32,
        stream: cuda.CUstream,
    ):
        """Launch one-split dense BF16 relative-bias attention directly.

        Args:
            q_bshd, k_bshd, v_bshd: Dense BF16 Q/K/V tensors.
            o_bshd: BF16 output tensor written by the main kernel.
            bias_bshr: Compact or statically selected Prefill-sheared BF16 bias.
            scale_s: Softmax scale.
            stream: CUDA stream for the launch.

        Returns:
            None. The single main kernel normalizes and writes ``o_bshd``.
        """
        assert not self.is_paged_kv and self.do_direct_red
        assert not self.qk_blockscaled and not self.v_dequant
        assert q_bshd.dtype == k_bshd.dtype == v_bshd.dtype == BFloat16
        assert o_bshd.dtype == bias_bshr.dtype == BFloat16
        return self.__call__(
            Int32(1),
            q_bshd,
            k_bshd,
            v_bshd,
            None,
            None,
            None,
            o_bshd,
            bias_bshr,
            None,
            None,
            None,
            scale_s,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            stream,
        )

    @cute.jit
    def paged_bf16_no_bias(
        self,
        kv_splits: Int32,
        mSeqUsedK: cute.Tensor,
        mPageTableOffsets: cute.Tensor,
        mPageTable: cute.Tensor,
        k_bshd: cute.Tensor,
        v_bshd: cute.Tensor,
        q_bshd: cute.Tensor,
        o_bshd: cute.Tensor,
        o_partial_bshd: cute.Tensor,
        m_partial_bsh: cute.Tensor,
        l_partial_bsh: cute.Tensor,
        scale_s: Float32,
        stream: cuda.CUstream,
    ):
        """Launch the paged BF16 deterministic no-bias specialization.

        This narrow ABI avoids per-call adaptation of unrelated MXFP8, bias,
        dense, and packed-varlen operands. The device kernels are shared with
        the general entry point.

        Args:
            kv_splits: Number of KV sequence splits.
            mSeqUsedK: Per-batch logical KV lengths.
            mPageTableOffsets: Per-batch offsets into the flattened page table.
            mPageTable: Logical-to-physical page indices.
            k_bshd, v_bshd, q_bshd: BF16 paged K/V pools and fixed Q.
            o_bshd: BF16 output tensor.
            o_partial_bshd: FP32 partial outputs by split.
            m_partial_bsh, l_partial_bsh: FP32 split max/sum workspaces.
            scale_s: Softmax scale.
            stream: CUDA stream for both launches.

        Returns:
            None. The main and deterministic reduction kernels write o_bshd.
        """
        assert self.is_paged_kv and self.do_kernel_red
        assert not self.qk_blockscaled and not self.v_dequant
        assert q_bshd.dtype == k_bshd.dtype == v_bshd.dtype == BFloat16
        assert o_bshd.dtype == BFloat16
        return self.__call__(
            kv_splits,
            q_bshd,
            k_bshd,
            v_bshd,
            None,
            None,
            None,
            o_bshd,
            None,
            o_partial_bshd,
            m_partial_bsh,
            l_partial_bsh,
            scale_s,
            None,
            None,
            None,
            mSeqUsedK,
            None,
            None,
            mPageTable,
            mPageTableOffsets,
            stream,
        )

    @cute.jit
    def __call__(
        self,
        kv_splits: Int32,  # threadblocks per sequence (flash decoding)
        q_bshd: cute.Tensor,
        k_bshd: cute.Tensor,
        v_bshd: cute.Tensor,
        q_sf: Optional[cute.Tensor],
        k_sf: Optional[cute.Tensor],
        v_sf: Optional[cute.Tensor],
        o_bshd: cute.Tensor,  # must be zero initialized for atomic reduction
        bias_bshr: Optional[cute.Tensor],  # None, compact, or Prefill-sheared bias
        # Workspace tensors for kernel reduction
        o_partial_bshd: Optional[cute.Tensor],  # partial O per kv split
        m_partial_bsh: Optional[cute.Tensor],  # partial colmax_s per kv split
        l_partial_bsh: Optional[cute.Tensor],  # partial colsum_p per kv split
        scale_s: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        max_seqlen_q: Int32 | int | None = None,
        max_seqlen_k: Int32 | int | None = None,
        mPageTable: Optional[cute.Tensor] = None,
        mPageTableOffsets: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        """Launch fixed-length, native packed-varlen, or paged-KV Decode.

        ``bias_bshr=None`` selects static no-relative-bias codegen.
        ``rel_bias_layout="compact"`` consumes (B, Sq, Hq, R).
        ``rel_bias_layout="sheared"`` consumes the corresponding Prefill
        ShearingBias allocation (B, round_up(Sq, 128), Hq, R + 256). The layout
        selection is static and is never inferred from the tensor shape.

        Packed varlen uses rank-3 Q/K/V/O and rank-3 bias tensors. Both
        ``mCuSeqlensQ`` and ``mCuSeqlensK`` are CUDA int32 prefix sums, and the
        corresponding maximum sequence lengths size the launch grid. The
        initial packed path requires ``reduction_mode="direct"``, one KV split,
        and leaves ``mSeqUsedQ/K`` unset. MXFP8 callers provide one padded SF
        plane per ``(batch, KV head)``. ``stream`` remains the final argument,
        matching the Prefill calling convention.

        Paged KV uses rank-4 physical K/V pools with shape
        ``(physical_pages, page_size, heads_k, head_dim)``. ``mPageTable`` is a
        flattened int32 logical-to-physical mapping, ``mPageTableOffsets``
        gives each batch's table start, and ``mSeqUsedK`` gives the logical KV
        length. Q/K may use BF16/FP16 or MXFP8; paged MXFP8 K supplies compact
        page-local UE8M0 SFK shaped (physical_pages, page_size, heads_k, D/32).
        SFK uses compact head-major backing with the element strides enforced by
        ``can_implement`` so each page/head plane is one aligned TMA payload.
        V may use BF16/FP16 or E4M3/E5M2 plus page-local UE8M0 SFV with shape
        (physical_pages, ceil(page_size / 32), heads_k, head_dim).
        FP8 V is dequantized into v_mma_dtype after TMA and GEMM2 remains a
        normal BF16/FP16 multiply with FP32 accumulation.

        Returns:
            None. Results are written to ``o_bshd`` and, for non-direct fixed
            modes, the supplied reduction workspaces.
        """
        ##############################
        # TiledMma creation
        ##############################
        qk_dtype = q_bshd.dtype
        v_dtype = v_bshd.dtype
        v_mma_dtype = (
            self.v_mma_dtype if cutlass.const_expr(self.v_dequant) else v_dtype
        )
        p_dtype = v_mma_dtype
        sf_dtype = q_sf.dtype if cutlass.const_expr(q_sf is not None) else qk_dtype
        vsf_dtype = v_sf.dtype if cutlass.const_expr(v_sf is not None) else v_dtype
        acc_dtype = Float32
        is_paged_kv = self.is_paged_kv
        parallel_v_dequant = self.v_dequant
        is_varlen_q = (
            mCuSeqlensQ is not None or mSeqUsedQ is not None
        ) and not is_paged_kv
        is_varlen_k = (
            mCuSeqlensK is not None or mSeqUsedK is not None
        ) and not is_paged_kv
        assert k_bshd.dtype == qk_dtype
        if cutlass.const_expr(bias_bshr is not None):
            assert bias_bshr.dtype == v_mma_dtype
        if cutlass.const_expr(is_paged_kv):
            assert mPageTable is not None and mPageTableOffsets is not None
            assert mSeqUsedK is not None
            assert mCuSeqlensQ is None and mCuSeqlensK is None
            assert mSeqUsedQ is None
            assert max_seqlen_q is None and max_seqlen_k is None
            assert cute.rank(q_bshd) == cute.rank(o_bshd) == 4
            assert cute.rank(k_bshd) == cute.rank(v_bshd) == 4
            if cutlass.const_expr(bias_bshr is not None):
                assert cute.rank(bias_bshr) == 4
            assert cute.rank(mPageTable) == cute.rank(mPageTableOffsets) == 1
            assert cute.rank(mSeqUsedK) == 1
            assert mPageTable.dtype == mPageTableOffsets.dtype == Int32
            assert mSeqUsedK.dtype == Int32
            if cutlass.const_expr(self.qk_blockscaled):
                assert qk_dtype in (Float8E4M3FN, Float8E5M2)
                assert q_sf is not None and k_sf is not None
                assert cute.rank(k_sf) == 4
            else:
                assert qk_dtype in (BFloat16, Float16)
            if cutlass.const_expr(self.v_dequant):
                assert v_sf is not None and cute.rank(v_sf) == 4
        if cutlass.const_expr(is_varlen_q or is_varlen_k):
            assert (
                self.do_direct_red
            ), "varlen decode initially requires direct reduction"
            assert mCuSeqlensQ is not None and mCuSeqlensK is not None
            assert mSeqUsedQ is None and mSeqUsedK is None
            assert max_seqlen_q is not None and max_seqlen_k is not None
            assert cute.rank(q_bshd) == cute.rank(k_bshd) == 3
            assert cute.rank(v_bshd) == cute.rank(o_bshd) == 3
            if cutlass.const_expr(bias_bshr is not None):
                assert cute.rank(bias_bshr) == 3
            if cutlass.const_expr(self.v_dequant):
                assert v_sf is not None and cute.rank(v_sf) == 6
        if cutlass.const_expr(self.qk_blockscaled):
            assert q_sf is not None and k_sf is not None
            assert q_sf.dtype == k_sf.dtype == Float8E8M0FNU
            assert qk_dtype in (Float8E4M3FN, Float8E5M2)
            assert not self.v_dequant or v_dtype in (Float8E4M3FN, Float8E5M2)
            if cutlass.const_expr(is_varlen_q):
                assert cute.rank(q_sf) == cute.rank(k_sf) == 6
        else:
            assert q_sf is None and k_sf is None
            assert self.v_dequant or qk_dtype == v_dtype
        if cutlass.const_expr(self.v_dequant):
            assert v_sf is not None and v_sf.dtype == Float8E8M0FNU
            assert v_dtype in (Float8E4M3FN, Float8E5M2)
        else:
            assert v_sf is None
            assert v_dtype in (BFloat16, Float16)

        # Block tile sets the granularity at which threadblocks consume work (BMM1/BMM2)
        blk_tile_s = self.sequence_tile
        blk_tile_h = self.grouped_head_tile
        blk_tile_p = self.prediction_tile
        blk_tile_d = self.headdim
        blk_tile_shpd = (blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d)

        # MMA tile sets the granularity at which TMAs + MMAs are staged in smem
        mma_tile_m = 128
        qk_mma_tile_k = (
            128
            if cutlass.const_expr(self.qk_blockscaled)
            else 128 * 8 // qk_dtype.width
        )
        pv_mma_tile_k = 128 * 8 // v_mma_dtype.width
        k_tma_tokens = (
            min(self.page_size, mma_tile_m)
            if cutlass.const_expr(is_paged_kv)
            else mma_tile_m
        )
        v_tma_tokens = (
            min(self.page_size, pv_mma_tile_k)
            if cutlass.const_expr(is_paged_kv)
            else pv_mma_tile_k
        )
        # N-major 8b B in smem requires N multiple of 16
        min_mma_tile_n = 16 if qk_dtype.width == 8 else 8
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_n = max(min_mma_tile_n, blk_tile_n)
        qk_mma_tile_mnk = (mma_tile_m, mma_tile_n, qk_mma_tile_k)
        pv_mma_tile_mnk = (mma_tile_m, mma_tile_n, pv_mma_tile_k)

        # MMA tiles per block tile
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_dm = math.ceil(blk_tile_d / mma_tile_m)
        tiles_dk = math.ceil(blk_tile_d / qk_mma_tile_k)
        assert blk_tile_s % mma_tile_m == 0
        assert mma_tile_n % blk_tile_n == 0
        assert not is_paged_kv or mma_tile_m % k_tma_tokens == 0
        assert not is_paged_kv or self.page_size % k_tma_tokens == 0
        assert not is_paged_kv or pv_mma_tile_k % v_tma_tokens == 0
        assert not is_paged_kv or self.page_size % v_tma_tokens == 0

        # GEMM1: (S_K, (H_R, S_Q), D, (H_K, B))
        if cutlass.const_expr(self.qk_blockscaled):
            tiled_mma_kq = sm100_utils.make_blockscaled_trivial_tiled_mma(
                qk_dtype,
                qk_dtype,
                OperandMajorMode.K,  # A = K
                OperandMajorMode.K,  # B = Q
                sf_dtype,
                self.qk_sf_vec_size,
                tcgen05.CtaGroup.ONE,
                qk_mma_tile_mnk[:2],
            )
            # SFB's SMEM/TMA representation is padded to 128 N rows.
            qsf_mma_tile_mnk = (
                mma_tile_m,
                cute.round_up(mma_tile_n, 128),
                qk_mma_tile_k,
            )
            tiled_mma_qsf = sm100_utils.make_blockscaled_trivial_tiled_mma(
                qk_dtype,
                qk_dtype,
                OperandMajorMode.K,
                OperandMajorMode.K,
                sf_dtype,
                self.qk_sf_vec_size,
                tcgen05.CtaGroup.ONE,
                qsf_mma_tile_mnk[:2],
            )
        else:
            tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(
                qk_dtype,
                qk_dtype,
                OperandMajorMode.K,  # K
                OperandMajorMode.K,  # Q
                acc_dtype,
                tcgen05.CtaGroup.ONE,
                qk_mma_tile_mnk[:2],
            )
            qsf_mma_tile_mnk = qk_mma_tile_mnk
            tiled_mma_qsf = tiled_mma_kq

        # GEMM2: (D, (H_R, S_Q), S_K, (H_K, B))
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(
            v_mma_dtype,
            p_dtype,
            OperandMajorMode.MN,  # V
            OperandMajorMode.MN,  # P
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            pv_mma_tile_mnk[:2],
        )
        if cutlass.const_expr(self.v_dequant):
            vq_mma_tile_mnk = (mma_tile_m, 128, pv_mma_tile_k)
            # The SF layout atom carries four 32-token groups, so its layout-
            # only MMA uses K=128 even though GEMM2 consumes K=64 per stage.
            vsf_mma_tile_mnk = (mma_tile_m, 128, 128)
            tiled_mma_vq = sm100_utils.make_trivial_tiled_mma(
                v_dtype,
                v_dtype,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                acc_dtype,
                tcgen05.CtaGroup.ONE,
                vq_mma_tile_mnk[:2],
            )
            tiled_mma_vsf = sm100_utils.make_blockscaled_trivial_tiled_mma(
                v_dtype,
                v_dtype,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                vsf_dtype,
                self.v_sf_vec_size,
                tcgen05.CtaGroup.ONE,
                vsf_mma_tile_mnk[:2],
            )
        else:
            vq_mma_tile_mnk = pv_mma_tile_mnk
            vsf_mma_tile_mnk = pv_mma_tile_mnk
            tiled_mma_vq = tiled_mma_vsf = tiled_mma_vp

        ##############################
        # Calculate stage counts
        ##############################
        # Fixed stage counts
        self.p_stages = p_stages = 4  # smem P (BMM2 B)
        self.o_stages = o_stages = 2  # tmem O (BMM2 C)

        # Calculate tmem alloc
        tmem_capacity_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        tmem_s_stage_cols = tiles_sm * mma_tile_n
        tmem_alloc_cols = mma_tile_n * o_stages  # per-thread colsum
        tmem_alloc_cols += tiles_dm * mma_tile_n * o_stages  # O
        sf_tmem_cols = 32 if cutlass.const_expr(self.qk_blockscaled) else 0
        max_s_stages = (
            tmem_capacity_cols - tmem_alloc_cols - sf_tmem_cols
        ) // tmem_s_stage_cols
        self.s_stages = s_stages = min(max_s_stages, p_stages)

        tmem_alloc_cols += tmem_s_stage_cols * s_stages  # S
        tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))  # po2
        self.tmem_alloc_cols = tmem_alloc_cols
        assert tmem_alloc_cols <= tmem_capacity_cols

        # Calculate smem alloc
        smem_alloc_bits = 0
        mbarrier_bits = Int64.width
        pipe_stage_bits = mbarrier_bits * 2  # producer + consumer
        k_stage_bits = mma_tile_m * qk_mma_tile_k * qk_dtype.width
        vq_stage_bits = mma_tile_m * pv_mma_tile_k * v_dtype.width
        v_stage_bits = mma_tile_m * pv_mma_tile_k * v_mma_dtype.width
        vsf_groups = (
            (pv_mma_tile_k // self.page_size) * math.ceil(self.page_size / 32)
            if cutlass.const_expr(is_paged_kv)
            else 4
        )
        vsf_stage_bits = (
            mma_tile_m * vsf_groups * vsf_dtype.width
            if cutlass.const_expr(self.v_dequant)
            else 0
        )
        q_stage_bits = mma_tile_n * qk_mma_tile_k * qk_dtype.width
        p_stage_bits = mma_tile_m * mma_tile_n * p_dtype.width
        ksf_stage_bits = (
            mma_tile_m * (qk_mma_tile_k // self.qk_sf_vec_size) * sf_dtype.width
            if cutlass.const_expr(self.qk_blockscaled)
            else 0
        )
        # Compact paged SFK is TMA-staged through a separately aligned,
        # ordered scratch before the consumer repacks it into standard SFA.
        ksf_scratch_stage_bits = (
            (mma_tile_m // self.page_size)
            * max(
                128 * 8,
                self.page_size
                * (qk_mma_tile_k // self.qk_sf_vec_size)
                * sf_dtype.width,
            )
            if cutlass.const_expr(is_paged_kv and self.qk_blockscaled)
            else 0
        )
        qsf_stage_bits = (
            128 * (qk_mma_tile_k // self.qk_sf_vec_size) * sf_dtype.width
            if cutlass.const_expr(self.qk_blockscaled)
            else 0
        )
        # tmem ptr
        smem_alloc_bits += Int32.width
        # colmax + colsum
        smem_alloc_bits += blk_tile_n * acc_dtype.width
        smem_alloc_bits += blk_tile_n * warpgroup_warps * acc_dtype.width
        if cutlass.const_expr(self.do_cluster_red):
            smem_alloc_bits += max_reduction_iters * blk_tile_n * acc_dtype.width * 2
            smem_alloc_bits += max_reduction_iters * mbarrier_bits * 2
        # Q, S, P, O
        smem_alloc_bits += tiles_dk * (q_stage_bits + qsf_stage_bits) + mbarrier_bits
        smem_alloc_bits += s_stages * pipe_stage_bits  # s in tmem
        smem_alloc_bits += p_stages * (tiles_sm * p_stage_bits + pipe_stage_bits)
        smem_alloc_bits += o_stages * pipe_stage_bits  # o in tmem
        alignment_bits = 1024 - (smem_alloc_bits % 1024)
        # K, V
        smem_capacity_bits = utils.get_smem_capacity_in_bytes("sm_100") * 8
        remaining_bits = smem_capacity_bits - smem_alloc_bits - alignment_bits
        kv_stage_bits = (
            k_stage_bits
            + ksf_stage_bits
            + vq_stage_bits
            + vsf_stage_bits
            + v_stage_bits
            if cutlass.const_expr(self.v_dequant)
            else max(k_stage_bits + ksf_stage_bits, v_stage_bits)
        )
        kv_stage_bits += ksf_scratch_stage_bits
        kv_stages = remaining_bits // kv_stage_bits
        kv_pipeline_count = 3 if parallel_v_dequant else 1
        kv_stages -= (
            1
            if (kv_stages * pipe_stage_bits * kv_pipeline_count > alignment_bits)
            else 0
        )

        print(f"\ts stages: {s_stages}\tkv stages: {kv_stages}")

        ##############################
        # TMA creation
        ##############################
        o_bshd_ = o_bshd if self.do_cluster_red else o_partial_bshd
        if cutlass.const_expr(is_varlen_q):
            h_k = k_bshd.shape[1]
            mQ_nkl = self.packed_gqa_view(q_bshd, h_k, True)
            mK_mkl = self.packed_kv_view(k_bshd, True)
            mV_mkl = self.packed_kv_view(v_bshd, False)
            mO_mnl = self.packed_ragged_o_view(o_bshd_, h_k)
        else:
            h_k = k_bshd.shape[2]
            # ((h_g, s_q), d, (h_k, b))
            mQ_nkl = self.gemm_view(self.gqa_pack(q_bshd, h_k), True)
            mK_mkl = self.gemm_view(k_bshd, True)
            mV_mkl = self.gemm_view(v_bshd, False)
            # (d, (h_g, s_q), (h_k, b_partial))
            mO_mnl = self.gemm_view(self.gqa_pack(o_bshd_, h_k), False)
        if cutlass.const_expr(self.qk_blockscaled):
            if cutlass.const_expr(is_varlen_q):
                # A block-scaled TMA tile cannot start in the middle of its
                # 128-row scale-factor atom.  Keep one padded SF plane per
                # (batch, KV head), while Q/K/V themselves remain packed.
                # Recover each plane's capacity from blocked SF storage.
                qsf_m_capacity = q_sf.shape[1] * 128
                sf_l = (h_k, mCuSeqlensQ.shape[0] - 1)
                qsf_shape = (
                    (
                        self.grouped_head_tile,
                        qsf_m_capacity // self.grouped_head_tile,
                    ),
                    mQ_nkl.shape[1],
                    sf_l,
                )
            else:
                qsf_shape = (
                    (self.grouped_head_tile, mQ_nkl.shape[0][1]),
                    mQ_nkl.shape[1],
                    mQ_nkl.shape[2],
                )
            mQSF_nkl = cute.make_tensor(
                q_sf.iterator,
                blockscaled_utils.tile_atom_to_shape_SF(qsf_shape, self.qk_sf_vec_size),
            )
            if cutlass.const_expr(is_paged_kv):
                mKSF_mkl = self.paged_ksf_view(k_sf)
            else:
                ksf_shape = (
                    (
                        k_sf.shape[1] * 128,
                        mK_mkl.shape[1],
                        sf_l,
                    )
                    if cutlass.const_expr(is_varlen_q)
                    else mK_mkl.shape
                )
                mKSF_mkl = cute.make_tensor(
                    k_sf.iterator,
                    blockscaled_utils.tile_atom_to_shape_SF(
                        ksf_shape, self.qk_sf_vec_size
                    ),
                )
        else:
            mQSF_nkl = None
            mKSF_mkl = None

        if cutlass.const_expr(self.v_dequant):
            if cutlass.const_expr(is_paged_kv):
                # Compact physical-page SFV: (pages, groups, heads, D).
                mVSF_mkl = self.gemm_view(v_sf, False)
            else:
                if cutlass.const_expr(is_varlen_q):
                    vsf_shape = (
                        self.headdim,
                        max_seqlen_k,
                        (h_k, mCuSeqlensK.shape[0] - 1),
                    )
                else:
                    vsf_shape = mV_mkl.shape
                mVSF_mkl = cute.make_tensor(
                    v_sf.iterator,
                    blockscaled_utils.tile_atom_to_shape_SF(
                        vsf_shape, self.v_sf_vec_size
                    ),
                )
        else:
            mVSF_mkl = None

        # (MMA, MMA_M/N, MMA_K, stages)
        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, qk_mma_tile_mnk, qk_dtype, tiles_dk
        )
        smem_layout_k = sm100_utils.make_smem_layout_a(
            tiled_mma_kq, qk_mma_tile_mnk, qk_dtype, kv_stages
        )
        smem_layout_v = sm100_utils.make_smem_layout_a(
            tiled_mma_vp, pv_mma_tile_mnk, v_mma_dtype, kv_stages
        )
        if cutlass.const_expr(self.v_dequant):
            smem_layout_vq = sm100_utils.make_smem_layout_b(
                tiled_mma_vq, vq_mma_tile_mnk, v_dtype, kv_stages
            )
            if cutlass.const_expr(is_paged_kv):
                smem_layout_vsf = cute.make_ordered_layout(
                    (mma_tile_m, vsf_groups, kv_stages), order=(0, 1, 2)
                )
            else:
                smem_layout_vsf = blockscaled_utils.make_smem_layout_sfb(
                    tiled_mma_vsf,
                    vsf_mma_tile_mnk,
                    self.v_sf_vec_size,
                    kv_stages,
                )
        else:
            smem_layout_vq = smem_layout_v
            smem_layout_vsf = None
        if cutlass.const_expr(is_paged_kv):
            # TMA cannot span a non-affine page-table mapping.  Re-express the
            # MMA stage as page/chunk slots so multiple page-local TMAs
            # reconstruct the same contiguous K/V tile in shared memory.
            smem_layout_k_mk = cute.composition(
                smem_layout_k,
                cute.make_layout((mma_tile_m, qk_mma_tile_k, kv_stages)),
            )
            smem_layout_vq_mk = cute.composition(
                smem_layout_vq,
                cute.make_layout((mma_tile_m, pv_mma_tile_k, kv_stages)),
            )
            smem_layout_k_tma = cute.tiled_divide(
                smem_layout_k_mk, (k_tma_tokens, qk_mma_tile_k)
            )
            smem_layout_k_tma = cute.select(smem_layout_k_tma, [0, 1, 3])
            smem_layout_vq_tma = cute.tiled_divide(
                smem_layout_vq_mk, (mma_tile_m, v_tma_tokens)
            )
            smem_layout_vq_tma = cute.select(smem_layout_vq_tma, [0, 2, 3])
            if cutlass.const_expr(self.v_dequant):
                groups_per_page = math.ceil(self.page_size / self.v_sf_vec_size)
                smem_layout_vsf_tma = cute.tiled_divide(
                    smem_layout_vsf, (mma_tile_m, groups_per_page)
                )
                smem_layout_vsf_tma = cute.select(smem_layout_vsf_tma, [0, 2, 3])
            else:
                smem_layout_vsf_tma = None
        else:
            smem_layout_k_tma = None
            smem_layout_vq_tma = None
            smem_layout_vsf_tma = None
        if cutlass.const_expr(self.qk_blockscaled):
            smem_layout_ksf = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma_kq,
                qk_mma_tile_mnk,
                self.qk_sf_vec_size,
                kv_stages,
            )
            smem_layout_qsf = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma_qsf,
                qsf_mma_tile_mnk,
                self.qk_sf_vec_size,
                tiles_dk,
            )
        else:
            smem_layout_ksf = None
            smem_layout_qsf = None
        if cutlass.const_expr(is_paged_kv and self.qk_blockscaled):
            ksf_groups = qk_mma_tile_k // self.qk_sf_vec_size
            ksf_page_elems = self.page_size * ksf_groups
            ksf_page_stride = max(128, ksf_page_elems)
            ksf_pages_per_stage = mma_tile_m // self.page_size
            ksf_scratch_stage_elems = ksf_page_stride * ksf_pages_per_stage
            smem_layout_ksf_scratch = cute.make_layout(
                (ksf_scratch_stage_elems, kv_stages),
                stride=(1, ksf_scratch_stage_elems),
            )
            ksf_payload_bytes = mma_tile_m * ksf_groups * sf_dtype.width // 8
            assert (
                cute.size_in_bytes(
                    sf_dtype,
                    cute.slice_(smem_layout_ksf_scratch, (None, 0)),
                )
                == ksf_scratch_stage_elems
            )
            assert (
                cute.size_in_bytes(
                    sf_dtype, cute.slice_(smem_layout_ksf, (None, None, None, 0))
                )
                == ksf_payload_bytes
            )
            smem_layout_ksf_scratch_tma = cute.make_layout(
                (ksf_page_elems, ksf_pages_per_stage, kv_stages),
                stride=(1, ksf_page_stride, ksf_scratch_stage_elems),
            )
        else:
            smem_layout_ksf_scratch = None
            smem_layout_ksf_scratch_tma = None

        o_smem_dtype = mO_mnl.dtype
        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_smem_dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (max(blk_tile_d, mma_tile_m), mma_tile_n), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))
        # (MMA_TILE_M, MMA_TILE_N, #TILE_DM)
        smem_layout_o = cute.select(smem_layout_o, (0, 1, 2))

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = (
            cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp()
            if self.do_atomic_red
            else cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        )

        # Construct multimode gmem tiler
        tma_tile_n = (blk_tile_h, mma_tile_n // blk_tile_h)
        qk_tma_tile_mnk = (mma_tile_m, tma_tile_n, qk_mma_tile_k)
        pv_tma_tile_mnk = (mma_tile_m, tma_tile_n, pv_mma_tile_k)
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_nkl,
            cute.select(smem_layout_q, mma_modes),
            qk_tma_tile_mnk,
            tiled_mma_kq,
        )
        if cutlass.const_expr(is_paged_kv):
            tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op,
                mK_mkl,
                smem_layout_k_tma[0],
                (k_tma_tokens, qk_mma_tile_k),
            )
            tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op,
                mV_mkl,
                smem_layout_vq_tma[0],
                (mma_tile_m, v_tma_tokens),
            )
        else:
            tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mK_mkl,
                cute.select(smem_layout_k, mma_modes),
                qk_tma_tile_mnk,
                tiled_mma_kq,
            )
            if cutlass.const_expr(self.v_dequant):
                tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
                    tma_load_op,
                    mV_mkl,
                    cute.select(smem_layout_vq, mma_modes),
                    vq_mma_tile_mnk,
                    tiled_mma_vq,
                )
            else:
                tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
                    tma_load_op,
                    mV_mkl,
                    cute.select(smem_layout_v, mma_modes),
                    pv_tma_tile_mnk,
                    tiled_mma_vp,
                )
        if cutlass.const_expr(self.v_dequant and is_paged_kv):
            groups_per_page = math.ceil(self.page_size / self.v_sf_vec_size)
            tma_atom_vsf, tma_tensor_vsf = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_op,
                mVSF_mkl,
                smem_layout_vsf_tma[0],
                (mma_tile_m, groups_per_page),
                internal_type=cutlass.Int16,
            )
        elif cutlass.const_expr(self.v_dequant):
            tma_atom_vsf, tma_tensor_vsf = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mVSF_mkl,
                cute.slice_(smem_layout_vsf, (None, None, None, 0)),
                vsf_mma_tile_mnk,
                tiled_mma_vsf,
                internal_type=cutlass.Int16,
            )
        else:
            tma_atom_vsf = tma_tensor_vsf = None
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            mO_mnl,
            cute.select(smem_layout_o, mode=[0, 1]),
            pv_tma_tile_mnk[:2],
        )
        if cutlass.const_expr(self.qk_blockscaled):
            if cutlass.const_expr(is_paged_kv):
                tma_atom_ksf, tma_tensor_ksf = cute.nvgpu.cpasync.make_tiled_tma_atom(
                    tma_load_op,
                    mKSF_mkl,
                    smem_layout_ksf_scratch_tma[0],
                    (ksf_page_elems,),
                    internal_type=cutlass.Int16,
                )
            else:
                tma_atom_ksf, tma_tensor_ksf = cute.nvgpu.make_tiled_tma_atom_A(
                    tma_load_op,
                    mKSF_mkl,
                    cute.slice_(smem_layout_ksf, (None, None, None, 0)),
                    qk_mma_tile_mnk,
                    tiled_mma_kq,
                    internal_type=cutlass.Int16,
                )
            tma_atom_qsf, tma_tensor_qsf = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mQSF_nkl,
                cute.slice_(smem_layout_qsf, (None, None, None, 0)),
                qsf_mma_tile_mnk,
                tiled_mma_qsf,
                internal_type=cutlass.Int16,
            )
        else:
            tma_atom_ksf = tma_tensor_ksf = None
            tma_atom_qsf = tma_tensor_qsf = None

        # GEMM views for workspace tensors
        mM_partial_nl = mL_partial_nl = None
        if cutlass.const_expr(self.do_kernel_red):
            assert (
                m_partial_bsh.dtype
                == l_partial_bsh.dtype
                == o_partial_bshd.dtype
                == acc_dtype
            )

            # ((h_g, s_q), (h_k, b), kv_splits)
            mM_partial_nl = self.gemm_view_bsh(m_partial_bsh, h_k)
            mL_partial_nl = self.gemm_view_bsh(l_partial_bsh, h_k)

        ##############################
        # Launch kernel(s)
        ##############################
        scale_s_log2_e = scale_s * log2_e

        if cutlass.const_expr(is_varlen_q):
            tiles_hp = (
                cute.ceil_div(mQ_nkl.shape[0][0], blk_tile_h),
                cute.ceil_div(max_seqlen_q, blk_tile_p),
            )
            n_tiles = cute.size(tiles_hp)
            l_tiles = h_k * (mCuSeqlensQ.shape[0] - 1)
        else:
            n_tiles = cute.size(
                cute.ceil_div(mQ_nkl.shape[0], (blk_tile_h, blk_tile_p))
            )
            l_tiles = cute.size(mQ_nkl.shape[2])
        grid = (kv_splits, n_tiles, l_tiles)
        cluster_x = kv_splits if self.do_atomic_red else 1

        self.decode(
            # MMA
            blk_tile_shpd,
            qk_mma_tile_mnk,
            pv_mma_tile_mnk,
            tiled_mma_kq,
            tiled_mma_qsf,
            tiled_mma_vp,
            tiled_mma_vq,
            tiled_mma_vsf,
            qk_dtype,
            v_dtype,
            v_mma_dtype,
            sf_dtype,
            vsf_dtype,
            o_smem_dtype,
            # Q
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            smem_layout_qsf,
            tma_atom_qsf,
            tma_tensor_qsf,
            # K
            smem_layout_k,
            smem_layout_k_tma,
            tma_atom_k,
            tma_tensor_k,
            smem_layout_ksf,
            smem_layout_ksf_scratch,
            smem_layout_ksf_scratch_tma,
            tma_atom_ksf,
            tma_tensor_ksf,
            # V
            smem_layout_v,
            smem_layout_vq,
            smem_layout_vq_tma,
            smem_layout_vsf,
            smem_layout_vsf_tma,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_vsf,
            tma_tensor_vsf,
            # O
            smem_layout_o,
            tma_atom_o,
            tma_tensor_o,
            bias_bshr,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            max_seqlen_q,
            mPageTable,
            mPageTableOffsets,
            # Rest
            mM_partial_nl,
            mL_partial_nl,
            scale_s_log2_e,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[cluster_x, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

        if cutlass.const_expr(self.do_kernel_red):
            self.launch_reduction(
                self.headdim,
                o_bshd,
                o_partial_bshd,
                m_partial_bsh,
                l_partial_bsh,
                stream,
                self.use_pdl,
            )

    @cute.jit
    def sheared_bias_column(
        self,
        query_idx: Int32,
        key_idx: Int32,
        query_head_in_group: Int32,
        grouped_heads: int,
        prediction: Int32,
        seqlen: Int32,
        bias_storage_extent: int,
    ) -> Int32:
        """Map one logical score to Prefill ShearingBias's physical column."""
        packed_m = query_idx * grouped_heads + query_head_in_group
        attn_m_block = packed_m // 128
        m_idx_max = cute.ceil_div((attn_m_block + 1) * 128, grouped_heads)
        attn_n_block_max = cutlass.min(
            cute.ceil_div(seqlen, 128),
            cute.ceil_div(m_idx_max + seqlen - prediction, 128),
        )
        return key_idx + bias_storage_extent - attn_n_block_max * 128

    @cute.jit
    def dequant_v_stage_dense_contiguous(
        self,
        sVq: cute.Tensor,
        sSFV: cute.Tensor,
        sV: cute.Tensor,
        src_stage: Int32,
        dst_stage: Int32,
        scale_group_offset: int,
        pv_mma_tile_mnk: cute.Tile,
        kv_stages: int,
        tidx: Int32,
    ):
        """Convert one dense FP8-V stage with the correction warpgroup."""
        mma_tile_m, _, pv_mma_tile_k = pv_mma_tile_mnk
        transposed_shape = (pv_mma_tile_k, mma_tile_m, kv_stages)
        transposed_layout = cute.make_ordered_layout(transposed_shape, order=(1, 0, 2))
        sVq_transposed = cute.composition(sVq, transposed_layout)
        sV_transposed = cute.composition(sV, transposed_layout)
        scale_layout = cute.make_ordered_layout(
            (mma_tile_m, 128, kv_stages), order=(0, 1, 2)
        )
        sSFV_logical = cute.composition(sSFV, scale_layout)
        sSFV_stage = sSFV_logical[None, None, src_stage]

        num_load_elems = 128 // sVq.element_type.width // 2
        num_store_elems = 128 // sV.element_type.width
        assert num_load_elems == num_store_elems
        threads_per_copy_row = 16
        tiled_copy_s2r = copy_utils.tiled_copy_2d(
            sVq.element_type,
            threads_per_row=threads_per_copy_row,
            num_threads=warpgroup_threads,
            num_copy_elems=num_load_elems,
        )
        tiled_copy_r2s = copy_utils.tiled_copy_2d(
            sV.element_type,
            threads_per_row=threads_per_copy_row,
            num_threads=warpgroup_threads,
            num_copy_elems=num_store_elems,
        )
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        source = thr_copy_s2r.partition_S(sVq_transposed)[None, None, None, src_stage]
        raw_fragment = cute.make_fragment_like(source)
        destination = thr_copy_r2s.partition_S(sV_transposed)[
            None, None, None, dst_stage
        ]
        coordinates = cute.make_identity_tensor((pv_mma_tile_k, mma_tile_m))
        thread_coordinates = thr_copy_s2r.partition_S(coordinates)
        num_rows = cute.size(raw_fragment.shape[1])
        num_cols = cute.size(raw_fragment.shape[2])
        copy_token_rows = warpgroup_threads // threads_per_copy_row
        rows_per_sf_group = self.v_sf_vec_size // copy_token_rows
        num_sf_groups = pv_mma_tile_k // self.v_sf_vec_size
        assert num_sf_groups == 2
        assert self.v_sf_vec_size % copy_token_rows == 0
        assert num_rows == rows_per_sf_group * num_sf_groups

        cute.copy(tiled_copy_s2r, source, raw_fragment)
        for group in cutlass.range_constexpr(num_cols):
            coordinate = thread_coordinates[0, 0, group]
            dim_begin = coordinate[1]
            dst_template = destination[None, 0, group]
            scales_ue8m0_lo = cute.make_fragment_like(dst_template, sSFV.element_type)
            scales_ue8m0_hi = cute.make_fragment_like(dst_template, sSFV.element_type)
            scales_bf16_lo = cute.make_fragment_like(dst_template, BFloat16)
            scales_bf16_hi = cute.make_fragment_like(dst_template, BFloat16)
            scale_base = scale_group_offset * num_sf_groups
            for elem in cutlass.range_constexpr(num_load_elems):
                scale_slice = sSFV_stage[dim_begin + elem, (0, None)]
                scale_coord = scales_ue8m0_lo.layout.get_hier_coord(elem)
                scales_ue8m0_lo[scale_coord] = scale_slice[scale_base]
                scales_ue8m0_hi[scale_coord] = scale_slice[scale_base + 1]
            cvt_tensor_ue8m0_to_bf16(scales_ue8m0_lo, scales_bf16_lo)
            cvt_tensor_ue8m0_to_bf16(scales_ue8m0_hi, scales_bf16_hi)
            scale_values_lo = scales_bf16_lo.load().to(sV.element_type)
            scale_values_hi = scales_bf16_hi.load().to(sV.element_type)
            converted = cute.make_fragment_like(dst_template)
            for row in cutlass.range_constexpr(num_rows):
                raw_values = raw_fragment[None, row, group]
                dst_values = destination[None, row, group]
                converted.store(raw_values.load().to(Float32).to(sV.element_type))
                if cutlass.const_expr(row < rows_per_sf_group):
                    converted.store(converted.load() * scale_values_lo)
                else:
                    converted.store(converted.load() * scale_values_hi)
                cute.copy(tiled_copy_r2s, converted, dst_values)

        cute.arch.fence_view_async_shared()

    @cute.jit
    def dequant_v_tile_dense_pipeline(
        self,
        sVq: cute.Tensor,
        sSFV: cute.Tensor,
        sV: cute.Tensor,
        raw_v_pipeline: pipeline.PipelineTmaAsync,
        v_mma_pipeline: pipeline.PipelineAsyncUmma,
        raw_v_consumer_state: pipeline.PipelineState,
        v_mma_producer_state: pipeline.PipelineState,
        v_dequant_nbar: nbar,
        tiles_sk: int,
        tiles_dm: int,
        pv_mma_tile_mnk: cute.Tile,
        kv_stages: int,
        tidx: Int32,
    ):
        """Convert one logical V tile between independent pipelines."""
        for sk in cutlass.range_constexpr(tiles_sk):
            for _ in cutlass.range_constexpr(tiles_dm):
                raw_v_pipeline.consumer_wait(raw_v_consumer_state)
                v_mma_pipeline.producer_acquire(v_mma_producer_state)
                if cutlass.const_expr(self.is_paged_kv):
                    self.dequant_v_stage(
                        sVq,
                        sSFV,
                        sV,
                        raw_v_consumer_state.index,
                        v_mma_producer_state.index,
                        0,
                        pv_mma_tile_mnk,
                        kv_stages,
                        warpgroup_threads,
                        tidx,
                    )
                else:
                    self.dequant_v_stage_dense_contiguous(
                        sVq,
                        sSFV,
                        sV,
                        raw_v_consumer_state.index,
                        v_mma_producer_state.index,
                        sk % 2,
                        pv_mma_tile_mnk,
                        kv_stages,
                        tidx,
                    )
                v_dequant_nbar.arrive_and_wait()
                raw_v_pipeline.consumer_release(raw_v_consumer_state)
                v_mma_pipeline.producer_commit(v_mma_producer_state)
                raw_v_consumer_state.advance()
                v_mma_producer_state.advance()
        return raw_v_consumer_state, v_mma_producer_state

    @cute.jit
    def dequant_v_stage(
        self,
        sVq: cute.Tensor,
        sSFV: cute.Tensor,
        sV: cute.Tensor,
        src_stage: Int32,
        dst_stage: Int32,
        scale_group_offset: int,
        pv_mma_tile_mnk: cute.Tile,
        kv_stages: int,
        num_threads: int,
        tidx: Int32,
    ):
        """Vectorize one page-local FP8 V stage along contiguous head rows."""
        mma_tile_m, _, pv_mma_tile_k = pv_mma_tile_mnk
        transposed_shape = (pv_mma_tile_k, mma_tile_m, kv_stages)
        transposed_layout = cute.make_ordered_layout(transposed_shape, order=(1, 0, 2))
        sVq_transposed = cute.composition(sVq, transposed_layout)
        sV_transposed = cute.composition(sV, transposed_layout)
        sSFV_stage = sSFV[None, None, src_stage]

        num_load_elems = 128 // sVq.element_type.width // 2
        num_store_elems = 128 // sV.element_type.width
        assert num_load_elems == num_store_elems
        threads_per_copy_row = 16
        tiled_copy_s2r = copy_utils.tiled_copy_2d(
            sVq.element_type,
            threads_per_row=threads_per_copy_row,
            num_threads=num_threads,
            num_copy_elems=num_load_elems,
        )
        tiled_copy_r2s = copy_utils.tiled_copy_2d(
            sV.element_type,
            threads_per_row=threads_per_copy_row,
            num_threads=num_threads,
            num_copy_elems=num_store_elems,
        )
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        source = thr_copy_s2r.partition_S(sVq_transposed)[None, None, None, src_stage]
        raw_fragment = cute.make_fragment_like(source)
        destination = thr_copy_r2s.partition_S(sV_transposed)[
            None, None, None, dst_stage
        ]
        coordinates = cute.make_identity_tensor((pv_mma_tile_k, mma_tile_m))
        thread_coordinates = thr_copy_s2r.partition_S(coordinates)
        num_rows = cute.size(raw_fragment.shape[1])
        num_cols = cute.size(raw_fragment.shape[2])
        copy_token_rows = num_threads // threads_per_copy_row
        scale_group_width = min(self.page_size, self.v_sf_vec_size)
        rows_per_sf_group = scale_group_width // copy_token_rows
        num_scale_groups = pv_mma_tile_k // scale_group_width
        assert num_rows == rows_per_sf_group * num_scale_groups

        cute.copy(tiled_copy_s2r, source, raw_fragment)
        for column in cutlass.range_constexpr(num_cols):
            coordinate = thread_coordinates[0, 0, column]
            dim_begin = coordinate[1]
            dst_template = destination[None, 0, column]
            converted = cute.make_fragment_like(dst_template)
            for sf_group in cutlass.range_constexpr(num_scale_groups):
                scales_ue8m0 = cute.make_fragment_like(dst_template, sSFV.element_type)
                scales_bf16 = cute.make_fragment_like(dst_template, BFloat16)
                for elem in cutlass.range_constexpr(num_load_elems):
                    scale_coord = scales_ue8m0.layout.get_hier_coord(elem)
                    scales_ue8m0[scale_coord] = sSFV_stage[dim_begin + elem, sf_group]
                cvt_tensor_ue8m0_to_bf16(scales_ue8m0, scales_bf16)
                scale_values = scales_bf16.load().to(sV.element_type)
                row_begin = sf_group * rows_per_sf_group
                for row_offset in cutlass.range_constexpr(rows_per_sf_group):
                    row = row_begin + row_offset
                    raw_values = raw_fragment[None, row, column]
                    dst_values = destination[None, row, column]
                    converted.store(
                        raw_values.load().to(Float32).to(sV.element_type) * scale_values
                    )
                    cute.copy(tiled_copy_r2s, converted, dst_values)

        cute.arch.fence_view_async_shared()
        if cutlass.const_expr(num_threads == warp_threads):
            cute.arch.sync_warp()

    @cute.jit
    def repack_paged_ksf_stage(
        self,
        sKSFScratch: cute.Tensor,
        sKSF: cute.Tensor,
        stage: Int32,
        mma_tile_m: int,
        qk_mma_tile_k: int,
        page_size: int,
        kv_stages: int,
        tidx: Int32,
    ):
        """Repack one compact paged-SFK stage into block-scaled SFA."""
        ksf_groups = qk_mma_tile_k // self.qk_sf_vec_size
        assert mma_tile_m == 128 and ksf_groups == 4
        scratch_page_elems = page_size * ksf_groups
        scratch_page_stride = max(128, scratch_page_elems)
        scratch_stage_elems = scratch_page_stride * (mma_tile_m // page_size)
        scratch_stage_words = scratch_stage_elems // 4
        scratch_word_layout = cute.make_layout(
            (scratch_stage_words, kv_stages), stride=(1, scratch_stage_words)
        )
        sfa_word_layout = cute.make_layout(
            (mma_tile_m, kv_stages), stride=(1, mma_tile_m)
        )
        sKSFScratch32 = cute.make_tensor(
            cute.recast_ptr(sKSFScratch.iterator, dtype=cutlass.Uint32),
            scratch_word_layout,
        )
        sKSF32 = cute.make_tensor(
            cute.recast_ptr(sKSF.iterator, dtype=cutlass.Uint32), sfa_word_layout
        )
        cute.arch.fence_view_async_shared()
        rows_per_lane = mma_tile_m // warp_threads
        for row in cutlass.range_constexpr(rows_per_lane):
            raw_row = tidx + row * warp_threads
            raw_page = raw_row // page_size
            raw_word = raw_page * (scratch_page_stride // 4) + raw_row % page_size
            sKSF32[tidx * rows_per_lane + row, stage] = sKSFScratch32[raw_word, stage]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.kernel
    def decode(
        self,
        # MMA
        blk_tile_shpd: cute.Tile,
        qk_mma_tile_mnk: cute.Tile,
        pv_mma_tile_mnk: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_qsf: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        tiled_mma_vq: cute.TiledMma,
        tiled_mma_vsf: cute.TiledMma,
        qk_dtype: Type[cutlass.Numeric],
        v_dtype: Type[cutlass.Numeric],
        v_mma_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        vsf_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        # Q
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        smem_layout_qsf: Optional[cute.Layout],
        tma_atom_qsf: Optional[cute.CopyAtom],
        mQSF: Optional[cute.Tensor],
        # K
        smem_layout_k: cute.ComposedLayout,
        smem_layout_k_tma: Optional[cute.ComposedLayout],
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        smem_layout_ksf: Optional[cute.Layout],
        smem_layout_ksf_scratch: Optional[cute.Layout],
        smem_layout_ksf_scratch_tma: Optional[cute.Layout],
        tma_atom_ksf: Optional[cute.CopyAtom],
        mKSF: Optional[cute.Tensor],
        # V
        smem_layout_v: cute.ComposedLayout,
        smem_layout_vq: cute.ComposedLayout,
        smem_layout_vq_tma: Optional[cute.ComposedLayout],
        smem_layout_vsf: Optional[cute.Layout],
        smem_layout_vsf_tma: Optional[cute.Layout],
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        tma_atom_vsf: Optional[cute.CopyAtom],
        mVSF: Optional[cute.Tensor],
        # O
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: cute.CopyAtom,
        mO: cute.Tensor,
        # Optional bias, compact or Prefill-sheared as selected by rel_bias_layout
        mBias: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        max_seqlen_q: Int32 | int | None,
        mPageTable: Optional[cute.Tensor],
        mPageTableOffsets: Optional[cute.Tensor],
        # Rest
        mM_partial: Optional[cute.Tensor],
        mL_partial: Optional[cute.Tensor],
        scale_s_log2_e: Float32,
    ):
        ##############################
        # Static variables
        ##############################
        # Smem alloc helper
        svector_align = 16
        stensor_align = 128
        smem = utils.SmemAllocator()

        # No multicast
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))  # vmnk

        # Alias types
        q_dtype = k_dtype = qk_dtype
        p_dtype = v_mma_dtype
        o_dtype = out_dtype
        acc_dtype = Float32

        # Shapes for MMA tile indexing (Read TMA partition for example)
        blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d = blk_tile_shpd
        blk_tile_hp = (blk_tile_h, blk_tile_p)  # multimode tiler
        blk_tile_n = blk_tile_h * blk_tile_p  # linearized tiler
        mma_tile_m, mma_tile_n, qk_mma_tile_k = qk_mma_tile_mnk
        _, _, pv_mma_tile_k = pv_mma_tile_mnk
        k_tma_tokens = (
            min(self.page_size, mma_tile_m)
            if cutlass.const_expr(self.is_paged_kv)
            else mma_tile_m
        )
        v_tma_tokens = (
            min(self.page_size, pv_mma_tile_k)
            if cutlass.const_expr(self.is_paged_kv)
            else pv_mma_tile_k
        )
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_sk = blk_tile_s // pv_mma_tile_k
        tiles_dm = cute.ceil_div(blk_tile_d, mma_tile_m)
        tiles_dk = cute.ceil_div(blk_tile_d, qk_mma_tile_k)

        # Static control flow
        do_kernel_red = self.do_kernel_red
        do_atomic_red = self.do_atomic_red
        do_cluster_red = self.do_cluster_red
        window_size_left = self.window_size_left
        is_local = window_size_left is not None
        is_paged_kv = self.is_paged_kv
        parallel_v_dequant = self.v_dequant
        is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        is_varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        if cutlass.const_expr(is_paged_kv):
            is_varlen_q = False
            is_varlen_k = False

        ##############################
        # Warp specialization
        ##############################
        # Warp assignments
        warpgroup_id = 0
        mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        tma_qo_warp_id = warpgroup_id * warpgroup_warps + 3
        reduction_warp_id = tma_kv_warp_id
        warpgroup_id += 1

        softmax_warpgroups = 2
        softmax_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + softmax_warpgroups)
        )
        warpgroup_id += softmax_warpgroups

        correction_warpgroup_id = warpgroup_id
        warpgroup_id += 1
        assert self.threads_per_cta == warpgroup_id * warpgroup_threads

        # Register allocations
        use_reg_reconfig = blk_tile_n > 16
        max_sw_regs_per_wg_thread = 256  # CUDA limitation
        max_hw_regs_per_wg_thread = 64 * 1024 // warpgroup_threads  # 64K regs per SM
        mma_tma_regs = 96 if cutlass.const_expr(self.v_dequant) else 64
        softmax_regs = 120
        correction_regs = min(
            max_sw_regs_per_wg_thread,
            max_hw_regs_per_wg_thread - mma_tma_regs - softmax_regs * 2,
        )
        assert (
            mma_tma_regs + softmax_regs * 2 + correction_regs
        ) <= max_hw_regs_per_wg_thread

        # Read thread indices
        kv_splits, tiles_hp, tiles_hb = cute.arch.grid_dim()
        kv_split_idx, coord_hp, coord_hb = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 1  # warp 0 does all mbarrier inits for now

        # Unpack multimodes
        grouped_heads = mQ.shape[0][0]
        heads_k = mK.shape[2]
        heads_q = bias_storage_extent = bias_extent = 0
        if cutlass.const_expr(is_varlen_q):
            batches = mCuSeqlensQ.shape[0] - 1
            tiles_hb = (heads_k, batches)
            tiles_hp = (
                cute.ceil_div(grouped_heads, blk_tile_h),
                cute.ceil_div(max_seqlen_q, blk_tile_p),
            )
        else:
            _, prediction = mQ.shape[0]
            heads_k, batches = tiles_hb = mQ.shape[2]
            tiles_hp = cute.ceil_div(mQ.shape[0], blk_tile_hp)
        if cutlass.const_expr(mBias is not None):
            if cutlass.const_expr(is_varlen_q):
                _, heads_q, bias_storage_extent = mBias.shape
            else:
                _, _, heads_q, bias_storage_extent = mBias.shape
            bias_extent = (
                bias_storage_extent - 256
                if cutlass.const_expr(self.bias_is_sheared)
                else bias_storage_extent
            )
        coord_hb = cute.idx2crd(coord_hb, tiles_hb)
        coord_hp = cute.idx2crd(coord_hp, tiles_hp)
        coord_hg, coord_p = coord_hp
        coord_hk, coord_b = coord_hb
        if cutlass.const_expr(is_varlen_q):
            offset_q = mCuSeqlensQ[coord_b]
            offset_k = mCuSeqlensK[coord_b]
            prediction = mCuSeqlensQ[coord_b + 1] - offset_q
            seqlen = mCuSeqlensK[coord_b + 1] - offset_k
            table_offset = Int32(0)
            page_count = Int32(0)
        elif cutlass.const_expr(is_paged_kv):
            offset_q = Int32(0)
            offset_k = Int32(0)
            seqlen = mSeqUsedK[coord_b]
            table_offset = mPageTableOffsets[coord_b]
            page_count = cute.ceil_div(seqlen, self.page_size)
        else:
            offset_q = Int32(0)
            offset_k = Int32(0)
            seqlen = mK.shape[0]
            table_offset = Int32(0)
            page_count = Int32(0)

        # Runtime control flow
        tiles_s = cute.ceil_div(seqlen, blk_tile_s)
        tile_begin_s = Int32(0)
        tile_end_s = tiles_s
        if cutlass.const_expr(is_local):
            query_begin = coord_p * blk_tile_p
            query_end = cutlass.min(query_begin + blk_tile_p, prediction)
            key_begin = cutlass.max(
                seqlen - prediction + query_begin - window_size_left,
                Int32(0),
            )
            key_end = cutlass.min(seqlen - prediction + query_end, seqlen)
            tile_begin_s = key_begin // blk_tile_s
            tile_end_s = cute.ceil_div(key_end, blk_tile_s)
        active_tiles_s = tile_end_s - tile_begin_s
        iters_s = cute.ceil_div(
            cutlass.max(active_tiles_s - kv_split_idx, Int32(0)), kv_splits
        )
        prefetch_iters = min(2, self.s_stages - 1)  # MMA KQ iters to hide first softmax
        exit_early = (
            kv_split_idx >= active_tiles_s or coord_p * blk_tile_p >= prediction
        )
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        ##############################
        # Prefetch TMA descriptor
        ##############################
        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
            if cutlass.const_expr(self.v_dequant):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_vsf)
            if cutlass.const_expr(self.qk_blockscaled):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_qsf)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_ksf)
        init_warp += 1

        ##############################
        # Tmem Allocation
        ##############################
        tmem_alloc_cols = self.tmem_alloc_cols
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)
        if warp_idx == init_warp and not exit_early:
            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        ##############################
        # Pipeline Allocation + Init
        ##############################
        # Initialize named barriers
        softmax_threads = warpgroup_threads
        correction_threads = warpgroup_threads
        reduction_threads = warp_threads
        tma_threads = warp_threads
        # Preserve KQ/VP dispatch cadence for shared and split KV pipelines.
        mma_order_kq_nbar = nbar(1, tma_threads + tma_threads)
        mma_order_vp_nbar = nbar(2, tma_threads + tma_threads)
        sM_producer_nbar = nbar(3, softmax_threads + correction_threads)
        sM_consumer_nbar = nbar(4, softmax_threads + correction_threads)
        tL_producer_nbar = nbar(5, softmax_threads + correction_threads)
        tL_consumer_nbar = nbar(7, softmax_threads + correction_threads)
        sM_final_nbar = nbar(9, correction_threads + reduction_threads)
        sL_final_nbar = nbar(10, correction_threads + reduction_threads)
        sO_final_nbar = nbar(11, correction_threads + tma_threads)
        sM_mutex_nbar = nbar(12, softmax_threads * softmax_warpgroups)
        v_dequant_nbar = nbar(14, correction_threads)

        # named barrier stage helper
        def with_phase(nbar_, phase):
            return nbar(nbar_.barrier_id + phase, nbar_.num_threads)

        # Alias thread cooperatives
        thr_cg = lambda t: CooperativeGroup(Agent.Thread, t)
        elect_one_cooperative = thr_cg(1)
        warpgroup_cooperative = thr_cg(warpgroup_threads)
        correction_warps_cooperative = thr_cg(warpgroup_warps)
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        softmax_group = warpgroup_cooperative
        correction_group = warpgroup_cooperative

        # Initialize cluster colmax + colsum mbar (even if this split exits early)
        if cutlass.const_expr(do_cluster_red):
            reduction_mbars_ptr = smem.allocate_array(Int64, max_reduction_iters * 2)
            if warp_idx == init_warp:
                if lane_idx < max_reduction_iters * 2:
                    mbar_ptr = reduction_mbars_ptr + lane_idx
                    arrive_count = 1
                    expect_tx_bytes = blk_tile_n * acc_dtype.width // 8
                    cute.arch.mbarrier_init(mbar_ptr, arrive_count)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expect_tx_bytes)
            init_warp += 1
            cute.arch.cluster_arrive_relaxed()

        # Initialize Q load mbarrier
        q_load_mbar = smem.allocate_array(Int64, 1)
        if warp_idx == init_warp:
            expect_tx_bytes = cute.size_in_bytes(q_dtype, smem_layout_q)
            if cutlass.const_expr(self.qk_blockscaled):
                expect_tx_bytes += cute.size_in_bytes(sf_dtype, smem_layout_qsf)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_load_mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(q_load_mbar, expect_tx_bytes)
        init_warp += 1

        # Initialize pipelines
        kv_stages = smem_layout_k.shape[-1]
        k_stage_bytes = mma_tile_m * qk_mma_tile_k * k_dtype.width // 8
        v_stage_bytes = mma_tile_m * pv_mma_tile_k * v_dtype.width // 8
        if cutlass.const_expr(self.v_dequant):
            vsf_stage = (
                cute.slice_(smem_layout_vsf, (None, None, 0))
                if cutlass.const_expr(is_paged_kv)
                else cute.slice_(smem_layout_vsf, (None, None, None, 0))
            )
            v_stage_bytes += cute.size_in_bytes(vsf_dtype, vsf_stage)
        if cutlass.const_expr(self.qk_blockscaled):
            # Count only the 128 * (D / 32) scale bytes delivered by TMA.
            # The consumer-side SFA repack is not another barrier transaction.
            k_stage_bytes += (
                mma_tile_m
                * (qk_mma_tile_k // self.qk_sf_vec_size)
                * sf_dtype.width
                // 8
            )
        kv_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        kv_producer, kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=kv_stages,
            producer_group=tma_group,
            consumer_group=mma_group,
            tx_count=k_stage_bytes,
            barrier_storage=kv_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        if cutlass.const_expr(parallel_v_dequant):
            raw_v_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
            raw_v_pipeline = pipeline.PipelineTmaAsync.create(
                num_stages=kv_stages,
                producer_group=tma_group,
                consumer_group=correction_warps_cooperative,
                tx_count=v_stage_bytes,
                barrier_storage=raw_v_pipeline_ptr,
                cta_layout_vmnk=mcast_layout,
                tidx=warpgroup_tidx,
                defer_sync=True,
            )
            raw_v_producer = raw_v_pipeline.make_producer()

            v_mma_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
            v_mma_pipeline = pipeline.PipelineAsyncUmma.create(
                num_stages=kv_stages,
                producer_group=correction_group,
                consumer_group=mma_group,
                barrier_storage=v_mma_pipeline_ptr,
                cta_layout_vmnk=mcast_layout,
                defer_sync=True,
            )
            v_mma_consumer = v_mma_pipeline.make_consumer()

        s_stages = self.s_stages
        s_pipeline_ptr = smem.allocate_array(Int64, s_stages * 2)
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=s_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        p_stages = self.p_stages
        p_pipeline_ptr = smem.allocate_array(Int64, p_stages * 2)
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=p_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        o_stages = self.o_stages
        o_pipeline_ptr = smem.allocate_array(Int64, o_stages * 2)
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=o_stages,
            producer_group=mma_group,
            consumer_group=correction_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        ##############################
        # Smem Tensor Allocation
        ##############################
        # Threadblock slice
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_qsf = tiled_mma_qsf.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)
        thrblk_mma_vq = tiled_mma_vq.get_slice(0)
        thrblk_mma_vsf = tiled_mma_vsf.get_slice(0)

        # Q, K, V
        tAsK = smem.allocate_tensor(
            k_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )  # (MMA, #MMA_M, #MMA_K, kv_stages)
        if cutlass.const_expr(self.v_dequant):
            tAsVq = smem.allocate_tensor(
                v_dtype, smem_layout_vq.outer, stensor_align, smem_layout_vq.inner
            )
            tAsV = smem.allocate_tensor(
                v_mma_dtype,
                smem_layout_v.outer,
                stensor_align,
                smem_layout_v.inner,
            )
            sVSF = smem.allocate_tensor(vsf_dtype, smem_layout_vsf, stensor_align)
        else:
            tAsV = cute.make_tensor(
                cute.recast_ptr(tAsK.iterator, smem_layout_v.inner, dtype=v_mma_dtype),
                smem_layout_v.outer,
            )
            tAsVq = tAsV
            sVSF = None
        # Place Q, P, M, L on the second sub-bank (128-227KiB) {$nv-internal-release}
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )  # (MMA, #MMA_N, #MMA_K, q_stages)
        if cutlass.const_expr(self.qk_blockscaled):
            sKSF = smem.allocate_tensor(sf_dtype, smem_layout_ksf, stensor_align)
            if cutlass.const_expr(is_paged_kv):
                sKSFScratch = smem.allocate_tensor(
                    sf_dtype, smem_layout_ksf_scratch, stensor_align
                )
            else:
                sKSFScratch = None
            sQSF = smem.allocate_tensor(sf_dtype, smem_layout_qsf, stensor_align)
        else:
            sKSF = None
            sKSFScratch = None
            sQSF = None

        # S
        # (MMA_MN, #MMA_M=1, #MMA_N=1, #TILE_SM, s_stages)
        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m, mma_tile_n, tiles_sm, s_stages)
        )
        tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)

        # P - Treat MN C tile of BMM0 as NM B tile of BMM1
        # (MMA_NK, #MMA_N=1, #MMA_K=TILE_S/MMA_K, p_stages)
        blk_tile_nm = (None, mma_tile_n, mma_tile_m * tiles_sm)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, blk_tile_nm, p_dtype, p_stages
        )
        tBsP_nm = smem.allocate_tensor(
            p_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )

        # Tile for NK B tile iteration
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B(
            (mma_tile_n, pv_mma_tile_k)
        )  # (MMA_NK, #MMA_N=1, #MMA_K=MMA_TILE_K/MMA_K, #TILE_SK=TILE_S/MMA_TILE_K, p_stages)
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

        # Reshape NM B tile of BMM1 to become MN C tile of BMM0
        # (MMA_NK, #MMA_N, #MMA_K=TILE_S/MMA_K, p_stages) ->
        # (MMA_MN, #MMA_M, #MMA_N, #TILE_SM, p_stages)
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4, 5))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        # O
        # Reuse KV smem for O TMA store
        sO_iterator = cute.recast_ptr(tAsK.iterator, smem_layout_o.inner, dtype=o_dtype)
        # (MMA_TILE_M, MMA_TILE_N, #TILE_DM)
        sO_mma = cute.make_tensor(sO_iterator, smem_layout_o.outer)
        # (MMA, #MMA_M, #MMA_N, #TILE_DM, o_stages)
        tCsO = thrblk_mma_vp.partition_C(sO_mma)
        tCtO = thrblk_mma_vp.make_fragment_C((*tCsO.shape, o_stages))

        # M - colmax
        sM_layout = cute.make_layout(blk_tile_n)
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        if warp_idx == init_warp:
            if lane_store_max:
                sM[lane_idx] = -Float32.inf
        init_warp += 1

        # L - colsum
        sL_layout = cute.make_layout((blk_tile_n, warpgroup_warps))
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        if warp_idx == init_warp:
            for i in cutlass.range_constexpr(0, cute.size(sL), warp_threads):
                if i + lane_idx < cute.size(sL):
                    sL[i + lane_idx] = Float32(0)
        init_warp += 1

        # per-thread colsum
        # (MMA_MN, #MMA_M=1, #MMA_N=1, o_stages)
        tCtL_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, o_stages))
        tCtL = thrblk_mma_kq.make_fragment_C(tCtL_shape)

        # R - cluster reduction buffers for colmax + colsum
        if cutlass.const_expr(do_cluster_red):
            sR_layout = cute.make_layout((blk_tile_n, max_reduction_iters, 2))
            sR = smem.allocate_tensor(acc_dtype, sR_layout, svector_align)

        ##############################
        # Sync
        ##############################
        # Ensure visibility of cluster mbarriers
        if cutlass.const_expr(do_cluster_red):
            cute.arch.cluster_wait()

        # Ensure visibility of local mbarrier inits and tmem alloc
        cute.arch.sync_threads()
        assert init_warp < (
            self.threads_per_cta // warp_threads
        ), f"used {init_warp + 1} init warps, {self.threads_per_cta // warp_threads} warps available"

        ##############################
        # Tmem tensor allocation
        ##############################
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtL = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtL.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtL)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        if cutlass.const_expr(self.qk_blockscaled):
            tCtKSF_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_kq,
                qk_mma_tile_mnk,
                self.qk_sf_vec_size,
                cute.slice_(smem_layout_ksf, (None, None, None, 0)),
            )
            tCtKSF = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + tmem_offset, dtype=sf_dtype),
                tCtKSF_layout,
            )
            tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtKSF)

            tCtQSF_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_kq,
                qk_mma_tile_mnk,
                self.qk_sf_vec_size,
                cute.slice_(smem_layout_qsf, (None, None, None, 0)),
            )
            tCtQSF = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + tmem_offset, dtype=sf_dtype),
                tCtQSF_layout,
            )
            tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtQSF)

            sf_copy_atom = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype
            )
            tiled_copy_ksf = tcgen05.make_s2t_copy(
                sf_copy_atom, cute.filter_zeros(tCtKSF)
            )
            thr_copy_ksf = tiled_copy_ksf.get_slice(0)
            tCsKSF = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_ksf,
                thr_copy_ksf.partition_S(cute.filter_zeros(sKSF)),
            )
            tCtKSF_copy = thr_copy_ksf.partition_D(cute.filter_zeros(tCtKSF))

            tiled_copy_qsf = tcgen05.make_s2t_copy(
                sf_copy_atom, cute.filter_zeros(tCtQSF)
            )
            thr_copy_qsf = tiled_copy_qsf.get_slice(0)
            tCsQSF = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_qsf,
                thr_copy_qsf.partition_S(cute.filter_zeros(sQSF)),
            )
            tCtQSF_copy = thr_copy_qsf.partition_D(cute.filter_zeros(tCtQSF))
        else:
            tCtKSF = tCtQSF = None
            tiled_copy_ksf = tiled_copy_qsf = None
            tCsKSF = tCsQSF = None
            tCtKSF_copy = tCtQSF_copy = None

        assert (
            tmem_offset <= tmem_alloc_cols
        ), f"\t{tmem_offset} tmem cols used, {tmem_alloc_cols} tmem cols allocated"

        ##############################
        # Exit early
        ##############################
        if exit_early:
            if warpgroup_idx == correction_warpgroup_id:
                sM_final_nbar.arrive()
                sL_final_nbar.arrive()

        ##############################
        # TMA KV Dispatch
        ##############################
        elif warp_idx == tma_kv_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            if cutlass.const_expr(is_paged_kv):
                # The physical-page coordinate is the final GMEM mode.  Each
                # page TMA fills one slot of the original MMA stage.
                sK_page = cute.make_tensor(tAsK.iterator, smem_layout_k_tma.outer)
                gK_page = cute.local_tile(
                    mK,
                    (k_tma_tokens, qk_mma_tile_k),
                    coord=(None, None, (coord_hk, None)),
                )
                tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=sK_page,
                    gmem_tensor=cute.group_modes(gK_page, 0, 2),
                )
                if cutlass.const_expr(self.qk_blockscaled):
                    ksf_groups = qk_mma_tile_k // self.qk_sf_vec_size
                    ksf_page_elems = self.page_size * ksf_groups
                    sKSF_page = cute.make_tensor(
                        sKSFScratch.iterator, smem_layout_ksf_scratch_tma
                    )
                    gKSF_page = cute.local_tile(
                        mKSF,
                        (ksf_page_elems,),
                        coord=(0, (coord_hk, None)),
                    )
                    tGSsKSF, tGSgKSF = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_ksf,
                        mcast_coord,
                        mcast_layout,
                        smem_tensor=sKSF_page,
                        gmem_tensor=cute.group_modes(gKSF_page, 0, 1),
                    )

                sV_page = cute.make_tensor(tAsVq.iterator, smem_layout_vq_tma.outer)
                gV_page = cute.local_tile(
                    mV,
                    (mma_tile_m, v_tma_tokens),
                    coord=(None, None, (coord_hk, None)),
                )
                tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=sV_page,
                    gmem_tensor=cute.group_modes(gV_page, 0, 2),
                )
                if cutlass.const_expr(self.v_dequant):
                    groups_per_page = math.ceil(self.page_size / self.v_sf_vec_size)
                    sVSF_page = cute.make_tensor(sVSF.iterator, smem_layout_vsf_tma)
                    gVSF_page = cute.local_tile(
                        mVSF,
                        (mma_tile_m, groups_per_page),
                        coord=(None, 0, (coord_hk, None)),
                    )
                    tGSsVSF, tGSgVSF = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_vsf,
                        mcast_coord,
                        mcast_layout,
                        smem_tensor=sVSF_page,
                        gmem_tensor=cute.group_modes(gVSF_page, 0, 2),
                    )
                k_tmas_per_stage = mma_tile_m // k_tma_tokens
                k_chunks_per_page = self.page_size // k_tma_tokens
                pages_k = pv_mma_tile_k // v_tma_tokens
                v_chunks_per_page = self.page_size // v_tma_tokens
            else:
                # Apply block tiler and slice for contiguous or packed KV.
                if cutlass.const_expr(is_varlen_k):
                    mK_cur = cute.domain_offset((offset_k, 0, 0), mK)
                    mV_cur = cute.domain_offset((0, offset_k, 0), mV)
                    coord_kv = coord_hk
                    if cutlass.const_expr(self.v_dequant):
                        mVSF_cur = mVSF
                        coord_vsf = (
                            0,
                            coord_hk + heads_k * coord_b,
                        )
                else:
                    mK_cur, mV_cur = mK, mV
                    coord_kv = coord_hb
                    if cutlass.const_expr(self.v_dequant):
                        mVSF_cur = mVSF
                        coord_vsf = (
                            0,
                            coord_hk + heads_k * coord_b,
                        )
                gK = cute.local_tile(
                    mK_cur,
                    tiler=(blk_tile_s, blk_tile_d),
                    coord=(None, 0, coord_kv),
                )
                gV = cute.local_tile(
                    mV_cur,
                    tiler=(blk_tile_d, blk_tile_s),
                    coord=(0, None, coord_kv),
                )
                if cutlass.const_expr(self.v_dequant):
                    gVSF = cute.local_tile(
                        mVSF_cur,
                        tiler=(blk_tile_d, blk_tile_s),
                        coord=(0, None, coord_vsf),
                    )
                gK_mma = cute.flat_divide(gK, (mma_tile_m, qk_mma_tile_k))
                gV_mma = cute.flat_divide(gV, (mma_tile_m, pv_mma_tile_k))
                tAgK = thrblk_mma_kq.partition_A(gK_mma)
                tAgV = (
                    thrblk_mma_vq.partition_B(gV_mma)
                    if cutlass.const_expr(self.v_dequant)
                    else thrblk_mma_vp.partition_A(gV_mma)
                )
                tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsK, 0, 3),
                    gmem_tensor=cute.group_modes(tAgK, 0, 3),
                )
                tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(tAsVq, 0, 3),
                    gmem_tensor=cute.group_modes(tAgV, 0, 3),
                )
                if cutlass.const_expr(self.v_dequant):
                    gVSF_mma = cute.flat_divide(gVSF, (mma_tile_m, 128))
                    tAgVSF = thrblk_mma_vsf.partition_B(gVSF_mma)
                    tGSsVSF, tGSgVSF = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_vsf,
                        mcast_coord,
                        mcast_layout,
                        smem_tensor=cute.group_modes(sVSF, 0, 3),
                        gmem_tensor=cute.group_modes(tAgVSF, 0, 3),
                    )
                    tGSsVSF = cute.filter_zeros(tGSsVSF)
                    tGSgVSF = cute.filter_zeros(tGSgVSF)

                if cutlass.const_expr(self.qk_blockscaled):
                    sf_coord_l = (0, coord_hk + heads_k * coord_b)
                    gKSF = cute.local_tile(
                        mKSF,
                        tiler=(blk_tile_s, blk_tile_d),
                        coord=(None, 0, sf_coord_l),
                    )
                    gKSF_mma = cute.flat_divide(gKSF, (mma_tile_m, qk_mma_tile_k))
                    tAgKSF = thrblk_mma_kq.partition_A(gKSF_mma)
                    tGSsKSF, tGSgKSF = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_ksf,
                        mcast_coord,
                        mcast_layout,
                        smem_tensor=cute.group_modes(sKSF, 0, 3),
                        gmem_tensor=cute.group_modes(tAgKSF, 0, 3),
                    )
                    tGSsKSF = cute.filter_zeros(tGSsKSF)
                    tGSgKSF = cute.filter_zeros(tGSgKSF)

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # Sequence loop
            prefetch_tiles = prefetch_iters * kv_splits
            kv_token = True  # Producer always acquires first
            raw_v_token = True
            for s in cutlass.range(
                kv_split_idx, prefetch_tiles + active_tiles_s, kv_splits
            ):
                # Load K
                if s < active_tiles_s:
                    tile_s = tile_begin_s + s
                    if cutlass.const_expr(is_paged_kv):
                        for sm in cutlass.range_constexpr(tiles_sm):
                            for dk in cutlass.range_constexpr(tiles_dk):
                                kv_handle = kv_producer.acquire_and_advance(kv_token)
                                kv_token = kv_producer.try_acquire()
                                for pm in cutlass.range_constexpr(k_tmas_per_stage):
                                    logical_k_chunk_idx = (
                                        tile_s * (blk_tile_s // k_tma_tokens)
                                        + sm * k_tmas_per_stage
                                        + pm
                                    )
                                    logical_page_idx = (
                                        logical_k_chunk_idx // k_chunks_per_page
                                    )
                                    k_page_chunk = (
                                        logical_k_chunk_idx % k_chunks_per_page
                                    )
                                    physical_page_idx = Int32(-1)
                                    if logical_page_idx < page_count:
                                        physical_page_idx = mPageTable[
                                            table_offset + logical_page_idx
                                        ]
                                    cute.copy(
                                        tma_atom_k,
                                        tGSgK[
                                            None,
                                            k_page_chunk,
                                            dk,
                                            physical_page_idx,
                                        ],
                                        tGSsK[None, pm, kv_handle.index],
                                        tma_bar_ptr=kv_handle.barrier,
                                    )
                                    if cutlass.const_expr(self.qk_blockscaled):
                                        cute.copy(
                                            tma_atom_ksf,
                                            tGSgKSF[None, physical_page_idx],
                                            tGSsKSF[None, pm, kv_handle.index],
                                            tma_bar_ptr=kv_handle.barrier,
                                        )
                    else:
                        tGSgK_s = tGSgK[None, None, None, tile_s]
                        for sm in cutlass.range_constexpr(tiles_sm):
                            for dk in cutlass.range_constexpr(tiles_dk):
                                kv_handle = kv_producer.acquire_and_advance(kv_token)
                                kv_token = kv_producer.try_acquire()
                                cute.copy(
                                    tma_atom_k,
                                    tGSgK_s[None, sm, dk],
                                    tGSsK[None, kv_handle.index],
                                    tma_bar_ptr=kv_handle.barrier,
                                )
                                if cutlass.const_expr(self.qk_blockscaled):
                                    cute.copy(
                                        tma_atom_ksf,
                                        tGSgKSF[None, sm, dk, tile_s],
                                        tGSsKSF[None, kv_handle.index],
                                        tma_bar_ptr=kv_handle.barrier,
                                    )

                # Load V
                if s >= prefetch_tiles:
                    tile_v = tile_begin_s + s - prefetch_tiles
                    if cutlass.const_expr(is_paged_kv):
                        for sk in cutlass.range_constexpr(tiles_sk):
                            for dm in cutlass.range_constexpr(tiles_dm):
                                if cutlass.const_expr(parallel_v_dequant):
                                    v_load_handle = raw_v_producer.acquire_and_advance(
                                        raw_v_token
                                    )
                                    raw_v_token = raw_v_producer.try_acquire()
                                else:
                                    v_load_handle = kv_producer.acquire_and_advance(
                                        kv_token, expected_tx=v_stage_bytes
                                    )
                                    kv_token = kv_producer.try_acquire()
                                for pk in cutlass.range_constexpr(pages_k):
                                    logical_v_chunk_idx = (
                                        tile_v * (blk_tile_s // v_tma_tokens)
                                        + sk * pages_k
                                        + pk
                                    )
                                    logical_page_idx = (
                                        logical_v_chunk_idx // v_chunks_per_page
                                    )
                                    v_page_chunk = (
                                        logical_v_chunk_idx % v_chunks_per_page
                                    )
                                    physical_page_idx = Int32(-1)
                                    if logical_page_idx < page_count:
                                        physical_page_idx = mPageTable[
                                            table_offset + logical_page_idx
                                        ]
                                    cute.copy(
                                        tma_atom_v,
                                        tGSgV[
                                            None, dm, v_page_chunk, physical_page_idx
                                        ],
                                        tGSsV[None, pk, v_load_handle.index],
                                        tma_bar_ptr=v_load_handle.barrier,
                                    )
                                    if cutlass.const_expr(self.v_dequant):
                                        cute.copy(
                                            tma_atom_vsf,
                                            tGSgVSF[None, dm, physical_page_idx],
                                            tGSsVSF[None, pk, v_load_handle.index],
                                            tma_bar_ptr=v_load_handle.barrier,
                                        )
                    else:
                        tGSgV_s = tGSgV[None, None, None, tile_v]
                        for sk in cutlass.range_constexpr(tiles_sk):
                            for dm in cutlass.range_constexpr(tiles_dm):
                                if cutlass.const_expr(parallel_v_dequant):
                                    v_load_handle = raw_v_producer.acquire_and_advance(
                                        raw_v_token
                                    )
                                    raw_v_token = raw_v_producer.try_acquire()
                                else:
                                    v_load_handle = kv_producer.acquire_and_advance(
                                        kv_token, expected_tx=v_stage_bytes
                                    )
                                    kv_token = kv_producer.try_acquire()
                                cute.copy(
                                    tma_atom_v,
                                    tGSgV_s[None, dm, sk],
                                    tGSsV[None, v_load_handle.index],
                                    tma_bar_ptr=v_load_handle.barrier,
                                )
                                if cutlass.const_expr(self.v_dequant):
                                    cute.copy(
                                        tma_atom_vsf,
                                        tGSgVSF[None, dm, sk // 2, tile_v],
                                        tGSsVSF[None, v_load_handle.index],
                                        tma_bar_ptr=v_load_handle.barrier,
                                    )

        ##############################
        # TMA QO Dispatch
        ##############################
        elif warp_idx == tma_qo_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Slice and partition Q
            if cutlass.const_expr(is_varlen_q):
                mQ_cur = cute.domain_offset(((0, offset_q), 0, 0), mQ)
                coord_q = coord_hk
            else:
                mQ_cur = mQ
                coord_q = coord_hb
            # (TILE_H, TILE_D)
            gQ = cute.local_tile(
                mQ_cur,
                tiler=(blk_tile_hp, blk_tile_d),
                coord=(coord_hp, 0, coord_q),
            )
            # Apply MMA tiler and MMA partition
            # (MMA_TILE_N, MMA_TILE_K, #TILE_HN=1, #TILE_DK)
            gQ_mma = cute.flat_divide(gQ, (mma_tile_n, qk_mma_tile_k))
            gQ_mma = gQ_mma[None, None, 0, None]
            # (MMA, #MMA_N, #MMA_K, #TILE_DK)
            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)
            # (TMA, #TILE_DK)
            tBsQ_tma, tBgQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )
            if cutlass.const_expr(self.qk_blockscaled):
                qsf_tile_n = cute.round_up(mma_tile_n, 128)
                coord_qsf_l = (0, coord_hk + heads_k * coord_b)
                gQSF = cute.local_tile(
                    mQSF,
                    tiler=((blk_tile_h, qsf_tile_n // blk_tile_h), blk_tile_d),
                    coord=((0, 0), 0, coord_qsf_l),
                )
                gQSF_mma = cute.flat_divide(gQSF, (qsf_tile_n, qk_mma_tile_k))
                gQSF_mma = gQSF_mma[None, None, 0, None]
                tBgQSF = thrblk_mma_qsf.partition_B(gQSF_mma)
                tBsQSF_tma, tBgQSF_tma = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_qsf,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(sQSF, 0, 3),
                    gmem_tensor=cute.group_modes(tBgQSF, 0, 3),
                )
                tBsQSF_tma = cute.filter_zeros(tBsQSF_tma)
                tBgQSF_tma = cute.filter_zeros(tBgQSF_tma)

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # Load Q
            cute.copy(
                tma_atom_q,
                tBgQ_tma,
                tBsQ_tma,
                tma_bar_ptr=q_load_mbar,
            )
            if cutlass.const_expr(self.qk_blockscaled):
                cute.copy(
                    tma_atom_qsf,
                    tBgQSF_tma,
                    tBsQSF_tma,
                    tma_bar_ptr=q_load_mbar,
                )

            # Slice and partition O
            # (TILE_D, TILE_H)
            if cutlass.const_expr(is_varlen_q):
                ragged_big = mO.shape[1][1]
                mO_cur = cute.domain_offset(
                    (0, (0, ragged_big - prediction), (0, 0)), mO
                )
                coord_o_l = (coord_hk, offset_q + prediction)
            else:
                mO_cur = mO
                coord_b_partial = (
                    coord_b if do_cluster_red else kv_split_idx * batches + coord_b
                )
                coord_o_l = (coord_hk, coord_b_partial)
            gO = cute.local_tile(
                mO_cur,
                tiler=(blk_tile_d, blk_tile_hp),
                coord=(0, coord_hp, coord_o_l),
            )
            # (MMA_TILE_M, MMA_TILE_N, #TILE_DM, #TILE_HN=1)
            gO_mma = cute.flat_divide(gO, (mma_tile_m, mma_tile_n))
            gO_mma = gO_mma[None, None, None, 0]
            # (TMA, #TILE_DM)
            sO_tma, gO_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(sO_mma, 0, 2),
                gmem_tensor=cute.group_modes(gO_mma, 0, 2),
            )

            # Store O to gmem
            # TODO: Prefetch O for L2 reduction {$nv-internal-release}
            sO_final_nbar.arrive_and_wait()
            cute.copy(tma_atom_o, sO_tma, gO_tma)

        ##############################
        # MMA KQ (BMM1) Dispatch
        ##############################
        elif warp_idx == mma_kq_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Setup mma descriptors
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            # Wait for Q
            cute.arch.mbarrier_wait(q_load_mbar, phase=0)

            # Sequence loop
            for s in cutlass.range(iters_s):
                s_token = s_producer.try_acquire()

                # Advance for MMA VP
                # Could relax the wait under certain tile + stage counts {$nv-internal-release}
                if cutlass.const_expr(not parallel_v_dequant):
                    if s >= prefetch_iters + 1:
                        for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                            kv_consumer.advance()
                mma_order_vp_nbar.arrive_and_wait()
                k_token = kv_consumer.try_wait()

                s_handle = s_producer.acquire_and_advance(s_token)
                for sm in cutlass.range_constexpr(tiles_sm):
                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for dk in cutlass.range_constexpr(tiles_dk):
                        k_handle = kv_consumer.wait_and_advance(k_token)
                        is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                        if is_last_iter:
                            mma_order_kq_nbar.arrive()
                        else:
                            k_token = kv_consumer.try_wait()

                        if cutlass.const_expr(self.qk_blockscaled):
                            if cutlass.const_expr(is_paged_kv):
                                self.repack_paged_ksf_stage(
                                    sKSFScratch,
                                    sKSF,
                                    k_handle.index,
                                    mma_tile_m,
                                    qk_mma_tile_k,
                                    self.page_size,
                                    kv_stages,
                                    lane_idx,
                                )
                            cute.copy(
                                tiled_copy_ksf,
                                tCsKSF[None, None, None, None, k_handle.index],
                                tCtKSF_copy,
                            )
                            cute.copy(
                                tiled_copy_qsf,
                                tCsQSF[None, None, None, None, dk],
                                tCtQSF_copy,
                            )
                            cute.gemm(
                                tiled_mma_kq,
                                tCtS[mma_dice + (sm, s_handle.index)],
                                [
                                    tAsK_desc[None, None, None, k_handle.index],
                                    tCtKSF,
                                ],
                                [tBsQ_desc[None, None, None, dk], tCtQSF],
                                tCtS[mma_dice + (sm, s_handle.index)],
                            )
                            tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                        else:
                            mmas_k = cute.size(tAsK.shape[2])
                            for mma_k in cutlass.range_constexpr(mmas_k):
                                cute.gemm(
                                    tiled_mma_kq,
                                    tCtS[mma_dice + (sm, s_handle.index)],
                                    tAsK_desc[None, None, mma_k, k_handle.index],
                                    tBsQ_desc[None, None, mma_k, dk],
                                    tCtS[mma_dice + (sm, s_handle.index)],
                                )
                                if dk == 0 and mma_k == 0:
                                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                        k_handle.release()
                s_handle.commit()

            # Tail loop
            for s in cutlass.range_constexpr(prefetch_iters):
                mma_order_vp_nbar.arrive_and_wait()
                mma_order_kq_nbar.arrive()

        ##############################
        # MMA VP (BMM2) Dispatch
        ##############################
        elif warp_idx == mma_vp_warp_id:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)

            # Setup mma descriptors
            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tAsV_desc = thrblk_mma_vp.make_fragment_A(tAsV)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)

            # Prefetch loop
            mma_order_vp_nbar.arrive()
            for s in cutlass.range_constexpr(prefetch_iters):
                if cutlass.const_expr(not parallel_v_dequant):
                    if s < iters_s:
                        for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                            kv_consumer.advance()
                mma_order_kq_nbar.arrive_and_wait()
                mma_order_vp_nbar.arrive()

            # Sequence loop
            for s in cutlass.range(iters_s):
                p_token = p_consumer.try_wait()
                o_token = o_producer.try_acquire()

                # Advance for MMA KQ
                if cutlass.const_expr(not parallel_v_dequant):
                    if s < iters_s - prefetch_iters:
                        for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                            kv_consumer.advance()
                mma_order_kq_nbar.arrive_and_wait()
                if cutlass.const_expr(parallel_v_dequant):
                    v_token = v_mma_consumer.try_wait()
                else:
                    v_token = kv_consumer.try_wait()

                p_handle = p_consumer.wait_and_advance(p_token)
                o_handle = o_producer.acquire_and_advance(o_token)
                for sk in cutlass.range_constexpr(tiles_sk):
                    for dm in cutlass.range_constexpr(tiles_dm):
                        if cutlass.const_expr(parallel_v_dequant):
                            v_handle = v_mma_consumer.wait_and_advance(v_token)
                        else:
                            v_handle = kv_consumer.wait_and_advance(v_token)
                        is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                        if is_last_iter:
                            mma_order_vp_nbar.arrive()
                        else:
                            if cutlass.const_expr(parallel_v_dequant):
                                v_token = v_mma_consumer.try_wait()
                            else:
                                v_token = kv_consumer.try_wait()

                        if cutlass.const_expr(
                            self.v_dequant and is_paged_kv and not parallel_v_dequant
                        ):
                            self.dequant_v_stage(
                                tAsVq,
                                sVSF,
                                tAsV,
                                v_handle.index,
                                v_handle.index,
                                0,
                                pv_mma_tile_mnk,
                                kv_stages,
                                warp_threads,
                                lane_idx,
                            )
                        mmas_k = cute.size(tAsV.shape[2])
                        for mma_k in cutlass.range_constexpr(mmas_k):
                            cute.gemm(
                                tiled_mma_vp,
                                tCtO[mma_dice + (dm, o_handle.index)],
                                tAsV_desc[None, None, mma_k, v_handle.index],
                                tBsP_desc[None, None, mma_k, sk, p_handle.index],
                                tCtO[mma_dice + (dm, o_handle.index)],
                            )
                        v_handle.release()
                p_handle.release()
                o_handle.commit()

            # Wait for signal to dealloc tmem, then dealloc
            if iters_s == 1:
                # Epilogue still reads the empty buffer
                o_producer.commit()
                o_producer.advance()
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        ##############################
        # Softmax Dispatch
        ##############################
        elif warpgroup_idx in softmax_warpgroup_ids:
            # Free registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(softmax_regs)

            # Initialize for dual warpgroups
            softmax_phase = (warpgroup_idx - 1) % softmax_warpgroups
            tL_producer_nbar = with_phase(tL_producer_nbar, softmax_phase)
            tL_consumer_nbar = with_phase(tL_consumer_nbar, softmax_phase)
            sM_acquire_nbar = with_phase(sM_mutex_nbar, softmax_phase)
            sM_release_nbar = with_phase(sM_mutex_nbar, softmax_phase ^ 1)
            if softmax_phase == 1:
                s_consumer.advance()
                p_producer.advance()
                sM_release_nbar.arrive()
                if iters_s == 1:
                    tL_producer_nbar.arrive()

            # Construct copy atom for S
            tmem_repeat_op_s = blk_tile_n
            if cutlass.const_expr(mma_tile_n == blk_tile_n and tiles_sm in (2, 4)):
                tmem_repeat_op_s *= tiles_sm
            tmem_repeat_op_s = tcgen05.Repetition(tmem_repeat_op_s)
            tmem_load_op_s = tcgen05.Ld32x32bOp(tmem_repeat_op_s)
            tmem_load_atom_s = cute.make_copy_atom(tmem_load_op_s, acc_dtype)
            # Tile atom and slice
            tCtS_stage = tCtS[mma_dice + (None, 0)]
            tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS_stage)
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)
            # Partition S and P
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #CPY_SM, stages)
            tStS = thr_load_s.partition_S(tCtS)
            tSsP = thr_load_s.partition_D(tCsP)
            # Slice unused modes
            tStS = tStS[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, s_stages)
            tSsP = tSsP[None, 0, 0, 0, None, None]  # (CPY, #CPY_SM, p_stages)

            # Construct copy atom for L
            tmem_repeat_op_l = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_l = tcgen05.Ld32x32bOp(tmem_repeat_op_l)
            tmem_store_op_l = tcgen05.St32x32bOp(tmem_repeat_op_l)
            tmem_load_atom_l = cute.make_copy_atom(tmem_load_op_l, acc_dtype)
            tmem_store_atom_l = cute.make_copy_atom(tmem_store_op_l, acc_dtype)
            # Tile atom and slice
            tCtL_phase = tCtL[mma_dice + (softmax_phase,)]
            tmem_load_l = tcgen05.make_tmem_copy(tmem_load_atom_l, tCtL_phase)
            thr_load_l = tmem_load_l.get_slice(warpgroup_tidx)
            # Partition L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N)
            tStL = thr_load_l.partition_S(tCtL_phase)
            tSrL_shape = thr_load_l.partition_D(tCtL_phase).shape
            # Slice unused modes
            # (CPY,)
            tStL = tStL[None, 0, 0, 0]
            tSrL_shape = tSrL_shape[:1]

            # Sequence loop
            for s in cutlass.range(softmax_phase, iters_s, softmax_warpgroups):
                s_token = s_consumer.try_wait()
                p_token = p_producer.try_acquire()

                # Load S from tmem and notify BMM1
                s_handle = s_consumer.wait_and_advance(s_token)
                tStS_s = tStS[None, None, s_handle.index]
                tSrS_s = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                cute.copy(tmem_load_atom_s, tStS_s, tSrS_s)
                cute.arch.fence_view_async_tmem_load()
                s_handle.release()

                scores = tSrS_s.load().reshape((blk_tile_n, tiles_sm))

                # Convert QK to log2-softmax space and fuse compact relative
                # bias.  For bottom-right causal attention, query q attends
                # key k with relative coordinate
                #
                #   rel = q + (seqlen_k - seqlen_q) - k.
                #
                # Split-K CTAs visit sequence tiles in a round-robin order;
                # recover the global tile coordinate before forming k.
                score_tile_s = tile_begin_s + kv_split_idx + s * kv_splits
                scores_log2_rmem = cute.make_rmem_tensor(
                    (blk_tile_h, blk_tile_p, tiles_sm), acc_dtype
                )
                scores_log2_rmem.store(
                    (scale_s_log2_e * scores).reshape(scores_log2_rmem.shape)
                )
                if cutlass.const_expr(is_local):
                    # Like Prefill, classify whole interior tiles before
                    # falling back to the exact per-score boundary mask.
                    # The bounds below are conservative across every query
                    # packed into this CTA, so the fast path is also valid
                    # for prediction_tile > 1 and partial query tiles.
                    query_tile_begin = coord_p * blk_tile_p
                    query_tile_end = query_tile_begin + blk_tile_p
                    query_abs_min = seqlen - prediction + query_tile_begin
                    query_abs_max = seqlen - prediction + query_tile_end - 1
                    interior_key_min = cutlass.max(
                        query_abs_max - window_size_left, Int32(0)
                    )
                    score_key_begin = score_tile_s * blk_tile_s
                    score_key_end = score_key_begin + blk_tile_s
                    tile_is_interior = query_tile_end <= prediction
                    tile_is_interior = (
                        tile_is_interior and score_key_begin >= interior_key_min
                    )
                    tile_is_interior = (
                        tile_is_interior and score_key_end <= query_abs_min + 1
                    )
                    tile_is_interior = tile_is_interior and score_key_end <= seqlen
                    if not tile_is_interior:
                        for sm in cutlass.range_constexpr(tiles_sm):
                            window_key_idx = (
                                score_key_begin + sm * mma_tile_m + warpgroup_tidx
                            )
                            for p in cutlass.range_constexpr(blk_tile_p):
                                window_query_idx = query_tile_begin + p
                                window_query_abs = (
                                    seqlen - prediction + window_query_idx
                                )
                                key_min = cutlass.max(
                                    window_query_abs - window_size_left, Int32(0)
                                )
                                window_is_oob = (
                                    window_key_idx < key_min
                                    or window_key_idx > window_query_abs
                                )
                                window_is_oob = (
                                    window_is_oob or window_key_idx >= seqlen
                                )
                                window_is_oob = (
                                    window_is_oob or window_query_idx >= prediction
                                )
                                window_score_p = scores_log2_rmem[None, p, sm]
                                window_mask = (
                                    -Float32.inf if window_is_oob else Float32(0)
                                )
                                window_score_p.store(
                                    window_score_p.load() + window_mask
                                )
                else:
                    # Tail tiles need per-score masking for BOTH the
                    # sequence end (key >= seqlen) and bottom-right
                    # causality between prediction rows (key > query_abs:
                    # row t sits at seqlen - prediction + t and must not
                    # attend the later rows' keys). The causal edge
                    # reaches down to key seqlen - prediction + 1, which
                    # can straddle a tile boundary — hence the window
                    # condition instead of last-tile-only. At
                    # prediction == 1 this reduces exactly to the old
                    # key >= seqlen tail mask.
                    if (score_tile_s + 1) * blk_tile_s > seqlen - prediction:
                        for sm in cutlass.range_constexpr(tiles_sm):
                            tail_key_idx = (
                                score_tile_s * blk_tile_s
                                + sm * mma_tile_m
                                + warpgroup_tidx
                            )
                            for p in cutlass.range_constexpr(blk_tile_p):
                                tail_query_abs = (
                                    seqlen - prediction + coord_p * blk_tile_p + p
                                )
                                tail_is_oob = (
                                    tail_key_idx > tail_query_abs
                                    or tail_key_idx >= seqlen
                                )
                                tail_score_p = scores_log2_rmem[None, p, sm]
                                tail_mask = -Float32.inf if tail_is_oob else Float32(0)
                                tail_score_p.store(tail_score_p.load() + tail_mask)
                if cutlass.const_expr(mBias is not None):
                    # Almost all tiles in a long decode sequence precede the
                    # compact bias window. Skip the per-score coordinate/load
                    # path for those tiles with a warpgroup-uniform branch.
                    bias_window_begin = seqlen - prediction - bias_extent
                    tile_end = (score_tile_s + 1) * blk_tile_s
                    if tile_end > bias_window_begin:
                        for sm in cutlass.range_constexpr(tiles_sm):
                            key_idx = (
                                score_tile_s * blk_tile_s
                                + sm * mma_tile_m
                                + warpgroup_tidx
                            )
                            for p in cutlass.range_constexpr(blk_tile_p):
                                query_idx = coord_p * blk_tile_p + p
                                rel_idx = query_idx + (seqlen - prediction) - key_idx
                                for h in cutlass.range_constexpr(blk_tile_h):
                                    query_head = (
                                        coord_hk * grouped_heads
                                        + coord_hg * blk_tile_h
                                        + h
                                    )
                                    score_log2 = scores_log2_rmem[h, p, sm]
                                    if query_idx < prediction:
                                        if query_head < heads_q:
                                            if rel_idx >= 0:
                                                if rel_idx < bias_extent:
                                                    bias_idx = rel_idx
                                                    if cutlass.const_expr(
                                                        self.bias_is_sheared
                                                    ):
                                                        bias_idx = (
                                                            self.sheared_bias_column(
                                                                query_idx,
                                                                key_idx,
                                                                coord_hg * blk_tile_h
                                                                + h,
                                                                grouped_heads,
                                                                prediction,
                                                                seqlen,
                                                                bias_storage_extent,
                                                            )
                                                        )
                                                    if cutlass.const_expr(is_varlen_q):
                                                        bias_value = mBias[
                                                            offset_q + query_idx,
                                                            query_head,
                                                            bias_idx,
                                                        ]
                                                    else:
                                                        bias_value = mBias[
                                                            coord_b,
                                                            query_idx,
                                                            query_head,
                                                            bias_idx,
                                                        ]
                                                    bias_log2 = (
                                                        bias_value.to(acc_dtype)
                                                        * log2_e
                                                    )
                                                    score_log2 += bias_log2
                                    scores_log2_rmem[h, p, sm] = score_log2
                scores_log2 = scores_log2_rmem.load().reshape((blk_tile_n, tiles_sm))

                # Reduce colmax in thread RF
                rM = cute.make_rmem_tensor_like(sM)
                rM.store(
                    scores_log2.reduce(
                        cute.ReductionOp.MAX,
                        init_val=-Float32.inf,
                        reduction_profile=(None, 0),
                    )
                )

                # Reduce colmax in warp RF
                rM_lane = Float32(0)
                for n in cutlass.range_constexpr(blk_tile_n):
                    rM[n] = warp_fmax(rM[n])  # warp reduction
                    # Avoid dynamic register indexing (creates spills)
                    if n == lane_idx:
                        rM_lane = rM[n]

                # Reduce colmax in smem
                sM_acquire_nbar.arrive_and_wait()
                sM_consumer_nbar.arrive_and_wait()
                if lane_store_max:
                    smem_fmax(sM.iterator + sM.layout(lane_idx), rM_lane)

                # Wait for colmax and load
                sM_producer_nbar.arrive_and_wait()
                colmax = sM.load()
                sM_release_nbar.arrive()

                # Handle if we never saw any in-bounds values
                if cutlass.const_expr(is_local):
                    rM = cute.make_rmem_tensor_like(colmax)
                    rM.store(colmax)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        if rM[n] == -Float32.inf:
                            rM[n] = Float32(0)
                    colmax = rM.load()

                # Wait for empty P buffer
                # Here so we can interleave ex2 with convert ops
                p_handle = p_producer.acquire_and_advance(p_token)
                tSsP_s = tSsP[None, None, p_handle.index]

                # Compute online softmax
                probs = exp2(scores_log2 - colmax)

                # Store P to smem and notify BMM2
                tSsP_s.store(probs.to(p_dtype).reshape(tSsP_s.shape))
                cute.arch.fence_view_async_shared()
                p_handle.commit()

                # Accumulate per-thread colsum
                # TensorSSA.reduce doesn't elide the zero add {$nv-internal-release}
                colsum = probs[None, 0]
                for sm in cutlass.range_constexpr(1, tiles_sm, 1):
                    colsum += probs[None, sm]
                tSrL = cute.make_rmem_tensor(tSrL_shape, acc_dtype)
                tSrL.store(colsum.reshape(tSrL.shape))

                # Store per-thread colsum to tmem
                tL_consumer_nbar.arrive_and_wait()
                cute.copy(tmem_store_atom_l, tSrL, tStL)
                cute.arch.fence_view_async_tmem_store()
                tL_producer_nbar.arrive()

                # Advance again for dual warpgroups
                s_consumer.advance()
                p_producer.advance()

        ##############################
        # Correction Dispatch
        ##############################
        elif warpgroup_idx == correction_warpgroup_id:
            # Alloc registers
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_increase(correction_regs)

            # Select copy atoms for O and L
            tmem_repeat_op_o = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_o = tcgen05.Ld32x32bOp(tmem_repeat_op_o)
            tmem_store_op_o = tcgen05.St32x32bOp(tmem_repeat_op_o)
            tmem_load_atom_o = cute.make_copy_atom(tmem_load_op_o, acc_dtype)
            tmem_store_atom_o = cute.make_copy_atom(tmem_store_op_o, acc_dtype)
            # Tile atoms and slice
            tCtO_dm = tCtO[mma_dice + (0, 0)]
            tmem_load_o = tcgen05.make_tmem_copy(tmem_load_atom_o, tCtO_dm)
            thr_load_o = tmem_load_o.get_slice(warpgroup_tidx)
            # Partition O and L
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, #TILE_DM, o_stages)
            tOtO = thr_load_o.partition_S(tCtO)
            tOsO = thr_load_o.partition_D(tCsO)
            # (CPY, #CPY_MMA, #CPY_M, #CPY_N, o_stages)
            tOtL = thr_load_o.partition_S(tCtL)
            # Slice unused modes
            tOtO = tOtO[None, 0, 0, 0, None, None]  # (CPY, #TILE_DM, o_stages)
            tOsO = tOsO[None, 0, 0, 0, None]  # (CPY, #TILE_DM)
            tOtL = tOtL[None, 0, 0, 0, None]  # (CPY, o_stages)

            # colsum load helper
            def colsum_load(
                phase,
                blk_tile_n=blk_tile_n,
                tOtL=tOtL,
                tOrO_shape=tOsO.shape[:1],
                tmem_load_atom_o=tmem_load_atom_o,
                tL_producer_nbar=tL_producer_nbar,
                tL_consumer_nbar=tL_consumer_nbar,
            ):
                with_phase(tL_producer_nbar, phase).arrive_and_wait()
                tOtL_s = tOtL[None, phase]
                tOrL_s = cute.make_rmem_tensor(tOrO_shape, Float32)
                cute.copy(tmem_load_atom_o, tOtL_s, tOrL_s)
                cute.arch.fence_view_async_tmem_load()
                with_phase(tL_consumer_nbar, phase).arrive()
                return tOrL_s.load().reshape(blk_tile_n)

            # Initialize O and colsum
            tOrO = cute.make_rmem_tensor(tOsO.shape, acc_dtype)
            tOrO.fill(Float32(0))
            for phase in cutlass.range_constexpr(o_stages):
                cute.copy(tmem_store_atom_o, tOrO, tOtO[None, None, phase])
            cute.copy(tmem_store_atom_o, tOrO[None, 0], tOtL[None, 1])
            cute.arch.fence_view_async_tmem_store()

            # Initialize consumer barriers
            sM_consumer_nbar.arrive()
            for phase in cutlass.range_constexpr(o_stages):
                with_phase(tL_consumer_nbar, phase).arrive()

            # Initialize colsum in RF
            colsum_p = cute.make_rmem_tensor((blk_tile_n, o_stages), Float32)
            colsum_0, colsum_1 = colsum_p[None, 0], colsum_p[None, 1]
            colsum_p.fill(Float32(0))

            # Set up conversion only after both O stages have been zeroed.
            # Each initial tile is published after its colmax handshake so
            # softmax can produce P before the converted-V ring fills.
            if cutlass.const_expr(parallel_v_dequant):
                raw_v_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, kv_stages
                )
                v_mma_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, kv_stages
                )
                dequant_v_tile = partial(
                    self.dequant_v_tile_dense_pipeline,
                    sVq=tAsVq,
                    sSFV=sVSF,
                    sV=tAsV,
                    raw_v_pipeline=raw_v_pipeline,
                    v_mma_pipeline=v_mma_pipeline,
                    v_dequant_nbar=v_dequant_nbar,
                    tiles_sk=tiles_sk,
                    tiles_dm=tiles_dm,
                    pv_mma_tile_mnk=pv_mma_tile_mnk,
                    kv_stages=kv_stages,
                    tidx=warpgroup_tidx,
                )

            # Load colmax of s-2, s-1
            sM_lane_prev_prev = sM_lane_prev = -Float32.inf
            for s in cutlass.range_constexpr(o_stages):
                sM_lane_prev_prev = sM_lane_prev
                if not (s == 1 and iters_s == 1):
                    sM_producer_nbar.arrive_and_wait()
                    if lane_store_max:
                        sM_lane_prev = sM[lane_idx]
                    sM_consumer_nbar.arrive()
                    if cutlass.const_expr(parallel_v_dequant):
                        raw_v_consumer_state, v_mma_producer_state = dequant_v_tile(
                            raw_v_consumer_state=raw_v_consumer_state,
                            v_mma_producer_state=v_mma_producer_state,
                        )

            # Sequence loop
            phase = 0
            for s in cutlass.range(iters_s - o_stages, unroll=o_stages):
                # Load colsum of s-2
                colsum_s = colsum_load(phase)

                # Load colmax of s
                sM_producer_nbar.arrive_and_wait()
                if s == iters_s - o_stages - 1:  # Notify for final colmax
                    sM_final_nbar.arrive()
                sM_lane = Float32(0)
                if lane_store_max:
                    sM_lane = sM[lane_idx]
                sM_consumer_nbar.arrive()

                # Wait for O of s-2
                # Here so we can interleave shuffle_sync with correction muls
                o_token = o_consumer.try_wait()
                o_handle = o_consumer.wait_and_advance(o_token)

                # Compute correction of s-2
                if cutlass.const_expr(is_local):
                    correction_lane = (
                        Float32(0)
                        if sM_lane_prev_prev == -Float32.inf
                        else exp2(sM_lane_prev_prev - sM_lane)
                    )
                else:
                    correction_lane = exp2(sM_lane_prev_prev - sM_lane)
                correction = cute.make_rmem_tensor_like(sM)
                for n in cutlass.range_constexpr(blk_tile_n):
                    correction[n] = cute.arch.shuffle_sync(correction_lane, n)
                correction = correction.load()

                # Correct O of s-2 and notify MMA VP
                correction_o = correction.reshape(tOsO.shape[:1])
                for dm in cutlass.range_constexpr(tiles_dm):
                    tOtO_dm = tOtO[None, dm, phase]
                    tOrO_dm = cute.make_rmem_tensor(tOsO.shape[:1], acc_dtype)
                    cute.copy(tmem_load_atom_o, tOtO_dm, tOrO_dm)
                    tOrO_dm.store(correction_o * tOrO_dm.load())
                    cute.copy(tmem_store_atom_o, tOrO_dm, tOtO_dm)
                cute.arch.fence_view_async_tmem_store()
                o_handle.release()

                # Correct and accumulate colsum of s-2
                colsum_s *= correction
                if phase == 0:
                    colsum_0.store(correction * colsum_0.load() + colsum_s)
                elif phase == 1:
                    colsum_1.store(correction * colsum_1.load() + colsum_s)

                if cutlass.const_expr(parallel_v_dequant):
                    raw_v_consumer_state, v_mma_producer_state = dequant_v_tile(
                        raw_v_consumer_state=raw_v_consumer_state,
                        v_mma_producer_state=v_mma_producer_state,
                    )

                # Advance loop
                sM_lane_prev_prev, sM_lane_prev = sM_lane_prev, sM_lane
                phase ^= 1

            # Notify for final colmax if we didn't enter loop
            if iters_s <= o_stages:
                sM_final_nbar.arrive()

            # Compute correction of s-1
            if cutlass.const_expr(is_local):
                correction_lane = (
                    Float32(0)
                    if sM_lane_prev_prev == -Float32.inf
                    else exp2(sM_lane_prev_prev - sM_lane_prev)
                )
            else:
                correction_lane = exp2(sM_lane_prev_prev - sM_lane_prev)
            correction = cute.make_rmem_tensor_like(sM)
            for n in cutlass.range_constexpr(blk_tile_n):
                correction[n] = cute.arch.shuffle_sync(correction_lane, n)
            correction = correction.load()

            # Correct and accumulate final colsum
            tail_phase = iters_s % o_stages
            for phase in cutlass.range_constexpr(o_stages):
                if tail_phase == phase:
                    # Accumulate in thread RF
                    colsum_prev = colsum_load(phase)
                    colsum_final = colsum_load(phase ^ 1)
                    colsum_prev += colsum_p[None, phase].load()
                    colsum_final += colsum_p[None, phase ^ 1].load()
                    colsum_final += correction * colsum_prev
                    # Reduce colsum in warp RF
                    rL_lane = Float32(0.0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rL_n = cute.arch.warp_reduction_sum(colsum_final[n])
                        if n == lane_idx:
                            rL_lane = rL_n
                    # Store partial colsum in smem and notify
                    if lane_store_max:
                        sL[lane_idx, warpgroup_widx] = rL_lane
                    sL_final_nbar.arrive()

            # Load O of s-1, s
            tOrO_tail = cute.make_rmem_tensor((*tOsO.shape, o_stages), acc_dtype)
            for s in cutlass.range_constexpr(o_stages):
                phase_s = tail_phase ^ s
                tOtO_phase = tOtO[None, None, phase_s]
                tOrO_tail_s = tOrO_tail[None, None, s]
                o_handle = o_consumer.wait_and_advance()
                cute.copy(tmem_load_atom_o, tOtO_phase, tOrO_tail_s)
                cute.arch.fence_view_async_tmem_load()
                o_handle.release()  # Final release signals tmem dealloc
            tOrO_prev = tOrO_tail[None, None, 0].load()
            tOrO_final = tOrO_tail[None, None, 1].load()

            # Correct and accumulate output
            output_prev = tOrO_prev.reshape((blk_tile_n, tiles_dm))
            output_final = tOrO_final.reshape((blk_tile_n, tiles_dm))
            output_final += correction * output_prev

            # Apply final correction
            if cutlass.const_expr(do_cluster_red):
                # final correction stored in sM
                sM_final_nbar.arrive_and_wait()
                correction = sM.load()
                output_final *= correction

            # Store O to smem and notify
            # TODO: shuffle in tmem, transpose to smem {$nv-internal-release}
            tOsO.store(output_final.to(o_dtype).reshape(tOsO.shape))
            cute.arch.fence_view_async_shared()
            sO_final_nbar.arrive()

        ##############################
        # Reduction Dispatch
        ##############################
        if warp_idx == reduction_warp_id:
            if cutlass.const_expr(do_kernel_red):
                self.reduction_epilogue(
                    blk_tile_hp,
                    coord_hp,
                    coord_hb,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM,
                    sL,
                    mM_partial,
                    mL_partial,
                )
            else:
                self.reduction_cluster(
                    blk_tile_n,
                    kv_splits,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    reduction_mbars_ptr,
                    sM,
                    sL,
                    sR,
                )
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        return

    @staticmethod
    @cute.jit
    def reduction_epilogue(
        blk_tile_hp: Tuple[int, int],
        coord_hp: Tuple[Int32, Int32],
        coord_hb: Tuple[Int32, Int32],
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
    ):
        # get gmem colmax + colsum to store to
        coord_h = (coord_hp, coord_hb, kv_split_idx)
        gM_partial = cute.local_tile(mM_partial, (blk_tile_hp,), coord_h)
        gL_partial = cute.local_tile(mL_partial, (blk_tile_hp,), coord_h)

        # tile predication
        blk_tile_h, blk_tile_p = blk_tile_hp
        blk_tile_n = blk_tile_h * blk_tile_p
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        # gmem predication
        grouped_heads, prediction = mM_partial.shape[0]
        cM = cute.make_identity_tensor(mM_partial.shape[0])
        cM = cute.local_tile(cM, blk_tile_hp, coord_hp)
        idx_hg, idx_p = cM[lane_idx]
        lane_store_max &= idx_hg < grouped_heads
        lane_store_max &= idx_p < prediction

        # Store this split's partial colmax. The final max is folded from
        # the partials inside the reduction kernel — a gmem running max
        # accumulated across launches (the retired m workspace) poisoned
        # reused buffers: a stale larger max underflows every
        # smaller-scale launch's rescale to NaN.
        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after barrier
        sM_final_nbar.arrive_and_wait()
        if lane_store_max:
            gM_partial[lane_idx] = sM[lane_idx]

        # Load partial colsum and reduce
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane_wg = sL[lane_idx, None]
            sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
            gL_partial[lane_idx] = sL_lane

    @staticmethod
    @cute.jit
    def reduction_cluster(
        blk_tile_n: int,
        kv_splits: Int32,
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        reduction_mbars_ptr: cute.Pointer,
        sM: cute.Tensor,
        sL: cute.Tensor,
        sR: cute.Tensor,
    ):
        acc_dtype = sM.dtype
        colmax_bits = blk_tile_n * acc_dtype.width
        copy_vec_bits = min(colmax_bits, 128)
        dsmem_store_threads = colmax_bits // copy_vec_bits
        dsmem_store_values = copy_vec_bits // acc_dtype.width
        dsmem_store_atom_r = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyDsmemStoreOp(),
            acc_dtype,
            num_bits_per_copy=copy_vec_bits,
        )
        dsmem_store_r = cute.make_tiled_copy(
            dsmem_store_atom_r,
            cute.make_ordered_layout(
                (dsmem_store_threads, dsmem_store_values), order=(1, 0)
            ),
            (blk_tile_n,),
        )
        thr_store_r = dsmem_store_r.get_slice(lane_idx)
        tRsM = thr_store_r.partition_S(sM)  # (CPY, #CPY)
        tRsL = thr_store_r.partition_S(sL)  # (CPY, #CPY, warpgroup_warps)
        tRsR = thr_store_r.partition_S(sR)  # (CPY, #CPY, max_red_iters, 2)

        tRrM_shape = thr_store_r.partition_D(sM).shape
        tRrM_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrM_prev = cute.make_rmem_tensor(tRrM_shape, acc_dtype)

        # Wait for last colmax
        cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after barrier
        sM_final_nbar.arrive_and_wait()
        is_reduction_lane = lane_idx < dsmem_store_threads
        if is_reduction_lane:
            tRrM_prev.store(tRsM.load())
            tRrM_final.store(tRrM_prev.load())

            # Cluster butterfly reduction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRsR_local = tRsR[None, None, i, 0]
                    tRsR_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx),
                        tRsR_local.layout,
                    )
                    local_mbar = reduction_mbars_ptr + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(
                        dsmem_store_atom_r, tRrM_final, tRsR_peer, mbar_ptr=peer_mbar
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    tRrR = tRsR_local.load()
                    for j in cutlass.range_constexpr(cute.size(tRrM_final)):
                        tRrM_final[j] = cute.arch.fmax(tRrM_final[j], tRrR[j])

        # Wait for last colsum
        sL_final_nbar.arrive_and_wait()
        if is_reduction_lane:
            # Warpgroup reduction
            colsum = tRsL[None, None, 0].load()
            for i in cutlass.range_constexpr(1, warpgroup_warps, 1):
                colsum += tRsL[None, None, i].load()

            # Compute final correction and correct local colsum
            correction = exp2(tRrM_prev.load() - tRrM_final.load())
            correction = correction.reshape(colsum.shape)
            colsum *= correction

            # Cluster butterfly reduction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 0x01 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRrL_local = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
                    tRrL_local.store(colsum)
                    tRsR_local = tRsR[None, None, i, 1]
                    tRsR_peer = cute.make_tensor(
                        cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx),
                        tRsR_local.layout,
                    )
                    local_mbar = reduction_mbars_ptr + max_reduction_iters + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(
                        dsmem_store_atom_r, tRrL_local, tRsR_peer, mbar_ptr=peer_mbar
                    )
                    cute.arch.fence_acq_rel_cta()  # dont reorder dsmem store after wait
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    colsum += tRsR_local.load()

            # Divide by final colsum and store
            rcp_colsum = cute.make_rmem_tensor(colsum.shape, acc_dtype)
            for i in cutlass.range(cute.size(colsum.shape)):
                rcp_colsum[i] = cute.arch.rcp_approx(colsum[i])
            tRsM.store(correction * rcp_colsum.load())

        # Notify for final correction
        sM_final_nbar.arrive()

    ##############################
    # Reduction Kernel launch
    ##############################
    @staticmethod
    @cute.jit
    def launch_reduction(
        d_per_blk: int,
        o_bshd: cute.Tensor,
        o_partial_bshd: cute.Tensor,  # partial O per kv split
        m_partial_bsh: cute.Tensor,  # partial colmax_s per kv split
        l_partial_bsh: cute.Tensor,  # partial colsum_p per kv split
        stream: cuda.CUstream,
        use_pdl: cutlass.Constexpr[bool] = True,
    ):
        splits, b, s_q, h_q, d = o_partial_bshd.shape

        # Tile in headdim first
        def reverse(t: cute.Tensor):
            modes = tuple(reversed(range(cute.rank(t))))
            layout = cute.select(t.layout, modes)
            return cute.make_tensor(t.iterator, layout)

        o_dhsb = reverse(o_bshd)
        o_partial_dhsb = reverse(o_partial_bshd)
        m_partial_hsb = reverse(m_partial_bsh)
        l_partial_hsb = reverse(l_partial_bsh)

        d_per_thr = 32 // o_bshd.dtype.width
        thr_per_blk = d_per_blk // d_per_thr
        d_blks = cute.ceil_div(d, d_per_blk)
        smem_bytes = (splits * 2) * Float32.width // 8

        FlashAttentionDecodeSm100Bias.reduction_kernel(
            (thr_per_blk, d_per_thr, d_per_blk),
            o_dhsb,
            o_partial_dhsb,
            m_partial_hsb,
            l_partial_hsb,
            use_pdl,
        ).launch(
            grid=[d_blks, h_q * s_q, b],
            block=[thr_per_blk, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )

    @staticmethod
    @cute.kernel
    def reduction_kernel(
        tile_d: cute.Tile,
        o_dhsb: cute.Tensor,
        o_partial_dhsb: cute.Tensor,
        m_partial_hsb: cute.Tensor,
        l_partial_hsb: cute.Tensor,
        use_pdl: cutlass.Constexpr[bool],
    ):
        thr_per_blk, d_per_thr, d_per_blk = tile_d
        d, h_q, s_q, b, splits = o_partial_dhsb.shape
        d_blk_idx, coord_hs, coord_b = cute.arch.block_idx()
        coord_h, coord_s = cute.idx2crd(coord_hs, (h_q, s_q))
        tidx, _, _ = cute.arch.thread_idx()

        not_oob_d = True
        if d % d_per_blk != 0:
            not_oob_d = d_blk_idx * d_per_blk + tidx * d_per_thr < d

        coord_o = (d_blk_idx, coord_h, coord_s, coord_b, None)
        gO = cute.local_tile(o_dhsb, (d_per_blk,), coord_o[:-1])
        gO_partial = cute.local_tile(o_partial_dhsb, (d_per_blk,), coord_o)

        gM_partial = cute.local_tile(m_partial_hsb, (1,), coord_o[1:])
        gL_partial = cute.local_tile(l_partial_hsb, (1,), coord_o[1:])
        gM_partial_0 = gM_partial[None, tidx]
        gL_partial_0 = gL_partial[None, tidx]

        smem_ptr = cute.arch.get_dyn_smem(Float32)
        partial_layout = cute.make_layout((1, splits))
        sM_partial = cute.make_tensor(smem_ptr, partial_layout)
        sL_partial = cute.make_tensor(smem_ptr + splits, partial_layout)
        sL_partial_0 = sL_partial[None, tidx]
        sM_partial_0 = sM_partial[None, tidx]

        cpasync_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32
        )

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        tv_layout = cute.make_ordered_layout((thr_per_blk, d_per_thr), order=(1, 0))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (d_per_blk,))
        thr_copy = tiled_copy.get_slice(tidx)

        tCgO_partial = thr_copy.partition_S(gO_partial)  # (CPY, #CPY=1, splits)
        tCgO_partial = tCgO_partial[None, 0, None]  # (CPY, splits)
        tCgO = thr_copy.partition_D(gO)  # (CPY, #CPY=1)
        tCgO = tCgO[None, 0]  # (CPY)
        tCrO_final = cute.zeros_like(tCgO, Float32)

        if cutlass.const_expr(use_pdl):
            cute.arch.fence_acq_rel_cta()  # Don't reorder partitioning after PDL wait
            cute.arch.griddepcontrol_wait()

        if tidx < splits:
            cute.copy(cpasync_atom, gL_partial_0, sL_partial_0)
            cute.copy(cpasync_atom, gM_partial_0, sM_partial_0)

        for split_idx in cutlass.range(thr_per_blk + tidx, splits, thr_per_blk):
            gL_partial_n = gL_partial[None, split_idx]
            sL_partial_n = sL_partial[None, split_idx]
            cute.copy(cpasync_atom, gL_partial_n, sL_partial_n)

            gM_partial_n = gM_partial[None, split_idx]
            sM_partial_n = sM_partial[None, split_idx]
            cute.copy(cpasync_atom, gM_partial_n, sM_partial_n)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Fold the final colmax from the per-split partials (inactive splits publish -inf, the fmax identity).
        max_final = -Float32.inf
        for max_split_idx in cutlass.range(splits, unroll=8):
            max_final = cute.arch.fmax(max_final, sM_partial[0, max_split_idx])
        sum_final = Float32(0)
        if max_final > -Float32.inf and not_oob_d:
            for split_idx in cutlass.range(splits, unroll=8):
                max_partial = sM_partial[0, split_idx]
                if max_partial > -Float32.inf:
                    correction = exp2(max_partial - max_final)
                    sum_final += correction * sL_partial[0, split_idx]
                    tCrO_final += correction * tCgO_partial[None, split_idx].load()
            tCrO_final *= cute.arch.rcp_approx(sum_final)

        if cutlass.const_expr(use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if not_oob_d:
            tCgO.store(tCrO_final.to(o_dhsb.dtype))

        return


def run(
    batches: int,
    prediction: int,
    seqlen: int,
    heads_q: int,
    heads_k: int,
    headdim: int,
    rel_bias_extent: int,
    kv_splits: int,
    reduction: str,
    qkv_dtype: Type[cutlass.Numeric],
    qk_mode: str,
    pv_dtype: Optional[Type[cutlass.Numeric]],
    o_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    tolerance: float,
    scale_s: float,
    v_mode: str = "dense",
    window_size_left: int | None = None,
    warmup_iterations: int = 0,
    iterations: int = 0,
    skip_ref_check: bool = False,
    use_warm_l2: bool = False,
    use_cuda_graphs: bool = True,
    quiet: bool = False,
    return_debug_tensors: bool = False,
    seed: int = 1111,
    bias_mode: str = "random",
    scale_pattern: str = "random",
    qk_init_range: tuple[int, int] = (-8, 7),
    v_init=None,
    v_dequant: bool = False,
    page_size: int | None = None,
    sequence_tile: int = 256,
):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")
    if min(batches, prediction, seqlen, heads_q, heads_k, headdim) <= 0:
        raise ValueError(
            "batch, sequence lengths, heads, and head_dim must be positive"
        )
    if seqlen < prediction:
        raise ValueError("KV sequence length must be at least the query length")
    if heads_q % heads_k != 0:
        raise ValueError("heads_q must be divisible by heads_k")
    if rel_bias_extent <= 0:
        raise ValueError("rel_bias_extent must be positive")
    if window_size_left is not None and window_size_left < 0:
        raise ValueError("window_size_left must be nonnegative or None")
    if page_size is not None and page_size not in (8, 16, 32, 64, 128, 256):
        raise ValueError("page_size must be one of 8, 16, 32, 64, 128, 256, or None")
    if sequence_tile <= 0 or sequence_tile % 128:
        raise ValueError("sequence_tile must be a positive multiple of 128")
    if bias_mode not in ("random", "zero", "structured", "impulse"):
        raise ValueError("bias_mode must be random, zero, structured, or impulse")
    if scale_pattern not in ("random", "unit", "structured", "row", "column"):
        raise ValueError(
            "scale_pattern must be random, unit, structured, row, or column"
        )
    if len(qk_init_range) != 2 or qk_init_range[0] >= qk_init_range[1]:
        raise ValueError("qk_init_range must be an increasing (min, max) pair")
    qk_blockscaled = qk_mode.upper() == "MXFP8"
    v_dequant = v_dequant or v_mode.upper() == "MXFP8"
    if v_mode.upper() not in ("DENSE", "MXFP8"):
        raise ValueError("v_mode must be dense or MXFP8")
    if page_size in (128, 256) and (qk_blockscaled or v_dequant):
        raise ValueError(
            "page_size=128/256 currently supports dense BF16/FP16 Q/K/V only"
        )
    if qk_blockscaled:
        if qkv_dtype not in (Float8E4M3FN, Float8E5M2):
            qkv_dtype = Float8E4M3FN
        if headdim != 128:
            raise ValueError("the initial MXFP8 decode path supports head_dim=128")
        if prediction * (heads_q // heads_k) > 32:
            raise ValueError(
                "MXFP8 currently requires prediction * grouped_heads <= 32"
            )
    if qk_blockscaled or v_dequant:
        pv_dtype = BFloat16 if pv_dtype is None else pv_dtype
    else:
        pv_dtype = qkv_dtype
    v_storage_dtype = Float8E4M3FN if v_dequant else pv_dtype
    bf16_eager = (
        not qk_blockscaled
        and not v_dequant
        and qkv_dtype == pv_dtype == o_dtype == BFloat16
    )

    npo2 = lambda x: 2 ** math.ceil(math.log2(x))

    grouped_heads = heads_q // heads_k
    grouped_head_tile = npo2(grouped_heads)
    grouped_head_tile = min(32, grouped_head_tile)

    # Crashes when converting single f32 to f8 {$nv-internal-release}
    if grouped_head_tile == 1 and prediction == 1 and qkv_dtype.width == 8:
        grouped_head_tile = 2

    prediction_tile = (
        choose_swa_prediction_tile(
            prediction,
            grouped_head_tile,
            window_size_left,
            bf16_eager,
        )
        if window_size_left is not None
        else min(32 // grouped_head_tile, npo2(prediction))
    )
    grouped_head_tiles = math.ceil(grouped_heads / grouped_head_tile)
    prediction_tiles = math.ceil(prediction / prediction_tile)
    base_ctas = batches * heads_k * grouped_head_tiles * prediction_tiles
    hardware_info = cutlass.utils.HardwareInfo()
    sm_count = hardware_info.get_device_multiprocessor_count()
    sm_count = 148 if sm_count <= 0 else sm_count

    key_begin = (
        0
        if window_size_left is None
        else max(seqlen - prediction - window_size_left, 0)
    )
    active_sequence_tiles = (
        math.ceil(seqlen / sequence_tile) - key_begin // sequence_tile
    )
    swa_plan = None

    # Automatic KV splits
    if kv_splits == 0:
        if window_size_left is not None:
            swa_plan = plan_swa_decode(
                batches,
                prediction,
                seqlen,
                heads_q,
                heads_k,
                grouped_head_tile,
                prediction_tile,
                sequence_tile,
                headdim,
                window_size_left,
                sm_count,
                bf16_eager=bf16_eager,
            )
            kv_splits = swa_plan.kv_splits
        else:
            # Preserve the established full-attention policy exactly.
            kv_splits = sm_count // base_ctas  # 1 wave
            kv_splits = max(1, kv_splits)
            if sm_count == 148 and base_ctas == 32:
                kv_splits = 9  # 2 waves
            # At least 256 tokens per split
            kv_splits = min(kv_splits, active_sequence_tiles)
            # Cluster reduction requires po2 splits
            if reduction == "atomic":
                kv_splits = max(
                    split for split in (1, 2, 4, 8, 16) if split <= kv_splits
                )

    # Automatic reduction mode
    if reduction == "auto":
        if window_size_left is not None:
            # Direct is safe for one split.  Multi-split SWA stays deterministic;
            # the existing cluster atomic path requires O to be cleared before
            # every invocation and is therefore not a general callable fast path.
            reduction = "direct" if kv_splits == 1 else "kernel"
        elif o_dtype in (Float32, Float16, BFloat16) and kv_splits in (1, 2, 4, 8):
            reduction = "atomic"
        else:
            reduction = "kernel"
    atomic = reduction == "atomic"
    kernel_reduction = reduction == "kernel"
    if atomic and iterations > 0:
        raise ValueError(
            "atomic reduction requires output.zero_() before every launch; "
            "use --iterations 0 for correctness or --reduction kernel for timing"
        )

    paged_plan = (
        None
        if page_size is None
        else _paged_cache_plan(batches, seqlen, page_size, seed)
    )

    print(
        f"Command: python {__file__.split('/')[-1]}"
        f" --d {headdim} --h_q {heads_q} --h_k {heads_k}"
        f" --b {batches} --p {prediction} --s {seqlen}"
        f"{f' --page-size {page_size}' if page_size is not None else ''}"
        f" --rel_bias_extent {rel_bias_extent}"
        f"{f' --window_size_left {window_size_left}' if window_size_left is not None else ''}"
        f" --kv_splits {kv_splits} --reduction {reduction}"
        f" --qk_mode {qk_mode} --mma_dtype {qkv_dtype}"
        f" --v_mode {v_mode}{' --v_dequant' if v_dequant else ''}"
        f" --pv_dtype {pv_dtype} --out_dtype {o_dtype}"
        f" --atol {tolerance}{' --skip_ref_check' if skip_ref_check else ''}"
        f" --scale {scale_s}"
        f" --sequence_tile {sequence_tile}"
        f" --iterations {iterations} --warmups {warmup_iterations}{' --use_warm_l2' if use_warm_l2 else ''}"
        f"{' --no_cuda_graphs' if not use_cuda_graphs else ''}"
        f"{' --quiet' if quiet else ''}"
    )

    if not quiet:
        swa_plan_description = (
            ""
            if swa_plan is None
            else "\tSWA active tiles/query tile: "
            f"{swa_plan.active_tiles_per_query_tile}; "
            f"direct tile limit: {swa_plan.direct_tile_limit}; "
            f"candidates: {swa_plan.candidates}\n"
        )
        print(
            "Running Blackwell SM100 GQA Decode test with:\n"
            f"\theaddim: {headdim}\theads_q: {heads_q}\theads_k: {heads_k}\n"
            f"\tbatches: {batches}\tprediction: {prediction}\tseqlen: {seqlen}\n"
            f"\tKV storage: {'dense' if page_size is None else f'paged ({page_size} tokens/page)'}\n"
            f"\trelative bias extent: {rel_bias_extent}\n"
            f"\tcausal window left: {window_size_left}\n"
            f"\tkv_splits: {kv_splits}\treduction: {reduction}\n"
            f"\tdecode CTAs: {base_ctas * kv_splits} for {sm_count} SMs "
            f"(base CTAs: {base_ctas})\n"
            + swa_plan_description
            + f"\tqk: {qkv_dtype}\tv storage: {v_storage_dtype}\t"
            f"pv: {pv_dtype}\to: {o_dtype}\t\n"
            f"\tatol: {tolerance if not skip_ref_check else 'skip'}"
            f"\tscale_s: {f'1 / sqrt({headdim})' if scale_s == 0 else scale_s}\n"
            f"\titerations: {iterations}\twarmups: {warmup_iterations}\tL2 warm: {use_warm_l2}\n"
            f"\tCUDA graph timing: {use_cuda_graphs}"
        )

    # Automatic scale
    if scale_s == 0:
        scale_s = headdim**-0.5

    #
    # Config Kernel
    #
    fmha = FlashAttentionDecodeSm100Bias(
        headdim,
        grouped_head_tile,
        prediction_tile=prediction_tile,
        sequence_tile=sequence_tile,
        reduction_mode=reduction,
        qk_blockscaled=qk_blockscaled,
        v_dequant=v_dequant,
        v_mma_dtype=pv_dtype,
        window_size_left=window_size_left,
        page_size=page_size,
    )

    seqlen_q = prediction
    seqlen_k = seqlen
    qo_shape = (kv_splits, batches, seqlen_q, heads_q, headdim)
    logical_kv_shape = (
        (batches, seqlen_k, heads_k, headdim)
        if paged_plan is None
        else (
            batches,
            paged_plan["padded_seqlen"],
            heads_k,
            headdim,
        )
    )
    kv_shape = (
        logical_kv_shape
        if paged_plan is None
        else (
            paged_plan["physical_pages"],
            page_size,
            heads_k,
            headdim,
        )
    )
    bias_shape = (batches, seqlen_q, heads_q, rel_bias_extent)
    k_sf_shape = None
    k_sf_stride = None
    v_sf_shape = None
    if paged_plan is not None and qk_blockscaled:
        k_sf_groups = math.ceil(headdim / 32)
        k_sf_shape = (*kv_shape[:3], k_sf_groups)
        k_sf_stride = (
            page_size * heads_k * k_sf_groups,
            k_sf_groups,
            page_size * k_sf_groups,
            1,
        )
    if paged_plan is not None and v_dequant:
        v_sf_shape = (
            paged_plan["physical_pages"],
            math.ceil(page_size / 32),
            heads_k,
            headdim,
        )

    fmha.can_implement(
        qo_shape[0],
        qo_shape[1:],
        kv_shape,
        bias_shape,
        qkv_dtype,
        v_storage_dtype,
        o_dtype,
        Float8E8M0FNU if qk_blockscaled else None,
        Float8E8M0FNU if v_dequant else None,
        v_shape_bshd=kv_shape,
        v_sf_shape=v_sf_shape,
        k_sf_shape=k_sf_shape,
        k_sf_stride=k_sf_stride,
    )

    #
    # Allocate Tensors
    #
    torch_ref_dtype = torch.float16
    torch_device = torch.device("cuda")
    torch.manual_seed(seed)

    def create_tensor(shape, dtype, init=None):
        init_type = cutlass.torch.TensorInitType.SKIP
        init_config = None
        if isinstance(init, int) or isinstance(init, float):
            init_type = cutlass.torch.TensorInitType.SCALAR
            init_config = cutlass.torch.ScalarInitConfig(value=init)
        elif isinstance(init, tuple) or isinstance(init, list):
            if len(init) == 2:
                init_type = cutlass.torch.TensorInitType.RANDOM
                init_config = cutlass.torch.RandomInitConfig(
                    min_val=init[0], max_val=init[1]
                )
            if len(init) == 3:
                init_type = cutlass.torch.TensorInitType.GAUSSIAN
                init_config = cutlass.torch.RandomInitConfig(
                    mean=init[0], std=init[1], scale=init[2]
                )

        ref_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
            shape,
            torch_ref_dtype,
            permute_order=None,
            init_type=init_type,
            init_config=init_config,
            device=torch_device,
        )

        cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
            ref_torch_tensor,
            dtype,
            is_dynamic_layout=True,
            assumed_align=16,
        )

        return (
            ref_torch_tensor,
            cute_tensor,
            torch_tensor,
        )

    q_ref, q_cute, q_torch = create_tensor(qo_shape[1:], qkv_dtype, init=qk_init_range)
    k_ref, k_cute, k_torch = create_tensor(
        logical_kv_shape, qkv_dtype, init=qk_init_range
    )
    v_ref, v_cute, v_torch = create_tensor(
        logical_kv_shape,
        v_storage_dtype,
        init=[-8, 7] if v_init is None else v_init,
    )
    logical_k_torch = k_torch
    logical_v_torch = v_torch
    if paged_plan is not None and paged_plan["padded_seqlen"] > seqlen:
        logical_k_torch[:, seqlen:].zero_()
        logical_v_torch[:, seqlen:].zero_()
    _, o_cute, o_torch = create_tensor(
        qo_shape[1:], o_dtype, init=(0 if atomic else None)
    )
    bias_ref, bias_cute, bias_torch = create_tensor(bias_shape, pv_dtype, init=[-2, 2])
    if bias_mode == "zero":
        bias_ref.zero_()
        bias_torch.zero_()
    elif bias_mode == "impulse":
        bias_torch.zero_()
        head_sign = 2 * (torch.arange(heads_q, device=torch_device) % 2) - 1
        bias_torch[..., 0] = 1.5 * head_sign
        bias_torch[..., -1] = -1.25 * head_sign
        if rel_bias_extent > 2:
            bias_torch[..., rel_bias_extent // 2] = 0.75
        bias_ref.copy_(bias_torch)
    elif bias_mode == "structured":
        b_idx = torch.arange(batches, device=torch_device)[:, None, None, None]
        q_idx = torch.arange(prediction, device=torch_device)[None, :, None, None]
        h_idx = torch.arange(heads_q, device=torch_device)[None, None, :, None]
        r_idx = torch.arange(rel_bias_extent, device=torch_device)[None, None, None, :]
        pattern = ((7 * b_idx + 5 * q_idx + 3 * h_idx + r_idx) % 17 - 8) * 0.125
        bias_ref.copy_(pattern)
        bias_torch.copy_(pattern)
    else:
        # Match the standalone prefill runner's compact relative-bias range.
        bias_ref.mul_(0.25)
        bias_torch.mul_(0.25)
    q_sf_cute = k_sf_cute = None
    q_sf_torch = k_sf_torch = None
    q_sf_ref = k_sf_ref = None
    v_sf_cute = v_sf_torch = v_sf_ref = None
    logical_k_sf_pages = logical_v_sf_pages = None
    if qk_blockscaled:
        q_sf_flat, q_sf_cute, q_sf_torch = create_mxfp8_scale_factor_tensor(
            prediction * grouped_head_tile,
            headdim,
            batches * heads_k,
            device=torch_device,
            pattern=scale_pattern,
            phase=0,
        )
        q_sf_ref = (
            q_sf_flat.reshape(prediction, grouped_head_tile, headdim, batches, heads_k)[
                :, :grouped_heads
            ]
            .permute(3, 0, 4, 1, 2)
            .reshape(batches, prediction, heads_q, headdim)
        )
        if paged_plan is None:
            k_sf_flat, k_sf_cute, k_sf_torch = create_mxfp8_scale_factor_tensor(
                seqlen,
                headdim,
                batches * heads_k,
                device=torch_device,
                pattern=scale_pattern,
                phase=1,
            )
            k_sf_ref = (
                k_sf_flat.reshape(seqlen, headdim, batches, heads_k)
                .permute(2, 0, 3, 1)
                .contiguous()
            )
        else:
            logical_k_sf_pages = _paged_scale_values(
                (
                    batches,
                    paged_plan["pages_per_batch"],
                    page_size,
                    heads_k,
                    math.ceil(headdim / 32),
                ),
                scale_pattern,
                1,
                torch_device,
                "k",
                page_size,
            )
            k_sf_ref = (
                logical_k_sf_pages.repeat_interleave(32, dim=-1)[..., :headdim]
                .reshape(logical_kv_shape)
                .contiguous()
            )
        q_ref = q_torch.float() * q_sf_ref
        k_ref = logical_k_torch.float() * k_sf_ref
    else:
        q_ref = q_torch.float()
        k_ref = logical_k_torch.float()
    if v_dequant:
        if paged_plan is None:
            v_sf_flat, v_sf_cute, v_sf_torch = create_mxfp8_scale_factor_tensor(
                headdim,
                seqlen,
                batches * heads_k,
                device=torch_device,
                pattern=scale_pattern,
                phase=2,
            )
            v_sf_ref = (
                v_sf_flat.reshape(headdim, seqlen, batches, heads_k)
                .permute(2, 1, 3, 0)
                .contiguous()
            )
        else:
            logical_v_sf_pages = _paged_scale_values(
                (
                    batches,
                    paged_plan["pages_per_batch"],
                    math.ceil(page_size / 32),
                    heads_k,
                    headdim,
                ),
                scale_pattern,
                2,
                torch_device,
                "v",
                page_size,
            )
            v_sf_ref = (
                logical_v_sf_pages.repeat_interleave(32, dim=2)[:, :, :page_size]
                .reshape(logical_kv_shape)
                .contiguous()
            )
        pv_torch_dtype = cutlass_torch.dtype(pv_dtype)
        v_ref = (
            logical_v_torch.to(pv_torch_dtype) * v_sf_ref.to(pv_torch_dtype)
        ).float()
    else:
        v_ref = logical_v_torch.float()

    seq_used_k_cute = page_table_cute = table_offsets_cute = None
    seq_used_k_torch = page_table_torch = table_offsets_torch = None
    if paged_plan is not None:
        page_table_values = paged_plan["page_table"]
        table_offset_values = paged_plan["table_offsets"]
        logical_k_pages = logical_k_torch.reshape(
            batches,
            paged_plan["pages_per_batch"],
            page_size,
            heads_k,
            headdim,
        )
        logical_v_pages = logical_v_torch.reshape(logical_k_pages.shape)
        k_torch = _scatter_paged_pages(
            logical_k_pages, page_table_values, table_offset_values
        )
        v_torch = _scatter_paged_pages(
            logical_v_pages, page_table_values, table_offset_values
        )
        k_cute = _to_cute_host_tensor(k_torch)
        v_cute = _to_cute_host_tensor(v_torch)

        if logical_k_sf_pages is not None:
            k_sf_source = torch.full(
                (
                    paged_plan["physical_pages"],
                    heads_k,
                    page_size,
                    math.ceil(headdim / 32),
                ),
                16.0,
                dtype=torch.float32,
                device=torch_device,
            ).permute(0, 2, 1, 3)
            k_sf_source.copy_(
                _scatter_paged_pages(
                    logical_k_sf_pages,
                    page_table_values,
                    table_offset_values,
                    fill_value=16.0,
                )
            )
            k_sf_cute, k_sf_torch = cutlass_torch.cute_tensor_like(
                k_sf_source,
                Float8E8M0FNU,
                is_dynamic_layout=True,
                assumed_align=16,
            )
        if logical_v_sf_pages is not None:
            v_sf_source = _scatter_paged_pages(
                logical_v_sf_pages,
                page_table_values,
                table_offset_values,
                fill_value=16.0,
            )
            v_sf_cute, v_sf_torch = cutlass_torch.cute_tensor_like(
                v_sf_source,
                Float8E8M0FNU,
                is_dynamic_layout=True,
                assumed_align=16,
            )

        seq_used_k_torch = torch.full(
            (batches,), seqlen, dtype=torch.int32, device=torch_device
        )
        page_table_torch = torch.tensor(
            page_table_values, dtype=torch.int32, device=torch_device
        )
        table_offsets_torch = torch.tensor(
            table_offset_values, dtype=torch.int32, device=torch_device
        )
        seq_used_k_cute = _to_cute_host_tensor(seq_used_k_torch, assumed_align=4)
        page_table_cute = _to_cute_host_tensor(page_table_torch, assumed_align=4)
        table_offsets_cute = _to_cute_host_tensor(table_offsets_torch, assumed_align=4)
    bias_ref = bias_torch.float()
    # Workspace tensors
    o_partial_cute = m_partial_cute = l_partial_cute = None
    if kernel_reduction:
        _, o_partial_cute, o_partial_torch = create_tensor(qo_shape, acc_dtype)
        _, m_partial_cute, m_partial_torch = create_tensor(qo_shape[:-1], acc_dtype)
        _, l_partial_cute, l_partial_torch = create_tensor(qo_shape[:-1], acc_dtype)

    #
    # Compile
    #
    current_stream = cutlass_torch.default_stream()
    compiled_fmha = cute.compile(
        fmha,
        kv_splits,
        q_cute,
        k_cute,
        v_cute,
        q_sf_cute,
        k_sf_cute,
        v_sf_cute,
        o_cute,
        bias_cute,
        o_partial_cute,
        m_partial_cute,
        l_partial_cute,
        scale_s,
        None,  # mCuSeqlensQ
        None,  # mCuSeqlensK
        None,  # mSeqUsedQ
        seq_used_k_cute,
        None,  # max_seqlen_q
        None,  # max_seqlen_k
        page_table_cute,
        table_offsets_cute,
        current_stream,
    )
    print("Finished Compiling")

    #
    # Refcheck
    #
    def run_torch_fmha(q_bshd, k_bshd, v_bshd, bias_bshr, scale_s):
        """Bottom-right causal/SWA GQA reference with compact relative bias."""
        qf = q_bshd.float().transpose(1, 2)
        kf = k_bshd.float().transpose(1, 2)
        vf = v_bshd.float().transpose(1, 2)
        if heads_q != heads_k:
            repeats = heads_q // heads_k
            kf = kf.repeat_interleave(repeats, dim=1)
            vf = vf.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale_s

        q_idx = torch.arange(prediction, device=q_bshd.device)[:, None]
        k_idx = torch.arange(seqlen, device=q_bshd.device)[None, :]
        rel_idx = q_idx + (seqlen - prediction) - k_idx
        gather_idx = rel_idx.clamp(0, rel_bias_extent - 1)
        gather_idx = gather_idx[None, None].expand(batches, heads_q, -1, -1)
        bias = bias_bshr.permute(0, 2, 1, 3).gather(-1, gather_idx)
        valid_bias = (rel_idx >= 0) & (rel_idx < rel_bias_extent)
        scores += bias.float().masked_fill(~valid_bias[None, None], 0.0)

        causal_mask = k_idx > q_idx + (seqlen - prediction)
        if window_size_left is not None:
            causal_mask |= rel_idx > window_size_left
        scores.masked_fill_(causal_mask[None, None], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = probabilities.to(cutlass_torch.dtype(pv_dtype)).float()
        return torch.matmul(probabilities, vf).transpose(1, 2)

    o_ref = None
    bias_effect = scale_effect = v_scale_effect = None
    if not skip_ref_check:
        # Execute kernel once for reference checking
        print("Running...")
        compiled_fmha(
            kv_splits,
            q_cute,
            k_cute,
            v_cute,
            q_sf_cute,
            k_sf_cute,
            v_sf_cute,
            o_cute,
            bias_cute,
            o_partial_cute,
            m_partial_cute,
            l_partial_cute,
            scale_s,
            None,
            None,
            None,
            seq_used_k_cute,
            None,
            None,
            page_table_cute,
            table_offsets_cute,
            current_stream,
        )
        print("Verifying results...")
        if qk_blockscaled or v_dequant:
            # PyTorch Inductor cannot reliably lower FP8 source casts yet.
            reference_fn = run_torch_fmha
        else:
            reference_fn = torch.compile(run_torch_fmha, mode="max-autotune")
        o_ref = reference_fn(
            q_ref, k_ref[:, :seqlen], v_ref[:, :seqlen], bias_ref, scale_s
        )
        output_fp32 = o_torch.float()
        max_abs_error = (o_ref - output_fp32).abs().max().item()
        torch.testing.assert_close(output_fp32, o_ref, atol=tolerance, rtol=1e-05)
        print(f"PASS (max abs error: {max_abs_error:.6f})")
        if bias_mode != "zero":
            o_ref = o_ref.clone()
            no_bias_ref = reference_fn(
                q_ref,
                k_ref[:, :seqlen],
                v_ref[:, :seqlen],
                torch.zeros_like(bias_ref),
                scale_s,
            )
            bias_effect = (o_ref - no_bias_ref).abs().max().item()
            print(f"Relative-bias negative control: max effect {bias_effect:.6f}")
        if qk_blockscaled:
            unscaled_ref = run_torch_fmha(
                q_torch.float(),
                logical_k_torch[:, :seqlen].float(),
                v_ref[:, :seqlen],
                bias_ref,
                scale_s,
            )
            scale_effect = (o_ref - unscaled_ref).abs().max().item()
            if scale_effect <= 1e-3:
                raise AssertionError(
                    "MXFP8 negative control did not observe UE8M0 scale factors"
                )
            print(
                "MXFP8 scale negative control: "
                f"ignoring UE8M0 changes output by {scale_effect:.6f}"
            )
        if v_dequant and scale_pattern != "unit":
            unscaled_v = (
                logical_v_torch[:, :seqlen].to(cutlass_torch.dtype(pv_dtype)).float()
            )
            unscaled_v_ref = run_torch_fmha(
                q_ref, k_ref[:, :seqlen], unscaled_v, bias_ref, scale_s
            )
            v_scale_effect = (o_ref - unscaled_v_ref).abs().max().item()
            if v_scale_effect <= 1e-3:
                raise AssertionError("FP8 V negative control did not observe SFV")
            print(
                "FP8 V scale negative control: "
                f"ignoring SFV changes output by {v_scale_effect:.6f}"
            )

    else:
        print("SKIP")

    if return_debug_tensors:
        if o_ref is None:
            raise ValueError("return_debug_tensors requires reference checking")
        return {
            "q": q_torch,
            "k": logical_k_torch[:, :seqlen],
            "v": logical_v_torch[:, :seqlen],
            "physical_k": k_torch if paged_plan is not None else None,
            "physical_v": v_torch if paged_plan is not None else None,
            "page_table": page_table_torch,
            "page_table_offsets": table_offsets_torch,
            "seq_used_k": seq_used_k_torch,
            "bias": bias_torch,
            "output": o_torch,
            "reference": o_ref,
            "q_sf_elementwise": q_sf_ref,
            "k_sf_elementwise": (None if k_sf_ref is None else k_sf_ref[:, :seqlen]),
            "q_sf_storage": q_sf_torch,
            "k_sf_storage": k_sf_torch,
            "v_sf_elementwise": (None if v_sf_ref is None else v_sf_ref[:, :seqlen]),
            "v_sf_storage": v_sf_torch,
            "kv_splits": kv_splits,
            "sequence_tile": sequence_tile,
            "grouped_head_tile": grouped_head_tile,
            "prediction_tile": prediction_tile,
            "bias_effect": bias_effect,
            "scale_effect": scale_effect,
            "v_scale_effect": v_scale_effect,
        }

    #
    # Profile
    #
    if iterations <= 0:
        return 0.0

    # Create non-default stream for CUDA graph profiling
    torch_stream = torch.cuda.Stream()
    profile_stream = cuda.CUstream(torch_stream.cuda_stream)

    def workspace_generator():
        _, q_cute, _ = create_tensor(qo_shape[1:], qkv_dtype, init=[-8, 7])
        workspace_q_sf_cute = q_sf_cute
        workspace_k_sf_cute = k_sf_cute
        workspace_v_sf_cute = v_sf_cute
        if q_sf_torch is not None:
            workspace_q_sf_cute = _to_cute_host_tensor(q_sf_torch.clone())
        if k_sf_torch is not None:
            workspace_k_sf_cute = _to_cute_host_tensor(k_sf_torch.clone())
        if v_sf_torch is not None:
            workspace_v_sf_cute = _to_cute_host_tensor(v_sf_torch.clone())
        workspace_seq_used_k_cute = None
        workspace_page_table_cute = None
        workspace_table_offsets_cute = None
        if paged_plan is None:
            _, k_cute, _ = create_tensor(kv_shape, qkv_dtype, init=[-8, 7])
            _, v_cute, _ = create_tensor(kv_shape, v_storage_dtype, init=[-8, 7])
        else:
            k_cute = _to_cute_host_tensor(k_torch.clone())
            v_cute = _to_cute_host_tensor(v_torch.clone())
            workspace_seq_used_k_cute = _to_cute_host_tensor(
                seq_used_k_torch.clone(), assumed_align=4
            )
            workspace_page_table_cute = _to_cute_host_tensor(
                page_table_torch.clone(), assumed_align=4
            )
            workspace_table_offsets_cute = _to_cute_host_tensor(
                table_offsets_torch.clone(), assumed_align=4
            )
        _, o_cute, _ = create_tensor(
            qo_shape[1:], o_dtype, init=(0 if atomic else None)
        )
        _, bias_cute, _ = create_tensor(bias_shape, pv_dtype, init=[-2, 2])
        o_partial_cute = m_partial_cute = l_partial_cute = None
        if kernel_reduction:
            _, o_partial_cute, _ = create_tensor(qo_shape, acc_dtype)
            _, m_partial_cute, _ = create_tensor(qo_shape[:-1], acc_dtype)
            _, l_partial_cute, _ = create_tensor(qo_shape[:-1], acc_dtype)
        return testing.JitArguments(
            kv_splits,
            q_cute,
            k_cute,
            v_cute,
            workspace_q_sf_cute,
            workspace_k_sf_cute,
            workspace_v_sf_cute,
            o_cute,
            bias_cute,
            o_partial_cute,
            m_partial_cute,
            l_partial_cute,
            scale_s,
            None,
            None,
            None,
            workspace_seq_used_k_cute,
            None,
            None,
            workspace_page_table_cute,
            workspace_table_offsets_cute,
            profile_stream,
        )

    workspace_count = 1
    qkvo_bytes = (
        q_torch.nbytes
        + logical_k_torch[:, :seqlen].nbytes
        + logical_v_torch[:, :seqlen].nbytes
        + o_torch.nbytes
        + bias_torch.nbytes
    )
    allocated_workspace_bytes = (
        q_torch.nbytes
        + k_torch.nbytes
        + v_torch.nbytes
        + o_torch.nbytes
        + bias_torch.nbytes
    )
    for auxiliary in (
        q_sf_torch,
        k_sf_torch,
        v_sf_torch,
        seq_used_k_torch,
        page_table_torch,
        table_offsets_torch,
    ):
        if auxiliary is not None:
            allocated_workspace_bytes += auxiliary.nbytes
    if not use_warm_l2:
        one_workspace_bytes = allocated_workspace_bytes
        if kernel_reduction:
            one_workspace_bytes += (
                o_partial_torch.nbytes + m_partial_torch.nbytes + l_partial_torch.nbytes
            )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    runtime_us = testing.benchmark(
        compiled_fmha,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        stream=profile_stream,
        workspace_generator=workspace_generator,
        workspace_count=workspace_count,
        use_cuda_graphs=use_cuda_graphs,
    )

    # Print throughputs
    terabytes_per_s = qkvo_bytes / runtime_us * 1.0e-6
    flops = batches * heads_q * seqlen_q * seqlen_k * headdim * 2 * 2
    teraflops_per_s = flops / runtime_us * 1.0e-6

    print(
        f"{runtime_us:.3f} us\n"
        f"{terabytes_per_s:.3f} TB/s\n"
        f"{teraflops_per_s:.3f} TFLOPS/s"
    )

    return (runtime_us, teraflops_per_s, terabytes_per_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example of MHA/GQA decode on Blackwell."
    )

    parser.add_argument(
        "--batches",
        "--batch",
        "--b",
        type=int,
        default=1,
        help="batch size",
    )

    parser.add_argument(
        "--prediction",
        "--p",
        type=int,
        default=4,
        help="number of predicted tokens",
    )

    parser.add_argument(
        "--seqlen",
        "--seq",
        "--s",
        type=int,
        default=10240,
        help="key/value sequence length",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        choices=(8, 16, 32, 64, 128, 256),
        default=None,
        help="physical KV page size; omit for dense KV",
    )

    parser.add_argument(
        "--heads_q",
        "--h_q",
        type=int,
        default=32,
        help="query heads",
    )

    parser.add_argument(
        "--heads_k",
        "--h_k",
        type=int,
        default=4,
        help="key/value heads",
    )

    parser.add_argument(
        "--headdim",
        "--d",
        type=int,
        default=128,
        help="head dimension",
    )

    parser.add_argument(
        "--rel_bias_extent",
        "--bias_extent",
        type=int,
        default=128,
        help="number of compact bottom-right relative-bias entries per query",
    )

    parser.add_argument(
        "--window_size_left",
        "--window-left",
        type=int,
        default=None,
        help="causal sliding-window left extent; L includes keys center-L through center",
    )

    parser.add_argument(
        "--kv_splits",
        "--splits",
        type=int,
        default=0,
        help="threadblocks per sequence",
    )

    parser.add_argument(
        "--sequence_tile",
        "--sequence-tile",
        type=int,
        default=256,
        help="KV tokens consumed by one threadblock loop iteration",
    )

    parser.add_argument(
        "--reduction",
        type=str,
        default="auto",
        help="split KV reduction mode, can be kernel, atomic, or auto",
    )

    parser.add_argument(
        "--qkv_dtype",
        "--mma_dtype",
        type=cutlass.dtype,
        default=BFloat16,
        help="Q/K storage data type (dense mode also uses it for dense V)",
    )

    parser.add_argument(
        "--qk_mode",
        choices=("dense", "MXFP8"),
        default="MXFP8",
        help="use dense Q/K or true E4M3+UE8M0 block-scaled MXFP8 Q/K",
    )

    parser.add_argument(
        "--v_mode",
        choices=("dense", "MXFP8"),
        default="dense",
        help="store V densely or as FP8 plus UE8M0 SFV",
    )

    parser.add_argument(
        "--v_dequant",
        "--v-dequant",
        action="store_true",
        help=(
            "enable FP8 V plus UE8M0 SFV storage and dequantize V to "
            "BF16/FP16 before PV MMA"
        ),
    )

    parser.add_argument(
        "--pv_dtype",
        type=cutlass.dtype,
        default=None,
        help="P/V dtype for MXFP8 mode (default: BFloat16)",
    )

    parser.add_argument(
        "--o_dtype",
        "--out_dtype",
        type=cutlass.dtype,
        default=BFloat16,
        help="output data type",
    )

    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=Float32,
        help="accumulator/reduction data type",
    )

    parser.add_argument(
        "--tolerance",
        "--atol",
        type=float,
        default=1e-01,
        help="Absolute tolerance for validation",
    )

    parser.add_argument(
        "--scale_s",
        "--scale",
        type=float,
        default=0,
        help="score (Q*K) scale factor; if zero, defaults to 1/sqrt(D)",
    )

    parser.add_argument(
        "--warmup_iterations",
        "--warmups",
        type=int,
        default=10,
        help="Number of iterations for warmup",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations after warmup",
    )

    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference check",
    )

    parser.add_argument(
        "--use_warm_l2",
        action="store_true",
        help="dont rotate profiling workspace and dont flush L2 before profiling",
    )

    parser.add_argument(
        "--no_cuda_graphs",
        dest="use_cuda_graphs",
        action="store_false",
        help="time queued direct compiled calls instead of CUDA graph replay",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose prints",
    )

    kwargs = vars(parser.parse_args())

    run(**kwargs)
