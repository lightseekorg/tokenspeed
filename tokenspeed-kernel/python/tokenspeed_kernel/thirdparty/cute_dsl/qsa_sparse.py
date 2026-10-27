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

"""Blackwell QSA attention with grouped query and KV operand reuse."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_tensor
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_sparse.common import (
    _CONVERT_WARPGROUPS,
    _HEAD_DIM,
    _SELECTED_WIDTH,
    _SMALL_MAX_CLUSTERS,
    _can_group_six_queries,
    _num_splits,
    _scalar,
    _select_head_tile,
    _select_launch_config,
    _sm_count,
    _to_tvm_meta,
    _wide_cluster_capacity,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_sparse.kernel import (
    MixedInputFusedMultiHeadAttentionDecode,
)
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_sparse.prefix import (
    _prepare_query_groups,
)

_COMPILED_KERNELS = {}
_WARMED = set()


def _query_group_size(
    num_rows: int,
    num_kv_heads: int,
    heads_per_kv: int,
    max_seqlen_q: int,
    bf16_kv: bool,
    sm_count: int,
) -> int:
    """Group only consecutive queries in the same uniform request."""
    output_groups = num_rows * num_kv_heads
    if (
        not bf16_kv
        or heads_per_kv not in (6, 12)
        or max_seqlen_q % 2
        or output_groups < sm_count
    ):
        return 1
    if heads_per_kv == 12 and num_kv_heads == 2:
        if max_seqlen_q % 8 == 0 and output_groups >= sm_count * 6:
            return 8
        if max_seqlen_q % 6 == 0 and _can_group_six_queries(output_groups, sm_count):
            return 6
    if max_seqlen_q % 4 == 0 and output_groups >= sm_count * 3:
        return 4
    if heads_per_kv == 6 and max_seqlen_q % 3 == 0:
        return 3
    return 2


def _query_head_tile(heads_per_kv: int, head_tile: int, bf16_kv: bool) -> int:
    # Native 48/72/96 tiles pack twelve real heads per query without padding.
    if head_tile in (48, 72, 96):
        return heads_per_kv
    return _select_head_tile(heads_per_kv, bf16_kv, 1)


def _configuration(rows, q_heads, kv_heads, max_seqlen_q, bf16_kv, sm_count, device):
    heads_per_kv = q_heads // kv_heads
    query_group = _query_group_size(
        rows, kv_heads, heads_per_kv, max_seqlen_q, bf16_kv, sm_count
    )
    head_tile = _select_head_tile(heads_per_kv * query_group, bf16_kv, query_group)
    head_tiles_per_row = kv_heads * cute.ceil_div(heads_per_kv, head_tile)
    query_rows_per_cta, kv_splits, smem_slots, async_ready = _select_launch_config(
        rows,
        head_tiles_per_row,
        bf16_kv,
        sm_count,
        (
            _wide_cluster_capacity(device)
            if rows * head_tiles_per_row <= _SMALL_MAX_CLUSTERS
            else 0
        ),
    )
    return (
        head_tile,
        query_group,
        kv_splits,
        query_rows_per_cta,
        2 if head_tile in (48, 64, 72, 96) else smem_slots,
        smem_slots,
        async_ready,
        heads_per_kv == 12 and kv_heads == 2 and query_group == 2 and rows > 128,
    )


def warmup(query, key_cache, value_cache, *, max_rows, enable_pdl):
    """Warm every launch configuration for this model's cache and row budget.

    Args:
        query: BF16 query prototype supplying device and fixed head geometry.
        key_cache: Model's BF16 or FP8 key cache; its layout stays fixed.
        value_cache: Value cache matching the key cache.
        max_rows: Largest row count whose execution configurations to warm.
        enable_pdl: Programmatic dependent launch setting to compile.

    Returns:
        None. Compiled configurations remain cached for subsequent calls.
    """
    key = (
        query.device.index,
        query.shape[1],
        key_cache.shape,
        key_cache.dtype,
        max_rows,
        enable_pdl,
    )
    if key in _WARMED:
        return
    bf16_kv = key_cache.dtype is torch.bfloat16
    sm_count = _sm_count(query.device.index)
    configurations = {}
    # These are the divisibility cases in _query_group_size: width 1 disables
    # grouping, 2/4 select pairs/quads, 6 covers groups of three/six, and 8
    # reaches the widest group. Row counts independently select launch policy.
    for rows in range(1, max_rows + 1):
        for width in (1, 2, 4, 6, 8):
            if rows % width == 0:
                config = _configuration(
                    rows,
                    query.shape[1],
                    key_cache.shape[1],
                    width,
                    bf16_kv,
                    sm_count,
                    query.device.index,
                )
                configurations.setdefault(config, (rows, width))
    warmup_rows = max(rows for rows, _ in configurations.values())
    sample = query.new_zeros((warmup_rows, *query.shape[1:]))
    selected = torch.full(
        (warmup_rows, _SELECTED_WIDTH),
        -1,
        dtype=torch.int32,
        device=query.device,
    )
    for rows, width in configurations.values():
        kernel(
            sample[:rows],
            key_cache,
            value_cache,
            selected[:rows],
            scale=1.0,
            max_seqlen_q=width,
            k_scale=None,
            v_scale=None,
            enable_pdl=enable_pdl,
        )
    _WARMED.add(key)


def _compile_kernel(
    query: torch.Tensor,
    original_query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    output: torch.Tensor,
    kv_splits: int,
    enable_pdl: bool,
    query_rows_per_cta: int,
    bf16_smem_slots: int,
    bf16_async_ready: bool,
    head_tile: int,
    query_group: int,
    prefix_shared: torch.Tensor,
    distinct_bf16_smem_slots: int,
):
    bf16_kv = key_cache.dtype is torch.bfloat16
    kv_heads = key_cache.shape[1]
    heads_per_kv = original_query.shape[1] // kv_heads
    query_head_tile = _query_head_tile(heads_per_kv, head_tile, bf16_kv)
    problem_shape = (
        # Tensor layouts carry rows; the remaining dimensions are model facts.
        None,
        kv_heads * query_head_tile * query_group if query_group > 1 else query.shape[1],
        kv_heads,
        key_cache.shape[0],
        _HEAD_DIM,
    )
    native_head_tile = head_tile in (48, 72, 96)
    single_softmax_g2 = head_tile == 32 and query_group == 2 and heads_per_kv == 12
    softmax_warpgroups = (
        3 if native_head_tile else 2 if head_tile == 64 or query_group in (3, 4) else 1
    )
    if heads_per_kv == 12:
        register_budget = (
            (112, 48, 80)
            if native_head_tile and kv_heads == 2
            else (
                (104, 48, 72)
                if native_head_tile
                else (
                    (168, 48, 96)
                    if head_tile == 64
                    else (208, 80, 112) if single_softmax_g2 else None
                )
            )
        )
    else:
        register_budget = None
    fmha = MixedInputFusedMultiHeadAttentionDecode(
        kv_splits,
        enable_pdl,
        query_rows_per_cta,
        bf16_smem_slots,
        bf16_async_ready,
        head_tile,
        problem_shape,
        query_group,
        heads_per_kv,
        query_group > 1,
        False,
        softmax_warpgroups,
        _sm_count(query.device.index),
        register_budget=register_budget,
        distinct_bf16_smem_slots=distinct_bf16_smem_slots,
        launch_threads_per_cta=None,
        scalar_softmax_stats=head_tile in (72, 96),
        direct_query=query_group > 1,
        bucket_prefix=False,
        leader_first=head_tile in (72, 96),
        producer_warpgroups=(
            1
            if heads_per_kv == 12
            and kv_heads == 2
            and head_tile == 16
            and query_group == 1
            and bf16_async_ready
            and bf16_smem_slots == 3
            else _CONVERT_WARPGROUPS
        ),
        private_tail_workers=(
            2
            if query_group == 1
            and heads_per_kv == 6
            and kv_heads == 1
            and bf16_kv
            and kv_splits == 1
            and bf16_async_ready
            else 0
        ),
        cached_bf16_lookahead=False,
        work_sync_threads=0,
        prefix_in_cta=prefix_shared.numel() == 0,
        large_query_grid=original_query.shape[0] > 128,
    )

    def dynamic_rows(tensor):
        return _to_tvm_meta(tensor, 16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )

    prefix_meta = make_fake_tensor(
        cutlass.Int32,
        (cute.sym_int64(),),
        stride=(cute.sym_int64(),),
        assumed_align=4,
    )
    return cute.compile(
        fmha,
        dynamic_rows(query),
        dynamic_rows(original_query),
        _to_tvm_meta(key_cache, 16),
        _to_tvm_meta(value_cache, 16),
        make_fake_tensor(
            cutlass.Int32,
            (cute.sym_int64(divisibility=query_group), _SELECTED_WIDTH),
            stride=(cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        ),
        prefix_meta,
        dynamic_rows(output),
        1.0,
        1.0,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3",
    )


def kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int,
    k_scale: float | torch.Tensor | None,
    v_scale: float | torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor:
    """Run direct-slot QSA with exact prefix reuse within uniform requests.

    Query heads form contiguous groups for each KV head. Non-positive selected
    slots are masked, and each query retains its own last three candidates.
    K/V offsets are 64-bit. The enable_pdl flag is retained by CUDA graph capture.
    Returns BF16 output with the query's shape.
    """
    if query.ndim != 3 or query.shape[-1] != _HEAD_DIM:
        raise ValueError("QSA requires query shape [tokens, query_heads, 256]")
    if key_cache.ndim != 3 or key_cache.shape[-1] != _HEAD_DIM:
        raise ValueError("QSA requires key cache shape [slots, kv_heads, 256]")
    num_q_heads, num_kv_heads = query.shape[1], key_cache.shape[1]
    if num_q_heads < 1 or num_kv_heads < 1 or num_q_heads % num_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if num_q_heads == num_kv_heads:
        raise ValueError("QSA requires at least two query heads per KV head")
    if value_cache.shape != key_cache.shape:
        raise ValueError("QSA requires matching K/V cache shapes")
    if selected_slots.shape != (query.shape[0], _SELECTED_WIDTH):
        raise ValueError("QSA requires selected slots shape [tokens, 2051]")
    if query.dtype is not torch.bfloat16:
        raise TypeError("QSA requires BF16 query")
    if key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("QSA requires BF16 or FP8 E4M3FN K/V cache")
    if value_cache.dtype is not key_cache.dtype:
        raise TypeError("QSA requires matching K/V cache dtypes")
    if selected_slots.dtype is not torch.int32:
        raise TypeError("QSA requires int32 selected slots")
    if max_seqlen_q < 1 or query.shape[0] % max_seqlen_q:
        raise ValueError("max_seqlen_q must be positive and divide the query rows")
    if query.shape[0] == 0:
        return torch.empty_like(query)
    if not query.is_contiguous():
        query = query.contiguous()
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    scale_qs = float(scale) * _scalar(k_scale, 1.0)
    scale_o = _scalar(v_scale, 1.0)
    bf16_kv = key_cache.dtype is torch.bfloat16
    heads_per_kv = num_q_heads // num_kv_heads
    sm_count = _sm_count(query.device.index)
    config = _configuration(
        query.shape[0],
        num_q_heads,
        num_kv_heads,
        max_seqlen_q,
        bf16_kv,
        sm_count,
        query.device.index,
    )
    (
        head_tile,
        query_group,
        kv_splits,
        query_rows_per_cta,
        bf16_smem_slots,
        distinct_bf16_smem_slots,
        bf16_async_ready,
        _,
    ) = config
    query_head_tile = _query_head_tile(heads_per_kv, head_tile, bf16_kv)
    prepared_query = query
    prefix_shared = selected_slots[:, 0]
    if query_group > 1 and heads_per_kv == 12:
        if query_group == 2:
            # The empty flag selects exact detection inside the attention CTA.
            prefix_shared = (
                selected_slots[:0, 0]
                if num_kv_heads == 2
                else torch.empty((0,), dtype=torch.int32, device=query.device)
            )
        else:
            if num_kv_heads == 2 and query_head_tile == heads_per_kv:
                prepared_query = query.view(
                    query.shape[0] // query_group,
                    num_kv_heads * query_head_tile * query_group,
                    _HEAD_DIM,
                )
            prefix_shared = torch.empty(
                (query.shape[0] // query_group,), dtype=torch.int32, device=query.device
            )
            _prepare_query_groups(
                selected_slots,
                prefix_shared,
                num_kv_heads,
                query_group,
                enable_pdl,
            )
    cache_key = (
        query.device.index,
        num_q_heads,
        num_kv_heads,
        key_cache.shape[0],
        key_cache.dtype,
        config,
        enable_pdl,
    )
    compiled = _COMPILED_KERNELS.get(cache_key)
    if compiled is None:
        compiled = _compile_kernel(
            prepared_query,
            query,
            key_cache,
            value_cache,
            selected_slots,
            output,
            kv_splits,
            enable_pdl,
            query_rows_per_cta,
            bf16_smem_slots,
            bf16_async_ready,
            head_tile,
            query_group,
            prefix_shared,
            distinct_bf16_smem_slots,
        )
        _COMPILED_KERNELS[cache_key] = compiled
    compiled(
        prepared_query,
        query,
        key_cache.view(torch.uint8) if not bf16_kv else key_cache,
        value_cache.view(torch.uint8) if not bf16_kv else value_cache,
        selected_slots,
        prefix_shared,
        output,
        scale_qs,
        scale_o,
    )
    return output
