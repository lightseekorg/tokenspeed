# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Copyright (c) 2026 LightSeek Foundation

"""Shared layouts and helpers for GFX1250 Gluon MLA kernels.

The device implementation is ported from ROCm/AITER commit
4a1cc773f34cbfc74387259e51262556ee38edd0.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

e4m3_info = torch.finfo(torch.float8_e4m3fn)
e5m2_info = torch.finfo(torch.float8_e5m2)


def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)):
        return (
            str(int(value))
            if isinstance(value, float) and value.is_integer()
            else str(value)
        )
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        return "_".join(_sanitize_constexpr_value(item) for item in items) or "NONE"
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned.upper() if cleaned else "NONE"


def make_kernel_repr(base_name, config_keys):
    """Build stable compiled-kernel names from Gluon constexpr arguments."""

    def _repr(specialization):
        parts = [
            f"{key}_{_sanitize_constexpr_value(specialization.constants.get(key))}"
            for key in config_keys
        ]
        return f"{base_name}_{'_'.join(parts)}" if parts else base_name

    return _repr


@gluon.constexpr_function
def absorbed_mla_layouts(
    KV_LORA_RANK,
    QK_ROPE_HEAD_DIM,
    BLOCK_SIZE,
    BLOCK_M,
    NUM_WARPS,
    WARP_SIZE,
    K_WIDTH,
):
    """Build operand, load, and LDS layouts for absorbed MLA."""
    assert WARP_SIZE == 32
    assert NUM_WARPS == 1 or NUM_WARPS == 2 or NUM_WARPS == 4 or NUM_WARPS == 8
    assert K_WIDTH == 8 or K_WIDTH == 16
    is_fp8 = K_WIDTH == 16
    instr_shape = [16, 16, 64] if is_fp8 else [16, 16, 32]

    if NUM_WARPS == 1:
        warp_bases_qk = []
        warp_bases_pv = []
    elif NUM_WARPS == 2:
        warp_bases_qk = [(1, 0)]
        warp_bases_pv = [(0, 1)]
    elif NUM_WARPS == 4:
        warp_bases_qk = [(1, 0), (2, 0)]
        warp_bases_pv = [(0, 1), (0, 2)]
    else:
        warp_bases_qk = [(1, 0), (2, 0), (4, 0)]
        warp_bases_pv = [(0, 1), (0, 2), (0, 4)]

    qk_wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases_qk,
        reg_bases=[],
        instr_shape=instr_shape,
    )
    pv_wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases_pv,
        reg_bases=[],
        instr_shape=instr_shape,
    )
    q_dot_layout = gl.DotOperandLayout(0, qk_wmma_layout, k_width=K_WIDTH)
    k_dot_layout = gl.DotOperandLayout(1, qk_wmma_layout, k_width=K_WIDTH)
    p_dot_layout = gl.DotOperandLayout(0, pv_wmma_layout, k_width=8)
    v_dot_layout = gl.DotOperandLayout(1, pv_wmma_layout, k_width=8)

    q_lora_shared_layout = gl.PaddedSharedLayout.with_identity_for(
        [[KV_LORA_RANK, K_WIDTH]], [BLOCK_M, KV_LORA_RANK], [1, 0]
    )
    q_rope_shared_layout = gl.PaddedSharedLayout.with_identity_for(
        [[QK_ROPE_HEAD_DIM, K_WIDTH]], [BLOCK_M, QK_ROPE_HEAD_DIM], [1, 0]
    )
    kv_lora_shared_layout = gl.PaddedSharedLayout.with_identity_for(
        [[KV_LORA_RANK, K_WIDTH]], [BLOCK_SIZE, KV_LORA_RANK], [1, 0]
    )
    k_rope_shared_layout = gl.PaddedSharedLayout.with_identity_for(
        [[QK_ROPE_HEAD_DIM, K_WIDTH]], [BLOCK_SIZE, QK_ROPE_HEAD_DIM], [1, 0]
    )

    load_vec = K_WIDTH
    lora_threads = min(max(KV_LORA_RANK // load_vec, 1), WARP_SIZE)
    q_lora_load_layout = gl.BlockedLayout(
        [1, load_vec],
        [WARP_SIZE // lora_threads, lora_threads],
        [NUM_WARPS, 1],
        [1, 0],
    )
    rope_threads = min(max(QK_ROPE_HEAD_DIM // load_vec, 1), WARP_SIZE)
    q_rope_load_layout = gl.BlockedLayout(
        [1, load_vec],
        [WARP_SIZE // rope_threads, rope_threads],
        [NUM_WARPS, 1],
        [1, 0],
    )

    return (
        qk_wmma_layout,
        pv_wmma_layout,
        q_dot_layout,
        k_dot_layout,
        p_dot_layout,
        v_dot_layout,
        q_lora_load_layout,
        q_rope_load_layout,
        q_lora_shared_layout,
        q_rope_shared_layout,
        kv_lora_shared_layout,
        k_rope_shared_layout,
    )


@gluon.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@gluon.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: gl.constexpr,
    use_q_block_mode: gl.constexpr,
):
    left: gl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = gl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1
