# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CuTe-DSL helpers adapted from vLLM's ``vllm/cute_utils`` package."""

import torch
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Int64,
    Uint32,
    cute,
)
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl._mma import (
    _fp8x4_to_fp16x4,
    _mma_sync,
)

_TORCH_TO_CUTE_DTYPE = {
    torch.bfloat16: BFloat16,
    torch.float8_e4m3fn: Float8E4M3FN,
}


# https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
EVICT_FIRST = Int64(0x12F0000000000000)


def simple_tma_copy(atom, src, dst, mbar=None, cache_policy=None):
    """A simple helper that wraps group_modes() and tma_partition().

    NOTE: this should be called WITHOUT cute.elect_one().
    """
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        gmem = src
        smem = dst
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        smem = src
        gmem = dst
    else:
        raise ValueError

    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )

    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar, cache_policy=cache_policy)
    elif isinstance(atom.op, cpasync.CopyBulkTensorTileS2GOp):
        cute.copy(atom, s_part, g_part, cache_policy=cache_policy)
    else:
        raise ValueError


@dsl_user_op
def mma_sync(a, b, c: cute.Tensor, *, loc=None, ip=None):
    return _mma_sync(a, b, c, loc=loc, ip=ip, vector_kwargs={"ip": ip})


@dsl_user_op
def fp8x4_to_fp16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    return _fp8x4_to_fp16x4(x, loc=loc, ip=ip, vector_kwargs={"ip": ip})
