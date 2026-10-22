# Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the Triton project
# SPDX-License-Identifier: MIT
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

"""Local copies of small triton-kernels utilities used by AMD MoE kernels."""

# fmt: off
# isort: off
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from tokenspeed_kernel_amd._triton import tl, triton


# activation metadata
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FnSpecs:
    name: str
    fn: object
    fn_arg_names: tuple[str, ...]
    fn_arg_do_not_specialize: tuple[str, ...] = tuple()
    reduction_n: int = 1

    @staticmethod
    def default():
        return FnSpecs("dflt", None, tuple())


@dataclass(frozen=True)
class FusedActivation:
    specs: FnSpecs = FnSpecs.default()
    fn_args: tuple[object, ...] = tuple()


# swiglu
# ---------------------------------------------------------------------------- #
@triton.jit
def _swiglu_clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _compute_swiglu(gelu, linear, scale, alpha, limit):
    gelu = gelu.to(tl.float32) * scale
    if limit is not None:
        gelu = _swiglu_clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32) * scale
    if limit is not None:
        linear = _swiglu_clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp(-alpha * gelu))
    return tl.fma(s, linear, s)


@triton.jit(repr=lambda _: "_swiglu")
def swiglu_fn(input, alpha, limit):
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    return _compute_swiglu(gelu, linear, 1.0, alpha, limit)


# data types
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntegerType:
    bitwidth: int
    is_signed: bool


@dataclass(frozen=True)
class FloatType:
    bitwidth_exponent: int
    bitwidth_mantissa: int
    is_signed: bool
    unsigned_zero: bool = False

    @property
    def bitwidth(self):
        return int(self.is_signed) + self.bitwidth_exponent + self.bitwidth_mantissa


UINT8 = IntegerType(8, is_signed=False)
FP4 = FloatType(bitwidth_exponent=2, bitwidth_mantissa=1, is_signed=True)
FP8_E4M3FN = FloatType(bitwidth_exponent=4, bitwidth_mantissa=3, is_signed=True)
FP8_E4M3FNUZ = FloatType(
    bitwidth_exponent=4, bitwidth_mantissa=3, is_signed=True, unsigned_zero=True
)
FP8_E5M2 = FloatType(bitwidth_exponent=5, bitwidth_mantissa=2, is_signed=True)
BF16 = FloatType(bitwidth_exponent=8, bitwidth_mantissa=7, is_signed=True)
FP16 = FloatType(bitwidth_exponent=5, bitwidth_mantissa=10, is_signed=True)
FP32 = FloatType(bitwidth_exponent=8, bitwidth_mantissa=23, is_signed=True)
FP64 = FloatType(bitwidth_exponent=11, bitwidth_mantissa=52, is_signed=True)
INT16 = IntegerType(16, is_signed=True)
INT32 = IntegerType(32, is_signed=True)
INT64 = IntegerType(64, is_signed=True)

DataType: TypeAlias = IntegerType | FloatType


# layout utilities
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StridedLayout:
    major_dim: int = -1

    def __post_init__(self):
        if not isinstance(self.major_dim, int):
            raise TypeError(
                f"StridedLayout(major_dim=...) must be an int, got {type(self.major_dim)}"
            )

    @property
    def name(self):
        return "STRIDED"

    def swizzle_block_shape(self, block_shape):
        return block_shape

    def order(self, rank: int) -> list[int]:
        """
        Returns the minor->major dimension order for a given tensor rank.

        `self.major_dim` supports negative indexing (like Python).
        """
        if rank <= 0:
            return []
        if not (-rank <= self.major_dim < rank):
            raise ValueError(
                f"Invalid StridedLayout.major_dim={self.major_dim} for rank={rank}"
            )
        major_dim = self.major_dim if self.major_dim >= 0 else self.major_dim + rank
        base = list(reversed(range(rank)))
        idx = base.index(major_dim)
        base[0], base[idx] = base[idx], base[0]
        return base


# storage
# ---------------------------------------------------------------------------- #
@dataclass
class Storage:
    data: torch.Tensor
    layout: StridedLayout

    @property
    def device(self):
        return self.data.device


# main tensor class
# ---------------------------------------------------------------------------- #
@dataclass
class Tensor:
    storage: Storage
    dtype: IntegerType | FloatType
    shape: list[int] | None = None
    shape_max: list[int] | None = None

    def __post_init__(self):
        assert isinstance(self.storage, Storage)
        # initialize dtype
        if self.dtype.bitwidth < 8 and self.shape is None:
            raise ValueError("shape must be provided for sub-byte types")
        # initialize shape
        if self.shape is None:
            self.shape = list(self.storage.data.shape)
        self.shape = list(self.shape)
        # validate shape: all elements must be `int` or numel-1 `torch.Tensor`
        is_int = lambda s: isinstance(s, int)
        is_item = lambda s: hasattr(s, "numel") and s.numel() == 1
        assert all(map(lambda s: is_int(s) or is_item(s), self.shape))
        # initialize shape_max
        if self.shape_max is None:
            self.shape_max = [None] * len(self.shape)
        for i, (s, smax) in enumerate(zip(self.shape, self.shape_max)):
            if smax is not None and not is_int(smax):
                raise ValueError(
                    f"shape_max[{i}] must be `int` or `None`; got {type(smax)}"
                )
            if smax is None:
                self.shape_max[i] = s
        # validate shape_max: all elements must be `int`
        assert all(map(is_int, self.shape_max))

    # torch compatibility layer
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def device(self):
        return self.storage.device

    def stride(self, i=None):
        return self.storage.data.stride() if i is None else self.storage.data.stride(i)

    def data_ptr(self):
        return self.storage.data.data_ptr()

    def numel(self):
        return self.storage.data.numel()

    def element_size(self):
        return self.dtype.bitwidth // 8

    @property
    def data(self):
        t = self.storage
        return t.data if isinstance(t, Storage) else t

    def dim(self):
        return self.ndim

    def size(self, i=None):
        if i is None:
            return self.shape
        return self.shape[i]


def torch_dtype_to_dtype(dtype: torch.dtype) -> DataType:
    if isinstance(dtype, DataType):
        return dtype
    id = str(dtype).split(".")[-1]
    vals = {
        "uint8": UINT8,
        "float8_e4m3fn": FP8_E4M3FN,
        "float8_e4m3fnuz": FP8_E4M3FNUZ,
        "float8_e5m2": FP8_E5M2,
        "float16": FP16,
        "bfloat16": BF16,
        "float32": FP32,
        "float64": FP64,
        "int16": INT16,
        "int32": INT32,
        "int64": INT64,
    }
    if id in vals:
        return vals[id]
    if "float8" in id:
        return FP8_E4M3FN
    assert False, f"Unknown dtype: {id}"


def wrap_torch_tensor(
    torch_tensor, dtype=None, shape=None, shape_max=None, layout=None
):
    if dtype is None:
        dtype = torch_tensor.dtype
    dtype = torch_dtype_to_dtype(dtype)
    if shape is None:
        shape = list(torch_tensor.shape)
        if dtype == FP4:
            shape[torch_tensor.stride().index(1)] *= (
                8 * torch_tensor.dtype.itemsize
            ) // dtype.bitwidth
    if shape_max is None:
        shape_max = list(shape)
    if layout is None:
        # For a strided (dense) tensor we only track which dimension has unit stride.
        # This is consistent with how we expand `shape` for packed sub-byte dtypes.
        major_dim = torch_tensor.stride().index(1) if 1 in torch_tensor.stride() else -1
        layout = StridedLayout(major_dim=major_dim - torch_tensor.ndim)
    return Tensor(
        Storage(torch_tensor, layout), dtype=dtype, shape=shape, shape_max=shape_max
    )


# ragged tensor metadata
# ---------------------------------------------------------------------------- #
@dataclass
class RaggedTensorMetadata:
    """
    Example:
    `slice_sizes`= [15 17 0 127]
    `slice_offs`= [0 15 32 32 332]
    `block_offs_data` = {
        16: [0 1 3 3 11]
        32: [0 1 2 2 6]
        64: [0 1 2 2 4]
        128: [0 1 2 2 3]
    }
    `block_schedule_data` = {
        16:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) ... (7, 3) -1 ... -1]
        32:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) (3, 3) -1 ...     -1]
        64:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) -1 ...            -1]
        128: [(0, 0) (0, 1) (0, 3) (1, 3) -1 ...                   -1]
    }
    """

    # slice_sizes[i] is the number of elements in slice i along the ragged dimension
    slice_sizes: torch.Tensor
    # slice_offs = [0] + cumsum(slice_sizes)
    # i.e., slice_offs[i] is the offset of the first element in slice `i`
    slice_offs: torch.Tensor
    # block_offs_data[k] = [0] + cumsum(ceil_div(slice_sizes, 16 * k))
    # i.e., `block_offs_data[k][i]` is the offset of the first block of
    # `16*k`` token for batch `i` in a `bath_sizes`-shaped ragged tensor
    block_offs_data: torch.Tensor
    # let `num_blocks[k] = block_offs_data[k, 1:] - block_offs_data[k, :-1]
    # block_schedule_data[k] = cat(*[[(batch, blk) for blk in range(blks)] for batch, blks in enumerate(num_blocks)])
    # i.e., if the schedule of batch `i` is [(i, 0), (i, 1), ..., (i, num_blocks[k][i] - 1)]
    # then `block_schedule_data[k]` is the concatenation of the schedules for all batches
    # NOTE 1: `block_schedule_data[k][j]` is a packed 32-bit integer
    # NOTE 2: because the size of `block_schedule_data[k]` is data-dependent, we pad it with -1s
    # up to an user-provided upper bound
    block_schedule_data: torch.Tensor
    # expected slice size (for heuristics)
    expected_slice_size: int | None = None
    # divisibility hint for values in `slice_sizes`
    slice_sizes_divisibility: int = None

    def __post_init__(self):
        assert self.block_offs_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_schedule_data.shape[0] == len(
            RaggedTensorMetadata.block_sizes()
        )
        assert self.block_offs_data.dtype == torch.int32
        assert self.block_schedule_data.dtype == torch.int32
        if self.slice_sizes is not None:
            assert self.slice_sizes.dtype == torch.int32
        if self.slice_offs is not None:
            assert self.slice_offs.dtype == torch.int32

    @property
    def n_slices(self):
        return self.slice_sizes.shape[0]

    def block_offs(self, block_size):
        return self.block_offs_data[
            RaggedTensorMetadata.block_sizes().index(block_size)
        ]

    def block_schedule(self, block_size):
        return self.block_schedule_data[
            RaggedTensorMetadata.block_sizes().index(block_size)
        ]

    @staticmethod
    def n_blocks(n_slices, n_total_rows, block_size):
        if n_total_rows <= n_slices:
            return n_total_rows
        return n_slices - 1 - ((n_slices - n_total_rows - 1) // block_size)

    @staticmethod
    def max_n_blocks(n_slices, n_total_rows):
        return RaggedTensorMetadata.n_blocks(
            n_slices, n_total_rows, min(RaggedTensorMetadata.block_sizes())
        )

    @staticmethod
    def block_sizes_log2():
        return range(4, 9)

    @staticmethod
    def block_sizes():
        return [2**x for x in RaggedTensorMetadata.block_sizes_log2()]


def cdiv(x, y):
    return (x + y - 1) // y


def exact_div(x, y):
    assert x % y == 0
    return x // y


def empty_aligned(shape, dtype, device, pad_size):
    pad = lambda x: cdiv(x, pad_size) * pad_size
    ret = torch.empty((*shape[:-1], pad(shape[-1])), dtype=dtype, device=device)
    ret_slices = (*[slice(None)] * (len(shape) - 1), slice(0, shape[-1]))
    return ret[ret_slices], ret.numel()


@triton.jit
def _cdiv_pow2(n, log2_k):
    # ceil_div(n, 2**log2_k)
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _ragged_tensor_metadata_memset(
    SliceSizes,
    n_slices,
    BlockOffs,
    slice_offs_stride_m,
    BlockSchedule,
    first_block_size_log2,
    SIZES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid <= SIZES:
        BlockOffs += pid * slice_offs_stride_m
        BlockOffsPtrs = BlockOffs + tl.arange(0, BLOCK)
        block_size_log2 = tl.where(pid == 0, 0, pid + first_block_size_log2 - 1)
        # total number of blocks in slice processed as the loop iterates
        n_blocks_tot = tl.zeros([BLOCK], dtype=BlockOffs.dtype.element_ty)
        for i in range(0, n_slices + 1, BLOCK):
            # load slice sizes
            offs = tl.arange(0, BLOCK) + i
            mask = offs < n_slices
            slice_sizes = tl.load(SliceSizes + offs, mask=mask, other=0)
            # number of blocks in the slices loaded
            n_blocks = _cdiv_pow2(slice_sizes, block_size_log2)
            # start index of the blocks for the slices loaded
            block_starts = tl.cumsum(n_blocks, 0) + n_blocks_tot
            n_blocks_tot += tl.sum(n_blocks, 0)
            tl.store(BlockOffsPtrs, block_starts - n_blocks)
            BlockOffsPtrs += BLOCK
    else:
        # initialize block schedule to -1
        pid -= SIZES + 1
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(BlockSchedule + offs, 0xFFFFFFFF)


@triton.jit
def _ragged_tensor_metadata_compute(
    SliceSizes,  #
    BlockOffs,
    block_offs_stride_m,  #
    BlockSchedule,
    block_schedule_stride_m,  #
    first_block_size_log2,  #
    SIZES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    slice_id = pid // SIZES
    block_size_id = pid % SIZES
    # offset pointers
    BlockOffs += block_size_id * block_offs_stride_m
    BlockSchedule += block_size_id * block_schedule_stride_m
    # load slice sizes
    slice_sizes = tl.load(SliceSizes + slice_id)
    # number of blocks in the slices loaded
    block_size_log2 = first_block_size_log2 + block_size_id
    n_blocks = _cdiv_pow2(slice_sizes, block_size_log2)
    # compute block schedule
    block_off = tl.load(BlockOffs + slice_id)
    BlockSchedule += block_off
    for block_off in range(0, n_blocks, BLOCK):
        block_offs = block_off + tl.arange(0, BLOCK)
        data = (block_offs << 16) + slice_id
        tl.store(BlockSchedule + block_offs, data, mask=block_offs < n_blocks)


def make_ragged_tensor_metadata(slice_sizes, n_total_rows):
    assert slice_sizes.ndim == 1
    n_slices = slice_sizes.shape[0]
    block_sizes_log2 = RaggedTensorMetadata.block_sizes_log2()
    block_size_num = len(block_sizes_log2)
    MEMSET_BLOCK = 512
    dtype = torch.int32
    device = slice_sizes.device
    max_n_blocks = RaggedTensorMetadata.max_n_blocks(n_slices, n_total_rows)
    slice_offs_combined, _ = empty_aligned(
        (block_size_num + 1, n_slices + 1), dtype, device, MEMSET_BLOCK
    )
    block_schedule_data, n_memset_elts = empty_aligned(
        (block_size_num, max_n_blocks), dtype, device, MEMSET_BLOCK
    )
    slice_offs, block_offs_data = slice_offs_combined[0], slice_offs_combined[1:]
    n_memset_blocks = exact_div(n_memset_elts, MEMSET_BLOCK)

    _ragged_tensor_metadata_memset[(slice_offs_combined.shape[0] + n_memset_blocks,)](
        slice_sizes,
        n_slices,  #
        slice_offs_combined,
        slice_offs_combined.stride(0),  #
        block_schedule_data,  #
        block_sizes_log2[0],
        SIZES=len(block_sizes_log2),
        BLOCK=MEMSET_BLOCK,  # optimization parameters
        num_warps=4,
    )

    _ragged_tensor_metadata_compute[(block_size_num * n_slices,)](
        slice_sizes,
        block_offs_data,
        block_offs_data.stride(0),
        block_schedule_data,
        block_schedule_data.stride(0),  # outputs
        block_sizes_log2[0],
        SIZES=len(block_sizes_log2),
        BLOCK=512,  # optimization parameters
        num_warps=4,
    )

    return RaggedTensorMetadata(
        slice_sizes, slice_offs, block_offs_data, block_schedule_data
    )


__all__ = [
    "make_ragged_tensor_metadata",
]
