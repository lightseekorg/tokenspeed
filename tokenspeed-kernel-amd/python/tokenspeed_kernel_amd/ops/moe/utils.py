"""Local copies of small triton-kernels utilities used by AMD MoE kernels."""

# fmt: off
# isort: off
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeAlias, Union

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


BIT = IntegerType(1, is_signed=False)
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


def dtype_to_torch_dtype(dtype: DataType) -> torch.dtype:
    if dtype is None:
        return None
    if not isinstance(dtype, DataType):
        return dtype
    return {
        FP4: torch.uint8,
        UINT8: torch.uint8,
        FP8_E4M3FN: torch.float8_e4m3fn,
        FP8_E4M3FNUZ: torch.float8_e4m3fnuz,
        FP8_E5M2: torch.float8_e5m2,
        BF16: torch.bfloat16,
        FP32: torch.float32,
        FP16: torch.float16,
        FP64: torch.float64,
        INT16: torch.int16,
        INT32: torch.int32,
        INT64: torch.int64,
    }[dtype]


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


# sum bitmatrix rows
# ---------------------------------------------------------------------------- #
@triton.jit
def vpopc(x):
    """
    Vertical popcount
    Input  x : uint32[..., N]
    Output y : uint32[..., 32]
    semantics : y[..., i] = sum_j((x[..., j] >> i) & 1)
    credits: @apgoucher
    """

    tl.static_assert(
        x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers"
    )

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    # create 8-way sums in 4-bit fields:
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // sa1, 4]
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    # create 128-way sums in 8-bit fields:
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0F0F0F0F
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // (sa1 * sa2), 2, 4]
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    # create N-way sums in 32-bit fields:
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000FF
    y = tl.sum(y, 2)  # [BATCHES, 4, 8]
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def _sum_bitmatrix_rows(
    B,
    shape_bm,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,  # input bitmatrix
    Out,
    OutPartials,
    stride_pm: tl.constexpr,
    stride_pn,
    shape_pn,  # outputs
    BLOCK_MM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tl.static_assert(BLOCK_MM % BLOCK_M == 0)
    TILE_SIZE: tl.constexpr = BLOCK_MM // BLOCK_M
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    # load input bits
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_bm = pid_m * BLOCK_MM + tl.arange(0, BLOCK_MM)
    bits = tl.load(
        B + pid_n * stride_bn + offs_bm * stride_bm, mask=offs_bm < shape_bm, other=0
    )
    bits = tl.reshape(bits, [TILE_SIZE, BLOCK_M])
    # partial row sum
    partial_row_sum = vpopc(bits)  # [TILE_SIZE, 32]
    # write-back partial row sum
    offs_pm = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs_n = pid_n * 32 + tl.arange(0, 32)
    tl.store(
        OutPartials + offs_pm[:, None] * stride_pm + offs_n[None, :] * stride_pn,
        partial_row_sum,
    )
    # update final row sum
    tl.atomic_add(Out + offs_n, tl.sum(partial_row_sum, 0), sem="relaxed")


def cdiv(x, y):
    return (x + y - 1) // y


def sum_bitmatrix_rows(x, partials_block_size=None):
    assert partials_block_size is not None
    PARTIALS_BLOCK_M = partials_block_size
    n_rows, n_cols = x.shape
    n_rows_max = x.shape_max[0]

    TILE_SIZE = max(1, 128 // PARTIALS_BLOCK_M)
    BLOCK_MM = PARTIALS_BLOCK_M * TILE_SIZE

    grid_m = cdiv(n_rows_max, BLOCK_MM)
    grid_n = cdiv(n_cols, 32)
    out = torch.zeros((cdiv(n_cols, 128) * 128,), device=x.device, dtype=torch.int32)[
        :n_cols
    ]
    out_partials = torch.empty(
        (grid_n * 32, grid_m * TILE_SIZE), device=x.device, dtype=torch.int32
    )
    out_partials = torch.transpose(out_partials, 0, 1)
    # output tensors
    _sum_bitmatrix_rows[(grid_m, grid_n)](
        x.storage.data,
        n_rows,
        x.stride(0),
        x.stride(1),  # input
        out,  # output [final reduction]
        out_partials,
        out_partials.stride(0),
        out_partials.stride(1),
        out_partials.shape[1],  # output [partial reductions]
        BLOCK_M=PARTIALS_BLOCK_M,
        BLOCK_MM=BLOCK_MM,  # constants
        num_warps=8,
    )
    out_partials = out_partials[: cdiv(n_rows_max, PARTIALS_BLOCK_M), :]
    return out, out_partials


# bitmatrix metadata
# ---------------------------------------------------------------------------- #
@dataclass
class BitmatrixMetadata:
    """
    Example:
    `bitmatrix` = [0 0 1 0 1 1 0
                   0 1 0 0 0 1 0
                   1 1 1 0 0 0 1
                   0 0 1 0 1 0 0]
    `col_sum` = [1 2 3 0 2 2 1]
    `col_sorted_indx` = cat([5], [3 6], [0 7], [], [9 1 10], [2 4], [8])
    `row_sorted_indx` = cat([3 6 8], [1 9], [0 2 4 10], [5 7])
    """

    # the number of entries equal to 1 in each column
    col_sum: torch.Tensor
    # indices of nonzero values numbered row-major, grouped by cols, concatenated
    col_sorted_indx: torch.Tensor
    # indices of nonzero values numbered col-major, grouped by rows, concatenated
    row_sorted_indx: torch.Tensor


@triton.jit
def _keyed_add(x, y):
    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xFFFF0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _bitmatrix_metadata_compute_stage2(
    ColSortedIndx,
    RowSortedIndx,
    NonzeroIndx,
    n_tokens,
    ColPartialSum,
    stride_pm,
    stride_pn,
    ColOffs,
    TOKS_PER_ROW: tl.constexpr,
    BLOCK_PER_TOK: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = BLOCK_PER_TOK * TOKS_PER_ROW
    tl.static_assert(BLOCK_SIZE <= 32768)
    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    nonzero_indx_size = n_tokens * TOKS_PER_ROW
    pid_m = tl.program_id(0)
    # load column indices
    offs_local = tl.arange(0, BLOCK_SIZE)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < nonzero_indx_size
    col_indx = tl.load(NonzeroIndx + offs_global, mask=mask, other=-1).to(tl.uint32)
    # stable-sort by columns index
    kv_pairs = ((col_indx << 16) | offs_local).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    col_indx = kv_pairs >> 16
    offs_global = pid_m * BLOCK_SIZE + (kv_pairs & 0xFFFF)
    mask = col_indx != 0xFFFF
    # compute run lengths in column-sorted order:
    x = kv_pairs & 0xFFFF0000 | 0x00000001
    cols_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (cols_and_inclusive_run_lengths - 1) & 0xFFFF
    # compute output
    row_sorted_indx = tl.load(
        ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=mask
    )
    row_sorted_indx += tl.load(ColOffs + col_indx, mask=mask)
    row_sorted_indx += exclusive_run_lengths
    # write back output
    tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=mask)
    tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=mask)


@triton.jit
def _bitmatrix_metadata_compute_stage1(
    CombinedIndx,
    n_combined_indx,
    sentinel,
    BLOCK: tl.constexpr,
    ColSum,
    ColOffs,
    n_cols,
    PartialColSum,
    shape_pm,
    stride_pm,
    stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    # compute col_partial_sums
    if pid < n_cols:
        PartialColSum += pid * stride_pn
        curr_sum = 0
        for start in range(0, shape_pm, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M) * stride_pm
            partial_col_sum = tl.load(PartialColSum + offs, mask=offs < shape_pm)
            out = tl.cumsum(partial_col_sum, 0) - partial_col_sum + curr_sum
            curr_sum += tl.sum(partial_col_sum, 0)
            tl.store(PartialColSum + offs, out, mask=offs < shape_pm)
    # compute col_offs
    elif pid == n_cols:
        curr_sum = 0
        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            col_sum = tl.load(ColSum + offs, mask=offs < n_cols)
            col_offs = tl.cumsum(col_sum, 0) - col_sum + curr_sum
            curr_sum += tl.sum(col_sum, 0)
            tl.store(ColOffs + offs, col_offs, mask=offs < n_cols)
    # memset `combined_indx` to `sentinel`
    else:
        offs = (pid - n_cols - 1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(CombinedIndx + offs, sentinel, mask=offs < n_combined_indx)


def make_bitmatrix_metadata(nonzero_indx, bitmatrix):
    assert nonzero_indx.ndim == 2
    PARTIAL_BLOCK_M = 32
    col_sum, col_partial_sum = sum_bitmatrix_rows(
        bitmatrix, partials_block_size=PARTIAL_BLOCK_M
    )
    # allocate memory
    device = bitmatrix.device
    n_indx = nonzero_indx.numel()
    n_cols = bitmatrix.shape[1]
    col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
    combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
    col_sorted_indx = combined_indx[:n_indx]
    row_sorted_indx = combined_indx[n_indx:]
    # this kernel:
    # - initializes `{row,col}_sorted_indx` to `sentinel`
    # - computes col_offs; necessary for computing `{row,col}_sorted_indx`
    # - computes col_partial_sums; necessary for computing `{row,col}_sorted_indx`
    MEMSET_BLOCK = 1024
    memset_grid = (cdiv(n_indx * 2, MEMSET_BLOCK) + n_cols + 1,)
    _bitmatrix_metadata_compute_stage1[memset_grid](
        combined_indx,
        n_indx * 2,
        -1,
        MEMSET_BLOCK,
        col_sum,  #
        col_offs,
        col_sum.shape[0],
        col_partial_sum,  # inputs
        col_partial_sum.shape[0],
        col_partial_sum.stride(0),
        col_partial_sum.stride(1),  # outputs
        BLOCK_M=512,
        BLOCK_N=512,  # tunable parameters
    )
    # this kernel computes valid entries of `{row,col}_sorted_indx`
    # using `col_offs` and `col_partial_sums`
    n_indx = nonzero_indx.numel()
    toks_per_row = nonzero_indx.shape[-1]
    compute_grid = (cdiv(bitmatrix.shape_max[0], PARTIAL_BLOCK_M),)
    _bitmatrix_metadata_compute_stage2[compute_grid](
        col_sorted_indx,
        row_sorted_indx,  # outputs
        nonzero_indx,
        bitmatrix.shape[0],
        col_partial_sum,
        col_partial_sum.stride(0),
        col_partial_sum.stride(1),  # inputs
        col_offs,  #
        TOKS_PER_ROW=toks_per_row,
        BLOCK_PER_TOK=PARTIAL_BLOCK_M,  #
    )
    return BitmatrixMetadata(
        col_sum=col_sum,
        col_sorted_indx=col_sorted_indx,
        row_sorted_indx=row_sorted_indx,
    )


# sparse matrix
# ---------------------------------------------------------------------------- #
@dataclass
class SparseMatrix:
    indx: torch.Tensor
    vals: torch.Tensor
    mask: Tensor

    def __post_init__(self):
        self.mask_metadata = make_bitmatrix_metadata(self.indx, self.mask)


# topk forward kernels
# ---------------------------------------------------------------------------- #
@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(
        x.dtype.is_int_unsigned(), "floating-point value must be passed as bits"
    )
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


# stable top-k tie-breaks to value with smaller index
@triton.jit
def indx_to_key(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def key_to_indx(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def streaming_topk(
    X,
    stride_xm,
    n_expts_tot,
    offs_m,
    mask_m,
    N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:
        # this ensures that we leave at least 16 bits for expert index
        # even if the input dtype is smaller than 16 bits:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = X.dtype.element_ty

    # subtract 1 from loop iterations because we peel the first (masked) iteration:
    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1
    offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot

    # first iteration:
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    acc = tl.topk(x, N_EXPTS_ACT, dim=1)

    # subsequent iterations:
    for _i in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
        acc = tl.bitonic_merge(acc)  # ensure sorted ascending for the merge
        X_ptrs -= BLOCK_N
        offs_x_n -= BLOCK_N
        x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
        x = fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
        acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))

    # sort packed (value_key, index_key) descending:
    # this keeps outputs ordered by gate value and uses smaller expert index for ties
    acc = tl.sort(acc, dim=1, descending=True)
    # 0000vvvvvvvviiii --> 0000iiii:
    y_indices_raw = (acc & 0xFFFF).to(tl.uint32)
    y_indices = key_to_indx(y_indices_raw, N_EXPTS_PAD)
    # 0000vvvvvvvviiii --> vvvvvvvv:
    y_values_raw = (acc >> 16).to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    return y_values, y_indices


@triton.jit
def _topk_forward(
    X,
    stride_xm,  # inputs
    PeerYvs,
    PeerYis,
    stride_ym,  # topk values/indices
    USE_PROVIDED_INDX: tl.constexpr,
    PeerBits,
    stride_rm: tl.constexpr,
    stride_rn: tl.constexpr,  # bitmatrix
    n_rows,
    n_expts_tot,  # shape
    dst_offs_m,
    APPLY_SOFTMAX: tl.constexpr,  # constant
    BLOCK_M: tl.constexpr,
    N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    N_PEERS: tl.constexpr = len(PeerYvs)

    pid = tl.program_id(0)
    if isinstance(n_rows, tl.tensor) and n_rows.dtype.is_ptr():
        n_rows = tl.load(n_rows)

    if pid * BLOCK_M >= n_rows:
        # early exit:
        return

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty

    # load logits
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    if USE_PROVIDED_INDX:
        tl.static_assert(len(PeerYis) == 1)
        Yi_ptrs = (
            PeerYis[0] + (dst_offs_m + offs_m[:, None]) * stride_ym + offs_y_n[None, :]
        )
        y_indices = tl.load(Yi_ptrs, mask=mask_m)
        Xv_ptrs = X + offs_m[:, None] * stride_xm + y_indices
        y_values = tl.load(Xv_ptrs, mask=mask_m)
    else:
        y_values, y_indices = streaming_topk(
            X,
            stride_xm,
            n_expts_tot,
            offs_m,
            mask_m,
            N_EXPTS_PAD,
            N_EXPTS_ACT,
            BLOCK_N,
        )

    # normalize selected values
    if APPLY_SOFTMAX:
        y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True).to(
            x_dtype
        )

    # write back
    for rank in tl.static_range(N_PEERS):
        Yv_ptrs = (
            PeerYvs[rank]
            + (dst_offs_m + offs_m[:, None]) * stride_ym
            + offs_y_n[None, :]
        )
        tl.store(Yv_ptrs, y_values, mask=mask_m)
    if not USE_PROVIDED_INDX:
        for rank in tl.static_range(N_PEERS):
            Yi_ptrs = (
                PeerYis[rank]
                + (dst_offs_m + offs_m[:, None]) * stride_ym
                + offs_y_n[None, :]
            )
            tl.store(Yi_ptrs, y_indices, mask=mask_m)

    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    loop_iterations = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(
            y_div[:, :, None] == offs_r_n[None, None, :], (1 << y_rem)[:, :, None], 0
        )
        r = tl.reduce_or(y2, axis=1)
        for rank in tl.static_range(N_PEERS):
            BitsPtrs = (
                PeerBits[rank]
                + (dst_offs_m + offs_m[:, None]) * stride_rm
                + offs_r_n[None, :] * stride_rn
            )
            tl.store(BitsPtrs, r, mask=mask_m)


def make_empty(offset, shape, dtype, device, all_gather, symm_mem_pool):
    dtype = dtype_to_torch_dtype(dtype)
    if all_gather:
        rank_id = symm_mem_pool.mesh.local_rank
        ret_bufs = symm_mem_pool.make_empty(
            shape=shape, dtype=dtype, region="topk", region_offset=offset
        )
        ret = ret_bufs[rank_id]
        offset = symm_mem_pool.align_up(
            offset + ret.numel() * ret.element_size(),
            symm_mem_pool.regions["topk"].alignment,
        )
        return ret_bufs, ret, offset
    ret = torch.empty(shape, dtype=dtype, device=device)
    return (ret,), ret, 0


def topk_forward(
    x,
    k,
    apply_softmax=True,
    dim=1,
    y_indx=None,
    n_rows=None,
    all_gather=False,
    symm_mem_pool=None,
):
    if not isinstance(x, Tensor):
        x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
        x_shape_max = [x.shape[0], x.shape[1]]
        x = wrap_torch_tensor(x, shape=x_shape, shape_max=x_shape_max)
    BLOCK_M = 32
    BLOCK_N = 32
    use_provided_indx = y_indx is not None
    assert symm_mem_pool is not None or not all_gather
    assert len(x.shape) == 2
    assert x.shape_max[-1] < 32768
    assert dim == 1
    n_rows, n_cols = x.shape
    n_rows_max, _ = x.shape_max
    dev = x.device
    n_rows_out_max = (
        n_rows_max * symm_mem_pool.mesh.world_size if all_gather else n_rows_max
    )
    # scratchpad tensors
    # NOTE: these are not returned
    y_vals_bufs, y_vals, offset = make_empty(
        0,
        (n_rows_out_max, k),
        x.dtype,
        dev,
        all_gather=all_gather,
        symm_mem_pool=symm_mem_pool,
    )
    if y_indx is None:
        y_indx_bufs, y_indx, offset = make_empty(
            offset,
            (n_rows_out_max, k),
            torch.int16,
            dev,
            all_gather=all_gather,
            symm_mem_pool=symm_mem_pool,
        )
    else:
        y_indx_bufs = (y_indx,)
    # create bitmatrix in transposed memory layout:
    n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
    n_cols_words = n_cols_pad // 32
    bitmatrix_bufs, bitmatrix_data, offset = make_empty(
        offset,
        (n_cols_words, cdiv(n_rows_out_max, 32) * 32),
        torch.uint32,
        dev,
        all_gather=all_gather,
        symm_mem_pool=symm_mem_pool,
    )
    bitmatrix_data = torch.transpose(bitmatrix_data, 0, 1)[:n_rows_max]
    pids = cdiv(n_rows_max, BLOCK_M)
    _topk_forward[(pids,)](
        x.storage.data,
        x.stride(0),  # inputs
        y_vals_bufs,
        y_indx_bufs,
        y_vals.stride(0),
        use_provided_indx,  # output [topk]
        bitmatrix_bufs,
        bitmatrix_data.stride(0),
        bitmatrix_data.stride(1),  # output [bitmatrix]
        n_rows,
        n_cols,  # shapes
        symm_mem_pool.mesh.local_rank * n_rows_max if all_gather else 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,  # tunable parameter
        APPLY_SOFTMAX=apply_softmax,
        N_EXPTS_PAD=n_cols_pad,
        N_EXPTS_ACT=k,  # constants
    )
    if all_gather:
        symm_mem_pool.hdl.barrier(channel=0)
    bitmatrix_shape = [
        n_rows * symm_mem_pool.mesh.world_size if all_gather else n_rows,
        n_cols,
    ]
    bitmatrix_shape_max = [n_rows_out_max, None]
    bitmatrix = wrap_torch_tensor(
        bitmatrix_data, dtype=BIT, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max
    )
    return y_vals, y_indx, bitmatrix


def topk(
    x: Union[Tensor, torch.Tensor],
    k: int,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: Optional[torch.Tensor] = None,
    n_rows: Optional[int] = None,
    all_gather: bool = False,
    symm_mem_pool: object | None = None,
):
    """
    Computes the top-k values and indices along a specified dimension of a tensor.
    Note that the input can be either a `Tensor` or a `torch.Tensor`, but the output will always be a `torch.Tensor`.

    Parameters
    ----------
    x : Union[Tensor, torch.Tensor]
        Input tensor of shape (n_tokens, n_expts).
    k : int
        Number of top elements to retrieve.
    apply_softmax : bool, default True
        Whether to apply softmax to the input tensor before computing top-k.
    dim : int, default 1
        Dimension along which to compute top-k.
    y_indx : torch.Tensor, optional
        Pre-allocated tensor for storing indices of top-k elements with shape (n_tokens, k).
        If provided, we skip the computation of top-k indices and use this tensor instead.
    n_rows : int, optional
        Number of rows to apply top-k on. If None, we consider all rows in `x`.

    Returns
    -------
    SparseMatrix: sparse matrix equal to `x` with non-selected entries set to 0
    """
    y_vals, y_indx, bitmatrix = topk_forward(
        x,
        k,
        apply_softmax,
        dim,
        y_indx,
        n_rows,
        all_gather,
        symm_mem_pool,
    )
    return SparseMatrix(vals=y_vals, indx=y_indx, mask=bitmatrix)


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
        return range(4, 9) if torch.version.hip is not None else range(4, 8)

    @staticmethod
    def block_sizes():
        return [2**x for x in RaggedTensorMetadata.block_sizes_log2()]


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
    "topk",
    "make_ragged_tensor_metadata",
]
