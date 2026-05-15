# Copyright (c) 2026 LightSeek Foundation
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

"""
CuTe DSL MLA Decode Kernel Integration
=======================================

Wraps NVIDIA's CuTe DSL MLA decode kernels (FP16/BF16/FP8) for Blackwell SM100
and exposes them via a PyTorch API compatible with FlashInfer's MLA backend.
"""

import functools
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from tokenspeed_mla.mla_decode_fp8 import (
    BlackwellMultiHeadLatentAttentionForwardFP8,
)
from tokenspeed_mla.mla_decode_fp16 import (
    BlackwellMultiHeadLatentAttentionForwardFP16,
)
from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor
from tokenspeed_mla.utils import (
    get_max_active_clusters,
    get_num_sm,
    torch_to_cutlass_dtype,
)


@functools.cache
def _get_split_kv_and_workspace_size(
    B: int,
    q_len: int,
    H: int,
    kv_lora_rank: int,
    max_active_blocks: int,
) -> Tuple[int, int]:
    """Cache split_kv and workspace_size since they are deterministic for the same params."""
    split_kv = BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv_simplified(
        B, q_len, max_active_blocks
    )
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
        H, q_len, kv_lora_rank, B, split_kv, cutlass.Float32
    )
    return split_kv, workspace_size


@functools.cache
def _check_can_implement(
    torch_dtype: torch.dtype,
    page_size: int,
    num_heads: int,
    seq_len_q: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
) -> None:
    """Check if the kernel supports the given configuration (cached)."""
    mma_qk_tiler_mn = (128, 128)
    mma_pv_tiler_mn = (128, 256)

    is_fp8 = torch_dtype == torch.float8_e4m3fn
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    cutlass_dtype = torch_to_cutlass_dtype(torch_dtype)
    if not KernelClass.can_implement(
        1,  # B (runtime, use placeholder)
        seq_len_q,
        1,  # K (runtime, use placeholder)
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        cutlass_dtype,
        cutlass_dtype,
        cutlass.Float32,
        cutlass.Float32,
        mma_qk_tiler_mn,
        mma_pv_tiler_mn,
        1,  # split_kv placeholder; actual value is selected at runtime
        is_persistent,
        is_var_seq,
        is_var_split_kv,
        page_size,
    ):
        raise ValueError(
            f"tokenspeed_mla_decode: unsupported configuration "
            f"(q_len={seq_len_q}, num_heads={num_heads}, page_size={page_size}, "
            f"dtype={torch_dtype})"
        )


@functools.cache
def _get_compiled_mla_kernel(
    torch_dtype: torch.dtype,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    skip_correction_threshold: float = 0.0,
    is_workspace_size_zero: bool = False,
    fold_sq_factor: int = 1,
    causal_mask: bool = True,
    num_heads: int = 128,
    seq_len_q: int = 1,
    use_pdl: bool = False,
) -> Callable:
    """Compile and cache an MLA decode kernel.

    Returns a callable that accepts (q_latent, q_rope, c_latent, c_rope,
    page_table, o, lse (None to skip), workspace, split_kv_scalar, cache_seqs,
    block_split_kvs, softmax_scale_scalar, output_scale_scalar).

    All scalar arguments must be pre-wrapped as Int32/Float32.
    """
    # Tile sizes for Blackwell mma.
    # (128, 128) for QK and (128, 256) for PV.
    mma_qk_tiler_mn = (128, 128)
    mma_pv_tiler_mn = (128, 256)
    # 2 CTAs along M (num_heads)
    cluster_shape_mnk = (2, 1, 1)

    is_fp8 = torch_dtype == torch.float8_e4m3fn
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    cutlass_dtype = torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = cutlass.BFloat16 if is_fp8 else cutlass_dtype

    kernel_kwargs = dict(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=mma_qk_tiler_mn,
        mma_pv_tiler_mn=mma_pv_tiler_mn,
        max_active_clusters=get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        ),
        page_size=page_size,
        skip_correction_threshold=skip_correction_threshold,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        fold_sq_factor=fold_sq_factor,
    )
    if is_fp8:
        kernel_kwargs["is_causal"] = causal_mask
        kernel_kwargs["num_heads"] = num_heads
        kernel_kwargs["seq_len_q"] = seq_len_q
    kernel_obj = KernelClass(**kernel_kwargs)

    # All dimensions as sym_int — this matches the original kernel's use of
    # mark_compact_shape_dynamic, which makes ALL shapes dynamic CuTe Integers.
    # Static Python ints would cause cute.assume() to fail with AttributeError
    # inside initialize_workspace() since it expects DSL Integer types.
    sym_heads = cute.sym_int()
    sym_latent = cute.sym_int(divisibility=16)
    sym_seq_q = cute.sym_int()
    sym_rope = cute.sym_int(divisibility=16)
    sym_batch = cute.sym_int()  # query/output batch dimension
    sym_kv_batch = cute.sym_int()  # KV cache batch dim (flat pool, =1 in paged mode)
    sym_seq_kv = cute.sym_int()
    sym_page_count = cute.sym_int()
    sym_workspace_size = cute.sym_int()

    # q_latent, q_rope, c_latent, c_rope are slices of contiguous tensors on
    # the last dim (e.g. query[..., :kv_lora_rank]), so they are NOT contiguous:
    #   stride[-2] = D_qk (original full last dim), not the sliced shape.
    # Use make_fake_tensor with fully dynamic strides so the compiled kernel
    # reads actual strides from the runtime tensor.  Last-dim stride is always 1.

    # q_latent: [batch_size, seq_len_q, num_heads, latent_dim] — non-contiguous slice
    q_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # q_rope: [batch_size, seq_len_q, num_heads, rope_dim] — non-contiguous slice
    q_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # c_latent: [kv_batch, seq_len_k, latent_dim] — non-contiguous slice
    # kv_batch is a separate sym_int from query batch: paged KV cache uses a flat
    # pool so kv_batch=num_pages at runtime, while query batch can be any value.
    c_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # c_rope: [kv_batch, seq_len_k, rope_dim] — non-contiguous slice
    c_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # page_table: [batch_size, page_count] — contiguous
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch, sym_page_count),
        stride_order=(1, 0),
        assumed_align=4,
    )
    # o: [batch_size, seq_len_q, num_heads, latent_dim] — contiguous
    o_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_out_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    if is_workspace_size_zero:
        workspace_fake = None
    else:
        # workspace: 1-D int8 buffer. 32-byte alignment because workspace stores
        # fp32 partial sums internally, requiring stricter alignment than tensors.
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (sym_workspace_size,),
            assumed_align=32,
        )
    # cache_seqs: [batch_size] — int32
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=4,
    )
    # block_split_kvs: [batch_size] — int32 (only needed for is_var_split_kv=True)
    if is_var_split_kv:
        block_split_kvs_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_batch,),
            assumed_align=4,
        )
    else:
        block_split_kvs_fake = None

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        None,  # lse (disabled)
        workspace_fake,
        Int32(1),  # split_kv placeholder
        cache_seqs_fake,
        block_split_kvs_fake,
        Float32(1.0),  # softmax_scale placeholder
        Float32(1.0),  # output_scale placeholder
        stream_fake,
        use_pdl,
        options="--enable-tvm-ffi --opt-level 2",
    )

    return compiled_kernel


def tokenspeed_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    is_var_seq: bool = True,
    causal_mask: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """CuTe DSL MLA decode kernel for Blackwell SM100.

    Parameters
    ----------
    query : torch.Tensor
        [B, q_len, H, D_qk] where D_qk = kv_lora_rank + qk_rope_head_dim
    kv_cache : torch.Tensor
        [num_pages, page_size, D_ckv + D_kpe] (3D) or [num_pages, 1, page_size, D_ckv + D_kpe] (4D)
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer (uint8). Required size depends on batch size
        and split_kv (auto-computed from B, q_len, and number of SMs):

        - Formula: ``B * H * q_len * split_kv * (kv_lora_rank + 1) * 4`` bytes
          (0 when split_kv == 1, which happens when B >= num_SMs / 2)
        - The TokenSpeed runtime backend grows this buffer from the actual
          q_len before each decode launch.
    kv_lora_rank : int
        Latent dimension (e.g. 512).
    qk_rope_head_dim : int
        RoPE dimension (e.g. 64).
    block_tables : torch.Tensor
        [B, max_pages] — page table indices.
    seq_lens : torch.Tensor
        [B] — per-request KV sequence lengths.
    max_seq_len : int
        Maximum sequence length across the batch.
    softmax_scale : float
        Scale factor for QK^T before softmax.
    output_scale : float
        Scale factor applied to the output.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor [B, q_len, H, kv_lora_rank].
    is_var_seq : bool
        Whether the sequence length is variable.
        If True, the sequence length is variable.
        Otherwise,the sequence length is fixed for all the requests in the batch.
    causal_mask : bool
        Whether to enable causal masking in the CuTe DSL kernel.
        Currently this is effective for the FP8 kernel path.
    enable_pdl : bool
        When True, enables Programmatic Dependent Launch (PDL) on the
        underlying CuTe DSL decode kernel. Tokenspeed callers wire this from
        ``pdl_enabled()`` so ``--disable-pdl`` propagates through to the
        kernel binary; ``use_pdl`` is part of the kernel cache key.

    Returns
    -------
    torch.Tensor
        Output tensor [B, q_len, H, kv_lora_rank].
    """
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    assert (
        query.dtype in supported_dtypes
    ), f"tokenspeed_mla_decode only supports {supported_dtypes}, got {query.dtype}"
    assert (
        kv_cache.dtype == query.dtype
    ), f"kv_cache dtype {kv_cache.dtype} must match query dtype {query.dtype}"
    B, q_len, H, D_qk = query.shape
    assert D_qk == kv_lora_rank + qk_rope_head_dim

    q_dtype = query.dtype

    # Handle 3D vs 4D kv_cache: normalize to 3D [num_pages, page_size, D_total]
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1]

    # Split query into latent and rope components — keep contiguous [B, q_len, H, D].
    # The kernel's __call__ reinterprets to [H, D, q_len, B] via zero-cost make_tensor.
    q_latent_k = query[..., :kv_lora_rank]
    q_rope_k = query[..., kv_lora_rank:]

    # KV cache slices — keep contiguous [num_pages, page_size, D].
    # The kernel reinterprets to [page_size, D, num_pages] internally.
    c_latent_k = kv_cache[:, :, :kv_lora_rank]
    c_rope_k = kv_cache[:, :, kv_lora_rank:]

    # Page table: [B, max_pages]: passed directly, kernel reinterprets.
    page_table_k = block_tables

    # Runtime validation (int comparisons only, negligible overhead)
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    # H=128: standard config. When H is smaller than M tile, fold only by a
    # factor that exactly divides q_len; otherwise leave q_len on the scheduler
    # dimension.
    mma_m_tile = 128
    fold_sq_factor = get_mla_decode_fold_sq_factor(H, q_len, mma_m_tile)

    # Effective dimensions used by split_kv/workspace accounting.
    H_eff = H * fold_sq_factor
    q_len_eff = q_len // fold_sq_factor

    # Cached split_kv and workspace_size computation
    max_active_blocks = get_num_sm(query.device)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        B, q_len_eff, H_eff, kv_lora_rank, max_active_blocks
    )

    # Prepare workspace: slice of contiguous 1D buffer is already contiguous
    assert (
        workspace_buffer.dtype == torch.int8
    ), f"workspace_buffer must be torch.int8, got {workspace_buffer.dtype}"
    assert workspace_buffer.numel() >= workspace_size, (
        f"workspace_buffer too small: {workspace_buffer.numel()} bytes, "
        f"need {workspace_size} bytes"
    )
    is_workspace_size_zero = workspace_size == 0
    if is_workspace_size_zero:
        workspace_bytes = None
    else:
        workspace_bytes = workspace_buffer[:workspace_size]
    # Output buffer: contiguous [B, q_len, H, D].
    # Kernel reinterprets to [H, D, q_len, B] internally via zero-cost make_tensor.
    # FP8 kernel writes BF16 output for better downstream precision.
    out_dtype = torch.bfloat16 if q_dtype == torch.float8_e4m3fn else q_dtype
    if out is not None:
        o_k = out
    else:
        o_k = torch.empty(
            (B, q_len, H, kv_lora_rank), dtype=out_dtype, device=query.device
        )

    # cache_seqs: per-batch sequence lengths (skip .to() if already int32)
    cache_seqs = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)

    is_var_split_kv = False
    block_split_kvs = None
    skip_correction_threshold = 0.0

    # For fixed-length input, set is_persistent to True; otherwise, set to False.
    is_persistent = not is_var_seq

    # Validate configuration (cached, negligible overhead after first call)
    _check_can_implement(
        torch_dtype=q_dtype,
        page_size=page_size,
        num_heads=H,
        seq_len_q=q_len,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
    )

    # Get compiled kernel (cached after first compile)
    # Note: when is_workspace_size_zero is True, workspace_bytes is None and it will launch one kernel without workspace.
    # Otherwise, workspace_bytes is not None and it will launch two kernels.
    compiled_kernel = _get_compiled_mla_kernel(
        torch_dtype=q_dtype,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        skip_correction_threshold=skip_correction_threshold,
        is_workspace_size_zero=is_workspace_size_zero,
        fold_sq_factor=fold_sq_factor,
        causal_mask=causal_mask,
        num_heads=H,
        seq_len_q=q_len,
        use_pdl=enable_pdl,
    )

    # TVM FFI env stream must be set to PyTorch's current stream so the kernel
    # runs on the same stream as upstream PyTorch ops. CuTe tensors flow through
    # __tvm_ffi_object__ which does NOT auto-infer PyTorch's current stream;
    # use_torch_stream() binds it explicitly. Symmetric with mla_prefill.py.
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        compiled_kernel(
            q_latent_k,
            q_rope_k,
            c_latent_k,
            c_rope_k,
            page_table_k,
            o_k,
            None,  # lse (disabled)
            workspace_bytes,
            Int32(split_kv),
            cache_seqs,
            block_split_kvs,
            Float32(softmax_scale),
            Float32(output_scale),
        )

    # If out was provided, kernel already wrote into it — return directly.
    if out is not None:
        return out

    # o_k is [B, q_len, H, D] — return as-is to match trtllm-gen output shape.
    return o_k
