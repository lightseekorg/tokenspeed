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
CuTe DSL MLA Prefill Kernel Wrapper
====================================

Wraps BlackwellFusedMultiHeadAttentionForward for ragged MLA prefill on Blackwell SM100.
No padding required — kernel handles ragged varlen directly.
"""

import functools
import logging
import math
import os

LOG2_E = math.log2(math.exp(1.0))  # ≈ 1.4426950408889634
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.base_dsl.arch import Arch
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import BaseDSL
from tokenspeed_mla import fmha_binary as _fmha_binary
from tokenspeed_mla import fmha_helpers as fmha_utils
from tokenspeed_mla.fmha import (
    BlackwellFusedMultiHeadAttentionForward,
)
from tokenspeed_mla.utils import torch_to_cutlass_dtype

logger = logging.getLogger(__name__)

# Backend selection via env var. Values: "cutedsl" (default) or "binary" (AOT SO).
_PREFILL_BACKEND_ENV = os.environ.get(
    "TOKENSPEED_MLA_PREFILL_BACKEND", "cutedsl"
).lower()


@functools.lru_cache(maxsize=None)
def _resolve_backend() -> str:
    """Resolve the effective prefill backend, called once on first inference."""
    if _PREFILL_BACKEND_ENV == "binary":
        if not _fmha_binary.has_binary_prefill():
            raise RuntimeError(
                "TOKENSPEED_MLA_PREFILL_BACKEND=binary requested but no binary SO "
                "found for this GPU. Check tokenspeed_mla/objs/ or set "
                "TOKENSPEED_MLA_FMHA_BINARY_SO to the .so path."
            )
        logger.info("MLA prefill: using binary backend")
        return "binary"
    return "cutedsl"


def _to_cute(src: torch.Tensor, dtype):
    """Convert a torch tensor to cute tensor with TVM-FFI support."""
    ct = from_dlpack(src.detach(), assumed_align=16, enable_tvm_ffi=True)
    ct.element_type = dtype
    return ct


def _to_cute_1d(t: torch.Tensor):
    """Convert a 1D int32 tensor (cum_seqlen) to cute with TVM-FFI.
    Returns (cute_tensor, torch_tensor) to keep backing storage alive."""
    t = t.to(torch.int32) if t.dtype != torch.int32 else t
    if not t.is_cuda:
        t = t.cuda()
    ct = from_dlpack(t.detach(), assumed_align=4, enable_tvm_ffi=True)
    ct.element_type = cutlass.Int32
    return ct, t


def _enable_ex2_emulation() -> bool:
    arch = BaseDSL._get_dsl().get_arch_enum()
    if Arch.sm_100 <= arch <= Arch.sm_100f:
        # On sm100, default output is False.
        return False

    if Arch.sm_103 <= arch <= Arch.sm_103f:
        # On sm103, it must be False.
        return False

    raise NotImplementedError(f"MLA prefill not implemented for arch={arch}.")


def _compile_prefill_kernel(
    q_dtype: torch.dtype,
    head_dim_qk: int,
    head_dim_v: int,
    is_causal: bool,
    return_lse: bool,
    use_pdl: bool = False,
    enable_ex2_emulation: bool = True,
):
    """Compile FMHA prefill kernel with fully fake/symbolic tensors.

    New fmha.py accepts cute.Tensor (not .iterator) and handles ragged
    varlen without padding. All tensor shapes are symbolic — compile once,
    reuse for any buffer size.
    """
    cutlass_dtype = torch_to_cutlass_dtype(q_dtype)
    out_cutlass_dtype = cutlass.BFloat16

    mma_tiler_mn = (128, 128)
    head_dim = (head_dim_qk, head_dim_v) if head_dim_qk != head_dim_v else head_dim_qk
    mask_type = (
        fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
        if is_causal
        else fmha_utils.MaskEnum.RESIDUAL_MASK
    )

    kernel_obj = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=mma_tiler_mn,
        head_dim=head_dim,
        is_persistent=False,
        mask_type=mask_type,
        enable_ex2_emulation=enable_ex2_emulation,
        enable_skip_correction=True,
    )

    # Symbolic cum_seqlen — allows variable batch_size without recompile.
    sym_cum_len = cute.sym_int()
    cum_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_cum_len,),
        assumed_align=4,
    )

    problem_size_compile = tuple(Int32(1) for _ in range(8))
    stream_compile = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    window_right = Int32(0) if is_causal else None

    logger.info(
        f"Compiling CuteDSL FMHA prefill kernel: "
        f"head_dim=({head_dim_qk},{head_dim_v}), causal={is_causal}, lse={return_lse}"
    )

    # New API: pass cute.Tensor directly (not .iterator).
    # Fake tensors with correct row-major 5D strides for compile-once.
    # Layout: (1, total, h_k, h_r, d) with row-major strides.
    sym_s_q = cute.sym_int()
    sym_s_kv = cute.sym_int()

    # Row-major strides for Q/O: (1, total_q, h_k, h_r, d_qk)
    #   stride = (total*h_k*h_r*d, h_k*h_r*d, h_r*d, d, 1)
    # With symbolic total, intermediate strides are also symbolic.
    q_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        shape=(1, sym_s_q, cute.sym_int(), cute.sym_int(), head_dim_qk),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        shape=(1, sym_s_kv, cute.sym_int(), 1, head_dim_qk),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        shape=(1, sym_s_kv, cute.sym_int(), 1, head_dim_v),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    o_fake = cute.runtime.make_fake_tensor(
        out_cutlass_dtype,
        shape=(1, sym_s_q, cute.sym_int(), cute.sym_int(), head_dim_v),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )

    if return_lse:
        # LSE: (total_q, h_q) row-major
        lse_fake = cute.runtime.make_fake_tensor(
            cutlass.Float32,
            shape=(sym_s_q, cute.sym_int()),
            stride=(cute.sym_int(), 1),
            assumed_align=4,
        )
    else:
        lse_fake = None

    compiled = cute.compile(
        kernel_obj,
        q_fake,
        k_fake,
        v_fake,
        o_fake,
        problem_size_compile,
        cum_fake,
        cum_fake,
        lse_fake,
        Float32(1.0),
        Float32(1.0),
        Float32(1.0),
        None,
        None,
        window_right,
        None,
        None,
        stream_compile,
        use_pdl,
        options="--enable-tvm-ffi --opt-level 2",
    )
    return compiled


# Cache: (dtype, d_qk, d_v, h_q, h_k, is_causal, return_lse, pdl, ex2)
# New fmha.py accepts cute.Tensor and handles varlen without padding.
# TMA descriptors rebuilt from runtime tensors. Buffer sizes NOT in key.
_compiled_kernels = {}

_call_count = 0


def warmup_compile_prefill(
    q_dtype: torch.dtype = torch.float8_e4m3fn,
    d_qk: int = 192,
    d_v: int = 128,
    enable_pdl: bool = False,
) -> None:
    """Eagerly compile all prefill kernel variants used by tokenspeed_mla.

    In production, tokenspeed_mla_prefill is invoked with 2 distinct compile keys:
      - (q_dtype, d_qk, d_v, is_causal=True,  return_lse=True)  chunker causal path
      - (q_dtype, d_qk, d_v, is_causal=False, return_lse=True)  chunker chunk path
    Normally each compile happens on the first matching call (1.5-2 minutes after
    server start, depending on which request first triggers chunked prefill).
    This helper forces both compiles up front so no JIT happens during serving.

    Safe to call multiple times: if a config is already in `_compiled_kernels`,
    it is skipped. Called from `CuteDSLMLABackend.__init__` at server startup.
    """
    enable_ex2_emulation = _enable_ex2_emulation()
    for is_causal in (True, False):
        for return_lse in (True,):  # tokenspeed only uses return_lse=True
            config = (
                q_dtype,
                d_qk,
                d_v,
                is_causal,
                return_lse,
                enable_pdl,
                enable_ex2_emulation,
            )
            if config in _compiled_kernels:
                continue
            logger.info(
                f"Pre-compiling CuteDSL FMHA prefill kernel: "
                f"head_dim=({d_qk},{d_v}), causal={is_causal}, lse={return_lse} "
                f"use_pdl={enable_pdl}, ex2_emulation={enable_ex2_emulation} "
                f"(warmup, no serving traffic yet)"
            )
            _compiled_kernels[config] = _compile_prefill_kernel(
                q_dtype,
                d_qk,
                d_v,
                is_causal,
                return_lse,
                use_pdl=enable_pdl,
                enable_ex2_emulation=enable_ex2_emulation,
            )


def tokenspeed_mla_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens: torch.Tensor,
    max_seq_len: int,
    batch_size: int,
    softmax_scale: float,
    is_causal: bool = True,
    return_lse: bool = False,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    enable_pdl: bool = False,
) -> "torch.Tensor | Tuple[torch.Tensor, torch.Tensor]":
    """CuTe DSL FMHA prefill kernel for MLA on Blackwell SM100.

    Q/K/V are plain ragged tensors — no padding required:
      Q shape: [sum(q_lens), h_q, d_qk]
      K shape: [sum(kv_lens), h_k, d_qk]
      V shape: [sum(kv_lens), h_k, d_v]
    """
    total_q_tokens, h_q, d_qk = query.shape
    total_kv_tokens, h_k, _ = key.shape
    d_v = value.shape[2]
    h_r = h_q // h_k
    q_dtype = query.dtype
    cutlass_dtype = torch_to_cutlass_dtype(q_dtype)
    out_torch_dtype = torch.bfloat16
    out_cutlass_dtype = cutlass.BFloat16

    # Q and KV may have different cum_seq_lens and max_seq_len.
    cum_seq_lens_kv = cum_seq_lens
    max_seq_len_kv = max_seq_len
    if cum_seq_lens_q is None:
        cum_seq_lens_q = cum_seq_lens_kv
    if max_seq_len_q is None:
        max_seq_len_q = max_seq_len_kv

    global _call_count
    _call_count += 1

    if _call_count <= 3:
        logger.debug(
            f"tokenspeed_mla_prefill called (#{_call_count}): "
            f"q_tokens={total_q_tokens}, kv_tokens={total_kv_tokens}, "
            f"h_q={h_q}, h_k={h_k}, h_r={h_r}, d_qk={d_qk}, d_v={d_v}, "
            f"bs={batch_size}, max_sq={max_seq_len_q}, max_sk={max_seq_len_kv}, "
            f"lse={return_lse}"
        )

    # Reshape to 5D layout expected by fmha.py:
    #   Q/O: (1, total_q, h_k, h_r, d)
    #   K/V: (1, total_kv, h_k, 1, d)
    q_5d = query.reshape(1, total_q_tokens, h_k, h_r, d_qk)
    k_5d = key.reshape(1, total_kv_tokens, h_k, 1, d_qk)
    v_5d = value.reshape(1, total_kv_tokens, h_k, 1, d_v)

    q_ct = _to_cute(q_5d, cutlass_dtype)
    k_ct = _to_cute(k_5d, cutlass_dtype)
    v_ct = _to_cute(v_5d, cutlass_dtype)

    # Output: BF16, same 5D layout. The kernel writes (out=0, lse=-inf) for
    # rows in batches where seqlen_k==0, so no pre-init is required.
    o_torch = torch.empty(
        total_q_tokens, h_q, d_v, dtype=out_torch_dtype, device=query.device
    )
    o_5d = o_torch.view(1, total_q_tokens, h_k, h_r, d_v)
    o_ct = _to_cute(o_5d, out_cutlass_dtype)

    # Cumulative sequence lengths.
    # Keep backing tensors alive until after kernel launch.
    cum_q_ct, _cum_q_backing = _to_cute_1d(cum_seq_lens_q)
    cum_k_ct, _cum_k_backing = _to_cute_1d(cum_seq_lens_kv)

    if _resolve_backend() == "binary":
        # Binary backend: LSE layout is (1, h_k, h_r, total_q) — differs from CuteDSL's
        # (total_q, h_q). The binary SO was AOT-compiled with that layout.
        if return_lse:
            lse_binary = torch.empty(
                1, h_k, h_r, total_q_tokens, dtype=torch.float32, device=query.device
            )
            lse_binary_ct = _to_cute(lse_binary, cutlass.Float32)
        else:
            lse_binary = None
            lse_binary_ct = None

        _fmha_binary.call_binary_prefill(
            q_ct,
            k_ct,
            v_ct,
            o_ct,
            lse_binary_ct,
            (
                batch_size,
                max_seq_len_q,
                total_q_tokens,
                max_seq_len_kv,
                h_q,
                h_k,
                d_qk,
                d_v,
            ),
            cum_q_ct,
            cum_k_ct,
            softmax_scale * LOG2_E,
            softmax_scale,
            is_causal,
            return_lse,
        )

        if return_lse:
            # Reshape (1, h_k, h_r, total_q) → (total_q, h_q) to match CuteDSL output.
            lse_torch = (
                lse_binary.squeeze(0)
                .permute(2, 0, 1)
                .reshape(total_q_tokens, h_q)
                .contiguous()
            )
            return o_torch, lse_torch
        return o_torch

    # CuteDSL backend (default): JIT-compile kernel on first use, cache thereafter.
    if return_lse:
        lse_torch = torch.empty(
            (total_q_tokens, h_q),
            dtype=torch.float32,
            device=query.device,
        )
        lse_ct = _to_cute(lse_torch, cutlass.Float32)
    else:
        lse_torch = None
        lse_ct = None

    # Cache by static config — all shapes are symbolic at compile time, but
    # ``enable_pdl`` and ``enable_ex2_emulation`` flip constexpr paths inside
    # the kernel and produce different SASS, hence keying on both.
    enable_ex2_emulation = _enable_ex2_emulation()
    config = (
        q_dtype,
        d_qk,
        d_v,
        is_causal,
        return_lse,
        enable_pdl,
        enable_ex2_emulation,
    )
    if config not in _compiled_kernels:
        _compiled_kernels[config] = _compile_prefill_kernel(
            q_dtype,
            d_qk,
            d_v,
            is_causal,
            return_lse,
            use_pdl=enable_pdl,
            enable_ex2_emulation=enable_ex2_emulation,
        )
    compiled_kernel = _compiled_kernels[config]

    scale_softmax_log2 = softmax_scale * LOG2_E

    # problem_size: (b, s_q, s_lse, s_k, h_q, h_k, d, dv)
    problem_size = (
        Int32(batch_size),
        Int32(max_seq_len_q),
        Int32(total_q_tokens),
        Int32(max_seq_len_kv),
        Int32(h_q),
        Int32(h_k),
        Int32(d_qk),
        Int32(d_v),
    )
    window_right = Int32(0) if is_causal else None

    # TVM FFI env stream must be set to PyTorch's current stream.
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        compiled_kernel(
            q_ct,
            k_ct,
            v_ct,
            o_ct,
            problem_size,
            cum_q_ct,
            cum_k_ct,
            lse_ct,
            Float32(scale_softmax_log2),
            Float32(softmax_scale),
            Float32(1.0),
            None,
            None,
            window_right,
            None,
            None,
        )

    if return_lse:
        return o_torch, lse_torch
    return o_torch
