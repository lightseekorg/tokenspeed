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

"""Fused (residual + input) -> RMSNorm -> NVFP4 block-scale quantize.

Wraps the TRT-LLM ws_layernorm warp-specialized kernel ported under
``thirdparty/cuda/csrc/fused_add_rmsnorm_fp4_quant``. The kernel writes both
the FP4 packed output (consumed by the MoE GEMM) and, optionally, the bf16/fp16
high-precision normed values (consumed by the MoE gate / shared expert) in a
single pass over the hidden states.

Requires SM90+ (Hopper / Blackwell). Hidden dim must be in ``[2048, 16384]``
and divisible by 16.
"""

import functools
from pathlib import Path
from typing import Optional, Tuple

import torch


_FP4_BLOCK_SIZE = 16  # 16 FP4 values share one e4m3 SF
_M_PADDING = 32       # ws_layernorm vectorized stores assume m % 32 == 0


@functools.cache
def _load_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "fused_add_rmsnorm_fp4_quant"
    so_path = objs_dir / "fused_add_rmsnorm_fp4_quant.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel fused_add_rmsnorm_fp4_quant library not found at "
            f"{so_path}. Run: pip install -e tokenspeed-kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def fused_add_rmsnorm_fp4_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    sf_scale: Optional[torch.Tensor],
    *,
    eps: float,
    output_hp_norm: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run residual-add + RMSNorm + NVFP4 block-scale quant in a single kernel.

    Args:
        hidden_states: ``[M, N]`` fp16/bf16 contiguous activations from the
            previous projection (``hs + residual`` is the pre-norm).
        residual: ``[M, N]`` same dtype as ``hidden_states``; will be added
            into ``hidden_states`` to form the pre-norm. The addition result is
            also returned as the new residual for the next layer.
        weight: ``[N]`` RMSNorm gamma (same dtype as ``hidden_states``).
        sf_scale: optional ``[1]`` float32 scale (= ``1 / input_scale``) used
            by the FP4 block-scale quant. ``None`` defaults to ``1.0``.
        eps: RMSNorm epsilon.
        output_hp_norm: when True, also return the high-precision normed
            tensor (used by the MoE gate / shared expert path).

    Returns:
        ``(fp4_packed_uint8, residual_out, sf_uint8, hp_norm)``:

        * ``fp4_packed_uint8``: ``[M, N // 2]`` ``uint8`` (2 FP4 values per byte).
        * ``residual_out``: ``[M, N]`` same dtype as ``hidden_states``; this is
          ``hidden_states + residual`` (pre-norm), suitable as the residual
          input for the next decoder layer.
        * ``sf_uint8``: ``[M, N // 16]`` ``uint8`` block-scale factors in
          row-major LINEAR layout (consumer reinterprets as ``e4m3``); ready
          for ``trtllm_fp4_block_scale_moe`` without any unswizzle step.
        * ``hp_norm``: ``[M, N]`` same dtype as ``hidden_states`` if
          ``output_hp_norm`` else ``None``.
    """
    if hidden_states.dim() != 2:
        raise ValueError(
            f"fused_add_rmsnorm_fp4_quant expects 2-D input [M, N]; got {tuple(hidden_states.shape)}"
        )
    if residual.shape != hidden_states.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} must match hidden_states {tuple(hidden_states.shape)}"
        )
    if weight.dim() != 1 or weight.size(0) != hidden_states.size(1):
        raise ValueError(
            f"weight must be 1-D [N={hidden_states.size(1)}]; got {tuple(weight.shape)}"
        )
    dtype = hidden_states.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"fused_add_rmsnorm_fp4_quant only supports fp16/bf16; got {dtype}")
    if residual.dtype != dtype or weight.dtype != dtype:
        raise TypeError(
            f"hidden_states/residual/weight dtypes must match; got "
            f"{dtype}/{residual.dtype}/{weight.dtype}"
        )

    m, n = hidden_states.shape
    if n < 2048 or n > 16384 or n % _FP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"hidden dim N={n} must be in [2048, 16384] and divisible by {_FP4_BLOCK_SIZE}"
        )

    device = hidden_states.device
    m_padded = (m + _M_PADDING - 1) // _M_PADDING * _M_PADDING

    # FP4 packed output: kernel writes uint32_t (8 FP4 values per int32). Allocate
    # as int32 [m_padded, N//8] to match the kernel's stride, then expose to
    # downstream consumers as uint8 [M, N//2] (2 FP4 values per byte).
    normed_fp4_padded_i32 = torch.empty(
        (m_padded, n // 8), dtype=torch.int32, device=device
    )
    residual_out_padded = torch.empty((m_padded, n), dtype=dtype, device=device)

    sf_out_padded = torch.empty(
        (m_padded, n // _FP4_BLOCK_SIZE), dtype=torch.uint8, device=device
    )

    hp_norm_padded: Optional[torch.Tensor] = None
    if output_hp_norm:
        hp_norm_padded = torch.empty((m_padded, n), dtype=dtype, device=device)

    if sf_scale is not None:
        if sf_scale.dtype != torch.float32 or sf_scale.numel() != 1:
            raise TypeError("sf_scale must be a [1] float32 tensor")
        sf_scale_arg = sf_scale
    else:
        sf_scale_arg = None

    _load_module().fused_add_rmsnorm_fp4_quant(
        hidden_states,
        residual,
        weight,
        sf_scale_arg,
        float(eps),
        bool(output_hp_norm),
        normed_fp4_padded_i32,
        sf_out_padded,
        residual_out_padded,
        hp_norm_padded,
    )

    # Narrow back to [M, ...] views, and reinterpret the FP4 packed buffer as
    # uint8 [M, N // 2] so callers can pass it directly into the MoE GEMM.
    normed_fp4_i32 = (
        normed_fp4_padded_i32 if m_padded == m else normed_fp4_padded_i32[:m]
    )
    fp4_packed_uint8 = normed_fp4_i32.view(torch.uint8).view(m, n // 2)
    residual_out = residual_out_padded if m_padded == m else residual_out_padded[:m]
    sf_out = sf_out_padded if m_padded == m else sf_out_padded[:m]
    hp_norm = None
    if output_hp_norm:
        hp_norm = hp_norm_padded if m_padded == m else hp_norm_padded[:m]

    return fp4_packed_uint8, residual_out, sf_out, hp_norm
