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

"""merge_state: vendored from flashinfer's MergeStateKernel + lse_scale knob + PDL.

The CUDA source lives at ``thirdparty/cuda/csrc/merge_state.cu``. This module is
the public Python API — it loads the prebuilt ``.so`` lazily, validates inputs,
allocates outputs, and forwards to the kernel.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Tuple

import torch

# Common scale presets — multiplying LSE by these brings it into log2 domain,
# which is what the kernel uses internally (ex2.approx is the native PTX
# intrinsic on NVIDIA, so we always work in log2 and pre-/post-rescale instead).
LSE_LOG2 = 1.0
LSE_LN = math.log2(math.e)  # ≈ 1.4426950408889634


@functools.cache
def _load_merge_state_module():
    import tvm_ffi

    so_path = Path(__file__).parent / "objs" / "merge_state" / "merge_state.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel merge_state library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    *,
    inplace: bool = False,
    lse_scale_log2: float = LSE_LN,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention partials.

    Vendored from flashinfer's ``MergeStateKernel`` (see
    ``thirdparty/cuda/csrc/merge_state.cu``) with two additions: ``lse_scale_log2``
    so callers can pass LSE in any base, and PDL hooks so the kernel can overlap
    its preamble with upstream / downstream PDL-aware kernels.

    The kernel works in log2 space internally (PTX-native ``ex2.approx``).
    ``lse_scale_log2`` is the multiplier that converts the caller's LSE into
    log2 domain — pass ``LSE_LN = log2(e)`` (default) for natural-log LSE
    (trtllm-gen / FA cute / cuteDSL MLA convention) or ``LSE_LOG2 = 1.0`` if
    the caller's LSE is already in log2 (flashinfer C++ convention). The
    output LSE is returned in the same basis as the input.

    Parameters
    ----------
    v_a, v_b : ``[seq_len, num_heads, head_dim]``, ``bfloat16`` or ``float16``.
    s_a, s_b : ``[seq_len, num_heads]``, must be ``float32``.
    inplace : when ``True``, the merged output is written back into ``v_a`` and
        ``s_a`` (mutated) and the same tensors are returned. When ``False``,
        fresh buffers are allocated.
    lse_scale_log2 : multiplier mapping caller's LSE basis to log2.
    enable_pdl : opt into Programmatic Dependent Launch (Hopper+). Caller must
        also enable PDL on the upstream / downstream kernels for the overlap
        to materialize.

    Returns
    -------
    v_merged : ``[seq_len, num_heads, head_dim]`` in ``v_a.dtype``.
    s_merged : ``[seq_len, num_heads]`` in fp32, same basis as inputs.
    """
    assert v_a.is_contiguous() and v_b.is_contiguous()
    assert s_a.is_contiguous() and s_b.is_contiguous()
    assert v_a.shape == v_b.shape
    assert s_a.shape == s_b.shape
    assert v_a.shape[:2] == s_a.shape
    assert v_a.dtype == v_b.dtype
    assert v_a.dtype in (
        torch.bfloat16,
        torch.float16,
    ), f"merge_state V must be bf16/fp16, got {v_a.dtype}"
    assert (
        s_a.dtype == torch.float32 and s_b.dtype == torch.float32
    ), f"merge_state expects fp32 LSE, got s_a={s_a.dtype} s_b={s_b.dtype}"

    if inplace:
        v_merged = v_a
        s_merged = s_a
    else:
        v_merged = torch.empty_like(v_a)
        s_merged = torch.empty_like(s_a)
    _load_merge_state_module().merge_state(
        v_a,
        s_a,
        v_b,
        s_b,
        v_merged,
        s_merged,
        float(lse_scale_log2),
        bool(enable_pdl),
    )
    return v_merged, s_merged
