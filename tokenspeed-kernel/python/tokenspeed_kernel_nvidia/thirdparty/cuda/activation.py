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

"""Activation ops: SiLU+Mul fused with FP8 / NVFP4 block quantize."""

import functools
from pathlib import Path
from typing import Optional, Tuple

import torch


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@functools.cache
def _load_silu_fuse_block_quant_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "silu_fuse_block_quant"
    so_path = objs_dir / "silu_fuse_block_quant.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel silu_fuse_block_quant library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def silu_and_mul_fuse_block_quant(
    input: torch.Tensor,
    scale_out: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
    num_tokens_per_expert: Optional[torch.Tensor] = None,
    num_tokens_hint: Optional[int] = None,
    num_experts: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=torch.float8_e4m3fn,
        )
    mod = _load_silu_fuse_block_quant_module()
    if num_tokens_per_expert is not None:
        assert num_tokens_hint is not None
        assert num_experts is not None
        mod.silu_and_mul_fused_block_quant_ep(
            out,
            scale_out,
            input,
            bool(enable_pdl),
            num_tokens_per_expert,
            int(num_tokens_hint),
            int(num_experts),
        )
    else:
        mod.silu_and_mul_fused_block_quant(
            out,
            scale_out,
            input,
            bool(enable_pdl),
        )
    return out, scale_out


@functools.cache
def _load_silu_fuse_nvfp4_quant_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "silu_fuse_nvfp4_quant"
    so_path = objs_dir / "silu_fuse_nvfp4_quant.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel silu_fuse_nvfp4_quant library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def silu_and_mul_fuse_nvfp4_quant(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    enable_pdl: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU_and_Mul + NVFP4 quantize for dense MLPs (SM100+).

    Takes a concatenated gate|up tensor of shape ``[M, 2*I]`` (bf16/fp16)
    and writes:
      - Packed NVFP4 output of shape ``[M, I/2]`` (uint8, two e2m1 per byte).
      - Block scale factors of shape ``[padded_M, padded_K]`` (float8_e4m3fn)
        in the 128x4 swizzled layout that ``mm_fp4`` / cuBLASLt consume
        directly, where ``padded_M = round_up(M, 128)`` and
        ``padded_K = round_up(I / 16, 4)``.

    The kernel is PDL-wired (``griddepcontrol.wait`` / ``launch_dependents``)
    so it can overlap with the surrounding GEMMs when launched with PDL.

    Args:
        input: ``[M, 2*I]`` bf16 or fp16, concatenated gate|up.
        global_scale: ``[1]`` float32. The scale-up factor
            (i.e. ``layer.input_scale_inv`` = ``448 * 6 / amax``).
        enable_pdl: honor upstream/downstream PDL if True.

    Returns:
        ``(out_fp4, out_sf)``.
    """
    assert input.dim() == 2, "input must be 2-D [M, 2*I]"
    assert input.dtype in (torch.bfloat16, torch.float16), "input must be bf16 or fp16"
    M, two_I = input.shape
    assert two_I % 32 == 0, "2*I must be multiple of 32"
    I = two_I // 2
    sf_vec_size = 16
    padded_m = _round_up(M, 128)
    padded_k = _round_up(I // sf_vec_size, 4)

    out = torch.empty(M, I // 2, dtype=torch.uint8, device=input.device)
    # Scale buffer is [padded_M, padded_K] fp8_e4m3fn laid out as 128x4
    # swizzle. The kernel writes via uint32* (4 scales per uint32), so
    # padded_K must be a multiple of 4 (enforced by round_up above).
    scale_out = torch.empty(
        padded_m, padded_k, dtype=torch.float8_e4m3fn, device=input.device
    )
    if global_scale.dim() == 0:
        global_scale = global_scale.view(1)

    mod = _load_silu_fuse_nvfp4_quant_module()
    mod.silu_and_mul_fuse_nvfp4_quant(
        out,
        scale_out,
        input.contiguous() if not input.is_contiguous() else input,
        global_scale,
        bool(enable_pdl),
    )
    return out, scale_out
