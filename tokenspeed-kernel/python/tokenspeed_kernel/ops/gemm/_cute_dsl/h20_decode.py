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

"""Bounded BF16 M=1 projections for NVIDIA H20.

The shape table is a candidate dispatch set, not a performance claim. Other
shapes/devices keep their existing GEMM. Capture and serving only use a shape
that completed an eager call with the same device and PDL policy.
"""

from __future__ import annotations

import functools
import threading
from types import MappingProxyType

import torch
import torch.nn.functional as F
from tokenspeed_kernel.compile_monitor import is_serving
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16 import LLBf16Router
from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
    ShapeDynamicSkinnyGemm,
    SkinnyGemmConfig,
)

# Keep the candidate limited to these exact M=1 shapes. Larger batches or
# a different SM90 product need separate validation.
H20_DECODE_ROUTES = MappingProxyType(
    {
        (1, 2624, 6144): "ll_bf16",
        (1, 4096, 2048): "skinny",
        (1, 160, 6144): "ll_bf16",
        (1, 512, 2048): "ll_bf16",
        (1, 512, 6144): "ll_bf16",
        (1, 128, 6144): "ll_bf16",
        (1, 1152, 6144): "ll_bf16",
    }
)
# Explicit skinny configuration for M=1, N=4096, K=2048.
_SKINNY_CONFIG = SkinnyGemmConfig(
    num_rows=1, block_size=64, outputs_per_block=2, k_unroll=2, vector_width=8
)
# The retained skinny driver's compile key has no PDL field. Separate owners
# prevent a policy change from reusing a callable compiled for the other mode.
_DRIVERS = {
    False: (LLBf16Router(), ShapeDynamicSkinnyGemm()),
    True: (LLBf16Router(), ShapeDynamicSkinnyGemm()),
}
_WarmKey = tuple[int, int, int, int, bool]
_warmed: set[_WarmKey] = set()
_warmed_lock = threading.Lock()


def _shape_supported(m: int, n: int, k: int) -> bool:
    return (m, n, k) in H20_DECODE_ROUTES


@functools.lru_cache(maxsize=16)
def _is_h20_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (
        9,
        0,
    ) and torch.cuda.get_device_name(device_index) in {"H20", "NVIDIA H20"}


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.numel() != 0
        and b.numel() != 0
        and a.device == b.device
        and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    )


def _call_key(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> _WarmKey | None:
    if (
        x.ndim != 2
        or weight.ndim != 2
        or not x.is_cuda
        or x.device != weight.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.shape[1] != weight.shape[1]
        or x.stride() != (x.shape[1], 1)
        or weight.stride() != (weight.shape[1], 1)
        or not _shape_supported(x.shape[0], weight.shape[0], weight.shape[1])
        or x.data_ptr() % 32
        or weight.data_ptr() % 32
    ):
        return None
    if out is not None and (
        tuple(out.shape) != (x.shape[0], weight.shape[0])
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
        or out.stride() != (weight.shape[0], 1)
        or out.data_ptr() % 32
        or _shares_storage(out, x)
        or _shares_storage(out, weight)
    ):
        return None
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    # Capture state is queried for the current device. A foreign-device call
    # keeps Torch's device guard instead of compiling through this shortcut.
    if device_index != torch.cuda.current_device():
        return None
    if not _is_h20_device(device_index):
        return None
    return (device_index, x.shape[0], weight.shape[0], weight.shape[1], pdl_enabled())


def _key_supported(key: _WarmKey, x: torch.Tensor, weight: torch.Tensor) -> bool:
    if (
        is_serving() or torch.cuda.is_current_stream_capturing()
    ) and key not in _warmed:
        return False
    _, m, n, k, pdl = key
    ll_driver, skinny_driver = _DRIVERS[pdl]
    # Both drivers need quack's fake-tensor utilities at compile time. The
    # LL availability check covers that import as well as CuTe DSL.
    if not ll_driver.is_available():
        return False
    if H20_DECODE_ROUTES[(m, n, k)] == "ll_bf16":
        return ll_driver.supports(x, weight, m)
    return skinny_driver.is_available() and skinny_driver.supports(
        _SKINNY_CONFIG, m, n, k
    )


def h20_decode_supported(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> bool:
    """Whether this call may use a warmed or startup H20 candidate projection.

    Args:
        x: Contiguous BF16 activation with shape ``[1, K]``.
        weight: Contiguous BF16 weight with shape ``[N, K]``.
        out: Optional aligned BF16 destination with shape ``[1, N]``, with
            storage separate from both operands.

    Returns:
        False for another device/shape/layout or an unwarmed capture/serving
        call. Startup may compile only one of the seven fixed shapes. As with
        the other GEMMs, the process PDL policy must stay fixed during a call.
    """
    key = _call_key(x, weight, out)
    return key is not None and _key_supported(key, x, weight)


def _torch_fallback(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    if out is not None and (_shares_storage(out, x) or _shares_storage(out, weight)):
        raise ValueError("out storage must be separate from both operands")
    if x.dtype == torch.bfloat16 and weight.dtype == torch.float32:
        x = x.float()
    if out is None:
        return F.linear(x, weight)
    if (
        x.ndim != 2
        or weight.ndim != 2
        or tuple(out.shape) != (x.shape[0], weight.shape[0])
        or out.dtype != x.dtype
        or out.device != x.device
    ):
        raise ValueError(
            "out must match the GEMM shape, result dtype and device, "
            "with storage separate from both operands"
        )
    return torch.mm(x, weight.t(), out=out)


def cute_dsl_h20_decode(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute ``x @ weight.T`` through the bounded H20 BF16 candidate.

    Args:
        x: Activation with shape ``[M, K]``.
        weight: Projection weight with shape ``[N, K]``.
        out: Optional destination; invalid shape/dtype/device or shared operand
            storage raises, including disjoint views of a shared allocation.

    Returns:
        BF16 output for supported inputs, otherwise the ordinary Torch GEMM.
        A successful eager call makes this shape safe for later capture and
        serving; those phases never compile a missing candidate specialization.
    """
    key = _call_key(x, weight, out)
    if key is None or not _key_supported(key, x, weight):
        return _torch_fallback(x, weight, out)
    _, m, n, k, pdl = key
    ll_driver, skinny_driver = _DRIVERS[pdl]
    if H20_DECODE_ROUTES[(m, n, k)] == "ll_bf16":
        result = ll_driver(x.detach(), weight.detach(), out, out_dtype=torch.bfloat16)
    else:
        result = skinny_driver(x.detach(), weight.detach(), _SKINNY_CONFIG, out=out)
    if not is_serving() and not torch.cuda.is_current_stream_capturing():
        with _warmed_lock:
            _warmed.add(key)
    return result
