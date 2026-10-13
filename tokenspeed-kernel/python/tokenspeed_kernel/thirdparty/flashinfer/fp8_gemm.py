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

"""Keep one cuBLASLt FP8 GEMM runner so its algorithm cache survives calls.

FlashInfer 0.7.x builds a new runner, with an empty per-shape cuBLASLt
algorithm cache, inside every ``fp8_gemm_sm100`` call, so each eager call
re-enumerates algorithms (~120 us of host time). One runner of the same class
keeps the tuned cache keys; its cache is bounded because prefill M varies.
"""

from __future__ import annotations

import collections
import functools
import inspect
import threading
import types

import torch
from flashinfer.gemm import gemm_base

# Recent exact shapes kept; each entry holds a 6.4 KB algorithm list.
_ALGO_CACHE_SHAPES = 256


class _RecentShapes(collections.OrderedDict):
    """The runner's per-shape algorithm cache, evicting the least recently used."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            if key not in self:
                return default
            self.move_to_end(key)
            return self[key]

    def __setitem__(self, key, value) -> None:
        with self._lock:
            super().__setitem__(key, value)
            if len(self) > _ALGO_CACHE_SHAPES:
                self.popitem(last=False)


@functools.cache
def _cublas_fp8_runner(device_index: int):
    upstream = type(gemm_base.get_gemm_module().cublas_fp8_gemm_runner())

    # The upstream class name keeps tuned tactics keyed to this runner.
    class CublasFp8GemmRunner(upstream):
        def __init__(self) -> None:
            super().__init__()
            self._algo_cache = _RecentShapes()

    return CublasFp8GemmRunner()


@functools.cache
def _private_fp8_gemm_sm100(device_index: int):
    raw = inspect.unwrap(gemm_base.fp8_gemm_sm100)
    namespace = dict(raw.__globals__)
    # cuBLASLt algorithms are enumerated per device, so each device keeps its own runner.
    namespace["get_gemm_module"] = lambda: types.SimpleNamespace(
        cublas_fp8_gemm_runner=lambda: _cublas_fp8_runner(device_index)
    )
    return types.FunctionType(
        raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__
    )


def cublas_fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Write ``a @ b`` dequantized by both scales into ``out`` with cuBLASLt.

    Args:
        a: ``[batch, M, K]`` row-major FP8 activations.
        b: ``[batch, K, N]`` column-major FP8 weights.
        a_scale: One-element FP32 dequant scale of ``a``.
        b_scale: One-element FP32 dequant scale of ``b``.
        out: ``[batch, M, N]`` dense BF16 or FP16 output.
    """
    # Streams can overlap (a forked shared expert), so each keeps its own scratch.
    stream = torch.cuda.current_stream(a.device).stream_id
    workspace = gemm_base._get_cache_buf(
        f"bmm_fp8_workspace_{stream}", gemm_base.DEFAULT_WORKSPACE_SIZE, a.device
    )
    _private_fp8_gemm_sm100(a.device.index)(
        a, b, a_scale, b_scale, out, workspace, ["cublas"]
    )
