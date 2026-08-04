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

"""Non-owning tensor view for CUDA-graph break closures.

``weak_ref_tensor(t)`` returns a tensor aliasing ``t``'s memory WITHOUT owning
its storage (``at::from_blob`` with a no-op deleter). A breakable-CUDA-graph
break closure that holds such a view does not pin ``t``'s mempool block, so the
graph pool can recycle blocks across segments and captures -- graph memory
drops from sum-over-buckets of break inputs back to ~peak-live. Correctness
relies on replay order: the captured segment rewrites the aliased address
before the break reads it, every replay (the same discipline that lets the
decode graph share one pool across batch sizes).

Adapted from vLLM ``csrc/ops.h`` / SGLang ``sgl-kernel/csrc/memory/
weak_ref_tensor.cpp``. Compiled lazily via ``torch.utils.cpp_extension``
(pure C++ against ATen -- no nvcc, no wheel rebuild); falls back to identity
(strong ref: correct, more memory) if compilation is unavailable.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_CPP_SOURCE = r"""
#include <ATen/ATen.h>
#include <vector>

at::Tensor weak_ref_tensor(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_cuda(), "weak_ref_tensor expects a CUDA tensor");
  void* data_ptr = tensor.data_ptr();
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();
  auto options = tensor.options();
  return at::from_blob(data_ptr, sizes, strides, options);
}
"""

_module = None
_load_failed = False


def _load():
    global _module, _load_failed
    if _module is not None or _load_failed:
        return _module
    try:
        from torch.utils.cpp_extension import load_inline

        _module = load_inline(
            name="tokenspeed_weak_ref_tensor",
            cpp_sources=_CPP_SOURCE,
            functions=["weak_ref_tensor"],
            with_cuda=False,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - host without a C++ toolchain
        _load_failed = True
        logger.warning(
            "weak_ref_tensor extension unavailable (%s: %s); falling back to "
            "identity (strong refs pin graph-pool blocks -- correct, but "
            "breakable-graph capture memory scales with the bucket sum).",
            type(exc).__name__,
            exc,
        )
    return _module


def weak_ref_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return a non-owning view of CUDA tensor ``t`` (alias, same shape/strides).

    Args:
        t: A CUDA tensor. The caller must guarantee the aliased memory is
            rewritten (or still valid) whenever the view is consumed -- for
            breakable-graph closures this holds because the captured segment
            producing ``t`` replays before the break reads the view.

    Returns:
        A tensor sharing ``t``'s memory without owning its storage, or ``t``
        itself if the extension is unavailable (identity fallback).
    """
    mod = _load()
    if mod is None:
        return t
    return mod.weak_ref_tensor(t)
