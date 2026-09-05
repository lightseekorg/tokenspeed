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

"""FlashInfer's low-M split-K BF16 GEMM, driven by a caller-chosen tactic.

``mm_bf16(backend="cute-dsl")`` reaches this kernel through two policy
decisions that a block drafter should not inherit:

* the tactic comes from ``default_tactic``, a generic occupancy heuristic;
* ``_MAX_M`` refuses M above 32.

Both cost real time here. The drafter's projections are cold-weight and
grid-starved -- ncu puts them at 0.18-0.74 waves per SM, and DRAM throughput
tracks that ratio rather than the kernel's arithmetic -- so the tactic that
wins is the one that fills the machine, not the one the heuristic picks. And
the drafter's M is its batch times its block width, which reaches 64.

Public M rides the kernel's MMA-N axis, so M above the cutover is tiled, not
truncated; ``test_flashinfer_splitk.py`` checks that directly. Raising the
cutover is therefore a change of policy, not of contract, and this module only
does it after confirming the vendor's constants are exactly the ones that
behaviour was measured against.
"""

from __future__ import annotations

import functools
import threading

import torch

__all__ = ["MAX_M", "is_available", "splitk_mm", "supports"]

#: Largest M this adapter serves. The vendor cutover is 32; the kernel itself
#: tiles public M and stays exact to here.
MAX_M = 64

#: What the vendor module must look like for the measured tactics to mean
#: anything. A wheel that differs turns the adapter off instead of guessing.
_EXPECTED = {
    "_MAX_M": 32,
    "_SUPPORTED_MMA_M": (64, 128),
    "_SUPPORTED_MMA_N": (8, 16, 32),
    "_SUPPORTED_SPLIT_K": (1, 2, 3, 4),
    "_CTA_K": 128,
}
_TACTIC_FIELDS = ("mma_m", "mma_n", "split_k", "ab_stages")

_widen_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def _module():
    """The vendor kernel module, or None when it is not the one measured."""
    try:
        import dataclasses

        from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as mod
    except ImportError:
        return None
    for name, value in _EXPECTED.items():
        if getattr(mod, name, None) != value:
            return None
    tactic = getattr(mod, "SplitKTactic", None)
    if tactic is None or not dataclasses.is_dataclass(tactic):
        return None
    if tuple(f.name for f in dataclasses.fields(tactic)) != _TACTIC_FIELDS:
        return None
    if not all(callable(getattr(mod, f, None)) for f in ("run_splitk_dense",)):
        return None
    with _widen_lock:
        # One permanent widening, after the guard above pinned every constant
        # the measurement was taken against. Scoped to this process; the only
        # other reader is mm_bf16's own cute-dsl path, which this makes serve
        # M <= 64 with its default tactic -- measured faster than the cuBLAS
        # it would otherwise fall back to.
        mod._MAX_M = MAX_M
    return mod


def is_available() -> bool:
    """Whether the vendor kernel is present and matches the measured API."""
    return _module() is not None


def supports(m: int, n: int, k: int, tactic: tuple[int, int, int, int]) -> bool:
    """Whether ``tactic`` can serve ``(m, n, k)`` on this device."""
    mod = _module()
    if mod is None or not 1 <= m <= MAX_M:
        return False
    try:
        mod.validate_tactic(mod.SplitKTactic(*tactic), m, n, k)
    except (ValueError, AttributeError):
        return False
    return True


def splitk_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    tactic: tuple[int, int, int, int],
    out: torch.Tensor | None = None,
    *,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """``x @ weight.T`` through the vendor split-K kernel with ``tactic``.

    Args:
        x: ``[M, K]`` contiguous BF16 activation.
        weight: ``[N, K]`` contiguous BF16 weight; its transpose is the
            ``(K, N)`` operand the kernel wants, with no copy.
        tactic: ``(mma_m, mma_n, split_k, ab_stages)``, as measured for this
            exact ``(M, N, K)``.
        out: optional ``[M, N]`` BF16 destination; allocated when omitted.
        enable_pdl: launch with programmatic dependent launch.

    Returns:
        ``[M, N]`` BF16 tensor, ``out`` when it was given.

    Raises:
        RuntimeError: the vendor kernel is absent or not the measured build.
    """
    mod = _module()
    if mod is None:
        raise RuntimeError("flashinfer split-K BF16 GEMM is not available here")
    if out is None:
        out = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    # CuTe DSL reads these through DLPack, which rejects autograd views.
    mod.run_splitk_dense(
        x.detach(),
        weight.detach().t(),
        None,
        out,
        enable_pdl,
        mod.SplitKTactic(*tactic),
    )
    return out
