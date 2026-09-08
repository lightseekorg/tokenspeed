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

``mm_bf16(backend="cute-dsl")`` reaches the same kernel under two policies a
block drafter should not inherit: the tactic comes from ``default_tactic``, a
generic occupancy heuristic, and ``_MAX_M`` refuses M above 32. These
projections are cold-weight and grid-starved, so the tactic they need is the
one that fills the machine, and a block drafter's M is its batch times its
block width, which reaches 64. Public M rides the kernel's MMA-N axis, so M
past the cutover is tiled and not truncated: a policy bound, not a correctness
one.

``test/gemm_tuning/tune_splitk_tactic.py`` measures the tactics.
``tokenspeed-kernel/test/ops/gemm/test_routed_gemv.py`` checks both the
exactness past the cutover and that the shared vendor module stays untouched.
"""

from __future__ import annotations

import functools

import torch

__all__ = ["MAX_M", "is_available", "splitk_mm", "supports"]

_VENDOR_MODULE = "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk"
#: Where this adapter's private instance is registered. Never the vendor's own
#: key, so ``import flashinfer...`` keeps returning the vendor's module.
_PRIVATE_MODULE = f"{__name__}._vendor_splitk"

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


@functools.lru_cache(maxsize=1)
def _module():
    """A private instance of the vendor kernel module, or None.

    Executed into its own namespace rather than imported, so raising the M
    cutover below cannot be observed through the shared module.
    """
    import dataclasses
    import importlib.util
    import sys
    import types

    try:
        spec = importlib.util.find_spec(_VENDOR_MODULE)
        source = spec.loader.get_source(_VENDOR_MODULE) if spec else None
    except (ImportError, ValueError, AttributeError, OSError):
        return None
    if source is None:
        return None
    # Compiled into a module of our own rather than loaded through the vendor's
    # spec: the loader refuses to execute a spec under a different name, and
    # @dataclass resolves ``sys.modules[cls.__module__]`` while the source
    # runs, so the instance has to be registered under the name it carries.
    # That name is ours, so the vendor's own sys.modules entry is never
    # written and ``import flashinfer...`` still yields the vendor's module.
    mod = types.ModuleType(_PRIVATE_MODULE)
    mod.__file__ = spec.origin
    mod.__package__ = _VENDOR_MODULE.rpartition(".")[0]
    sys.modules[_PRIVATE_MODULE] = mod
    try:
        exec(compile(source, spec.origin, "exec"), mod.__dict__)  # noqa: S102
    except Exception:  # noqa: BLE001  (any import-time failure disables us)
        sys.modules.pop(_PRIVATE_MODULE, None)
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
    # Every constant the tactics were measured against is now pinned, so this
    # relaxes a policy bound and never a correctness one. It lands on this
    # module's own copy of the vendor namespace.
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
