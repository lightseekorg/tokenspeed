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

"""FlashInfer's low-M direct BF16 GEMM, driven by a measured tactic."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["MAX_M", "direct_mm", "is_available", "supports"]

_VENDOR_MODULE = "flashinfer.gemm.kernels.dense_bf16_gemm_direct"

#: Largest M the vendor kernel serves.
MAX_M = 32

#: What the vendor module must look like for the measured tactics to mean
#: anything. A wheel that differs turns the adapter off instead of guessing.
_EXPECTED = {
    "_MAX_M": MAX_M,
    "_VECTOR_WIDTH": 8,
    "_SUPPORTED_BLOCK_SIZES": (32, 64, 96, 128, 192, 256, 384),
    "_SUPPORTED_OUTPUTS_PER_BLOCK": (1, 2, 4),
}
_TACTIC_FIELDS = ("block_size", "outputs_per_block", "rows_per_block")


@functools.lru_cache(maxsize=1)
def _module():
    """The vendor kernel module when it matches the measured API, else None."""
    try:
        mod = importlib.import_module(_VENDOR_MODULE)
    except Exception:  # noqa: BLE001  (any import-time failure disables us)
        return None
    drift = [
        name for name, value in _EXPECTED.items() if getattr(mod, name, None) != value
    ]
    tactic = getattr(mod, "DirectTactic", None)
    if tactic is None or not dataclasses.is_dataclass(tactic):
        drift.append("DirectTactic")
    elif tuple(f.name for f in dataclasses.fields(tactic)) != _TACTIC_FIELDS:
        drift.append("DirectTactic fields")
    drift += [
        name
        for name in ("run_direct_dense", "validate_tactic")
        if not callable(getattr(mod, name, None))
    ]
    if drift:
        # A silent decline would hand the measured shapes back to the heuristic.
        logger.warning(
            "flashinfer's direct BF16 GEMM differs from the measured build (%s); "
            "its route entries are off",
            ", ".join(drift),
        )
        return None
    return mod


def is_available() -> bool:
    """Whether the vendor kernel is present and matches the measured API."""
    return _module() is not None


def supports(m: int, n: int, k: int, tactic: tuple[int, int, int]) -> bool:
    """Whether the vendor validates ``tactic`` for ``(m, n, k)``."""
    mod = _module()
    if mod is None:
        return False
    try:
        mod.validate_tactic(mod.DirectTactic(*tactic), m, n, k)
    except (ValueError, TypeError, AssertionError, AttributeError):
        return False
    return True


def direct_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    tactic: tuple[int, int, int],
    out: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor:
    """``x @ weight.T`` through the vendor direct kernel with ``tactic``.

    Args:
        x: ``[M, K]`` contiguous 32-byte-aligned BF16 activation.
        weight: ``[N, K]`` contiguous 32-byte-aligned BF16 weight.
        tactic: ``(block_size, outputs_per_block, rows_per_block)`` for this shape.
        out: ``[M, N]`` contiguous BF16 destination, or None to allocate one.
        enable_pdl: Launch with programmatic dependent launch.

    Returns:
        ``[M, N]`` BF16 tensor, ``out`` when it was given.
    """
    mod = _module()
    if mod is None:
        raise RuntimeError("flashinfer direct BF16 GEMM is not available here")
    if out is None:
        out = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    # CuTe DSL reads these through DLPack, which rejects autograd views.
    mod.run_direct_dense(
        x.detach(), weight.detach().t(), out, enable_pdl, mod.DirectTactic(*tactic)
    )
    return out
