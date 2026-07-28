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

"""Single import point for the optional third-party flash-linear-attention (``fla``).

Kimi-K3's KDA (Kimi Delta Attention) uses ``fla``'s per-channel
``chunk_kda`` and ``fused_recurrent_kda`` implementations. All ``fla`` imports
funnel through this module so the dependency stays optional and isolated
(mirroring ``_triton.py`` for tokenspeed-triton). Imports remain lazy so
importing tokenspeed-kernel never requires ``fla``.
"""

from __future__ import annotations

_INSTALL_HINT = (
    "Kimi-K3 KDA requires flash-linear-attention for this backend. Install it "
    "into the active environment: `pip install flash-linear-attention`."
)

# Stock-triton / tokenspeed-triton coexistence shim.
#
# fla compiles its kernels against the stock ``triton`` package (a torch dep),
# whose JIT only accepts host helpers wrapped as ``ConstexprFunction`` inside a
# kernel body. Importing tokenspeed-triton (pulled in by the runtime's triton
# ops) un-wraps stock ``triton.next_power_of_2`` back to a plain function, so
# fla's kernels then fail to compile ("Unsupported function referenced:
# next_power_of_2"). Re-wrap the affected constexpr builtins before each fla
# call. Isolated to the stock ``triton`` module (fla's), so it does not affect
# tokenspeed-triton kernels.
_CONSTEXPR_BUILTINS = ("next_power_of_2",)


def _ensure_triton_constexpr() -> None:
    try:
        import triton
        from triton.runtime.jit import ConstexprFunction
    except Exception:  # pragma: no cover - triton always present with fla
        return
    for name in _CONSTEXPR_BUILTINS:
        fn = getattr(triton, name, None)
        if fn is not None and not isinstance(fn, ConstexprFunction):
            setattr(triton, name, ConstexprFunction(fn))


def chunk_kda(*args, **kwargs):
    """``fla.ops.kda.chunk_kda`` (chunked prefill gated-delta-rule scan)."""
    try:
        from fla.ops.kda import chunk_kda as _chunk_kda
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(_INSTALL_HINT) from exc
    _ensure_triton_constexpr()
    return _chunk_kda(*args, **kwargs)


def fused_recurrent_kda(*args, **kwargs):
    """``fla.ops.kda.fused_recurrent.fused_recurrent_kda`` (single-step decode)."""
    try:
        from fla.ops.kda.fused_recurrent import fused_recurrent_kda as _fused
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(_INSTALL_HINT) from exc
    _ensure_triton_constexpr()
    return _fused(*args, **kwargs)
