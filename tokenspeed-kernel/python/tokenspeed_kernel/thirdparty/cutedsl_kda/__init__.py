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

"""Single import point for the CuteDSL KDA AOT prefill kernel.

The kernel, its wrapper, and the AOT ``.so`` payload ship as the standalone
``tokenspeed-cutedsl-kda`` distribution (import package ``tokenspeed_cutedsl_kda``,
a CUDA-only dependency declared in ``requirements/cuda-thirdparty.txt``).
All ``tokenspeed_cutedsl_kda`` imports funnel through this module so the
dependency remains optional, mirroring the sibling
``thirdparty/flash_kda.py`` loader. Lazy imports keep the attention registry
usable when the package is absent, notably on AMD/ROCm.
``is_cutedsl_kda_installed()`` returns False there, so explicit selection fails
at startup with an actionable error rather than an import crash.

The package ships as the ``tokenspeed-cutedsl-kda`` wheel:

    pip install tokenspeed-cutedsl-kda

Requirements: sm_100a (B200) / sm_103a (B300), CUDA 13.
"""

from __future__ import annotations

import math
from functools import lru_cache
from importlib.util import find_spec

DEFAULT_SCALE: float = 1.0 / math.sqrt(128)

_INSTALL_HINT = (
    "CuteDSL KDA is not installed. Install it with "
    "`pip install tokenspeed-cutedsl-kda` (sm_100a / sm_103a, CUDA 13)."
)


@lru_cache(maxsize=1)
def is_cutedsl_kda_installed() -> bool:
    """Whether the ``tokenspeed_cutedsl_kda`` package loads on this device."""
    if find_spec("tokenspeed_cutedsl_kda") is None:
        return False
    try:
        import tokenspeed_cutedsl_kda
    except Exception:
        # Present but unloadable (missing runtime wheels, wrong arch, ...).
        return False
    return bool(tokenspeed_cutedsl_kda.is_cutedsl_kda_installed())


def _module():
    """Return the ``tokenspeed_cutedsl_kda`` module or raise an install hint."""
    if not is_cutedsl_kda_installed():
        raise ImportError(_INSTALL_HINT)
    import tokenspeed_cutedsl_kda

    return tokenspeed_cutedsl_kda


def cutedsl_kda_check_config(gate_lower_bound: float) -> None:
    """Validate the model gate bound against the compiled kernel."""
    _module().cutedsl_kda_check_config(gate_lower_bound)


@lru_cache(maxsize=1)
def cutedsl_kda_supports_host_hint() -> bool:
    """Whether the installed package accepts ``cu_seqlens_cpu`` host hints.

    Older package builds read ``cu_seqlens`` back to the host (one D2H plus a
    stream sync per new boundary tensor); newer builds accept the
    already-known host copy and stay sync-free. Callers probe this once and
    only then thread the hint through, so either package version keeps
    working. The probe inspects the explicit-signature size query because
    ``cutedsl_kda_forward`` is a passthrough in every package version and
    would always match.
    """
    import inspect

    try:
        sig = inspect.signature(_module().cutedsl_kda_workspace_size)
    except (ImportError, ValueError, TypeError):
        return False
    return "cu_seqlens_cpu" in sig.parameters


def cutedsl_kda_workspace_size(cu_seqlens, heads: int, cu_seqlens_cpu=None) -> int:
    """Decomposition-route workspace bytes for this shape (0 on the engine route).

    ``cu_seqlens_cpu`` optionally carries a CPU copy of the boundary contents
    so the wrapper can plan without a stream-synchronizing D2H read; it is
    forwarded only when set, so hint-unaware package builds keep working.
    """
    if cu_seqlens_cpu is not None:
        return _module().cutedsl_kda_workspace_size(
            cu_seqlens, heads, cu_seqlens_cpu=cu_seqlens_cpu
        )
    return _module().cutedsl_kda_workspace_size(cu_seqlens, heads)


def cutedsl_kda_forward(*args, **kwargs):
    """Run the token-major KDA forward and return ``(out, new_state)``."""
    return _module().cutedsl_kda_forward(*args, **kwargs)
