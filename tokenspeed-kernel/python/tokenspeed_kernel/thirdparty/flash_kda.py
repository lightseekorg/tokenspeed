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

"""Single import point for the optional third-party ``flash-kda`` package.

FlashKDA (MoonshotAI) is a two-kernel CUTLASS implementation of chunked KDA
prefill. It applies the safe gate, beta sigmoid, and QK L2 normalization
in-kernel. All ``flash_kda`` imports funnel through this module, and lazy imports
keep the optional dependency isolated from unrelated tokenspeed-kernel
operations.

The package ships as the ``tokenspeed-flashkda`` wheel (declared in
``requirements/cuda-thirdparty.txt``):

    pip install tokenspeed-flashkda

Requirements: SM90+, CUDA 12.9+, PyTorch 2.4+ (K = V = 128 only).
"""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec

_INSTALL_HINT = (
    "FlashKDA is not installed. Install it with "
    "`pip install tokenspeed-flashkda` (SM90+, CUDA 12.9+)."
)


@lru_cache(maxsize=1)
def is_flash_kda_installed() -> bool:
    """Whether the ``flash_kda`` package (and its extension) is importable."""
    if find_spec("flash_kda") is None:
        return False
    try:
        import flash_kda  # noqa: F401
    except Exception:
        # Present but unloadable (e.g. extension built for another
        # torch/CUDA ABI). Treat as unavailable rather than failing import.
        return False
    return True


def flash_kda_fwd():
    """Return ``flash_kda.fwd`` or raise with an install hint.

    Returns:
        The ``fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=None, cu_seqlens=None)`` entry point.
        It writes ``out``/``final_state`` in place and applies sigmoid(beta),
        the safe gate, and QK L2 normalization internally.
    """
    if not is_flash_kda_installed():
        raise ImportError(_INSTALL_HINT)
    import flash_kda

    return flash_kda.fwd
