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

"""Single import boundary for the Triton-Ascend distribution."""

from types import SimpleNamespace

import triton
from triton import language as tl
from triton.language.extra import libdevice

try:
    from triton import profiler as proton
except ImportError:
    # Triton-Ascend's offline bundle may omit optional Proton profiling.
    # TokenSpeed's profiling boundary treats None as an unavailable profiler.
    proton = None


# The Ascend Triton release lacks the newer alignment-only specialization hint.
# Disabling all specialization for those arguments is conservative: it preserves
# the requested alignment behavior at the cost of fewer compiled variants.
_ascend_jit = triton.jit


def _jit_with_ascend_compatibility(fn=None, **kwargs):
    alignment = kwargs.pop("do_not_specialize_on_alignment", ())
    if alignment:
        unspecialized = kwargs.get("do_not_specialize") or ()
        kwargs["do_not_specialize"] = tuple(dict.fromkeys((*unspecialized, *alignment)))
    return _ascend_jit(fn, **kwargs)


triton.jit = _jit_with_ascend_compatibility


@triton.jit
def _unsupported_pdl_noop():
    """Keep disabled CUDA PDL branches parseable on Triton-Ascend."""
    pass


if not hasattr(tl.extra, "cuda"):
    tl.extra.cuda = SimpleNamespace()
if not hasattr(tl.extra.cuda, "gdc_wait"):
    tl.extra.cuda.gdc_wait = _unsupported_pdl_noop
if not hasattr(tl.extra.cuda, "gdc_launch_dependents"):
    tl.extra.cuda.gdc_launch_dependents = _unsupported_pdl_noop


__all__ = ["libdevice", "proton", "tl", "triton"]
