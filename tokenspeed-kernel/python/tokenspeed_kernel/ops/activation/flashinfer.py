# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""FlashInfer activation kernels."""

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

gelu_and_mul = error_fn
gelu_tanh_and_mul = error_fn
silu_and_mul = error_fn

if current_platform().is_nvidia:
    try:
        from flashinfer import (
            gelu_and_mul,
            gelu_tanh_and_mul,
            silu_and_mul,
        )
    except ImportError:
        pass

__all__ = ["gelu_and_mul", "gelu_tanh_and_mul", "silu_and_mul"]

_reviewed_silu_implementation = silu_and_mul


def record_silu_profile_engine(metadata, *, implementation):
    """Record the actual imported binding; certify artifacts after collection."""
    import sys

    module = sys.modules.get("flashinfer")
    metadata["activation_import_matches_binding"] = (
        implementation is _reviewed_silu_implementation
    )
    metadata["activation_package_version"] = getattr(module, "__version__", None)
    metadata["activation_source_branch"] = "flashinfer.silu_and_mul:unclamped"
