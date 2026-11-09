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

"""Registers the CUDA build of the EAGLE-Harness ops (NVIDIA), including the
device-chain ops."""

from __future__ import annotations

from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature

if current_platform().is_nvidia:
    import tokenspeed_kernel.thirdparty.cuda.eagle_harness as _tree
    import tokenspeed_kernel.thirdparty.cuda.eagle_harness_chain as _chain
    import tokenspeed_kernel.thirdparty.cuda.eagle_harness_policy as _policy

    # The wrappers load their library on first call, so registering needs no build.
    for _module in (_tree, _policy, _chain):
        for _op in _module.__all__:
            register_kernel(
                "sampling",
                _op,
                name=f"cuda_{_op}",
                solution="cuda",
                capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
                signatures=frozenset({format_signature()}),
                priority=Priority.PERFORMANT,
            )(getattr(_module, _op))
