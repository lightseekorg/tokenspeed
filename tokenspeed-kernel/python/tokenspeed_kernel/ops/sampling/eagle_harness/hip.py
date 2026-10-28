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

"""Registers the HIP build of the EAGLE-Harness ops (AMD gfx950). There are no
device-chain ops: HIP has no device-side graph launch, so the host launches
each draft depth once its go flag lands."""

from __future__ import annotations

from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature

if current_platform().is_amd:
    import tokenspeed_kernel.thirdparty.hip.eagle_harness as _tree
    import tokenspeed_kernel.thirdparty.hip.eagle_harness_policy as _policy

    # The wrappers load their library on first call, so registering needs no build.
    for _module in (_tree, _policy):
        for _op in _module.__all__:
            register_kernel(
                "sampling",
                _op,
                name=f"hip_{_op}",
                solution="hip",
                capability=CapabilityRequirement(
                    vendors=frozenset({"amd"}),
                    min_arch_version=ArchVersion(9, 5),
                    max_arch_version=ArchVersion(9, 5),
                ),
                signatures=frozenset({format_signature()}),
                priority=Priority.PERFORMANT,
            )(getattr(_module, _op))
