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

"""LongCat verify-window eligibility for the shared DSA implementations."""

import sys
from dataclasses import replace
from typing import Any

from tokenspeed_kernel.registry import KernelRegistry, KernelSpec

_PROVIDERS = {
    "deep_gemm_dsa_decode_topk": ("deep_gemm", "dsa_decode_topk"),
    "flashinfer_trtllm_dsa_decode": ("flashinfer", "dsa_decode"),
    "triton_dsa_decode": ("triton", "dsa_decode"),
}


def longcat_dsa_candidates(spec: KernelSpec, traits: dict[str, Any]) -> KernelSpec:
    """Adapt a registered ``spec`` to LongCat's request ``traits``.

    Return a spec allowing 7/8-token windows for verified builtins, or the
    original spec when no extension applies.
    """
    provider = _PROVIDERS.get(spec.name)
    if traits.get("q_len") not in (7, 8) or provider is None:
        return spec
    module, mode = provider
    if (
        spec.family != "attention"
        or spec.mode != mode
        or spec.traits.get("q_len") != frozenset(range(1, 7))
    ):
        return spec
    builtin = getattr(
        sys.modules.get(f"tokenspeed_kernel.ops.attention.dsa.{module}"),
        spec.name,
        None,
    )
    if builtin is None or KernelRegistry.get().get_impl(spec.name) is not builtin:
        return spec
    return replace(spec, traits={**spec.traits, "q_len": frozenset(range(1, 9))})
