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

"""Registration shims for AMD Gluon attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_decode_fp16_gfx950 import (
        gluon_mha_decode_fp16_gfx950 as _decode_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_prefill_fp16_gfx950 import (
        gluon_mha_prefill_fp16_gfx950 as _prefill_impl,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc
    _decode_impl = None
    _prefill_impl = None
else:
    _IMPORT_ERROR = None


if _decode_impl is not None:

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="gluon_mha_decode_fp16_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64}),
            "page_size": frozenset({64}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_mha_decode_fp16_gfx950(*args, **kwargs):
        return _decode_impl(*args, **kwargs)

else:

    def gluon_mha_decode_fp16_gfx950(*args, **kwargs):
        raise ImportError(
            "gluon_mha_decode_fp16_gfx950 requires tokenspeed-kernel-amd"
        ) from _IMPORT_ERROR


if _prefill_impl is not None:

    @register_kernel(
        "attention",
        "mha_prefill",
        name="gluon_mha_prefill_fp16_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_prefill_fp16_gfx950(*args, **kwargs):
        return _prefill_impl(*args, **kwargs)

else:

    def gluon_mha_prefill_fp16_gfx950(*args, **kwargs):
        raise ImportError(
            "gluon_mha_prefill_fp16_gfx950 requires tokenspeed-kernel-amd"
        ) from _IMPORT_ERROR


__all__ = [
    "gluon_mha_decode_fp16_gfx950",
    "gluon_mha_prefill_fp16_gfx950",
]
