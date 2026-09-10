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

"""Registration shims for AMD Gluon relative MHA kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.decode import (
        gluon_rel_mha_decode_gfx950 as _rel_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.extend import (
        gluon_rel_mha_extend_gfx950 as _rel_extend_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.prefill import (
        gluon_rel_mha_prefill_gfx950 as _rel_prefill_impl,
    )

    @register_kernel(
        "attention",
        "rel_mha_prefill",
        name="gluon_rel_mha_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_rel_mha_prefill_gfx950(*args, **kwargs):
        kwargs.pop("enable_pdl", None)
        tau = kwargs.pop("tau", None)
        if tau is not None:
            # No fused per-row logit scale in the gluon backend; fold tau
            # into q and the rel bias: tau*(scale*qk + rel).
            kwargs["q"] = kwargs["q"] * tau[:, None, None].to(kwargs["q"].dtype)
            kwargs["rel_logits"] = kwargs["rel_logits"] * tau[:, None, None].to(
                kwargs["rel_logits"].dtype
            )
        return _rel_prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_extend_with_kvcache",
        name="gluon_rel_mha_extend_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_rel_mha_extend_gfx950(*args, **kwargs):
        kwargs.pop("enable_pdl", None)
        tau = kwargs.pop("tau", None)
        if tau is not None:
            # No fused per-row logit scale in the gluon backend; fold tau
            # into q and the rel bias: tau*(scale*qk + rel).
            kwargs["q"] = kwargs["q"] * tau[:, None, None].to(kwargs["q"].dtype)
            kwargs["rel_logits"] = kwargs["rel_logits"] * tau[:, None, None].to(
                kwargs["rel_logits"].dtype
            )
        return _rel_extend_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "rel_mha_decode_with_kvcache",
        name="gluon_rel_mha_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64, 128, 256}),
            "sliding_window": frozenset({False, True}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_rel_mha_decode_gfx950(*args, **kwargs):
        kwargs.pop("enable_pdl", None)
        tau = kwargs.pop("tau", None)
        if tau is not None:
            # No fused per-row logit scale in the gluon backend; fold tau
            # into q and the rel bias: tau*(scale*qk + rel).
            kwargs["q"] = kwargs["q"] * tau[:, None, None].to(kwargs["q"].dtype)
            kwargs["rel_logits"] = kwargs["rel_logits"] * tau[:, None, None].to(
                kwargs["rel_logits"].dtype
            )
        return _rel_decode_impl(*args, **kwargs)
