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

"""gfx950 Gluon attention prologue registration."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.prologue.types import (
    BOOLS,
    FP8TensorScales,
    GQAPrologueOutput,
    HeadKVCache,
    HeadNorm,
    Rotary,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.prologue import (
        launch_gluon_gqa_prologue_gfx950,
    )

    # The gfx950 GQA prologue: the Triton prologue's bytes from Gluon device
    # functions a fused decoder layer also runs, so a layer's prologue and the
    # standalone one share one body (attention-prologue.md).
    @register_kernel(
        "attention",
        "gqa_prologue",
        name="gluon_gqa_prologue_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(("q",), "dense", {torch.float16, torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "has_norm": frozenset({True}),
            "kv_format": frozenset({"native", "fp8", "fp8_per_tensor"}),
            "kv_convert": BOOLS,
            "mrope": frozenset({False}),
            "partial_rotary": frozenset({False}),
            "return_kv": frozenset({False}),
            "rope_style": frozenset({"none", "gptj"}),
        },
    )
    def gluon_gqa_prologue_gfx950(
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        norm: HeadNorm | None,
        rotary: Rotary | None,
        cache: HeadKVCache,
        query_dtype: torch.dtype,
        return_kv: bool,
        enable_pdl: bool,
    ) -> GQAPrologueOutput:
        scales = cache.scales if isinstance(cache.scales, FP8TensorScales) else None
        query = launch_gluon_gqa_prologue_gfx950(
            q, k, v, norm.q_weight, norm.k_weight, norm.eps, norm.weight_offset,
            None if rotary is None else rotary.cos_sin_cache,
            None if rotary is None else rotary.positions,
            cache.k_cache, cache.v_cache, cache.slots,
            1.0 if scales is None else 1.0 / scales.k,
            1.0 if scales is None else 1.0 / scales.v,
            scales is not None, query_dtype,
        )  # fmt: skip
        return GQAPrologueOutput(q=query, k=None, v=None)
