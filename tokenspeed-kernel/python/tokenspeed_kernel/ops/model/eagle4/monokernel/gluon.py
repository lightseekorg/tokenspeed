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

"""Shims that register the AMD Gluon EAGLE-4 draft monokernel."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.eagle4 import (
        layer as _eagle4_layer,
    )
except ImportError:
    _eagle4_layer = None


if _eagle4_layer is not None:
    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    # One draft row per request; the projections' GEMV covers 16 rows.
    _TRAITS = {"rows": frozenset(range(1, _eagle4_layer.MAX_ROWS + 1))}

    @register_kernel(
        "monokernel",
        "eagle4_draft_layer_prepare",
        name="gluon_eagle4_draft_layer_prepare_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_eagle4_draft_layer_prepare_gfx950(
        *, tensors: dict[str, torch.Tensor], max_rows: int
    ) -> tuple[Any, int]:
        weights = _eagle4_layer.pack_eagle4_draft_layer(tensors)
        # A plan's scratch grows with its rows: the largest serves them all.
        return weights, _eagle4_layer.layer_plan(max_rows).scratch_pairs

    @register_kernel(
        "monokernel",
        "eagle4_draft_layer_decode",
        name="gluon_eagle4_draft_layer_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_eagle4_draft_layer_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        layer_index: int,
        compile_only: bool,
        **operands: torch.Tensor,
    ) -> None:
        _eagle4_layer.launch_gluon_eagle4_draft_layer_gfx950(
            weights,
            scratch=scratch,
            step=step,
            layer=layer_index,
            trace_buf=None,
            compile_only=compile_only,
            **operands,
        )
