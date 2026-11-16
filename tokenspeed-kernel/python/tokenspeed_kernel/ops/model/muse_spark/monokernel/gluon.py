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

"""Shims that register the AMD Gluon Muse Spark monokernels."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
        _extract_gluon_raw_s,
        _extract_gluon_raw_w,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel import stuck as _stuck
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        layer as _muse_spark_layer,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        stuck_report as _muse_spark_stuck_report,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        weights as _muse_spark_weights,
    )
except ImportError:
    _muse_spark_layer = None

try:
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        draft_context as _muse_spark_draft_context,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        draft_head as _muse_spark_draft_head,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (
        draft_layer as _muse_spark_draft_layer,
    )
except ImportError:
    _muse_spark_draft_context = None
    _muse_spark_draft_head = None
    _muse_spark_draft_layer = None


if _muse_spark_layer is not None:
    _MUSE_GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    # The rows of a launch: one request's decode or verify (up to 8 rows), or
    # two requests' (up to 16), a compile-time row count.
    _MUSE_TRAITS = {"rows": frozenset(range(1, _muse_spark_layer.MAX_ROWS + 1))}

    @register_kernel(
        "monokernel",
        "muse_spark_layer_prepare",
        name="gluon_muse_spark_layer_prepare_gfx950",
        solution="gluon",
        capability=_MUSE_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_MUSE_TRAITS,
    )
    def gluon_muse_spark_layer_prepare_gfx950(
        *, tensors: dict[str, Any], max_rows: int
    ) -> tuple[Any, int, None]:
        # The routed experts as the regular gluon MXFP4 MoE prepared them: the
        # kernel reads their raw storage in place, so it serves only the MoE
        # kernels whose weight preprocessor is the gfx950 gluon MXFP4 one.
        # Imported here: that module defines it only on AMD platforms.
        from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp4_gfx950_moe_weights

        tensors = dict(tensors)
        experts_kernel = tensors.pop("experts_kernel")
        spec = KernelRegistry.get().get_by_name(experts_kernel)
        preprocessor = None if spec is None else spec.weight_preprocessor
        if isinstance(preprocessor, tuple):
            preprocessor = preprocessor[1]
        if preprocessor is not gluon_mxfp4_gfx950_moe_weights:
            raise ValueError(
                f"experts: the monokernel reads the gfx950 gluon MXFP4 layout, "
                f"not {experts_kernel}'s"
            )
        w13, w13_scale, w2, w2_scale = tensors["experts"]
        experts = (
            _extract_gluon_raw_w(w13),
            _extract_gluon_raw_s(w13_scale),
            _extract_gluon_raw_w(w2),
            _extract_gluon_raw_s(w2_scale),
        )
        weights = _muse_spark_weights.pack_layer(**{**tensors, "experts": experts})
        # One scratch serves launches of every row count up to max_rows.
        scratch_pairs = max(
            # The scratch's size depends on the rows, not on how they chain,
            # on the warm stage or on qnext (every plan lays out the same
            # regions).
            _muse_spark_layer.layer_plan(
                full=weights.full,
                rows=rows,
                chain=rows,
                warm_next=True,
                own_qkvg=True,
                next_qkvg=False,
                # Off in serving (see the launch).
                check_parity=False,
            ).scratch_pairs
            for rows in range(1, max_rows + 1)
        )
        return weights, scratch_pairs, None

    @register_kernel(
        "monokernel",
        "muse_spark_layer_decode",
        name="gluon_muse_spark_layer_gfx950",
        solution="gluon",
        capability=_MUSE_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_MUSE_TRAITS,
    )
    def gluon_muse_spark_layer_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        peer_bases: torch.Tensor,
        layer_index: int,
        chain: int,
        whole_rows: bool,
        tap: torch.Tensor | None,
        tap_slot: int | None,
        warm_next: Any,
        proj_in: torch.Tensor | None,
        proj_out: torch.Tensor | None,
        stuck_rec: torch.Tensor | None,
        compile_only: bool,
        **operands: torch.Tensor,
    ) -> None:
        _muse_spark_layer.launch_gluon_muse_spark_layer_gfx950(
            weights,
            scratch=scratch,
            step=step,
            peer_bases=peer_bases,
            layer=layer_index,
            chain=chain,
            whole_rows=whole_rows,
            tap=tap,
            tap_slot=tap_slot,
            warm_next=warm_next,
            proj_in=proj_in,
            proj_out=proj_out,
            trace_buf=None,
            stuck_rec=stuck_rec,
            # Serving's launches alternate parity by construction: every
            # layer 1-60 launches each decode step, and layer 60 is followed
            # by the next step's layer 1. The check would cost CTA 0 a load,
            # a barrier and a store before each launch's first put.
            check_parity=False,
            compile_only=compile_only,
            **operands,
        )

    class _MuseSparkStuckRecord:
        """The gfx950 stuck-poll record buffer of Muse Spark layer launches
        (``muse_spark_stuck_record``)."""

        def __init__(
            self,
            grid: int,
            waves: int,
            path: str | None,
            scratch: torch.Tensor,
            rank: int,
        ) -> None:
            self._buf = _stuck.stuck_buffer(grid, waves, path, scratch.data_ptr(), rank)
            self.path = path

        @property
        def tensor(self) -> torch.Tensor:
            return self._buf.tensor

        def any(self) -> bool:
            return bool(_stuck.records(self._buf.tensor))

        def close(self, remove: bool) -> None:
            _stuck.close(self._buf, remove)

        def layer_kinds(self, layers: dict) -> dict[int, dict]:
            return {i: {"full": bool(f.weights.full)} for i, f in layers.items()}

        def describe(self, layers: dict, launches: list) -> list[str]:
            # Serving's launches do not check the peer parity (see the launch).
            return _muse_spark_stuck_report.describe(
                self._buf.tensor, self.layer_kinds(layers), launches, False, None
            )

    @register_kernel(
        "monokernel",
        "muse_spark_stuck_record",
        name="gluon_muse_spark_stuck_record_gfx950",
        solution="gluon",
        capability=_MUSE_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_MUSE_TRAITS,
    )
    def gluon_muse_spark_stuck_record_gfx950(
        *, grid: int, waves: int, path: str | None, scratch: torch.Tensor, rank: int
    ) -> _MuseSparkStuckRecord:
        return _MuseSparkStuckRecord(grid, waves, path, scratch, rank)


if _muse_spark_draft_layer is not None:
    _DRAFT_GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    # The blocks of one or two requests (8 rows each) per launch.
    _DRAFT_TRAITS = {
        "rows": frozenset(
            range(
                _muse_spark_draft_layer.BLOCK_ROWS,
                _muse_spark_draft_layer.MAX_ROWS + 1,
                _muse_spark_draft_layer.BLOCK_ROWS,
            )
        )
    }

    @register_kernel(
        "monokernel",
        "muse_spark_draft_layer_prepare",
        name="gluon_muse_spark_draft_layer_prepare_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_DRAFT_TRAITS,
    )
    def gluon_muse_spark_draft_layer_prepare_gfx950(
        *, tensors: dict[str, torch.Tensor], rank: int, max_rows: int
    ) -> tuple[Any, int, None]:
        weights = _muse_spark_draft_layer.pack_draft_layer(tensors, rank)
        # The last layer's plan (with the final norm) is the larger.
        plan = _muse_spark_draft_layer.layer_plan(max_rows, True)
        return weights, plan.scratch_pairs, None

    @register_kernel(
        "monokernel",
        "muse_spark_draft_layer_decode",
        name="gluon_muse_spark_draft_layer_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_DRAFT_TRAITS,
    )
    def gluon_muse_spark_draft_layer_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        peer_bases: torch.Tensor,
        layer_index: int,
        compile_only: bool,
        **operands: torch.Tensor,
    ) -> None:
        _muse_spark_draft_layer.launch_gluon_muse_spark_draft_layer_gfx950(
            weights,
            scratch=scratch,
            step=step,
            peer_bases=peer_bases,
            layer=layer_index,
            trace_buf=None,
            compile_only=compile_only,
            **operands,
        )

    # The draft steps (each block's rows 1 .. 7) of one or two requests per
    # launch.
    _HEAD_TRAITS = {
        "rows": frozenset(
            _muse_spark_draft_head.STEPS * n
            for n in range(1, _muse_spark_draft_head.MAX_REQUESTS + 1)
        )
    }

    @register_kernel(
        "monokernel",
        "muse_spark_draft_head_prepare",
        name="gluon_muse_spark_draft_head_prepare_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_HEAD_TRAITS,
    )
    def gluon_muse_spark_draft_head_prepare_gfx950(
        *,
        tensors: dict[str, torch.Tensor],
        rank: int,
        max_rows: int,
        scale: float,
        softcap: float,
    ) -> tuple[Any, int, None]:
        weights = _muse_spark_draft_head.pack_draft_head(tensors, rank, scale, softcap)
        return weights, _muse_spark_draft_head.head_plan(max_rows).scratch_pairs, None

    @register_kernel(
        "monokernel",
        "muse_spark_draft_head_scratch",
        name="gluon_muse_spark_draft_head_scratch_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_HEAD_TRAITS,
    )
    def gluon_muse_spark_draft_head_scratch_gfx950(*, max_rows: int) -> int:
        return _muse_spark_draft_head.head_plan(max_rows).scratch_pairs

    @register_kernel(
        "monokernel",
        "muse_spark_draft_head_decode",
        name="gluon_muse_spark_draft_head_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_HEAD_TRAITS,
    )
    def gluon_muse_spark_draft_head_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        peer_bases: torch.Tensor,
        compile_only: bool,
        **operands: torch.Tensor,
    ) -> None:
        _muse_spark_draft_head.launch_gluon_muse_spark_draft_head_gfx950(
            weights,
            scratch=scratch,
            step=step,
            peer_bases=peer_bases,
            trace_buf=None,
            compile_only=compile_only,
            **operands,
        )

    # The verify rows of one or two requests per launch.
    _CONTEXT_TRAITS = {
        "rows": frozenset(
            _muse_spark_draft_context.ROWS * n
            for n in range(1, _muse_spark_draft_context.MAX_REQUESTS + 1)
        )
    }

    @register_kernel(
        "monokernel",
        "muse_spark_draft_context_prepare",
        name="gluon_muse_spark_draft_context_prepare_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_CONTEXT_TRAITS,
    )
    def gluon_muse_spark_draft_context_prepare_gfx950(
        *, tensors: dict[str, torch.Tensor], rank: int, max_rows: int
    ) -> tuple[Any, int, None]:
        weights = _muse_spark_draft_context.pack_draft_context(tensors, rank)
        plan = _muse_spark_draft_context.context_plan(max_rows)
        return weights, plan.scratch_pairs, None

    @register_kernel(
        "monokernel",
        "muse_spark_draft_context_decode",
        name="gluon_muse_spark_draft_context_gfx950",
        solution="gluon",
        capability=_DRAFT_GFX950,
        signatures=format_signatures("h", "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_CONTEXT_TRAITS,
    )
    def gluon_muse_spark_draft_context_gfx950(
        *,
        weights: Any,
        scratch: torch.Tensor,
        step: torch.Tensor,
        peer_bases: torch.Tensor,
        compile_only: bool,
        **operands: Any,
    ) -> None:
        _muse_spark_draft_context.launch_gluon_muse_spark_draft_context_gfx950(
            weights,
            scratch=scratch,
            step=step,
            peer_bases=peer_bases,
            trace_buf=None,
            compile_only=compile_only,
            **operands,
        )
