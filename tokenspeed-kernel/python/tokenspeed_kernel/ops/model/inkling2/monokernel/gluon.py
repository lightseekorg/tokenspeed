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

"""Shims that register the AMD Gluon Inkling2 monokernel."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (
        dims as _inkling2_dims,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (
        layer as _inkling2_layer,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import (
        plan as _inkling2_plan,
    )
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.weights import bf16_tiles
except ImportError:
    _inkling2_layer = None


if _inkling2_layer is not None:
    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    _SIGNATURES = format_signatures("h", "dense", {torch.bfloat16})
    # Verify batches of 1, 2, 4 and 8 requests, a verify block of 4 or 8
    # rows each.
    _TRAITS = {
        "rows": frozenset({4, 8, 16, 32, 64}),
        "experts": frozenset({"nvfp4", "mxfp8"}),
    }

    def _dims(tensors: dict[str, Any], experts: str):
        """The layer's per-rank dimensions from its weights' shapes."""
        # NVFP4 packs two values a byte, MXFP8 one.
        per_byte = 2 if experts == "nvfp4" else 1
        return _inkling2_dims.Dims(
            hidden=tensors["qkvr"].shape[1],
            q_heads=tensors["head"][0].shape[0],
            kv_heads=tensors["head"][3].shape[0] // _inkling2_dims.HEAD_DIM,
            routed=tensors["router"].shape[0],
            inter=tensors["w2"][0].shape[2] * per_byte,
            world=4,
        )

    def _served(tensors: dict[str, Any], experts: str, full: bool) -> bool:
        """Whether the layer's weights have a served geometry's per-rank shapes."""
        ip = _inkling2_plan
        try:
            d = _dims(tensors, experts)
        except ValueError:
            return False
        if d not in ip.SERVED_DIMS:
            return False
        shapes = {
            "qkvr": (d.full_qkvr if full else d.qkvr, d.hidden),
            "wo": (d.hidden, d.q_dim),
            "router": (d.routed, d.hidden),
            "router_bias": (d.routed,),
            "attn_taps": (d.hidden, 4),
            "mlp_taps": (d.hidden, 4),
        }
        if any(tuple(tensors[k].shape) != v for k, v in shapes.items()):
            return False
        per_byte = 2 if experts == "nvfp4" else 1
        w13, w2 = tensors["w13"][0], tensors["w2"][0]
        return tuple(w13.shape) == (
            d.local_experts,
            2 * d.inter,
            d.hidden // per_byte,
        ) and tuple(w2.shape) == (d.local_experts, d.hidden, d.inter // per_byte)

    @register_kernel(
        "monokernel",
        "inkling2_local_layer_prepare",
        name="gluon_inkling2_local_layer_prepare_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_inkling2_local_layer_prepare_gfx950(
        *,
        tensors: dict[str, Any],
        rows: tuple[int, ...],
        experts: str,
        boundary: bool,
        first: bool,
        full: bool,
        later_passes: bool,
    ) -> tuple[Any, int] | None:
        """The packed weights (every row count's launches read them) and the
        mailbox pairs of the largest launch, or None for weights of other
        shapes."""
        if not _served(tensors, experts, full):
            return None
        t = tensors
        ip = _inkling2_plan
        d = _dims(tensors, experts)

        def plan_of(r: int):
            if full:
                # Every split count's plan has the same regions.
                return ip.full_layer_plan(d, r, r, experts, ip.FULL_SPLITS)
            return ip.local_layer_plan(d, r, r, experts, boundary, first)

        kv_proj = ()
        if not full and later_passes:
            # A looped core's later passes project a local layer's separate K
            # and V sources: the K and V rows of its projection.
            kv_proj = tuple(
                bf16_tiles(
                    torch.nn.functional.pad(
                        t["qkvr"][off : off + d.kv_dim],
                        (0, 0, 0, d.kv_proj_padded - d.kv_dim),
                    )
                )
                for off in (d.k_off, d.v_off)
            )
        if full:
            # The K and V sources' projections: their values' rows and taps'
            # rows of the full projection, zero rows to the padded rows.
            taps = d.kv_heads * 4
            kv_proj = tuple(
                bf16_tiles(
                    torch.nn.functional.pad(
                        torch.cat(
                            (
                                t["qkvr"][
                                    d.k_off
                                    + i * d.kv_dim : d.k_off
                                    + (i + 1) * d.kv_dim
                                ],
                                t["qkvr"][
                                    d.taps_off + i * taps : d.taps_off + (i + 1) * taps
                                ],
                            )
                        ),
                        (0, 0, 0, d.kv_proj_padded - d.kv_proj),
                    )
                )
                for i in range(2)
            )
        weights = {
            "dims": d,
            # The GEMV stages read the BF16 projections in N16 tile order: a
            # copy, since prefill keeps reading the row-major weights.
            "attention": (
                # Zero rows to QKVR_PADDED: the projection tasks of many rows
                # take 64 rows each.
                bf16_tiles(
                    torch.nn.functional.pad(
                        t["qkvr"],
                        (0, 0, 0, d.qkvr_padded - t["qkvr"].shape[0]),
                    )
                ),
                tuple(t["head"]),
                bf16_tiles(t["wo"]),
                t["attn_taps"],
            ),
            "kv_proj": kv_proj,
            "mlp": (
                bf16_tiles(t["router"]),
                t["router_bias"],
                tuple(t["w13"]),
                tuple(t["w2"]),
                t["mlp_taps"],
            ),
        }
        # A plan's mailbox depends on its rows only, not on the chain or the
        # experts' format.
        pairs = max(plan_of(r).scratch_pairs for r in rows)
        return weights, pairs

    @register_kernel(
        "monokernel",
        "inkling2_local_layer_decode",
        name="gluon_inkling2_local_layer_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_inkling2_local_layer_gfx950(
        *,
        part: str,
        inputs: tuple[torch.Tensor, ...],
        weights: Any,
        experts: str,
        rank: int,
        rows: int,
        chain: int,
        scratch: torch.Tensor,
        step: torch.Tensor,
        layer_index: int,
        parity: int,
        bases: torch.Tensor,
        terms: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        queries: tuple[tuple, tuple],
        boundary: bool,
        first: bool,
        full_splits: int,
        sweep: tuple,
        tau: torch.Tensor | None,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        locations: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        page_size: int,
        ring: torch.Tensor,
        ring_slots: torch.Tensor,
    ) -> None:
        d = weights["dims"]
        if part != "layer":
            plan = _inkling2_plan.later_pass_plan(
                d, rows, chain, experts, full_splits or None, part
            )
        elif full_splits > 0:
            plan = _inkling2_plan.full_layer_plan(d, rows, chain, experts, full_splits)
        else:
            plan = _inkling2_plan.local_layer_plan(
                d, rows, chain, experts, boundary, first
            )
        _inkling2_layer.launch_gluon_inkling2_local_layer_gfx950(
            plan.key,
            inputs,
            experts,
            # A later pass's parts run no stage of a boundary layer or layer
            # 0 of their own: one compiled part serves every layer.
            boundary and part == "layer",
            first and part == "layer",
            full_splits,
            rank,
            scratch,
            step,
            layer_index,
            parity,
            bases,
            terms,
            outputs,
            queries,
            sweep,
            weights["attention"],
            weights["kv_proj"],
            tau,
            (
                (k_cache, v_cache, locations),
                (
                    k_cache,
                    v_cache,
                    positions,
                    page_table,
                    page_table.stride(0),
                    page_size,
                ),
            ),
            (ring, ring_slots, positions),
            weights["mlp"],
            None,
        )
