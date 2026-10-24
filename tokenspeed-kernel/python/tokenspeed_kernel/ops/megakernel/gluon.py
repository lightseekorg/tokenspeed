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

"""Shims that register the AMD Gluon megakernels."""

from __future__ import annotations

from typing import Any

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import (
        layer as _qwen4_exp_layer,
    )
    from tokenspeed_kernel_amd.ops.gfx950.megakernel.qwen4_exp import (
        weights as _qwen4_exp_weights,
    )
except ImportError:
    _qwen4_exp_layer = None


if _qwen4_exp_layer is not None:
    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    _SIGNATURES = format_signatures("h", "dense", {torch.bfloat16})

    def _pack_moe(t: dict[str, Any], dense_tiled: bool) -> Any:
        """Pop the MoE block's tensors from ``t`` and pack them, in the
        format that ``t["experts"]`` names (``layer.EXPERTS``)."""
        experts = t.pop("experts")
        if experts == "bf16":
            return _qwen4_exp_weights.pack_moe_bf16(
                router_w=t.pop("router_w"),
                shared_gate_w=t.pop("shared_gate_w"),
                w13=t.pop("w13"),
                w2=t.pop("w2"),
                shared_gu=t.pop("shared_gu"),
                shared_dn=t.pop("shared_dn"),
            )
        if experts != "mxfp4":
            raise ValueError(f"unknown experts format {experts!r}")
        return _qwen4_exp_weights.pack_moe(
            router_w=t.pop("router_w"),
            shared_gate_w=t.pop("shared_gate_w"),
            banks=t.pop("banks"),
            shared_gu=t.pop("shared_gu"),
            shared_dn=t.pop("shared_dn"),
            dense_tiled=dense_tiled,
        )

    _TRAITS = {"rows": frozenset(range(1, _qwen4_exp_layer.MAX_ROWS + 1))}

    @register_kernel(
        "megakernel",
        "qwen4_exp_delta_layer_prepare",
        name="gluon_qwen4_exp_delta_layer_prepare_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_qwen4_exp_delta_layer_prepare_gfx950(
        *, tensors: dict[str, Any], dense_tiled: bool, max_rows: int
    ) -> tuple[Any, int]:
        t = dict(tensors)
        ple = t.pop("ple", None)
        weights = _qwen4_exp_weights.pack_delta_layer(
            attn_mixer=_qwen4_exp_weights.pack_mixer(*t.pop("attn_mixer")),
            mlp_mixer=_qwen4_exp_weights.pack_mixer(*t.pop("mlp_mixer")),
            moe=_pack_moe(t, dense_tiled),
            ple=(None if ple is None else _qwen4_exp_weights.pack_ple(**ple)),
            **t,
        )
        # Size the scratch for the widest launch. Fewer rows use a prefix of it.
        plan = _qwen4_exp_layer.delta_plan(
            ple is not None, max_rows, 1, _qwen4_exp_layer.experts_format(weights)
        )
        return weights, plan.scratch_pairs

    @register_kernel(
        "megakernel",
        "qwen4_exp_delta_layer_decode",
        name="gluon_qwen4_exp_delta_layer_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_qwen4_exp_delta_layer_gfx950(
        *,
        weights: Any,
        h: torch.Tensor,
        h_out: torch.Tensor,
        scratch: torch.Tensor,
        step: torch.Tensor,
        layer_index: int,
        ple: dict[str, torch.Tensor] | None,
        **state: Any,
    ) -> None:
        _qwen4_exp_layer.launch_gluon_qwen4_exp_delta_layer_gfx950(
            h,
            h_out,
            weights,
            scratch=scratch,
            step=step,
            layer=layer_index,
            trace_buf=None,
            ple_step=None if ple is None else _qwen4_exp_layer.PleStep(**ple),
            **state,
        )

    @register_kernel(
        "megakernel",
        "qwen4_exp_qsa_layer_prepare",
        name="gluon_qwen4_exp_qsa_layer_prepare_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_qwen4_exp_qsa_layer_prepare_gfx950(
        *, tensors: dict[str, Any], dense_tiled: bool, max_rows: int
    ) -> tuple[Any, int]:
        t = dict(tensors)
        final_mixer = t.pop("final_mixer", None)
        weights = _qwen4_exp_weights.pack_qsa_layer(
            attn_mixer=_qwen4_exp_weights.pack_mixer(*t.pop("attn_mixer")),
            mlp_mixer=_qwen4_exp_weights.pack_mixer(*t.pop("mlp_mixer")),
            moe=_pack_moe(t, dense_tiled),
            final_mixer=(
                None
                if final_mixer is None
                else _qwen4_exp_weights.pack_mixer(*final_mixer)
            ),
            **t,
        )
        plan = _qwen4_exp_layer.qsa_plan(
            final_mixer is not None,
            max_rows,
            1,
            _qwen4_exp_layer.experts_format(weights),
            False,
        )
        return weights, plan.scratch_pairs

    @register_kernel(
        "megakernel",
        "qwen4_exp_qsa_layer_decode",
        name="gluon_qwen4_exp_qsa_layer_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_SIGNATURES,
        priority=Priority.SPECIALIZED,
        traits=_TRAITS,
    )
    def gluon_qwen4_exp_qsa_layer_gfx950(
        *,
        weights: Any,
        h: torch.Tensor,
        h_out: torch.Tensor,
        scratch: torch.Tensor,
        step: torch.Tensor,
        layer_index: int,
        **caches: Any,
    ) -> None:
        _qwen4_exp_layer.launch_gluon_qwen4_exp_qsa_layer_gfx950(
            h,
            h_out,
            weights,
            scratch=scratch,
            step=step,
            layer=layer_index,
            trace_buf=None,
            **caches,
        )
