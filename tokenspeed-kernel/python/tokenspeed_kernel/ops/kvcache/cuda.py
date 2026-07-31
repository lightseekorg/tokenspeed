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

"""CUDA KV cache transfer kernels."""

from __future__ import annotations

from functools import partial

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signature

try:
    from tokenspeed_kernel.thirdparty.cuda import kvcacheio as _kvcacheio
    from tokenspeed_kernel.thirdparty.cuda.kvcacheio import (
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_lf_ph,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
        transfer_kv_per_layer_ph_lf,
    )
except ImportError:
    _kvcacheio = None
    transfer_kv_all_layer_lf_pf = error_fn
    transfer_kv_all_layer_lf_ph = error_fn
    transfer_kv_all_layer_mla = error_fn
    transfer_kv_all_layer_mla_lf_pf = error_fn
    transfer_kv_direct = error_fn
    transfer_kv_per_layer_mla = error_fn
    transfer_kv_per_layer_mla_pf_lf = error_fn
    transfer_kv_per_layer_pf_lf = error_fn
    transfer_kv_per_layer_ph_lf = error_fn


if _kvcacheio is not None:

    @register_kernel(
        "kvcache",
        "h2d_scatter",
        name="cuda_kvcache_h2d_scatter",
        solution="cuda",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=frozenset({format_signature()}),
        priority=Priority.PERFORMANT,
        tags={"throughput"},
    )
    def cuda_prepare_h2d_scatter_plan(
        *,
        src_layers: list[torch.Tensor],
        dst_layers: list[torch.Tensor],
        entry_ids: list[int],
    ):
        plan, reason = _kvcacheio.prepare_kv_direct_h2d_scatter_plan(
            src_layers,
            dst_layers,
            entry_ids,
        )
        if plan is None:
            return None, reason
        return partial(_kvcacheio.transfer_kv_direct_h2d_scatter_prepared, plan), ""


__all__ = [
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_lf_ph",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_per_layer_pf_lf",
    "transfer_kv_per_layer_ph_lf",
]
