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

"""Ascend MLA kernel registrations."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def ascend_mla_dcp_kernels():
    """Return the Ascend packed MLA partial/merge and graph-length adapter."""
    if not current_platform().is_npu:
        raise RuntimeError("Ascend MLA DCP kernels require an NPU platform")
    from tokenspeed_kernel_npu.ops import mla_dcp

    return mla_dcp


if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.mla import (
        mla_decode_with_kvcache as _mla_decode_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mla import (
        mla_extend_with_kvcache as _mla_extend_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mla import (
        mla_normalize_project_query as _mla_normalize_project_query,
    )
    from tokenspeed_kernel_npu.ops.mla import mla_prefill as _mla_prefill
    from tokenspeed_kernel_npu.ops.mla import mla_project_value as _mla_project_value

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _DTYPES = {torch.float16, torch.bfloat16}

    @register_kernel(
        "attention",
        "mla_normalize_project_query",
        name="ascend_mla_normalize_project_query",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(
            ("query", "kv", "projection_weight", "out"), "dense", _DTYPES
        ),
        priority=Priority.PERFORMANT,
        traits={"split_output": frozenset({False})},
        features={"portability"},
    )
    def mla_normalize_project_query(**kwargs):
        return _mla_normalize_project_query(**kwargs)

    @register_kernel(
        "attention",
        "mla_project_value",
        name="ascend_mla_project_value",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("attention", "weight", "out"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            "gate_kind": frozenset({"none", "sigmoid"}),
            "inputs_contiguous": frozenset({True}),
        },
        features={"portability"},
    )
    def mla_project_value(**kwargs):
        return _mla_project_value(**kwargs)

    @register_kernel(
        "attention",
        "mla_prefill",
        name="ascend_mla_prefill",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            "is_causal": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
        features={"portability"},
    )
    def mla_prefill(**kwargs):
        return _mla_prefill(**kwargs)

    @register_kernel(
        "attention",
        "mla_extend_with_kvcache",
        name="ascend_mla_extend_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "kv_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            "page_size": frozenset({64, 128}),
            "qk_nope_head_dim": frozenset({128}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "is_causal": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
        features={"portability"},
    )
    def mla_extend_with_kvcache(**kwargs):
        return _mla_extend_with_kvcache(**kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="ascend_mla_decode_with_kvcache",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "kv_cache"), "dense", _DTYPES),
        priority=Priority.PERFORMANT,
        traits={
            "page_size": frozenset({64, 128}),
            "q_len": frozenset({1}),
            "qk_nope_head_dim": frozenset({128}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "sliding_window": frozenset({False}),
            "block_on_query_axis": frozenset({True}),
        },
        features={"portability"},
    )
    def mla_decode_with_kvcache(**kwargs):
        return _mla_decode_with_kvcache(**kwargs)


__all__ = [
    "mla_decode_with_kvcache",
    "mla_extend_with_kvcache",
    "mla_normalize_project_query",
    "mla_prefill",
    "mla_project_value",
]
