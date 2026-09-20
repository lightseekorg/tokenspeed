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


import torch
from tokenspeed_kernel.ops.attention.mha._triton.context import *  # noqa: F403
from tokenspeed_kernel.ops.attention.mha._triton.decode import (
    _triton_mha_decode_with_kvcache_impl,
)
from tokenspeed_kernel.ops.attention.mha._triton.prefill import (
    _triton_mha_extend_with_kvcache_impl,
    _triton_mha_prefill_impl,
)
from tokenspeed_kernel.ops.attention.mha._triton.qkv_rotary import *  # noqa: F403
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_PORTABLE_CAPABILITY = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))
_PORTABLE_DTYPES = {torch.float16, torch.bfloat16}


@register_kernel(
    "attention",
    "mha_prefill",
    name="triton_mha_prefill",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures(("q", "k", "v"), "dense", _PORTABLE_DTYPES),
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
        "support_skip_softmax": frozenset({False}),
    },
)
def triton_mha_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _triton_mha_prefill_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max_seqlen,
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


@register_kernel(
    "attention",
    "mha_extend_with_kvcache",
    name="triton_mha_extend_with_kvcache",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", _PORTABLE_DTYPES
    ),
    priority=Priority.PORTABLE,
    traits={
        "is_causal": frozenset({False, True}),
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
)
def triton_mha_extend_with_kvcache(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _triton_mha_extend_with_kvcache_impl(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        is_causal=is_causal,
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=enable_pdl,
    )


@register_kernel(
    "attention",
    "mha_decode_with_kvcache",
    name="triton_mha_decode_with_kvcache",
    solution="triton",
    capability=_PORTABLE_CAPABILITY,
    signatures=format_signatures(
        ("q", "k_cache", "v_cache"), "dense", _PORTABLE_DTYPES
    ),
    priority=Priority.PORTABLE,
    traits={
        "sliding_window": frozenset({False, True}),
        "support_sinks": frozenset({False, True}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False}),
    },
)
def triton_mha_decode_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    return _triton_mha_decode_with_kvcache_impl(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        window_left=window_left,
        logit_cap=logit_cap,
        sinks=sinks,
        return_lse=return_lse,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=enable_pdl,
    )


if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.mha import (
        mha_decode_with_kvcache as _mha_decode_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mha import (
        mha_extend_with_kvcache as _mha_extend_with_kvcache,
    )
    from tokenspeed_kernel_npu.ops.mha import mha_prefill as _mha_prefill

    _NPU_CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    _NPU_OPTIONS = {
        "sliding_window": frozenset({False}),
        "support_sinks": frozenset({False}),
        "support_logit_cap": frozenset({False}),
        "support_skip_softmax": frozenset({False}),
        "return_lse": frozenset({False}),
    }

    @register_kernel(
        "attention",
        "mha_prefill",
        name="torch_npu_mha_prefill",
        solution="torch_npu",
        capability=_NPU_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _PORTABLE_DTYPES),
        priority=Priority.PERFORMANT,
        traits=_NPU_OPTIONS,
    )
    def torch_npu_mha_prefill(**kwargs):
        return _mha_prefill(**kwargs)

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="torch_npu_mha_extend_with_kvcache",
        solution="torch_npu",
        capability=_NPU_CAPABILITY,
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", _PORTABLE_DTYPES
        ),
        priority=Priority.PERFORMANT,
        traits={
            **_NPU_OPTIONS,
            "page_size": frozenset({64, 128}),
            "is_causal": frozenset({False, True}),
        },
    )
    def torch_npu_mha_extend_with_kvcache(**kwargs):
        return _mha_extend_with_kvcache(**kwargs)

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="torch_npu_mha_decode_with_kvcache",
        solution="torch_npu",
        capability=_NPU_CAPABILITY,
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"), "dense", _PORTABLE_DTYPES
        ),
        priority=Priority.PERFORMANT,
        traits={
            **_NPU_OPTIONS,
            "page_size": frozenset({64, 128}),
            "q_len": frozenset({1}),
        },
    )
    def torch_npu_mha_decode_with_kvcache(**kwargs):
        return _mha_decode_with_kvcache(**kwargs)
