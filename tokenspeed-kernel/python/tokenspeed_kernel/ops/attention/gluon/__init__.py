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
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

if current_platform().is_amd:
    _DSA_FULL_TOPK_WIDTHS = frozenset({512, 1024, 2048})
    _DSA_PREFILL_TOPK_WIDTHS = _DSA_FULL_TOPK_WIDTHS

    from tokenspeed_kernel_amd.ops.attention.gluon.dsa_gfx950 import (
        gluon_dsa_decode_gfx950 as _dsa_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.dsa_gfx950 import (
        gluon_dsa_prefill_gfx950 as _dsa_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.dsa_topk_gfx950 import (
        gluon_dsa_decode_topk_fp8_gfx950 as _dsa_decode_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.dsa_topk_gfx950 import (
        gluon_dsa_prefill_topk_fp8_gfx950 as _dsa_prefill_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_decode_gfx950 import (
        gluon_mha_decode_gfx950 as _decode_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_extend_gfx950 import (
        gluon_mha_extend_gfx950 as _extend_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_prefill_gfx950 import (
        gluon_mha_prefill_gfx950 as _prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_prefill_gfx1250 import (
        gluon_mha_prefill_gfx1250 as _prefill_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.mla_decode_bf16_gfx950 import (
        gluon_mla_decode_bf16_gfx950 as _mla_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.mla_prefill_bf16_gfx950 import (
        gluon_mla_prefill_bf16_gfx950 as _mla_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.rel_mha_decode_gfx950 import (
        gluon_rel_mha_decode_gfx950 as _rel_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.rel_mha_extend_gfx950 import (
        gluon_rel_mha_extend_gfx950 as _rel_extend_impl,
    )
    from tokenspeed_kernel_amd.ops.attention.gluon.rel_mha_prefill_gfx950 import (
        gluon_rel_mha_prefill_gfx950 as _rel_prefill_impl,
    )

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="gluon_mha_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_mha_decode_gfx950(*args, **kwargs):
        return _decode_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_prefill",
        name="gluon_mha_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_prefill_gfx950(*args, **kwargs):
        return _prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_prefill",
        name="gluon_mha_prefill_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_prefill_gfx1250(*args, **kwargs):
        return _prefill_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mha_extend_with_kvcache",
        name="gluon_mha_extend_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k_cache", "v_cache"),
            "dense",
            {
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            },
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({64, 128}),
            "page_size": frozenset({64}),
            "is_causal": frozenset({False, True}),
            "sliding_window": frozenset({False, True}),
            "support_sinks": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mha_extend_gfx950(*args, **kwargs):
        return _extend_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16_gfx950_bh16bn64",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "kv_cache"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "q_len": frozenset({1}),
            "num_q_heads": frozenset(range(1, 17)),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_bf16_gfx950_bh16bn64(*args, **kwargs):
        return _mla_decode_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16_gfx950_bh64",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "kv_cache"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "q_len": frozenset({1}),
            "num_q_heads": frozenset({64, 128}),
            "batch_size_div_64": frozenset({True}),
            "page_size": frozenset({64}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_decode_bf16_gfx950_bh64(*args, **kwargs):
        return _mla_decode_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_prefill",
        name="gluon_mla_prefill_bf16_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "qk_head_dim": frozenset({192}),
            "v_head_dim": frozenset({128}),
            "is_causal": frozenset({False, True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
        },
    )
    def gluon_mla_prefill_bf16_gfx950(*args, **kwargs):
        return _mla_prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_fp8_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "index_k_format": frozenset({"fp8_scaled"}),
        },
    )
    def gluon_dsa_decode_topk_fp8_gfx950(*args, **kwargs):
        return _dsa_decode_topk_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_fp8_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "index_k_format": frozenset({"fp8_scaled"}),
        },
    )
    def gluon_dsa_prefill_topk_fp8_gfx950(*args, **kwargs):
        return _dsa_prefill_topk_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_decode",
        name="gluon_dsa_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_decode_gfx950(*args, **kwargs):
        return _dsa_decode_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="gluon_dsa_prefill_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_prefill_gfx950(*args, **kwargs):
        return _dsa_prefill_impl(*args, **kwargs)

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
        return _rel_decode_impl(*args, **kwargs)
