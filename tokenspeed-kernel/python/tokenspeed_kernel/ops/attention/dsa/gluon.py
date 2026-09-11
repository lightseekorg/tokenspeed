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

"""Registration shims for AMD Gluon DSA attention kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    _DSA_FULL_TOPK_WIDTHS = frozenset({512, 1024, 2048, 2049, 2050, 2051})
    _DSA_PREFILL_TOPK_WIDTHS = _DSA_FULL_TOPK_WIDTHS

    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
        gluon_dsa_decode_gfx950 as _dsa_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
        gluon_dsa_prefill_gfx950 as _dsa_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_decode_topk_fp8_gfx950 as _dsa_decode_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_decode_topk_standard_gfx950 as _dsa_decode_topk_standard_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_fp8_gfx950 as _dsa_prefill_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_standard_gfx950 as _dsa_prefill_topk_standard_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.attention import (
        gluon_dsa_decode_gfx1250 as _dsa_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.attention import (
        gluon_dsa_prefill_gfx1250 as _dsa_prefill_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        gluon_dsa_decode_topk_fp8_gfx1250 as _dsa_decode_topk_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        gluon_dsa_decode_topk_standard_gfx1250 as _dsa_decode_topk_standard_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_fp8_gfx1250 as _dsa_prefill_topk_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        gluon_dsa_prefill_topk_standard_gfx1250 as _dsa_prefill_topk_standard_gfx1250_impl,
    )

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_standard_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(q_dtype),
                    weights=dense_tensor_format(weight_dtype),
                )
                for q_dtype in (torch.bfloat16, torch.float8_e4m3fn)
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
    )
    def gluon_dsa_decode_topk_standard_gfx950(*args, **kwargs):
        return _dsa_decode_topk_standard_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_standard_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(q_dtype),
                    weights=dense_tensor_format(weight_dtype),
                )
                for q_dtype in (torch.bfloat16, torch.float8_e4m3fn)
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
    )
    def gluon_dsa_prefill_topk_standard_gfx950(*args, **kwargs):
        return _dsa_prefill_topk_standard_impl(*args, **kwargs)

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
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
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
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": frozenset({512, 1024, 2048}),
            "page_size": frozenset({64}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
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
                format_signature(q=dense_tensor_format(torch.float8_e5m2)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_decode_gfx950(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
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
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_prefill_gfx950(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="gluon_dsa_prefill_fp8_dense_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
                format_signature(q=dense_tensor_format(torch.float8_e5m2)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_prefill_fp8_dense_gfx950(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_standard_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(q_dtype),
                    weights=dense_tensor_format(weight_dtype),
                )
                for q_dtype in (torch.bfloat16, torch.float8_e4m3fn)
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_decode_topk_standard_gfx1250(*args, **kwargs):
        return _dsa_decode_topk_standard_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_standard_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(q_dtype),
                    weights=dense_tensor_format(weight_dtype),
                )
                for q_dtype in (torch.bfloat16, torch.float8_e4m3fn)
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "page_size": frozenset({64}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_prefill_topk_standard_gfx1250(*args, **kwargs):
        return _dsa_prefill_topk_standard_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_fp8_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1, 2, 3, 4, 5, 6}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_decode_topk_fp8_gfx1250(*args, **kwargs):
        return _dsa_decode_topk_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_fp8_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.float32),
                ),
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(torch.bfloat16),
                ),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "head_dim": frozenset({128}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "page_size": frozenset({64}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_prefill_topk_fp8_gfx1250(*args, **kwargs):
        return _dsa_prefill_topk_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_decode",
        name="gluon_dsa_decode_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.bfloat16)),
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
                format_signature(q=dense_tensor_format(torch.float8_e5m2)),
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
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_decode_gfx1250(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_decode_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="gluon_dsa_prefill_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
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
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_prefill_gfx1250(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_prefill_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill",
        name="gluon_dsa_prefill_fp8_dense_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
                format_signature(q=dense_tensor_format(torch.float8_e5m2)),
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({True}),
            "sparse_kv_cache_available": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
        tags={"amd", "gfx1250"},
    )
    def gluon_dsa_prefill_fp8_dense_gfx1250(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_prefill_gfx1250_impl(*args, **kwargs)
