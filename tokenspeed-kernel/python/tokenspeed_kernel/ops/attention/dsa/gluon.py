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
from tokenspeed_kernel.ops.tuning import is_warming_up
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
        launch_gluon_dsa_decode_gfx950 as _dsa_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
        launch_gluon_dsa_prefill_gfx950 as _dsa_prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        launch_gluon_dsa_decode_topk_bf16_gfx950 as _dsa_decode_topk_bf16_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        launch_gluon_dsa_decode_topk_fp8_gfx950 as _dsa_decode_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        launch_gluon_dsa_decode_topk_fp8_rows_gfx950 as _dsa_decode_topk_fp8_rows_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        launch_gluon_dsa_decode_topk_standard_gfx950 as _dsa_decode_topk_standard_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        launch_gluon_dsa_prefill_topk_bf16_gfx950 as _dsa_prefill_topk_bf16_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        launch_gluon_dsa_prefill_topk_fp8_gfx950 as _dsa_prefill_topk_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        launch_gluon_dsa_prefill_topk_fp8_rows_gfx950 as _dsa_prefill_topk_fp8_rows_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
        launch_gluon_dsa_prefill_topk_standard_gfx950 as _dsa_prefill_topk_standard_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.attention import (
        launch_gluon_dsa_decode_gfx1250 as _dsa_decode_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.attention import (
        launch_gluon_dsa_prefill_gfx1250 as _dsa_prefill_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        launch_gluon_dsa_decode_topk_fp8_gfx1250 as _dsa_decode_topk_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        launch_gluon_dsa_decode_topk_standard_gfx1250 as _dsa_decode_topk_standard_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        launch_gluon_dsa_prefill_topk_fp8_gfx1250 as _dsa_prefill_topk_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.dsa.sparse_mla import (
        launch_gluon_dsa_prefill_topk_standard_gfx1250 as _dsa_prefill_topk_standard_gfx1250_impl,
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
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
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
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
    )
    def gluon_dsa_prefill_topk_standard_gfx950(*args, **kwargs):
        return _dsa_prefill_topk_standard_impl(*args, **kwargs)

    # E4M3 queries share each gathered key tile across rows; BF16 stays standard.
    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_fp8_rows_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.float8_e4m3fn),
                    weights=dense_tensor_format(weight_dtype),
                )
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 2,
        traits={
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
    )
    def gluon_dsa_decode_topk_fp8_rows_gfx950(*args, **kwargs):
        return _dsa_decode_topk_fp8_rows_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_fp8_rows_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.float8_e4m3fn),
                    weights=dense_tensor_format(weight_dtype),
                )
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 2,
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
    )
    def gluon_dsa_prefill_topk_fp8_rows_gfx950(*args, **kwargs):
        return _dsa_prefill_topk_fp8_rows_impl(*args, **kwargs)

    # BF16 plane scorers; radix ties are not lowest-index, so not batch_invariant.
    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_bf16_gfx950",
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
                    weights=dense_tensor_format(weight_dtype),
                )
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "index_k_format": frozenset({"bf16"}),
            "index_k_layout": frozenset({"packed"}),
        },
    )
    def gluon_dsa_decode_topk_bf16_gfx950(*args, **kwargs):
        return _dsa_decode_topk_bf16_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "dsa_prefill_topk",
        name="gluon_dsa_prefill_topk_bf16_gfx950",
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
                    weights=dense_tensor_format(weight_dtype),
                )
                for weight_dtype in (torch.bfloat16, torch.float32)
            }
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
            "index_k_format": frozenset({"bf16"}),
            "index_k_layout": frozenset({"packed"}),
        },
    )
    def gluon_dsa_prefill_topk_bf16_gfx950(*args, **kwargs):
        return _dsa_prefill_topk_bf16_impl(*args, **kwargs)

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
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
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
            "page_size": frozenset({64}),
            "topk": frozenset({512, 1024, 2048}),
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
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "page_size": frozenset({64}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "has_kv_cache": frozenset({False, True}),
            "has_sparse_kv_cache": frozenset({False, True}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
        },
    )
    def gluon_dsa_decode_gfx950(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_decode_impl(*args, warm_split_variants=is_warming_up(), **kwargs)

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
            "q_len": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "page_size": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "has_kv_cache": frozenset({False, True}),
            "has_sparse_kv_cache": frozenset({False, True}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
        },
    )
    def gluon_dsa_prefill_gfx950(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_prefill_impl(*args, warm_split_variants=is_warming_up(), **kwargs)

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
            "q_len": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "page_size": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "has_kv_cache": frozenset({True}),
            "has_sparse_kv_cache": frozenset({False}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
        },
    )
    def gluon_dsa_prefill_fp8_dense_gfx950(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_prefill_impl(*args, warm_split_variants=is_warming_up(), **kwargs)

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
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
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
            "page_size": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
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
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
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
            "page_size": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
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
            "q_len": frozenset({1, 2, 3, 4, 5, 6}),
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "page_size": frozenset({64}),
            "topk": _DSA_FULL_TOPK_WIDTHS,
            "has_kv_cache": frozenset({False, True}),
            "has_sparse_kv_cache": frozenset({False, True}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
        },
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
            "q_len": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "page_size": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "has_kv_cache": frozenset({False, True}),
            "has_sparse_kv_cache": frozenset({False, True}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
        },
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
            "q_len": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192, 256}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({0, 64}),
            "page_size": frozenset({64}),
            "topk": _DSA_PREFILL_TOPK_WIDTHS,
            "has_kv_cache": frozenset({True}),
            "has_sparse_kv_cache": frozenset({False}),
            "logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
            "topk_layout": frozenset({"global_slots"}),
        },
    )
    def gluon_dsa_prefill_fp8_dense_gfx1250(*args, enable_pdl: bool = False, **kwargs):
        kwargs.pop("kv_seq_lens", None)
        return _dsa_prefill_gfx1250_impl(*args, **kwargs)

    # This experimental leaf is selected only by an explicit solution. The
    # portable candidate implementation keeps automatic/default selection.
    from tokenspeed_kernel.ops.attention.dsa import (
        INDEX_CANDIDATE_PAGES_FEATURE,
        DSAIndexCandidatePages,
    )

    @register_kernel(
        "attention",
        "dsa_index_candidates",
        name="gluon_dsa_index_candidates_gfx950",
        solution="gluon",
        features={INDEX_CANDIDATE_PAGES_FEATURE},
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            format_signature(
                q=dense_tensor_format(torch.float8_e4m3fn),
                weights=dense_tensor_format(dtype),
            )
            for dtype in (torch.bfloat16, torch.float32)
        ),
        traits={
            "index_heads": frozenset({32, 64}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({64}),
            "index_k_layout": frozenset({"packed", "page_planar"}),
        },
        priority=Priority.REFERENCE,
    )
    def gluon_dsa_index_candidates_gfx950(
        q,
        weights,
        index_k_cache,
        local_page_table,
        query_requests,
        causal_lens,
        *,
        page_size,
        topk,
        softmax_scale,
        initial_tokens,
        local_tokens,
        pages: DSAIndexCandidatePages | None = None,
        q_scales: torch.Tensor | None = None,
        query_group_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tokenspeed_kernel.ops.attention.dsa._triton.index_candidates import (
            compact_index_pages,
            gather_index_candidates,
            mask_index_scores,
        )
        from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (
            launch_gluon_dsa_index_candidates_gfx950,
        )
        from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.sparse_mla import (
            gluon_dsa_logical_topk_gfx950,
        )

        if not 0 < topk <= 2048:
            raise ValueError("gfx950 owned-page top-k supports at most2048 candidates")
        if q_scales is None:
            raise ValueError("gfx950 owned-page scorer requires FP8 query scales")
        if pages is None:
            physical, positions, lengths = compact_index_pages(
                local_page_table, query_requests, causal_lens, page_size
            )
        else:
            physical, positions, lengths = pages.pages, pages.positions, pages.lengths
        lengths = lengths.reshape(-1)
        scores = launch_gluon_dsa_index_candidates_gfx950(
            q.contiguous(),
            q_scales.contiguous(),
            weights.contiguous(),
            index_k_cache,
            physical,
            lengths,
            page_size=page_size,
            softmax_scale=softmax_scale,
            query_group_size=query_group_size,
        )
        mask_index_scores(
            scores,
            positions,
            lengths,
            causal_lens,
            page_size,
            initial_tokens,
            local_tokens,
            mask_tail=False,
        )
        select_width = next(width for width in (512, 1024, 2048) if width >= topk)
        offsets, _ = gluon_dsa_logical_topk_gfx950(
            scores, torch.zeros_like(lengths), lengths, topk=select_width
        )
        if select_width != topk:
            # Small generic/test top-k counts refine only this bounded
            # candidate pool, never the unwritten capacity tail.
            selected = scores.gather(1, offsets.clamp_min(0).long())
            selected = torch.where(offsets >= 0, selected, -float("inf"))
            order = torch.topk(selected, topk, dim=1).indices
            offsets = offsets.gather(1, order)
        return gather_index_candidates(
            offsets, scores, positions, page_size, lengths=lengths
        )
