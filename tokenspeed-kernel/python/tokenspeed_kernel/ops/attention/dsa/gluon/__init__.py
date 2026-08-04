"""AMD Gluon DSA registrations."""

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
        gluon_dsa_prefill_topk_fp8_gfx950 as _dsa_prefill_topk_impl,
    )

    _CAPABILITY = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    _FULL_TOPK_WIDTHS = frozenset({512, 1024, 2048})

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="gluon_dsa_decode_topk_fp8_gfx950",
        solution="gluon",
        capability=_CAPABILITY,
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
            "topk": _FULL_TOPK_WIDTHS,
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
        capability=_CAPABILITY,
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
            "topk": _FULL_TOPK_WIDTHS,
            "page_size": frozenset({64}),
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
        capability=_CAPABILITY,
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
            "topk": _FULL_TOPK_WIDTHS,
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
        capability=_CAPABILITY,
        signatures=frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))}),
        priority=Priority.SPECIALIZED,
        traits={
            "page_size": frozenset({64}),
            "q_len_per_req": frozenset({1}),
            "qk_nope_head_dim": frozenset({128, 192}),
            "kv_lora_rank": frozenset({128, 512}),
            "qk_rope_head_dim": frozenset({64}),
            "topk": _FULL_TOPK_WIDTHS,
            "kv_cache_available": frozenset({False, True}),
            "sparse_kv_cache_available": frozenset({False, True}),
            "topk_layout": frozenset({"global_slots"}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_dsa_prefill_gfx950(*args, **kwargs):
        return _dsa_prefill_impl(*args, **kwargs)
