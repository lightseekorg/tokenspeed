"""AMD Gluon MLA registrations."""

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
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_bf16xbf16_gfx950 as _decode_bf16xbf16_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_bf16xfp8_gfx950 as _decode_bf16xfp8_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.decode import (
        gluon_mla_decode_fp8xfp8_gfx950 as _decode_fp8xfp8_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.mla.prefill import (
        gluon_mla_prefill_bf16_gfx950 as _prefill_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.decode import (
        gluon_mla_decode_bf16_gfx1250 as _decode_bf16_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.extend import (
        gluon_mla_extend_bf16_gfx1250 as _extend_bf16_gfx1250_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.attention.mla.prefill import (
        gluon_mla_prefill_bf16_gfx1250 as _prefill_gfx1250_impl,
    )

    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )
    _GFX1250 = CapabilityRequirement(
        min_arch_version=ArchVersion(12, 5),
        max_arch_version=ArchVersion(12, 5),
        vendors=frozenset({"amd"}),
    )
    _DECODE_TRAITS = {
        "q_len": frozenset({1}),
        "page_size": frozenset({64}),
        "kv_lora_rank": frozenset({512}),
        "qk_rope_head_dim": frozenset({64}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
    }

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16xbf16_gfx950_bh16bn64",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(("q", "kv_cache"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={**_DECODE_TRAITS, "num_q_heads": frozenset(range(1, 17))},
    )
    def gluon_mla_decode_bf16xbf16_gfx950_bh16bn64(*args, **kwargs):
        return _decode_bf16xbf16_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16xfp8_gfx950_bh16bn128",
        solution="gluon",
        capability=_GFX950,
        signatures=frozenset(
            format_signature(
                q=dense_tensor_format(q_dtype),
                kv_cache=dense_tensor_format(kv_dtype),
            )
            for q_dtype, kv_dtype in (
                (torch.bfloat16, torch.float8_e4m3fn),
                (torch.bfloat16, torch.float8_e5m2),
            )
        ),
        priority=Priority.SPECIALIZED,
        traits={**_DECODE_TRAITS, "num_q_heads": frozenset(range(1, 17))},
    )
    def gluon_mla_decode_bf16xfp8_gfx950_bh16bn128(*args, **kwargs):
        return _decode_bf16xfp8_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_fp8xfp8_gfx950_bh16bn128",
        solution="gluon",
        capability=_GFX950,
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.float8_e4m3fn),
                    kv_cache=dense_tensor_format(torch.float8_e4m3fn),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={**_DECODE_TRAITS, "num_q_heads": frozenset(range(1, 17))},
    )
    def gluon_mla_decode_fp8xfp8_gfx950_bh16bn128(*args, **kwargs):
        return _decode_fp8xfp8_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16xbf16_gfx950_bh64",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(("q", "kv_cache"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            **_DECODE_TRAITS,
            "num_q_heads": frozenset({64, 128}),
            "batch_size_div_64": frozenset({True}),
        },
    )
    def gluon_mla_decode_bf16xbf16_gfx950_bh64(*args, **kwargs):
        return _decode_bf16xbf16_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_decode_with_kvcache",
        name="gluon_mla_decode_bf16_gfx1250",
        solution="gluon",
        capability=_GFX1250,
        signatures=format_signatures(("q", "kv_cache"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={**_DECODE_TRAITS, "num_q_heads": frozenset(range(1, 129))},
    )
    def gluon_mla_decode_bf16_gfx1250(*args, **kwargs):
        return _decode_bf16_gfx1250_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_extend_with_kvcache",
        name="gluon_mla_extend_bf16_gfx1250",
        solution="gluon",
        capability=_GFX1250,
        signatures=format_signatures(("q", "kv_cache"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "num_q_heads": frozenset(range(1, 129)),
            "max_seqlen_q": frozenset(range(1, 257)),
            "page_size": frozenset({64}),
            "qk_nope_head_dim": frozenset({128}),
            "kv_lora_rank": frozenset({512}),
            "qk_rope_head_dim": frozenset({64}),
            "is_causal": frozenset({True}),
            "support_logit_cap": frozenset({False}),
            "return_lse": frozenset({False}),
        },
    )
    def gluon_mla_extend_bf16_gfx1250(*args, **kwargs):
        return _extend_bf16_gfx1250_impl(*args, **kwargs)

    _PREFILL_TRAITS = {
        "qk_head_dim": frozenset({192}),
        "v_head_dim": frozenset({128}),
        "is_causal": frozenset({False, True}),
        "support_logit_cap": frozenset({False}),
        "return_lse": frozenset({False, True}),
    }

    @register_kernel(
        "attention",
        "mla_prefill",
        name="gluon_mla_prefill_bf16_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_PREFILL_TRAITS,
    )
    def gluon_mla_prefill_bf16_gfx950(*args, **kwargs):
        return _prefill_impl(*args, **kwargs)

    @register_kernel(
        "attention",
        "mla_prefill",
        name="gluon_mla_prefill_bf16_gfx1250",
        solution="gluon",
        capability=_GFX1250,
        signatures=format_signatures(("q", "k", "v"), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits=_PREFILL_TRAITS,
    )
    def gluon_mla_prefill_bf16_gfx1250(*args, **kwargs):
        return _prefill_gfx1250_impl(*args, **kwargs)
