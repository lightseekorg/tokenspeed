"""AMD Gluon relative-MHA registrations."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.decode import (
        gluon_rel_mha_decode_gfx950 as _rel_decode_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.extend import (
        gluon_rel_mha_extend_gfx950 as _rel_extend_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.prefill import (
        gluon_rel_mha_prefill_gfx950 as _rel_prefill_impl,
    )

    _CAPABILITY = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )

    @register_kernel(
        "attention",
        "rel_mha_prefill",
        name="gluon_rel_mha_prefill_gfx950",
        solution="gluon",
        capability=_CAPABILITY,
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
        capability=_CAPABILITY,
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
        capability=_CAPABILITY,
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
