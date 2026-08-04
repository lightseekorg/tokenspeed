"""AMD Gluon KDA registrations."""

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
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_recurrent_decode_gfx950 as _kda_decode_impl,
    )

    @register_kernel(
        "attention",
        "kda_paged_decode",
        name="gluon_kda_paged_decode_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "indexed_state": frozenset({True}),
            "single_token": frozenset({True}),
        },
        tags={"amd", "gfx950", "paged_cache", "cuda_graph"},
    )
    def gluon_kda_paged_decode_gfx950(**kwargs):
        """Run specialized gfx950 KDA decode against the canonical K-major pool."""
        return _kda_decode_impl(**kwargs)
