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

"""CuTe DSL registration for LongCat packed over-embedding."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.over_embedding.spec import OverEmbeddingSpec
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.embedding import _cutedsl_available

platform = current_platform()

if platform.is_nvidia and _cutedsl_available():
    from tokenspeed_kernel.thirdparty.cute_dsl.longcat.embedding import (
        longcat_oe_append_packed_lookup,
    )

    @register_kernel(
        "over_embedding",
        "append_packed_lookup",
        name="cute_dsl_longcat_oe_append_packed_lookup",
        solution="cute_dsl",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=format_signatures(("table",), "dense", {torch.bfloat16}),
        priority=Priority.SPECIALIZED,
        traits={
            "fragment_count": frozenset({2}),
            "fragment_widths": frozenset({(512, 512)}),
        },
    )
    def cute_dsl_longcat_oe_append_packed_lookup(
        *,
        input_ids: torch.Tensor,
        input_start_offsets: torch.Tensor,
        req_pool_indices: torch.Tensor,
        active_request_mask: torch.Tensor,
        history_token_ids: torch.Tensor,
        committed_lengths: torch.Tensor,
        oe_tables: tuple[torch.Tensor, ...],
        out: torch.Tensor,
        spec: OverEmbeddingSpec,
        enable_pdl: bool,
    ) -> None:
        """Launch packed LongCat OE lookup for two 512-column fragments."""
        longcat_oe_append_packed_lookup(
            input_ids,
            input_start_offsets,
            req_pool_indices,
            active_request_mask,
            history_token_ids,
            committed_lengths,
            oe_tables,
            out,
            vocab_size=spec.vocab_size,
            fragment_configs=tuple(
                (
                    fragment.ngram_order,
                    fragment.modulus,
                    fragment.feature_width,
                )
                for fragment in spec.fragments
            ),
            ignored_token_ids=spec.ignored_token_ids,
            eos_token_id=spec.eos_token_id,
            enable_pdl=enable_pdl,
        )
