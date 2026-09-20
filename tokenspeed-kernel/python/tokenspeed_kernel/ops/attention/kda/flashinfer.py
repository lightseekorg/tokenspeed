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

"""Register BF16 FlashInfer decode and frozen verify with TS producers."""

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.flashinfer.kda import (
    FLASHINFER_KDA_MAX_VERIFY_TOKENS,
    flashinfer_kda_producer_decode,
    flashinfer_kda_producer_verify,
    flashinfer_kda_recurrent_available,
)

_CAPABILITY = CapabilityRequirement(
    vendors=frozenset({"nvidia"}),
    min_arch_version=ArchVersion(10, 0),
    max_arch_version=ArchVersion(10, 3),
)
_SIGNATURES = format_signatures(("q", "k", "v"), "dense", {torch.bfloat16})

if flashinfer_kda_recurrent_available():
    register_kernel(
        "attention",
        "kda_fused_paged_decode",
        name="flashinfer_kda_recurrent_producer_decode",
        solution="flashinfer",
        capability=_CAPABILITY,
        signatures=_SIGNATURES,
        # Prefer this BF16-only path over native decode (SPECIALIZED).
        priority=Priority.SPECIALIZED + 1,
        traits={
            "paged_state": frozenset({True}),
            "fused_output_norm": frozenset({False, True}),
            "state_dtype": frozenset({torch.bfloat16}),
            "num_heads": frozenset({12}),
            "head_dim": frozenset({128}),
            "conv_kernel_size": frozenset({4}),
            "recurrent_layout": frozenset({"v_major"}),
        },
        tags={"nvidia", "paged_cache", "cuda_graph"},
    )(flashinfer_kda_producer_decode)

    register_kernel(
        "attention",
        "kda_fused_paged_verify",
        name="flashinfer_kda_recurrent_producer_verify",
        solution="flashinfer",
        capability=_CAPABILITY,
        signatures=_SIGNATURES,
        # Native split verify uses SPECIALIZED + 1.
        priority=Priority.SPECIALIZED + 2,
        traits={
            "paged_state": frozenset({True}),
            "store_states": frozenset({False}),
            "draft_token_num": frozenset(
                range(1, FLASHINFER_KDA_MAX_VERIFY_TOKENS + 1)
            ),
            "fused_replay_payload": frozenset({True}),
            "split_producers": frozenset({False}),
            "state_dtype": frozenset({torch.bfloat16}),
            "num_heads": frozenset({12}),
            "head_dim": frozenset({128}),
            "recurrent_layout": frozenset({"v_major"}),
        },
        tags={"nvidia", "paged_cache", "cuda_graph", "speculative"},
    )(flashinfer_kda_producer_verify)
