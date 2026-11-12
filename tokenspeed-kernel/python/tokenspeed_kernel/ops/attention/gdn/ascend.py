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

"""Ascend Gated Delta Net kernel registrations.

The leaves live in ``tokenspeed_kernel_npu.ops.gdn`` and drive CANN's native
Gated Delta Rule ops. What they cover, and what they refuse, is documented
there; this module maps that onto the vendor-neutral ``attention`` family so
``gdn_chunk_prefill`` / ``gdn_decode_step`` / ``gdn_decode_mtp`` select them
on Ascend.

``gdn_chain_verify`` and ``gdn_replay_commit`` are not registered: the native
recurrent op cannot stop a window at an accepted prefix, and ReplaySSM needs
per-layer address tables no Ascend kernel reads.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.gdn import gdn_chunk_prefill as _chunk_prefill
    from tokenspeed_kernel_npu.ops.gdn import gdn_decode_mtp as _decode_mtp
    from tokenspeed_kernel_npu.ops.gdn import gdn_decode_step as _decode_step

    _CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
    # The native ops take bfloat16 activations, which is Qwen3.5/3.6's dtype.
    _DTYPES = {torch.bfloat16}
    # The recurrent pool is bf16 or fp32 (Qwen3.6 selects fp32 through
    # ``--mamba-ssm-dtype``); the native ops accept both.
    _STATE_DTYPES = frozenset({torch.bfloat16, torch.float32})

    @register_kernel(
        "attention",
        "gdn_chunk_prefill",
        name="ascend_gdn_chunk_prefill",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            # No intermediate checkpoints: the native scan returns the final
            # state only.
            "output_h": frozenset({False}),
            "qk_l2norm": frozenset({False, True}),
            "state_checkpoints": frozenset({False}),
        },
        features={"ascend"},
    )
    def ascend_gdn_chunk_prefill(**kwargs):
        result = _chunk_prefill(**kwargs)
        # The vendor-neutral entry returns the structured result type.
        from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult

        out, final_state = result
        return GdnChunkPrefillResult(out=out, final_state=final_state)

    @register_kernel(
        "attention",
        "gdn_decode_step",
        name="ascend_gdn_decode_step",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            # The native op reads a row per token, which rejects q/k/v rows
            # spaced wider than one packed projection row.
            "strided_qkv": frozenset({False}),
            # Negative rows are redirected to the null page, whose zero state
            # they read, and that page is restored after the launch. The
            # kernel therefore serves both callers that require the reserved
            # null page and those that do not.
            "padding_safe": frozenset({True, False}),
            "state_dtype": _STATE_DTYPES,
        },
        features={"ascend"},
    )
    def ascend_gdn_decode_step(**kwargs):
        return _decode_step(**kwargs)

    @register_kernel(
        "attention",
        "gdn_decode_mtp",
        name="ascend_gdn_decode_mtp",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures(("q", "k", "v"), "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            # Draft trees need per-node parent walks the native op lacks.
            "tree": frozenset({False}),
            "padding_safe": frozenset({True, False}),
            # The window stages each position's state in its own row of the
            # caller's `output_state` pool, leaving the committed rows alone.
            "separate_output_state": frozenset({True}),
            # The op names one row per token, so it reads packed projection
            # rows only.
            "strided_qkv": frozenset({False}),
            "state_dtype": _STATE_DTYPES,
        },
        features={"ascend"},
    )
    def ascend_gdn_decode_mtp(**kwargs):
        return _decode_mtp(**kwargs)


__all__: list[str] = []
