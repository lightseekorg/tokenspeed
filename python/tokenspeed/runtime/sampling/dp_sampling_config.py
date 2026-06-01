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

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class DpSamplingSupport:
    requested: bool
    enabled: bool
    infra_supports: bool
    drafter_available: bool
    backend_supports_verify: bool
    tp_size: int
    tp_group_set: bool

    def unsupported_message(self) -> str:
        return (
            "--dp-sampling was set but Batch-DP spec-verify "
            "preconditions are not met: "
            f"drafter={self.drafter_available}, "
            f"backend_supports_dp_verify={self.backend_supports_verify}, "
            f"processor.tp_size={self.tp_size}, "
            f"processor.tp_group_set={self.tp_group_set}"
        )


def resolve_dp_sampling_support(
    *,
    requested: bool,
    drafter: Any,
    sampling_backend: Any,
    logits_processor: Any,
) -> DpSamplingSupport:
    backend_supports_verify = bool(
        getattr(sampling_backend, "_SUPPORTS_DP_VERIFY", False)
    )
    drafter_available = drafter is not None
    tp_size = int(logits_processor.tp_size)
    tp_group_set = logits_processor.tp_group is not None
    infra_supports = (
        drafter_available and backend_supports_verify and tp_size > 1 and tp_group_set
    )
    support = DpSamplingSupport(
        requested=bool(requested),
        enabled=infra_supports and bool(requested),
        infra_supports=infra_supports,
        drafter_available=drafter_available,
        backend_supports_verify=backend_supports_verify,
        tp_size=tp_size,
        tp_group_set=tp_group_set,
    )
    if support.requested and not support.infra_supports:
        raise RuntimeError(support.unsupported_message())
    return support


def dp_sampling_comm_vocab_size(
    *,
    lm_head_rows: int,
    tp_size: int,
    skip_all_gather: bool,
) -> int:
    vocab_size = int(lm_head_rows)
    if not skip_all_gather:
        vocab_size *= int(tp_size)
    return ((vocab_size + int(tp_size) - 1) // int(tp_size)) * int(tp_size)


def validate_dp_sampling_lm_head_vocab(
    *,
    lm_head_rows: int,
    vocab_size: int,
    tp_size: int,
    skip_all_gather: bool,
    tie_word_embeddings: bool,
) -> None:
    if skip_all_gather and int(lm_head_rows) < int(vocab_size):
        raise RuntimeError(
            "Batch-DP sampling with skip_all_gather requires a replicated/"
            "full-vocab LM head. Got a sharded LM head with "
            f"lm_head_rows={lm_head_rows}, vocab_size={vocab_size}, "
            f"tp_size={tp_size}, tie_word_embeddings={tie_word_embeddings}. "
            "Disable --dp-sampling or use a model path that resolves a "
            "replicated LM head."
        )
