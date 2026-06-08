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

import torch

from tokenspeed.runtime.sampling.logits_layout import resolve_dp_sampling_min_bs


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


@dataclasses.dataclass(frozen=True)
class DpSamplingRuntimeConfig:
    vocab_size: int | None = None
    max_bucket_bs: int | None = None
    min_bs: int | None = None


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


def configure_dp_sampling_runtime(
    *,
    support: DpSamplingSupport,
    model: Any,
    sampling_backend: Any,
    logits_processor: Any,
    runtime_vocab_size: int,
    max_num_seqs: int,
    data_parallel_size: int,
    num_tokens_per_req: int,
    configured_min_bs: int | None,
    device: Any,
) -> DpSamplingRuntimeConfig:
    if not support.enabled:
        return DpSamplingRuntimeConfig()

    lm_head = model.lm_head
    weight = lm_head.weight
    if weight.ndim < 1:
        raise RuntimeError(
            f"dp_sampling LM head weight must be at least 1D, got {weight.ndim}D"
        )
    lm_head_rows = int(weight.shape[0])
    validate_dp_sampling_lm_head_vocab(
        lm_head_rows=lm_head_rows,
        vocab_size=runtime_vocab_size,
        tp_size=logits_processor.tp_size,
        skip_all_gather=logits_processor.skip_all_gather,
        tie_word_embeddings=bool(
            getattr(logits_processor.config, "tie_word_embeddings", False)
        ),
    )
    dp_vocab_size = dp_sampling_comm_vocab_size(
        lm_head_rows=lm_head_rows,
        tp_size=logits_processor.tp_size,
        skip_all_gather=logits_processor.skip_all_gather,
    )
    configure_dp_vocab = getattr(
        sampling_backend, "configure_dp_sampling_vocab_size", None
    )
    if configure_dp_vocab is not None:
        configure_dp_vocab(dp_vocab_size)
    max_bs = max_num_seqs // max(data_parallel_size, 1)
    max_bucket_bs = (
        (max_bs + logits_processor.tp_size - 1) // logits_processor.tp_size
    ) * logits_processor.tp_size
    min_bs = resolve_dp_sampling_min_bs(
        tp_size=logits_processor.tp_size,
        configured_min_bs=configured_min_bs,
    )
    logits_processor.configure_dp_sampling(
        dp_num_tokens_per_req=num_tokens_per_req,
        dp_sampling_min_bs=min_bs,
        max_bucket_bs=max_bucket_bs,
        vocab_size=dp_vocab_size,
        device=device,
    )
    return DpSamplingRuntimeConfig(
        vocab_size=dp_vocab_size,
        max_bucket_bs=max_bucket_bs,
        min_bs=min_bs,
    )


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


def resolve_dp_sampling_vocab_size_update(
    *,
    has_comm: bool,
    current_vocab_size: int,
    requested_vocab_size: int,
    tp_size: int,
    comm_initialized: bool,
) -> int | None:
    if not has_comm:
        return None
    if requested_vocab_size == current_vocab_size:
        return None
    if requested_vocab_size % int(tp_size) != 0:
        raise RuntimeError(
            f"DP sampling vocab_size={requested_vocab_size} must be divisible by "
            f"tp_size={tp_size}"
        )
    if comm_initialized:
        raise RuntimeError("Cannot resize DP sampling comm after use")
    return requested_vocab_size


def slice_dp_vocab_mask(
    vocab_mask: torch.Tensor | None,
    *,
    full_bs: int,
    pad_bs: int,
    num_tokens_per_req: int,
    shard: slice,
) -> torch.Tensor | None:
    if vocab_mask is None:
        return None
    n = num_tokens_per_req
    if pad_bs > full_bs:
        vocab_mask = torch.nn.functional.pad(
            vocab_mask,
            (0, 0, 0, (pad_bs - full_bs) * n),
            value=-1,
        )
    return vocab_mask.view(pad_bs, n, -1)[shard].reshape(-1, vocab_mask.shape[-1])


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
