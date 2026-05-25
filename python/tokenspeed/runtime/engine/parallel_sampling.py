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

"""Tokenized-payload prep for parallel-sampling (n>1) fan-out.

The scheduler treats each replica as a standalone rid (the n>1 expansion
happens at the AsyncLLM frontend, not on the scheduler), so there is no
per-replica scheduler state here. What we need to centralize is the
tokenized-payload mutation the fan-out loop performs on every replica:
rid regeneration, the ``max_new_tokens=0`` warmup variant, and the plain
replica copy.

The two functions below are pure (no engine state, no I/O) and are
called from ``AsyncLLM._handle_batch_request`` once for the prefix
warmup and ``parallel_sample_num`` times for the replica expansion.
"""

from __future__ import annotations

import copy

from tokenspeed.runtime.engine.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)


def _own_multimodal_inputs(tokenized_copy) -> None:
    """Give this fan-out copy its own multimodal feature tensors.

    ``MultimodalInputs.publish_shm_features`` rewrites ``item.feature``
    in place from tensor to SHM handle, and consumers later unlink the
    segment. The prefix warmup and each n>1 replica are shallow
    ``copy.copy`` of the same tokenized payload, so without an independent
    copy they would publish only once (the 2nd+ ``publish`` is a no-op
    because the feature is already a handle) and share the SHM segment.
    When the first replica's scheduler step consumes-and-unlinks, the
    later replicas hit ``FileNotFoundError`` on their own attach.
    """
    mm = getattr(tokenized_copy, "multimodal_inputs", None)
    if mm is not None:
        tokenized_copy.multimodal_inputs = copy.deepcopy(mm)


def prepare_prefix_warmup(
    tmp_obj: GenerateReqInput | EmbeddingReqInput,
    tokenized_obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput,
) -> TokenizedGenerateReqInput | TokenizedEmbeddingReqInput:
    """Build the prefix-warmup variant used before parallel-sampling
    fan-out. Mutates ``tmp_obj`` to receive a fresh rid; returns a
    copy of ``tokenized_obj`` with ``max_new_tokens`` forced to 0
    and streaming disabled so the scheduler caches the common
    prefix before the replicas are dispatched.
    """
    tokenized_copy = copy.copy(tokenized_obj)
    _own_multimodal_inputs(tokenized_copy)
    tokenized_copy.rid = tmp_obj.regenerate_rid()
    tokenized_copy.sampling_params = copy.copy(tokenized_copy.sampling_params)
    tokenized_copy.sampling_params.max_new_tokens = 0
    tokenized_copy.stream = False
    return tokenized_copy


def prepare_parallel_sampling_replica(
    tmp_obj: GenerateReqInput | EmbeddingReqInput,
    tokenized_obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput,
) -> TokenizedGenerateReqInput | TokenizedEmbeddingReqInput:
    """Build one tokenized replica for parallel-sampling fan-out.

    Mutates ``tmp_obj`` to receive a fresh rid; returns a copy of
    ``tokenized_obj`` sharing that rid. The rest of the tokenized
    payload (sampling_params, input_ids, etc.) is unchanged because
    the replicas share everything except their request identity.
    """
    tokenized_copy = copy.copy(tokenized_obj)
    _own_multimodal_inputs(tokenized_copy)
    tokenized_copy.rid = tmp_obj.regenerate_rid()
    return tokenized_copy
