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

"""Incremental detokenization state machine and helpers.

This module hosts the pure state machine used by AsyncLLM's inline
detokenizer path. Everything here is tokenizer-agnostic — callers
pass a HuggingFace-shaped tokenizer (with a ``batch_decode`` method)
plus a ``BatchTokenIDOut`` and a mutable ``decode_status`` dict. The
state machine mutates ``decode_status`` in place and returns the
per-request incremental output strings to emit.

The per-request ``IncrementalDetokenizer`` class wraps a single
``DecodeStatus`` and is the preferred entry point for AsyncLLM; the
batch function ``incremental_decode_batch`` remains as the test
harness driver (``test/runtime/test_detokenizer_parity.py``).
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict, defaultdict
from typing import Any

from tokenspeed.runtime.engine.io_struct import BatchTokenIDOut
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.text import find_printable_text

# Maximum number of request states that the detokenizer can hold.
# When exceeded, the oldest entries are evicted. Default: 65536 (1<<16).
DETOKENIZER_MAX_STATES = envs.TOKENSPEED_DETOKENIZER_MAX_STATES.get()


@dataclasses.dataclass
class DecodeStatus:
    """Per-request incremental decoding state."""

    decoded_text: str
    decode_ids: list[int]
    surr_offset: int
    read_offset: int
    # Offset into ``decoded_text`` that has already been streamed to
    # the consumer; the next call emits ``output_str[sent_offset:]``.
    sent_offset: int = 0


class LimitedCapacityDict(OrderedDict):
    """FIFO-evicting ordered dict used as the detokenizer's request table.

    Only inserting a *new* key at capacity triggers eviction — updating an
    existing key is a size-preserving operation and must never drop the
    oldest entry. Production detokenizer code writes `self.decode_status[rid]
    = s` only on the new-request path, so this guard is defensive for any
    future caller that uses the dict for updates.
    """

    def __init__(self, capacity: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key: Any, value: Any) -> None:
        if key not in self and len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        super().__setitem__(key, value)


def trim_matched_stop(
    output: str | list[int],
    finished_reason: dict[str, Any],
    no_stop_trim: bool,
) -> str | list[int]:
    """Trim a matched stop string or drop a matched stop token.

    If ``no_stop_trim`` is set or ``finished_reason`` is falsy, the
    output is returned unchanged. Otherwise:
    - When ``matched`` is a ``str`` and ``output`` is also a ``str``,
      the output is truncated at the first occurrence of the stop
      string.
    - When ``matched`` is an ``int`` and ``output`` is a ``list``
      (the raw-token id path), the last id is dropped.
    Any other shape combination returns ``output`` unchanged.
    """
    if no_stop_trim or not finished_reason:
        return output

    matched = finished_reason.get("matched", None)
    if not matched:
        return output

    # Trim stop str.
    if isinstance(matched, str) and isinstance(output, str):
        pos = output.find(matched)
        return output[:pos] if pos != -1 else output

    # Trim stop token.
    if isinstance(matched, int) and isinstance(output, list):
        assert len(output) > 0
        return output[:-1]
    return output


def decode_grouped_batch(
    tokenizer: Any, ids: list[list[int]], recv_obj: BatchTokenIDOut
) -> list[str]:
    """Batch-decode requests that disagree on skip/spaces settings.

    Groups requests by ``(skip_special_tokens, spaces_between_special_tokens)``
    so each group can go through a single ``tokenizer.batch_decode``
    call with the correct kwargs, then scatters the results back into
    their original positions.
    """
    groups: dict[Any, list[Any]] = defaultdict(list)
    for i, id in enumerate(ids):
        key = (
            recv_obj.skip_special_tokens[i],
            recv_obj.spaces_between_special_tokens[i],
        )
        groups[key].append((i, id))

    texts: list[Any] = [None] * len(ids)

    for (skip, spaces), items in groups.items():
        indices, group_ids = zip(*items)
        decoded_batch = tokenizer.batch_decode(
            group_ids,
            skip_special_tokens=skip,
            spaces_between_special_tokens=spaces,
        )
        for idx, text in zip(indices, decoded_batch):
            texts[idx] = text

    return texts


def incremental_decode_batch(
    tokenizer: Any,
    decode_status: dict[str, DecodeStatus],
    recv_obj: BatchTokenIDOut,
) -> list[str]:
    """Run the incremental detokenizer state machine on a single batch.

    Mutates ``decode_status`` in place: each request's DecodeStatus is
    either freshly created or has its decode_ids extended, offsets
    advanced, and decoded_text committed. Returns the list of
    incremental output strings to emit (one per request in the batch).

    Raises RuntimeError if a request disappears from ``decode_status``
    mid-call, which happens when the capacity-limited dict evicts an
    earlier rid during a later rid's assignment in the first loop.
    """
    bs = len(recv_obj.rids)

    # Initialize decode status for each request and prepare the
    # surr_ids / read_ids slices the tokenizer will decode.
    read_ids, surr_ids = [], []
    for i in range(bs):
        rid = recv_obj.rids[i]
        if rid not in decode_status:
            s = DecodeStatus(
                decoded_text=recv_obj.decoded_texts[i],
                decode_ids=recv_obj.decode_ids[i],
                surr_offset=0,
                read_offset=recv_obj.read_offsets[i],
            )
            decode_status[rid] = s
        else:
            s = decode_status[rid]
            s.decode_ids.extend(recv_obj.decode_ids[i])

        read_ids.append(
            trim_matched_stop(
                s.decode_ids[s.surr_offset :],
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
        )
        surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

    all_same = (len(set(recv_obj.skip_special_tokens)) <= 1) and (
        len(set(recv_obj.spaces_between_special_tokens)) <= 1
    )
    if all_same:
        surr_texts = tokenizer.batch_decode(
            surr_ids,
            skip_special_tokens=recv_obj.skip_special_tokens[0],
            spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
        )
        read_texts = tokenizer.batch_decode(
            read_ids,
            skip_special_tokens=recv_obj.skip_special_tokens[0],
            spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
        )
    else:
        surr_texts = decode_grouped_batch(tokenizer, surr_ids, recv_obj)
        read_texts = decode_grouped_batch(tokenizer, read_ids, recv_obj)

    # Incremental decoding
    output_strs: list[str] = []
    for i in range(bs):
        try:
            s = decode_status[recv_obj.rids[i]]
        except KeyError:
            raise RuntimeError(
                f"Decode status not found for request {recv_obj.rids[i]}. "
                "It may be due to the request being evicted from the decode status due to memory pressure. "
                "Please increase the maximum number of requests by setting "
                "the TOKENSPEED_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                f"The current value is {DETOKENIZER_MAX_STATES}."
            )
        new_text = read_texts[i][len(surr_texts[i]) :]
        if recv_obj.finished_reasons[i] is None:
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):
                s.decoded_text = s.decoded_text + new_text
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decode_ids)
                new_text = ""
            else:
                new_text = find_printable_text(new_text)

        output_str = trim_matched_stop(
            s.decoded_text + new_text,
            recv_obj.finished_reasons[i],
            recv_obj.no_stop_trim[i],
        )
        # Incrementally send text.
        incremental_output = output_str[s.sent_offset :]
        s.sent_offset = len(output_str)
        output_strs.append(incremental_output)

    return output_strs


class IncrementalDetokenizer:
    """Per-request incremental detokenizer wrapping a single ``DecodeStatus``.

    Each instance owns a per-request slice of the state machine that
    ``incremental_decode_batch`` runs across an entire batch. The
    semantics are byte-for-byte identical to the per-i inner loop of
    the batch function for a single-request batch — ``process`` is just
    a stateful facade for call sites where one-request-at-a-time
    processing is more natural than a shared ``decode_status`` dict.

    Stop authority stays with the scheduler. The ``process`` method
    does not return a matched stop string or invent finish reasons —
    it only consumes ``finished_reason`` as an input flag exactly
    like the batch function does.
    """

    def __init__(self, decoded_text: str = "", read_offset: int = 0) -> None:
        self._status = DecodeStatus(
            decoded_text=decoded_text,
            decode_ids=[],
            surr_offset=0,
            read_offset=read_offset,
        )

    @property
    def status(self) -> DecodeStatus:
        """Expose the underlying DecodeStatus for cross-checks / telemetry.

        The returned object is the live mutable state, not a copy.
        Callers must not mutate it directly — use ``process`` to advance
        the state machine.
        """
        return self._status

    def process(
        self,
        tokenizer: Any,
        *,
        new_decode_ids: list[int],
        finished_reason: dict[str, Any] | None = None,
        no_stop_trim: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> str:
        """Process one frame for this request and return the incremental emit.

        Mutates ``self.status`` in place. Semantically equivalent to one
        iteration of the per-i loop in ``incremental_decode_batch`` for a
        single-request batch: extend decode_ids with the delta, build
        surr_ids/read_ids slices, batch_decode both (single-element
        batch), run the partial-UTF-8 deferral / commit machinery, then
        emit ``output_str[sent_offset:]``.
        """
        s = self._status
        s.decode_ids.extend(new_decode_ids)

        read_ids = trim_matched_stop(
            s.decode_ids[s.surr_offset :],
            finished_reason,
            no_stop_trim,
        )
        surr_ids = s.decode_ids[s.surr_offset : s.read_offset]

        surr_texts = tokenizer.batch_decode(
            [surr_ids],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        read_texts = tokenizer.batch_decode(
            [read_ids],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

        new_text = read_texts[0][len(surr_texts[0]) :]
        if finished_reason is None:
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):
                s.decoded_text = s.decoded_text + new_text
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decode_ids)
                new_text = ""
            else:
                new_text = find_printable_text(new_text)

        output_str = trim_matched_stop(
            s.decoded_text + new_text,
            finished_reason,
            no_stop_trim,
        )
        # Incrementally send text.
        incremental_output = output_str[s.sent_offset :]
        s.sent_offset = len(output_str)
        return incremental_output
