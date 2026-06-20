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

"""Output-processing helpers for the async frontend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Streaming merge policy for the logprob meta_info fields. These are the UNION
# of both dialects' per-position list keys; only one dialect is present per
# request (the renderer emits one), so the union is harmless.
#
# Each value is a per-position list that GROWS as frames arrive, so it must be
# appended rather than overwritten -- hence merged by ``_extend_sequence``. That
# helper is dict-safe: its prefix check (``_is_prefix``) compares elements only
# with ``==``, which both ``dict`` and the ``Logprob`` dataclass support, so the
# entries never need to be hashed or ordered.
#
# ``cumulative_logprob`` is a scalar handled separately (see _SUM_META_KEYS):
# under streaming each frame recomputes it from a fresh dict, so each frame's
# value is the sum of only that frame's positions (a delta). Since the
# per-position ``logprobs`` are appended across frames, the scalar must be
# *summed* (not overwritten) to stay consistent with the appended list.
_APPEND_META_KEYS = {
    # vLLM dialect
    "logprobs",
    # SGLang dialect
    "input_token_logprobs",
    "output_token_logprobs",
    "input_top_logprobs",
    "output_top_logprobs",
    "input_token_ids_logprobs",
    "output_token_ids_logprobs",
}

# Scalar logprob metadata accumulated by addition across coalesced frames.
_SUM_META_KEYS = {
    "cumulative_logprob",
}


class RequestOutputCollector:
    """Coalesce pending per-request outputs into a single visible response.

    Streaming merges mutate an owned pending dict in place so that N sequential
    ``put`` calls cost O(total_delta) instead of O(N * total_delta). The first
    merge after a take/reset clones the held output once to detach it from the
    producer's reference; subsequent merges extend the cloned lists directly.
    """

    def __init__(self) -> None:
        self._pending: dict[str, Any] | None = None
        self._pending_owned: bool = False

    def has_pending(self) -> bool:
        return self._pending is not None

    def take(self) -> dict[str, Any] | None:
        output = self._pending
        self._pending = None
        self._pending_owned = False
        return output

    def put(self, output: dict[str, Any], *, stream: bool) -> None:
        if self._pending is None or not stream:
            self._pending = output
            self._pending_owned = False
            return
        if not self._pending_owned:
            self._pending = self._clone_for_merge(self._pending)
            self._pending_owned = True
        self._merge_into_pending(output)

    def _merge_into_pending(self, output: dict[str, Any]) -> None:
        pending = self._pending
        assert pending is not None  # guarded by put()
        pending_kind = self._output_kind(pending)
        output_kind = self._output_kind(output)
        if pending_kind != output_kind:
            raise ValueError(
                f"Cannot merge different output kinds: {pending_kind} vs {output_kind}"
            )

        if output_kind == "embedding":
            # Embedding outputs are latest-wins; drop the owned pending.
            self._pending = output
            self._pending_owned = False
            return

        pending_meta = pending.setdefault("meta_info", {})
        self._merge_meta_info_into(pending_meta, output.get("meta_info") or {})

        if output_kind == "text" and "text" in output:
            pending["text"] = output["text"]

        self._extend_sequence(pending, "output_ids", output.get("output_ids"))

        if "output_multi_ids" in pending or "output_multi_ids" in output:
            self._extend_sequence(
                pending, "output_multi_ids", output.get("output_multi_ids")
            )

        if "output_extra_info" in output:
            pending["output_extra_info"] = output["output_extra_info"]

    def _merge_meta_info_into(
        self, pending: dict[str, Any], output: dict[str, Any]
    ) -> None:
        for key, value in output.items():
            if key == "id":
                existing = pending.get("id")
                if existing is not None and existing != value:
                    raise ValueError(
                        f"Cannot merge outputs for different request ids: "
                        f"{existing} vs {value}"
                    )
                pending["id"] = value
                continue
            if key in _APPEND_META_KEYS:
                self._extend_sequence(pending, key, value)
                continue
            if key in _SUM_META_KEYS:
                if value is not None:
                    pending[key] = (pending.get(key) or 0.0) + value
                continue
            pending[key] = value

    def _extend_sequence(self, container: dict[str, Any], key: str, value: Any) -> None:
        if value is None:
            return
        existing = container.get(key)
        if existing is None:
            # Adopt a fresh owned copy so later extends stay private to us.
            container[key] = list(value)
            return
        if not value:
            # Follow-up empty list: preserve already-populated values
            # (input-logprob producers emit once, then send empty
            # lists on subsequent frames).
            return
        if not existing:
            container[key] = list(value)
            return
        if self._is_prefix(existing, value):
            # Cumulative producer: extend with just the tail of `value`.
            existing.extend(value[len(existing) :])
        else:
            existing.extend(value)

    def _clone_for_merge(self, pending: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = dict(pending)
        meta = pending.get("meta_info")
        if isinstance(meta, dict):
            cloned_meta = dict(meta)
            for key in _APPEND_META_KEYS:
                seq = cloned_meta.get(key)
                if isinstance(seq, list):
                    cloned_meta[key] = list(seq)
            cloned["meta_info"] = cloned_meta
        for key in ("output_ids", "output_multi_ids"):
            seq = cloned.get(key)
            if isinstance(seq, list):
                cloned[key] = list(seq)
        return cloned

    def _output_kind(self, output: dict[str, Any]) -> str:
        if "embedding" in output:
            return "embedding"
        if "text" in output:
            return "text"
        return "tokens"

    def _is_prefix(self, pending: Sequence[Any], output: Sequence[Any]) -> bool:
        pending_len = len(pending)
        if pending_len > len(output):
            return False
        for index in range(pending_len):
            if output[index] != pending[index]:
                return False
        return True
