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

"""The per-round operator log lines ("Prefill batch." / "Decode batch.").

These describe a SCHEDULER round -- what was admitted, how deep the queue
is, how full the KV pool is, how fast committed tokens are coming out --
so they belong to the control plane, where all of those numbers already
live. Logging them from the executor instead meant threading page and
queue counts down through the dispatch path and across the forward thread
purely to reach a ``logger.info``, and left the decode counters written on
the control plane (at commit) but read on the forward thread (at execute).

Here, the loop dispatches, commits, and logs in one place on one thread:
``log_dispatch`` prints the round, ``record_decode`` folds committed
results into the throughput window, and the executor stays free of
scheduler arguments.
"""

from __future__ import annotations

import logging
import time

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs

logger = get_colorful_logger(__name__)

LOG_SPEC_ACCEPT_LENGTHS = envs.TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS.get()

# The prefill line reports first-touch cached tokens, so it remembers which
# requests it has already counted. Bound the growth: this is log dedup only.
MAX_SEEN_PREFILL_IDS = 100_000


class BatchLogger:
    """Per-round batch logging for one rank's event loop.

    Args:
        enabled: Emit lines on this scheduler's representative rank; other
            ranks still update counters.
        decode_log_interval: Rounds between two "Decode batch." lines.
        num_total_pages: Device KV pages, for the active/total page ratio.
        spec_num_steps: Draft steps per verify, 0 when speculation is off.
        spec_num_tokens: Verify width, used by the accept-length debug log.
        cache_state_group_ids: State-family cache-group ids, appended to
            each decode line at DEBUG. Empty for pools with no such group.
        cache_group_pages: ``group_id -> (total, available)`` page counts,
            read from the C++ scheduler. None disables the extra line.
        dp_rank: Attention-DP scheduler coordinate printed on every line;
            each DP rank runs its own scheduler, so their lines differ.
        pd_lifecycle: Query returning the PD request lifecycle counts
            ``(bootstrapping, prefilling, remote_prefilling, decoding,
            pd_pinned)`` read from the C++ scheduler, appended to each batch
            line. None on a fused engine, where every count but prefilling
            and decoding is zero by construction.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        decode_log_interval: int,
        num_total_pages: int,
        spec_num_steps: int,
        spec_num_tokens: int,
        dp_rank: int,
        pd_lifecycle,
        cache_state_group_ids=(),
        cache_group_pages=None,
    ) -> None:
        self._enabled = enabled
        self._decode_log_interval = decode_log_interval
        self._num_total_pages = num_total_pages
        self._spec_num_steps = spec_num_steps
        self._spec_num_tokens = spec_num_tokens
        self._cache_state_group_ids = tuple(cache_state_group_ids)
        self._cache_group_pages = cache_group_pages
        self._dp_rank = dp_rank
        self._pd_lifecycle = pd_lifecycle

        self._step = 0
        self._seen_prefill_ids: set[str] = set()
        # Decode throughput window: committed tokens since the last line.
        self._num_generated_tokens = 0
        self._num_decode_steps = 0
        self._last_decode_tic = time.time()

    def _lifecycle_suffix(self) -> str:
        """The PD request-state counts, read only when a line is emitted."""
        if self._pd_lifecycle is None:
            return ""
        counts = self._pd_lifecycle()
        return (
            ", #req-state(bootstrap/prefill/remote-prefill/decode/pd-pinned): "
            + "/".join(str(count) for count in counts)
        )

    def log_dispatch(self, forward_op, stats: dict) -> None:
        """Log the round being dispatched; ``stats`` is its scheduler sample.

        Extend rounds log every time (they are rare and each one matters);
        decode rounds log once per ``decode_log_interval`` rounds, with the
        throughput accumulated by ``record_decode`` since the last line.
        """
        self._step += 1
        if not self._enabled:
            return
        num_extends = forward_op.num_extends()
        bs = len(forward_op.request_ids)
        if num_extends > 0:
            self._log_extend(forward_op, num_extends, bs, stats)
        elif self._step % self._decode_log_interval == 0:
            self._log_decode(bs, stats)

    def _log_extend(self, forward_op, num_extends: int, bs: int, stats: dict) -> None:
        mode = "Prefill" if num_extends == bs else "Mix"
        total_tokens = sum(forward_op.input_lengths)
        cached_tokens = sum(
            prefix_len
            for rid, prefix_len in zip(
                forward_op.request_ids[:num_extends],
                forward_op.extend_prefix_lens,
            )
            if rid not in self._seen_prefill_ids
        )
        if len(self._seen_prefill_ids) > MAX_SEEN_PREFILL_IDS:
            self._seen_prefill_ids.clear()
        self._seen_prefill_ids.update(forward_op.request_ids[:num_extends])
        logger.info(
            f"{mode!s} batch. #dp-rank: {self._dp_rank!s}, "
            f"#new-seq: {num_extends!s}, #new-token: {total_tokens!s}, "
            f"#cached-token: {cached_tokens!s}, "
            f"#running-req: {bs!s}, #queue-req: {stats['num_queue_reqs']!s}"
            f"{self._lifecycle_suffix()!s}",
        )

    def _log_decode(self, bs: int, stats: dict) -> None:
        now = time.time()
        gap = now - self._last_decode_tic
        gen_throughput = self._num_generated_tokens / gap if gap > 0 else 0
        avg_accept = (
            self._num_generated_tokens / self._num_decode_steps
            if self._num_decode_steps > 0
            else 0
        )
        num_active_pages = stats["num_active_pages"]
        page_ratio = (
            num_active_pages / self._num_total_pages if self._num_total_pages > 0 else 0
        )
        if self._spec_num_steps:
            logger.info(
                f"Decode batch. #dp-rank: {self._dp_rank!s}, #running-req: {bs!s}, "
                f"#pages(active/cached/total): {num_active_pages!s}/"
                f"{stats['num_cached_pages']!s}/{self._num_total_pages!s}, "
                f"page ratio: {page_ratio:.2f}, gen throughput (token/s): "
                f"{gen_throughput:.2f}, "
                f"avg_accept_len: {avg_accept:.2f}, accept_rate: "
                f"{(avg_accept - 1) / self._spec_num_steps:.2f}, #queue-req: "
                f"{stats['num_queue_reqs']!s}{self._lifecycle_suffix()!s}",
            )
        else:
            logger.info(
                f"Decode batch. #dp-rank: {self._dp_rank!s}, #running-req: {bs!s}, "
                f"#pages(active/cached/total): {num_active_pages!s}/"
                f"{stats['num_cached_pages']!s}/{self._num_total_pages!s}, "
                f"page ratio: {page_ratio:.2f}, gen throughput (token/s): "
                f"{gen_throughput:.2f}, "
                f"#queue-req: {stats['num_queue_reqs']!s}{self._lifecycle_suffix()!s}",
            )
        self._log_cache_state_group_pages()
        self._num_generated_tokens = 0
        self._num_decode_steps = 0
        self._last_decode_tic = now

    def _log_cache_state_group_pages(self) -> None:
        """Append per-group page usage for state-family groups, at DEBUG.

        Recurrent/conv state groups are sized separately from the KV groups,
        so the decode line's single page ratio does not show when one of them
        is the binding constraint. Pure scheduler-counter reads.
        """
        if self._cache_group_pages is None or not self._cache_state_group_ids:
            return
        if not logger.isEnabledFor(logging.DEBUG):
            return
        parts = []
        for group_id in self._cache_state_group_ids:
            total, available = self._cache_group_pages(group_id)
            parts.append(
                f"{group_id}: used={total - available}/{total}, available={available}"
            )
        logger.debug(
            f"Cache state group pages. #dp-rank: {self._dp_rank!s}, "
            f"{'; '.join(parts)!s}"
        )

    def record_decode(self, results, bs: int) -> None:
        """Fold one committed decode step into the throughput window.

        Reads host tensors of an already-synced result — no GPU sync.
        """
        accept_lengths = results.output_lengths
        self._num_generated_tokens += int(accept_lengths.sum().item())
        self._num_decode_steps += bs
        if not (LOG_SPEC_ACCEPT_LENGTHS and self._enabled and self._spec_num_steps):
            return
        accepted_widths = [int(value) for value in accept_lengths.tolist()]
        logger.info(
            f"Spec verify step. #dp-rank: {self._dp_rank!s}, "
            f"accept_lengths={accepted_widths!s}, "
            "accepted_draft_tokens="
            f"{[max(0, value - 1) for value in accepted_widths]!s}",
        )
        candidates = results.spec_candidate_tokens
        if candidates is None:
            return
        verify_width = int(self._spec_num_tokens)
        candidate_rows = candidates.view(bs, verify_width)
        target_rows = results.output_tokens.view(bs, verify_width)
        # Candidate column j+1 is verified by the target token sampled from
        # column j. The final target column is the bonus token.
        draft_rows = candidate_rows[:, 1:]
        target_draft_rows = target_rows[:, :-1]
        logger.info(
            f"Spec token compare. #dp-rank: {self._dp_rank!s}, "
            f"anchor={candidate_rows[:, 0].tolist()!s}, draft="
            f"{draft_rows.tolist()!s}, target={target_draft_rows.tolist()!s}, match="
            f"{draft_rows.eq(target_draft_rows).tolist()!s}",
        )
