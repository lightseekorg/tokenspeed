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
        enabled: Emit lines at all (rank 0 only; other ranks still count).
        decode_log_interval: Rounds between two "Decode batch." lines.
        num_total_pages: Device KV pages, for the active/total page ratio.
        spec_num_steps: Draft steps per verify, 0 when speculation is off.
        spec_num_tokens: Verify width, used by the accept-length debug log.
        token_to_kv_pool: Pool asked to log its per-group page usage
            alongside each decode line.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        decode_log_interval: int,
        num_total_pages: int,
        spec_num_steps: int,
        spec_num_tokens: int,
        token_to_kv_pool,
    ) -> None:
        self._enabled = enabled
        self._decode_log_interval = decode_log_interval
        self._num_total_pages = num_total_pages
        self._spec_num_steps = spec_num_steps
        self._spec_num_tokens = spec_num_tokens
        self._token_to_kv_pool = token_to_kv_pool

        self._step = 0
        self._seen_prefill_ids: set[str] = set()
        # Decode throughput window: committed tokens since the last line.
        self._num_generated_tokens = 0
        self._num_decode_steps = 0
        self._last_decode_tic = time.time()

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
            "%s batch. #new-seq: %s, #new-token: %s, #cached-token: %s, "
            "#running-req: %s, #queue-req: %s",
            mode,
            num_extends,
            total_tokens,
            cached_tokens,
            bs,
            stats["num_queue_reqs"],
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
                "Decode batch. #running-req: %s, "
                "#pages(active/cached/total): %s/%s/%s, "
                "page ratio: %.2f, gen throughput (token/s): %.2f, "
                "avg_accept_len: %.2f, accept_rate: %.2f, #queue-req: %s",
                bs,
                num_active_pages,
                stats["num_cached_pages"],
                self._num_total_pages,
                page_ratio,
                gen_throughput,
                avg_accept,
                (avg_accept - 1) / self._spec_num_steps,
                stats["num_queue_reqs"],
            )
        else:
            logger.info(
                "Decode batch. #running-req: %s, "
                "#pages(active/cached/total): %s/%s/%s, "
                "page ratio: %.2f, gen throughput (token/s): %.2f, "
                "#queue-req: %s",
                bs,
                num_active_pages,
                stats["num_cached_pages"],
                self._num_total_pages,
                page_ratio,
                gen_throughput,
                stats["num_queue_reqs"],
            )
        self._token_to_kv_pool.maybe_log_cache_group_pages()
        self._num_generated_tokens = 0
        self._num_decode_steps = 0
        self._last_decode_tic = now

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
            "Spec verify step. accept_lengths=%s, accepted_draft_tokens=%s",
            accepted_widths,
            [max(0, value - 1) for value in accepted_widths],
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
            "Spec token compare. anchor=%s, draft=%s, target=%s, match=%s",
            candidate_rows[:, 0].tolist(),
            draft_rows.tolist(),
            target_draft_rows.tolist(),
            draft_rows.eq(target_draft_rows).tolist(),
        )
