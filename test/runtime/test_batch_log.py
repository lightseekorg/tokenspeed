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

"""Per-round batch logging (the control-plane "Prefill/Decode batch." lines)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.engine import batch_log as batch_log_module
from tokenspeed.runtime.engine.batch_log import BatchLogger

STATS = {"num_active_pages": 40, "num_cached_pages": 15, "num_queue_reqs": 7}


def _logger(**overrides) -> BatchLogger:
    kwargs = dict(
        enabled=True,
        decode_log_interval=2,
        num_total_pages=100,
        spec_num_steps=0,
        spec_num_tokens=0,
    )
    kwargs.update(overrides)
    return BatchLogger(**kwargs)


def _extend_op(request_ids, num_extends, input_lengths, extend_prefix_lens):
    return SimpleNamespace(
        request_ids=request_ids,
        input_lengths=input_lengths,
        extend_prefix_lens=extend_prefix_lens,
        num_extends=lambda: num_extends,
    )


def _decode_op(bs):
    return SimpleNamespace(
        request_ids=[f"r{i}" for i in range(bs)],
        input_lengths=[1] * bs,
        extend_prefix_lens=[],
        num_extends=lambda: 0,
    )


def test_extend_round_counts_cached_tokens_once_per_request():
    logger = _logger()
    op = _extend_op(["a", "b"], 2, [10, 20], [4, 6])

    with mock.patch.object(batch_log_module.logger, "info") as log:
        logger.log_dispatch(op, STATS)
        # Chunked prefill re-dispatches the same rids; their prefix is not
        # cached-token news a second time.
        logger.log_dispatch(op, STATS)

    assert log.call_args_list[0].args[1:] == ("Prefill", 2, 30, 10, 2, 7)
    assert log.call_args_list[1].args[1:] == ("Prefill", 2, 30, 0, 2, 7)


def test_mixed_round_is_labelled_mix():
    logger = _logger()
    op = _extend_op(["a", "b", "c"], 1, [10, 1, 1], [4])

    with mock.patch.object(batch_log_module.logger, "info") as log:
        logger.log_dispatch(op, STATS)

    assert log.call_args.args[1] == "Mix"


def test_decode_rounds_log_once_per_interval_with_committed_throughput():
    logger = _logger(decode_log_interval=3)

    with mock.patch.object(batch_log_module.logger, "info") as log:
        for _ in range(3):
            logger.record_decode(
                SimpleNamespace(output_lengths=torch.tensor([2, 2])), 2
            )
            logger.log_dispatch(_decode_op(2), STATS)

    # Rounds 1 and 2 are throttled; round 3 prints the window.
    log.assert_called_once()
    args = log.call_args.args
    assert args[1:5] == (2, 40, 15, 100)  # running-req, pages active/cached/total
    assert args[5] == 0.4  # page ratio
    assert args[6] > 0  # gen throughput over the window


def test_state_group_pages_ride_the_decode_line_at_debug():
    """Recurrent/conv state groups are sized apart from the KV groups, so the
    decode line's single page ratio cannot show one of them binding."""
    queried = []

    def pages(group_id):
        queried.append(group_id)
        return {"state_a": (10, 4), "state_b": (8, 8)}[group_id]

    logger = _logger(
        decode_log_interval=1,
        cache_state_group_ids=("state_a", "state_b"),
        cache_group_pages=pages,
    )

    with mock.patch.object(batch_log_module.logger, "isEnabledFor", return_value=True):
        with mock.patch.object(batch_log_module.logger, "debug") as debug:
            logger.log_dispatch(_decode_op(2), STATS)

    assert queried == ["state_a", "state_b"]
    assert debug.call_args.args[1] == (
        "state_a: used=6/10, available=4; state_b: used=0/8, available=8"
    )


def test_a_pool_with_no_state_group_never_queries_the_scheduler():
    def pages(group_id):  # pragma: no cover - must not be reached
        raise AssertionError("queried with no state group")

    logger = _logger(decode_log_interval=1, cache_group_pages=pages)

    with mock.patch.object(batch_log_module.logger, "isEnabledFor", return_value=True):
        with mock.patch.object(batch_log_module.logger, "debug") as debug:
            logger.log_dispatch(_decode_op(2), STATS)

    debug.assert_not_called()


def test_disabled_rank_still_counts_but_never_logs():
    logger = _logger(enabled=False, decode_log_interval=1)

    with mock.patch.object(batch_log_module.logger, "info") as log:
        logger.record_decode(SimpleNamespace(output_lengths=torch.tensor([3])), 1)
        logger.log_dispatch(_decode_op(1), STATS)

    log.assert_not_called()


def test_step_acceptance_log_separates_committed_and_draft_tokens():
    logger = _logger(spec_num_steps=7)
    result = SimpleNamespace(
        output_lengths=torch.tensor([1, 3, 8]),
        spec_candidate_tokens=None,
    )

    with (
        mock.patch.object(batch_log_module, "LOG_SPEC_ACCEPT_LENGTHS", True),
        mock.patch.object(batch_log_module.logger, "info") as log,
    ):
        logger.record_decode(result, bs=3)

    log.assert_called_once_with(
        "Spec verify step. accept_lengths=%s, accepted_draft_tokens=%s",
        [1, 3, 8],
        [0, 2, 7],
    )


def test_step_token_log_aligns_drafts_with_predecessor_target_logits():
    logger = _logger(spec_num_steps=3, spec_num_tokens=4)
    result = SimpleNamespace(
        output_lengths=torch.tensor([3]),
        output_tokens=torch.tensor([11, 12, 99, 100]),
        spec_candidate_tokens=torch.tensor([10, 11, 12, 13]),
    )

    with (
        mock.patch.object(batch_log_module, "LOG_SPEC_ACCEPT_LENGTHS", True),
        mock.patch.object(batch_log_module.logger, "info") as log,
    ):
        logger.record_decode(result, bs=1)

    assert log.call_args_list[1] == mock.call(
        "Spec token compare. anchor=%s, draft=%s, target=%s, match=%s",
        [10],
        [[11, 12, 13]],
        [[11, 12, 99]],
        [[True, True, False]],
    )
