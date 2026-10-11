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

"""Streaming merges of ``RequestOutputCollector``.

When the consumer of a streamed request falls behind, ``put(stream=True)``
merges the next frames into the pending output. Streamed frames carry only
their own ids and logprobs, so a merge appends them even when they repeat the
pending ones. The raw-token path without ``--stream-output`` sends every id so
far instead, so its ids are replaced.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.io_struct import BatchTokenIDOut  # noqa: E402
from tokenspeed.runtime.engine.logprobs import Logprob  # noqa: E402
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)


def _merge(outputs: list[dict], *, cumulative: bool) -> dict:
    """Put every output with no take in between, then take the merged one."""
    collector = RequestOutputCollector()
    for output in outputs:
        collector.put(output, stream=True, cumulative=cumulative)
    return collector.take()


def _ids(output_ids: list[int]) -> dict:
    return {"text": "", "output_ids": output_ids, "meta_info": {"id": "r"}}


class TestCollectorStreamMerge(unittest.TestCase):
    def test_repeated_token_is_kept(self):
        # Two decode steps that both sampled token 6.
        merged = _merge([_ids([6]), _ids([6])], cumulative=False)
        self.assertEqual(merged["output_ids"], [6, 6])

    def test_repeated_run_is_kept(self):
        steps = [_ids([2]), _ids([2]), _ids([2]), _ids([2]), _ids([8]), _ids([5])]
        merged = _merge(steps, cumulative=False)
        self.assertEqual(merged["output_ids"], [2, 2, 2, 2, 8, 5])
        # Speculative decoding: several ids per step.
        merged = _merge([_ids([4, 4]), _ids([4, 4, 1])], cumulative=False)
        self.assertEqual(merged["output_ids"], [4, 4, 4, 4, 1])

    def test_repeated_logprob_entries_are_kept(self):
        # Streamed logprob lists hold only their own frame's positions too.
        outputs = [
            {
                "text": "",
                "output_ids": [6],
                "meta_info": {
                    "id": "r",
                    "output_token_logprobs": [(0.0, 6, None)],
                    "logprobs": [{6: Logprob(logprob=0.0)}],
                },
            }
            for _ in range(2)
        ]
        meta = _merge(outputs, cumulative=False)["meta_info"]
        self.assertEqual(meta["output_token_logprobs"], [(0.0, 6, None)] * 2)
        self.assertEqual(meta["logprobs"], [{6: Logprob(logprob=0.0)}] * 2)

    def test_cumulative_ids_are_replaced(self):
        merged = _merge([_ids([7]), _ids([7, 7]), _ids([7, 7, 6])], cumulative=True)
        self.assertEqual(merged["output_ids"], [7, 7, 6])


class _DigitTokenizer:
    """Decodes token id n as the text of n."""

    def batch_decode(self, batch: list[list[int]], **kwargs) -> list[str]:
        return ["".join(str(token_id) for token_id in ids) for ids in batch]


class _Request:
    stream = True
    sampling_params = {}
    return_logprob = True
    top_logprobs_num = 0
    token_ids_logprob = None
    return_text_in_logprobs = False


def _step(ids: list[int], logprobs: list[float]) -> BatchTokenIDOut:
    """One decode step of request "r" as the scheduler streams it."""
    return BatchTokenIDOut(
        rids=["r"],
        finished_reasons=[None],
        decoded_texts=[""],
        decode_ids=[ids],
        read_offsets=[0],
        output_ids=[ids],
        output_multi_ids=[[]],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
        no_stop_trim=[False],
        prompt_tokens=[1],
        completion_tokens=[1],
        cached_tokens=[0],
        spec_verify_ct=[],
        input_token_logprobs_val=[[]],
        input_token_logprobs_idx=[[]],
        output_token_logprobs_val=[logprobs],
        output_token_logprobs_idx=[ids],
        input_top_logprobs_val=[],
        input_top_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        input_token_ids_logprobs_val=[],
        input_token_ids_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        output_hidden_states=[],
        batch_accept_draft_tokens=[],
        output_extra_infos=[],
        generated_time=0.0,
    )


def _stream(steps: list[tuple[list[int], list[float]]], *, inline: bool) -> dict:
    """Feed the steps to the output processor with no take in between (the
    consumer is behind), then take the merged output."""
    engine = SimpleNamespace(
        server_args=SimpleNamespace(
            weight_version="test",
            enable_inline_detokenizer=inline,
            skip_tokenizer_init=not inline,
            stream_output=False,
            speculative_algorithm=None,
        ),
        tokenizer=_DigitTokenizer() if inline else None,
        rid_to_state={},
        enable_metrics=False,
        dump_requests_folder=False,
    )
    state = ReqState(
        RequestOutputCollector(), False, asyncio.Event(), _Request(), created_time=0.0
    )
    engine.rid_to_state["r"] = state
    processor = OutputProcessor(engine)
    for ids, logprobs in steps:
        processor.handle_batch_output(_step(ids, logprobs))
    return state.collector.take()


def _logprob_ids(output: dict) -> list[int]:
    return [entry[1] for entry in output["meta_info"]["output_token_logprobs"]]


class TestOutputProcessorStreamMerge(unittest.TestCase):
    def test_inline_detokenizer_keeps_repeated_token(self):
        out = _stream([([6], [0.0]), ([6], [0.0])], inline=True)
        self.assertEqual(out["text"], "66")
        self.assertEqual(out["output_ids"], [6, 6])
        self.assertEqual(_logprob_ids(out), [6, 6])

    def test_raw_token_cumulative_ids_are_not_duplicated(self):
        # Without --stream-output a raw-token frame carries every id so far,
        # while its logprobs still cover only that frame.
        out = _stream([([7], [0.0]), ([7], [0.0]), ([6], [-0.5])], inline=False)
        self.assertEqual(out["output_ids"], [7, 7, 6])
        self.assertEqual(_logprob_ids(out), [7, 7, 6])


if __name__ == "__main__":
    unittest.main(verbosity=2)
