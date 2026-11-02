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

"""The prefill/decode interval's TTFT bound, driven against the real C++
scheduler built through ``make_config`` as the event loop builds it.

Prompts of one to five chunks arrive while others decode. With room for every
request, a round that runs decodes alone while a prompt waits is a round the
interval held it back, and no prompt may be held back more than N such rounds
in total, however many chunks it has (docs/design/scheduler.md §3.3).
"""

from __future__ import annotations

import random
from test.ci_system.ci_register import register_cuda_ci

import pytest

register_cuda_ci(est_time=5, suite="runtime-1gpu")

ts = pytest.importorskip("tokenspeed_scheduler")

from test.runtime.scheduler_driver import (  # noqa: E402
    advance_scheduler,
    make_extend_result_event,
    make_finish_event,
)

from tokenspeed.runtime.engine.scheduler_utils import make_config  # noqa: E402

PAGE = 2
CHUNK = 8  # tokens per prefill round
INTERVAL = 3
PROMPTS = 40
MAX_PROMPT = 5 * CHUNK
MAX_NEW_TOKENS = 64
# Room for every prompt at once, so no round waits for pages or a slot and
# every decode-only round while a prompt waits is the interval's doing.
PAGES = PROMPTS * (MAX_PROMPT + MAX_NEW_TOKENS) // PAGE + 1


def _scheduler() -> ts.Scheduler:
    return ts.Scheduler(
        make_config(
            num_device_pages=PAGES,
            max_scheduled_tokens=CHUNK,
            max_batch_size=PROMPTS,
            prefix_granularity=PAGE,
            num_host_pages=0,
            disable_l2_cache=True,
            enable_l3_storage=False,
            role="fused",
            num_snapshot_pages=1,
            host_group_packing=(),
            max_retracted_requests=0,
            host_write_policy="write_through",
            prefill_decode_interval=INTERVAL,
            storage_config=None,
            cache_groups=[
                ts.CacheGroupConfig(
                    group_id="history",
                    block_granularity=PAGE,
                    total_pages=PAGES,
                    attn_kind=ts.AttnKind.Full,
                )
            ],
        )
    )


def test_no_prompt_is_held_back_more_than_the_interval():
    rng = random.Random(0)
    scheduler = _scheduler()
    prompt_left: dict[str, int] = {}  # prompt tokens not yet prefilled
    decodes_left: dict[str, int] = {}  # decode rounds until the finish
    held_back: dict[str, int] = {}
    chunks: dict[str, int] = {}
    num_prompts = 0
    for _ in range(2000):
        if num_prompts < PROMPTS and rng.random() < 0.15:
            rid = f"r{num_prompts}"
            length = rng.randint(1, MAX_PROMPT)
            spec = ts.RequestSpec()
            spec.request_id = rid
            spec.tokens = [1000 * num_prompts + i for i in range(length)]
            spec.max_new_tokens = MAX_NEW_TOKENS
            scheduler.submit_requests([spec])
            prompt_left[rid] = length
            decodes_left[rid] = rng.randint(4, 40)
            held_back[rid] = chunks[rid] = 0
            num_prompts += 1
        plan = scheduler.next_execution_plan()
        forwards = [op for op in plan.forward if op.request_ids]
        if not forwards:
            if num_prompts == PROMPTS and not prompt_left and not decodes_left:
                break
            continue
        (op,) = forwards
        ids = list(op.request_ids)
        if op.num_extends() == 0:
            # Decodes alone: every prompt still waiting was held back.
            for rid, left in prompt_left.items():
                if left > 0:
                    held_back[rid] += 1
        events = []
        for row, rid in enumerate(ids):
            if row < op.num_extends():
                prompt_left[rid] -= op.input_lengths[row]
                chunks[rid] += 1
                if prompt_left[rid] > 0:
                    events.append(make_extend_result_event(rid, []))
                    continue
                del prompt_left[rid]
            events.append(make_extend_result_event(rid, [7]))
            if row >= op.num_extends():
                decodes_left[rid] -= 1
        advance_scheduler(scheduler, events)
        finished = [rid for rid in ids if decodes_left.get(rid) == 0]
        for rid in finished:
            del decodes_left[rid]
        if finished:
            advance_scheduler(scheduler, [make_finish_event(r) for r in finished])
    assert num_prompts == PROMPTS and not prompt_left and not decodes_left
    assert max(held_back.values()) <= INTERVAL, held_back
    # The bound was exercised: some prompt was held back the whole interval,
    # and some prompt of several chunks was held back at all.
    assert max(held_back.values()) == INTERVAL
    assert any(held_back[r] > 0 and chunks[r] > 1 for r in held_back)
