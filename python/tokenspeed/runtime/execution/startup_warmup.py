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

"""Startup coverage of the request path against the final cache allocation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tokenspeed_kernel.ops.tuning import autotune, kernel_warmup
from tokenspeed_scheduler import (
    ExecutionEvent,
    ForwardEvent,
    RequestSpec,
    Scheduler,
)

from tokenspeed.runtime.execution.types import PlannedForward, StopCriteria
from tokenspeed.runtime.sampling.sampling_params import SamplingParams


def warmup_request_path(
    device,
    scheduler_config,
    *,
    phases: Sequence[tuple[int, int]],
    decode_width: int,
    decode_steps: int,
) -> int:
    """Run bounded private startup batches before admitting serving requests.

    A fresh scheduler owns each phase's temporary pages and request slots.
    The live scheduler is not constructed or modified here. The ordinary
    device path clears those pages, prepares inputs, extends cached prefixes,
    verifies drafts and commits accepted state. This also reaches kernels
    whose pointer-range specialization changed when the probe cache was
    replaced, and bookkeeping that model-only warmup never calls.

    Args:
        device: The final DeviceHandle, before it is given to the serving loop.
        scheduler_config: Its plain-engine scheduler geometry, with no host store.
        phases: Explicit (batch size, prompt length) startup cases.
        decode_width: Fixed number of candidate inputs per decode.
        decode_steps: Decode rounds to complete at each case's full batch size.

    Returns:
        Number of actual forwards completed. No outputs leave this function.
    """
    from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan

    if decode_width < 1 or decode_steps < 1:
        raise ValueError("startup decode width and steps must be positive")
    forwards = 0
    with (
        torch.no_grad(),
        kernel_warmup(enabled=True),
        autotune(tune_mode=False, tuning_buckets=None, round_up=None),
    ):
        for phase, (batch_size, prompt_length) in enumerate(phases):
            if batch_size < 1 or prompt_length < 1:
                raise ValueError("startup batches and prompt lengths must be positive")
            scheduler = Scheduler(scheduler_config)
            requests = []
            criteria = {}
            # Leave room for partial acceptance and the pending draft round.
            output_budget = (decode_steps + 2) * decode_width + 1
            for row in range(batch_size):
                request = RequestSpec()
                request.request_id = f"__startup_{phase}_{row}"
                # Distinct prefixes prevent prefix sharing from erasing the
                # input/cache work this phase is meant to prepare.
                request.tokens = [row + 1] * prompt_length
                requests.append(request)
                criteria[request.request_id] = StopCriteria(
                    stop_token_ids=(), token_limit=prompt_length + output_budget
                )
            scheduler.submit_requests(requests)
            # A normalized greedy request; unlike temperature=0 it needs no
            # frontend normalization before the device's sampling preparation.
            sampling = SamplingParams(
                temperature=1.0,
                top_k=1,
                max_new_tokens=output_budget,
                ignore_eos=True,
            )
            completed_decodes = 0
            # Full chunks, admission transitions and a few accepted rounds.
            limit = (
                batch_size * prompt_length + scheduler_config.max_scheduled_tokens - 1
            ) // scheduler_config.max_scheduled_tokens + 4 * (
                batch_size + decode_steps + 1
            )
            try:
                for _ in range(limit):
                    plan = scheduler.next_execution_plan()
                    cache_ops = cache_ops_from_plan(plan)
                    if cache_ops.storage or cache_ops.device:
                        raise RuntimeError(
                            "startup rehearsal requires no host-cache transfers"
                        )
                    batches = plan.forward
                    if len(batches) > 1:
                        raise RuntimeError(
                            "startup rehearsal requires one local forward"
                        )
                    planned = None
                    if batches:
                        batch = batches[0]
                        planned = PlannedForward(
                            forward_op=batch,
                            sampling_params_list=[sampling] * len(batch.request_ids),
                            dp_metadata=None,
                            grammar_inputs=None,
                            multimodal_context=None,
                            ngram_inputs=None,
                            request_history_seeds=None,
                            input_logprob_plan=None,
                            sampled_top_k=0,
                            stop_criteria=tuple(
                                criteria[rid] for rid in batch.request_ids
                            ),
                        )
                    result = device.execute(plan, planned, cache_ops=())
                    if result is None:
                        continue
                    # Startup alone waits synchronously. Serving keeps its
                    # normal asynchronous device/commit protocol.
                    result.sync()
                    forwards += 1
                    lengths = result.output_lengths.tolist()
                    tokens = result.output_tokens.tolist()
                    if len(lengths) != len(batch.request_ids):
                        raise RuntimeError("startup result lost a request row")
                    if result.output_nan_flags is not None and any(
                        result.output_nan_flags.tolist()
                    ):
                        raise RuntimeError(
                            "startup rehearsal produced a nonfinite output"
                        )
                    events = ExecutionEvent()
                    offset = 0
                    extends = batch.num_extends()
                    for row, (rid, count) in enumerate(
                        zip(batch.request_ids, lengths, strict=True)
                    ):
                        count = max(0, count)
                        values = tokens[offset : offset + count]
                        if len(values) != count:
                            raise RuntimeError("startup result lost sampled tokens")
                        offset += count if row < extends else decode_width
                        middle = row < extends and (
                            batch.extend_prefix_lens[row] + batch.input_lengths[row]
                            < batch.prefill_lengths[row]
                        )
                        event = ForwardEvent.ExtendResult()
                        event.request_id = rid
                        event.tokens = [] if middle else values
                        events.add_event(event)
                        if row >= extends:
                            reserve = ForwardEvent.UpdateReserveNumTokens()
                            reserve.request_id = rid
                            reserve.reserve_num_tokens_in_next_schedule_event = count
                            events.add_event(reserve)
                    scheduler.advance(events)
                    if extends == 0 and len(batch.request_ids) == batch_size:
                        completed_decodes += 1
                        if completed_decodes == decode_steps:
                            break
                else:
                    raise RuntimeError(
                        "startup rehearsal did not reach its decode bound"
                    )
            finally:
                # The last forward can leave the next draft on another stream.
                # No startup work may overlap the first serving admission.
                device.settle()
                device.synchronize()
    return forwards
