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

"""Drive the C++ scheduler from a test: admission records and the feedback
events a control plane sends after each forward. The engine's own control
plane is Rust (``ts-control``); these build the same events through the
Python binding."""

from collections.abc import Sequence

from tokenspeed_scheduler import ExecutionEvent, ForwardEvent, RequestSpec

# "No bound" for RequestSpec.max_cached_prefix_tokens (the C++ default).
UNBOUNDED_CACHED_PREFIX_TOKENS = 2**31 - 1


def make_spec(
    rid: str,
    tokens: list[int],
    *,
    max_cached_prefix_tokens: int,
    max_new_tokens: int = 0,
) -> RequestSpec:
    """One request's admission record; ``max_cached_prefix_tokens`` bounds
    the prefix the admission probe may claim from the cache."""
    spec = RequestSpec()
    spec.request_id = rid
    spec.tokens = tokens
    spec.max_new_tokens = max_new_tokens
    spec.max_cached_prefix_tokens = max_cached_prefix_tokens
    return spec


def make_extend_result_event(
    request_id: str,
    tokens: Sequence[int] = (),
    spec_candidate_ids: Sequence[int] | None = None,
) -> "ForwardEvent.ExtendResult":
    """A forward's result for one request: the tokens it kept and, on a P
    node's final chunk, the drafter's candidates."""
    event = ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = list(tokens)
    if spec_candidate_ids:
        event.spec_candidate_ids = list(spec_candidate_ids)
    return event


def make_finish_event(request_id: str) -> "ForwardEvent.Finish":
    event = ForwardEvent.Finish()
    event.request_id = request_id
    return event


def make_abort_event(request_id: str) -> "ForwardEvent.Abort":
    """Finish without caching the request's KV."""
    event = ForwardEvent.Abort()
    event.request_id = request_id
    return event


def make_update_reserve_tokens_event(
    request_id: str, reserve_num_tokens: int
) -> "ForwardEvent.UpdateReserveNumTokens":
    """A decode row's tokens to reserve for its next round."""
    event = ForwardEvent.UpdateReserveNumTokens()
    event.request_id = request_id
    event.reserve_num_tokens_in_next_schedule_event = reserve_num_tokens
    return event


def advance_scheduler(scheduler, events: list) -> None:
    """Feed forward results (or cache-op acknowledgments) back to the
    scheduler."""
    batch = ExecutionEvent()
    for event in events:
        batch.add_event(event)
    scheduler.advance(batch)
