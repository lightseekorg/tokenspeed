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

"""EventLoop-side PD transfer-event integration.

The KV transfer executors (prefill/decode) DECIDE — they own the senders /
receivers and surface progress as PD events. ``PdTransferHooks`` here ACTS on
those events with the event loop's collaborators (output processor, data
plane) and returns the enriched event list for the loop to advance the
scheduler with — feedback into the scheduler stays an explicit
``advance_scheduler`` call in the loop body.
"""

from __future__ import annotations

from tokenspeed_scheduler import PD, ForwardEvent

from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor


class PdTransferHooks:
    """EventLoop-side PD transfer hooks. Stateless glue: holds only a loop
    back-reference; a cheap no-op when PD is disabled (``kv_transfer=None``).
    """

    def __init__(self, loop, device) -> None:
        self._loop = loop
        # Injected, not reached through the loop — see PauseHooks.
        self._device = device

    def poll_transfer_events(self) -> list:
        """Poll the KV transfer executor, act on its events, and return the
        (possibly enriched) event list for the scheduler advance. Empty when
        PD is disabled.
        """
        loop = self._loop
        if loop.kv_transfer is None:
            return []

        processed = []
        for event in loop.kv_transfer.generate_events():
            processed.append(event)
            if isinstance(event, PD.SucceededEvent) and isinstance(
                loop.kv_transfer, DisaggPrefillExecutor
            ):
                req_id = event.request_id
                processed.extend(loop.output_processor.finish_prefill_request(req_id))
            elif isinstance(event, PD.RemotePrefillDoneEvent):
                req_id = event.request_id
                bootstrap_token = event.bootstrap_token
                state = loop.output_processor.rid_to_state.get(req_id)
                if state is None or not state.to_abort:
                    loop.output_processor.on_remote_prefill_done(
                        req_id, bootstrap_token
                    )
                if loop._pd_cache_enabled:
                    processed.extend(
                        loop.output_processor.finish_remote_prefill_only_request(req_id)
                    )
                if isinstance(loop.kv_transfer, DisaggDecodeExecutor):
                    remote_cache_slot = loop.kv_transfer.pop_remote_cache_slot(req_id)
                    candidate_info = loop.kv_transfer.pop_remote_spec_candidate_ids(
                        req_id
                    )
                    remaining_state = loop.output_processor.rid_to_state.get(req_id)
                    still_decoding = (
                        remaining_state is not None
                        and not remaining_state.to_abort
                        and not remaining_state.finished
                    )
                    # Both are device writes read by the request's first local
                    # decode, so they cross as one ordered submission rather
                    # than as two calls from this handler.
                    self._device.run_remote_prefill_landing(
                        candidate_info,
                        remote_cache_slot if still_decoding else None,
                    )
            elif isinstance(event, PD.FailedEvent):
                # A PD/EPD transfer failed: the decode KV receiver timed out (e.g. the
                # prefill aborted on embedding timeout so the KV never arrives), or a
                # transfer errored. Publish the client-visible failure here. An
                # encode-only EPD flow still needs a following Forward.Abort because
                # its C++ FailedEvent handler is a no-op; CachePD FailedEvent
                # atomically terminalizes and fences the leased scheduler resources.
                req_id = event.request_id
                state = loop.output_processor.rid_to_state.get(req_id)
                if state is not None:
                    if state.finished:
                        loop.output_processor.reap_finished_orphan(req_id, state)
                    else:
                        state.set_finish_with_abort(
                            "PD/EPD remote transfer failed or timed out"
                        )
                        loop.output_processor.publish_finished_at_admission(
                            req_id, state
                        )
                    if not loop._pd_cache_enabled:
                        abort = ForwardEvent.Abort()
                        abort.request_id = req_id
                        processed.append(abort)
        return processed
