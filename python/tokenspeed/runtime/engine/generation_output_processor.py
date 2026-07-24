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

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tokenspeed.runtime.engine.io_struct import BatchTokenIDOut
from tokenspeed.runtime.engine.request_stats import (
    NOOP_STATS,
    RequestStats,
    RequestStatsTracker,
)
from tokenspeed.runtime.engine.request_types import (
    ABORT_CODE,
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    INIT_INCREMENTAL_DETOKENIZATION_OFFSET,
    BaseFinishReason,
)
from tokenspeed.runtime.engine.scheduler_utils import (
    make_abort_event,
    make_extend_result_event,
    make_finish_event,
    make_update_reserve_tokens_event,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.io_struct import TokenizedGenerateReqInput
    from tokenspeed.runtime.execution.types import ModelExecutionResult
    from tokenspeed.runtime.metrics.collector import EngineMetrics
    from tokenspeed.runtime.grammar.base_grammar_backend import (
        BaseGrammarObject,
    )

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

logger = get_colorful_logger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


class RequestState:
    """Per-request state needed for incremental streaming output.

    Extracts only the fields required by process_output from the incoming
    request. Does not hold a reference to Req or any scheduler object.
    """

    def __init__(
        self,
        prompt_input_ids: list[int],
        sampling_params: SamplingParams,
        stream: bool,
        tokenizer,
        eos_token_ids: list[int] = None,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: list[int] | None = None,
        multimodal_inputs=None,
        prompt_input_ids_unpadded: list[int] | None = None,
        created_time: float = 0.0,
    ) -> None:
        # --- Extracted from recv_req (immutable) ---
        self.prompt_input_ids: list[int] = prompt_input_ids
        self.prompt_input_ids_unpadded: list[int] = (
            prompt_input_ids_unpadded
            if prompt_input_ids_unpadded is not None
            else prompt_input_ids
        )
        self.multimodal_inputs = multimodal_inputs
        self.sampling_params = sampling_params
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.tokenizer = tokenizer
        self.computed_length = 0
        self.return_logprob = return_logprob
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob

        # --- generation state (updated with forward step) ---
        self.output_ids: list[int] = []
        self.finished_reason: BaseFinishReason | None = None
        self.cached_tokens: int = 0
        self.prefix_len: int = 0
        self.spec_verify_ct: int = 0
        self.accept_draft_tokens: float | None = None

        # request stats (host-side); tracker attached only with --enable-log-request-stats
        self.created_time: float = created_time
        self.stats: RequestStatsTracker = NOOP_STATS
        # Sampled-token logprobs, accumulated per generated token.
        # None when return_logprob is False.
        self.output_token_logprobs_val: list[float] | None = (
            [] if return_logprob else None
        )
        self.output_token_logprobs_idx: list[int] | None = (
            [] if return_logprob else None
        )

        # --- Streaming bookkeeping (internal) ---
        self._surr_offset: int | None = None
        self._read_offset: int | None = None
        self.decoded_text: str = ""
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        self.finished_output: bool = False

        # abort related
        self.to_abort = False
        self.to_abort_message = None
        # Client-initiated aborts skip streaming a finish (the TM already tore
        # down its state). Pause-initiated aborts set this so the passive client
        # still receives a terminating finish.
        self.abort_notify_client = False

        # cached tokenizer ids
        self._eos_token_id_cached = None
        self._additional_stop_token_ids_cached = None

        # Constrained-decoding state.
        self.grammar: BaseGrammarObject | None = None
        self.grammar_key: tuple[str, str] | None = None
        self.grammar_queued_ts: float = 0.0

    def set_finish_with_abort(self, message: str, notify_client: bool = False) -> None:
        """Mark this request as aborted with ``message``; finished_reason is
        materialized immediately so callers don't need a check_finished() pass.

        ``notify_client`` streams a terminating finish to the client (used for
        pause-initiated aborts, where the client did not tear down its state).
        """
        self.to_abort = True
        self.to_abort_message = message
        self.abort_notify_client = notify_client
        self.finished_reason = FINISH_ABORT(message=message)

    @classmethod
    def from_recv_req(
        cls,
        recv_req: TokenizedGenerateReqInput,
        tokenizer,
        eos_token_ids: list[int],
    ) -> RequestState:
        return cls(
            prompt_input_ids=recv_req.input_ids,
            sampling_params=recv_req.sampling_params,
            stream=recv_req.stream,
            tokenizer=tokenizer,
            eos_token_ids=eos_token_ids,
            return_logprob=getattr(recv_req, "return_logprob", False),
            top_logprobs_num=getattr(recv_req, "top_logprobs_num", 0),
            token_ids_logprob=getattr(recv_req, "token_ids_logprob", None),
            multimodal_inputs=getattr(recv_req, "multimodal_inputs", None),
            prompt_input_ids_unpadded=getattr(recv_req, "input_ids_unpadded", None),
            created_time=getattr(recv_req, "created_time", 0.0),
        )

    @property
    def finished(self) -> bool:
        return self.finished_reason is not None

    @property
    def input_length(self) -> int:
        return len(self.prompt_input_ids)

    @property
    def output_length(self) -> int:
        return len(self.output_ids)

    @property
    def prefill_finished(self):
        return self.computed_length >= self.input_length

    def add_computed_length(self, incr: int):
        self.computed_length += incr

    def maybe_extend_multimodal_mrope_positions(self) -> None:
        mm = self.multimodal_inputs
        if mm is None or mm.mrope_positions is None:
            return

        target_len = self.input_length + self.output_length
        current_len = mm.mrope_positions.shape[-1]
        if current_len >= target_len:
            return

        from tokenspeed.runtime.multimodal.mrope import (
            extend_mrope_positions_for_retracted_request,
        )

        mm.mrope_positions = extend_mrope_positions_for_retracted_request(
            mm.mrope_positions, target_len - current_len
        )

    def release_pending_multimodal_features(self) -> None:
        mm = self.multimodal_inputs
        if mm is not None and hasattr(mm, "release_shm_features"):
            mm.release_shm_features()

    def init_incremental_detokenize(self):
        """Return (all_ids_from_surr_offset, read_offset_relative_to_surr)."""
        if self._surr_offset is None or self._read_offset is None:
            self._read_offset = len(self.prompt_input_ids_unpadded)
            self._surr_offset = max(
                self._read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
        all_ids = self.prompt_input_ids_unpadded + self.output_ids
        return (
            all_ids[self._surr_offset :],
            self._read_offset - self._surr_offset,
        )

    def check_finished(self, skip_grammar_termination: bool = False):

        if self.finished:
            return

        if self.to_abort:
            self.finished_reason = FINISH_ABORT(
                message=self.to_abort_message,
            )
            return

        # When the capturable-grammar hostfunc is authoritative, the
        # caller identifies the terminating token itself (see
        # post_process_forward_op); firing here would re-trigger on
        # every later token and trim content via trim_matched_stop.
        if not skip_grammar_termination and self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
                return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            return

        last_token_id = self.output_ids[-1]

        if not self.sampling_params.ignore_eos:
            matched_eos = False

            # Check stop token ids
            if self.sampling_params.stop_token_ids:
                matched_eos = last_token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= last_token_id in self.eos_token_ids
            if self._eos_token_id_cached is None:
                self.set_cached_id()
            if self._eos_token_id_cached is not None:
                matched_eos |= last_token_id == self._eos_token_id_cached
            if self._additional_stop_token_ids_cached:
                matched_eos |= last_token_id in self._additional_stop_token_ids_cached
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=last_token_id)
                return

        # Check stop strings
        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                    return

    def set_cached_id(self):
        """Assign tokenizer and cache ids needed by check_finished()."""
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        self._eos_token_id_cached = int(eos_id) if eos_id is not None else None
        extra = getattr(self.tokenizer, "additional_stop_token_ids", None)
        self._additional_stop_token_ids_cached = (
            set(int(x) for x in extra) if extra else None
        )


class OutputProcesser:
    """Streams generation output to the detokenizer.

    Logprob support is intentionally omitted.
    """

    # Upper bound on how long a pending abort stays buffered waiting for
    # its matching register(). Generous — a client reorder of more than
    # a few seconds is already pathological; 5 min gives plenty of slack
    # while preventing unbounded growth on stray/post-completion aborts.
    _PENDING_ABORT_TTL_S = 300.0

    def __init__(
        self,
        send_to_tokenizer,
        attn_tp_rank: int = 0,
        spec_algorithm=None,
        spec_num_tokens: int | None = None,
        stream_interval: int = 1,
        enable_log_request_stats: bool = False,
        *,
        metrics: EngineMetrics,
    ) -> None:
        # BatchTokenIDOut is pushed directly to
        # ``send_to_tokenizer`` (AsyncLLM's input socket). The
        # inline detokenizer inside AsyncLLM is the only
        # detokenization path.
        self.send_to_tokenizer = send_to_tokenizer
        # Per-request logs fire on each DP replica's TP leader (attn_tp_rank == 0),
        # NOT the global rank 0 — otherwise DP replicas > 0 would log nothing and
        # their requests would be missing from the logs entirely.
        self.attn_tp_rank = attn_tp_rank
        self.spec_algorithm = spec_algorithm
        self.spec_num_tokens = spec_num_tokens
        self.stream_interval = stream_interval
        self.metrics = metrics
        self.enable_log_request_stats = enable_log_request_stats
        # previous forward step ts, for host-side preempt timing
        self._last_step_ts: float = 0.0
        self.log_cnt = 0
        self.rid_to_state: dict[str, RequestState] = {}
        # rid → monotonic ts at which the abort was seen. Covers the
        # "abort arrives before register()" race (pre-arrival reorder),
        # plus grammar-queued aborts that publish_finished_at_admission
        # handles. Entries for rids that never register are swept by TTL
        # to keep this bounded across a long-running server.
        self.pending_aborts: dict[str, float] = {}

    def log_accept_length(self, rid, request_state: RequestState):
        # When --enable-log-request-stats is on, the richer RequestStats line (which
        # already carries acc_len) replaces this one — see _log_request_stats.
        if self.attn_tp_rank == 0 and not self.enable_log_request_stats:
            logger.info(
                "Req: %s Finish! Accept_num_tokens_avg: %s",
                rid,
                request_state.accept_draft_tokens,
            )

    def _log_request_stats(
        self, rid: str, rs: RequestState, finish_time: float
    ) -> None:
        # Single guard for the whole stats path: no tracker (flag off) or non-zero
        # rank => nothing to do. Keeps the forward-loop call sites trivial and the
        # derivation in from_state total (it always sees a tracker).
        if rs.stats is NOOP_STATS or self.attn_tp_rank != 0:
            return
        rs.stats.mark_finish(finish_time)
        stats = RequestStats.from_state(rs, self.spec_algorithm, self.spec_num_tokens)
        # Fused into the scheduler's per-request finish line (supersedes the
        # Accept_num_tokens_avg variant in log_accept_length).
        logger.info("Req: %s Finish! %s", rid, stats)

    def sweep_pending_aborts(self) -> None:
        """Drop TTL-expired entries from ``pending_aborts``.

        Safe to call anytime. pending_aborts is insertion-ordered so we
        can stop at the first non-expired entry. Called both inside
        ``mark_abort`` (so adds are bounded) and periodically from the
        event loop (so entries also age out when aborts stop arriving).
        """
        cutoff = time.monotonic() - self._PENDING_ABORT_TTL_S
        while self.pending_aborts:
            oldest_rid = next(iter(self.pending_aborts))
            if self.pending_aborts[oldest_rid] >= cutoff:
                break
            self.pending_aborts.pop(oldest_rid)

    def mark_abort(self, rid: str, notify_client: bool = False):
        """Mark a request for abort. Safe to call before or after register().

        Routes through ``RequestState.set_finish_with_abort`` so
        ``finished_reason`` is materialized immediately. Without that,
        the gate ``request_state.to_abort and request_state.finished``
        in ``post_process_forward_op`` never fires (``.finished`` is
        ``finished_reason is not None``), so the scheduler keeps
        running the request until natural ``max_tokens``/EOS — the
        cancelled request burns up to ``max_tokens`` forward steps and
        latches a ``--max-num-seqs`` slot in the meantime.

        ``notify_client`` streams a terminating finish to the client (for
        pause-initiated aborts; client-initiated aborts leave it False since
        the tokenizer manager has already cleaned up its own state).
        """
        state = self.rid_to_state.get(rid)
        if state is not None:
            msg = "Aborted by pause" if notify_client else "AbortReq from client"
            state.set_finish_with_abort(msg, notify_client=notify_client)
            return

        self.sweep_pending_aborts()
        self.pending_aborts[rid] = time.monotonic()

    def register(self, rid, state):
        self.rid_to_state[rid] = state
        if self.enable_log_request_stats:
            state.stats = RequestStatsTracker()
        if self.pending_aborts.pop(rid, None) is not None:
            # Same reasoning as ``mark_abort``: drive the abort all the
            # way to ``finished_reason`` so the slot-release gate fires.
            state.set_finish_with_abort("AbortReq from client")

    def publish_finished_at_admission(self, rid: str, state: RequestState) -> None:
        """Stream and release a request finished outside normal forward output.

        Used for grammar admission failures and scheduler-internal aborts so
        the client receives a finish reason without another forward step.
        """
        self.rid_to_state[rid] = state
        try:
            state.finished_output = False
            self.stream_output([rid], [state])
        finally:
            state.release_pending_multimodal_features()
            self.rid_to_state.pop(rid, None)
            # Drop any queued abort marker so pending_aborts does not leak and
            # a reused rid is not immediately re-aborted on the next register.
            self.pending_aborts.pop(rid, None)

    def reap_finished_orphan(self, rid: str, state: RequestState) -> None:
        """Resolve a finished request that no future forward op will reap.

        Stream the terminating finish to a passive client (pause-initiated
        aborts still have the client waiting on the stream); client-initiated
        aborts already tore down their own state, so just drop the registered
        state so the rid does not leak.
        """
        if state.abort_notify_client:
            self.publish_finished_at_admission(rid, state)
        else:
            self.rid_to_state.pop(rid, None)

    def publish_scheduler_abort(self, rid: str, message: str) -> None:
        """Publish a terminal error decided internally by the scheduler."""
        state = self.rid_to_state.get(rid)
        if state is None:
            return
        state.set_finish_with_abort(message, notify_client=True)
        self._log_request_stats(rid, state, time.time())
        self.publish_finished_at_admission(rid, state)

    def _host_advance_matcher(self, completion, model_execution_results):
        """Host-side fallback for the grammar matcher advance.

        Reads already-synced CPU tensors. Fires when no next step arrives to run
        the hostfunc (e.g., last live request finished).
        """
        grammars = completion.grammars or []
        stride = completion.tokens_per_req
        bs = completion.bs
        advance_mask = completion.advance_mask or [True] * bs
        output_tokens = model_execution_results.output_tokens
        accept_lengths = model_execution_results.output_lengths
        terminated_at = [-1] * bs
        for i, grammar in enumerate(grammars):
            if (
                grammar is None
                or grammar.finished
                or grammar.is_terminated()
                or not advance_mask[i]
            ):
                continue
            n_accepted = int(accept_lengths[i].item())
            for j in range(n_accepted):
                tok = int(output_tokens[i * stride + j].item())
                try:
                    grammar.accept_token(tok)
                except Exception:
                    break
                if grammar.is_terminated():
                    terminated_at[i] = j
                    break
        completion.terminated_at = terminated_at

    def add_computed_length(self, rids, input_lengths, extend_prefix_lens):
        for i, rid in enumerate(rids):
            if rid not in self.rid_to_state:
                continue
            if i < len(extend_prefix_lens):
                self.rid_to_state[rid].computed_length = (
                    input_lengths[i] + extend_prefix_lens[i]
                )  # Avoid accumulation here so chunked prefill does not distort the value.
            else:
                self.rid_to_state[rid].add_computed_length(input_lengths[i])

    @staticmethod
    def _aggregate_spec_decode_step(
        *,
        forward_op,
        output_lengths,
        rid_to_state,
    ) -> tuple[int, int]:
        n_ext = forward_op.num_extends()
        accepted = 0
        num_slots = 0
        for i in range(n_ext, len(forward_op.request_ids)):
            rid = forward_op.request_ids[i]
            rs = rid_to_state.get(rid)
            if rs is None or not rs.prefill_finished:
                continue
            out_len = int(output_lengths[i].item())
            accepted += max(0, out_len - 1)
            num_slots += 1
        return num_slots, accepted

    def _emit_spec_decode_metrics(
        self, forward_op, model_execution_results: ModelExecutionResult
    ) -> None:
        if not self.metrics.enabled:
            return
        if forward_op.num_extends() > 0:
            return
        if self.spec_algorithm is None or self.spec_num_tokens is None:
            return
        if model_execution_results.output_lengths is None:
            return
        num_slots, accepted_draft_tokens = self._aggregate_spec_decode_step(
            forward_op=forward_op,
            output_lengths=model_execution_results.output_lengths,
            rid_to_state=self.rid_to_state,
        )
        if num_slots > 0:
            self.metrics.record_spec_decode_step(
                num_decode_slots=num_slots,
                accepted_draft_tokens=accepted_draft_tokens,
                draft_width=self.spec_num_tokens,
            )

    def add_cached_tokens(self, rids: list[str], extend_prefix_lens: list[int]) -> None:
        for rid, prefix_len in zip(rids, extend_prefix_lens):
            if rs := self.rid_to_state.get(rid):
                rs.cached_tokens += max(0, prefix_len - rs.computed_length)

    def post_process_forward_op(
        self,
        forward_op,
        model_execution_results: ModelExecutionResult,
        is_prefill_instance: bool = False,
        on_first_token=None,
    ):
        self.add_cached_tokens(
            forward_op.request_ids,
            forward_op.extend_prefix_lens,
        )
        with nvtx_range("commit:sync", color="red"):
            model_execution_results.sync()

        self._emit_spec_decode_metrics(forward_op, model_execution_results)

        # Wait briefly for the next step's build hostfunc to advance
        # the matcher; if it doesn't come, advance on host. The lock
        # on the completion ensures exactly one path wins.
        grammar_completion = model_execution_results.grammar_completion
        grammar_terminated_at = None
        if grammar_completion is not None:
            if not grammar_completion.event.wait(timeout=0.005):
                with grammar_completion.lock:
                    if not grammar_completion.event.is_set():
                        self._host_advance_matcher(
                            grammar_completion, model_execution_results
                        )
                        grammar_completion.event.set()
            grammar_terminated_at = grammar_completion.terminated_at
        self.log_cnt += 1
        self.add_computed_length(
            forward_op.request_ids,
            forward_op.input_lengths,
            forward_op.extend_prefix_lens,
        )
        num_extends = forward_op.num_extends()

        # per-request stats timing (host-side, only when --enable-log-request-stats)
        stats_now = time.time() if self.enable_log_request_stats else 0.0
        step_dt = 0.0
        prefilling_others = False
        if self.enable_log_request_stats:
            step_dt = (
                stats_now - self._last_step_ts if self._last_step_ts > 0.0 else 0.0
            )
            self._last_step_ts = stats_now
            prefilling_others = num_extends > 0

        request_changes = []
        stream_out_rids = []
        stream_out_states = []
        output_logprobs_list = (
            model_execution_results.output_logprobs.tolist()
            if model_execution_results.output_logprobs is not None
            else None
        )
        # NaN-guard flags, aligned with forward_op.request_ids (None when disabled).
        nan_flags_list = (
            model_execution_results.output_nan_flags.tolist()
            if model_execution_results.output_nan_flags is not None
            else None
        )
        # Per-slot total prefill length as the OP sees it (C++ Request::PrefillSize()).
        # After a flat retract the victim's generated tokens are rebased into the
        # prefill window (RebasePrefill), so this can exceed the original prompt
        # length that RequestState.prefill_finished compares against.
        prefill_lengths = getattr(forward_op, "prefill_lengths", None)
        pt = 0
        for i, rid in enumerate(forward_op.request_ids):
            output_length = model_execution_results.output_lengths[i].item()
            model_output_ids = model_execution_results.output_tokens.tolist()[
                pt : pt + output_length
            ]
            model_output_logprobs = (
                output_logprobs_list[pt : pt + output_length]
                if output_logprobs_list is not None
                else None
            )
            is_decode_slot = i >= num_extends
            if self.spec_num_tokens is not None and is_decode_slot:
                pt += self.spec_num_tokens
            else:
                pt += output_length

            if rid not in self.rid_to_state:
                # means it's delayed token, do not process
                continue

            request_state: RequestState = self.rid_to_state[rid]
            # scheduled_time is stamped pre-forward in the event loop (queue end)

            # Mid-chunk extend slot by the op's own prefill_lengths (rebased after
            # flat retract; C++ owes no result and the sampled token is garbage).
            # Fresh requests: prefill_length == prompt length, same as the gate below.
            if (
                not is_decode_slot
                and prefill_lengths is not None
                and forward_op.extend_prefix_lens[i] + forward_op.input_lengths[i]
                < prefill_lengths[i]
            ):
                continue

            # Do not output chunking result
            if not request_state.prefill_finished:
                continue

            request_state.stats.mark_prefill_done(stats_now)
            if i >= num_extends:
                request_state.stats.record_decode_step(step_dt, prefilling_others)

            nan_detected = nan_flags_list is not None and nan_flags_list[i]
            if nan_detected and not request_state.finished:
                request_state.finished_reason = FINISH_ABORT(
                    message=(
                        "Request terminated: numerical corruption (NaN logits"
                        " or out-of-vocab sample) detected during generation."
                    ),
                    err_type=ABORT_CODE.NumericalError,
                )
                # Keep one sanitized token so accounting matches a mid-step finish.
                model_output_ids = model_output_ids[:1]
                if model_output_logprobs is not None:
                    model_output_logprobs = model_output_logprobs[:1]
                self.metrics.record_nan_abort()
                if self.attn_tp_rank == 0:
                    logger.warning(
                        "Req %s terminated: NaN detected in logits (or an"
                        " out-of-vocab sample escaped the sampler);"
                        " isolating it from the batch.",
                        rid,
                    )

            # Notify caller of first output token (used by prefill node to hand off
            # bootstrap token and speculative candidates to the KV transfer layer).
            # NaN-terminated requests skip the handoff: their KV is suspect.
            if on_first_token is not None and model_output_ids and not nan_detected:
                bootstrap_token = int(model_output_ids[0])
                spec_candidate_ids = None
                if model_execution_results.next_input_ids is not None and i < len(
                    model_execution_results.next_input_ids
                ):
                    spec_candidate_ids = [
                        int(x)
                        for x in model_execution_results.next_input_ids[i].tolist()
                    ]

                on_first_token(
                    rid,
                    forward_op.request_pool_indices[i],
                    bootstrap_token,
                    spec_candidate_ids,
                )

            if is_decode_slot and self.spec_algorithm is not None:
                request_state.spec_verify_ct += 1

            # With the capturable grammar pipeline the matcher is
            # advanced by the hostfunc; here we just read which token
            # (if any) terminated it so FINISH_MATCHED_TOKEN fires on
            # the right token and check_finished skips the now-stale
            # grammar.is_terminated() probe.
            use_hostfunc = grammar_terminated_at is not None
            advance_grammar = not use_hostfunc and request_state.grammar is not None
            term_idx = (
                grammar_terminated_at[i]
                if use_hostfunc and request_state.grammar is not None
                else -1
            )
            new_ids = []
            for j, model_output_id in enumerate(model_output_ids):
                request_state.output_ids.append(model_output_id)
                if advance_grammar:
                    request_state.grammar.accept_token(model_output_id)
                if (
                    request_state.return_logprob
                    and request_state.output_token_logprobs_val is not None
                    and model_output_logprobs is not None
                ):
                    request_state.output_token_logprobs_val.append(
                        model_output_logprobs[j]
                    )
                    request_state.output_token_logprobs_idx.append(model_output_id)
                if term_idx == j:
                    # Grammar termination takes precedence over
                    # length/EOS/stop_str at the same step (matching
                    # check_finished's original order).
                    request_state.finished_reason = FINISH_MATCHED_TOKEN(
                        matched=model_output_id
                    )
                else:
                    request_state.check_finished(skip_grammar_termination=use_hostfunc)
                new_ids.append(model_output_id)
                if request_state.finished:
                    request_state.accept_draft_tokens = (
                        (len(request_state.output_ids) - 1)
                        / request_state.spec_verify_ct
                        if request_state.spec_verify_ct > 0
                        else 0
                    )
                    self.log_accept_length(rid, request_state)
                    break

            # first output token == TTFT anchor
            if request_state.output_ids:
                request_state.stats.mark_first_token(stats_now)

            # For aborted requests, skip output to detokenizer (the tokenizer
            # manager already cleaned up), just notify the scheduler to finish.
            # Exception: pause-initiated aborts (abort_notify_client) leave a
            # passive client that still needs a terminating finish streamed.
            if request_state.to_abort and request_state.finished:
                request_changes.append(make_extend_result_event(rid, new_ids))
                request_changes.append(make_finish_event(rid))
                if request_state.abort_notify_client:
                    stream_out_rids.append(rid)
                    stream_out_states.append(request_state)
                self._log_request_stats(rid, request_state, stats_now)
                request_state.release_pending_multimodal_features()
                self.rid_to_state.pop(rid)
                continue

            request_changes.append(make_extend_result_event(rid, new_ids))
            if is_prefill_instance:
                # Prefill instances: never stream intermediate output to detokenizer.
                # The finish packet is sent exactly once by finish_prefill_request()
                # when SucceededEvent arrives (KV transfer complete).  Sending output
                # here would either give the client partial data or trigger a double-
                # finish on the TM side.
                pass
            elif request_state.finished:
                stream_out_rids.append(rid)
                stream_out_states.append(request_state)
                # Abort (vs Finish) keeps corrupted KV out of the prefix caches.
                request_changes.append(
                    make_abort_event(rid) if nan_detected else make_finish_event(rid)
                )
                self._log_request_stats(rid, request_state, stats_now)
                request_state.release_pending_multimodal_features()
                self.rid_to_state.pop(rid)
            else:
                stream_out_rids.append(rid)
                stream_out_states.append(request_state)
                if is_decode_slot:
                    request_changes.append(
                        make_update_reserve_tokens_event(rid, output_length)
                    )

        self.stream_output(stream_out_rids, stream_out_states)
        return request_changes

    def on_remote_prefill_done(self, req_id: str, bootstrap_token: int) -> None:
        """Record the bootstrap token on a decode-node request (RemotePrefillDoneEvent).

        The bootstrap_token is the first real output token produced by the prefill node.
        It is appended to output_ids so the decode side starts generation from the
        correct position.

        bootstrap_token == -1 means the prefill side did not (or could not) supply a
        token (e.g. it was generated on a rank whose ZMQ message arrived after the
        success barrier had already been satisfied).
        """
        if req_id not in self.rid_to_state:
            return
        if bootstrap_token == -1:
            logger.warning(
                "[on_remote_prefill_done] rid=%s received bootstrap_token=-1, skipping append to output_ids",
                req_id,
            )
            return
        state = self.rid_to_state[req_id]
        state.output_ids.append(bootstrap_token)
        state.check_finished()

    def finish_prefill_request(self, req_id: str) -> list:
        """Finish a prefill-instance request when KV transfer succeeds (SucceededEvent).

        Called by event_loop._process_kv_transfer_events at the correct moment — the
        SucceededEvent itself drives the C++ FSM transition Decoding → Finished,
        so we must NOT emit an additional make_finish_event here.

        We send a finished BatchTokenIDOut to the detokenizer so the Prefill TM
        can resolve its HTTP coroutine and let the HTTP load balancer unblock.
        Without this, the load balancer waits forever for the prefill side's HTTP
        response while the decode side has already finished — client hangs
        indefinitely.
        """
        if req_id not in self.rid_to_state:
            return []
        rs = self.rid_to_state.pop(req_id)
        rs.release_pending_multimodal_features()

        # Ensure a finish reason is set so TokenizerManager marks the request done.
        if not rs.finished:
            rs.finished_reason = FINISH_LENGTH(length=len(rs.output_ids))
        rs.finished_output = False
        # PD prefill node's terminal path (the other finish/abort logging lives in
        # post_process_forward_op). Self-guarded, so a no-op when the flag is off.
        self._log_request_stats(req_id, rs, time.time())
        self.stream_output([req_id], [rs])
        # SucceededEvent already finishes the C++ FSM; no extra FinishEvent needed
        return []

    def stream_output(
        self, stream_out_rids: list[str], output_states: list[RequestState]
    ) -> None:
        """Collect per-step results and forward them to the detokenizer."""
        if len(output_states) == 0:
            return

        rids_to_send = []
        finished_reasons = []
        decoded_texts: list[str] = []
        decode_ids_list = []
        read_offsets: list[int] = []
        output_ids = []
        output_multi_ids = []
        skip_special_tokens: list[bool] = []
        spaces_between_special_tokens: list[bool] = []
        no_stop_trim: list[bool] = []
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        cached_tokens: list[int] = []
        spec_verify_ct: list[int] = []
        batch_accept_draft_tokens: list[float] = []
        output_extra_infos: list[dict] = []
        output_token_logprobs_val: list[list[float]] = []
        output_token_logprobs_idx: list[list[int]] = []

        for i, rs in enumerate(output_states):
            # For finished requests, always output (unless already output)
            if rs.finished:
                if rs.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                rs.finished_output = True
                should_output = True
            else:
                # For ongoing requests, use stream interval logic
                if rs.stream:
                    stream_interval = getattr(
                        rs.sampling_params, "stream_interval", None
                    )
                    if stream_interval is None:
                        stream_interval = self.stream_interval
                    should_output = (
                        rs.output_length % stream_interval == 1
                        if stream_interval > 1
                        else rs.output_length % stream_interval == 0
                    )
                else:
                    stream_interval = DEFAULT_FORCE_STREAM_INTERVAL
                    should_output = (
                        rs.output_length == 1 or rs.output_length % stream_interval == 0
                    )

            if not should_output:
                continue

            rids_to_send.append(stream_out_rids[i])
            send_token_offset = rs.send_token_offset

            finished_reasons.append(
                rs.finished_reason.to_json() if rs.finished_reason else None
            )
            decoded_texts.append(rs.decoded_text)

            decode_ids, read_offset = rs.init_incremental_detokenize()
            decode_ids_list.append(decode_ids[rs.send_decode_id_offset :])
            rs.send_decode_id_offset = len(decode_ids)

            read_offsets.append(read_offset)
            output_ids.append(rs.output_ids[send_token_offset:])
            rs.send_token_offset = rs.output_length

            output_multi_ids.append([])

            skip_special_tokens.append(rs.sampling_params.skip_special_tokens)
            spaces_between_special_tokens.append(
                rs.sampling_params.spaces_between_special_tokens
            )
            no_stop_trim.append(rs.sampling_params.no_stop_trim)
            prompt_tokens.append(rs.input_length)
            completion_tokens.append(rs.output_length)
            cached_tokens.append(rs.cached_tokens)

            if self.spec_algorithm is not None:
                spec_verify_ct.append(rs.spec_verify_ct)
                batch_accept_draft_tokens.append(rs.accept_draft_tokens)

            output_extra_infos.append({"decode_prefix_len": rs.prefix_len})

            if rs.return_logprob and rs.output_token_logprobs_val is not None:
                # Send only the slice not yet shipped; send_token_offset was
                # just advanced above, so use the logprob list tail.
                n_new = rs.output_length - send_token_offset
                output_token_logprobs_val.append(
                    rs.output_token_logprobs_val[-n_new:] if n_new > 0 else []
                )
                output_token_logprobs_idx.append(
                    rs.output_token_logprobs_idx[-n_new:] if n_new > 0 else []
                )
            else:
                output_token_logprobs_val.append([])
                output_token_logprobs_idx.append([])

        # Don't send empty batch to detokenizer
        if len(rids_to_send) == 0:
            return

        batch_id_out = BatchTokenIDOut(
            rids=rids_to_send,
            finished_reasons=finished_reasons,
            decoded_texts=decoded_texts,
            decode_ids=decode_ids_list,
            read_offsets=read_offsets,
            output_ids=output_ids,
            output_multi_ids=output_multi_ids,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            no_stop_trim=no_stop_trim,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            spec_verify_ct=spec_verify_ct,
            input_token_logprobs_val=[],
            input_token_logprobs_idx=[],
            output_token_logprobs_val=output_token_logprobs_val,
            output_token_logprobs_idx=output_token_logprobs_idx,
            input_top_logprobs_val=[],
            input_top_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            input_token_ids_logprobs_val=[],
            input_token_ids_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
            output_hidden_states=[],
            batch_accept_draft_tokens=batch_accept_draft_tokens,
            output_extra_infos=output_extra_infos,
            generated_time=time.time(),
        )

        # Push BatchTokenIDOut directly to AsyncLLM via the shared
        # tokenizer-ipc socket. AsyncLLM runs IncrementalDetokenizer
        # inline — there is no detokenizer subprocess anymore.
        self.send_to_tokenizer.send_pyobj(batch_id_out)
