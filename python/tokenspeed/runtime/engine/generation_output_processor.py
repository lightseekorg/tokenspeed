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
from tokenspeed.runtime.execution.types import StopCriteria
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.io_struct import TokenizedGenerateReqInput
    from tokenspeed.runtime.execution.types import ModelExecutionResult
    from tokenspeed.runtime.metrics.collector import EngineMetrics
    from tokenspeed.runtime.grammar.base_grammar_backend import (
        BaseGrammarObject,
    )

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


def resolve_stop_token_ids(
    sampling_params: SamplingParams, eos_token_ids, tokenizer
) -> frozenset[int]:
    """The tokens that end a request: its own ``stop_token_ids``, the
    model's EOS ids and the tokenizer's EOS and additional stop ids; none
    under ``ignore_eos``.

    The one definition both the control plane's finish check and the
    device's verify cut (``StopCriteria``) read, so the two always end a
    request on the same token.
    """
    if sampling_params.ignore_eos:
        return frozenset()
    ids = set(sampling_params.stop_token_ids or ())
    ids.update(eos_token_ids or ())
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        ids.add(eos_id)
    ids.update(getattr(tokenizer, "additional_stop_token_ids", None) or ())
    return frozenset(int(token_id) for token_id in ids)


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
        *,
        computes_prompt_logprobs: bool,
        eos_token_ids: list[int] = None,
        return_logprob: bool = False,
        logprob_start_len: int = -1,
        top_logprobs_num: int = 0,
        token_ids_logprob: list[int] | None = None,
        multimodal_inputs=None,
        prompt_input_ids_unpadded: list[int] | None = None,
        created_time: float = 0.0,
    ) -> None:
        """Per-request output state.

        Args:
            computes_prompt_logprobs: Whether this engine runs the prompt rows
                (False on the PD decode role, whose prefill node returns the
                prompt logprobs). Selects what ``return_logprob`` accumulates,
                so it is always stated.
        """
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
        self.stop_token_ids: frozenset[int] = resolve_stop_token_ids(
            sampling_params, eos_token_ids, tokenizer
        )
        self._stop_criteria: StopCriteria | None = None
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
        # Queue-time metric (tokenspeed:request_queue_time_seconds): wall time
        # the request entered the scheduler's waiting queue, stamped when its
        # spec is submitted, and the once-per-request observation flag. A
        # retracted request re-enters the C++ queue without a new stamp, so
        # preemption never records a second sample.
        self.scheduler_enqueue_time: float = 0.0
        self.queue_time_recorded: bool = False
        # Sampled-token logprobs, accumulated per generated token.
        # None when return_logprob is False.
        self.output_token_logprobs_val: list[float] | None = (
            [] if return_logprob else None
        )
        self.output_token_logprobs_idx: list[int] | None = (
            [] if return_logprob else None
        )
        # Prompt (input) logprobs, SGLang dialect: ``input_token_logprobs``
        # is ``[(None, ids[start]), (lp, ids[start+1]), ...]`` over positions
        # ``[logprob_start_len, input_length)``. The start is resolved here
        # (-1 -> last prompt token, the default that needs no logits; the
        # ingress rejected out-of-range values). ``input_token_logprobs`` is
        # the position-indexed accumulator for the prompt positions that need
        # logits, ``[start, input_length - 1)``, filled across prefill chunks;
        # ``_val``/``_idx`` are the finalized lists, None until the prompt's
        # final chunk commits. A decode-role request never finalizes -- the
        # prefill node returns them -- so it ships [] like a request that did
        # not ask.
        last_position = max(len(self.prompt_input_ids) - 1, 0)
        self.logprob_start_len: int = (
            last_position
            if logprob_start_len < 0
            else min(logprob_start_len, last_position)
        )
        self.returns_input_logprobs: bool = return_logprob and computes_prompt_logprobs
        self.input_token_logprobs: list[float | None] | None = (
            [None] * (last_position - self.logprob_start_len)
            if self.wants_input_logprobs
            else None
        )
        self.input_token_logprobs_val: list[float | None] | None = None
        self.input_token_logprobs_idx: list[int] | None = None
        self.input_logprob_sent: bool = False
        # Top-k alternatives (``top_logprobs_num``, ``return_logprob``'s
        # dialect): one ``(values, ids)`` per scored prompt position, beside
        # ``input_token_logprobs`` and finalized into ``[None] + rows``; and
        # one ``values`` / ``ids`` list per generated token. None when the
        # request asks for none.
        self.input_top_logprobs: list[tuple[list[float], list[int]] | None] | None = (
            [None] * len(self.input_token_logprobs)
            if self.input_token_logprobs is not None and top_logprobs_num > 0
            else None
        )
        self.input_top_logprobs_val: list[list[float] | None] | None = None
        self.input_top_logprobs_idx: list[list[int] | None] | None = None
        wants_top = return_logprob and top_logprobs_num > 0
        self.output_top_logprobs_val: list[list[float]] | None = (
            [] if wants_top else None
        )
        self.output_top_logprobs_idx: list[list[int]] | None = [] if wants_top else None
        # The NaN guard flagged this request on a prefill chunk that owed no
        # token yet (a NaN prompt logprob, or NaN logits of the chunk's last
        # row). The abort is issued when the prompt completes, like a flag on
        # the final chunk.
        self.numerical_error_detected: bool = False

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

        # Constrained-decoding state.
        self.grammar: BaseGrammarObject | None = None
        self.grammar_key: tuple[str, str] | None = None
        self.grammar_queued_ts: float = 0.0

    def set_finish_with_abort(
        self,
        message: str,
        notify_client: bool = False,
        status_code: int | None = None,
    ) -> None:
        """Mark this request as aborted with ``message``; finished_reason is
        materialized immediately so callers don't need a check_finished() pass.

        ``notify_client`` streams a terminating finish to the client (used for
        pause-initiated aborts, where the client did not tear down its state).
        ``status_code`` is the HTTP status the frontend reports; None leaves
        the abort an internal error.
        """
        self.to_abort = True
        self.to_abort_message = message
        self.abort_notify_client = notify_client
        self.finished_reason = FINISH_ABORT(message=message, status_code=status_code)

    @classmethod
    def from_recv_req(
        cls,
        recv_req: TokenizedGenerateReqInput,
        tokenizer,
        eos_token_ids: list[int],
        *,
        computes_prompt_logprobs: bool,
    ) -> RequestState:
        """Build the state for an admitted request.

        Args:
            recv_req: The tokenized request.
            tokenizer: Tokenizer for stop-string handling.
            eos_token_ids: EOS ids that end generation.
            computes_prompt_logprobs: Whether this engine runs the prompt rows
                (False on the PD decode role, whose prefill node returns the
                prompt logprobs).
        """
        return cls(
            prompt_input_ids=recv_req.input_ids,
            sampling_params=recv_req.sampling_params,
            stream=recv_req.stream,
            tokenizer=tokenizer,
            computes_prompt_logprobs=computes_prompt_logprobs,
            eos_token_ids=eos_token_ids,
            return_logprob=recv_req.return_logprob,
            logprob_start_len=recv_req.logprob_start_len,
            top_logprobs_num=recv_req.top_logprobs_num,
            token_ids_logprob=recv_req.token_ids_logprob,
            multimodal_inputs=recv_req.multimodal_inputs,
            prompt_input_ids_unpadded=recv_req.input_ids_unpadded,
            created_time=recv_req.created_time,
        )

    @property
    def finished(self) -> bool:
        return self.finished_reason is not None

    @property
    def stop_criteria(self) -> StopCriteria:
        """Where the device cuts this request's verify rounds. Built on the
        first forward: admission clamps ``max_new_tokens`` after construction
        and nothing changes it later."""
        if self._stop_criteria is None:
            self._stop_criteria = StopCriteria(
                stop_token_ids=tuple(sorted(self.stop_token_ids)),
                token_limit=self.input_length + self.sampling_params.max_new_tokens,
            )
        return self._stop_criteria

    @property
    def input_length(self) -> int:
        return len(self.prompt_input_ids)

    @property
    def output_length(self) -> int:
        return len(self.output_ids)

    @property
    def prefill_finished(self):
        return self.computed_length >= self.input_length

    @property
    def wants_input_logprobs(self) -> bool:
        """Whether any prompt position needs logits for its logprob.

        The default start (the last prompt token) yields the single
        ``(None, last_token)`` entry and no GPU work.
        """
        return (
            self.returns_input_logprobs
            and self.logprob_start_len < self.input_length - 1
        )

    def record_input_token_logprobs(
        self,
        start: int,
        logprobs: list[float],
        top: tuple[list[list[float]], list[list[int]]] | None = None,
    ) -> None:
        """Store one chunk's prompt logprobs for positions ``[start, start+n)``,
        and their top-k alternatives (``top``: per-row values and ids, most
        likely first) cut to the request's ``top_logprobs_num``."""
        if self.input_token_logprobs is None:
            raise RuntimeError("prompt logprobs recorded for a request that wants none")
        offset = start - self.logprob_start_len
        if offset < 0 or offset + len(logprobs) > len(self.input_token_logprobs):
            raise RuntimeError(
                f"prompt logprob rows [{start}, {start + len(logprobs)}) fall outside "
                f"[{self.logprob_start_len}, {self.input_length - 1})"
            )
        self.input_token_logprobs[offset : offset + len(logprobs)] = logprobs
        if self.input_top_logprobs is not None:
            if top is None:
                raise RuntimeError(
                    "prompt logprobs recorded without the top-k alternatives the "
                    "request asked for"
                )
            k = self.top_logprobs_num
            values, ids = top
            self.input_top_logprobs[offset : offset + len(logprobs)] = [
                (row_values[:k], row_ids[:k])
                for row_values, row_ids in zip(values, ids)
            ]

    def finalize_input_token_logprobs(self) -> None:
        """Assemble the SGLang lists once the prompt's final chunk committed.

        Idempotent: a second call keeps the lists already built. Element 0 is
        ``(None, ids[start])``; element
        ``k`` is the logprob of ``ids[start + k]`` given its prefix. The ids are
        the tokenizer-valid prompt (multimodal pad hashes are not tokens).
        """
        if not self.returns_input_logprobs or self.input_token_logprobs_val is not None:
            return
        logprobs: list[float | None] = (
            list(self.input_token_logprobs)
            if self.input_token_logprobs is not None
            else []
        )
        if any(lp is None for lp in logprobs):
            # Every scored position was recomputed by a chunk that commits its
            # rows before the prompt completes (the admission probe is capped
            # at the start), so a gap is a planning bug, not a runtime state.
            raise RuntimeError(
                "prompt logprobs incomplete: "
                f"{sum(lp is None for lp in logprobs)} of {len(logprobs)} positions "
                "have no value"
            )
        self.input_token_logprobs_val = [None] + logprobs
        self.input_token_logprobs_idx = list(
            self.prompt_input_ids_unpadded[self.logprob_start_len :]
        )
        if len(self.input_token_logprobs_val) != len(self.input_token_logprobs_idx):
            raise RuntimeError(
                f"prompt logprobs have {len(self.input_token_logprobs_val)} values "
                f"for {len(self.input_token_logprobs_idx)} tokens"
            )
        self.input_token_logprobs = None
        if self.top_logprobs_num > 0:
            tops = (
                list(self.input_top_logprobs)
                if self.input_top_logprobs is not None
                else []
            )
            if len(tops) != len(logprobs) or any(top is None for top in tops):
                raise RuntimeError(
                    "prompt top-k logprobs incomplete: "
                    f"{sum(top is not None for top in tops)} of {len(logprobs)} "
                    "positions have alternatives"
                )
            # Element 0 has no alternatives, like its value.
            self.input_top_logprobs_val = [None] + [top[0] for top in tops]
            self.input_top_logprobs_idx = [None] + [top[1] for top in tops]
            self.input_top_logprobs = None

    def has_pending_multimodal_features(self) -> bool:
        mm = self.multimodal_inputs
        return (
            mm is not None
            and hasattr(mm, "has_pending_shm_features")
            and mm.has_pending_shm_features()
        )

    def release_pending_multimodal_features(self) -> None:
        mm = self.multimodal_inputs
        if mm is not None and hasattr(mm, "release_shm_features"):
            mm.release_shm_features()

    def take_unsent_decode_ids(self) -> tuple[list[int], int]:
        """Return the detokenizer ids not sent yet and the read offset.

        The detokenizer's ids are the prompt from the surrounding offset on
        followed by the output; this returns the part past
        ``send_decode_id_offset`` and advances it, slicing only that tail
        instead of copying the whole output on every frame. The read offset
        is relative to the surrounding offset.
        """
        if self._surr_offset is None or self._read_offset is None:
            self._read_offset = len(self.prompt_input_ids_unpadded)
            self._surr_offset = max(
                self._read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
        prompt = self.prompt_input_ids_unpadded
        offset = self._surr_offset
        output_start = max(offset - len(prompt), 0)
        prompt_part = max(len(prompt) - offset, 0)
        sent = self.send_decode_id_offset
        unsent = (
            prompt[offset + sent :]
            + self.output_ids[output_start + max(sent - prompt_part, 0) :]
        )
        self.send_decode_id_offset = prompt_part + max(
            len(self.output_ids) - output_start, 0
        )
        return unsent, self._read_offset - offset

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
        if last_token_id in self.stop_token_ids:
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
        physical_context_len: int | None = None,
        *,
        metrics: EngineMetrics,
        defer_to_device=None,
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
        # Tripwire bound: per-request buffers (page tables, draft tables) are
        # sized for this physical extent, and spec verify may legitimately
        # commit up to it for a finished request lingering one overlap step.
        # Committing PAST it means the one-lingering-step assumption broke
        # (scheduler eviction timing, overlap depth) — the next KV write would
        # go out of bounds and hang the attention kernel, so fail loudly here
        # instead. None disables the check.
        self.physical_context_len = physical_context_len
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
        # ``DeviceHandle.run_multimodal_work``, or None where there is no
        # data plane (unit tests, CPU-only harnesses).
        self._defer_to_device = defer_to_device

    def _release_multimodal_features(self, request_state: RequestState) -> None:
        """Release a request's SHM features once no forward can still read them.

        The embedder consumes ``feature_shm`` inside the forward, and at
        in-flight depth >= 1 a queued forward can still contain the request
        this commit finishes — so the release rides the data plane's FIFO,
        behind it.

        Args:
            request_state: The finished/aborted request's state.
        """
        if self._defer_to_device is None or not (
            request_state.has_pending_multimodal_features()
        ):
            request_state.release_pending_multimodal_features()
            return
        self._defer_to_device(
            request_state.release_pending_multimodal_features, wait=False
        )

    def _check_physical_extent(
        self, rid, request_state: RequestState, output_length: int
    ) -> None:
        """Raise if a decode step committed past the physical KV extent.

        Verify may legitimately commit up to physical_context_len for a
        finished request lingering one overlap step; every per-request table
        is sized for exactly that. Going PAST it means the one-lingering-step
        assumption broke (scheduler eviction timing, overlap depth) and the
        next KV write would run out of bounds and hang the attention kernel —
        fail loudly here instead.
        """
        if self.physical_context_len is None:
            return
        total_tokens = (
            len(request_state.prompt_input_ids)
            + len(request_state.output_ids)
            + output_length
        )
        if total_tokens > self.physical_context_len:
            raise RuntimeError(
                "Committed token count exceeds the physical KV extent for "
                f"request {rid}: prompt+output reaches {total_tokens} > "
                f"physical_context_len={self.physical_context_len}. A "
                "finished request lingered more than one overlap step past "
                "context_len (see _SPEC_OVERSHOOT_SPANS in server_args.py); "
                "the next KV write would run out of bounds and hang the "
                "attention kernel."
            )

    def log_accept_length(self, rid, request_state: RequestState):
        # When --enable-log-request-stats is on, the richer RequestStats line (which
        # already carries acc_len) replaces this one — see _log_request_stats.
        if self.attn_tp_rank == 0 and not self.enable_log_request_stats:
            logger.info(
                f"Req: {rid!s} Finish! Accept_num_tokens_avg: "
                f"{request_state.accept_draft_tokens!s}",
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
        logger.info(f"Req: {rid!s} Finish! {stats!s}")

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
        """Stream a finish for a request that was finished before admission.

        Used for grammar-aborted requests (invalid/timed-out compile, missing
        backend) so the client gets a finish_reason without us wasting a
        scheduler slot or a forward step on them.
        """
        self.rid_to_state[rid] = state
        try:
            state.finished_output = False
            self.stream_output([rid], [state])
        finally:
            self._release_multimodal_features(state)
            self.rid_to_state.pop(rid, None)
            # This path replaces register() for grammar-aborted rids —
            # drop any queued abort marker so pending_aborts doesn't leak
            # and a reused rid isn't instantly re-aborted on next register.
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

    def _host_advance_matcher(self, completion, model_execution_results):
        """Host-side fallback for the grammar matcher advance.

        Reads already-synced CPU tensors. Fires when no next step arrives to run
        the hostfunc (e.g., last live request finished).
        """
        grammars = completion.grammars or []
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
            offset = completion.output_layout.token_offset(i)
            for j in range(n_accepted):
                tok = int(output_tokens[offset + j].item())
                try:
                    grammar.accept_token(tok)
                except Exception:
                    break
                if grammar.is_terminated():
                    terminated_at[i] = j
                    break
        completion.terminated_at = terminated_at

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
        # One read: each read of a scheduler-binding vector copies all of it.
        request_ids = forward_op.request_ids
        lengths = output_lengths.tolist()
        for i in range(n_ext, len(request_ids)):
            rid = request_ids[i]
            rs = rid_to_state.get(rid)
            if rs is None or not rs.prefill_finished:
                continue
            out_len = lengths[i]
            accepted += max(0, out_len - 1)
            num_slots += 1
        return num_slots, accepted

    def _emit_spec_decode_metrics(
        self, forward_op, model_execution_results: ModelExecutionResult
    ) -> None:
        if not self.metrics.enabled:
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
                draft_width=self.spec_num_tokens - 1,
            )

    def add_cached_tokens(
        self,
        rids: list[str],
        extend_prefix_lens: list[int],
        extend_replay_lens: list[int],
    ) -> None:
        # Replayed rows re-feed cached positions: the hit reaches to the end of
        # the replay window, not to where the model input starts.
        for rid, prefix_len, replay_len in zip(
            rids, extend_prefix_lens, extend_replay_lens
        ):
            if rs := self.rid_to_state.get(rid):
                rs.cached_tokens += max(0, prefix_len + replay_len - rs.computed_length)

    def post_process_forward_op(
        self,
        forward_op,
        model_execution_results: ModelExecutionResult,
        is_prefill_instance: bool,
    ):
        # Each read of a scheduler-binding vector copies all of it; read once.
        request_ids = forward_op.request_ids
        input_lengths = forward_op.input_lengths
        extend_prefix_lens = forward_op.extend_prefix_lens
        self.add_cached_tokens(
            request_ids, extend_prefix_lens, forward_op.extend_replay_lens
        )
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
        output_lengths_list = model_execution_results.output_lengths.tolist()
        output_tokens_list = model_execution_results.output_tokens.tolist()
        # The prefill rows' sampled-token alternatives, in prefill output
        # order: extend slot i's rows start where its tokens do (``pt``).
        output_top_val_list = output_top_idx_list = None
        if model_execution_results.output_top_logprobs_val is not None:
            output_top_val_list = (
                model_execution_results.output_top_logprobs_val.tolist()
            )
            output_top_idx_list = (
                model_execution_results.output_top_logprobs_idx.tolist()
            )
        self._record_input_token_logprobs(forward_op, model_execution_results)
        # Per-slot total prefill length as the OP sees it (C++
        # Request::PrefillSize()): the prompt length. A capacity retraction
        # restores the request's image and re-prefills nothing, and an L3 hit
        # is prefetched before admission, so no path extends the window past
        # the prompt.
        prefill_lengths = forward_op.prefill_lengths
        num_prefix_lens = len(extend_prefix_lens)
        pt = 0
        for i, rid in enumerate(request_ids):
            output_length = output_lengths_list[i]
            model_output_ids = output_tokens_list[pt : pt + output_length]
            model_output_logprobs = (
                output_logprobs_list[pt : pt + output_length]
                if output_logprobs_list is not None
                else None
            )
            is_decode_slot = i >= num_extends
            model_output_top = (
                (
                    output_top_val_list[pt : pt + output_length],
                    output_top_idx_list[pt : pt + output_length],
                )
                if output_top_val_list is not None and not is_decode_slot
                else None
            )
            # A decode row stores its whole verify window (one token without
            # speculation) however many it keeps -- none for a request the
            # device stopped; a prefill row stores the one token it samples.
            if is_decode_slot:
                pt += self.spec_num_tokens if self.spec_num_tokens is not None else 1
            else:
                pt += output_length

            # Mid-chunk extend slot by the op's own prefill_lengths: C++ owes
            # no token and the sampled one is garbage.
            mid_chunk = (
                not is_decode_slot
                and extend_prefix_lens[i] + input_lengths[i] < prefill_lengths[i]
            )

            request_state: RequestState | None = self.rid_to_state.get(rid)
            if request_state is None:
                # The request finished (or was aborted) on an earlier result
                # while this forward was already dispatched. Its output goes
                # nowhere, but the forward landed: the scheduler counts it
                # against the request's pages and, for a finished request,
                # publishes the frontier it consumed once nothing is in
                # flight. The tokens keep the frontier exact (TokenSize - 1);
                # a forward behind the round the device stopped the request
                # on keeps none, so the frontier stays before the stop token.
                request_changes.append(
                    make_extend_result_event(rid, [] if mid_chunk else model_output_ids)
                )
                continue

            # An extend sets the computed length rather than adding to it, so a chunked prefill cannot distort it.
            if i < num_prefix_lens:
                request_state.computed_length = input_lengths[i] + extend_prefix_lens[i]
            else:
                request_state.computed_length += input_lengths[i]
            # scheduled_time is stamped pre-forward in the event loop (queue end)

            if mid_chunk:
                # It owes no token, but the chunk's KV has landed -- report
                # that much, so the scheduler stops counting a forward
                # against these pages and may retract the request (imaging
                # the landed chunk) if the next round needs them. A NaN flag
                # on this chunk (its prompt logprobs, or its last row's
                # logits) terminates the request when the prompt completes.
                if nan_flags_list is not None and nan_flags_list[i]:
                    request_state.numerical_error_detected = True
                request_changes.append(make_extend_result_event(rid))
                continue

            # Do not output chunking result
            if not request_state.prefill_finished:
                continue
            if is_decode_slot and output_length == 0:
                raise RuntimeError(
                    f"decode request {rid} kept no token: only a request the "
                    "device already stopped keeps none, and that request has "
                    "finished"
                )

            if not is_decode_slot:
                request_state.finalize_input_token_logprobs()
            request_state.stats.mark_prefill_done(stats_now)
            if i >= num_extends:
                request_state.stats.record_decode_step(step_dt, prefilling_others)

            nan_detected = (
                nan_flags_list is not None and nan_flags_list[i]
            ) or request_state.numerical_error_detected
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
                        f"Req {rid!s} terminated: NaN detected in logits (or an"
                        " out-of-vocab sample escaped the sampler);"
                        " isolating it from the batch.",
                    )

            # P-side final chunk: the drafter candidates join the chunk's
            # ExtendResult, so the scheduler's remote-decode operation is
            # self-contained (the bootstrap token itself is new_ids[0], which
            # the scheduler reads back as LastToken). NaN-terminated requests
            # abort at the scheduler below instead: their KV is suspect and
            # must never transfer.
            spec_candidate_ids = None
            if (
                is_prefill_instance
                and model_output_ids
                and not nan_detected
                and model_execution_results.next_input_ids is not None
                and i < len(model_execution_results.next_input_ids)
            ):
                spec_candidate_ids = [
                    int(x) for x in model_execution_results.next_input_ids[i].tolist()
                ]
                if spec_candidate_ids and spec_candidate_ids[0] != int(
                    model_output_ids[0]
                ):
                    raise RuntimeError(
                        "Prefill bootstrap token mismatch: sampled token "
                        f"{int(model_output_ids[0])} != speculative candidate "
                        f"{spec_candidate_ids[0]} for request {rid}"
                    )

            if is_decode_slot and self.spec_algorithm is not None:
                request_state.spec_verify_ct += 1
                self._check_physical_extent(rid, request_state, output_length)

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
                if request_state.output_top_logprobs_val is not None:
                    if model_output_top is None or j >= len(model_output_top[0]):
                        # The ingress takes a top-k request only when every
                        # token of it is sampled in a prefill, which reports
                        # the alternatives; end the request rather than ship
                        # a token without them.
                        if not request_state.finished:
                            request_state.finished_reason = FINISH_ABORT(
                                message=(
                                    "top-k logprobs are not computed for this "
                                    "step (decode steps return none)"
                                ),
                            )
                    else:
                        k = request_state.top_logprobs_num
                        request_state.output_top_logprobs_val.append(
                            model_output_top[0][j][:k]
                        )
                        request_state.output_top_logprobs_idx.append(
                            model_output_top[1][j][:k]
                        )
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

            # The device cuts a round at the request's stop tokens and token
            # limit, so a request ending on those finishes on its last kept
            # token. One ending earlier on a condition only this side checks
            # (a stop string, a grammar, an abort) still consumed the whole
            # round on the device: the scheduler gets every kept token, so
            # the frontier it publishes is the state and KV the device holds.
            # A NaN abort keeps one sanitized token instead; it is safe because
            # an abort releases the pages without publishing them.
            reported_ids = (
                model_output_ids if len(new_ids) < len(model_output_ids) else new_ids
            )

            # first output token == TTFT anchor
            if request_state.output_ids:
                request_state.stats.mark_first_token(stats_now)

            # Numerical corruption outranks every other ending: this KV must
            # never be reused or transferred, whatever else is true of the
            # request. Abort (not Finish) keeps it out of the prefix caches,
            # and on a P node it also stops the scheduler from emitting the
            # remote decode that would ship the suspect pages to the peer --
            # the peer's receiver then fails through the transfer layer
            # instead of waiting for KV that never comes.
            if is_prefill_instance and nan_detected:
                request_changes.append(
                    make_extend_result_event(rid, reported_ids, spec_candidate_ids)
                )
                request_changes.append(make_abort_event(rid))
                self._log_request_stats(rid, request_state, stats_now)
                self._release_multimodal_features(request_state)
                self.rid_to_state.pop(rid)
                continue

            # For aborted requests, skip output to detokenizer (the tokenizer
            # manager already cleaned up), just notify the scheduler to finish.
            # Exception: pause-initiated aborts (abort_notify_client) leave a
            # passive client that still needs a terminating finish streamed.
            if request_state.to_abort and request_state.finished:
                request_changes.append(
                    make_extend_result_event(rid, reported_ids, spec_candidate_ids)
                )
                if is_prefill_instance:
                    # PD owns these pages until SucceededEvent or FailedEvent
                    # fences the transfer. Keep the request state alive so the
                    # KV transfer can finish before the scheduler releases it.
                    continue
                # Abort (vs Finish) keeps corrupted KV out of the prefix
                # caches -- an ending the client asked for does not make the
                # numbers trustworthy.
                request_changes.append(
                    make_abort_event(rid) if nan_detected else make_finish_event(rid)
                )
                if request_state.abort_notify_client:
                    stream_out_rids.append(rid)
                    stream_out_states.append(request_state)
                self._log_request_stats(rid, request_state, stats_now)
                self._release_multimodal_features(request_state)
                self.rid_to_state.pop(rid)
                continue

            request_changes.append(
                make_extend_result_event(rid, reported_ids, spec_candidate_ids)
            )
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
                self._release_multimodal_features(request_state)
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

    def _record_input_token_logprobs(
        self, forward_op, model_execution_results: ModelExecutionResult
    ) -> None:
        """Place this forward's prompt logprobs into their requests.

        The flat result follows the plan's extend-slot order; each slot's
        rows start at the prompt position the plan recorded, so a chunk lands
        at its own positions whatever earlier chunks delivered.
        """
        plan = model_execution_results.input_logprob_plan
        logprobs = model_execution_results.input_token_logprobs
        if plan is None or logprobs is None:
            return
        values = logprobs.tolist()
        if len(values) != plan.num_rows:
            raise RuntimeError(
                f"prompt logprob result has {len(values)} rows for a plan of "
                f"{plan.num_rows}"
            )
        top_values = top_ids = None
        if model_execution_results.input_top_logprobs_val is not None:
            top_values = model_execution_results.input_top_logprobs_val.tolist()
            top_ids = model_execution_results.input_top_logprobs_idx.tolist()
        cursor = 0
        request_ids = forward_op.request_ids
        for i, count in enumerate(plan.counts):
            if count == 0:
                continue
            state = self.rid_to_state.get(request_ids[i])
            if state is not None and state.input_token_logprobs is not None:
                state.record_input_token_logprobs(
                    plan.position_starts[i],
                    values[cursor : cursor + count],
                    (
                        (
                            top_values[cursor : cursor + count],
                            top_ids[cursor : cursor + count],
                        )
                        if top_values is not None
                        else None
                    ),
                )
            cursor += count

    def on_remote_prefill_done(
        self,
        req_id: str,
        bootstrap_token: int,
        cached_tokens: int,
        bootstrap_logprob: float | None,
    ) -> None:
        """Record the bootstrap token on a decode-node request (RemotePrefillDoneEvent).

        The bootstrap_token is the first real output token produced by the prefill node.
        It is appended to output_ids so the decode side starts generation from the
        correct position.

        ``cached_tokens`` reports prefill cache reuse and determines uncached prompt
        work. Decode-local cache hits only reduce KV transfer.

        A constrained request also advances its matcher here. Prefill and decode
        each compiled their own matcher for the request, and this token is the
        only one the decode node never samples itself: without accepting it here
        the matcher would mask every decode step from a state one token behind
        the text it is constraining.

        bootstrap_token == -1 means the prefill side did not (or could not) supply a
        token (e.g. it was generated on a rank whose ZMQ message arrived after the
        success barrier had already been satisfied).

        ``bootstrap_logprob`` is the prefill node's logprob of that token, so a
        request returning logprobs sees one entry per generated token; None when
        the prefill node sent none (logprobs off, or an older sender).
        """
        if req_id not in self.rid_to_state:
            return
        state = self.rid_to_state[req_id]
        state.cached_tokens = cached_tokens
        if bootstrap_token == -1:
            logger.warning(
                f"[on_remote_prefill_done] rid={req_id!s} received bootstrap_token=-1, "
                "skipping append to output_ids",
            )
            if state.grammar is not None:
                # Nothing will ever bring this matcher in sync, and masking
                # from a stale state corrupts the output more quietly than
                # not masking at all. Drop it, loudly.
                logger.warning(
                    f"[on_remote_prefill_done] rid={req_id!s} lost its bootstrap token;"
                    " "
                    "structured output is no longer enforced for it",
                )
                state.grammar = None
            return
        state.output_ids.append(bootstrap_token)
        if state.output_token_logprobs_val is not None:
            if bootstrap_logprob is not None:
                state.output_token_logprobs_val.append(bootstrap_logprob)
                state.output_token_logprobs_idx.append(bootstrap_token)
            else:
                # Compat with a prefill peer that sends no logprob (older
                # version, or launched without --enable-output-logprobs):
                # the response's output logprobs are one entry short, which
                # the client cannot tell from the data. Say so here.
                logger.warning(
                    f"[on_remote_prefill_done] rid={req_id!s} returns logprobs but "
                    "the prefill peer sent no bootstrap logprob; its first output "
                    "token has none (is the prefill node started with "
                    "--enable-output-logprobs and at the same version?)",
                )
        if state.grammar is not None:
            state.grammar.accept_token(bootstrap_token)
        state.check_finished()

    def finish_scheduler_aborted_requests(self, aborts) -> None:
        """Finish the requests the scheduler aborted while building a plan.

        The last resort of a capacity retraction whose victim could not be
        imaged (``ExecutionPlan.aborts``, ``docs/configuration/server.md``,
        "Retraction snapshot pool"): the scheduler released the request's
        pages and slot in that very round and emits nothing further for it,
        so this is the request's whole finish -- the client gets an
        ``ABORT_CODE.CapacityAbort`` finish carrying the scheduler's detail
        (the shortfall and the knobs to raise), which a caller that can
        resubmit retries on. Returns no scheduler event: the request is
        already ``Finished`` there. A victim is quiescent when chosen, so no
        in-flight forward names it. A client abort may cross the same round:
        its state is already finished with the client's reason (and the
        tokenizer manager has torn down, unless a pause-initiated abort left
        a passive client waiting for its terminating finish), so it is
        released here without a capacity finish, count or stream; an id
        already gone is skipped.

        Args:
            aborts: The plan's ``SchedulerAbort`` entries (``request_id``,
                ``reason``, ``detail``).
        """
        for abort in aborts:
            rid = str(abort.request_id)
            state = self.rid_to_state.get(rid)
            if state is None:
                continue
            detail = str(abort.detail)
            if not state.finished:
                state.finished_reason = FINISH_ABORT(
                    message=f"Request aborted by the scheduler: {detail}",
                    err_type=ABORT_CODE.CapacityAbort,
                )
                self.metrics.record_capacity_abort()
                if self.attn_tp_rank == 0:
                    logger.warning(
                        f"Req {rid!s} aborted by the scheduler ({abort.reason!s}): "
                        f"{detail}"
                    )
                self.stream_output([rid], [state])
            elif state.abort_notify_client:
                self.stream_output([rid], [state])
            self._log_request_stats(rid, state, time.time())
            self._release_multimodal_features(state)
            self.rid_to_state.pop(rid)

    def finish_remote_prefill_only_request(self, req_id: str) -> list:
        """Finish after remote prefill when no decode step is needed.

        A PD Prefill computes the first real output token.  A one-token
        request is therefore already complete when Decode receives
        ``RemotePrefillDoneEvent`` and must not be scheduled for an additional
        decode forward. An aborted request follows the same transport fence
        before its scheduler Abort is emitted.
        """
        state = self.rid_to_state.get(req_id)
        if state is None or not state.finished:
            return []

        now = time.time()
        state.stats.mark_prefill_done(now)
        if not state.to_abort:
            state.stats.mark_first_token(now)
        self._log_request_stats(req_id, state, now)
        if not state.to_abort or state.abort_notify_client:
            self.stream_output([req_id], [state])
        self._release_multimodal_features(state)
        self.rid_to_state.pop(req_id)
        return [
            make_abort_event(req_id) if state.to_abort else make_finish_event(req_id)
        ]

    def finish_prefill_request(self, req_id: str) -> list:
        """Finish a prefill-instance request when KV transfer succeeds (SucceededEvent).

        Called by PdTransferHooks.poll_transfer_events at the correct moment — the
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
        self._release_multimodal_features(rs)

        # Ensure a finish reason is set so TokenizerManager marks the request done.
        if not rs.finished:
            rs.finished_reason = FINISH_LENGTH(length=len(rs.output_ids))
        rs.finished_output = False
        # PD prefill node's terminal path (the other finish/abort logging lives in
        # post_process_forward_op). Self-guarded, so a no-op when the flag is off.
        self._log_request_stats(req_id, rs, time.time())
        if not rs.to_abort or rs.abort_notify_client:
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
        input_token_logprobs_val: list[list[float | None]] = []
        input_token_logprobs_idx: list[list[int]] = []
        input_top_logprobs_val: list[list] = []
        input_top_logprobs_idx: list[list] = []
        output_top_logprobs_val: list[list[list[float]]] = []
        output_top_logprobs_idx: list[list[list[int]]] = []
        # The top-k columns get one entry per request only in a frame where a
        # request asked for top-k. Otherwise they stay [] for the whole
        # frame, which the frontend reads as no alternatives for every
        # request, so the default path adds no per-request list to the frame.
        ships_top_logprobs = any(
            rs.output_top_logprobs_val is not None for rs in output_states
        )

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

            unsent_decode_ids, read_offset = rs.take_unsent_decode_ids()
            decode_ids_list.append(unsent_decode_ids)

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
            # Prompt logprobs ship once, on the first frame after the prompt's
            # final chunk committed; later frames (and a decode-role request,
            # which never finalizes) carry [] and the frontend keeps the lists.
            ships_prompt_logprobs = (
                rs.input_token_logprobs_val is not None and not rs.input_logprob_sent
            )
            if ships_prompt_logprobs:
                rs.input_logprob_sent = True
                input_token_logprobs_val.append(rs.input_token_logprobs_val)
                input_token_logprobs_idx.append(rs.input_token_logprobs_idx)
            else:
                input_token_logprobs_val.append([])
                input_token_logprobs_idx.append([])

            if ships_top_logprobs:
                # The prompt's alternatives ride with its logprobs.
                if ships_prompt_logprobs and rs.input_top_logprobs_val is not None:
                    input_top_logprobs_val.append(rs.input_top_logprobs_val)
                    input_top_logprobs_idx.append(rs.input_top_logprobs_idx)
                else:
                    input_top_logprobs_val.append([])
                    input_top_logprobs_idx.append([])
                if rs.output_top_logprobs_val is not None:
                    # One entry per generated token, shipped with its logprob.
                    n_new = rs.output_length - send_token_offset
                    output_top_logprobs_val.append(
                        rs.output_top_logprobs_val[-n_new:] if n_new > 0 else []
                    )
                    output_top_logprobs_idx.append(
                        rs.output_top_logprobs_idx[-n_new:] if n_new > 0 else []
                    )
                else:
                    output_top_logprobs_val.append([])
                    output_top_logprobs_idx.append([])

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
            input_token_logprobs_val=input_token_logprobs_val,
            input_token_logprobs_idx=input_token_logprobs_idx,
            output_token_logprobs_val=output_token_logprobs_val,
            output_token_logprobs_idx=output_token_logprobs_idx,
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            output_top_logprobs_val=output_top_logprobs_val,
            output_top_logprobs_idx=output_top_logprobs_idx,
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
