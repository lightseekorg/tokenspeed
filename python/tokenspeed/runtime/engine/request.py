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

import copy
import time
from typing import Any

import torch

from tokenspeed.runtime.cache.req_to_token_pool import (
    ReqToTokenPoolInfo,
)
from tokenspeed.runtime.engine.request_types import (  # noqa: F401
    ABORT_CODE,
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    INIT_INCREMENTAL_DETOKENIZATION_OFFSET,
    BaseFinishReason,
)
from tokenspeed.runtime.grammar.base_grammar_backend import BaseGrammarObject
from tokenspeed.runtime.metrics.collector import TimeStats
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class Req:
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: tuple[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: list[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: tuple[int] | None = None,
        input_embeds: list[list[float]] | None = None,
        input_extra_infos: list[dict] | None = None,
        session_id: str | None = None,
        custom_logit_processor: str | None = None,
        return_hidden_states: bool = False,
        eos_token_ids: set[int] | None = None,
        bootstrap_host: str | None = None,
        bootstrap_port: int | None = None,
        bootstrap_room: int | None = None,
        origin_input_multi_ids: list[list[int]] | None = None,
        created_time: float | None = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # Before image padding
        )
        self.origin_input_ids = origin_input_ids
        self.origin_input_multi_ids = origin_input_multi_ids
        # Each decode stage's output ids
        self.output_ids = []
        self.output_multi_ids = []
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.
        self.fill_ids = None
        self.fill_multi_ids = None
        self.fill_input_embeds = None
        # For Eagle and chunked prefill, remove first token when chunked prefill
        self.draft_fill_ids = None
        self.session_id = session_id
        self.input_embeds = input_embeds
        self.input_extra_infos = input_extra_infos

        # Sampling info
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params

        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states

        # Memory pool info
        self.req_pool_idx: int | None = None
        self.req_to_token_pool_info: ReqToTokenPoolInfo | None = None
        # substitute for prefix_indices
        self.prefix_page_ids = []
        self.prefix_len = 0

        # Check finish
        self.tokenizer = None
        # Cached tokenizer-related ids to avoid repeated HF attribute lookups in check_finished().
        self._eos_token_id_cached: int | None = None
        self._additional_stop_token_ids_cached: set[int] | None = None
        self.finished_reason = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop, set this to true
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_abort = False
        # This carries the error message for `.to_abort` and will be attached to the finished_reason at the end of the event loop
        self.to_abort_message: str = "Unknown error"
        self.stream = stream
        self.eos_token_ids = eos_token_ids

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices = []
        # Number of tokens to run prefill.
        self.extend_input_len = 0
        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0
        self.last_node = None

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.is_chunked = 0

        # For retraction
        self.is_retracted = False

        # Incremental streamining
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # because the decode server does not have the first output token logprobs
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob

        # Logprobs (return values)
        self.input_logprob_sent: bool = False
        self.input_token_logprobs_val: list[float] | None = None
        self.input_token_logprobs_idx: list[int] | None = None
        self.input_top_logprobs_val: list[float] | None = None
        self.input_top_logprobs_idx: list[int] | None = None
        self.input_token_ids_logprobs_val: list[float] | None = None
        self.input_token_ids_logprobs_idx: list[int] | None = None
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: list[tuple[int]] | None = None
        self.temp_input_top_logprobs_val: list[torch.Tensor] | None = None
        self.temp_input_top_logprobs_idx: list[int] | None = None
        self.temp_input_token_ids_logprobs_val: list[float] | None = None
        self.temp_input_token_ids_logprobs_idx: list[int] | None = None

        if return_logprob:
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            self.output_token_ids_logprobs_val = []
            self.output_token_ids_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
                self.output_token_ids_logprobs_idx
            ) = None
        self.hidden_states = []

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar: BaseGrammarObject | None = None

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0
        self.last_host_node: Any = None
        self.host_hit_length = 0

        # The number of verification forward passes in the speculative decoding.
        # This is used to compute the average acceptance length per request.
        self.spec_verify_ct = 0

        # Time of obj created
        # Use the created_time from tokenizer if provided, otherwise use current time
        if created_time is not None:
            self.created_time = created_time
        else:
            self.created_time = time.time()
        # Calculate the time from receiving the request at TokenizerManager to reaching process_input_requests in the scheduling process
        self.tokenizer_to_scheduler_latency = time.time() - self.created_time
        # For metrics
        self.time_stats: TimeStats = TimeStats()
        self.has_log_time_stats: bool = False
        self.queue_time_start = None
        self.queue_time_end = None
        self.last_tic = time.monotonic()
        self.first_latency_recorded = (
            False  # Flag to track if first latency has been recorded
        )
        self.prefill_waiting_recorded = False
        self.first_chunk_forward_start_time = None

        self.reserve_num_tokens = 0
        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: int | None = bootstrap_port
        self.bootstrap_room: int | None = bootstrap_room

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1
        # Only meaningful in speculative reasoning.
        self.accept_draft_tokens: float | None = None

        self.output_extra_info: dict[str, Any] = {}

    def set_tokenizer(self, tokenizer):
        """Assign tokenizer and cache ids needed by check_finished()."""
        self.tokenizer = tokenizer
        if tokenizer is None:
            self._eos_token_id_cached = None
            self._additional_stop_token_ids_cached = None
            return
        eos_id = getattr(tokenizer, "eos_token_id", None)
        self._eos_token_id_cached = int(eos_id) if eos_id is not None else None
        extra = getattr(tokenizer, "additional_stop_token_ids", None)
        self._additional_stop_token_ids_cached = (
            set(int(x) for x in extra) if extra else None
        )

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
            # self.surr_offset = self.read_offset

        all_ids = self.origin_input_ids_unpadded + self.output_ids
        return all_ids[self.surr_offset :], self.read_offset - self.surr_offset

    def check_finished(self):
        if self.finished():
            return

        if self.to_abort:
            self.finished_reason = FINISH_ABORT(
                message=self.to_abort_message,
            )
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            return

        if self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
                return

        last_token_id = self.output_ids[-1]

        if not self.sampling_params.ignore_eos:
            matched_eos = False

            # Check stop token ids
            if self.sampling_params.stop_token_ids:
                matched_eos = last_token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= last_token_id in self.eos_token_ids
            if self.tokenizer is not None and self._eos_token_id_cached is None:
                self.set_tokenizer(self.tokenizer)
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

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={len(self.origin_input_ids)}, output_ids={len(self.output_ids)})"
        )
