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

"""Per-request output state and batch-output handling for the async frontend.

Hosts:

* ``ReqState`` — per-request bookkeeping that ``AsyncLLM`` keeps in
  its ``rid_to_state`` map.
* ``OutputProcessor`` — owns the hot-path translation from scheduler
  output frames (``BatchStrOut`` / ``BatchTokenIDOut`` /
  ``BatchEmbeddingOut`` / ``BatchMultimodalOut``) into the dict-
  shaped payload the per-request ``RequestOutputCollector`` merges.
  Also owns logprob detokenization, per-request streaming metrics,
  and request dumping. Stop authority stays with the scheduler —
  finish reasons are consumed as input flags, not invented here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.engine.collector import RequestOutputCollector
from tokenspeed.runtime.engine.detokenizer import IncrementalDetokenizer
from tokenspeed.runtime.engine.io_struct import (
    BatchEmbeddingOut,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
)
from tokenspeed.runtime.engine.logprobs import LogprobsProcessor
from tokenspeed.runtime.metrics.collector import RequestFinishStats

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    collector: RequestOutputCollector
    finished: bool
    event: asyncio.Event
    obj: Any

    # For metrics
    created_time: float
    tokenized_time: float = 0.0
    finished_time: float = 0.0
    first_token_time: float = 0.0
    first_completion_tokens: int = 1
    last_time: float = 0.0
    last_pure_time: float = 0.0
    last_completion_tokens: int = 1

    # For streaming output
    last_output_offset: int = 0

    # For incremental state update.
    text: str = ""
    output_ids: list[int] = dataclasses.field(default_factory=list)
    logprobs_info: dict = dataclasses.field(default_factory=dict)

    # Inline detokenizer: lazily constructed on the first
    # BatchTokenIDOut frame for this request. Stays None for
    # raw-token mode (skip_tokenizer_init or tokenizer absent).
    # See runtime/engine/detokenizer.py::IncrementalDetokenizer.
    inline_detokenizer: IncrementalDetokenizer | None = None


class OutputProcessor:
    """Translate scheduler output frames into per-request collector payloads.

    Owns the batch-output dispatch, logprob detokenization, streaming
    metrics collection, and request dumping. The engine reference lets
    this class read ``rid_to_state``, ``tokenizer``, ``server_args``,
    and the metrics / dump-state fields that live on ``AsyncLLM``
    without cloning them here.
    """

    def __init__(self, engine: AsyncLLM):
        self.engine = engine
        self.logprobs_processor = LogprobsProcessor(engine)

    def handle_batch_output(
        self,
        recv_obj: (
            BatchStrOut | BatchEmbeddingOut | BatchMultimodalOut | BatchTokenIDOut
        ),
    ):
        for i, rid in enumerate(recv_obj.rids):
            state: ReqState = self.engine.rid_to_state.get(rid, None)
            if state is None:
                logger.error(
                    "Received output for rid=%r but the state was deleted in AsyncLLM.",
                    rid,
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
            }
            logprobs_info = state.logprobs_info if not state.obj.stream else {}

            if getattr(state.obj, "return_logprob", False):
                try:
                    self.logprobs_processor.convert_logprob_style(
                        logprobs_info,
                        state.obj.top_logprobs_num,
                        state.obj.token_ids_logprob,
                        state.obj.return_text_in_logprobs,
                        recv_obj,
                        i,
                    )
                    meta_info.update(logprobs_info)
                except Exception as e:
                    logger.warning(
                        "Failed to attach logprobs for rid=%s: %s. Returning response without logprobs.",
                        rid,
                        e,
                    )

            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if isinstance(recv_obj, BatchStrOut):
                if len(recv_obj.batch_accept_draft_tokens) > 0:
                    meta_info.update(
                        {"accept_draft_tokens": recv_obj.batch_accept_draft_tokens[i]}
                    )
                state.text += recv_obj.output_strs[i]
                if state.obj.stream:
                    state.logprobs_info = logprobs_info
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.logprobs_info.update(logprobs_info)
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "text": state.text,
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
                if len(recv_obj.output_extra_infos):
                    out_dict["output_extra_info"] = recv_obj.output_extra_infos[i]
            elif isinstance(recv_obj, BatchTokenIDOut):
                if (
                    self.engine.server_args.enable_inline_detokenizer
                    and self.engine.tokenizer is not None
                ):
                    # Inline detokenizer path: run
                    # IncrementalDetokenizer per request and produce
                    # a BatchStrOut-shaped out_dict that
                    # RequestOutputCollector merges.
                    if state.inline_detokenizer is None:
                        state.inline_detokenizer = IncrementalDetokenizer(
                            decoded_text=recv_obj.decoded_texts[i],
                            read_offset=recv_obj.read_offsets[i],
                        )
                    incremental_emit = state.inline_detokenizer.process(
                        self.engine.tokenizer,
                        new_decode_ids=recv_obj.decode_ids[i],
                        finished_reason=recv_obj.finished_reasons[i],
                        no_stop_trim=recv_obj.no_stop_trim[i],
                        skip_special_tokens=recv_obj.skip_special_tokens[i],
                        spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[
                            i
                        ],
                    )
                    if len(recv_obj.batch_accept_draft_tokens) > 0:
                        meta_info.update(
                            {
                                "accept_draft_tokens": recv_obj.batch_accept_draft_tokens[
                                    i
                                ]
                            }
                        )
                    state.text += incremental_emit
                    if state.obj.stream:
                        state.logprobs_info = logprobs_info
                        state.output_ids.extend(recv_obj.decode_ids[i])
                        output_token_ids = state.output_ids[state.last_output_offset :]
                        state.last_output_offset = len(state.output_ids)
                    else:
                        state.logprobs_info.update(logprobs_info)
                        state.output_ids.extend(recv_obj.decode_ids[i])
                        output_token_ids = state.output_ids.copy()

                    out_dict = {
                        "text": state.text,
                        "output_ids": output_token_ids,
                        "meta_info": meta_info,
                    }
                    if len(recv_obj.output_extra_infos):
                        out_dict["output_extra_info"] = recv_obj.output_extra_infos[i]
                else:
                    # Raw-token path: skip_tokenizer_init, mm_mode=multi_ids,
                    # or ``enable_inline_detokenizer`` is on but
                    # ``self.tokenizer is None`` unexpectedly. Keep the
                    # response shape aligned with the BatchStrOut path by
                    # always populating ``text`` from the accumulated state.
                    if (
                        self.engine.server_args.enable_inline_detokenizer
                        and self.engine.tokenizer is None
                        and not self.engine.server_args.skip_tokenizer_init
                    ):
                        logger.warning(
                            "AsyncLLM raw-token branch fired with "
                            "enable_inline_detokenizer=True and "
                            "skip_tokenizer_init=False; "
                            "self.tokenizer is unexpectedly None. "
                            "Output text will be empty for rid=%s.",
                            rid,
                        )

                    output_multi_ids = None
                    if self.engine.server_args.stream_output and state.obj.stream:
                        state.output_ids.extend(recv_obj.output_ids[i])
                        output_token_ids = state.output_ids[state.last_output_offset :]
                        if recv_obj.output_multi_ids is not None:
                            output_multi_ids = recv_obj.output_multi_ids[i][
                                state.last_output_offset :
                            ]
                        state.last_output_offset = len(state.output_ids)
                    else:
                        state.output_ids.extend(recv_obj.output_ids[i])
                        output_token_ids = state.output_ids.copy()
                        if recv_obj.output_multi_ids is not None:
                            output_multi_ids = recv_obj.output_multi_ids[i]

                    if len(recv_obj.batch_accept_draft_tokens) > 0:
                        meta_info.update(
                            {
                                "accept_draft_tokens": recv_obj.batch_accept_draft_tokens[
                                    i
                                ]
                            }
                        )

                    out_dict = {
                        "text": state.text,
                        "output_ids": output_token_ids,
                        "meta_info": meta_info,
                    }
                    if len(recv_obj.output_extra_infos):
                        out_dict["output_extra_info"] = recv_obj.output_extra_infos[i]
                    if output_multi_ids is not None:
                        out_dict["output_multi_ids"] = output_multi_ids
            elif isinstance(recv_obj, BatchMultimodalOut):
                raise NotImplementedError()
            else:
                assert isinstance(recv_obj, BatchEmbeddingOut)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                if self.engine.server_args.speculative_algorithm:
                    meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time

            state.collector.put(
                out_dict, stream=bool(getattr(state.obj, "stream", False))
            )
            state.event.set()

            # Log metrics and dump
            if self.engine.enable_metrics and not isinstance(
                recv_obj, BatchEmbeddingOut
            ):
                self.collect_metrics(state, recv_obj, i)
            if (
                self.engine.dump_requests_folder
                and state.finished
                and state.obj.log_metrics
            ):
                self.dump_requests(state, out_dict)

    def collect_metrics(self, state: ReqState, recv_obj, i: int):
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        if state.first_token_time == 0.0:
            state.first_token_time = state.last_time = time.time()
            state.last_pure_time = recv_obj.generated_time
            state.last_completion_tokens = completion_tokens
            state.first_completion_tokens = completion_tokens
            self.engine.metrics.observe_time_to_first_token(
                state.first_token_time - state.created_time
            )
        else:
            num_new_tokens = completion_tokens - state.last_completion_tokens
            if num_new_tokens:
                new_time = time.time()
                interval = new_time - state.last_time
                pure_interval = recv_obj.generated_time - state.last_pure_time
                self.engine.metrics.observe_inter_token_latency(
                    interval,
                    num_new_tokens,
                )
                self.engine.metrics.observe_inter_token_latency(
                    pure_interval, num_new_tokens
                )
                state.last_pure_time = recv_obj.generated_time
                state.last_time = new_time
                state.last_completion_tokens = completion_tokens

        if state.finished:
            fr = recv_obj.finished_reasons[i]
            # TODO: consolidate the return type of fr.
            finished_ok = not (
                fr.get("type") == "abort"
                if isinstance(fr, dict)
                else getattr(fr, "is_error", False)
            )
            cached_prompt = (
                recv_obj.cached_tokens[i]
                if getattr(recv_obj, "cached_tokens", None) is not None
                else 0
            )
            self.engine.metrics.record_request_finish(
                RequestFinishStats(
                    prompt_tokens=recv_obj.prompt_tokens[i],
                    generation_tokens=completion_tokens,
                    e2e_latency=state.finished_time - state.created_time,
                    cached_prompt_tokens=cached_prompt,
                    finished_ok=finished_ok,
                )
            )
            if (completion_tokens - state.first_completion_tokens) > 0:
                self.engine.metrics.observe_inter_token_latency(
                    state.finished_time - state.first_token_time,
                    completion_tokens - state.first_completion_tokens,
                )

    def dump_requests(self, state: ReqState, out_dict: dict):
        import pickle as _pickle

        self.engine.dump_request_list.append(
            (state.obj, out_dict, state.created_time, time.time())
        )

        if len(self.engine.dump_request_list) >= self.engine.dump_requests_threshold:
            filename = os.path.join(
                self.engine.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            logger.info(
                "Dump %s requests to %s", len(self.engine.dump_request_list), filename
            )

            to_dump = self.engine.dump_request_list
            self.engine.dump_request_list = []

            dump_folder = self.engine.dump_requests_folder

            def background_task():
                os.makedirs(dump_folder, exist_ok=True)
                with open(filename, "wb") as f:
                    _pickle.dump(to_dump, f)

            # Schedule the task to run in the background without awaiting it
            asyncio.create_task(asyncio.to_thread(background_task))
