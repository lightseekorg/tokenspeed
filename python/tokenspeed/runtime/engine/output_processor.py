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
* ``EngineStream`` — what the frontend tracks of a request's engine-side
  outputs: its tokens, the detokenizer's window over the prompt and the
  stop strings, which the engine does not match.
* ``OutputProcessor`` — owns the hot-path translation from the engines'
  outputs (``tokenspeed_engine.Frontend.next_outputs``) into the dict-
  shaped payload the per-request ``RequestOutputCollector`` merges.
  Also owns logprob detokenization, per-request streaming metrics,
  and request dumping. The engines decide every ending but a stop
  string; a stop string ends the request here, and the engine-side
  request is aborted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.engine.collector import RequestOutputCollector
from tokenspeed.runtime.engine.detokenizer import IncrementalDetokenizer
from tokenspeed.runtime.engine.io_struct import (
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
)
from tokenspeed.runtime.engine.logprobs import LogprobsProcessor
from tokenspeed.runtime.engine.request_types import (
    FINISH_ABORT,
    FINISH_MATCHED_STR,
    INIT_INCREMENTAL_DETOKENIZATION_OFFSET,
)
from tokenspeed.runtime.metrics.collector import (
    RequestFinishStats,
    finish_reason_label,
)

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

    # The request's engine-side outputs; None for a request the engines do
    # not serve (an embedding).
    stream: EngineStream | None = None


@dataclasses.dataclass
class EngineStream:
    """A request's outputs as the engines produce them, tracked to render
    them as the in-process scheduler did.

    The engines match stop tokens, budgets and grammars; stop strings need
    detokenized text, so the frontend matches them here, token by token in
    the order the scheduler's finish check ran: a stop string ends the
    request at the token that completes it, unless the engine ended it on
    that very token.

    Attributes:
        prompt_ids: The tokenizer-valid prompt, which the detokenizer's
            first window reads (``input_ids_unpadded``).
        stop_strs: The request's stop strings.
        stop_str_max_len: The longest stop string, in tokens.
        skip_special_tokens: The request's detokenization options.
        spaces_between_special_tokens: Likewise.
        no_stop_trim: Likewise.
        output_ids: The tokens kept so far.
        stopped: The finish reason (``FINISH_MATCHED_STR.to_json()``) once a
            stop string ended the request; later outputs are the engine's
            until its abort lands, and are dropped.
    """

    prompt_ids: list[int]
    stop_strs: tuple[str, ...]
    stop_str_max_len: int
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    no_stop_trim: bool
    output_ids: list[int] = dataclasses.field(default_factory=list)
    stopped: dict | None = None
    _surr_offset: int | None = None
    _read_offset: int | None = None
    _sent_decode_ids: int = 0

    def keep(self, tokens: list[int], engine_finished: bool, tokenizer) -> int:
        """Keep this output's ``tokens``, up to one that completes a stop
        string; return how many were kept. ``engine_finished``: the engine
        ended the request on the last of them, which then outranks a stop
        string completed by the same token."""
        if not self.stop_strs:
            self.output_ids.extend(tokens)
            return len(tokens)
        for kept, token in enumerate(tokens, start=1):
            self.output_ids.append(token)
            if engine_finished and kept == len(tokens):
                return kept
            tail = tokenizer.decode(self.output_ids[-(self.stop_str_max_len + 1) :])
            for stop_str in self.stop_strs:
                if stop_str in tail:
                    self.stopped = FINISH_MATCHED_STR(matched=stop_str).to_json()
                    return kept
        return len(tokens)

    def take_unsent_decode_ids(self) -> tuple[list[int], int]:
        """The detokenizer ids not sent yet and the read offset
        (``RequestState.take_unsent_decode_ids``): the prompt from the
        surrounding offset on, then the output; the read offset is relative
        to the surrounding offset."""
        if self._surr_offset is None or self._read_offset is None:
            self._read_offset = len(self.prompt_ids)
            self._surr_offset = max(
                self._read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
        prompt_tail = len(self.prompt_ids) - self._surr_offset
        sent = self._sent_decode_ids
        if sent < prompt_tail:
            unsent = self.prompt_ids[self._surr_offset + sent :] + self.output_ids
        else:
            unsent = self.output_ids[sent - prompt_tail :]
        self._sent_decode_ids = prompt_tail + len(self.output_ids)
        return unsent, self._read_offset - self._surr_offset


def _engine_error_finish(error: Exception) -> dict:
    """The finish of a request whose engines went away mid-stream."""
    return FINISH_ABORT(message=f"The engine stream ended: {error!s}").to_json()


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

    def handle_engine_outputs(self, outputs: list[tuple[str, Any]]) -> list[str]:
        """Render a batch of engine outputs (``Frontend.next_outputs``: a
        ``(rid, chunk)`` per output, the chunk a dict or the error that ended
        its stream) as the scheduler's ``BatchTokenIDOut`` rows and hand them
        to :meth:`handle_batch_output`. Returns the requests a stop string
        ended here, whose engine-side requests the caller aborts."""
        engine = self.engine
        speculative = engine.server_args.speculative_algorithm is not None
        columns: dict[str, list] = {
            name: []
            for name in (
                "rids",
                "finished_reasons",
                "decoded_texts",
                "decode_ids",
                "read_offsets",
                "output_ids",
                "output_multi_ids",
                "skip_special_tokens",
                "spaces_between_special_tokens",
                "no_stop_trim",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "spec_verify_ct",
                "batch_accept_draft_tokens",
                "output_extra_infos",
                "output_token_logprobs_val",
                "output_token_logprobs_idx",
                "input_token_logprobs_val",
                "input_token_logprobs_idx",
                "input_top_logprobs_val",
                "input_top_logprobs_idx",
                "output_top_logprobs_val",
                "output_top_logprobs_idx",
            )
        }
        stopped: list[str] = []
        ships_top_logprobs = False
        for rid, chunk in outputs:
            state = engine.rid_to_state.get(rid)
            if state is None or state.stream is None:
                # Aborted by the client, or finished here by a stop string:
                # the engine's remaining outputs go nowhere.
                continue
            stream = state.stream
            if stream.stopped is not None:
                continue
            if isinstance(chunk, Exception):
                tokens, finish, kept = [], _engine_error_finish(chunk), 0
                chunk = {}
            else:
                tokens = chunk["output_ids"]
                finish = chunk["finish_details"]
                kept = stream.keep(tokens, finish is not None, engine.tokenizer)
                if stream.stopped is not None:
                    finish = stream.stopped
                    stopped.append(rid)
            decode_ids, read_offset = stream.take_unsent_decode_ids()
            columns["rids"].append(rid)
            columns["finished_reasons"].append(finish)
            columns["decoded_texts"].append("")
            columns["decode_ids"].append(decode_ids)
            columns["read_offsets"].append(read_offset)
            columns["output_ids"].append(tokens[:kept])
            columns["output_multi_ids"].append([])
            columns["skip_special_tokens"].append(stream.skip_special_tokens)
            columns["spaces_between_special_tokens"].append(
                stream.spaces_between_special_tokens
            )
            columns["no_stop_trim"].append(stream.no_stop_trim)
            columns["prompt_tokens"].append(chunk.get("prompt_tokens", 0))
            columns["completion_tokens"].append(len(stream.output_ids))
            columns["cached_tokens"].append(chunk.get("cached_tokens", 0))
            if speculative:
                columns["spec_verify_ct"].append(chunk.get("spec_verify_ct") or 0)
                columns["batch_accept_draft_tokens"].append(
                    chunk.get("accept_draft_tokens")
                )
            columns["output_extra_infos"].append({"decode_prefix_len": 0})
            logprobs = chunk.get("output_token_logprobs") or []
            columns["output_token_logprobs_val"].append(logprobs[:kept])
            columns["output_token_logprobs_idx"].append(
                tokens[:kept] if logprobs else []
            )
            prompt = chunk.get("input_token_logprobs")
            columns["input_token_logprobs_val"].append(prompt[0] if prompt else [])
            columns["input_token_logprobs_idx"].append(prompt[1] if prompt else [])
            prompt_top = chunk.get("input_top_logprobs")
            output_top = chunk.get("output_top_logprobs")
            ships_top_logprobs |= prompt_top is not None or output_top is not None
            columns["input_top_logprobs_val"].append(
                prompt_top[0] if prompt_top else []
            )
            columns["input_top_logprobs_idx"].append(
                prompt_top[1] if prompt_top else []
            )
            columns["output_top_logprobs_val"].append(
                output_top[0][:kept] if output_top else []
            )
            columns["output_top_logprobs_idx"].append(
                output_top[1][:kept] if output_top else []
            )
        if not columns["rids"]:
            return stopped
        if not ships_top_logprobs:
            # The frame shape the scheduler sends without top-k requests.
            for name in (
                "input_top_logprobs_val",
                "input_top_logprobs_idx",
                "output_top_logprobs_val",
                "output_top_logprobs_idx",
            ):
                columns[name] = []
        self.handle_batch_output(
            BatchTokenIDOut(
                **columns,
                input_token_ids_logprobs_val=[],
                input_token_ids_logprobs_idx=[],
                output_token_ids_logprobs_val=[],
                output_token_ids_logprobs_idx=[],
                output_hidden_states=[],
                generated_time=time.time(),
            )
        )
        return stopped

    def handle_batch_output(
        self,
        recv_obj: BatchStrOut | BatchEmbeddingOut | BatchTokenIDOut,
    ):
        for i, rid in enumerate(recv_obj.rids):
            state: ReqState = self.engine.rid_to_state.get(rid, None)
            if state is None:
                logger.error(
                    f"Received output for rid={rid!r} but the state was deleted in "
                    "AsyncLLM.",
                )
                continue

            # Whether out_dict's output_ids hold every id so far instead of
            # this frame's. Only the raw-token path streams them that way;
            # a non-streamed output replaces the pending one anyway.
            cumulative_ids = False

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.engine.server_args.weight_version,
            }
            logprobs_info = state.logprobs_info if not state.obj.stream else {}
            if state.obj.stream and "cumulative_logprob" in state.logprobs_info:
                # Lists cover this frame; the scalar covers the whole request.
                logprobs_info["cumulative_logprob"] = state.logprobs_info[
                    "cumulative_logprob"
                ]

            obj = state.obj
            sp = getattr(obj, "sampling_params", None) or {}
            vllm_req = sp.get("logprobs") is not None
            sglang_req = bool(getattr(obj, "return_logprob", False))
            if vllm_req or sglang_req:
                # Render the dialect the request asked for; default = match the
                # request (vLLM via sampling_params.logprobs, else SGLang).
                fmt = getattr(obj, "logprob_format", None) or (
                    "vllm" if vllm_req else "sglang"
                )
                try:
                    self.logprobs_processor.convert_logprob_style(
                        logprobs_info,
                        fmt,
                        getattr(obj, "top_logprobs_num", 0) or 0,
                        getattr(obj, "token_ids_logprob", None),
                        bool(getattr(obj, "return_text_in_logprobs", False)),
                        recv_obj,
                        i,
                    )
                    meta_info.update(logprobs_info)
                except Exception as exc:
                    logger.warning(
                        f"Failed to attach logprobs for rid={rid!s}: {exc!s}. Returning"
                        " response without logprobs.",
                    )
            state.logprobs_info = logprobs_info

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
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
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
                        state.output_ids.extend(recv_obj.output_ids[i])
                        output_token_ids = state.output_ids[state.last_output_offset :]
                        state.last_output_offset = len(state.output_ids)
                    else:
                        state.output_ids.extend(recv_obj.output_ids[i])
                        output_token_ids = state.output_ids.copy()

                    out_dict = {
                        "text": state.text,
                        "output_ids": output_token_ids,
                        "meta_info": meta_info,
                    }
                    if len(recv_obj.output_extra_infos):
                        out_dict["output_extra_info"] = recv_obj.output_extra_infos[i]
                else:
                    # Raw-token path: skip_tokenizer_init, or
                    # ``enable_inline_detokenizer`` is on but
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
                            f"Output text will be empty for rid={rid!s}.",
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
                        cumulative_ids = True
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
            else:
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                if self.engine.server_args.speculative_algorithm:
                    verify_ct = recv_obj.spec_verify_ct[i]
                    meta_info["spec_verify_ct"] = verify_ct
                    if verify_ct > 0:
                        # Prefill emits 1 token and each verify step 1 more;
                        # the rest are accepted drafts.
                        n = self.engine.server_args.speculative_num_draft_tokens
                        meta_info["spec_accepted_tokens"] = (
                            recv_obj.completion_tokens[i] - 1 - verify_ct
                        )
                        meta_info["spec_draft_tokens"] = verify_ct * (n - 1)
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time

            state.collector.put(
                out_dict,
                stream=bool(getattr(state.obj, "stream", False)),
                cumulative=cumulative_ids,
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
                state.first_token_time - state.created_time,
                stream=bool(state.obj.stream),
            )
        else:
            num_new_tokens = completion_tokens - state.last_completion_tokens
            if num_new_tokens > 0:
                new_time = time.time()
                interval = new_time - state.last_time
                pure_interval = recv_obj.generated_time - state.last_pure_time
                self.engine.metrics.observe_inter_token_latency(
                    interval,
                    num_new_tokens,
                )
                self.engine.metrics.observe_scheduler_inter_token_latency(
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
                    finish_reason=finish_reason_label(fr),
                    stream=bool(state.obj.stream),
                )
            )
            if (completion_tokens - state.first_completion_tokens) > 0:
                self.engine.metrics.observe_decode_time_per_output_token(
                    (state.last_time - state.first_token_time)
                    / (completion_tokens - state.first_completion_tokens)
                )

    def dump_requests(self, state: ReqState, out_dict: dict):
        import pickle as _pickle

        self.engine.dump_request_list.append(
            (state.obj, out_dict, state.created_time, time.time())
        )

        if len(self.engine.dump_request_list) >= self.engine.dump_requests_threshold:
            dump_folder = Path(self.engine.dump_requests_folder)
            filename = dump_folder / (
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
            )
            logger.info(
                f"Dump {len(self.engine.dump_request_list)!s} requests to {filename!s}",
            )

            to_dump = self.engine.dump_request_list
            self.engine.dump_request_list = []

            def background_task():
                dump_folder.mkdir(parents=True, exist_ok=True)
                with filename.open("wb") as dump_file:
                    _pickle.dump(to_dump, dump_file)

            # Schedule the task to run in the background without awaiting it
            asyncio.create_task(asyncio.to_thread(background_task))
