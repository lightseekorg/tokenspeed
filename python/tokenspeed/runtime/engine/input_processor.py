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

"""Request tokenization helpers for the async frontend."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from tokenspeed.runtime.engine.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.grammar.reasoning_structural_tag import (
    structural_tag_for_reasoning_json_schema,
)
from tokenspeed.runtime.multimodal.embedder import pad_input_tokens
from tokenspeed.runtime.multimodal.mrope import compute_mrope_positions
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM


class InputProcessor:
    """Owns request-input logic: validation, tokenization, and the
    tokenized-object prep for parallel-sampling fan-out. Callers
    (``AsyncLLM``) stay thin — they route requests through this
    class and then dispatch the resulting tokenized payloads to the
    scheduler.
    """

    def __init__(self, engine: AsyncLLM):
        self.engine = engine

    def _maybe_wrap_json_schema_for_reasoning(self, sampling: dict) -> None:
        # Without this, xgrammar locks onto ``{`` at token 0 and the
        # model can't emit ``<think>…</think>`` before the JSON.
        if "json_schema" not in sampling:
            return
        reasoning_parser = getattr(self.engine.server_args, "reasoning_parser", None)
        if not reasoning_parser:
            return
        try:
            schema = sampling["json_schema"]
            if isinstance(schema, str):
                schema = json.loads(schema)
            wrapped = structural_tag_for_reasoning_json_schema(reasoning_parser, schema)
        except Exception as exc:
            self.engine.logger.warning(
                "reasoning-parser=%s: failed to wrap json_schema (%s); "
                "falling back.",
                reasoning_parser,
                exc,
            )
            return
        if wrapped is None:
            return
        sampling.pop("json_schema", None)
        sampling["structural_tag"] = wrapped

    def validate_request(self, obj: GenerateReqInput | EmbeddingReqInput) -> None:
        """Reject cross-type requests before any other processing.

        An ``EmbeddingReqInput`` arriving at a generation-only engine
        is a configuration mistake, not a runtime condition, so we
        raise eagerly instead of letting it reach tokenization.
        """
        if isinstance(obj, EmbeddingReqInput) and self.engine.is_generation:
            raise ValueError("Embedding and rerank model requests are not supported.")
        self._validate_data_parallel_rank(obj)

    def _validate_data_parallel_rank(
        self, obj: GenerateReqInput | EmbeddingReqInput
    ) -> None:
        """Validate the attention-DP hard pin on every ingress.

        This is the single choke point all GenerateReqInput paths traverse
        (Engine API, the SMG gRPC servicer, batch fan-out), unlike
        ``entrypoints/engine.py`` whose checks the servicer bypasses. Runs
        before ``normalize_batch_and_arguments``, so fields may be scalar
        or list.

        Semantics:
        - non-integer pins (bool/float/str) are a caller error and raise on
          EVERY engine shape — checked first, before the mode branches below.
        - engines without attention DP: the pin is meaningless; drop it.
        - disaggregation engines: the bootstrap_room residue is the
          authoritative rank carrier on BOTH the prefill and decode sides
          (pd/mooncake must re-derive the same rank from the room), so the
          pin is dropped; a conflicting pin is logged, never rejected —
          rejecting would turn gateway/room version skew into a 400 storm.
        - colocated attention-DP engines: an out-of-range pin is a caller
          error and raises (surfacing as a request-level failure).
        """
        ranks = getattr(obj, "data_parallel_rank", None)
        if ranks is None:
            return
        rank_list = ranks if isinstance(ranks, list) else [ranks]
        for rank in rank_list:
            # bool is an int subclass; both bool and float pins would pass a
            # bare range compare and then fail the int|None typed msgpack
            # decode inside the DP controller's dispatch loop.
            if rank is not None and (
                not isinstance(rank, int) or isinstance(rank, bool)
            ):
                raise ValueError(f"data_parallel_rank must be an int, got {rank!r}")

        server_args = self.engine.server_args
        mapping = server_args.mapping
        if not mapping.has_attn_dp:
            obj.data_parallel_rank = None
            return
        dp_size = mapping.attn.dp_size

        if server_args.disaggregation_mode != "null":
            rooms = obj.bootstrap_room
            room_list = rooms if isinstance(rooms, list) else [rooms]
            # Broadcast whichever side is scalar so a scalar pin is checked
            # against EVERY room residue (and vice versa), then pad the
            # shorter side with None — zip truncation would silently skip
            # conflicts past the shorter list, and a rank with no room to
            # check against is itself a conflict signal.
            width = max(len(rank_list), len(room_list), 1)
            if len(rank_list) == 1:
                rank_list = rank_list * width
            if len(room_list) == 1:
                room_list = room_list * width
            rank_list = rank_list + [None] * (width - len(rank_list))
            room_list = room_list + [None] * (width - len(room_list))
            for rank, room in zip(rank_list, room_list):
                if rank is None:
                    continue
                # A non-int room (gateway format bug) is unverifiable — take
                # the warn+drop path, never raise: on disaggregation engines
                # a pin must not turn room skew into request rejections.
                if (
                    not isinstance(room, int)
                    or isinstance(room, bool)
                    or room % dp_size != rank
                ):
                    self.engine.logger.warning(
                        "data_parallel_rank=%s conflicts with bootstrap_room "
                        "residue (room=%s, dp_size=%d); the room stays "
                        "authoritative on disaggregation engines — pin ignored.",
                        rank,
                        room,
                        dp_size,
                    )
                    break
            obj.data_parallel_rank = None
            return

        for rank in rank_list:
            if rank is not None and not 0 <= rank < dp_size:
                raise ValueError(
                    f"data_parallel_rank must be in [0, {dp_size}), got {rank}"
                )

    async def tokenize_batch(
        self,
        objs: list[GenerateReqInput | EmbeddingReqInput],
    ) -> list[TokenizedGenerateReqInput | TokenizedEmbeddingReqInput]:
        """Tokenize a list of requests in parallel.

        Used by the batched fan-out path in ``AsyncLLM._handle_batch_request``.
        The single-request path stays on ``tokenize_one_request`` —
        avoiding the ``asyncio.gather`` hop keeps the hot path flat.
        """
        return await asyncio.gather(*(self.tokenize_one_request(obj) for obj in objs))

    async def tokenize_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ) -> TokenizedGenerateReqInput | TokenizedEmbeddingReqInput:
        """Tokenize one request without changing current behavior."""
        input_embeds = None
        multimodal_inputs = None
        input_ids_unpadded = None
        input_text = obj.text
        input_ids = obj.input_ids

        if obj.input_embeds is not None:
            if self.engine.server_args.enable_prefix_caching:
                raise ValueError(
                    "input_embeds is provided while prefix caching is enabled. "
                    "Please add `--no-enable-prefix-caching` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
        elif input_ids is None:
            if self.engine.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )
            input_ids = self.engine.tokenizer.encode(input_text)

        precomputed_mm = (
            isinstance(obj, GenerateReqInput)
            and obj.precomputed_multimodal_inputs is not None
        )
        if (
            precomputed_mm
            and self.engine.server_args.disaggregation_mode == "decode"
            and self.engine.server_args.language_model_only
            and self.engine.model_config.is_multimodal
        ):
            # Decode only needs the expanded prompt length to size its remote-KV
            # destination; it never executes the multimodal prefill.
            precomputed_mm = False
        if precomputed_mm:
            # Gateway-side preprocess path (e.g. SMG): mm tensors are already
            # built by an upstream preprocessor and the input_ids carry the
            # expanded placeholder tokens (im_token_id) at the right offsets.
            # We still need to run pad_input_tokens so the engine's
            # MultimodalEmbedder can plan encoder-token scatter ranges from each
            # item's offsets — the bare placeholder token alone would not
            # encode per-item uniqueness needed by the single-table prefix layer.
            if not self.engine.model_config.is_multimodal_active:
                raise ValueError(
                    "precomputed_multimodal_inputs is provided for a text-only model."
                )
            multimodal_inputs = obj.precomputed_multimodal_inputs
            multimodal_inputs.ensure_pad_values()
            # MRoPE-aware models (Qwen2/3-VL, …) require 3-axis position_ids
            # derived from image_grid_thw + the image_token_id placeholders in
            # input_ids. SMG ships precomputed mm inputs with mrope_* unset; if
            # left None, model_executor falls back to a 1-D linear position
            # override — silently degrading OCR accuracy. Compute them here, on
            # the un-padded input_ids (so get_rope_index can still locate the
            # image regions) BEFORE pad_input_tokens substitutes per-image
            # pad_value over the placeholders, then pad for the embed splice.
            input_ids_list = None
            if input_ids is not None:
                input_ids_list = (
                    input_ids if isinstance(input_ids, list) else list(input_ids)
                )
            if (
                input_ids_list is not None
                and getattr(multimodal_inputs, "mrope_positions", None) is None
            ):
                mrope_positions, mrope_position_delta = compute_mrope_positions(
                    self.engine.model_config.hf_config,
                    input_ids_list,
                    multimodal_inputs.mm_items,
                )
                multimodal_inputs.mrope_positions = mrope_positions
                multimodal_inputs.mrope_position_delta = mrope_position_delta
                if mrope_position_delta is not None:
                    multimodal_inputs.mrope_position_delta_scalar = int(
                        mrope_position_delta.flatten()[0].item()
                    )
            if input_ids_list is not None:
                input_ids_unpadded = input_ids_list
                input_ids = pad_input_tokens(input_ids_list, multimodal_inputs)

        if self.engine.is_generation:
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        input_token_num = len(input_ids) if input_ids is not None else 0
        max_req_input_len = self.engine.max_req_input_len
        if max_req_input_len is None:
            max_req_input_len = self.engine.context_len - 1
        if input_token_num > max_req_input_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) exceeds the engine's "
                f"maximum input length ({max_req_input_len} tokens)."
            )

        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        # Resolve to a finite cap bounded by remaining context. Both
        # Req.check_finished and RequestState.check_finished read this field;
        # leaving it None lets a request reach the per-request page-table cap.
        adjusted_max_new_tokens = self.engine.context_len - input_token_num
        if max_new_tokens is None:
            obj.sampling_params.update({"max_new_tokens": adjusted_max_new_tokens})
        elif max_new_tokens + input_token_num >= self.engine.context_len:
            self.engine.logger.warning(
                "Requested(rid=%s) token count exceeds the model's maximum context length of %s tokens. You requested a total of %s tokens: %s tokens from the input messages and %s tokens for the completion. The max_new_tokens will be truncated to %s.",
                obj.rid,
                self.engine.context_len,
                max_new_tokens + input_token_num,
                input_token_num,
                max_new_tokens,
                adjusted_max_new_tokens,
            )
            obj.sampling_params.update({"max_new_tokens": adjusted_max_new_tokens})

        self._maybe_wrap_json_schema_for_reasoning(obj.sampling_params)

        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.resolve_seed(obj.rid)
        sampling_params.normalize(self.engine.tokenizer)
        sampling_params.verify(self.engine.model_config.vocab_size)

        # Output logprobs: two request dialects, one compute path. vLLM uses
        # sampling_params.logprobs; SGLang uses GenerateReqInput.return_logprob
        # (+ top_logprobs_num / logprob_start_len / token_ids_logprob). Either way
        # the scheduler computes only the sampled token's logprob; the response
        # dialect is chosen at render time. Gate unsupported CAPABILITIES loudly
        # here rather than silently clamping the request shape.
        sglang_req = bool(getattr(obj, "return_logprob", False))
        return_logprob = sampling_params.logprobs is not None or sglang_req
        # Output logprobs are gated by the static server arg enable_output_logprobs
        # (the sampler only gathers them when on). Reject loudly instead of
        # silently returning empty logprobs when the server cannot honor it.
        if return_logprob and not self.engine.server_args.enable_output_logprobs:
            raise ValueError(
                "logprobs were requested but the server was started without "
                "enable_output_logprobs; restart with enable_output_logprobs=True "
                "to return output logprobs."
            )
        if sglang_req:
            # vLLM top-k / full-vocab are gated in SamplingParams.verify(); gate
            # the SGLang capability knobs here for parity.
            if getattr(obj, "top_logprobs_num", 0):
                raise ValueError(
                    "top_logprobs_num > 0 (output top-k logprobs) is not supported "
                    "yet; use top_logprobs_num=0 (the sampled token's logprob)."
                )
            if (getattr(obj, "logprob_start_len", -1) or -1) >= 0:
                raise ValueError(
                    "logprob_start_len >= 0 (prompt logprobs) is not supported yet."
                )
            if getattr(obj, "token_ids_logprob", None):
                raise ValueError("token_ids_logprob is not supported yet.")
        logprob_start_len = -1
        top_logprobs_num = 0
        token_ids_logprob = None

        if isinstance(obj, GenerateReqInput):
            return TokenizedGenerateReqInput(
                rid=obj.rid,
                input_text=input_text,
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                token_ids_logprob=token_ids_logprob,
                stream=obj.stream,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=obj.bootstrap_room,
                data_parallel_rank=obj.data_parallel_rank,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                return_hidden_states=obj.return_hidden_states,
                created_time=time.time(),
                input_multi_ids=obj.input_multi_ids,
                input_extra_infos=obj.input_extra_infos,
                input_ids_unpadded=input_ids_unpadded,
                multimodal_inputs=multimodal_inputs,
            )

        return TokenizedEmbeddingReqInput(
            rid=obj.rid,
            input_text=input_text,
            input_ids=input_ids,
            sampling_params=sampling_params,
            created_time=time.time(),
        )
