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
        except Exception as e:
            self.engine.logger.warning(
                "reasoning-parser=%s: failed to wrap json_schema (%s); "
                "falling back.",
                reasoning_parser,
                e,
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
        if precomputed_mm:
            # Gateway-side preprocess path (e.g. SMG): mm tensors are already
            # built by an upstream preprocessor and the input_ids carry the
            # expanded placeholder tokens (im_token_id) at the right offsets.
            # We still need to run pad_input_tokens so the engine's
            # VisionEmbedder can plan vision-token scatter ranges from each
            # item's offsets — the bare placeholder token alone would not
            # encode per-item uniqueness needed by the radix prefix layer.
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
            if (
                input_ids is not None
                and getattr(multimodal_inputs, "mrope_positions", None) is None
            ):
                mrope_positions, mrope_position_delta = compute_mrope_positions(
                    self.engine.model_config.hf_config,
                    list(input_ids),
                    multimodal_inputs.mm_items,
                )
                multimodal_inputs.mrope_positions = mrope_positions
                multimodal_inputs.mrope_position_delta = mrope_position_delta
                if mrope_position_delta is not None:
                    multimodal_inputs.mrope_position_delta_scalar = int(
                        mrope_position_delta.flatten()[0].item()
                    )
            if input_ids is not None:
                input_ids_unpadded = list(input_ids)
                input_ids = pad_input_tokens(list(input_ids), multimodal_inputs)

        if self.engine.is_generation:
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        input_token_num = len(input_ids) if input_ids is not None else 0
        if input_token_num >= self.engine.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.engine.context_len} tokens)."
            )

        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            max_new_tokens is not None
            and max_new_tokens + input_token_num >= self.engine.context_len
        ):
            adjusted_max_new_tokens = self.engine.context_len - input_token_num
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
                obj.rid,
                input_text,
                input_ids,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                obj.stream,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=obj.bootstrap_room,
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
            obj.rid,
            input_text,
            input_ids,
            sampling_params,
            created_time=time.time(),
        )
