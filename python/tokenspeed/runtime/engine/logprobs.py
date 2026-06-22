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

"""Logprob assembly for the async frontend — two dialects, one compute path.

The scheduler/sampler emit format-neutral wire arrays on ``recv_obj``
(``recv_obj.{input,output}_{token,top}_logprobs_{val,idx}`` etc.). This
processor renders them into the ``logprobs_info`` payload the per-request
``RequestOutputCollector`` merges, in whichever dialect the request asked for:

- ``"vllm"`` -> ``meta_info["logprobs"]`` as ``list[dict[token_id, Logprob]]``
  (one dict per generated token) plus a running ``cumulative_logprob``.
- ``"sglang"`` -> ``meta_info["output_token_logprobs"]`` (and the top-k /
  token-id variants) as lists of ``(logprob, token_id, text|None)`` tuples.
- ``"both"`` -> emit both (opt-in; doubles the payload).

Only the renderers differ; the underlying wire arrays are computed once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM
    from tokenspeed.runtime.engine.io_struct import BatchStrOut


@dataclass
class Logprob:
    """Per-output-token logprob entry (vLLM-style).

    Attributes:
        logprob: log-probability of the sampled token.
        rank: slot rank of the entry; 0 for the sampled token. NOTE: this is the
            slot index, not the token's rank in the full-vocab distribution.
    """

    logprob: float
    rank: int = 0

    def to_dict(self) -> dict:
        """JSON-safe view for serving boundaries that can't ship dataclasses."""
        return {"logprob": self.logprob, "rank": self.rank}


class LogprobsProcessor:
    """Render sampler logprob wire arrays into per-request meta_info entries.

    Holds an engine reference for the live ``tokenizer`` used when the SGLang
    dialect requests text decoding (``return_text=True``). The vLLM dialect does
    not detokenize, so the tokenizer is never touched there.
    """

    def __init__(self, engine: AsyncLLM) -> None:
        self.engine = engine

    def convert_logprob_style(
        self,
        logprobs_info: dict,
        fmt: str,
        top_logprobs_num: int,
        token_ids_logprob: list[int] | None,
        return_text: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ) -> None:
        """Render ``recv_obj``'s logprob arrays into ``logprobs_info``.

        ``fmt`` selects the dialect: ``"vllm"``, ``"sglang"``, or ``"both"``.
        Lists EXTEND across streamed frames, so this may be called repeatedly
        for one request.
        """
        if fmt in ("vllm", "both"):
            self._render_vllm(logprobs_info, recv_obj, recv_obj_index)
        if fmt in ("sglang", "both"):
            self._render_sglang(
                logprobs_info,
                top_logprobs_num,
                token_ids_logprob,
                return_text,
                recv_obj,
                recv_obj_index,
            )

    # --- shared wire access -------------------------------------------------

    @staticmethod
    def _row(recv_obj, field: str, idx: int):
        # Defensive: sampler may not have populated logprobs for this request
        # (e.g. backend doesn't support logprobs, overlap race). Treat missing
        # or out-of-range wire fields as empty rather than crashing the loop.
        lst = getattr(recv_obj, field, None) or []
        return lst[idx] if idx < len(lst) else []

    # --- vLLM renderer ------------------------------------------------------

    def _render_vllm(
        self, logprobs_info: dict, recv_obj: BatchStrOut, idx: int
    ) -> None:
        """Emit ``logprobs`` (list[dict[int, Logprob]]) + ``cumulative_logprob``.

        Only the sampled token's logprob is materialized (rank 0).
        """
        out_sampled_val = self._row(recv_obj, "output_token_logprobs_val", idx)
        out_sampled_idx = self._row(recv_obj, "output_token_logprobs_idx", idx)
        positions = [
            {int(out_sampled_idx[p]): Logprob(logprob=float(out_sampled_val[p]))}
            for p in range(len(out_sampled_idx))
        ]
        logprobs_info.setdefault("logprobs", []).extend(positions)
        logprobs_info["cumulative_logprob"] = logprobs_info.get(
            "cumulative_logprob", 0.0
        ) + (float(sum(out_sampled_val)) if out_sampled_val else 0.0)

    # --- SGLang renderer ----------------------------------------------------

    def _render_sglang(
        self,
        logprobs_info: dict,
        top_logprobs_num: int,
        token_ids_logprob: list[int] | None,
        return_text: bool,
        recv_obj: BatchStrOut,
        idx: int,
    ) -> None:
        """Emit the SGLang tuple-list keys (``{input,output}_token_logprobs``,
        and the top-k / token-id variants when requested)."""

        def _get(field: str):
            return self._row(recv_obj, field, idx)

        input_token_logprobs = logprobs_info.get("input_token_logprobs", [])
        output_token_logprobs = logprobs_info.get("output_token_logprobs", [])
        input_token_logprobs.extend(
            self.detokenize_logprob_tokens(
                _get("input_token_logprobs_val"),
                _get("input_token_logprobs_idx"),
                return_text,
            )
        )
        output_token_logprobs.extend(
            self.detokenize_logprob_tokens(
                _get("output_token_logprobs_val"),
                _get("output_token_logprobs_idx"),
                return_text,
            )
        )
        logprobs_info["input_token_logprobs"] = input_token_logprobs
        logprobs_info["output_token_logprobs"] = output_token_logprobs

        if top_logprobs_num > 0:
            input_top_logprobs = logprobs_info.get("input_top_logprobs", [])
            output_top_logprobs = logprobs_info.get("output_top_logprobs", [])
            input_top_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    _get("input_top_logprobs_val"),
                    _get("input_top_logprobs_idx"),
                    return_text,
                )
            )
            output_top_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    _get("output_top_logprobs_val"),
                    _get("output_top_logprobs_idx"),
                    return_text,
                )
            )
            logprobs_info["input_top_logprobs"] = input_top_logprobs
            logprobs_info["output_top_logprobs"] = output_top_logprobs

        if token_ids_logprob is not None:
            input_token_ids_logprobs = logprobs_info.get("input_token_ids_logprobs", [])
            output_token_ids_logprobs = logprobs_info.get(
                "output_token_ids_logprobs", []
            )
            input_token_ids_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    _get("input_token_ids_logprobs_val"),
                    _get("input_token_ids_logprobs_idx"),
                    return_text,
                )
            )
            output_token_ids_logprobs.extend(
                self.detokenize_top_logprobs_tokens(
                    _get("output_token_ids_logprobs_val"),
                    _get("output_token_ids_logprobs_idx"),
                    return_text,
                )
            )
            logprobs_info["input_token_ids_logprobs"] = input_token_ids_logprobs
            logprobs_info["output_token_ids_logprobs"] = output_token_ids_logprobs

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        assert self.engine.tokenizer is not None
        token_texts = self.engine.tokenizer.batch_decode(token_logprobs_idx)
        return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: list,
        token_logprobs_idx: list,
        decode_to_text: bool,
    ):
        # One [k] entry per position (batch all top-k tokens across positions).
        ret = []
        for logprobs, token_ids in zip(token_logprobs_val, token_logprobs_idx):
            if logprobs:
                ret.append(
                    self.detokenize_logprob_tokens(logprobs, token_ids, decode_to_text)
                )
            else:
                ret.append(None)
        return ret
