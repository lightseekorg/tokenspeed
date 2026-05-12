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

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling.cuda import (
    verify_chain_greedy as _verify_chain_greedy_cuda,
)
from tokenspeed_kernel.registry import error_fn

from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.utils import nan_guard_logits, write_output_logprobs
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:

    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo


def _verify_chain_greedy_torch(
    predicts: torch.Tensor,  # [bs * N] int32, in/out
    accept_index: torch.Tensor,  # [bs, N] int32, in/out (-1-filled on entry)
    accept_token_num: torch.Tensor,  # [bs] int32, out
    candidates: torch.Tensor,  # [bs, N] int32
    target_predict: torch.Tensor,  # [bs, N] int64 (argmax output)
    batch_size: int,
    num_draft_tokens: int,
) -> None:
    """Pure-torch equivalent of tokenspeed_kernel.verify_chain_greedy.

    Used on non-CUDA devices and when the CUDA kernel is unavailable.
    """

    bs = batch_size
    n = num_draft_tokens

    # For i in 1..n-1: candidates[b, i] accepted iff it equals target_predict[b, i-1].
    # Accepted prefix length per row = longest-leading-1s of the match array.
    match = candidates[:, 1:] == target_predict[:, :-1].to(
        candidates.dtype
    )  # [bs, n-1]
    leading = torch.cumprod(match.to(torch.int32), dim=1)  # [bs, n-1]
    num_accepted = leading.sum(dim=1).to(torch.int32)  # [bs]

    # Fill all of `predicts` with target_predict; slots outside the accepted
    # prefix are harmless because accept_index keeps them at -1 and callers
    # mask on that. Matches the CUDA kernel's observable state.
    predicts.copy_(target_predict.reshape(-1).to(torch.int32))

    device = candidates.device
    pos = torch.arange(n, device=device).unsqueeze(0)  # [1, n]
    batch_off = torch.arange(bs, device=device).unsqueeze(1) * n  # [bs, 1]
    flat_idx = (batch_off + pos).to(torch.int32)  # [bs, n]
    valid = pos <= num_accepted.unsqueeze(1)  # [bs, n]
    accept_index.copy_(torch.where(valid, flat_idx, torch.full_like(accept_index, -1)))
    accept_token_num.copy_(num_accepted)


def _verify_chain_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    batch_size: int,
    num_draft_tokens: int,
    enable_pdl: bool = False,
) -> None:

    # Prefer the CUDA kernel when available AND the tensors are on CUDA.
    if _verify_chain_greedy_cuda is not error_fn and candidates.is_cuda:

        _verify_chain_greedy_cuda(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            target_predict=target_predict,
            batch_size=batch_size,
            num_draft_tokens=num_draft_tokens,
            enable_pdl=enable_pdl,
        )
        return

    _verify_chain_greedy_torch(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        target_predict=target_predict,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
    )


class GreedySamplingBackend(SamplingBackend):
    """Greedy-only backend: argmax for single-step, chain-greedy verify for
    multi-step verification. No flashinfer / min_p / penalty machinery, no
    coin buffers. Verify uses the fused CUDA kernel when available; falls
    back to a pure-torch implementation otherwise (CPU, ROCm, etc.).

    sampling_info is ignored for single-step (always argmax). verify() also
    treats every request as greedy — stochastic verification is not
    supported. Intended as the default backend and as a fallback when
    flashinfer is unavailable."""

    def __init__(self, config: SamplingBackendConfig) -> None:

        super().__init__(config)

        self._ones_buf = torch.ones(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )
        self._predict_buf = torch.zeros(
            (config.max_bs * config.max_draft_tokens_per_req,),
            dtype=torch.int32,
            device=config.device,
        )
        # Flat layout so [:bs * n].view(bs, n) is contiguous for any bs/n
        # (required by maybe_broadcast / NCCL).
        self._accept_index_buf = torch.zeros(
            (config.max_bs * config.max_draft_tokens_per_req,),
            dtype=torch.int32,
            device=config.device,
        )
        self._accept_length_buf = torch.zeros(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )

    @nvtx_range("sampling:sample", color="yellow")
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        logits = nan_guard_logits(
            logits_output.next_token_logits, self.config.enable_nan_detection
        )
        # Grammar bitmask apply — captured inside the CUDA graph. Buffer is
        # pre-bound by bind_grammar_mask_buf; non-grammar rows stay all-ones
        # so apply is a no-op.
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )
        tokens = torch.argmax(logits, -1).to(torch.int32)

        if self.config.enable_output_logprobs:

            write_output_logprobs(logits_output, logits, tokens)

        # Greedy backend allocates no sampling buffers — derive bs from logits.
        bs = logits.shape[0]

        return tokens, self._ones_buf[:bs]

    @nvtx_range("sampling:verify", color="yellow")
    def verify(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        bs = candidates.shape[0]
        num_tokens_per_req = candidates.shape[1]

        predict = self._predict_buf[: bs * num_tokens_per_req]
        accept_index = (
            self._accept_index_buf[: bs * num_tokens_per_req]
            .view(bs, num_tokens_per_req)
            .fill_(-1)
        )
        accept_length = self._accept_length_buf[:bs]

        # Per-draft-position grammar bitmask: buffer shape
        # [bs * num_tokens_per_req, V/32] matches the flat target logits.
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits_output.next_token_logits,
                vocab_mask=sampling_info.vocab_mask,
            )
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).reshape(
            bs, num_tokens_per_req
        )

        _verify_chain_greedy(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates.to(torch.int32),
            target_predict=target_predict,
            batch_size=bs,
            num_draft_tokens=num_tokens_per_req,
            enable_pdl=pdl_enabled(),
        )

        accept_length += 1

        if self.config.enable_output_logprobs:

            write_output_logprobs(
                logits_output, logits_output.next_token_logits, predict
            )

        return predict, accept_length


register_backend("greedy", GreedySamplingBackend)
