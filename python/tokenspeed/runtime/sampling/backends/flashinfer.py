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
    chain_speculative_sampling_target_only,
    verify_chain_greedy,
)
from tokenspeed_kernel.ops.sampling.flashinfer import (
    softmax,
    top_k_top_p_sampling_from_logits,
)
from tokenspeed_kernel.torch_compile import get_compiler_backend

from tokenspeed.runtime.sampling.backends.base import (
    SPECULATIVE_ACCEPT_THRESHOLD_ACC,
    SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.utils import (
    coin_eps,
    nan_guard_logits,
    top_k_top_p_renorm_torch,
    write_output_logprobs,
)
from tokenspeed.runtime.utils import crash_on_warnings
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class FlashInferSamplingBackend(SamplingBackend):
    """Fast fused backend: single-kernel top_k_top_p_sampling_from_logits
    for stochastic single-step sampling; cuda chain kernels (greedy +
    rejection) for multi-step verification.

    Scope is deliberately narrow — temperature / top_k / top_p only — so
    the hot path stays 1 kernel. Requests asking for min_p, penalties, or
    logit_bias are silently ignored; use `flashinfer_full` if any of those
    matter for the workload.
    """

    _HAS_POOL_STATE = True

    def __init__(self, config: SamplingBackendConfig) -> None:

        super().__init__(config)
        self._init_shared_buffers(config)
        self._init_pool_scalars(config)

    def _init_pool_scalars(self, config: SamplingBackendConfig) -> None:
        # Capture warm-up reads row 0 with req_pool_indices zeroed, so row 0
        # must carry neutral-sampling values that can't produce nan/inf.
        pool_rows = config.max_req_pool_size + 1

        self._temperature_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._top_k_pool = torch.ones(
            (pool_rows,), dtype=torch.int32, device=config.device
        )
        self._top_p_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._seed_pool = torch.zeros(
            (pool_rows,), dtype=torch.int64, device=config.device
        )

        # Per-slot CPU-side torch.Generators used to advance speculative
        # coin buffers outside the CUDA graph. Seeded on flip from sp.seed.
        # Slot 0 is pre-filled with _capture_gen so capture warm-up works
        # without any real request having been registered.
        #
        # Retract-resume note: if a request is retracted and later takes a
        # different pool slot on resume, _reset_slot re-seeds a fresh
        # Generator from sp.seed. Sampling stays deterministic given the same
        # seed, and flashinfer's Philox path (seed + seq_len offset) already
        # gives per-step uniqueness independent of the torch.Generator.
        self._generator_per_slot: list[torch.Generator | None] = [None] * pool_rows
        self._generator_per_slot[0] = self._capture_gen
        self._cpu_generator_per_slot: list[torch.Generator | None] = [None] * pool_rows
        self._cpu_generator_per_slot[0] = self._capture_gen

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        self._temperature_pool[pool_idx].fill_(float(sp.temperature))
        self._top_k_pool[pool_idx].fill_(int(sp.top_k))
        self._top_p_pool[pool_idx].fill_(float(sp.top_p))
        self._seed_pool[pool_idx].fill_(int(sp.seed))

        gen = torch.Generator(device=self.config.device)
        gen.manual_seed(int(sp.seed))
        self._generator_per_slot[pool_idx] = gen
        
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(int(sp.seed))
        self._cpu_generator_per_slot[pool_idx] = cpu_gen

    def _init_shared_buffers(self, config: SamplingBackendConfig) -> None:

        # Persistent coin buffers. Filled per-request in prepare() outside the
        # CUDA graph so verify() only reads from them.
        self._coins_buf = torch.zeros(
            (config.max_bs, config.max_draft_tokens_per_req),
            dtype=torch.float32,
            device=config.device,
        )
        self._final_coins_buf = torch.zeros(
            (config.max_bs,), dtype=torch.float32, device=config.device
        )

        self._cpu_coins_buf = torch.empty(
            config.max_bs, config.max_draft_tokens_per_req, dtype=torch.float32, pin_memory=True
        )
        self._cpu_final_coins_buf = torch.empty(config.max_bs, dtype=torch.float32, pin_memory=True)

        # Stub generator used during CUDA-graph capture/warm-up (no requests yet).
        self._capture_gen = torch.Generator(device=config.device)
        self._capture_gen.manual_seed(config.random_seed)

        # Pre-allocated persistent buffers — no per-step alloc in the hot path.
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

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _prepare_step_hook(
        self,
        num_tokens_per_req: int,
        bs: int,
        request_pool_indices: list[int] | None = None,
    ) -> None:
        """Refill persistent coin buffers outside the captured graph.
        request_pool_indices=None is the capture/warm-up path — uses
        _capture_gen for all rows. Otherwise reads per-slot generators
        populated via _reset_slot."""
        n = min(num_tokens_per_req, self.config.max_draft_tokens_per_req)
        lo = coin_eps(self._coins_buf.dtype)

        if bs <= 0:
            return

        if request_pool_indices is None:
            self._coins_buf[:bs, :n].uniform_(lo, 1.0, generator=self._capture_gen)
            self._final_coins_buf[:bs].uniform_(lo, 1.0, generator=self._capture_gen)
            return
            
        cpu_coins = self._cpu_coins_buf[:bs, :n]
        cpu_final = self._cpu_final_coins_buf[:bs]

        for i, pool_idx in enumerate(request_pool_indices):
            # No _reset_slot has run for this slot yet — fall back to
            # the stub generator. Should not happen in well-formed runs
            # because prepare_step's flip detection runs _reset_slot
            # before this hook.
            gen = self._cpu_generator_per_slot[pool_idx] or self._capture_gen
            cpu_coins[i, :n].uniform_(lo, 1.0, generator=gen)
            cpu_final[i].uniform_(lo, 1.0, generator=gen)

        self._coins_buf[:bs, :n].copy_(cpu_coins, non_blocking=True)
        self._final_coins_buf[:bs].copy_(cpu_final, non_blocking=True)

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _gather_scalars(
        self, req_pool_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pool_idx = req_pool_indices.long()
        return (
            self._temperature_pool.index_select(0, pool_idx),
            self._top_k_pool.index_select(0, pool_idx),
            self._top_p_pool.index_select(0, pool_idx),
            self._seed_pool.index_select(0, pool_idx),
        )

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _gather_offsets(self, sampling_info: SamplingBatchInfo) -> torch.Tensor:
        # Philox offset = current seq_len (= prefix + generated tokens so
        # far). Strictly increases across decode steps so consecutive
        # samples from the same request draw different uniforms.
        return sampling_info.valid_cache_lengths.index_select(
            0, sampling_info.req_pool_indices.long()
        ).to(torch.int64)

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
        # pre-bound by bind_grammar_mask_buf; non-grammar rows stay all-ones.
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )

        if sampling_info.is_all_greedy:

            batch_next_token_ids = torch.argmax(logits, -1)

        else:

            temperatures, top_ks, top_ps, seeds = self._gather_scalars(
                sampling_info.req_pool_indices
            )
            offsets = self._gather_offsets(sampling_info)

            # Fuses softmax + top_k + top_p + sample into one kernel; we only
            # need to pre-scale by temperature.
            check_nan = self.config.enable_nan_detection and crash_on_warnings()
            scaled_logits = logits.div_(temperatures.view(-1, 1))

            batch_next_token_ids = top_k_top_p_sampling_from_logits(
                scaled_logits,
                top_ks,
                top_ps,
                filter_apply_order="joint",
                check_nan=check_nan,
                seed=seeds,
                offset=offsets,
            )

        sampled = batch_next_token_ids.to(torch.int32)

        # TP-rank sync: rank 0 wins.
        self.maybe_broadcast(sampled)

        if self.config.enable_output_logprobs:

            write_output_logprobs(logits_output, logits, sampled)

        bs = logits.shape[0]

        return sampled, self._ones_buf[:bs]

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

        if sampling_info.is_all_greedy:

            target_predict = torch.argmax(
                logits_output.next_token_logits, dim=-1
            ).reshape(bs, num_tokens_per_req)

            verify_chain_greedy(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates,
                target_predict=target_predict,
                batch_size=bs,
                num_draft_tokens=num_tokens_per_req,
            )

        else:

            temperatures, top_ks, top_ps, _seeds = self._gather_scalars(
                sampling_info.req_pool_indices
            )

            # Each request's N verified positions share one (temp, top_k, top_p)
            # tuple; flat [bs*N] per-row knobs match the flat [bs*N, vocab] logits.
            n = num_tokens_per_req
            target_probs = softmax(
                logits_output.next_token_logits,
                temperature=torch.repeat_interleave(temperatures, n),
            )
            target_probs = top_k_top_p_renorm_torch(
                target_probs,
                torch.repeat_interleave(top_ks, n),
                torch.repeat_interleave(top_ps, n),
            )
            target_probs = target_probs.reshape(bs, n, -1)

            chain_speculative_sampling_target_only(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates,
                uniform_samples=self._coins_buf[:bs, :n],
                uniform_samples_for_final_sampling=self._final_coins_buf[:bs],
                target_probs=target_probs,
                draft_probs=torch.zeros_like(target_probs),
                threshold_single=SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
                threshold_acc=SPECULATIVE_ACCEPT_THRESHOLD_ACC,
            )

        accept_length += 1

        # TP-rank sync: rank 0 wins on the full verify-output triple.
        self.maybe_broadcast(predict, accept_index, accept_length)

        if self.config.enable_output_logprobs:

            write_output_logprobs(
                logits_output, logits_output.next_token_logits, predict
            )

        return predict, accept_length


register_backend("flashinfer", FlashInferSamplingBackend)
