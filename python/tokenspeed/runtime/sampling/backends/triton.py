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
from tokenspeed_kernel.ops.sampling.cute_dsl import argmax as cute_argmax
from tokenspeed_kernel.ops.sampling.probability import (
    build_top_k_top_p_probs_from_logits,
)
from tokenspeed_kernel.ops.sampling.triton import (
    gather_and_expand_scalars,
    gumbel_sample_from_pools,
    gumbel_sample_from_pools_compact,
    sample_rejection_from_pools,
    sample_top_k_top_p_from_pools,
    sample_top_k_top_p_from_pools_compact,
)
from tokenspeed_kernel.torch_compile import get_compiler_backend

_GUMBEL_BLOCK_SIZE = 1024
_TOP_K_FILTER_MAX_K = 128
_TOP_K_TOP_P_DIRECT_MAX_K = 64
_COMPACT_VOCAB_MAX_SIZE = 32768
_COMPACT_GUMBEL_BLOCK_SIZE = 4096
_COMPACT_TOP_K_TOP_P_BLOCK_SIZE = 4096
_TRITON_GUMBEL_MODE_NONE = 0
_TRITON_GUMBEL_MODE_NO_FILTER = 1
_TRITON_GUMBEL_MODE_TOP_K_TOP_P = 2
_TRITON_GUMBEL_MODE_GENERIC_REJECTION = 3
_TRITON_GUMBEL_MODE_GREEDY = 4
_TOP_P_REJECTION_TRIES = 4
_CUDA_GRAPH_VARIANT_DEFAULT = "default"
_CUDA_GRAPH_VARIANT_GREEDY = "greedy"
_CUDA_GRAPH_VARIANT_NO_FILTER = "no_filter"
_CUDA_GRAPH_VARIANT_TOP_K_TOP_P = "top_k_top_p"

from tokenspeed.runtime.sampling.backends.base import (
    SPECULATIVE_ACCEPT_THRESHOLD_ACC,
    SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.sampling_params import _SAMPLING_EPS, _TOP_K_DISABLED
from tokenspeed.runtime.sampling.utils import (
    coin_eps,
    nan_guard_logits,
    write_output_logprobs,
)
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


def _gumbel_num_blocks(vocab_size: int) -> int:
    return (vocab_size + _GUMBEL_BLOCK_SIZE - 1) // _GUMBEL_BLOCK_SIZE


class TritonSamplingBackend(SamplingBackend):
    """TokenSpeed-native Triton sampling backend.

    Scope is deliberately narrow — temperature / top_k / top_p for normal
    sampling, plus CUDA chain kernels for multi-step verification. Requests
    asking for min_p, penalties, or logit_bias need the full backend.
    The sampler adapts vLLM MRV2's GPU-native/Gumbel-Max direction while
    preserving TokenSpeed's existing pool slots and prepare_step lifecycle.
    """

    _HAS_POOL_STATE = True

    def __init__(self, config: SamplingBackendConfig) -> None:

        super().__init__(config)
        self._init_shared_buffers(config)
        self._init_pool_scalars(config)
        self._step_triton_gumbel_mode = _TRITON_GUMBEL_MODE_NONE

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
        # seed, and Philox path (seed + seq_len offset) already
        # gives per-step uniqueness independent of the torch.Generator.
        self._cpu_generator_per_slot: list[torch.Generator | None] = [None] * pool_rows
        self._cpu_generator_per_slot[0] = self._capture_gen

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        self._temperature_pool[pool_idx].fill_(float(sp.temperature))
        self._top_k_pool[pool_idx].fill_(int(sp.top_k))
        self._top_p_pool[pool_idx].fill_(float(sp.top_p))
        self._seed_pool[pool_idx].fill_(int(sp.seed))

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
            config.max_bs,
            config.max_draft_tokens_per_req,
            dtype=torch.float32,
            pin_memory=True,
        )
        self._cpu_final_coins_buf = torch.empty(
            config.max_bs, dtype=torch.float32, pin_memory=True
        )

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
        gumbel_blocks = max(1, _gumbel_num_blocks(config.vocab_size))
        self._gumbel_local_ids = torch.empty(
            (config.max_bs, gumbel_blocks),
            dtype=torch.int32,
            device=config.device,
        )
        self._gumbel_local_scores = torch.empty(
            (config.max_bs, gumbel_blocks),
            dtype=torch.float32,
            device=config.device,
        )
        self._gumbel_out = torch.empty(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )
        self._top_k_local_values = torch.empty(
            (config.max_bs, gumbel_blocks, _TOP_K_FILTER_MAX_K),
            dtype=torch.float32,
            device=config.device,
        )
        self._top_k_selected_values = torch.empty(
            (config.max_bs,), dtype=torch.float32, device=config.device
        )
        self._rejection_local_ids = torch.empty(
            (config.max_bs, _TOP_P_REJECTION_TRIES, gumbel_blocks),
            dtype=torch.int32,
            device=config.device,
        )
        self._rejection_local_scores = torch.empty(
            (config.max_bs, _TOP_P_REJECTION_TRIES, gumbel_blocks),
            dtype=torch.float32,
            device=config.device,
        )
        self._rejection_argmax_ids = torch.empty(
            (config.max_bs, gumbel_blocks),
            dtype=torch.int32,
            device=config.device,
        )
        self._rejection_argmax_scores = torch.empty(
            (config.max_bs, gumbel_blocks),
            dtype=torch.float32,
            device=config.device,
        )
        self._rejection_candidate_ids = torch.empty(
            (config.max_bs, _TOP_P_REJECTION_TRIES + 1),
            dtype=torch.int32,
            device=config.device,
        )
        self._rejection_candidate_logits = torch.empty(
            (config.max_bs, _TOP_P_REJECTION_TRIES + 1),
            dtype=torch.float32,
            device=config.device,
        )
        self._rejection_top_k_total_probs = torch.empty(
            (config.max_bs,),
            dtype=torch.float32,
            device=config.device,
        )
        self._rejection_local_total_probs = torch.empty(
            (config.max_bs, gumbel_blocks),
            dtype=torch.float32,
            device=config.device,
        )
        self._rejection_local_before_probs = torch.empty(
            (config.max_bs, _TOP_P_REJECTION_TRIES, gumbel_blocks),
            dtype=torch.float32,
            device=config.device,
        )
        self._rejection_local_before_counts = torch.empty(
            (config.max_bs, _TOP_P_REJECTION_TRIES, gumbel_blocks),
            dtype=torch.int32,
            device=config.device,
        )

    def _resolve_triton_gumbel_mode(
        self,
        sampling_params_list: list[SamplingParams],
        num_tokens_per_req: int,
    ) -> int:
        if len(sampling_params_list) == 0:
            return _TRITON_GUMBEL_MODE_NONE
        top_ks = [int(sp.top_k) for sp in sampling_params_list]
        if num_tokens_per_req != 1:
            if all(top_k <= 1 for top_k in top_ks):
                return _TRITON_GUMBEL_MODE_GREEDY
            return _TRITON_GUMBEL_MODE_NONE
        top_ps = [float(sp.top_p) for sp in sampling_params_list]
        if all(top_k == _TOP_K_DISABLED for top_k in top_ks):
            if all(abs(top_p - 1.0) <= _SAMPLING_EPS for top_p in top_ps):
                return _TRITON_GUMBEL_MODE_NO_FILTER
            return _TRITON_GUMBEL_MODE_GENERIC_REJECTION
        if all(1 <= top_k < _TOP_K_FILTER_MAX_K for top_k in top_ks):
            can_use_direct_top_k_top_p = all(
                top_k <= _TOP_K_TOP_P_DIRECT_MAX_K for top_k in top_ks
            )
            if can_use_direct_top_k_top_p:
                return _TRITON_GUMBEL_MODE_TOP_K_TOP_P
            return _TRITON_GUMBEL_MODE_GENERIC_REJECTION
        return _TRITON_GUMBEL_MODE_GENERIC_REJECTION

    def prepare_step(
        self,
        request_ids: list[str],
        request_pool_indices: list[int],
        sampling_params_list: list[SamplingParams],
        num_tokens_per_req: int = 1,
    ) -> None:
        super().prepare_step(
            request_ids=request_ids,
            request_pool_indices=request_pool_indices,
            sampling_params_list=sampling_params_list,
            num_tokens_per_req=num_tokens_per_req,
        )
        self._step_triton_gumbel_mode = self._resolve_triton_gumbel_mode(
            sampling_params_list, num_tokens_per_req
        )

    def cuda_graph_capture_variants(
        self, num_tokens_per_req: int = 1
    ) -> tuple[str, ...]:
        if not self.config.enable_output_logprobs and num_tokens_per_req == 1:
            variants = (
                _CUDA_GRAPH_VARIANT_DEFAULT,
                _CUDA_GRAPH_VARIANT_NO_FILTER,
                _CUDA_GRAPH_VARIANT_TOP_K_TOP_P,
            )
            return variants
        return (_CUDA_GRAPH_VARIANT_DEFAULT, _CUDA_GRAPH_VARIANT_GREEDY)

    def cuda_graph_replay_variant(self) -> str:
        if self._step_triton_gumbel_mode == _TRITON_GUMBEL_MODE_GREEDY:
            return _CUDA_GRAPH_VARIANT_GREEDY
        if self._step_triton_gumbel_mode == _TRITON_GUMBEL_MODE_NO_FILTER:
            return _CUDA_GRAPH_VARIANT_NO_FILTER
        if self._step_triton_gumbel_mode == _TRITON_GUMBEL_MODE_TOP_K_TOP_P:
            return _CUDA_GRAPH_VARIANT_TOP_K_TOP_P
        return _CUDA_GRAPH_VARIANT_DEFAULT

    def cuda_graph_capture_is_all_greedy(
        self,
        capture_variant: str,
        num_tokens_per_req: int = 1,
    ) -> bool:
        return num_tokens_per_req != 1 and capture_variant == _CUDA_GRAPH_VARIANT_GREEDY

    def prepare_capture(
        self,
        bs: int,
        num_tokens_per_req: int = 1,
        capture_variant: str | None = None,
    ) -> None:
        if num_tokens_per_req != 1 and capture_variant == _CUDA_GRAPH_VARIANT_GREEDY:
            self._step_triton_gumbel_mode = _TRITON_GUMBEL_MODE_GREEDY
        elif num_tokens_per_req == 1:
            if capture_variant == _CUDA_GRAPH_VARIANT_NO_FILTER:
                self._step_triton_gumbel_mode = _TRITON_GUMBEL_MODE_NO_FILTER
            elif capture_variant == _CUDA_GRAPH_VARIANT_TOP_K_TOP_P:
                self._step_triton_gumbel_mode = _TRITON_GUMBEL_MODE_TOP_K_TOP_P
            else:
                self._step_triton_gumbel_mode = _TRITON_GUMBEL_MODE_GENERIC_REJECTION
        else:
            self._step_triton_gumbel_mode = _TRITON_GUMBEL_MODE_NONE
        super().prepare_capture(
            bs=bs,
            num_tokens_per_req=num_tokens_per_req,
            capture_variant=capture_variant,
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

            batch_next_token_ids = cute_argmax(logits)

        else:

            triton_gumbel_mode = self._step_triton_gumbel_mode
            if triton_gumbel_mode != _TRITON_GUMBEL_MODE_NONE:
                bs = logits.shape[0]
                if triton_gumbel_mode == _TRITON_GUMBEL_MODE_NO_FILTER:
                    if self.config.vocab_size <= _COMPACT_VOCAB_MAX_SIZE:
                        batch_next_token_ids = gumbel_sample_from_pools_compact(
                            logits,
                            sampling_info.req_pool_indices,
                            self._temperature_pool,
                            self._seed_pool,
                            sampling_info.valid_cache_lengths,
                            self._gumbel_out[:bs],
                            block_size=_COMPACT_GUMBEL_BLOCK_SIZE,
                        )
                    else:
                        batch_next_token_ids = gumbel_sample_from_pools(
                            logits,
                            sampling_info.req_pool_indices,
                            self._temperature_pool,
                            self._seed_pool,
                            sampling_info.valid_cache_lengths,
                            self._gumbel_local_ids[:bs],
                            self._gumbel_local_scores[:bs],
                            self._gumbel_out[:bs],
                        )
                elif triton_gumbel_mode == _TRITON_GUMBEL_MODE_TOP_K_TOP_P:
                    if self.config.vocab_size <= _COMPACT_VOCAB_MAX_SIZE:
                        batch_next_token_ids = sample_top_k_top_p_from_pools_compact(
                            logits,
                            sampling_info.req_pool_indices,
                            self._temperature_pool,
                            self._top_k_pool,
                            self._top_p_pool,
                            self._seed_pool,
                            sampling_info.valid_cache_lengths,
                            self._top_k_local_values[:bs],
                            self._gumbel_out[:bs],
                            block_size=_COMPACT_TOP_K_TOP_P_BLOCK_SIZE,
                        )
                    else:
                        batch_next_token_ids = sample_top_k_top_p_from_pools(
                            logits,
                            sampling_info.req_pool_indices,
                            self._temperature_pool,
                            self._top_k_pool,
                            self._top_p_pool,
                            self._seed_pool,
                            sampling_info.valid_cache_lengths,
                            self._top_k_local_values[:bs],
                            self._top_k_selected_values[:bs],
                            self._gumbel_out[:bs],
                        )
                elif triton_gumbel_mode == _TRITON_GUMBEL_MODE_GENERIC_REJECTION:
                    batch_next_token_ids = sample_rejection_from_pools(
                        logits,
                        sampling_info.req_pool_indices,
                        self._temperature_pool,
                        self._top_k_pool,
                        self._top_p_pool,
                        self._seed_pool,
                        sampling_info.valid_cache_lengths,
                        self._rejection_local_ids[:bs],
                        self._rejection_local_scores[:bs],
                        self._rejection_argmax_ids[:bs],
                        self._rejection_argmax_scores[:bs],
                        self._rejection_candidate_ids[:bs],
                        self._rejection_candidate_logits[:bs],
                        self._top_k_local_values[:bs],
                        self._rejection_top_k_total_probs[:bs],
                        self._rejection_local_total_probs[:bs],
                        self._rejection_local_before_probs[:bs],
                        self._rejection_local_before_counts[:bs],
                        self._gumbel_out[:bs],
                    )
                else:
                    raise RuntimeError(
                        f"Unknown Triton sampling mode {triton_gumbel_mode}"
                    )
            else:
                raise RuntimeError(
                    "Triton sampling backend did not select a sampling mode "
                    "for this non-greedy step"
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
        logits = nan_guard_logits(
            logits_output.next_token_logits, self.config.enable_nan_detection
        )

        # Per-draft-position grammar bitmask: buffer shape
        # [bs * num_tokens_per_req, V/32] matches the flat target logits.
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits,
                vocab_mask=sampling_info.vocab_mask,
            )

        if sampling_info.is_all_greedy:

            target_predict = cute_argmax(logits).reshape(bs, num_tokens_per_req)

            verify_chain_greedy(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates,
                target_predict=target_predict,
                batch_size=bs,
                num_draft_tokens=num_tokens_per_req,
                enable_pdl=pdl_enabled(),
            )

        else:

            # Each request's N verified positions share one (temp, top_k, top_p)
            # tuple; flat [bs*N] per-row knobs match the flat [bs*N, vocab] logits.
            n = num_tokens_per_req
            temperatures, top_ks, top_ps, _, _, _ = gather_and_expand_scalars(
                sampling_info.req_pool_indices,
                temperature=self._temperature_pool,
                top_k=self._top_k_pool,
                top_p=self._top_p_pool,
                n=n,
                enable_pdl=pdl_enabled(),
            )

            target_probs = build_top_k_top_p_probs_from_logits(
                logits,
                temperatures,
                top_ks,
                top_ps,
                enable_pdl=pdl_enabled(),
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
                draft_probs=None,
                threshold_single=SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
                threshold_acc=SPECULATIVE_ACCEPT_THRESHOLD_ACC,
                deterministic=True,
                enable_pdl=pdl_enabled(),
            )

        accept_length += 1

        # TP-rank sync: rank 0 wins on the full verify-output triple.
        self.maybe_broadcast(predict, accept_index, accept_length)

        if self.config.enable_output_logprobs:

            write_output_logprobs(logits_output, logits, predict)

        return predict, accept_length


register_backend("triton", TritonSamplingBackend)
