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
import torch.distributed as dist
from tokenspeed_kernel.ops.sampling.cuda import (
    chain_speculative_sampling_target_only,
    fused_topk_topp_prepare,
    fused_topk_topp_renorm,
    verify_chain_greedy,
)
from tokenspeed_kernel.ops.sampling.cute_dsl import argmax as cute_argmax
from tokenspeed_kernel.ops.sampling.flashinfer import (
    softmax,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
)
from tokenspeed_kernel.ops.sampling.triton import gather_and_expand_scalars
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.torch_compile import get_compiler_backend

# Resolved once at import: the fused top-k + top-p kernel is NVIDIA-only.
# On non-NVIDIA platforms (e.g. ROCm) we fall back to the back-to-back
# flashinfer renorm calls. Defining this at module scope keeps the hot path
# branch-free in the captured graph.
_FUSED_TOPK_TOPP_AVAILABLE = current_platform().is_nvidia

from tokenspeed.runtime.distributed.dp_sampling_comm import DpSamplingComm
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.sampling.backends.base import (
    SPECULATIVE_ACCEPT_THRESHOLD_ACC,
    SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.utils import (
    coin_eps,
    gather_token_logprobs_torch,
    nan_guard_logits,
)
from tokenspeed.runtime.utils import crash_on_warnings
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class FlashInferSamplingBackend(SamplingBackend):
    """Fast backend: fused softmax(temperature) + top_k_top_p_sampling_from_probs
    for stochastic single-step sampling; cuda chain kernels (greedy +
    rejection) for multi-step verification.

    Scope is deliberately narrow — temperature / top_k / top_p only —
    keeping the hot path to 2 kernels. Requests asking for min_p, penalties,
    or logit_bias are silently ignored; use `flashinfer_full` if any of those
    matter for the workload.
    """

    _HAS_POOL_STATE = True
    _SUPPORTS_DP_VERIFY = True

    def __init__(self, config: SamplingBackendConfig) -> None:

        super().__init__(config)
        self._init_dp_geometry(config)
        self._init_shared_buffers(config)
        self._init_pool_scalars(config)
        # Pre-create the side stream used by fused_topk_topp_renorm. Must
        # happen before any CUDA graph capture — cudaStreamCreate is illegal
        # inside capture, and verify() runs from the captured graph.
        fused_topk_topp_prepare(config.device)

    def _init_dp_geometry(self, config: SamplingBackendConfig) -> None:
        tp_group = config.tp_group
        tp_size = len(tp_group) if tp_group is not None else 1

        self._dp_tp_size = tp_size
        self._dp_tp_group = tp_group
        self._dp_pg = None
        self._dp_rank = 0
        self._dp_comm: DpSamplingComm | None = None
        self._dp_comm_vocab_size = 0

        if tp_size > 1 and config.dp_sampling:
            self._dp_max_pad_bs = ((config.max_bs + tp_size - 1) // tp_size) * tp_size
            self._dp_max_reqs_per_rank = self._dp_max_pad_bs // tp_size

            self._dp_pg = pg_manager.get_process_group("nccl", tp_group)

            self._dp_rank = dist.get_rank(group=self._dp_pg)

            self._dp_comm_vocab_size = (
                (max(config.vocab_size, tp_size) + tp_size - 1) // tp_size
            ) * tp_size
            self._dp_comm = self._make_dp_comm(self._dp_comm_vocab_size, config)
        else:
            self._dp_max_pad_bs = config.max_bs
            self._dp_max_reqs_per_rank = config.max_bs

    def _make_dp_comm(
        self, vocab_size: int, config: SamplingBackendConfig
    ) -> DpSamplingComm:
        assert self._dp_tp_group is not None
        return DpSamplingComm(
            tp_size=self._dp_tp_size,
            rank=self._dp_rank,
            group=self._dp_tp_group,
            max_pad_bs=self._dp_max_pad_bs,
            num_tokens_per_req=config.max_draft_tokens_per_req,
            vocab_size=vocab_size,
            logits_dtype=None,
            device=config.device,
        )

    def configure_dp_sampling_vocab_size(self, vocab_size: int) -> None:
        """Use the target LM-head padded vocab size for DP logits exchange."""
        if self._dp_comm is None:
            return
        if vocab_size == self._dp_comm_vocab_size:
            return
        if vocab_size % self._dp_tp_size != 0:
            raise RuntimeError(
                f"DP sampling vocab_size={vocab_size} must be divisible by "
                f"tp_size={self._dp_tp_size}"
            )
        if self._dp_comm.is_initialized:
            raise RuntimeError("Cannot resize DP sampling comm after use")
        self._dp_comm_vocab_size = vocab_size
        self._dp_comm = self._make_dp_comm(vocab_size, self.config)

    @staticmethod
    def _slice_dp_vocab_mask(
        vocab_mask: torch.Tensor | None,
        *,
        full_bs: int,
        pad_bs: int,
        num_tokens_per_req: int,
        shard: slice,
    ) -> torch.Tensor | None:
        if vocab_mask is None:
            return None
        n = num_tokens_per_req
        if pad_bs > full_bs:
            vocab_mask = torch.nn.functional.pad(
                vocab_mask,
                (0, 0, 0, (pad_bs - full_bs) * n),
                value=-1,
            )
        return vocab_mask.view(pad_bs, n, -1)[shard].reshape(-1, vocab_mask.shape[-1])

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

        max_pad_bs = self._dp_max_pad_bs
        max_n = config.max_draft_tokens_per_req
        max_reqs_per_rank = self._dp_max_reqs_per_rank

        # Persistent coin buffers. Filled per-request in prepare() outside the
        # CUDA graph so verify() only reads from them.
        self._coins_buf = torch.zeros(
            (max_pad_bs, max_n),
            dtype=torch.float32,
            device=config.device,
        )
        self._final_coins_buf = torch.zeros(
            (max_pad_bs,), dtype=torch.float32, device=config.device
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
            (max_pad_bs,), dtype=torch.int32, device=config.device
        )
        self._predict_buf = torch.zeros(
            (max_pad_bs * max_n,),
            dtype=torch.int32,
            device=config.device,
        )
        # Flat layout so [:bs * n].view(bs, n) is contiguous for any bs/n
        # (required by maybe_broadcast / NCCL).
        self._accept_index_buf = torch.zeros(
            (max_pad_bs * max_n,),
            dtype=torch.int32,
            device=config.device,
        )
        self._accept_length_buf = torch.zeros(
            (max_pad_bs,), dtype=torch.int32, device=config.device
        )

        self._predict_local_buf = torch.zeros(
            (max_reqs_per_rank * max_n,), dtype=torch.int32, device=config.device
        )
        self._accept_index_local_buf = torch.zeros(
            (max_reqs_per_rank * max_n,), dtype=torch.int32, device=config.device
        )
        self._accept_length_local_buf = torch.zeros(
            (max_reqs_per_rank,), dtype=torch.int32, device=config.device
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

            temperatures, top_ks, top_ps, _, seeds, offsets = gather_and_expand_scalars(
                sampling_info.req_pool_indices,
                temperature=self._temperature_pool,
                top_k=self._top_k_pool,
                top_p=self._top_p_pool,
                seed=self._seed_pool,
                offsets=sampling_info.valid_cache_lengths,
                enable_pdl=pdl_enabled(),
            )

            check_nan = self.config.enable_nan_detection and crash_on_warnings()
            probs = softmax(
                logits,
                temperature=temperatures.view(-1, 1),
                enable_pdl=pdl_enabled(),
            )
            batch_next_token_ids = top_k_top_p_sampling_from_probs(
                probs,
                top_ks,
                top_ps,
                filter_apply_order="joint",
                check_nan=check_nan,
                seed=seeds,
                offset=offsets,
                deterministic=True,
            )

        sampled = batch_next_token_ids.to(torch.int32)

        # TP-rank sync: rank 0 wins.
        self.maybe_broadcast(sampled)

        if self.config.enable_output_logprobs:
            logits_output.next_token_logprobs = gather_token_logprobs_torch(
                logits, sampled
            )

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
        vocab_mask = sampling_info.vocab_mask

        if sampling_info.dp_sampling:
            assert (
                self._dp_tp_size > 1 and self._dp_pg is not None
            ), "dp_sampling requires tp_size > 1 and a resolved tp_group"
            tp_size = self._dp_tp_size
            rank = self._dp_rank
            full_bs = bs
            pad_bs = ((bs + tp_size - 1) // tp_size) * tp_size
            assert (
                pad_bs <= self._dp_max_pad_bs
            ), f"pad_bs={pad_bs} exceeds dp_max_pad_bs={self._dp_max_pad_bs}"
            bs = pad_bs // tp_size

            # Shard by request so each request's draft chain stays on one rank.
            shard = slice(rank * bs, (rank + 1) * bs)
            if pad_bs > full_bs:
                candidates = torch.nn.functional.pad(
                    candidates, (0, 0, 0, pad_bs - full_bs)
                )[shard]
                pool_indices = torch.nn.functional.pad(
                    sampling_info.req_pool_indices, (0, pad_bs - full_bs)
                )[shard]
            else:
                candidates = candidates[shard]
                pool_indices = sampling_info.req_pool_indices[shard]
            vocab_mask = self._slice_dp_vocab_mask(
                vocab_mask,
                full_bs=full_bs,
                pad_bs=pad_bs,
                num_tokens_per_req=num_tokens_per_req,
                shard=shard,
            )
            coins = self._coins_buf[shard]
            final_coins = self._final_coins_buf[shard]
            predict = self._predict_local_buf[: bs * num_tokens_per_req]
            accept_index = (
                self._accept_index_local_buf[: bs * num_tokens_per_req]
                .view(bs, num_tokens_per_req)
                .fill_(-1)
            )
            accept_length = self._accept_length_local_buf[:bs]
        else:
            pool_indices = sampling_info.req_pool_indices
            coins = self._coins_buf
            final_coins = self._final_coins_buf
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
        if vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits,
                vocab_mask=vocab_mask,
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
                pool_indices,
                temperature=self._temperature_pool,
                top_k=self._top_k_pool,
                top_p=self._top_p_pool,
                n=n,
                enable_pdl=pdl_enabled(),
            )

            target_probs = softmax(
                logits,
                temperature=temperatures,
                enable_pdl=pdl_enabled(),
            )
            if _FUSED_TOPK_TOPP_AVAILABLE:
                # Fused replacement for the back-to-back top_k_renorm_prob +
                # top_p_renorm_prob(is_deterministic=True) pair. Sentinel
                # K = 1<<30 in top_ks routes per-row through the radix top-p
                # only path.
                target_probs = fused_topk_topp_renorm(target_probs, top_ks, top_ps)
            else:
                target_probs = top_k_renorm_prob(target_probs, top_ks)
                target_probs = top_p_renorm_prob(
                    target_probs, top_ps, is_deterministic=True
                )
            target_probs = target_probs.reshape(bs, n, -1)

            chain_speculative_sampling_target_only(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates,
                uniform_samples=coins[:bs, :n],
                uniform_samples_for_final_sampling=final_coins[:bs],
                target_probs=target_probs,
                draft_probs=None,
                threshold_single=SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
                threshold_acc=SPECULATIVE_ACCEPT_THRESHOLD_ACC,
                deterministic=not sampling_info.dp_sampling,
                enable_pdl=pdl_enabled(),
            )

        accept_length += 1
        logprobs_local = None
        if self.config.enable_output_logprobs and sampling_info.dp_sampling:
            # DP verify logits are still sharded by request at this point.
            # Compute scalar logprobs for local predictions before gathering
            # predictions to full-batch shape; the non-DP writer requires
            # matching logits/token row counts.
            logprobs_local = gather_token_logprobs_torch(logits, predict).view(
                bs, num_tokens_per_req
            )

        if sampling_info.dp_sampling:
            n = num_tokens_per_req
            assert self._dp_comm is not None
            self._dp_comm.prepare_verify_outputs(logits_output.next_token_logits.dtype)
            (
                predict_full,
                accept_index_full,
                accept_length_full,
            ) = self._dp_comm.gather_verify_outputs(
                predict_local=predict.view(bs, n),
                accept_index_local=accept_index,
                accept_length_local=accept_length,
                pad_bs=pad_bs,
            )
            predict = predict_full.view(-1)[: full_bs * n]
            accept_index = accept_index_full[:full_bs]
            accept_length = accept_length_full[:full_bs]
            if logprobs_local is not None:
                logprobs_full = self._dp_comm.gather_verify_logprobs(
                    logprobs_local,
                    pad_bs=pad_bs,
                )
                logits_output.next_token_logprobs = logprobs_full.view(-1)[
                    : full_bs * n
                ]
        # TP-rank sync: rank 0 wins on the full verify-output triple.
        # Load-bearing: flashinfer top_k_renorm_prob has no is_deterministic
        # knob and produces non-bit-identical results across ranks (sub-ulp
        # FP accumulation order).
        # For fused top-k + top-p, the results are bit-identical across ranks.
        # So we don't need to broadcast the results.
        elif pdl_enabled():
            self.maybe_broadcast(predict, accept_index, accept_length)
        elif not _FUSED_TOPK_TOPP_AVAILABLE:
            self.maybe_broadcast(predict, accept_index, accept_length)

        if self.config.enable_output_logprobs and not sampling_info.dp_sampling:
            logits_output.next_token_logprobs = gather_token_logprobs_torch(
                logits, predict
            )

        return predict, accept_length


register_backend("flashinfer", FlashInferSamplingBackend)
