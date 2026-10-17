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

"""Full-featured, graph-safe Ascend sampling backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling.ascend import (
    SamplingWorkspace,
    chain_speculative_sampling_target_only,
    fused_topk_topp_renorm,
    min_p_renorm_prob,
    sample_from_probs,
)
from tokenspeed_kernel.ops.sampling.triton import gather_and_expand_scalars

from tokenspeed.runtime.sampling.backends.base import (
    SPECULATIVE_ACCEPT_THRESHOLD_ACC,
    SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.full_base import FullSamplingBackendBase
from tokenspeed.runtime.sampling.dp_sampling_config import slice_dp_vocab_mask
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.utils import gather_token_logprobs, nan_guard_logits
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.tree_verify import TreeVerifyBatch


class AscendFullSamplingBackend(FullSamplingBackendBase):
    """Ascend sampler with full parameter and Batch-DP verify support.

    Random coins are populated by :class:`PooledSamplingBackend` outside the
    captured graph. The current correctness path materializes dense FP32
    probabilities; compact finite-top-k kernels can replace `_sampling_probs`
    without changing backend state or DP communication.
    """

    _SUPPORTS_DP_VERIFY = True
    supports_tree_verify = False

    def _create_capture_generator(
        self, config: SamplingBackendConfig
    ) -> torch.Generator:
        del config
        return torch.Generator(device="cpu")

    def _fill_capture_coins(self, bs: int, n: int, lo: float) -> None:
        cpu_coins = torch.empty((bs, n), dtype=torch.float32, pin_memory=True)
        cpu_final = torch.empty((bs,), dtype=torch.float32, pin_memory=True)
        cpu_coins.uniform_(lo, 1.0, generator=self._capture_gen)
        cpu_final.uniform_(lo, 1.0, generator=self._capture_gen)
        self._coins_buf[:bs, :n].copy_(cpu_coins, non_blocking=True)
        self._final_coins_buf[:bs].copy_(cpu_final, non_blocking=True)

    def __init__(self, config: SamplingBackendConfig) -> None:
        super().__init__(config)
        self._sampling_workspace = SamplingWorkspace.create(
            max_rows=self._dp_max_pad_bs,
            vocab_size=config.vocab_size,
            device=config.device,
        )

    def _gather_sampling_scalars(
        self,
        pool_indices: torch.Tensor,
        *,
        num_tokens_per_req: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temperatures, top_ks, top_ps, min_ps, _, _ = gather_and_expand_scalars(
            pool_indices,
            temperature=self._temperature_pool,
            top_k=self._top_k_pool,
            top_p=self._top_p_pool,
            min_p=self._min_p_pool,
            n=num_tokens_per_req,
            enable_pdl=False,
        )
        assert min_ps is not None
        return temperatures, top_ks, top_ps, min_ps

    def _sampling_probs(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
    ) -> torch.Tensor:
        scaled_logits = logits.float() / temperatures.float().unsqueeze(-1)
        probs = fused_topk_topp_renorm(
            scaled_logits,
            top_ks.to(torch.int32),
            top_ps.float(),
        )
        return min_p_renorm_prob(probs, min_ps.float())

    @staticmethod
    def _accepted_prefix_mask(
        accept_length: torch.Tensor,
        num_tokens_per_req: int,
    ) -> torch.Tensor:
        positions = torch.arange(
            num_tokens_per_req,
            dtype=accept_length.dtype,
            device=accept_length.device,
        )
        return positions.unsqueeze(0) < accept_length.unsqueeze(1)

    def _accumulate_accepted_prefix(
        self,
        pool_indices: torch.Tensor,
        predict: torch.Tensor,
        accept_length: torch.Tensor,
        num_tokens_per_req: int,
    ) -> None:
        tokens = predict.view(-1, num_tokens_per_req)
        valid = self._accepted_prefix_mask(accept_length, num_tokens_per_req)
        expanded_pool_indices = (
            pool_indices.unsqueeze(1).expand(-1, num_tokens_per_req).reshape(-1)
        )
        self._accumulate_counts(
            expanded_pool_indices,
            tokens.reshape(-1),
            valid.reshape(-1).to(torch.int32),
        )

    @nvtx_range("sampling:sample", color="yellow")
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = nan_guard_logits(
            logits_output.next_token_logits, self.config.enable_nan_detection
        ).float()
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )

        raw_logits = logits.clone() if self.config.enable_output_logprobs else None
        pool_indices = sampling_info.req_pool_indices
        logits = self._apply_penalties_and_bias_for_indices(logits, pool_indices)
        temperatures, top_ks, top_ps, min_ps = self._gather_sampling_scalars(
            pool_indices
        )
        probs = self._sampling_probs(logits, temperatures, top_ks, top_ps, min_ps)

        bs = logits.shape[0]
        row0 = sampling_info.batch_row_offset
        sampled = self._predict_buf[:bs]
        sample_from_probs(
            probs,
            self._coins_buf[row0 : row0 + bs, 0],
            self._sampling_workspace,
            out=sampled,
        )
        lengths = self._accept_length_buf[:bs]
        lengths.fill_(1)
        self.maybe_broadcast(sampled)

        if raw_logits is not None:
            logits_output.next_token_logprobs = gather_token_logprobs(
                raw_logits, sampled.long(), logprob_order=self.config.logprob_order
            )
        self._accumulate_counts(
            pool_indices,
            sampled,
            torch.ones_like(sampled, dtype=torch.int32),
        )
        return sampled, lengths

    @nvtx_range("sampling:verify", color="yellow")
    def verify(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
        *,
        tree: TreeVerifyBatch | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tree is not None:
            raise NotImplementedError("Ascend sampling supports chain verify only")
        effective_bs = candidates.shape[0]
        num_tokens_per_req = candidates.shape[1]
        vocab_mask = sampling_info.vocab_mask
        logits_layout_plan = getattr(logits_output, "logits_layout_plan", None)
        dp_sampling = logits_layout_plan is not None

        if dp_sampling:
            if self._dp_comm is None:
                raise RuntimeError("DP sampling communication was not configured")
            pad_bs = logits_layout_plan.bucket_bs
            if logits_layout_plan.effective_bs != effective_bs:
                raise RuntimeError(
                    f"DP sampling effective_bs={logits_layout_plan.effective_bs} "
                    f"does not match candidate batch size {effective_bs}"
                )
            if (
                pad_bs < effective_bs
                or pad_bs > self._dp_max_pad_bs
                or pad_bs % self._dp_tp_size != 0
            ):
                raise RuntimeError(
                    f"invalid DP sampling pad_bs={pad_bs}, "
                    f"effective_bs={effective_bs}, tp_size={self._dp_tp_size}"
                )
            local_bs = pad_bs // self._dp_tp_size
            shard = slice(self._dp_rank * local_bs, (self._dp_rank + 1) * local_bs)
            if pad_bs > effective_bs:
                candidates_local = torch.nn.functional.pad(
                    candidates, (0, 0, 0, pad_bs - effective_bs)
                )[shard]
                pool_indices_local = torch.nn.functional.pad(
                    sampling_info.req_pool_indices, (0, pad_bs - effective_bs)
                )[shard]
            else:
                candidates_local = candidates[shard]
                pool_indices_local = sampling_info.req_pool_indices[shard]
            vocab_mask = slice_dp_vocab_mask(
                vocab_mask,
                full_bs=effective_bs,
                pad_bs=pad_bs,
                num_tokens_per_req=num_tokens_per_req,
                shard=shard,
            )
            coins = self._coins_buf[shard, :num_tokens_per_req]
            final_coins = self._final_coins_buf[shard]
            if (
                self._predict_local_buf is None
                or self._accept_index_local_buf is None
                or self._accept_length_local_buf is None
            ):
                raise RuntimeError("DP sampling verify buffers are not initialized")
            predict_local = self._predict_local_buf[: local_bs * num_tokens_per_req]
            accept_index_local = self._accept_index_local_buf[
                : local_bs * num_tokens_per_req
            ].view(local_bs, num_tokens_per_req)
            accept_length_local = self._accept_length_local_buf[:local_bs]
        else:
            pad_bs = effective_bs
            local_bs = effective_bs
            candidates_local = candidates
            pool_indices_local = sampling_info.req_pool_indices
            row0 = sampling_info.batch_row_offset
            coins = self._coins_buf[row0 : row0 + local_bs, :num_tokens_per_req]
            final_coins = self._final_coins_buf[row0 : row0 + local_bs]
            predict_local = self._predict_buf[: local_bs * num_tokens_per_req]
            accept_index_local = self._accept_index_buf[
                : local_bs * num_tokens_per_req
            ].view(local_bs, num_tokens_per_req)
            accept_length_local = self._accept_length_buf[:local_bs]

        accept_index_local.fill_(-1)
        logits = nan_guard_logits(
            logits_output.next_token_logits, self.config.enable_nan_detection
        ).float()
        expected_rows = local_bs * num_tokens_per_req
        if logits.shape[0] != expected_rows:
            raise RuntimeError(
                f"sampling logits rows {logits.shape[0]} != expected {expected_rows}"
            )
        if vocab_mask is not None:
            sampling_info.apply_vocab_mask(logits=logits, vocab_mask=vocab_mask)

        raw_logits = logits.clone() if self.config.enable_output_logprobs else None
        logits = self._apply_penalties_and_bias_for_indices(
            logits,
            pool_indices_local,
            num_tokens_per_req=num_tokens_per_req,
        )
        temperatures, top_ks, top_ps, min_ps = self._gather_sampling_scalars(
            pool_indices_local,
            num_tokens_per_req=num_tokens_per_req,
        )
        target_probs = self._sampling_probs(
            logits, temperatures, top_ks, top_ps, min_ps
        ).view(local_bs, num_tokens_per_req, -1)

        chain_speculative_sampling_target_only(
            predict_local,
            accept_index_local,
            accept_length_local,
            candidates_local.to(torch.int32),
            coins,
            final_coins,
            target_probs,
            self._sampling_workspace,
            threshold_single=SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
            threshold_acc=SPECULATIVE_ACCEPT_THRESHOLD_ACC,
        )
        accept_length_local += 1

        logprobs_local = None
        if raw_logits is not None:
            logprobs_local = gather_token_logprobs(
                raw_logits,
                predict_local.long(),
                logprob_order=self.config.logprob_order,
            ).view(local_bs, num_tokens_per_req)

        if dp_sampling:
            self._dp_comm.prepare_verify_outputs(logits_output.next_token_logits.dtype)
            predict_full, _, accept_length_full = self._dp_comm.gather_verify_outputs(
                predict_local=predict_local.view(local_bs, num_tokens_per_req),
                accept_index_local=accept_index_local,
                accept_length_local=accept_length_local,
                pad_bs=pad_bs,
            )
            predict = predict_full[:effective_bs].reshape(-1)
            accept_length = accept_length_full[:effective_bs]
            if logprobs_local is not None:
                logits_output.next_token_logprobs = (
                    self._dp_comm.gather_verify_logprobs(logprobs_local, pad_bs=pad_bs)[
                        :effective_bs
                    ].reshape(-1)
                )
            count_pool_indices = sampling_info.req_pool_indices
        else:
            predict = predict_local
            accept_length = accept_length_local
            self.maybe_broadcast(predict, accept_index_local, accept_length)
            if logprobs_local is not None:
                logits_output.next_token_logprobs = logprobs_local.reshape(-1)
            count_pool_indices = pool_indices_local

        self._accumulate_accepted_prefix(
            count_pool_indices,
            predict,
            accept_length,
            num_tokens_per_req,
        )
        return predict, accept_length


register_backend("ascend_full", AscendFullSamplingBackend)
