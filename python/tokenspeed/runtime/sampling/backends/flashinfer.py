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

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax
from tokenspeed_kernel.ops.sampling import chain_verify_coins
from tokenspeed_kernel.ops.sampling.block_verify import block_speculative_sampling
from tokenspeed_kernel.ops.sampling.cuda import (
    chain_speculative_sampling_target_only,
)
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_available as _FUSED_TOPK_TOPP_AVAILABLE,
)
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_prepare,
    fused_topk_topp_renorm,
)
from tokenspeed_kernel.ops.sampling.flashinfer import (
    softmax,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
)
from tokenspeed_kernel.ops.sampling.triton import (
    gather_and_expand_scalars,
    gumbel_sample_from_pools_generic,
)
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.distributed.dp_sampling_comm import DpSamplingComm
from tokenspeed.runtime.execution.drafter.eagle_harness.tree_verify import (
    TreeRejectionSampler,
)
from tokenspeed.runtime.execution.slot_state import aligned_slot_state_bytes
from tokenspeed.runtime.sampling.backends.base import (
    SPECULATIVE_ACCEPT_THRESHOLD_ACC,
    SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
    SamplingBackend,
    SamplingBackendConfig,
    SpeculativeSamplingPools,
)
from tokenspeed.runtime.sampling.backends.greedy import _verify_chain_greedy
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
    slice_dp_vocab_mask,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.tree_verify import (
    accepted_path_rows,
    packed_accepted_path_rows,
)
from tokenspeed.runtime.sampling.utils import (
    coin_eps,
    gather_token_logprobs,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams
    from tokenspeed.runtime.sampling.tree_verify import TreeVerifyBatch


# Greedy requests normalize to top_k=1 and ride the pool route, whose
# stochastic kernels resolve EXACT logit ties in reduction order: run-stable,
# not batch-invariant. Under the bitwise envelope (rl-bitwise) the helpers
# below give greedy rows the canonical lowest-index argmax instead. Both are
# elementwise over rows, so the graph-captured path stays one path.


def canonical_greedy_tokens(
    logits: torch.Tensor, top_ks: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Replace greedy rows' sampled tokens with the lowest-index argmax.

    Args:
        logits: ``[bs, vocab]`` logits the greedy choice maximizes.
        top_ks: ``[bs]`` per-row top-k; greedy rows carry 1.
        token_ids: ``[bs]`` tokens the pool route sampled.

    Returns:
        ``[bs]`` tokens in ``token_ids``' dtype.
    """
    canonical = sampling_argmax(logits)
    return torch.where(top_ks == 1, canonical.to(token_ids.dtype), token_ids)


def canonical_greedy_verify(
    *,
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    candidates: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
) -> None:
    """Rewrite greedy rows' verify outputs with the exact-match chain rule.

    Without ties the one-hot target makes chain speculative sampling accept
    exactly the drafts that equal the argmax, so this changes only tied rows.

    Args:
        logits: ``[bs * n, vocab]`` flat verify logits.
        top_ks: ``[bs * n]`` per-position top-k; greedy requests carry 1.
        candidates: ``[bs, n]`` draft chains.
        predict: ``[bs * n]`` int32 predictions, rewritten in place.
        accept_index: ``[bs, n]`` int32 accepted positions, rewritten in place.
        accept_length: ``[bs]`` int32 accepted draft counts (before the bonus
            token is added), rewritten in place.
    """
    bs, n = candidates.shape
    target_predict = sampling_argmax(logits).reshape(bs, n)
    # The CUDA chain kernel writes only the accepted prefix and the bonus
    # slot, but every slot is read (e.g. by the logprob gather): start from
    # the argmax, which is what an accepted greedy slot holds anyway.
    greedy_predict = target_predict.reshape(-1).to(predict.dtype)
    greedy_index = torch.full_like(accept_index, -1)
    greedy_length = torch.empty_like(accept_length)
    _verify_chain_greedy(
        predicts=greedy_predict,
        accept_index=greedy_index,
        accept_token_num=greedy_length,
        candidates=candidates.to(torch.int32),
        target_predict=target_predict,
        batch_size=bs,
        num_draft_tokens=n,
    )
    greedy_rows = top_ks.view(bs, n)[:, 0] == 1
    predict.copy_(
        torch.where(greedy_rows.repeat_interleave(n), greedy_predict, predict)
    )
    accept_index.copy_(
        torch.where(greedy_rows.unsqueeze(1), greedy_index, accept_index)
    )
    accept_length.copy_(torch.where(greedy_rows, greedy_length, accept_length))


class FlashInferSamplingBackend(SamplingBackend):
    """Fast backend: fused softmax(temperature) + top_k_top_p_sampling_from_probs
    for stochastic single-step sampling; cuda chain kernels (greedy +
    rejection) for multi-step verification.

    Scope is deliberately narrow — temperature / top_k / top_p only —
    keeping the hot path to 2 kernels. Requests asking for min_p, penalties,
    or logit_bias are silently ignored; use `flashinfer_full` if any of those
    matter for the workload.

    ``config.sampling_stream == "per-request"`` swaps the single-step kernel
    for the Gumbel-max pool route (``_sample_per_request``), including
    verification at N == 1. Multi-token chain verification uses independent
    coins for each round, keyed by request seed and first target position.
    Sampled adaptive V4.1 DSpark uses position-keyed CuTe probability sampling
    for every width.
    """

    _HAS_POOL_STATE = True
    supports_tree_verify = True
    supports_packed_verify: bool = True
    _SUPPORTS_DP_VERIFY = True
    _SUPPORTS_OUTPUT_CUMULATIVE_PROBS = True
    # Only over sampled drafts (speculative sampling); checked in __init__.
    _SUPPORTS_SYNTHETIC_ACCEPTANCE = True

    def __init__(self, config: SamplingBackendConfig) -> None:
        self._dspark_sampling = bool(
            global_server_args_dict["enable_dsv41_adaptive_dspark"]
            and config.enable_speculative_sampling
        )
        if self._dspark_sampling and (
            config.sampling_stream != "per-request"
            or global_server_args_dict["speculative_rejection_sample_method"]
            != "standard"
            or config.synthetic_acceptance_length is not None
        ):
            raise ValueError(
                "sampled adaptive DSpark requires the per-request sampling stream, "
                "standard rejection sampling, and no synthetic acceptance length"
            )
        # The rejection method is a server-wide choice read from the server
        # args like ``numerics``; block verification only applies under
        # speculative sampling.
        self._block_verify = (
            global_server_args_dict["speculative_rejection_sample_method"] == "block"
            and config.enable_speculative_sampling
        )
        if (
            config.synthetic_acceptance_length is not None
            and not config.enable_speculative_sampling
        ):
            raise ValueError(
                "synthetic acceptance on a flashinfer backend requires sampled "
                "drafts (speculative sampling); otherwise use the greedy "
                "sampling backend"
            )
        super().__init__(config)
        # Batch sampling and tree verify need legacy generators and coin
        # refills. Sampled adaptive DSpark uses its own position-keyed sampler.
        self._uses_legacy_coins = not self._dspark_sampling and (
            config.sampling_stream != "per-request" or config.draft_tree
        )
        self._init_dp_sampling(config)
        self._init_shared_buffers(config)
        self._init_pool_scalars(config)
        self._tree_sampler = TreeRejectionSampler(self._dp_max_pad_bs, config.device)
        if _FUSED_TOPK_TOPP_AVAILABLE:
            # Pre-create the fused renorm side stream: cudaStreamCreate is
            # illegal inside capture, and verify() runs from the captured graph.
            fused_topk_topp_prepare(config.device)

    def _init_dp_sampling(self, config: SamplingBackendConfig) -> None:
        self._dp_tp_group = config.tp_group
        self._dp_tp_size = (
            len(self._dp_tp_group) if self._dp_tp_group is not None else 1
        )
        self._dp_rank = 0
        self._dp_comm: DpSamplingComm | None = None
        self._dp_comm_vocab_size = 0

        if self._dp_tp_size <= 1:
            self._dp_max_pad_bs = config.max_bs
            self._dp_max_reqs_per_rank = config.max_bs
            return

        self._dp_max_pad_bs = (
            (config.max_bs + self._dp_tp_size - 1) // self._dp_tp_size
        ) * self._dp_tp_size
        self._dp_max_reqs_per_rank = self._dp_max_pad_bs // self._dp_tp_size

    def configure_dp_sampling(self, runtime: DpSamplingRuntimeConfig) -> None:
        if not runtime.enabled:
            return
        if (
            runtime.vocab_size is None
            or runtime.max_bucket_bs is None
            or runtime.topology is None
            or runtime.device is None
        ):
            raise RuntimeError("enabled DP sampling runtime is incomplete")
        topology = runtime.topology
        if topology.tp_size != self._dp_tp_size:
            raise RuntimeError(
                f"DP sampling runtime tp_size={topology.tp_size} "
                f"does not match backend tp_size={self._dp_tp_size}"
            )
        if topology.tp_group != self._dp_tp_group:
            raise RuntimeError("DP sampling runtime tp_group does not match backend")
        if self._dp_tp_group is None:
            raise RuntimeError("dp_sampling requires a tp_group")
        self._dp_rank = topology.tp_rank
        if runtime.max_bucket_bs > self._dp_max_pad_bs:
            raise RuntimeError(
                f"DP sampling max_bucket_bs={runtime.max_bucket_bs} exceeds "
                f"backend max_pad_bs={self._dp_max_pad_bs}"
            )
        if runtime.vocab_size % self._dp_tp_size != 0:
            raise RuntimeError(
                f"DP sampling vocab_size={runtime.vocab_size} must be divisible by "
                f"tp_size={self._dp_tp_size}"
            )
        self._init_dp_verify_buffers(runtime.device)
        if runtime.vocab_size == self._dp_comm_vocab_size:
            return
        if self._dp_comm is not None and self._dp_comm.is_initialized:
            raise RuntimeError("Cannot resize DP sampling comm after use")
        self._dp_comm_vocab_size = runtime.vocab_size
        self._dp_comm = DpSamplingComm(
            tp_size=self._dp_tp_size,
            rank=self._dp_rank,
            group=self._dp_tp_group,
            max_pad_bs=self._dp_max_pad_bs,
            num_tokens_per_req=runtime.num_tokens_per_req,
            vocab_size=runtime.vocab_size,
            logits_dtype=None,
            device=runtime.device,
        )

    def _init_dp_verify_buffers(self, device: torch.device | str) -> None:
        if self._predict_local_buf is not None:
            return

        max_n = self.config.max_draft_tokens_per_req
        self._predict_local_buf = torch.zeros(
            (self._dp_max_reqs_per_rank * max_n,), dtype=torch.int32, device=device
        )
        self._accept_index_local_buf = torch.zeros(
            (self._dp_max_reqs_per_rank * max_n,), dtype=torch.int32, device=device
        )
        self._accept_length_local_buf = torch.zeros(
            (self._dp_max_reqs_per_rank,), dtype=torch.int32, device=device
        )

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

        # The legacy coin stream advances outside the graph and its retraction
        # image carries the generator state. The position-keyed chain stream
        # needs only the seed and the restored cache frontier.
        self._cpu_generator_per_slot: list[torch.Generator | None] = [None] * pool_rows
        self._cpu_generator_per_slot[0] = self._capture_gen
        self._generator_state_bytes = (
            torch.Generator(device="cpu").get_state().numel()
            if self._uses_legacy_coins
            else 0
        )
        self._generator_image_bytes = aligned_slot_state_bytes(
            self._generator_state_bytes
        )

    def speculative_sampling_pools(self) -> SpeculativeSamplingPools:
        return SpeculativeSamplingPools(
            temperature=self._temperature_pool,
            top_k=self._top_k_pool,
            top_p=self._top_p_pool,
            seed=self._seed_pool,
        )

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        self._temperature_pool[pool_idx].fill_(float(sp.temperature))
        self._top_k_pool[pool_idx].fill_(int(sp.top_k))
        self._top_p_pool[pool_idx].fill_(float(sp.top_p))
        self._seed_pool[pool_idx].fill_(int(sp.seed))

        if self._uses_legacy_coins:
            cpu_gen = torch.Generator(device="cpu")
            cpu_gen.manual_seed(int(sp.seed))
            self._cpu_generator_per_slot[pool_idx] = cpu_gen

    # ------------------------------------------------------------------
    # Slot-state image (SlotStateExporter): the scalars and the coin generator
    # ------------------------------------------------------------------

    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        return [
            self._temperature_pool[slot],
            self._top_k_pool[slot],
            self._top_p_pool[slot],
            self._seed_pool[slot],
        ]

    def _slot_payload_bytes(self) -> int:
        return self._generator_image_bytes + super()._slot_payload_bytes()

    def _export_slot_payload(self, slot: int, out: torch.Tensor, stream) -> None:
        if self._uses_legacy_coins:
            # The base checks the prepared marker before exporting, so every
            # imaged slot already owns its CPU generator.
            generator = self._cpu_generator_per_slot[slot]
            if generator is None or generator.device.type != "cpu":
                raise RuntimeError(
                    f"sampling slot {slot} has no per-request coin generator to image"
                )
            out[: self._generator_state_bytes].copy_(generator.get_state())
        super()._export_slot_payload(slot, out[self._generator_image_bytes :], stream)

    def _import_slot_payload(self, slot: int, src: torch.Tensor, stream) -> None:
        if self._uses_legacy_coins:
            generator = torch.Generator(device="cpu")
            generator.set_state(src[: self._generator_state_bytes].clone())
            self._cpu_generator_per_slot[slot] = generator
        super()._import_slot_payload(slot, src[self._generator_image_bytes :], stream)

    def _init_shared_buffers(self, config: SamplingBackendConfig) -> None:

        max_pad_bs = self._dp_max_pad_bs
        max_n = config.max_draft_tokens_per_req
        # Persistent coin buffers, filled within verify() for the position-keyed
        # chain stream and outside the graph for the legacy stream.
        self._coins_buf = torch.zeros(
            (max_pad_bs, max_n),
            dtype=torch.float32,
            device=config.device,
        )
        self._final_coins_buf = torch.zeros(
            (max_pad_bs,), dtype=torch.float32, device=config.device
        )

        # Stub generator used during CUDA-graph capture/warm-up (no requests yet).
        self._capture_gen: torch.Generator | None = None
        if self._uses_legacy_coins:
            self._capture_gen = torch.Generator(device=config.device)
            self._capture_gen.manual_seed(config.random_seed)

        # Pre-allocated persistent buffers — no per-step alloc in the hot path.
        self._ones_buf = torch.ones(
            (max_pad_bs,), dtype=torch.int32, device=config.device
        )
        # DP padding may need more verify rows than max_bs.
        if max_pad_bs != config.max_bs:
            self._allocate_verify_outputs(max_pad_bs, max_n)
            self._allocate_pool_index_buffer(max_pad_bs)

        self._predict_local_buf: torch.Tensor | None = None
        self._accept_index_local_buf: torch.Tensor | None = None
        self._accept_length_local_buf: torch.Tensor | None = None

        # The native chain verifier reads batch-ordered [bs,N,V] rows. The
        # CuTe verifier addresses the RuntimeStates proposal pool directly.
        self._draft_probs_gather_buf: torch.Tensor | None = None
        if config.enable_speculative_sampling:
            if config.vocab_size <= 0:
                raise ValueError(
                    "enable_speculative_sampling needs vocab_size > 0 for "
                    f"proposal distributions, got {config.vocab_size}"
                )
            if not self._dspark_sampling:
                self._draft_probs_gather_buf = torch.empty(
                    (max_pad_bs, max_n, config.vocab_size),
                    dtype=torch.float32,
                    device=config.device,
                )

        # The per-request stream route (``_sample_per_request``) writes its
        # token ids into a caller-owned buffer.
        self._per_request_out = torch.empty(
            (max_pad_bs,), dtype=torch.int32, device=config.device
        )

    def _gather_draft_probs(
        self,
        draft_probs: torch.Tensor,
        pool_indices: torch.Tensor,
        bs: int,
        num_tokens_per_req: int,
    ) -> torch.Tensor:
        """Batch-order the recorded draft distributions for this verify.

        Args:
            draft_probs: ``[pool_rows, N, V]`` fp32 ``RuntimeStates.draft_probs``.
            pool_indices: ``[bs]`` pool slot per verify row (padding rows
                carry slot 0; their outputs are discarded).
            bs: Verify rows, padded under graph replay or DP sharding.
            num_tokens_per_req: The chain width N.

        Returns:
            The ``[bs, N, V]`` leading slice of the persistent gather buffer.
        """
        buf = self._draft_probs_gather_buf
        if buf is None:
            raise RuntimeError(
                "verify received draft_probs but the sampling backend was built "
                "without enable_speculative_sampling"
            )
        if num_tokens_per_req != buf.shape[1] or draft_probs.shape[1:] != buf.shape[1:]:
            raise RuntimeError(
                f"draft_probs geometry {tuple(draft_probs.shape[1:])} / N="
                f"{num_tokens_per_req} does not match the verify buffer "
                f"{tuple(buf.shape[1:])}"
            )
        out = buf[:bs]
        torch.index_select(draft_probs, 0, pool_indices, out=out)
        return out

    @staticmethod
    def _pack_single_verify_result(
        tokens: torch.Tensor,
        *,
        predict: torch.Tensor,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write one output per request into caller-owned verify buffers.

        ``tokens`` and ``predict`` are [bs], ``accept_index`` is [bs, 1], and
        ``accept_length`` is [bs]. Lengths count emitted tokens, including the
        sole target token. Explicit destinations also serve DP-local buffers;
        this adapter performs no sampling, communication or count updates.
        """
        predict.copy_(tokens)
        accept_length.fill_(1)
        torch.arange(
            tokens.numel(),
            dtype=accept_index.dtype,
            device=accept_index.device,
            out=accept_index.view(-1),
        )
        return predict, accept_length

    def _sample_per_request(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        *,
        min_p_pool: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample every row from its request's own Gumbel-max stream.

        flashinfer's ``*_sampling_from_probs`` kernels read one seed and
        offset and seed curand with the batch row, so a request's draw moves
        with its co-batch. The pool kernel keys the stream by
        ``(seed_pool[pool_idx], offsets_pool[pool_idx])`` — the request's seed
        and absolute target position — so the draw is the same alone and in
        any batch (``--sampling-stream per-request``). Temperature, top-k, top-p and
        min-p are read from the pool rows (no expanded per-row scalars are
        built); finite top-k is capped at the kernel's 128 candidates.

        Args:
            logits: ``[bs, vocab]`` logits (penalties already applied).
            sampling_info: Batch info carrying the pool indices and the
                borrowed frontier and prefill lengths.
            min_p_pool: Pool-indexed min-p values, or None for backends
                without min-p.

        Returns:
            ``(token_ids, top_ks)``: the ``[bs]`` int32 token ids and each
            row's top-k, which the bitwise envelopes' greedy overlay reads.
        """
        bs = logits.shape[0]
        pool_indices = self._req_pool_indices_for_kernels(
            sampling_info.req_pool_indices, bs
        )
        token_ids = gumbel_sample_from_pools_generic(
            logits,
            pool_indices,
            self._temperature_pool,
            self._top_k_pool,
            self._top_p_pool,
            self._seed_pool,
            self._prepare_sampling_offsets(sampling_info),
            self._per_request_out[:bs],
            min_p_pool=min_p_pool,
        )
        return token_ids, self._top_k_pool.index_select(0, pool_indices)

    def _prepare_step_hook(
        self,
        num_tokens_per_req: int,
        bs: int,
        request_pool_indices: list[int] | None = None,
    ) -> None:
        """Refill legacy coins outside the graph, for batch sampling and trees."""
        if bs > 0 and self._uses_legacy_coins:
            self._refill_legacy_coins(
                num_tokens_per_req=num_tokens_per_req,
                bs=bs,
                request_pool_indices=request_pool_indices,
            )

    def _refill_legacy_coins(
        self,
        *,
        num_tokens_per_req: int,
        bs: int,
        request_pool_indices: list[int] | None,
    ) -> None:
        """Advance per-step generators and fill the verifier's coin buffers.

        Live requests use their CPU generators; capture uses the stub stream.
        This is the existing coin protocol, not position-keyed randomness.
        """
        n = min(num_tokens_per_req, self.config.max_draft_tokens_per_req)
        lo = coin_eps(self._coins_buf.dtype)
        if request_pool_indices is None:
            assert self._capture_gen is not None
            self._coins_buf[:bs, :n].uniform_(lo, 1.0, generator=self._capture_gen)
            self._final_coins_buf[:bs].uniform_(lo, 1.0, generator=self._capture_gen)
            return

        cpu_coins = torch.empty((bs, n), dtype=torch.float32, pin_memory=True)
        cpu_final = torch.empty((bs,), dtype=torch.float32, pin_memory=True)

        for i, pool_idx in enumerate(request_pool_indices):
            gen = self._cpu_generator_per_slot[pool_idx]
            if gen is None:
                raise RuntimeError(
                    f"sampling slot {pool_idx} was not initialized before "
                    "coin-buffer refill"
                )
            cpu_coins[i, :n].uniform_(lo, 1.0, generator=gen)
            cpu_final[i].uniform_(lo, 1.0, generator=gen)

        self._coins_buf[:bs, :n].copy_(cpu_coins, non_blocking=True)
        self._final_coins_buf[:bs].copy_(cpu_final, non_blocking=True)

    def _chain_verify_coins(
        self,
        sampling_info: SamplingBatchInfo,
        pool_indices: torch.Tensor,
        *,
        width: int,
        row_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = pool_indices.numel()
        if self.config.sampling_stream == "per-request":
            # The CUDA verifier takes contiguous [bs, width] input even when
            # the backing scratch was allocated for a wider verify window.
            coins = self._coins_buf.view(-1)[: bs * width].view(bs, width)
            final_coins = self._final_coins_buf[:bs]
            # DP verify only draws its local rows. The offsets are prepared
            # for the real requests before their local padded indices read them.
            chain_verify_coins(
                pool_indices,
                self._seed_pool,
                self._prepare_sampling_offsets(sampling_info),
                coins=coins,
                final_coins=final_coins,
            )
            return coins, final_coins

        # Legacy coins were filled in full-batch order. Skip prefill rows in
        # a mixed batch, or take the verifier's DP shard.
        rows = slice(row_offset, row_offset + bs)
        return self._coins_buf[rows, :width], self._final_coins_buf[rows]

    @nvtx_range("sampling:sample", color="yellow")
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        logits = logits_output.next_token_logits

        if self._dspark_sampling:
            from tokenspeed_kernel.ops.sampling.packed_verify import (
                RNG_TAG_BONUS,
                sample_probs,
            )

            if sampling_info.vocab_mask is not None:
                sampling_info.apply_vocab_mask(
                    logits=logits, vocab_mask=sampling_info.vocab_mask
                )
            slots = sampling_info.req_pool_indices
            bs = logits.shape[0]
            sampled = self._predict_buf[:bs]
            sample_probs(
                probs=self._dspark_probabilities(logits, slots),
                seeds=self._seed_pool.index_select(0, slots),
                positions=self._prepare_sampling_offsets(sampling_info).index_select(
                    0, slots
                ),
                output=sampled,
                rng_tag=RNG_TAG_BONUS,
            )
            lengths = self._accept_length_buf[:bs]
            lengths.fill_(1)
            self.maybe_broadcast(sampled)
            if self.config.enable_output_logprobs:
                logits_output.next_token_logprobs = gather_token_logprobs(
                    logits, sampled, logprob_order=self.config.logprob_order
                )
            return sampled, lengths

        # Grammar bitmask apply — captured inside the CUDA graph. Buffer is
        # pre-bound by bind_grammar_mask_buf; non-grammar rows stay all-ones.
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )

        # Greedy requests normalize to top_k=1 (SamplingParams.__post_init__),
        # so the pool route serves them too — same path the CUDA graph
        # captures. Equivalence to argmax is pinned by
        # test_greedy_route_equivalence.py.
        if self.config.sampling_stream == "per-request":
            batch_next_token_ids, top_ks = self._sample_per_request(
                logits, sampling_info, min_p_pool=None
            )
        else:
            temperatures, top_ks, top_ps, _, seeds, offsets = gather_and_expand_scalars(
                sampling_info.req_pool_indices,
                temperature=self._temperature_pool,
                top_k=self._top_k_pool,
                top_p=self._top_p_pool,
                seed=self._seed_pool,
                offsets=self._prepare_sampling_offsets(sampling_info),
            )
            probs = softmax(
                logits,
                temperature=temperatures.view(-1, 1),
            )
            batch_next_token_ids = top_k_top_p_sampling_from_probs(
                probs,
                top_ks,
                top_ps,
                filter_apply_order="joint",
                seed=seeds,
                offset=offsets,
                deterministic=True,
            )
        if global_server_args_dict["numerics"] in BITWISE_ENVELOPES:
            batch_next_token_ids = canonical_greedy_tokens(
                logits, top_ks, batch_next_token_ids
            )

        bs = logits.shape[0]
        # Land the tokens in the packed output region so both outputs alias
        # _output_pack_buf and get_packed_output_d2h's single-D2H fast path
        # fires on the eager path exactly as it does after a verify().
        sampled = self._predict_buf[:bs]
        sampled.copy_(batch_next_token_ids)
        lengths = self._accept_length_buf[:bs]
        lengths.fill_(1)

        # TP-rank sync: rank 0 wins.
        self.maybe_broadcast(sampled)

        if self.config.enable_output_logprobs:
            logits_output.next_token_logprobs = gather_token_logprobs(
                logits, sampled, logprob_order=self.config.logprob_order
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
        query_start_loc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dspark_sampling:
            if tree is not None:
                raise ValueError(
                    "sampled adaptive DSpark requires plain chain verification"
                )
            return self._verify_dspark(
                logits_output, sampling_info, candidates, query_start_loc
            )
        if query_start_loc is not None:
            raise ValueError(
                "packed flashinfer verification requires sampled adaptive DSpark"
            )
        if tree is not None:
            return self._verify_tree(logits_output, sampling_info, candidates, tree)

        bs = candidates.shape[0]
        num_tokens_per_req = candidates.shape[1]
        vocab_mask = sampling_info.vocab_mask
        logits_layout_plan = getattr(logits_output, "logits_layout_plan", None)
        dp_sampling = logits_layout_plan is not None

        if dp_sampling:
            if self._dp_comm is None:
                raise RuntimeError(
                    "dp_sampling requires tp_size > 1, a resolved tp_group, "
                    "and a configured DP comm"
                )
            dp_comm = self._dp_comm
            tp_size = self._dp_tp_size
            rank = self._dp_rank
            effective_bs = logits_layout_plan.effective_bs
            pad_bs = logits_layout_plan.bucket_bs
            if effective_bs != bs:
                raise RuntimeError(
                    f"DP sampling effective_bs={effective_bs} must match "
                    f"candidate batch size {bs}"
                )
            if (
                pad_bs < effective_bs
                or pad_bs > self._dp_max_pad_bs
                or pad_bs % tp_size != 0
            ):
                raise RuntimeError(
                    f"invalid DP sampling pad_bs={pad_bs} for effective_bs={effective_bs}, "
                    f"max_pad_bs={self._dp_max_pad_bs}, tp_size={tp_size}"
                )
            bs = pad_bs // tp_size

            # Shard by request so each request's draft chain stays on one rank.
            shard = slice(rank * bs, (rank + 1) * bs)
            if pad_bs > effective_bs:
                candidates = torch.nn.functional.pad(
                    candidates, (0, 0, 0, pad_bs - effective_bs)
                )[shard]
                pool_indices = torch.nn.functional.pad(
                    sampling_info.req_pool_indices, (0, pad_bs - effective_bs)
                )[shard]
            else:
                candidates = candidates[shard]
                pool_indices = sampling_info.req_pool_indices[shard]
            vocab_mask = slice_dp_vocab_mask(
                vocab_mask,
                full_bs=effective_bs,
                pad_bs=pad_bs,
                num_tokens_per_req=num_tokens_per_req,
                shard=shard,
            )
            coin_row_offset = rank * bs
            if (
                self._predict_local_buf is None
                or self._accept_index_local_buf is None
                or self._accept_length_local_buf is None
            ):
                raise RuntimeError("DP sampling verify buffers are not initialized")
            predict = self._predict_local_buf[: bs * num_tokens_per_req]
            accept_index = (
                self._accept_index_local_buf[: bs * num_tokens_per_req]
                .view(bs, num_tokens_per_req)
                .fill_(-1)
            )
            accept_length = self._accept_length_local_buf[:bs]
        else:
            pool_indices = sampling_info.req_pool_indices
            coin_row_offset = sampling_info.batch_row_offset
            predict = self._predict_buf[: bs * num_tokens_per_req]
            accept_index = (
                self._accept_index_buf[: bs * num_tokens_per_req]
                .view(bs, num_tokens_per_req)
                .fill_(-1)
            )
            accept_length = self._accept_length_buf[:bs]

        logits = logits_output.next_token_logits
        if dp_sampling:
            expected_rows = bs * num_tokens_per_req
            if logits.shape[0] != expected_rows:
                raise RuntimeError(
                    f"DP sampling logits rows {logits.shape[0]} != expected "
                    f"{expected_rows}"
                )

        # Per-draft-position grammar bitmask: buffer shape
        # [bs * num_tokens_per_req, V/32] matches the flat target logits.
        if vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits,
                vocab_mask=vocab_mask,
            )

        # Greedy verifies through the same pool route (top_k=1); see sample().
        # Each request's N verified positions share one (temp, top_k, top_p)
        # tuple; flat [bs*N] per-row knobs match the flat [bs*N, vocab] logits.
        n = num_tokens_per_req

        per_request_stream = self.config.sampling_stream == "per-request"
        has_draft_candidates = n > 1
        # Random-stream policy and verification geometry are independent.
        # A one-token window needs only a target draw; wider windows must
        # retain acceptance and residual sampling, even under per-request.
        if per_request_stream and not has_draft_candidates:
            sampled, top_ks = self._sample_per_request(
                logits,
                replace(sampling_info, req_pool_indices=pool_indices),
                min_p_pool=None,
            )
            if global_server_args_dict["numerics"] in BITWISE_ENVELOPES:
                sampled = canonical_greedy_tokens(logits, top_ks, sampled)
            predict, accept_length = self._pack_single_verify_result(
                sampled,
                predict=predict,
                accept_index=accept_index,
                accept_length=accept_length,
            )
            # Match sample(): publish rank 0's direct draw on replicated TP.
            needs_tp_sync = True
        else:
            coins, final_coins = self._chain_verify_coins(
                sampling_info,
                pool_indices,
                width=n,
                row_offset=coin_row_offset,
            )
            temperatures, top_ks, top_ps, _, _, _ = gather_and_expand_scalars(
                pool_indices,
                temperature=self._temperature_pool,
                top_k=self._top_k_pool,
                top_p=self._top_p_pool,
                n=n,
            )

            target_probs = softmax(
                logits,
                temperature=temperatures,
            )
            if _FUSED_TOPK_TOPP_AVAILABLE:
                # Fused replacement for the back-to-back top_k_renorm_prob +
                # top_p_renorm_prob(is_deterministic=True) pair. Sentinel
                # K = 1<<30 in top_ks routes per-row through the radix top-p
                # only path. Availability decided once in
                # tokenspeed_kernel.ops.sampling.cuda.
                target_probs = fused_topk_topp_renorm(
                    target_probs,
                    top_ks,
                    top_ps,
                    in_place_skip=False,
                )
            else:
                target_probs = top_k_renorm_prob(target_probs, top_ks)
                target_probs = top_p_renorm_prob(
                    target_probs, top_ps, is_deterministic=True
                )
            target_probs = target_probs.reshape(bs, n, -1)

            # Draft-prob rule when the drafter recorded its distributions (the
            # DP shard gathers its own rows); target-only otherwise.
            use_draft_prob = sampling_info.draft_probs is not None
            draft_probs = (
                self._gather_draft_probs(sampling_info.draft_probs, pool_indices, bs, n)
                if use_draft_prob
                else None
            )
            synthetic = self.config.synthetic_acceptance_length is not None
            if (self._block_verify or synthetic) and draft_probs is None:
                raise RuntimeError(
                    "block verification and synthetic acceptance on this backend "
                    "require recorded draft probabilities"
                )
            # Synthetic acceptance mirrors vLLM's rejection_sample_method
            # "synthetic": the standard draft-probability chain rule below runs
            # for its cost, then the forced lengths replace its outcome. The
            # block verifier is a different rejection method and is not run.
            if self._block_verify and not synthetic:
                block_speculative_sampling(
                    predicts=predict,
                    accept_index=accept_index,
                    accept_token_num=accept_length,
                    candidates=candidates,
                    uniform_samples=coins,
                    uniform_samples_for_final_sampling=final_coins,
                    target_probs=target_probs,
                    draft_probs=draft_probs,
                    reject_draft_prob_threshold=self.config.spec_reject_draft_prob_threshold,
                )
            else:
                chain_speculative_sampling_target_only(
                    predicts=predict,
                    accept_index=accept_index,
                    accept_token_num=accept_length,
                    candidates=candidates,
                    uniform_samples=coins,
                    uniform_samples_for_final_sampling=final_coins,
                    target_probs=target_probs,
                    draft_probs=draft_probs,
                    threshold_single=SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
                    threshold_acc=SPECULATIVE_ACCEPT_THRESHOLD_ACC,
                    deterministic=per_request_stream or not dp_sampling,
                    use_draft_prob=use_draft_prob,
                    reject_draft_prob_threshold=self.config.spec_reject_draft_prob_threshold,
                )
            if global_server_args_dict["numerics"] in BITWISE_ENVELOPES:
                canonical_greedy_verify(
                    logits=logits,
                    top_ks=top_ks,
                    candidates=candidates,
                    predict=predict,
                    accept_index=accept_index,
                    accept_length=accept_length,
                )
            if synthetic:
                lengths = self.synthetic_lengths(
                    candidates, sampling_info.batch_row_offset
                )
                self.write_synthetic_outputs(
                    candidates,
                    self.sample_synthetic_targets(target_probs, lengths, final_coins),
                    lengths,
                    predict,
                    accept_index,
                    accept_length,
                )

            accept_length += 1
            # This is the chain implementation's agreement requirement, not
            # a property of the request's random-stream policy. The fused
            # renorm path skips TP sync only when PDL is disabled. The block
            # verifier meets the same requirement: fixed grids, per-program
            # reductions and no atomics, so equal inputs give equal outputs
            # on every rank.
            needs_tp_sync = pdl_enabled() or not _FUSED_TOPK_TOPP_AVAILABLE
        logprobs_local = None
        if self.config.enable_output_logprobs and dp_sampling:
            # DP verify logits are still sharded by request at this point.
            # Compute scalar logprobs for local predictions before gathering
            # predictions to full-batch shape; the non-DP writer requires
            # matching logits/token row counts.
            logprobs_local = gather_token_logprobs(
                logits, predict, logprob_order=self.config.logprob_order
            ).view(bs, num_tokens_per_req)

        if dp_sampling:
            n = num_tokens_per_req
            dp_comm.prepare_verify_outputs(logits_output.next_token_logits.dtype)
            (
                predict_full,
                accept_index_full,
                accept_length_full,
            ) = dp_comm.gather_verify_outputs(
                predict_local=predict.view(bs, n),
                accept_index_local=accept_index,
                accept_length_local=accept_length,
                pad_bs=pad_bs,
            )
            predict = predict_full.view(-1)[: effective_bs * n]
            accept_index = accept_index_full[:effective_bs]
            accept_length = accept_length_full[:effective_bs]
            if logprobs_local is not None:
                logprobs_full = dp_comm.gather_verify_logprobs(
                    logprobs_local,
                    pad_bs=pad_bs,
                )
                logits_output.next_token_logprobs = logprobs_full.view(-1)[
                    : effective_bs * n
                ]
        # DP above assembles disjoint request rows. Replicated TP follows
        # the selected implementation's agreement requirement instead.
        elif needs_tp_sync:
            self.broadcast_verify_outputs()

        if self.config.enable_output_logprobs and not dp_sampling:
            logits_output.next_token_logprobs = gather_token_logprobs(
                logits, predict, logprob_order=self.config.logprob_order
            )

        return predict, accept_length

    def _dspark_probabilities(self, logits, pool_indices):
        """Normalize the target and proposal with the same request filters."""
        temperatures, top_ks, top_ps, _, _, _ = gather_and_expand_scalars(
            pool_indices,
            temperature=self._temperature_pool,
            top_k=self._top_k_pool,
            top_p=self._top_p_pool,
            n=1,
        )
        probabilities = softmax(logits, temperature=temperatures)
        # Sub-tile vocabularies need the generic filter; the fused radix
        # implementation can leave those probability rows empty.
        if _FUSED_TOPK_TOPP_AVAILABLE and logits.shape[1] >= 1024:
            return fused_topk_topp_renorm(
                probabilities, top_ks, top_ps, in_place_skip=False
            )
        probabilities = top_p_renorm_prob(
            top_k_renorm_prob(probabilities, top_ks), top_ps, is_deterministic=True
        )
        # Generic threshold filtering can retain tied maxima. A top-k=1
        # request has the same canonical point mass as the fused filter.
        greedy = top_ks == 1
        maximum = sampling_argmax(logits).view(-1, 1)
        values = probabilities.gather(1, maximum)
        values = torch.where(greedy[:, None], 1.0, values)
        probabilities.mul_((~greedy)[:, None])
        probabilities.scatter_(1, maximum, values)
        return probabilities

    def sample_dspark_proposals(
        self,
        logits: torch.Tensor,
        pool_indices: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Draw proposals into ``output`` and return their distributions.

        Args:
            logits: Proposal logits with shape ``[rows, vocab]``.
            pool_indices: Request slot for each row, selecting its filters and seed.
            positions: Per-row position keys for the proposal RNG.
            output: Buffer overwritten with one sampled token per row.

        Returns:
            Normalized ``[rows, vocab]`` probability rows for the caller to record
            and reuse during verification.
        """
        from tokenspeed_kernel.ops.sampling.packed_verify import (
            RNG_TAG_DRAFT,
            sample_probs,
        )

        if not self._dspark_sampling:
            raise ValueError(
                "DSpark proposal sampling requires sampled adaptive verification"
            )
        probabilities = self._dspark_probabilities(logits, pool_indices)
        sample_probs(
            probs=probabilities,
            seeds=self._seed_pool.index_select(0, pool_indices),
            positions=positions,
            output=output,
            rng_tag=RNG_TAG_DRAFT,
        )
        return probabilities

    def _verify_dspark(self, logits_output, sampling_info, candidates, query_start_loc):
        """Verify DSpark candidate chains against packed target-logit rows.

        ``query_start_loc`` gives CSR row bounds per request; ``None`` uses the
        uniform candidate width. Proposal-probability sentinels stop a chain
        without locally recorded proposals. Writes ``_predict_buf``,
        ``_accept_index_buf`` and ``_accept_length_buf`` in place, returning
        prediction and acceptance-length views.
        """
        from tokenspeed_kernel.ops.sampling.packed_verify import verify_packed_chain

        if logits_output.logits_layout_plan is not None:
            raise ValueError("sampled adaptive DSpark does not support DP sampling")
        if sampling_info.draft_probs is None:
            raise ValueError(
                "sampled adaptive DSpark requires recorded proposal distributions"
            )
        bs, width = candidates.shape
        logits = logits_output.next_token_logits
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )
        slots = sampling_info.req_pool_indices
        if query_start_loc is None:
            query_start_loc = (
                torch.arange(bs + 1, device=logits.device, dtype=torch.int64) * width
            )
        rows = torch.arange(logits.shape[0], device=logits.device, dtype=torch.int64)
        requests = torch.searchsorted(query_start_loc[1:], rows, right=True).clamp_max(
            bs - 1
        )
        row_slots = slots.index_select(0, requests)
        predict = self._predict_buf[: bs * width]
        indices = self._accept_index_buf[: bs * width].view(bs, width)
        lengths = self._accept_length_buf[:bs]
        verify_packed_chain(
            predicts=predict,
            accept_index=indices,
            accept_length=lengths,
            candidates=candidates.to(torch.int32),
            target_probs=self._dspark_probabilities(logits, row_slots),
            draft_probs=sampling_info.draft_probs,
            draft_pool_indices=slots,
            reject_draft_prob_threshold=self.config.spec_reject_draft_prob_threshold,
            query_start_loc=query_start_loc,
            seeds=self._seed_pool.index_select(0, slots),
            positions=self._prepare_sampling_offsets(sampling_info).index_select(
                0, slots
            ),
        )
        self.broadcast_verify_outputs()
        if self.config.enable_output_logprobs:
            path_logits = logits.index_select(
                0, packed_accepted_path_rows(indices, query_start_loc)
            )
            logits_output.next_token_logprobs = gather_token_logprobs(
                path_logits, predict, logprob_order=self.config.logprob_order
            )
        return predict, lengths

    def _verify_tree(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
        tree: TreeVerifyBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accept each request's draft tree by rejection sampling (``TreeRejectionSampler``)."""
        if logits_output.logits_layout_plan is not None:
            raise NotImplementedError("draft trees take no DP sampling")
        bs, n = candidates.shape
        row0 = sampling_info.batch_row_offset
        logits = logits_output.next_token_logits
        predict = self._predict_buf[: bs * n]
        path = self._accept_index_buf[: bs * n].view(bs, n)
        accept_length = self._accept_length_buf[:bs]
        temperatures, top_ks, top_ps, _, _, _ = gather_and_expand_scalars(
            sampling_info.req_pool_indices,
            temperature=self._temperature_pool,
            top_k=self._top_k_pool,
            top_p=self._top_p_pool,
            n=n,
        )
        # The target p': the softmax at the request's temperature, cut to its
        # top-k and top-p and renormalized (a row that cuts nothing keeps its
        # softmax values). The unfused cut reads every row, so only the fused
        # one lets greedy rows skip the softmax and the cut.
        probs = softmax(
            logits,
            temperature=temperatures,
            skip_greedy_top_ks=top_ks if _FUSED_TOPK_TOPP_AVAILABLE else None,
        )
        if _FUSED_TOPK_TOPP_AVAILABLE:
            probs = fused_topk_topp_renorm(probs, top_ks, top_ps, in_place_skip=True)
        else:
            cut = top_k_renorm_prob(probs, top_ks)
            cut = top_p_renorm_prob(cut, top_ps, is_deterministic=True)
            uncut = (top_ks >= probs.shape[1]) & (top_ps >= 1.0)
            probs = torch.where(uncut[:, None], probs, cut)
        self._tree_sampler(
            probs,
            logits,
            candidates,
            tree,
            sampling_info.req_pool_indices,
            top_ks,
            coins=self._coins_buf[row0 : row0 + bs, :n],
            final_coins=self._final_coins_buf[row0 : row0 + bs],
            path=path,
            predict=predict,
            accept_length=accept_length,
        )
        # Rank 0's outputs win: the renorm is not bit-identical across ranks.
        self.broadcast_verify_outputs()
        if self.config.enable_output_logprobs:
            # predict is packed along the path; score it against the path's own rows.
            path_logits = logits.index_select(0, accepted_path_rows(path))
            logits_output.next_token_logprobs = gather_token_logprobs(
                path_logits, predict, logprob_order=self.config.logprob_order
            )
        return predict, accept_length


register_backend("flashinfer", FlashInferSamplingBackend)
