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

"""Logits processing."""

import dataclasses

import torch
import triton
import triton.language as tl
from tokenspeed_kernel.ops.communication.triton import all_gather_inner, create_state
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
)
from tokenspeed.runtime.sampling.logits_layout import (
    LogitsLayoutExecutor,
    LogitsLayoutPlan,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/tokenspeed/runtime/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # Used by speculative decoding.
    # The last hidden layers
    hidden_states: torch.Tensor | None = None
    logits_layout_plan: LogitsLayoutPlan | None = None

    ## Part 2: Populated by the active SamplingBackend during sample()/verify().
    # The logprobs of the next tokens.                              shape: [#seq]
    next_token_logprobs: torch.Tensor | None = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    next_token_top_logprobs_val: list | None = None
    next_token_top_logprobs_idx: list | None = None
    # The logprobs and ids of the requested token ids in output positions. shape: [#seq, n] (n is the number of requested token ids)
    next_token_token_ids_logprobs_val: list | None = None
    next_token_token_ids_logprobs_idx: list | None = None

    ## Part 3: Prefill-only. This part will be assigned in python/tokenspeed/runtime/layers/logits_processor.py::LogitsProcessor
    # The logprobs of input tokens.        shape: [#token]
    input_token_logprobs: torch.Tensor | None = None
    # The logprobs and ids of the top-k tokens in input positions.  shape: [#seq, #token, k]
    input_top_logprobs_val: list = None
    input_top_logprobs_idx: list = None
    # The logprobs and ids of the requested token ids in input positions. shape: [#seq, n] (n is the number of requested token ids)
    input_token_ids_logprobs_val: list | None = None
    input_token_ids_logprobs_idx: list | None = None


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    gather_ids: torch.Tensor | None = None

    extend_return_logprob: bool = False
    extend_return_top_logprob: bool = False
    extend_token_ids_logprob: bool = False
    extend_seq_lens_cpu: list[int] | None = None
    extend_logprob_start_lens_cpu: list[int] | None = None
    extend_logprob_pruned_lens_cpu: list[int] | None = None
    top_logprobs_nums: list[int] | None = None
    extend_input_logprob_token_ids_gpu: torch.Tensor | None = None
    token_ids_logprobs: list[list[int]] | None = None

    # logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    temperature: torch.Tensor = None
    top_p_normalized_logprobs: bool = False
    top_p: torch.Tensor = None

    # DP attention metadata. Not needed when DP attention is not used.
    # Number of tokens in the request.
    global_num_tokens_gpu: torch.Tensor | None = None
    # The start position of local hidden states.
    dp_local_start_pos: torch.Tensor | None = None
    dp_local_num_tokens: torch.Tensor | None = None
    gathered_buffer: torch.Tensor | None = None
    # Buffer to gather logits from all ranks.
    forward_batch_gathered_buffer: torch.Tensor | None = None

    @classmethod
    def from_forward_context(cls, ctx: ForwardContext):
        return cls(
            forward_mode=ctx.forward_mode,
            capture_hidden_mode=ctx.capture_hidden_mode,
            gather_ids=ctx.gather_ids,
        )


_FUSED_LM_HEAD_GEMM = None


def _get_fused_lm_head_gemm():
    """Lazily import the fused lm_head GEMM kernel.

    The kernel is only present when tokenspeed-kernel was built with a
    compatible nvcc. Cache a sentinel when unavailable so we fall back
    to ``torch.matmul`` silently on subsequent calls.
    """
    global _FUSED_LM_HEAD_GEMM
    if _FUSED_LM_HEAD_GEMM is not None:
        return _FUSED_LM_HEAD_GEMM
    if not current_platform().is_nvidia:
        _FUSED_LM_HEAD_GEMM = (None, None)
        return _FUSED_LM_HEAD_GEMM
    try:
        from tokenspeed_kernel.thirdparty.cuda.lm_head_gemm import (
            lm_head_gemm,
            should_use_fused,
        )

        _FUSED_LM_HEAD_GEMM = (should_use_fused, lm_head_gemm)
    except Exception:
        _FUSED_LM_HEAD_GEMM = (None, None)
    return _FUSED_LM_HEAD_GEMM


def _lm_head_matmul(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``hidden_states @ weight.T``.

    Routes to the fused ``lm_head_gemm`` when the shape matches a compiled
    template and the bench-driven perf gate accepts (``should_use_fused``).
    Otherwise falls back to ``torch.matmul``.

    Only enabled for Kimi (``model_type == "kimi_k2"``) at the call site —
    on DSv3 the fused kernel's PDL launch surface caused a downstream EAGLE3
    spec decode AR regression that we have not characterised end-to-end; on
    Kimi the perf win is the largest and the regression has not been
    reproduced, so we gate the fused path to Kimi only.
    """
    cast_hidden = hidden_states.to(weight.dtype)
    should_use_fused, lm_head_gemm = _get_fused_lm_head_gemm()
    if should_use_fused is not None and should_use_fused(cast_hidden, weight):
        return lm_head_gemm(cast_hidden, weight, enable_pdl=True)
    return torch.matmul(cast_hidden, weight.T)


class LogitsProcessor(nn.Module):

    _LOGITS_AG_MAX_TOKENS = 128
    _LOGITS_AG_STATE_UNINITIALIZED = object()

    def __init__(
        self,
        config,
        skip_all_gather: bool = False,
        logit_scale: float | None = None,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        tp_group: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.config = config
        self.skip_all_gather = skip_all_gather
        self.dp_sampling_enabled = False
        self.dp_num_tokens_per_req = 1
        self.dp_sampling_min_bs = 0
        self.logit_scale = logit_scale
        self._logits_layout_executor: LogitsLayoutExecutor | None = None

        if tp_rank is None:
            assert tp_size is None
            assert tp_group is None
            tp_rank, tp_size = 0, 1
        assert 0 <= tp_rank < tp_size
        assert tp_size == 1 or tp_group is not None
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group

        self._all_gather_state = self._LOGITS_AG_STATE_UNINITIALIZED

        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

        # Gate the fused lm_head GEMM to Kimi only. See ``_lm_head_matmul``.
        self._use_fused_lm_head = getattr(self.config, "model_type", None) == "kimi_k2"

    def configure_dp_logits_layout(self, runtime: DpSamplingRuntimeConfig) -> None:
        if (
            not runtime.enabled
            or runtime.topology is None
            or runtime.min_bs is None
            or runtime.max_bucket_bs is None
            or runtime.vocab_size is None
            or runtime.device is None
        ):
            raise RuntimeError("enabled DP sampling runtime is incomplete")
        topology = runtime.topology
        self.dp_sampling_enabled = True
        self.dp_num_tokens_per_req = runtime.num_tokens_per_req
        self.dp_sampling_min_bs = runtime.min_bs
        self._logits_layout_executor = LogitsLayoutExecutor(
            tp_rank=topology.tp_rank,
            tp_size=topology.tp_size,
            tp_group=topology.tp_group,
            max_bucket_bs=runtime.max_bucket_bs,
            num_tokens_per_req=runtime.num_tokens_per_req,
            vocab_size=runtime.vocab_size,
            device=runtime.device,
        )

    def _resolve_logits_layout_plan(
        self,
        hidden_states: torch.Tensor,
        logits_metadata: LogitsMetadata,
    ) -> LogitsLayoutPlan | None:
        if not self.dp_sampling_enabled:
            return None
        if not logits_metadata.forward_mode.is_decode():
            return None
        n = self.dp_num_tokens_per_req
        rows = hidden_states.shape[0]
        assert rows % n == 0, f"hidden_states have {rows} rows, not divisible by N={n}"
        effective_bs = rows // n
        bucket_bs = ((effective_bs + self.tp_size - 1) // self.tp_size) * self.tp_size
        if effective_bs < self.dp_sampling_min_bs:
            return None
        return LogitsLayoutPlan(
            effective_bs=effective_bs,
            bucket_bs=bucket_bs,
            tp_size=self.tp_size,
            num_tokens_per_req=n,
        )

    def _init_all_gather_state(self, lm_head: VocabParallelEmbedding):
        if not current_platform().is_nvidia:
            return None

        if self.tp_size == 1 or self.skip_all_gather:
            return None

        vocab_padded = lm_head.weight.size(0) * self.tp_size
        if vocab_padded % (self.tp_size * 8) != 0:
            return None

        return create_state(
            group=pg_manager.get_process_group("nccl", self.tp_group),
            rank_in_group=self.tp_rank,
            max_tokens=self._LOGITS_AG_MAX_TOKENS,
            hidden_size=vocab_padded,
        )

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        aux_hidden_states: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        # Get the last hidden states and last logits for the next token prediction
        if not logits_metadata.extend_return_logprob:
            gather_ids = logits_metadata.gather_ids
            if gather_ids is not None:
                # Shapes align iff midlayer already pruned to one row per request
                # (draft first-step reduce). Other paths emit [N, H] with N > bs.
                if gather_ids.shape[0] == hidden_states.shape[0]:
                    pruned_states = hidden_states
                    if aux_hidden_states is not None:
                        aux_pruned_states = list(aux_hidden_states)
                else:
                    pruned_states = hidden_states[gather_ids]
                    if aux_hidden_states is not None:
                        aux_pruned_states = [h[gather_ids] for h in aux_hidden_states]
            else:
                assert (
                    not logits_metadata.forward_mode.is_extend_or_mixed()
                ), "EXTEND/MIXED forward must set gather_ids on ForwardContext"
                pruned_states = hidden_states
                if aux_hidden_states is not None:
                    aux_pruned_states = list(aux_hidden_states)

            sample_indices = None
            input_logprob_indices = None
        else:
            # Input logprobs are required.
            # Find 3 different indices.
            # 1. pruned_states: hidden states that we want logprobs from.
            # 2. sample_indices: Indices that have sampled tokens.
            # 3. input_logprob_indices: Indices that have input logprob tokens.
            sample_index_pt = -1
            sample_indices = []
            input_logprob_indices_pt = 0
            input_logprob_indices = []
            pt, pruned_states = 0, []
            for extend_logprob_start_len, extend_len in zip(
                logits_metadata.extend_logprob_start_lens_cpu,
                logits_metadata.extend_seq_lens_cpu,
            ):
                # It can happen in chunked prefill. We still need to sample 1 token,
                # But we don't want to include it in input logprob.
                if extend_len == extend_logprob_start_len:
                    start_len = extend_logprob_start_len - 1
                else:
                    start_len = extend_logprob_start_len

                # We always need at least 1 token to sample because that's required
                # by a caller.
                assert extend_len > start_len
                pruned_states.append(hidden_states[pt + start_len : pt + extend_len])
                pt += extend_len
                sample_index_pt += extend_len - start_len
                sample_indices.append(sample_index_pt)
                input_logprob_indices.extend(
                    [
                        input_logprob_indices_pt + i
                        for i in range(extend_len - extend_logprob_start_len)
                    ]
                )
                input_logprob_indices_pt += extend_len - start_len

            pruned_states = torch.cat(pruned_states)
            sample_indices = torch.tensor(
                sample_indices, device=pruned_states.device, dtype=torch.int64
            )
            input_logprob_indices = torch.tensor(
                input_logprob_indices, device=pruned_states.device, dtype=torch.int64
            )

        # Compute logits for both input and sampled tokens.
        logits_layout_plan = self._resolve_logits_layout_plan(
            pruned_states, logits_metadata
        )
        logits = self._get_logits(
            pruned_states, lm_head, logits_metadata, plan=logits_layout_plan
        )
        sampled_logits = (
            logits[sample_indices] if sample_indices is not None else logits
        )

        hidden_states_to_store: torch.Tensor | None = None
        if logits_metadata.capture_hidden_mode.need_capture():
            if logits_metadata.capture_hidden_mode.is_full():
                if aux_hidden_states is not None:
                    aux_hidden_states = (
                        aux_hidden_states[0]
                        if len(aux_hidden_states) == 1
                        else torch.cat(aux_hidden_states, dim=-1)
                    )
                    hidden_states_to_store = aux_hidden_states
                else:
                    hidden_states_to_store = hidden_states
            elif logits_metadata.capture_hidden_mode.is_last():
                # Get the last token hidden states. If sample_indices is None,
                # pruned states only contain the last tokens already.
                if aux_hidden_states is not None:
                    aux_pruned_states = (
                        aux_pruned_states[0]
                        if len(aux_pruned_states) == 1
                        else torch.cat(aux_pruned_states, dim=-1)
                    )
                    hidden_states_to_store = (
                        aux_pruned_states[sample_indices]
                        if sample_indices is not None
                        else aux_pruned_states
                    )
                else:
                    hidden_states_to_store = (
                        pruned_states[sample_indices]
                        if sample_indices is not None
                        else pruned_states
                    )
            else:
                assert False, "Should never reach"

        if not logits_metadata.extend_return_logprob:
            # Decode mode or extend mode without return_logprob.
            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                hidden_states=hidden_states_to_store,
                logits_layout_plan=logits_layout_plan,
            )
        else:
            input_logprobs = logits[input_logprob_indices]
            del hidden_states, logits

            # Normalize the logprob w/o temperature, top-p
            pruned_lens = torch.tensor(
                logits_metadata.extend_logprob_pruned_lens_cpu,
                device=input_logprobs.device,
            )
            if logits_metadata.temp_scaled_logprobs:
                logits_metadata.temperature = torch.repeat_interleave(
                    logits_metadata.temperature.view(-1),
                    pruned_lens,
                ).view(-1, 1)
            if logits_metadata.top_p_normalized_logprobs:
                logits_metadata.top_p = torch.repeat_interleave(
                    logits_metadata.top_p,
                    pruned_lens,
                )
            input_logprobs = self.compute_temp_top_p_normalized_logprobs(
                input_logprobs, logits_metadata
            )

            # Get the logprob of top-k tokens
            if logits_metadata.extend_return_top_logprob:
                (
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                ) = self.get_top_logprobs(input_logprobs, logits_metadata)
            else:
                input_top_logprobs_val = input_top_logprobs_idx = None

            # Get the logprob of given token id
            if logits_metadata.extend_token_ids_logprob:
                (
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                ) = self.get_token_ids_logprobs(input_logprobs, logits_metadata)
            else:
                input_token_ids_logprobs_val = input_token_ids_logprobs_idx = None

            input_token_logprobs = input_logprobs[
                torch.arange(input_logprobs.shape[0], device=input_logprobs.device),
                logits_metadata.extend_input_logprob_token_ids_gpu,
            ]

            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                input_token_logprobs=input_token_logprobs,
                input_top_logprobs_val=input_top_logprobs_val,
                input_top_logprobs_idx=input_top_logprobs_idx,
                hidden_states=hidden_states_to_store,
                input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: torch.Tensor | None = None,
        plan: LogitsLayoutPlan | None = None,
    ) -> torch.Tensor:
        """Get logits from hidden_states.

        If sampled_logits_only is True, it means hidden_states only contain the
        last position (e.g., extend without input logprobs). The caller should
        guarantee the given hidden_states follow this constraint.
        """
        dp_sampling = plan is not None
        assert (not dp_sampling) or self.dp_sampling_enabled, (
            "DP logits layout plan was provided but LogitsProcessor was not "
            "configured with dp_sampling"
        )

        if dp_sampling and self.skip_all_gather:
            if self._logits_layout_executor is None:
                raise RuntimeError(
                    "dp_sampling logits layout executor is not configured"
                )
            hidden_states = self._logits_layout_executor.slice_hidden_states(
                hidden_states, plan
            )

        if hasattr(lm_head, "weight"):
            if self._use_fused_lm_head:
                logits = _lm_head_matmul(hidden_states, lm_head.weight)
            else:
                logits = torch.matmul(
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
        else:
            # GGUF models
            logits = lm_head.linear_method.apply(lm_head, hidden_states, embedding_bias)

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if dp_sampling and not self.skip_all_gather:
            if self._logits_layout_executor is None:
                raise RuntimeError(
                    "dp_sampling logits layout executor is not configured"
                )
            logits = self._logits_layout_executor.swap_batch_vocab(logits, plan)
        elif not dp_sampling and self.tp_size > 1 and not self.skip_all_gather:
            if self._all_gather_state is self._LOGITS_AG_STATE_UNINITIALIZED:
                self._all_gather_state = self._init_all_gather_state(lm_head)

            if (
                self._all_gather_state is not None
                and logits.size(0) <= self._LOGITS_AG_MAX_TOKENS
            ):
                # skip_entry_sync=True assumes other sync points existing between two all_gather_inner calls.
                logits = all_gather_inner(
                    self._all_gather_state,
                    logits,
                    tp_hidden_dim=logits.size(-1) * self.tp_size,
                    skip_entry_sync=True,
                    safe=False,
                )
            else:
                num_rows = logits.size(0)
                local_vocab_size = logits.size(1)
                gathered_logits = torch.empty(
                    self.tp_size * num_rows,
                    local_vocab_size,
                    dtype=logits.dtype,
                    device=logits.device,
                )
                all_gather_into_tensor(gathered_logits, logits, self.tp_group)
                logits = (
                    gathered_logits.view(self.tp_size, num_rows, local_vocab_size)
                    .transpose(0, 1)
                    .contiguous()
                    .view(num_rows, local_vocab_size * self.tp_size)
                )

        logits = logits[:, : self.config.vocab_size].contiguous()

        if self.final_logit_softcapping:
            fused_softcap_generic(logits, self.final_logit_softcapping)

        return logits

    @staticmethod
    def get_top_logprobs(all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata):
        max_k = max(logits_metadata.top_logprobs_nums)
        ret = all_logprobs.topk(max_k, dim=1)
        values = ret.values.tolist()
        indices = ret.indices.tolist()

        input_top_logprobs_val, input_top_logprobs_idx = [], []

        pt = 0
        for k, pruned_len in zip(
            logits_metadata.top_logprobs_nums,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_top_logprobs_val.append([])
                input_top_logprobs_idx.append([])
                continue

            input_top_logprobs_val.append(
                [values[pt + j][:k] for j in range(pruned_len)]
            )
            input_top_logprobs_idx.append(
                [indices[pt + j][:k] for j in range(pruned_len)]
            )
            pt += pruned_len

        return input_top_logprobs_val, input_top_logprobs_idx

    @staticmethod
    def get_token_ids_logprobs(
        all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
    ):
        input_token_ids_logprobs_val, input_token_ids_logprobs_idx = [], []
        pt = 0
        for token_ids, pruned_len in zip(
            logits_metadata.token_ids_logprobs,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_token_ids_logprobs_val.append([])
                input_token_ids_logprobs_idx.append([])
                continue

            input_token_ids_logprobs_val.append(
                [all_logprobs[pt + j, token_ids].tolist() for j in range(pruned_len)]
            )
            input_token_ids_logprobs_idx.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len

        return input_token_ids_logprobs_val, input_token_ids_logprobs_idx

    @staticmethod
    def compute_temp_top_p_normalized_logprobs(
        last_logits: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> torch.Tensor:
        """
        compute logprobs for the output token from the given logits.

        Returns:
            torch.Tensor: logprobs from logits
        """
        last_logits = last_logits.float()
        # Scale logits if temperature scaling is enabled
        if logits_metadata.temp_scaled_logprobs:
            last_logits = last_logits / logits_metadata.temperature

        # Normalize logprobs if top_p normalization is enabled
        #  only normalize logprobs when top_p is set and not equal to 1.0
        if (
            logits_metadata.top_p_normalized_logprobs
            and (logits_metadata.top_p != 1.0).any()
        ):
            from tokenspeed.runtime.sampling.utils import top_p_normalize_probs_torch

            probs = torch.softmax(last_logits, dim=-1)
            del last_logits
            probs = top_p_normalize_probs_torch(probs, logits_metadata.top_p)
            return torch.log(probs)
        else:
            return torch.nn.functional.log_softmax(last_logits, dim=-1)


@triton.jit
def fused_softcap_kernel(
    full_logits_ptr,
    softcapping_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(full_logits_ptr + offsets, mask=mask).to(tl.float32)

    # Perform operations in-place
    x = x / softcapping_value

    # Stable tanh form; the exp ratio overflows to inf/inf for large logits.
    x = 2 * tl.sigmoid(2 * x) - 1

    x = x * softcapping_value

    # Store result
    tl.store(full_logits_ptr + offsets, x, mask=mask)


def fused_softcap(full_logits, final_logit_softcapping):
    n_elements = full_logits.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    fused_softcap_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits


def fused_softcap_generic(full_logits, final_logit_softcapping):
    return fused_softcap(full_logits, final_logit_softcapping)
