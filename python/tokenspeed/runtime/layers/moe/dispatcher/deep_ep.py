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

from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    per_token_group_quant_fp8,
)
from tokenspeed_kernel.ops.moe.deepep import Buffer

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.moe.config import ExpertParallelConfig
from tokenspeed.runtime.utils import get_available_gpu_memory, get_colorful_logger

logger = get_colorful_logger(__file__)


class DeepEPMode(Enum):
    normal = "normal"
    low_latency = "low_latency"
    auto = "auto"

    def enable_normal(self):
        return self in [DeepEPMode.normal, DeepEPMode.auto]

    def enable_low_latency(self):
        return self in [DeepEPMode.low_latency, DeepEPMode.auto]

    def resolve(self, forward_mode):
        if self != DeepEPMode.auto:
            return self

        if forward_mode.is_decode():
            return DeepEPMode.low_latency
        else:
            return DeepEPMode.normal


class DeepEPDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()


class DeepEPBuffer:
    _buffer = None
    _dispatch_mode: DeepEPDispatchMode | None = None
    _hidden_size: int | None = None
    _num_max_dispatch_tokens_per_rank: int | None = None
    _num_experts: int | None = None

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = None,
        num_experts: int = None,
    ):
        if cls._buffer is not None:
            return cls._buffer

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                Buffer.get_dispatch_config(group.size()),
                Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank is not None
            assert num_experts is not None and num_experts % group.size() == 0
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        # Calculate num_qps_per_rank consistently with DeepEP examples:
        # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
        if deepep_mode == DeepEPMode.normal:
            num_qps_per_rank = Buffer.num_sms
        elif deepep_mode == DeepEPMode.low_latency:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py#L176
            num_qps_per_rank = num_experts // group.size()
        elif deepep_mode == DeepEPMode.auto:
            # low-latency and normal mode all need run
            num_qps_per_rank = max(Buffer.num_sms, num_experts // group.size())
        else:
            raise NotImplementedError

        free_gpu_memory_begin = get_available_gpu_memory(
            "cuda", torch.cuda.current_device()
        )
        cls._buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=deepep_mode.enable_low_latency(),
            num_qps_per_rank=num_qps_per_rank,
            allow_mnnvl=True,
        )
        free_gpu_memory_end = get_available_gpu_memory(
            "cuda", torch.cuda.current_device()
        )
        logger.info(
            "DeepEPBuffer use memory %s GB", free_gpu_memory_begin - free_gpu_memory_end
        )
        return cls._buffer

    @classmethod
    def clean_buffer(cls):
        if cls._buffer is None:
            return
        if not cls._buffer.low_latency_mode:
            return
        cls._buffer.clean_low_latency_buffer(
            cls._num_max_dispatch_tokens_per_rank,
            cls._hidden_size,
            cls._num_experts,
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls):
        cls._dispatch_mode = DeepEPDispatchMode.NORMAL

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):
        if cls._dispatch_mode == DeepEPDispatchMode.NORMAL:
            cls.clean_buffer()
        cls._dispatch_mode = DeepEPDispatchMode.LOW_LATENCY


class _DeepEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
        low_latency_max_num_tokens_per_gpu: int,
    ):
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        self.num_max_dispatch_tokens_per_rank = low_latency_max_num_tokens_per_gpu

        self.handle = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_origin_input: torch.Tensor = None,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        raise NotImplementedError


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.src2dst = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        hidden_states = per_token_group_quant_fp8(hidden_states, 128)
        topk_idx = topk_idx.to(torch.int64)
        topk_weights = topk_weights.to(torch.float32)
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, topk_idx, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_idx, topk_weights, previous_event):
        (
            hidden_states,
            topk_idx,
            topk_weights,
            num_recv_tokens_per_expert_list,
            event,
        ) = self._dispatch_core(hidden_states, topk_idx, topk_weights, previous_event)
        event.current_stream_wait() if self.async_finish else ()

        return (
            hidden_states,
            topk_idx,
            topk_weights,
            None,  # reorder_topk_ids
            num_recv_tokens_per_expert_list,
            None,  # seg_indptr
            None,  # masked_m
        )

    def _dispatch_core(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        # Note: We intentionally do not switch devices here.
        # DeepEP buffer is initialized on a specific device context and
        # switching devices during dispatch can cause "invalid resource handle" errors.
        # The caller is responsible for ensuring tensors are on the correct device.
        buffer = self._get_buffer()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            self.num_experts,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )

        # In principle ``handle`` should travel alongside the dispatched tokens
        # into combine(). Today that path triggers a synchronization issue, so
        # keep the handle on the dispatcher instance instead.

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            expert_alignment=128,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: tuple[torch.Tensor, torch.Tensor],
        moe_origin_input: torch.Tensor = None,
    ):
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, previous_event

    def combine_b(
        self,
        output: torch.Tensor,
        previous_event,
        topk_idx: torch.Tensor,
        topk_weights: tuple[torch.Tensor, torch.Tensor],
        moe_origin_input: torch.Tensor = None,
    ):
        hidden_states, event = self._combine_core(
            output, previous_event, topk_idx, topk_weights, moe_origin_input
        )
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(
        self,
        x: torch.Tensor,
        previous_event,
        topk_idx: torch.Tensor,
        topk_weights: tuple[torch.Tensor, torch.Tensor],
        moe_origin_input: torch.Tensor = None,
    ):
        topk_idx_ori, topk_weights_ori, topk_weights_recv = (
            (topk_idx, topk_weights[0], topk_weights[1])
            if moe_origin_input is not None
            else (topk_idx, None, topk_weights)
        )
        buffer = self._get_buffer()
        combine_args = {
            "x": x,
            "handle": self.handle,
            "async_finish": self.async_finish,
            "previous_event": previous_event,
            "allocate_on_comm_stream": previous_event is not None,
        }
        if moe_origin_input is not None:
            combine_args.update(
                {
                    "topk_weights": topk_weights_recv,
                    "topk_idx_ori": topk_idx_ori,
                    "topk_weights_ori": topk_weights_ori,
                    "x_ori": moe_origin_input,
                }
            )
        combined_x, _, event = buffer.combine(**combine_args)
        return combined_x, event

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_normal()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
    def __init__(self, return_recv_hook: bool, use_fp8: bool = False, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """
        self.return_recv_hook = return_recv_hook
        self.use_fp8 = use_fp8

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        # DeepEP requires independent contiguous tensors to prevent issues with
        # upstream tensor aliasing or non-standard strides. We clone to ensure
        # complete memory isolation, which is critical for low-latency dispatch.
        #
        # Dtype requirements:
        # - hidden_states: preserve original dtype (bf16/fp16/fp32)
        # - topk_idx: must be int64 (DeepEP C++ kernel API requirement for expert indices)
        # - topk_weights: use float32 for routing precision to avoid numerical issues
        hidden_states = hidden_states.contiguous().clone()
        topk_idx = topk_idx.to(torch.int64).contiguous().clone()
        topk_weights = topk_weights.to(torch.float32).contiguous().clone()
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_idx,
            use_fp8=self.use_fp8,
        )
        return (
            hidden_states,
            topk_idx,
            topk_weights,
            masked_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_idx,
        topk_weights,
        masked_m,
        event,
        hook,
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()

        return (
            hidden_states,
            topk_idx,
            topk_weights,
            None,  # reorder_topk_ids
            None,  # num_recv_tokens_per_expert_list
            None,  # seg_indptr
            masked_m,
        )

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        use_fp8: bool = False,
    ):
        """
        # For H20, there will be an CUDA error: DeepEP/csrc/kernels/internode_ll.cu:337 'too many blocks in cooperative launch'.
        # Please make sure to change DeepEP code in internode_ll.cu dispatch / combine as below first and then reinstall.
        # More details refer: https://github.com/deepseek-ai/DeepEP/issues/15#issuecomment-2709715782

        diff --git a/csrc/kernels/internode_ll.cu b/csrc/kernels/internode_ll.cu
        index 76ae2e2..8ecd08f 100644
        --- a/csrc/kernels/internode_ll.cu
        +++ b/csrc/kernels/internode_ll.cu
        @@ -310,8 +310,8 @@ void dispatch(void* packed_recv_x, float* packed_recv_x_scales,
                    int num_topk, int num_experts, int rank, int num_ranks, bool use_fp8,
                    void* workspace, cudaStream_t stream, int phases) {
            constexpr int kNumMaxTopK = 9;
        -    constexpr int kNumWarpsPerGroup = 10;
        -    constexpr int kNumWarpGroups = 3;
        +    constexpr int kNumWarpsPerGroup = 8;
        +    constexpr int kNumWarpGroups = 4;
            EP_STATIC_ASSERT(kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup, "Too many top-k selections");

            const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
        @@ -501,8 +501,8 @@ void combine(void* combined_x,
                    int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
                    int num_topk, int num_experts, int rank, int num_ranks,
                    void* workspace, cudaStream_t stream, int phases) {
        -    constexpr int kNumWarpsPerGroup = 10;
        -    constexpr int kNumWarpGroups = 3;
        +    constexpr int kNumWarpsPerGroup = 8;
        +    constexpr int kNumWarpGroups = 4;
            constexpr int kNumMaxTopk = 9;

            const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
        """
        buffer = self._get_buffer()
        packed_recv_hidden, packed_recv_count, self.handle, event, hook = (
            buffer.low_latency_dispatch(
                hidden_states,
                topk_idx,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
            )
        )
        return packed_recv_hidden, packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_origin_input: torch.Tensor = None,
    ):
        hidden_states, event, hook = self._combine_core(
            hidden_states, topk_idx, topk_weights, moe_origin_input
        )
        return hidden_states, event, hook

    def combine_b(self, hidden_states, event, hook):
        hook() if self.return_recv_hook else event.current_stream_wait()
        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_origin_input: torch.Tensor = None,
    ):
        buffer = self._get_buffer()
        combined_hidden_states, event, hook = buffer.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        )
        self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class DeepEPDispatcher:
    def __init__(
        self,
        config: ExpertParallelConfig,
        deepep_mode: DeepEPMode = DeepEPMode.auto,
        async_finish: bool = True,
        return_recv_hook: bool = True,
        use_fp8: bool = False,
    ):
        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=config.group,
            router_topk=config.top_k,
            permute_fusion=True,
            num_experts=config.num_experts,
            num_local_experts=config.num_experts // config.world_size,
            hidden_size=config.hidden_size,
            params_dtype=config.params_dtype,
            deepep_mode=deepep_mode,
            low_latency_max_num_tokens_per_gpu=config.low_latency_max_num_tokens_per_gpu,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                return_recv_hook=return_recv_hook,
                use_fp8=use_fp8,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )

    def dispatch(self, *args, **kwargs) -> tuple:
        self.dispatch_a(*args, **kwargs)
        return self.dispatch_b()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        topk_idx = topk_idx.to(torch.int64)
        inner_state = self._get_impl(forward_mode).dispatch_a(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        self._dispatch_intermediate_state = forward_mode, inner_state

    def dispatch_b(self):
        forward_mode, inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl(forward_mode).dispatch_b(*inner_state)

    def combine(self, *args, **kwargs) -> tuple:
        self.combine_a(*args, **kwargs)
        return self.combine_b()

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        forward_mode: ForwardMode,
        moe_origin_input: torch.Tensor = None,
    ):
        topk_idx = topk_idx.to(torch.int64)
        inner_state = self._get_impl(forward_mode).combine_a(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            moe_origin_input=moe_origin_input,
        )
        self._combine_intermediate_state = (
            forward_mode,
            inner_state,
            topk_idx,
            topk_weights,
            moe_origin_input,
        )

    def combine_b(self):
        forward_mode, inner_state, topk_idx, topk_weights, moe_origin_input = (
            self._combine_intermediate_state
        )
        if self.deepep_mode.resolve(forward_mode) == DeepEPMode.normal:
            inner_state = inner_state + (topk_idx, topk_weights, moe_origin_input)
        del self._combine_intermediate_state
        return self._get_impl(forward_mode).combine_b(*inner_state)

    def _get_impl(self, forward_mode: ForwardMode) -> _DeepEPDispatcherImplBase:
        resolved_deepep_mode = self.deepep_mode.resolve(forward_mode)
        if resolved_deepep_mode == DeepEPMode.normal:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.low_latency:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")
