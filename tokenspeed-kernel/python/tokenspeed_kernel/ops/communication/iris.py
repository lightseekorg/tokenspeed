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

import importlib
import logging
import math
import pkgutil
from typing import List, Tuple

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import (
    gl,
    gluon,
    redirect_triton_to_tokenspeed_triton,
    tl,
    triton,
)

# iris does plain ``import triton`` at module load time; route those bindings
# to the vendored ``tokenspeed_triton`` so iris and tokenspeed-kernel share a
# single triton distribution. See
# :func:`redirect_triton_to_tokenspeed_triton` for details.
with redirect_triton_to_tokenspeed_triton():
    import iris  # noqa: E402

    # Pre-import every iris kernel module that does ``import triton`` at module
    # load time (the CCL APIs above lazy-import them at call time, when the
    # redirect is no longer active).
    import iris.ccl.triton  # noqa: E402
    from iris.ccl import Config as _IrisConfig  # noqa: E402
    from iris.ccl.all_gather import all_gather as _iris_all_gather  # noqa: E402
    from iris.ccl.reduce_scatter import (  # noqa: E402
        reduce_scatter as _iris_reduce_scatter,
    )

    for _info in pkgutil.walk_packages(
        iris.ccl.triton.__path__, prefix="iris.ccl.triton."
    ):
        importlib.import_module(_info.name)

from tokenspeed_kernel.platform import current_platform  # noqa: E402

logger = logging.getLogger(__file__)

_platform = current_platform()

__all__ = [
    "IrisAllReduce",
    "IrisRSAG",
    "IrisAllReduceResidualRMSNorm",
    "create_iris_state",
    "iris_all_reduce",
    "iris_all_reduce_two",
    "iris_all_reduce_residual_attnres",
    "create_iris_rsag_state",
    "create_iris_ar_rmsnorm_state",
    "iris_allreduce_residual_rmsnorm",
    "IRIS_AR_STATES",
    "IRIS_AR_RMSNORM_STATES",
]


IRIS_AR_STATES: dict = {}
IRIS_AR_RMSNORM_STATES: dict = {}


def _get_available_gpu_memory(gpu_id: int, empty_cache: bool = True) -> float:
    if torch.cuda.is_available():
        with torch.cuda.device(gpu_id):
            if empty_cache:
                torch.cuda.empty_cache()
            free_gpu_memory, _ = torch.cuda.mem_get_info()
            return free_gpu_memory / (1 << 30)
    return 0.0


_iris_ctx_singleton = None


def _get_or_create_iris_context(heap_size: int):
    global _iris_ctx_singleton
    if _iris_ctx_singleton is None:
        _iris_ctx_singleton = iris.iris(heap_size=heap_size)
    return _iris_ctx_singleton


class IrisRSAG(object):
    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_tokens: int,
        hidden_size: int,
        device: torch.device = None,
        heap_size: int | None = None,
    ) -> None:
        assert (
            type(group) == dist.ProcessGroup
        ), f"Expected dist.ProcessGroup, got {type(group)}"
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "IrisRSAG; call dist.init_process_group() first."
        )
        assert (
            _platform.is_amd
        ), f"IrisRSAG currently targets AMD ROCm; got non-AMD platform: {_platform}"
        assert (
            group == dist.group.WORLD or group.size() == dist.get_world_size()
        ), "iris.ccl all_gather/reduce_scatter do not accept a sub-group."

        self.group = group
        self.rank_in_group = rank_in_group
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.dtype = torch.bfloat16
        self.world_size = group.size()

        # Heap holds in/out flat buffers plus iris bookkeeping; over-provision
        # similarly to ``IrisAllReduce`` to leave room for ring/spinlock flags.
        if heap_size is None:
            buf_bytes = max_tokens * hidden_size * self.dtype.itemsize
            heap_size = max(1 << 28, 4 * buf_bytes + (16 << 20))

        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        self._ctx = _get_or_create_iris_context(heap_size)
        self._in_buff = self._ctx.empty((max_tokens, hidden_size), dtype=self.dtype)
        self._out_buff = self._ctx.empty((max_tokens, hidden_size), dtype=self.dtype)
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris RSAG symmetric-heap buffers allocated: %s GB",
            free_gpu_memory_begin - free_gpu_memory_after,
        )

        assert self._ctx.get_num_ranks() == dist.get_world_size(), (
            f"Iris world size {self._ctx.get_num_ranks()} "
            f"!= torch world size {dist.get_world_size()}"
        )
        assert self.rank_in_group == self._ctx.get_rank(), (
            f"rank mismatch: rank_in_group={self.rank_in_group}, "
            f"iris rank={self._ctx.get_rank()}"
        )

    # -- token-distribution helpers (mirror sibling classes) ----------------

    def get_token_dist(self, total_tokens_in_group: int) -> list:
        token_list_in_group = []
        for rank in range(self.world_size):
            num_tokens_per_rank = total_tokens_in_group // self.world_size + (
                1 if (rank < total_tokens_in_group % self.world_size) else 0
            )
            token_list_in_group.append(num_tokens_per_rank)
        return token_list_in_group

    def get_context(self, token_list_in_group: list) -> Tuple[int, int, int]:
        total_num_tokens = sum(token_list_in_group)
        assert (
            total_num_tokens <= self.max_tokens
        ), f"The inner comm buffer is too small: {total_num_tokens=} is not <= {self.max_tokens=}"
        local_num_tokens = token_list_in_group[self.rank_in_group]
        local_token_offset = sum(token_list_in_group[: self.rank_in_group])
        return total_num_tokens, local_num_tokens, local_token_offset

    # -- internal helpers ---------------------------------------------------

    def _assert_uniform(self, token_list_in_group: List[int]) -> int:
        first = token_list_in_group[0]
        assert all(t == first for t in token_list_in_group), (
            "IrisRSAG requires uniform tokens per rank; got "
            f"token_list_in_group={token_list_in_group}"
        )
        return first

    @staticmethod
    def _pick_block_n(hidden_size: int) -> int:
        # Pick the largest power-of-two block that divides hidden_size, capped
        # at 256. This keeps the iris kernel on its no-mask fast path and
        # still produces enough tiles (world_size * hidden/block_n) to fill
        # ``comm_sms`` SMs on supported AMD chips.
        for cand in (256, 128, 64, 32, 16):
            if hidden_size % cand == 0:
                return cand
        return hidden_size

    def _make_config(self, local_num_tokens: int, hidden_size: int):
        # ``swizzle_size=1`` keeps tile_id ordering row-major in M, which is
        # required so that block-distribution (DISTRIBUTION=1) hands rank r
        # exactly the K tiles spanning rows [r*local, (r+1)*local) in the
        # reduce-scatter kernel. ``all_gather`` is rank-agnostic on tile order
        # so the same config is fine.
        return _IrisConfig(
            block_size_m=local_num_tokens,
            block_size_n=self._pick_block_n(hidden_size),
            swizzle_size=1,
            all_reduce_distribution=1,
        )

    # -- public collective ops ---------------------------------------------

    def reduce_scatter(
        self,
        hidden_states: torch.Tensor,
        tp_num_tokens: int = None,
        token_list_in_group: List[int] = None,
        safe=True,
    ) -> torch.Tensor:
        assert (
            tp_num_tokens is not None or token_list_in_group is not None
        ), "Either tp_num_tokens or token_list_in_group must be provided"
        if token_list_in_group is None:
            token_list_in_group = self.get_token_dist(tp_num_tokens)
        assert (
            hidden_states.dtype == self.dtype
        ), f"Only {self.dtype} is supported, got {hidden_states.dtype}"

        local_num_tokens = self._assert_uniform(token_list_in_group)
        total_num_tokens, _, local_token_offset = self.get_context(token_list_in_group)
        assert (hidden_states.shape[0] == total_num_tokens) and (
            hidden_states.shape[-1] == self.hidden_size
        ), (
            f"Mismatched shape, {hidden_states.shape[0]=} != {total_num_tokens=} "
            f"or {hidden_states.shape[-1]=} != {self.hidden_size=} "
            f"{hidden_states.shape=}"
        )

        if local_num_tokens == 0:
            return torch.empty(
                (0, self.hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        in_view = self._in_buff[:total_num_tokens, : self.hidden_size]
        out_view = self._out_buff[:total_num_tokens, : self.hidden_size]
        in_view.copy_(hidden_states)

        # Ensure every rank's shared input copy is visible before peer loads begin.
        self._ctx.device_barrier()

        config = self._make_config(local_num_tokens, self.hidden_size)
        _iris_reduce_scatter(out_view, in_view, self._ctx, config=config)

        output = out_view[local_token_offset : local_token_offset + local_num_tokens, :]
        return output.clone() if safe else output

    def all_gather(
        self,
        hidden_states: torch.Tensor,
        tp_num_tokens: int = None,
        token_list_in_group: List[int] = None,
        safe=True,
    ) -> torch.Tensor:
        assert (
            tp_num_tokens is not None or token_list_in_group is not None
        ), "Either tp_num_tokens or token_list_in_group must be provided"
        if token_list_in_group is None:
            token_list_in_group = self.get_token_dist(tp_num_tokens)
        assert (
            hidden_states.dtype == self.dtype
        ), f"Only {self.dtype} is supported, got {hidden_states.dtype}"

        local_num_tokens = self._assert_uniform(token_list_in_group)
        total_num_tokens, _, _ = self.get_context(token_list_in_group)
        hidden_size = hidden_states.shape[-1]
        assert (hidden_states.shape[0] == local_num_tokens) and (
            hidden_size <= self.hidden_size
        ), (
            f"{hidden_states.shape=}|{local_num_tokens=}|{hidden_states.device=} "
            "Mismatched shape"
        )

        if local_num_tokens == 0:
            return torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        in_view = self._in_buff[:local_num_tokens, :hidden_size]
        out_view = self._out_buff[:total_num_tokens, :hidden_size]
        in_view.copy_(hidden_states)

        self._ctx.device_barrier()

        config = self._make_config(local_num_tokens, hidden_size)
        _iris_all_gather(out_view, in_view, self._ctx, config=config)

        return out_view.clone() if safe else out_view


class IrisAllReduce(object):
    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        heap_size: int | None = None,
        device: torch.device = None,
        config=None,
    ) -> None:
        assert (
            type(group) == dist.ProcessGroup
        ), f"Expected dist.ProcessGroup, got {type(group)}"
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "IrisAllReduce; call dist.init_process_group() first."
        )
        assert _platform.is_amd, (
            "IrisAllReduce currently targets AMD ROCm; "
            f"got non-AMD platform: {_platform}"
        )

        self.group = group
        self.rank_in_group = rank_in_group
        self.max_numel = max_numel
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self._config = config or _IrisConfig(
            block_size_m=1,
            block_size_n=256,
            swizzle_size=4,
            comm_sms=64,
            all_reduce_variant="one_shot",
            all_reduce_distribution=1,
        )

        # Heap holds two flat buffers of ``max_numel * itemsize`` plus iris
        # bookkeeping; we leave generous headroom (~16 MiB) for internal
        # workspaces such as ring/spinlock flags.
        if heap_size is None:
            buf_bytes = max_numel * dtype.itemsize
            heap_size = max(1 << 28, 4 * buf_bytes + (16 << 20))

        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        self._ctx = _get_or_create_iris_context(heap_size)
        self.world_size = group.size()
        self._input_buf = self._ctx.zeros((max_numel,), dtype=dtype)
        self._block_size = 2048
        self._max_blocks = triton.cdiv(max_numel, self._block_size)
        # Private epochs are reserved for exact K3 specializations so ordinary
        # all-reduce launches cannot advance their staging counters. The K3
        # two-segment reduction uses 21 half-width staging blocks.
        self._ready_flags = self._ctx.zeros(
            (self._max_blocks + 64, self.world_size), dtype=torch.int32
        )
        self._heap_base_addresses = tuple(
            int(address) for address in self._ctx.get_heap_bases().tolist()
        )
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris all-reduce symmetric-heap buffers allocated: %s GB",
            free_gpu_memory_begin - free_gpu_memory_after,
        )

        self._rank_start = 0
        self._rank_stride = 1
        self._iris_rank = dist.get_rank()
        self._workspace = None

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op=None,
        safe: bool = True,
        async_op: bool = False,
    ) -> torch.Tensor:
        if op is None:
            op = dist.ReduceOp.SUM
        assert op == dist.ReduceOp.SUM, f"Iris all-reduce only supports SUM, got {op}"
        assert not async_op, "Iris all-reduce does not support async_op"
        assert tensor.dtype == self.dtype, (
            f"Iris all-reduce dtype mismatch: tensor={tensor.dtype}, "
            f"backend={self.dtype}"
        )
        numel = tensor.numel()
        assert (
            numel <= self.max_numel
        ), f"tensor numel ({numel}) exceeds iris buffer capacity ({self.max_numel})"
        if tensor.dim() >= 2:
            n_dim = tensor.shape[-1]
            m_dim = numel // n_dim
        else:
            m_dim, n_dim = 1, numel
        in_view = self._input_buf.narrow(0, 0, numel).view(m_dim, n_dim)
        iris_stage_one_shot_allreduce_kernel[(triton.cdiv(numel, self._block_size),)](
            tensor.view(-1),
            in_view.view(-1),
            tensor.view(-1),
            self._ready_flags,
            self._ctx.get_heap_bases(),
            numel,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            BLOCK_SIZE=self._block_size,
            num_warps=4,
        )

        return tensor.clone() if safe else tensor

    def all_reduce_two(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        op=None,
        safe: bool = True,
        async_op: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce two BF16 tensors with one launch and synchronization epoch.

        The tensors remain separate at the API and output boundaries. The
        kernel lays them out consecutively only in Iris's symmetric input
        buffer, avoiding an explicit concatenation or staging copy.

        Args:
            first: First contiguous tensor to reduce in place.
            second: Second contiguous tensor to reduce in place.
            op: Reduction operation. Only ``SUM`` is supported.
            safe: Return cloned outputs when true; otherwise return the input
                tensors after their in-place reductions.
            async_op: Must be false; asynchronous execution is unsupported.

        Returns:
            The two reduced tensors, in the same order as the inputs.
        """
        if op is None:
            op = dist.ReduceOp.SUM
        assert op == dist.ReduceOp.SUM, f"Iris all-reduce only supports SUM, got {op}"
        assert not async_op, "Iris all-reduce does not support async_op"
        assert first.dtype == self.dtype and second.dtype == self.dtype, (
            "Iris all-reduce dtype mismatch: "
            f"first={first.dtype}, second={second.dtype}, backend={self.dtype}"
        )
        assert (
            first.is_contiguous() and second.is_contiguous()
        ), "Iris two-tensor all-reduce requires contiguous inputs"
        assert first.device == second.device == self.device, (
            "Iris two-tensor all-reduce inputs must be on the state device: "
            f"first={first.device}, second={second.device}, state={self.device}"
        )
        first_numel = first.numel()
        second_numel = second.numel()
        total_numel = first_numel + second_numel
        assert (
            first_numel > 0 and second_numel > 0
        ), "Iris two-tensor all-reduce requires non-empty inputs"
        assert total_numel <= self.max_numel, (
            f"combined tensor numel ({total_numel}) exceeds iris buffer capacity "
            f"({self.max_numel})"
        )

        grid = (triton.cdiv(total_numel, self._block_size),)
        if (
            _platform.is_cdna4
            and self.world_size == 8
            and first_numel == 7168
            and second_numel == 3584
            and self._is_symmetric_staging_pair(first, second)
        ):
            first, second = self._all_reduce_two_symmetric(first, second)
            if safe:
                return first.clone(), second.clone()
            return first, second
        if (
            _platform.is_cdna4
            and self.world_size == 8
            and first_numel == 7168
            and second_numel == 3584
        ):
            self._all_reduce_two_rank_epoch(first, second, safe=False)
        else:
            iris_stage_one_shot_allreduce_two_kernel[grid](
                first.view(-1),
                second.view(-1),
                self._input_buf,
                first.view(-1),
                second.view(-1),
                self._ready_flags,
                self._ctx.get_heap_bases(),
                first_numel,
                total_numel,
                RANK=self._iris_rank,
                WORLD_SIZE=self.world_size,
                BLOCK_SIZE=self._block_size,
                num_warps=4,
            )

        if safe:
            return first.clone(), second.clone()
        return first, second

    def _all_reduce_two_rank_epoch(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        safe: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce the exact K3 MoE pair with one readiness epoch per rank."""
        assert _platform.is_cdna4 and self.world_size == 8
        assert first.dtype == second.dtype == self.dtype == torch.bfloat16
        assert first.is_contiguous() and second.is_contiguous()
        assert first.device == second.device == self.device
        assert first.numel() == 7168 and second.numel() == 3584
        iris_stage_one_shot_allreduce_two_rank_epoch_gluon_kernel[(21,)](
            first.view(-1),
            second.view(-1),
            self._input_buf,
            first.view(-1),
            second.view(-1),
            self._ready_flags,
            *self._heap_base_addresses,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            FIRST_NUMEL=first.numel(),
            TOTAL_NUMEL=first.numel() + second.numel(),
            BLOCK_SIZE=512,
            FLAG_BASE=self._max_blocks + 7,
            NUM_BLOCKS=21,
            ELEMENTS_PER_THREAD=2,
            NUM_WARPS=1,
            num_warps=1,
        )
        if safe:
            return first.clone(), second.clone()
        return first, second

    def staging_two(
        self,
        first_shape: tuple[int, ...],
        second_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return consecutive views of the symmetric Iris input buffer."""
        first_numel = math.prod(first_shape)
        second_numel = math.prod(second_shape)
        if first_numel + second_numel > self.max_numel:
            raise ValueError("Iris staging views exceed the input buffer")
        first = self._input_buf.narrow(0, 0, first_numel).view(first_shape)
        second = self._input_buf.narrow(0, first_numel, second_numel).view(second_shape)
        return first, second

    def _is_symmetric_staging_pair(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> bool:
        element_size = self._input_buf.element_size()
        return (
            first.data_ptr() == self._input_buf.data_ptr()
            and second.data_ptr()
            == self._input_buf.data_ptr() + first.numel() * element_size
        )

    def _all_reduce_two_symmetric(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce an exact K3 pair already staged in symmetric memory."""
        assert first.numel() == 7168 and second.numel() == 3584
        iris_reduce_symmetric_two_gluon_kernel[(21,)](
            self._input_buf,
            first.view(-1),
            second.view(-1),
            self._ready_flags,
            *self._heap_base_addresses,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            FIRST_NUMEL=7168,
            TOTAL_NUMEL=10752,
            BLOCK_SIZE=512,
            FLAG_BASE=self._max_blocks + 32,
            NUM_WARPS=1,
            num_warps=1,
        )
        return first, second

    def all_reduce_residual_attnres(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        score_weight: torch.Tensor,
        output_weight: torch.Tensor,
        scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float,
        op=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce a K3 attention partial and finish its AttnRes prefix mix.

        This exact-shape decode primitive combines Iris one-shot all-reduce,
        the BF16 residual accumulation, the split AttnRes prefix candidate,
        and the following RMSNorm.  It returns both values needed by the
        decoder: the new residual prefix and the normalized MoE input.

        Args:
            partial: Per-rank attention output shaped ``[1, 7168]`` in BF16.
            residual: Incoming residual prefix with the same shape and dtype.
            score_weight: Precomputed AttnRes ``rms_weight * projection_weight``.
            output_weight: Following RMSNorm weight shaped ``[7168]``.
            scratch: ``(m, s, acc)`` block-side online-softmax partial, where
                ``m`` and ``s`` are FP32 ``[1]`` and ``acc`` is FP32
                ``[1, 7168]``.
            eps: Shared K3 AttnRes/RMSNorm epsilon.
            op: Reduction operation. Only ``SUM`` is supported.

        Returns:
            ``(normalized_hidden, residual_out)``, both BF16 ``[1, 7168]``.
        """
        if op is None:
            op = dist.ReduceOp.SUM
        assert op == dist.ReduceOp.SUM, f"Iris all-reduce only supports SUM, got {op}"
        assert _platform.is_cdna4 and self.world_size == 8
        assert partial.shape == residual.shape == (1, 7168)
        assert partial.dtype == residual.dtype == self.dtype == torch.bfloat16
        assert partial.device == residual.device == self.device
        assert partial.is_contiguous() and residual.is_contiguous()
        assert score_weight.shape == output_weight.shape == (7168,)
        assert score_weight.dtype == output_weight.dtype == torch.bfloat16
        assert score_weight.device == output_weight.device == self.device
        assert score_weight.is_contiguous() and output_weight.is_contiguous()
        m, s_, acc = scratch
        assert m.shape == s_.shape == (1,)
        assert acc.shape == (1, 7168)
        assert m.dtype == s_.dtype == acc.dtype == torch.float32
        assert m.device == s_.device == acc.device == self.device
        assert m.is_contiguous() and s_.is_contiguous() and acc.is_contiguous()
        assert eps > 0.0

        hidden = torch.empty_like(partial)
        residual_out = torch.empty_like(residual)
        iris_stage_one_shot_allreduce_residual_attnres_gluon_kernel[(1,)](
            partial,
            residual,
            self._input_buf,
            score_weight,
            output_weight,
            m,
            s_,
            acc,
            hidden,
            residual_out,
            self._ready_flags,
            *self._heap_base_addresses,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            HIDDEN=7168,
            BLOCK=8192,
            EPS=eps,
            ELEMENTS_PER_THREAD=1,
            NUM_WARPS=4,
            num_warps=4,
        )
        return hidden, residual_out


@triton.jit
def iris_stage_one_shot_allreduce_kernel(
    input_ptr,
    input_sym_ptr,
    output_ptr,
    ready_flags,
    heap_bases,
    NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    local = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(input_sym_ptr + offsets, local, mask=mask, cache_modifier=".wt")
    tl.debug_barrier()

    flag_offset = block_id * WORLD_SIZE
    local_ready = ready_flags + flag_offset + RANK
    epoch = tl.load(local_ready).to(tl.int32) + 1
    tl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")

    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            seen = tl.full((), 0, dtype=tl.int32)
            while seen < epoch:
                seen = iris.load(
                    ready_flags + flag_offset + peer,
                    RANK,
                    peer,
                    heap_bases,
                    cache_modifier=".cv",
                    volatile=True,
                )

    acc = local.to(tl.float32)
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            acc += iris.load(
                input_sym_ptr + offsets,
                RANK,
                peer,
                heap_bases,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
                hint=BLOCK_SIZE,
            ).to(tl.float32)
    tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)


@gluon.jit
def _iris_heap_base(
    rank: gl.constexpr,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
):
    if rank == 0:
        return heap_base_0
    if rank == 1:
        return heap_base_1
    if rank == 2:
        return heap_base_2
    if rank == 3:
        return heap_base_3
    if rank == 4:
        return heap_base_4
    if rank == 5:
        return heap_base_5
    if rank == 6:
        return heap_base_6
    return heap_base_7


@gluon.jit
def _iris_wait_for_peers(
    ready_flags,
    block_id,
    epoch,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
):
    """Poll all remote ready flags together with coherent global loads."""
    ready_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    peer_ids = gl.arange(0, WORLD_SIZE, layout=ready_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)

    flags_heap_offset = tl.cast(ready_flags, gl.uint64) - local_heap
    peer_flags = tl.cast(
        peer_heaps + flags_heap_offset,
        gl.pointer_type(gl.int32),
    )
    peer_flags += block_id * WORLD_SIZE + peer_ids
    peer_mask = peer_ids != RANK
    seen = gl.full([WORLD_SIZE], 0, gl.int32, layout=ready_layout)
    while gl.min(gl.where(peer_mask, seen, epoch), axis=0) < epoch:
        seen = gl.load(
            peer_flags,
            mask=peer_mask,
            other=epoch,
            cache_modifier=".cv",
            volatile=True,
        )


@gluon.jit
def _iris_wait_for_local_and_peers(
    ready_flags,
    flag_id: gl.constexpr,
    epoch,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
):
    local_ready = ready_flags + flag_id * WORLD_SIZE + RANK
    seen = gl.full((), 0, dtype=gl.int32)
    while seen < epoch:
        seen = gl.load(
            local_ready,
            cache_modifier=".cv",
            volatile=True,
        )
    _iris_wait_for_peers(
        ready_flags,
        flag_id,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
    )


@gluon.jit
def _iris_wait_for_rank_epoch(
    ready_flags,
    block_id,
    epoch,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Poll rank epochs with a layout matching the specialized launch."""
    ready_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peer_ids = gl.arange(0, WORLD_SIZE, layout=ready_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    flags_heap_offset = tl.cast(ready_flags, gl.uint64) - local_heap
    peer_flags = tl.cast(
        peer_heaps + flags_heap_offset,
        gl.pointer_type(gl.int32),
    )
    peer_flags += block_id * WORLD_SIZE + peer_ids
    peer_mask = peer_ids != RANK
    seen = gl.full([WORLD_SIZE], 0, gl.int32, layout=ready_layout)
    while gl.min(gl.where(peer_mask, seen, epoch), axis=0) < epoch:
        seen = gl.load(
            peer_flags,
            mask=peer_mask,
            other=epoch,
            cache_modifier=".cv",
            volatile=True,
        )


@gluon.jit
def _unpack_bf16x4(packed):
    value_0 = ((packed & 0xFFFF).to(gl.uint32) << 16).to(gl.float32, bitcast=True)
    value_1 = (((packed >> 16) & 0xFFFF).to(gl.uint32) << 16).to(
        gl.float32, bitcast=True
    )
    value_2 = (((packed >> 32) & 0xFFFF).to(gl.uint32) << 16).to(
        gl.float32, bitcast=True
    )
    value_3 = (((packed >> 48) & 0xFFFF).to(gl.uint32) << 16).to(
        gl.float32, bitcast=True
    )
    return value_0, value_1, value_2, value_3


@gluon.jit
def _pack_bf16x4(value_0, value_1, value_2, value_3):
    bits_0 = value_0.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_1 = value_1.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_2 = value_2.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_3 = value_3.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint64)
    return bits_0 | (bits_1 << 16) | (bits_2 << 32) | (bits_3 << 48)


@gluon.jit
def iris_stage_one_shot_allreduce_two_gluon_kernel(
    first_input_ptr,
    second_input_ptr,
    input_sym_ptr,
    first_output_ptr,
    second_output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    FIRST_NUMEL,
    TOTAL_NUMEL,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    """Stage and reduce K3's shared and routed BF16 segments with buffer ops."""
    block_id = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [4], [0])
    packed_offset = block_id * (BLOCK_SIZE // 4) + gl.arange(
        0, BLOCK_SIZE // 4, layout=layout
    )
    first_packed = FIRST_NUMEL // 4
    total_packed = TOTAL_NUMEL // 4
    first_mask = packed_offset < first_packed
    second_offset = packed_offset - first_packed
    second_mask = (packed_offset >= first_packed) & (packed_offset < total_packed)

    first_input = gl.amd.cdna4.buffer_load(
        tl.cast(first_input_ptr, gl.pointer_type(gl.uint64)),
        packed_offset.to(gl.int32),
        mask=first_mask,
        other=0,
    )
    second_input = gl.amd.cdna4.buffer_load(
        tl.cast(second_input_ptr, gl.pointer_type(gl.uint64)),
        second_offset.to(gl.int32),
        mask=second_mask,
        other=0,
    )
    local_packed = first_input | second_input
    gl.amd.cdna4.buffer_store(
        local_packed,
        tl.cast(input_sym_ptr, gl.pointer_type(gl.uint64)),
        packed_offset.to(gl.int32),
        mask=packed_offset < total_packed,
        cache=".wt",
    )
    gl.barrier()

    flag_offset = block_id * WORLD_SIZE
    local_ready = ready_flags + flag_offset + RANK
    epoch = gl.load(local_ready).to(gl.int32) + 1
    gl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")

    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    _iris_wait_for_peers(
        ready_flags,
        block_id,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
    )

    acc_0, acc_1, acc_2, acc_3 = _unpack_bf16x4(local_packed)
    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            peer_heap = _iris_heap_base(
                peer,
                heap_base_0,
                heap_base_1,
                heap_base_2,
                heap_base_3,
                heap_base_4,
                heap_base_5,
                heap_base_6,
                heap_base_7,
            )
            peer_input = tl.cast(
                peer_heap + input_heap_offset,
                gl.pointer_type(gl.uint64),
            )
            peer_packed = gl.amd.cdna4.buffer_load(
                peer_input,
                packed_offset.to(gl.int32),
                mask=packed_offset < total_packed,
                other=0,
                cache=".cg",
            )
            peer_0, peer_1, peer_2, peer_3 = _unpack_bf16x4(peer_packed)
            acc_0 += peer_0
            acc_1 += peer_1
            acc_2 += peer_2
            acc_3 += peer_3

    packed_output = _pack_bf16x4(acc_0, acc_1, acc_2, acc_3)
    gl.amd.cdna4.buffer_store(
        packed_output,
        tl.cast(first_output_ptr, gl.pointer_type(gl.uint64)),
        packed_offset.to(gl.int32),
        mask=first_mask,
    )
    gl.amd.cdna4.buffer_store(
        packed_output,
        tl.cast(second_output_ptr, gl.pointer_type(gl.uint64)),
        second_offset.to(gl.int32),
        mask=second_mask,
    )


@gluon.jit
def iris_reduce_symmetric_two_gluon_kernel(
    input_sym_ptr,
    first_output_ptr,
    second_output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    FIRST_NUMEL: gl.constexpr,
    TOTAL_NUMEL: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    FLAG_BASE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Reduce K3 outputs which the producer placed in symmetric memory."""
    block_id = gl.program_id(0)
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    epoch_ptr = ready_flags + (FLAG_BASE + block_id) * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_wait_for_rank_epoch(
        ready_flags,
        FLAG_BASE + block_id,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
    )

    layout: gl.constexpr = gl.BlockedLayout([2], [64], [NUM_WARPS], [0])
    packed_offset = block_id * (BLOCK_SIZE // 4) + gl.arange(
        0, BLOCK_SIZE // 4, layout=layout
    )
    total_packed: gl.constexpr = TOTAL_NUMEL // 4
    mask = packed_offset < total_packed
    local_packed = gl.amd.cdna4.buffer_load(
        tl.cast(input_sym_ptr, gl.pointer_type(gl.uint64)),
        packed_offset.to(gl.int32),
        mask=mask,
        other=0,
    )
    acc_0, acc_1, acc_2, acc_3 = _unpack_bf16x4(local_packed)
    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            peer_heap = _iris_heap_base(
                peer,
                heap_base_0,
                heap_base_1,
                heap_base_2,
                heap_base_3,
                heap_base_4,
                heap_base_5,
                heap_base_6,
                heap_base_7,
            )
            peer_input = tl.cast(
                peer_heap + input_heap_offset, gl.pointer_type(gl.uint64)
            )
            peer_packed = gl.amd.cdna4.buffer_load(
                peer_input,
                packed_offset.to(gl.int32),
                mask=mask,
                other=0,
                cache=".cg",
            )
            peer_0, peer_1, peer_2, peer_3 = _unpack_bf16x4(peer_packed)
            acc_0 += peer_0
            acc_1 += peer_1
            acc_2 += peer_2
            acc_3 += peer_3

    packed_output = _pack_bf16x4(acc_0, acc_1, acc_2, acc_3)
    first_packed: gl.constexpr = FIRST_NUMEL // 4
    gl.amd.cdna4.buffer_store(
        packed_output,
        tl.cast(first_output_ptr, gl.pointer_type(gl.uint64)),
        packed_offset.to(gl.int32),
        mask=packed_offset < first_packed,
    )
    gl.amd.cdna4.buffer_store(
        packed_output,
        tl.cast(second_output_ptr, gl.pointer_type(gl.uint64)),
        (packed_offset - first_packed).to(gl.int32),
        mask=(packed_offset >= first_packed) & mask,
    )


@gluon.jit
def iris_stage_one_shot_allreduce_two_rank_epoch_gluon_kernel(
    first_input_ptr,
    second_input_ptr,
    input_sym_ptr,
    first_output_ptr,
    second_output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    FIRST_NUMEL: gl.constexpr,
    TOTAL_NUMEL: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    FLAG_BASE: gl.constexpr,
    NUM_BLOCKS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Stage K3 tiles, then synchronize once per rank before reduction."""
    block_id = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [64], [NUM_WARPS], [0]
    )
    packed_offset = block_id * (BLOCK_SIZE // 4) + gl.arange(
        0, BLOCK_SIZE // 4, layout=layout
    )
    first_packed: gl.constexpr = FIRST_NUMEL // 4
    total_packed: gl.constexpr = TOTAL_NUMEL // 4
    first_mask = packed_offset < first_packed
    second_offset = packed_offset - first_packed
    second_mask = (packed_offset >= first_packed) & (packed_offset < total_packed)

    first_input = gl.amd.cdna4.buffer_load(
        tl.cast(first_input_ptr, gl.pointer_type(gl.uint64)),
        packed_offset.to(gl.int32),
        mask=first_mask,
        other=0,
    )
    second_input = gl.amd.cdna4.buffer_load(
        tl.cast(second_input_ptr, gl.pointer_type(gl.uint64)),
        second_offset.to(gl.int32),
        mask=second_mask,
        other=0,
    )
    local_packed = first_input | second_input
    gl.amd.cdna4.buffer_store(
        local_packed,
        tl.cast(input_sym_ptr, gl.pointer_type(gl.uint64)),
        packed_offset.to(gl.int32),
        mask=packed_offset < total_packed,
        cache=".wt",
    )
    gl.barrier()

    seen_ptr = ready_flags + (FLAG_BASE + 3 + block_id) * WORLD_SIZE + RANK
    previous_epoch = gl.load(seen_ptr)
    arrival_ptr = ready_flags + FLAG_BASE * WORLD_SIZE + RANK
    old = gl.atomic_add(arrival_ptr, 1, sem="acq_rel", scope="gpu")
    if old == NUM_BLOCKS - 1:
        gl.atomic_xchg(arrival_ptr, 0, sem="relaxed", scope="gpu")
        epoch_ptr = ready_flags + (FLAG_BASE + 1) * WORLD_SIZE + RANK
        gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys")

    epoch_ptr = ready_flags + (FLAG_BASE + 1) * WORLD_SIZE + RANK
    epoch = gl.full((), 0, gl.int32)
    while epoch <= previous_epoch:
        epoch = gl.load(epoch_ptr, cache_modifier=".cv", volatile=True)
    gl.store(seen_ptr, epoch)

    coordinator_ptr = ready_flags + (FLAG_BASE + 2) * WORLD_SIZE + RANK
    if block_id == 0:
        local_heap = _iris_heap_base(
            RANK,
            heap_base_0,
            heap_base_1,
            heap_base_2,
            heap_base_3,
            heap_base_4,
            heap_base_5,
            heap_base_6,
            heap_base_7,
        )
        _iris_wait_for_rank_epoch(
            ready_flags,
            FLAG_BASE + 1,
            epoch,
            local_heap,
            heap_base_0,
            heap_base_1,
            heap_base_2,
            heap_base_3,
            heap_base_4,
            heap_base_5,
            heap_base_6,
            heap_base_7,
            RANK,
            WORLD_SIZE,
            NUM_WARPS,
        )
        gl.atomic_xchg(
            coordinator_ptr,
            epoch,
            sem="release",
            scope="gpu",
        )
    coordinator_epoch = gl.full((), 0, gl.int32)
    while coordinator_epoch < epoch:
        coordinator_epoch = gl.load(
            coordinator_ptr,
            cache_modifier=".cv",
            volatile=True,
        )

    acc_0, acc_1, acc_2, acc_3 = _unpack_bf16x4(local_packed)
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            peer_heap = _iris_heap_base(
                peer,
                heap_base_0,
                heap_base_1,
                heap_base_2,
                heap_base_3,
                heap_base_4,
                heap_base_5,
                heap_base_6,
                heap_base_7,
            )
            peer_input = tl.cast(
                peer_heap + input_heap_offset,
                gl.pointer_type(gl.uint64),
            )
            peer_packed = gl.amd.cdna4.buffer_load(
                peer_input,
                packed_offset.to(gl.int32),
                mask=packed_offset < total_packed,
                other=0,
                cache=".cg",
            )
            peer_0, peer_1, peer_2, peer_3 = _unpack_bf16x4(peer_packed)
            acc_0 += peer_0
            acc_1 += peer_1
            acc_2 += peer_2
            acc_3 += peer_3

    packed_output = _pack_bf16x4(acc_0, acc_1, acc_2, acc_3)
    gl.amd.cdna4.buffer_store(
        packed_output,
        tl.cast(first_output_ptr, gl.pointer_type(gl.uint64)),
        packed_offset.to(gl.int32),
        mask=first_mask,
    )
    gl.amd.cdna4.buffer_store(
        packed_output,
        tl.cast(second_output_ptr, gl.pointer_type(gl.uint64)),
        second_offset.to(gl.int32),
        mask=second_mask,
    )


@gluon.jit
def iris_stage_one_shot_allreduce_residual_attnres_gluon_kernel(
    partial_ptr,
    residual_ptr,
    input_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """K3 attention AR + residual + split AttnRes combine in one CTA.

    The residual accumulation and mixed AttnRes row are explicitly rounded to
    BF16 at the same boundaries as the unfused graph.  A single CTA owns the
    whole row because both the prefix score and output RMSNorm require
    reductions across all 7168 elements.
    """
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [64], [NUM_WARPS], [0]
    )
    offset = gl.arange(0, BLOCK, layout=layout)
    mask = offset < HIDDEN
    offset_i32 = offset.to(gl.int32)

    local = gl.amd.cdna4.buffer_load(
        partial_ptr,
        offset_i32,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        local.to(input_sym_ptr.dtype.element_ty),
        input_sym_ptr,
        offset_i32,
        mask=mask,
        cache=".wt",
    )
    gl.barrier()

    # This specialization uses the ordinary flag-0 epoch. All participating
    # ranks execute the same collective sequence, so it safely interleaves
    # with the tiled one-shot and two-segment Iris calls.
    local_ready = ready_flags + RANK
    epoch = gl.load(local_ready).to(gl.int32) + 1
    gl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")

    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    _iris_wait_for_peers(
        ready_flags,
        0,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    reduced = local
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            peer_heap = _iris_heap_base(
                peer,
                heap_base_0,
                heap_base_1,
                heap_base_2,
                heap_base_3,
                heap_base_4,
                heap_base_5,
                heap_base_6,
                heap_base_7,
            )
            peer_input = tl.cast(
                peer_heap + input_heap_offset,
                partial_ptr.dtype,
            )
            reduced += gl.amd.cdna4.buffer_load(
                peer_input,
                offset_i32,
                mask=mask,
                other=0.0,
                cache=".cg",
            ).to(gl.float32)

    # Preserve the all-reduce output and residual-add BF16 boundaries.
    reduced = reduced.to(gl.bfloat16).to(gl.float32)
    residual = gl.amd.cdna4.buffer_load(
        residual_ptr,
        offset_i32,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    prefix = (reduced + residual).to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        prefix.to(residual_out_ptr.dtype.element_ty),
        residual_out_ptr,
        offset_i32,
        mask=mask,
    )

    score_weight = gl.amd.cdna4.buffer_load(
        score_weight_ptr,
        offset_i32,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    square_sum = gl.sum(gl.where(mask, prefix * prefix, 0.0), axis=0)
    dot = gl.sum(gl.where(mask, prefix * score_weight, 0.0), axis=0)
    prefix_logit = dot * gl.rsqrt(square_sum / HIDDEN + EPS)

    block_m = gl.load(scratch_m_ptr)
    block_s = gl.load(scratch_s_ptr)
    maximum = gl.maximum(block_m, prefix_logit)
    block_correction = gl.exp(block_m - maximum)
    prefix_weight = gl.exp(prefix_logit - maximum)
    inverse_sum = 1.0 / (block_s * block_correction + prefix_weight)
    block_acc = gl.amd.cdna4.buffer_load(
        scratch_acc_ptr,
        offset_i32,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    mixed = (
        ((block_acc * block_correction + prefix_weight * prefix) * inverse_sum)
        .to(gl.bfloat16)
        .to(gl.float32)
    )

    output_square_sum = gl.sum(gl.where(mask, mixed * mixed, 0.0), axis=0)
    inverse_rms = gl.rsqrt(output_square_sum / HIDDEN + EPS)
    output_weight = gl.amd.cdna4.buffer_load(
        output_weight_ptr,
        offset_i32,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        (mixed * inverse_rms * output_weight).to(hidden_ptr.dtype.element_ty),
        hidden_ptr,
        offset_i32,
        mask=mask,
    )


@triton.jit
def iris_stage_one_shot_allreduce_two_kernel(
    first_input_ptr,
    second_input_ptr,
    input_sym_ptr,
    first_output_ptr,
    second_output_ptr,
    ready_flags,
    heap_bases,
    FIRST_NUMEL,
    TOTAL_NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    combined_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    combined_mask = combined_offsets < TOTAL_NUMEL
    first_mask = combined_offsets < FIRST_NUMEL
    second_offsets = combined_offsets - FIRST_NUMEL
    second_mask = (combined_offsets >= FIRST_NUMEL) & combined_mask

    # The masks are disjoint, so their sum forms the logical concatenation in
    # symmetric memory without allocating or copying a temporary torch.Tensor.
    local = tl.load(
        first_input_ptr + combined_offsets,
        mask=first_mask,
        other=0.0,
    )
    local += tl.load(
        second_input_ptr + second_offsets,
        mask=second_mask,
        other=0.0,
    )
    tl.store(
        input_sym_ptr + combined_offsets,
        local,
        mask=combined_mask,
        cache_modifier=".wt",
    )
    tl.debug_barrier()

    flag_offset = block_id * WORLD_SIZE
    local_ready = ready_flags + flag_offset + RANK
    epoch = tl.load(local_ready).to(tl.int32) + 1
    tl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")

    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            seen = tl.full((), 0, dtype=tl.int32)
            while seen < epoch:
                seen = iris.load(
                    ready_flags + flag_offset + peer,
                    RANK,
                    peer,
                    heap_bases,
                    cache_modifier=".cv",
                    volatile=True,
                )

    acc = local.to(tl.float32)
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            acc += iris.load(
                input_sym_ptr + combined_offsets,
                RANK,
                peer,
                heap_bases,
                mask=combined_mask,
                other=0.0,
                cache_modifier=".cg",
                hint=BLOCK_SIZE,
            ).to(tl.float32)

    acc = acc.to(first_output_ptr.type.element_ty)
    tl.store(first_output_ptr + combined_offsets, acc, mask=first_mask)
    tl.store(second_output_ptr + second_offsets, acc, mask=second_mask)


@triton.jit
def iris_allreduce_kernel(
    input_sym_ptr,
    output_ptr,
    NUMEL,
    heap_bases,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.static_range(0, world_size):
        remote_rank = rank_start + i * rank_stride
        acc += iris.load(
            input_sym_ptr + offsets,
            iris_rank,
            remote_rank,
            heap_bases,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    out_dtype = output_ptr.type.element_ty
    tl.store(output_ptr + offsets, acc.to(out_dtype), mask=mask)


@triton.jit
def iris_allreduce_residual_rmsnorm_kernel(
    input_sym_ptr,  # base of symmetric (M, HIDDEN_SIZE) input buffer
    residual_ptr,  # local (M, HIDDEN_SIZE)
    weight_ptr,  # local (HIDDEN_SIZE,)
    norm_out_ptr,  # local (M, HIDDEN_SIZE)
    residual_out_ptr,  # local (M, HIDDEN_SIZE)
    M,
    heap_bases,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_SIZE
    row_offsets = row * HIDDEN_SIZE + offsets
    in_row_ptr = input_sym_ptr + row_offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.static_range(0, world_size):
        remote_rank = rank_start + i * rank_stride
        acc += iris.load(
            in_row_ptr,
            iris_rank,
            remote_rank,
            heap_bases,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    residual_out = acc + residual

    res_out_dtype = residual_out_ptr.type.element_ty
    tl.store(
        residual_out_ptr + row_offsets,
        residual_out.to(res_out_dtype),
        mask=mask,
    )

    variance = tl.sum(residual_out * residual_out, axis=0) / HIDDEN_SIZE
    scale = tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm = residual_out * scale * weight

    norm_dtype = norm_out_ptr.type.element_ty
    tl.store(
        norm_out_ptr + row_offsets,
        norm.to(norm_dtype),
        mask=mask,
    )


@triton.jit
def iris_allreduce_residual_rmsnorm_kernel_persistent(
    input_sym_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    residual_out_ptr,
    M,
    heap_bases,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_SIZE
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    res_out_dtype = residual_out_ptr.type.element_ty
    norm_dtype = norm_out_ptr.type.element_ty

    for row in range(pid, M, num_programs):
        row_offsets = row * HIDDEN_SIZE + offsets
        in_row_ptr = input_sym_ptr + row_offsets

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in tl.static_range(0, world_size):
            remote_rank = rank_start + i * rank_stride
            acc += iris.load(
                in_row_ptr,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        residual_out = acc + residual

        tl.store(
            residual_out_ptr + row_offsets,
            residual_out.to(res_out_dtype),
            mask=mask,
        )

        variance = tl.sum(residual_out * residual_out, axis=0) / HIDDEN_SIZE
        scale = tl.rsqrt(variance + EPS)
        norm = residual_out * scale * weight

        tl.store(
            norm_out_ptr + row_offsets,
            norm.to(norm_dtype),
            mask=mask,
        )


class IrisAllReduceResidualRMSNorm(object):
    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_in_group: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        heap_size: int | None = None,
        device: torch.device = None,
        persistent: bool = False,
    ) -> None:
        assert (
            type(group) == dist.ProcessGroup
        ), f"Expected dist.ProcessGroup, got {type(group)}"
        assert dist.is_initialized(), (
            "torch.distributed must be initialized before constructing "
            "IrisAllReduceResidualRMSNorm; call dist.init_process_group() first."
        )
        assert _platform.is_amd, (
            "IrisAllReduceResidualRMSNorm currently targets AMD ROCm; "
            f"got non-AMD platform: {_platform}"
        )

        self.group = group
        self.rank_in_group = rank_in_group
        self.world_size = group.size()
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

        if heap_size is None:
            buf_bytes = max_token_num * hidden_dim * dtype.itemsize
            heap_size = max(1 << 28, 4 * buf_bytes + (16 << 20))
        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        self._ctx = _get_or_create_iris_context(heap_size)
        self._input_buf = self._ctx.zeros((max_token_num, hidden_dim), dtype=dtype)
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris AR+RMSNorm symmetric-heap buffer allocated: %s GB",
            free_gpu_memory_begin - free_gpu_memory_after,
        )

        self._rank_start = 0
        self._rank_stride = 1
        self._iris_rank = dist.get_rank()

        self.persistent = persistent
        self._num_programs = (
            torch.cuda.get_device_properties(self.device).multi_processor_count
            if persistent
            else 0
        )

    def fused(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        norm_out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert input_tensor.dtype == self.dtype, (
            f"Iris AR+RMSNorm dtype mismatch: input={input_tensor.dtype}, "
            f"backend={self.dtype}"
        )
        assert input_tensor.dim() == 2, (
            f"input must be 2-D (num_tokens, hidden_dim), got "
            f"shape={input_tensor.shape}"
        )
        assert (
            input_tensor.shape == residual.shape
        ), f"residual shape {residual.shape} != input shape {input_tensor.shape}"
        assert input_tensor.shape[1] == self.hidden_dim, (
            f"hidden_dim mismatch: input={input_tensor.shape[1]} vs "
            f"backend={self.hidden_dim}"
        )
        num_tokens = input_tensor.shape[0]
        assert (
            num_tokens <= self.max_token_num
        ), f"num_tokens ({num_tokens}) exceeds max_token_num ({self.max_token_num})"
        assert weight.shape == (
            self.hidden_dim,
        ), f"weight shape {weight.shape} != ({self.hidden_dim},)"
        assert input_tensor.is_contiguous() and residual.is_contiguous()

        in_view = self._input_buf[:num_tokens, :]
        in_view.copy_(input_tensor)

        if norm_out is None:
            norm_out = torch.empty_like(input_tensor)
        if residual_out is None:
            residual_out = torch.empty_like(residual)

        self._ctx.device_barrier()

        heap_bases = self._ctx.get_heap_bases()
        BLOCK_SIZE = triton.next_power_of_2(self.hidden_dim)
        if self.persistent:
            kernel = iris_allreduce_residual_rmsnorm_kernel_persistent
            grid = (min(num_tokens, self._num_programs),)
        else:
            kernel = iris_allreduce_residual_rmsnorm_kernel
            grid = (num_tokens,)
        kernel[grid](
            in_view,
            residual,
            weight,
            norm_out,
            residual_out,
            num_tokens,
            heap_bases,
            iris_rank=self._iris_rank,
            world_size=self.world_size,
            rank_start=self._rank_start,
            rank_stride=self._rank_stride,
            HIDDEN_SIZE=self.hidden_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            EPS=eps,
            num_warps=8,
        )
        # Ensure all peer loads finish before the next call reuses _input_buf.
        self._ctx.device_barrier()
        return norm_out, residual_out


def create_iris_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_numel: int,
    dtype: torch.dtype = torch.bfloat16,
    heap_size: int | None = None,
    device: torch.device = None,
) -> "IrisAllReduce":
    return IrisAllReduce(
        group=group,
        rank_in_group=rank_in_group,
        max_numel=max_numel,
        dtype=dtype,
        heap_size=heap_size,
        device=device,
    )


def iris_all_reduce(
    state: "IrisAllReduce",
    tensor: torch.Tensor,
    op=None,
    safe: bool = True,
    async_op: bool = False,
) -> torch.Tensor:
    return state.all_reduce(tensor, op=op, safe=safe, async_op=async_op)


def iris_all_reduce_two(
    state: "IrisAllReduce",
    first: torch.Tensor,
    second: torch.Tensor,
    op=None,
    safe: bool = True,
    async_op: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce two tensors through one Iris kernel launch."""
    return state.all_reduce_two(
        first,
        second,
        op=op,
        safe=safe,
        async_op=async_op,
    )


def iris_all_reduce_two_staging(
    state: "IrisAllReduce",
    first_shape: tuple[int, ...],
    second_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return consecutive symmetric producer-output views for Iris."""
    return state.staging_two(first_shape, second_shape)


def iris_all_reduce_residual_attnres(
    state: "IrisAllReduce",
    partial: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
    op=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finish the exact K3 attention reduction and AttnRes prefix mix."""
    return state.all_reduce_residual_attnres(
        partial,
        residual,
        score_weight,
        output_weight,
        scratch,
        eps,
        op=op,
    )


def create_iris_rsag_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device = None,
    heap_size: int | None = None,
) -> "IrisRSAG":
    return IrisRSAG(
        group=group,
        rank_in_group=rank_in_group,
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        device=device,
        heap_size=heap_size,
    )


def create_iris_ar_rmsnorm_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    heap_size: int | None = None,
    device: torch.device = None,
    persistent: bool = False,
) -> "IrisAllReduceResidualRMSNorm":
    return IrisAllReduceResidualRMSNorm(
        group=group,
        rank_in_group=rank_in_group,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        heap_size=heap_size,
        device=device,
        persistent=persistent,
    )


def iris_allreduce_residual_rmsnorm(
    state: "IrisAllReduceResidualRMSNorm",
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    norm_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return state.fused(
        input_tensor=input_tensor,
        residual=residual,
        weight=weight,
        eps=eps,
        norm_out=norm_out,
        residual_out=residual_out,
    )
