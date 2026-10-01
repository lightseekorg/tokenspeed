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

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import gl, triton
from tokenspeed_kernel.ops.communication._iris import (
    _iris_all_gather,
    _iris_reduce_scatter,
    _IrisConfig,
    iris,
)
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    iris_allreduce_residual_rmsnorm_kernel,
    iris_allreduce_residual_rmsnorm_kernel_persistent,
    iris_reduce_symmetric_gluon_kernel,
    iris_reduce_symmetric_two_stage_gluon_kernel,
    iris_stage_one_shot_allreduce_kernel,
    lamport_all_reduce_bf16,
)
from tokenspeed_kernel.ops.communication._iris.attnres import (
    ATTNRES_KERNEL_CONFIG,
    ATTNRES_MAX_ROWS,
    IrisAttnResWorkspace,
    attnres_supported,
)
from tokenspeed_kernel.ops.communication._iris.row_sharded import (
    REDUCE_PROGRAMS,
    IrisRowShardedWorkspace,
)
from tokenspeed_kernel.platform import current_platform

logger = logging.getLogger(__file__)

_platform = current_platform()

__all__ = [
    "IRIS_ALL_REDUCE_KERNEL_CONFIG",
    "IrisAllReduce",
    "IrisAllReduceKernelConfig",
    "KimiK3MoeAllReduceKernelConfig",
    "IrisRSAG",
    "IrisAllReduceResidualRMSNorm",
    "create_iris_state",
    "iris_all_reduce",
    "find_iris_state",
    "get_or_create_iris_state",
    "iris_acquire_outputs",
    "iris_all_reduce_symmetric",
    "iris_all_reduce_residual_attnres",
    "producer_direct_all_reduce_can_run",
    "create_iris_rsag_state",
    "create_iris_ar_rmsnorm_state",
    "iris_allreduce_residual_rmsnorm",
    "iris_kimi3_moe_tail",
    "iris_attention_mix",
    "IRIS_AR_STATES",
    "IRIS_AR_RMSNORM_STATES",
]


IRIS_AR_STATES: dict = {}
IRIS_AR_RMSNORM_STATES: dict = {}
_PRODUCER_DIRECT_GL_DTYPES = {
    torch.bfloat16: gl.bfloat16,
    torch.float16: gl.float16,
    torch.float32: gl.float32,
}


@dataclass(frozen=True)
class _StagedAllReduceKernelTuning:
    world_size: int
    dtype: torch.dtype
    numel: int
    block_size: int
    num_subgroups: int

    def __post_init__(self) -> None:
        if (
            self.world_size <= 1
            or self.numel <= 0
            or self.block_size <= 0
            or self.num_subgroups <= 0
        ):
            raise ValueError("invalid staged Iris kernel tuning")

    def num_programs(self) -> int:
        return triton.cdiv(self.numel, self.block_size)


@dataclass(frozen=True)
class _StagedAllReduceKernelConfig:
    block_size: int
    num_subgroups: int
    input_slots: int
    cdna4_tunings: tuple[_StagedAllReduceKernelTuning, ...]

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.num_subgroups <= 0 or self.input_slots < 2:
            raise ValueError("invalid staged Iris kernel configuration")

    def max_programs(self, max_numel: int) -> int:
        return triton.cdiv(max_numel, self.block_size)

    def tuning(
        self,
        numel: int,
        world_size: int,
        dtype: torch.dtype,
        is_cdna4: bool,
    ) -> _StagedAllReduceKernelTuning | None:
        if not is_cdna4:
            return None
        for tuning in self.cdna4_tunings:
            if (
                tuning.world_size == world_size
                and tuning.dtype == dtype
                and tuning.numel == numel
            ):
                return tuning
        return None


@dataclass(frozen=True)
class _ProducerDirectAllReduceKernelConfig:
    supported_world_sizes: tuple[int, ...]
    one_stage_block_size: int
    one_stage_max_programs: int
    one_stage_num_subgroups: int
    one_stage_words_per_lane: int
    two_stage_min_bytes: tuple[tuple[int, int], ...]
    publish_ready: bool

    def __post_init__(self) -> None:
        if (
            not self.supported_world_sizes
            or len(set(self.supported_world_sizes)) != len(self.supported_world_sizes)
            or any(world_size <= 1 for world_size in self.supported_world_sizes)
            or self.one_stage_block_size <= 0
            or self.one_stage_max_programs <= 0
            or self.one_stage_num_subgroups <= 0
            or self.one_stage_words_per_lane <= 0
        ):
            raise ValueError("invalid producer-direct Iris kernel configuration")
        threshold_world_sizes = tuple(
            world_size for world_size, _ in self.two_stage_min_bytes
        )
        if len(set(threshold_world_sizes)) != len(threshold_world_sizes) or any(
            world_size not in self.supported_world_sizes or min_bytes <= 0
            for world_size, min_bytes in self.two_stage_min_bytes
        ):
            raise ValueError("invalid producer-direct Iris two-stage thresholds")

    def supports_world_size(self, world_size: int) -> bool:
        return world_size in self.supported_world_sizes

    def two_stage_threshold(self, world_size: int) -> int | None:
        return dict(self.two_stage_min_bytes).get(world_size)


@dataclass(frozen=True)
class _TwoStageAllReduceKernelConfig:
    supported_world_sizes: tuple[int, ...]
    max_programs: int
    num_subgroups: int
    words_per_lane: int

    def __post_init__(self) -> None:
        if (
            not self.supported_world_sizes
            or len(set(self.supported_world_sizes)) != len(self.supported_world_sizes)
            or self.num_subgroups <= 0
            or any(
                world_size <= 1 or self.num_subgroups % world_size != 0
                for world_size in self.supported_world_sizes
            )
            or self.max_programs <= 0
            or self.words_per_lane <= 0
        ):
            raise ValueError("invalid two-stage Iris kernel configuration")

    def supports_world_size(self, world_size: int) -> bool:
        return world_size in self.supported_world_sizes

    def can_partition(
        self,
        world_size: int,
        total_numel: int,
        elements_per_word: int,
    ) -> bool:
        return (
            self.supports_world_size(world_size)
            and total_numel % (world_size * elements_per_word) == 0
        )

    def scratch_numel(self, max_numel: int, world_size: int) -> int:
        if max_numel == 0 or not self.supports_world_size(world_size):
            return 0
        return triton.cdiv(max_numel, world_size)

    def block_words(self, world_size: int, subgroup_size: int) -> int:
        assert self.supports_world_size(world_size)
        return self.num_subgroups * subgroup_size * self.words_per_lane // world_size


@dataclass(frozen=True)
class KimiK3MoeAllReduceKernelConfig:
    """Shape and mailbox contract for the CDNA4 K3 BF16 Lamport all-reduce.

    Attributes:
        world_size: Required communication group size.
        routed_hidden_size: Width of the routed-expert output.
        hidden_size: Width of the shared-expert output.
        lamport_max_rows: Largest row count using Lamport instead of pull.
        lamport_stages: Number of mailbox generations before reuse.
        lamport_block_elements: Elements owned by one workgroup.
        lamport_num_subgroups: Subgroups per workgroup; polling uses one.
        lamport_transaction_bytes: Width of each lane's publication/read.
    """

    world_size: int
    routed_hidden_size: int
    hidden_size: int
    lamport_max_rows: int
    lamport_stages: int
    lamport_block_elements: int
    lamport_num_subgroups: int
    lamport_transaction_bytes: int

    def __post_init__(self) -> None:
        if (
            self.world_size != 8
            or self.routed_hidden_size <= 0
            or self.hidden_size <= 0
            or self.lamport_max_rows <= 0
            or self.lamport_stages < 3
            or self.lamport_block_elements != 512
            or self.lamport_num_subgroups != 1
            or self.lamport_transaction_bytes != 16
            or self.row_numel % self.lamport_block_elements
        ):
            raise ValueError("invalid Kimi-K3 MoE Lamport kernel configuration")

    @property
    def row_numel(self) -> int:
        return self.routed_hidden_size + self.hidden_size

    @property
    def lamport_max_numel(self) -> int:
        return self.lamport_max_rows * self.row_numel

    def rows_for_shapes(self, shapes: tuple[tuple[int, ...], ...]) -> int | None:
        if len(shapes) != 2 or any(len(shape) != 2 for shape in shapes):
            return None
        rows = shapes[0][0]
        if (
            rows <= 0
            or shapes[1][0] != rows
            or (shapes[0][1], shapes[1][1])
            not in (
                (self.routed_hidden_size, self.hidden_size),
                (self.hidden_size, self.routed_hidden_size),
            )
        ):
            return None
        return rows


@dataclass(frozen=True)
class IrisAllReduceKernelConfig:
    """Launch and workspace contract for TokenSpeed's Iris all-reduces.

    ``staged`` and ``producer_direct`` are generic AMD all-reduce paths.
    The producer-direct path is used by Kimi-K3 MoE for both TP/TP and TP/EP;
    it is not EP-specific. Attention-residual fusion has its own workspace
    and launch contract in ``_iris/attnres.py``.

    Attributes:
        subgroup_size: Hardware subgroup width used by the Gluon kernels.
        packed_word_bytes: Packed element width used by the symmetric kernels.
        staged: Launch and workspace parameters for staged all-reduce.
        producer_direct: Launch and eligibility parameters for producer-direct
            all-reduce.
        two_stage: Launch and workspace parameters shared by ordinary staged and
            producer-direct two-stage all-reduce.
        kimi_k3_moe: Shape, launch, and mailbox parameters for K3 MoE Lamport.
    """

    subgroup_size: int
    packed_word_bytes: int
    staged: _StagedAllReduceKernelConfig
    producer_direct: _ProducerDirectAllReduceKernelConfig
    two_stage: _TwoStageAllReduceKernelConfig
    kimi_k3_moe: KimiK3MoeAllReduceKernelConfig

    def __post_init__(self) -> None:
        if (
            self.subgroup_size <= 0
            or self.subgroup_size & (self.subgroup_size - 1)
            or self.packed_word_bytes <= 0
        ):
            raise ValueError("invalid Iris all-reduce kernel configuration")
        if any(
            not self.two_stage.supports_world_size(world_size)
            for world_size, _ in self.producer_direct.two_stage_min_bytes
        ):
            raise ValueError(
                "producer-direct two-stage thresholds require kernel support"
            )
        if self.subgroup_size != 64:
            raise ValueError("Kimi-K3 Lamport requires a 64-thread subgroup")


IRIS_ALL_REDUCE_KERNEL_CONFIG = IrisAllReduceKernelConfig(
    subgroup_size=64,
    packed_word_bytes=8,
    staged=_StagedAllReduceKernelConfig(
        block_size=2048,
        num_subgroups=4,
        input_slots=2,
        # This CDNA4 TP4 tuning came from GLM-5.3-Flash decode.
        cdna4_tunings=(
            _StagedAllReduceKernelTuning(
                world_size=4,
                dtype=torch.bfloat16,
                numel=16 * 4096,
                block_size=512,
                num_subgroups=1,
            ),
        ),
    ),
    producer_direct=_ProducerDirectAllReduceKernelConfig(
        supported_world_sizes=(2, 4, 8),
        one_stage_block_size=512,
        one_stage_max_programs=84,
        one_stage_num_subgroups=1,
        one_stage_words_per_lane=2,
        two_stage_min_bytes=((4, 160 << 10), (8, 96 << 10)),
        publish_ready=False,
    ),
    two_stage=_TwoStageAllReduceKernelConfig(
        supported_world_sizes=(4, 8),
        max_programs=84,
        num_subgroups=8,
        words_per_lane=2,
    ),
    kimi_k3_moe=KimiK3MoeAllReduceKernelConfig(
        world_size=8,
        routed_hidden_size=3584,
        hidden_size=7168,
        lamport_max_rows=6,
        lamport_stages=3,
        lamport_block_elements=512,
        lamport_num_subgroups=1,
        lamport_transaction_bytes=16,
    ),
)


def _kimi_k3_moe_producer_direct_protocol(
    world_size: int,
    shapes: tuple[tuple[int, ...], ...],
    dtype: torch.dtype,
) -> str | None:
    config = IRIS_ALL_REDUCE_KERNEL_CONFIG.kimi_k3_moe
    if world_size != config.world_size or dtype != torch.bfloat16:
        return None
    rows = config.rows_for_shapes(shapes)
    return "lamport" if rows is not None and rows <= config.lamport_max_rows else None


def producer_direct_all_reduce_can_run(
    world_size: int,
    total_numel: int,
    dtype: torch.dtype,
    max_bytes: int,
) -> bool:
    """Check the generic AMD producer-direct kernel's payload requirements.

    Args:
        world_size: Number of ranks participating in the all-reduce.
        total_numel: Total number of elements in the payload.
        dtype: Element type of the payload.
        max_bytes: Byte capacity of the producer-direct symmetric input buffer.

    Returns:
        Whether the group size and element type are supported and the payload is
        positive, packed-word aligned, and within the input-buffer capacity.
    """
    kernel_config = IRIS_ALL_REDUCE_KERNEL_CONFIG
    config = kernel_config.producer_direct
    element_bytes = dtype.itemsize
    return (
        config.supports_world_size(world_size)
        and total_numel > 0
        and dtype in _PRODUCER_DIRECT_GL_DTYPES
        and kernel_config.packed_word_bytes % element_bytes == 0
        and total_numel % (kernel_config.packed_word_bytes // element_bytes) == 0
        and total_numel * element_bytes <= max_bytes
    )


def _use_two_stage_producer_direct(
    world_size: int,
    total_numel: int,
    dtype: torch.dtype,
) -> bool:
    kernel_config = IRIS_ALL_REDUCE_KERNEL_CONFIG
    min_bytes = kernel_config.producer_direct.two_stage_threshold(world_size)
    if min_bytes is None or dtype not in _PRODUCER_DIRECT_GL_DTYPES:
        return False
    elements_per_word = kernel_config.packed_word_bytes // dtype.itemsize
    return total_numel * dtype.itemsize >= min_bytes and (
        kernel_config.two_stage.can_partition(
            world_size=world_size,
            total_numel=total_numel,
            elements_per_word=elements_per_word,
        )
    )


def _use_two_stage_plain(
    world_size: int,
    numel: int,
    dtype: torch.dtype,
) -> bool:
    """Whether a plain all-reduce of ``numel`` should take the two-stage path.

    Args:
        world_size: Ranks participating in the reduction.
        numel: Elements in the tensor being reduced.
        dtype: Element type; sets how many elements pack into a 64-bit word.

    Returns:
        True when the two-stage reduce-scatter/all-gather can run this shape.

    Unlike the producer-direct threshold this carries no minimum size. Measured
    on gfx950 at world 8, the two forms are within noise of each other below
    about 16 tokens of hidden 7168 (one-shot is marginally ahead at some of those
    shapes), and two-stage pulls away above it: 1.11x at 16 tokens, 1.37x at 32,
    1.82x at 64. A minimum would buy nothing at the small end and risks sitting
    in the wrong place as shapes change, so the only condition kept is the
    kernel's structural one -- the payload has to split evenly into per-rank
    partitions of whole 64-bit words.
    """
    # The kernel packs elements into 64-bit words through
    # _PRODUCER_DIRECT_GL_DTYPES; anything outside it (or wider than a word,
    # which would make elements_per_word zero) stays on one-shot.
    if dtype not in _PRODUCER_DIRECT_GL_DTYPES:
        return False
    kernel_config = IRIS_ALL_REDUCE_KERNEL_CONFIG
    elements_per_word = kernel_config.packed_word_bytes // dtype.itemsize
    return kernel_config.two_stage.can_partition(
        world_size=world_size,
        total_numel=numel,
        elements_per_word=elements_per_word,
    )


def _select_staged_all_reduce_path(
    numel: int,
    world_size: int,
    dtype: torch.dtype,
    two_stage_supported: bool,
) -> tuple[_StagedAllReduceKernelTuning | None, bool]:
    """Resolve the tuned one-shot override and two-stage dispatch together."""
    tuning = IRIS_ALL_REDUCE_KERNEL_CONFIG.staged.tuning(
        numel=numel,
        world_size=world_size,
        dtype=dtype,
        is_cdna4=_platform.is_cdna4,
    )
    use_two_stage = (
        tuning is None
        and two_stage_supported
        and _use_two_stage_plain(world_size, numel, dtype)
    )
    return tuning, use_two_stage


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
    elif heap_size > _iris_ctx_singleton.heap_size:
        raise RuntimeError(
            f"Iris has a {_iris_ctx_singleton.heap_size}-byte symmetric heap, "
            f"but this state requires {heap_size} bytes; prepare the largest "
            "state first"
        )
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
        assert _platform.is_amd, (
            "IrisRSAG currently targets AMD ROCm; " f"got non-AMD platform: {_platform}"
        )
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
            "Iris RSAG symmetric-heap buffers allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
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
        staged_max_numel: int,
        producer_direct_max_numel: int,
        attnres_max_numel: int,
        attnres_max_rows: int,
        enable_lamport: bool,
        moe_tail_max_rows: int,
        dtype: torch.dtype,
        heap_size: int | None,
        device: torch.device | None,
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
        self.staged_max_numel = staged_max_numel
        self.producer_direct_max_numel = producer_direct_max_numel
        self.attnres_max_numel = attnres_max_numel
        self.attnres_max_rows = attnres_max_rows
        self.enable_lamport = enable_lamport
        self.moe_tail_max_rows = moe_tail_max_rows
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self.world_size = group.size()
        if (
            min(
                staged_max_numel,
                producer_direct_max_numel,
                attnres_max_numel,
                attnres_max_rows,
                moe_tail_max_rows,
            )
            < 0
        ):
            raise ValueError("Iris all-reduce capacities must be non-negative")
        if bool(attnres_max_numel) != bool(attnres_max_rows):
            raise ValueError(
                "AttnRes element and row capacities must both be zero or non-zero"
            )
        self._kernel_config = IRIS_ALL_REDUCE_KERNEL_CONFIG
        producer_config = self._kernel_config.producer_direct
        staged_config = self._kernel_config.staged
        two_stage_config = self._kernel_config.two_stage
        moe_config = self._kernel_config.kimi_k3_moe
        if moe_tail_max_rows:
            IrisRowShardedWorkspace.validate_capacity(
                moe_tail_max_rows, self.world_size, dtype, producer_direct_max_numel
            )
        self._elements_per_word = (
            self._kernel_config.packed_word_bytes // dtype.itemsize
        )
        # Reserve complete eligible rows. One program owns one tile for every
        # invocation, independently of the pull path's 84-program cap.
        self._kimi_k3_moe_lamport_max_numel = (
            min(
                producer_direct_max_numel // moe_config.row_numel,
                moe_config.lamport_max_rows,
            )
            * moe_config.row_numel
            if enable_lamport
            and _platform.is_cdna4
            and self.world_size == moe_config.world_size
            and dtype == torch.bfloat16
            else 0
        )
        self._kimi_k3_moe_lamport_max_programs = (
            self._kimi_k3_moe_lamport_max_numel // moe_config.lamport_block_elements
        )
        self._producer_direct_two_stage_workspace_required = (
            producer_direct_max_numel > 0
            and producer_config.two_stage_threshold(self.world_size) is not None
            and two_stage_config.supports_world_size(self.world_size)
        )
        self._producer_direct_scratch_numel = (
            two_stage_config.scratch_numel(
                max_numel=producer_direct_max_numel,
                world_size=self.world_size,
            )
            if self._producer_direct_two_stage_workspace_required
            else 0
        )
        self._producer_direct_max_programs = max(
            REDUCE_PROGRAMS if moe_tail_max_rows else 0,
            producer_config.one_stage_max_programs,
            (
                two_stage_config.max_programs
                if self._producer_direct_two_stage_workspace_required
                else 0
            ),
        )
        self._staged_max_programs = staged_config.max_programs(
            max_numel=staged_max_numel
        )
        self._staged_tunings = tuple(
            tuning
            for tuning in staged_config.cdna4_tunings
            if _platform.is_cdna4
            and tuning.world_size == self.world_size
            and tuning.dtype == dtype
            and tuning.numel <= staged_max_numel
        )

        # Fix capability-dependent storage at construction; only payload size
        # participates in per-call dispatch.
        self._staged_two_stage_supported = (
            _platform.is_cdna4
            and staged_max_numel > 0
            and two_stage_config.supports_world_size(self.world_size)
            and dtype in _PRODUCER_DIRECT_GL_DTYPES
        )
        self._staged_two_stage_scratch_numel = (
            two_stage_config.scratch_numel(
                max_numel=staged_max_numel,
                world_size=self.world_size,
            )
            if self._staged_two_stage_supported
            else 0
        )

        if heap_size is None:
            payload_numel = (
                producer_direct_max_numel
                + self._producer_direct_scratch_numel
                + staged_config.input_slots * staged_max_numel
                + staged_config.input_slots
                * sum(tuning.numel for tuning in self._staged_tunings)
                + (staged_max_numel if self._staged_two_stage_supported else 0)
                + self._staged_two_stage_scratch_numel
                + moe_config.lamport_stages
                * self.world_size
                * self._kimi_k3_moe_lamport_max_numel
            )
            flag_numel = self.world_size * (
                self._staged_max_programs
                + sum(tuning.num_programs() for tuning in self._staged_tunings)
                + (
                    self._producer_direct_max_programs
                    if producer_direct_max_numel
                    else 0
                )
                + (
                    two_stage_config.max_programs
                    if self._staged_two_stage_supported
                    else 0
                )
            )
            heap_size = max(
                1 << 28,
                payload_numel * dtype.itemsize
                + flag_numel * torch.int32.itemsize
                + IrisAttnResWorkspace.heap_bytes(
                    attnres_max_numel, attnres_max_rows, self.world_size, dtype
                )
                + (16 << 20),
            )
            # Preserve the base heap's headroom for other collective states.
            if moe_tail_max_rows:
                heap_size += IrisRowShardedWorkspace.heap_bytes(
                    moe_tail_max_rows, self.world_size, dtype
                )

        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        self._ctx = _get_or_create_iris_context(heap_size)
        group_ranks = dist.get_process_group_ranks(group)
        assert len(group_ranks) == self.world_size
        assert group_ranks[rank_in_group] == dist.get_rank()
        heap_bases = self._ctx.get_heap_bases()
        self._group_heap_bases = heap_bases[group_ranks].contiguous()
        group_heap_bases = [int(address) for address in self._group_heap_bases.tolist()]
        self._heap_base_addresses = tuple(
            group_heap_bases + [group_heap_bases[-1]] * (8 - self.world_size)
        )
        self._input_buf = (
            self._ctx.zeros((producer_direct_max_numel,), dtype=dtype)
            if producer_direct_max_numel
            else None
        )
        self._producer_direct_scratch_buf = (
            self._ctx.zeros((self._producer_direct_scratch_numel,), dtype=dtype)
            if self._producer_direct_scratch_numel
            else None
        )
        self._ready_flags = (
            self._ctx.zeros(
                (self._staged_max_programs, self.world_size), dtype=torch.int32
            )
            if staged_max_numel
            else None
        )
        # The staged one-shot rotates across its own slots rather than sharing
        # _input_buf: that buffer is handed out by acquire_outputs for
        # producer-direct reductions, so its layout is not ours to rotate.
        self._staged_input_buf = (
            self._ctx.zeros((staged_config.input_slots, staged_max_numel), dtype=dtype)
            if staged_max_numel
            else None
        )
        # Different block geometries cannot share per-block epochs or rotating
        # slots because their block IDs cover overlapping element ranges.
        self._staged_tuning_workspaces = {
            tuning: (
                self._ctx.zeros((staged_config.input_slots, tuning.numel), dtype=dtype),
                self._ctx.zeros(
                    (tuning.num_programs(), self.world_size), dtype=torch.int32
                ),
            )
            for tuning in self._staged_tunings
        }
        # Staging has a stable address for graph replay and dedicated storage.
        # The exit barrier completes peer reads before the next staging copy.
        # Producer-direct and one-shot slots have independent ownership.
        if self._staged_two_stage_supported:
            self._staged_two_stage_input_buf = self._ctx.zeros(
                (staged_max_numel,), dtype=dtype
            )
            # Each rank reduces only its own partition and peers read it at the
            # same offset, so scratch holds one partition, not the whole payload.
            self._staged_two_stage_scratch_buf = self._ctx.zeros(
                (self._staged_two_stage_scratch_numel,), dtype=dtype
            )
        else:
            self._staged_two_stage_input_buf = None
            self._staged_two_stage_scratch_buf = None
        self._producer_direct_ready_flags = (
            self._ctx.zeros(
                (
                    self._producer_direct_max_programs,
                    self.world_size,
                ),
                dtype=torch.int32,
            )
            if producer_direct_max_numel
            else None
        )
        self._kimi_k3_moe_lamport_region = (
            self._ctx.zeros(
                (
                    moe_config.lamport_stages,
                    self.world_size,
                    self._kimi_k3_moe_lamport_max_numel,
                ),
                dtype=dtype,
            )
            if self._kimi_k3_moe_lamport_max_numel
            else None
        )
        self._kimi_k3_moe_lamport_epochs = (
            torch.zeros(
                (self._kimi_k3_moe_lamport_max_programs,),
                dtype=torch.int32,
                device=self.device,
            )
            if self._kimi_k3_moe_lamport_max_numel
            else None
        )
        if self._kimi_k3_moe_lamport_region is not None:
            # Alternating +0/-0 halves: every lane's 16-byte pack has sentinel
            # halves. Legitimate -0 inputs are normalized before publication.
            self._kimi_k3_moe_lamport_region.view(torch.int32).fill_(-2147483648)
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
        # Separate epochs from the producer-direct reduce: in a tensor-parallel
        # MoE both collectives run inside one layer, and a shared counter would
        # let one path's epoch satisfy the other's barrier.
        self._staged_two_stage_ready_flags = (
            self._ctx.zeros(
                (two_stage_config.max_programs, self.world_size),
                dtype=torch.int32,
            )
            if self._staged_two_stage_supported
            else None
        )
        self._kimi_k3_moe_lamport_peer_addresses = None
        if self._kimi_k3_moe_lamport_region is not None:
            heap_offset = (
                self._kimi_k3_moe_lamport_region.data_ptr()
                - self._heap_base_addresses[rank_in_group]
            )
            self._kimi_k3_moe_lamport_peer_addresses = tuple(
                heap_base + heap_offset for heap_base in self._heap_base_addresses
            )
            assert all(
                address % moe_config.lamport_transaction_bytes == 0
                for address in self._kimi_k3_moe_lamport_peer_addresses
            )
        self.attnres: IrisAttnResWorkspace | None = None
        if attnres_max_numel:
            self.attnres = IrisAttnResWorkspace(
                self._ctx,
                group=group,
                rank=rank_in_group,
                device=self.device,
                heap_bases=self._heap_base_addresses,
                max_numel=attnres_max_numel,
                max_rows=attnres_max_rows,
                dtype=dtype,
            )
        self.row_sharded: IrisRowShardedWorkspace | None = None
        if moe_tail_max_rows:
            assert self._input_buf is not None
            assert self._producer_direct_scratch_buf is not None
            assert self._producer_direct_ready_flags is not None
            self.row_sharded = IrisRowShardedWorkspace(
                self._ctx,
                rank=rank_in_group,
                heap_bases=self._heap_base_addresses,
                inputs=self._input_buf,
                scratch=self._producer_direct_scratch_buf,
                reduce_flags=self._producer_direct_ready_flags,
                max_rows=moe_tail_max_rows,
            )
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris all-reduce symmetric-heap buffers allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
        )

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        out: torch.Tensor,
        op,
    ) -> torch.Tensor:
        if op is None:
            op = dist.ReduceOp.SUM
        assert op == dist.ReduceOp.SUM, f"Iris all-reduce only supports SUM, got {op}"
        assert tensor.dtype == self.dtype, (
            f"Iris all-reduce dtype mismatch: tensor={tensor.dtype}, "
            f"backend={self.dtype}"
        )
        numel = tensor.numel()
        assert 0 < numel <= self.staged_max_numel, (
            f"tensor numel ({numel}) exceeds iris buffer capacity "
            f"({self.staged_max_numel})"
        )
        assert tensor.is_contiguous() and out.is_contiguous()
        assert out.shape == tensor.shape and out.dtype == tensor.dtype
        assert out.device == tensor.device == self.device
        if out.data_ptr() != tensor.data_ptr():
            size_bytes = numel * tensor.element_size()
            assert (
                out.data_ptr() + size_bytes <= tensor.data_ptr()
                or tensor.data_ptr() + size_bytes <= out.data_ptr()
            ), "all-reduce input and output must be identical or disjoint"
        kernel_config = self._kernel_config.staged
        tuning, use_two_stage = _select_staged_all_reduce_path(
            numel=numel,
            world_size=self.world_size,
            dtype=self.dtype,
            two_stage_supported=self._staged_two_stage_supported,
        )
        if use_two_stage:
            return self._all_reduce_two_stage(tensor, numel, out=out)
        if tuning is None:
            block_size = kernel_config.block_size
            num_subgroups = kernel_config.num_subgroups
            input_buf = self._staged_input_buf
            ready_flags = self._ready_flags
            slot_stride = self.staged_max_numel
        else:
            block_size = tuning.block_size
            num_subgroups = tuning.num_subgroups
            input_buf, ready_flags = self._staged_tuning_workspaces[tuning]
            slot_stride = tuning.numel
        assert input_buf is not None and ready_flags is not None
        iris_stage_one_shot_allreduce_kernel[(triton.cdiv(numel, block_size),)](
            tensor.view(-1),
            input_buf.view(-1),
            out.view(-1),
            ready_flags,
            self._group_heap_bases,
            numel,
            RANK=self.rank_in_group,
            WORLD_SIZE=self.world_size,
            BLOCK_SIZE=block_size,
            SLOT_STRIDE=slot_stride,
            NUM_SLOTS=kernel_config.input_slots,
            num_warps=num_subgroups,
        )

        return out

    @staticmethod
    def _views(
        buffer: torch.Tensor,
        shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[torch.Tensor, ...]:
        views = []
        offset = 0
        for shape in shapes:
            numel = math.prod(shape)
            views.append(buffer.narrow(0, offset, numel).view(shape))
            offset += numel
        return tuple(views)

    def acquire_outputs(
        self,
        shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return consecutive views of the symmetric Iris input buffer."""
        if not shapes or any(math.prod(shape) <= 0 for shape in shapes):
            raise ValueError("Iris requires non-empty symmetric output shapes")
        if sum(math.prod(shape) for shape in shapes) > self.producer_direct_max_numel:
            raise ValueError("Iris symmetric outputs exceed the input buffer")
        assert self._input_buf is not None
        return self._views(self._input_buf, shapes)

    def owns_outputs(self, tensors: tuple[torch.Tensor, ...]) -> bool:
        """Whether tensors are consecutive views of this symmetric buffer."""
        if not tensors or any(
            tensor.dtype != self.dtype
            or tensor.device != self.device
            or not tensor.is_contiguous()
            or tensor.numel() <= 0
            for tensor in tensors
        ):
            return False
        if self._input_buf is None:
            return False
        element_size = self._input_buf.element_size()
        offset = 0
        for tensor in tensors:
            if tensor.data_ptr() != self._input_buf.data_ptr() + offset * element_size:
                return False
            offset += tensor.numel()
        return (
            offset % self._elements_per_word == 0
            and offset <= self.producer_direct_max_numel
        )

    def _all_reduce_two_stage(
        self, tensor: torch.Tensor, numel: int, *, out: torch.Tensor
    ) -> torch.Tensor:
        """Reduce into the caller's destination, using element stores if unaligned."""
        staged = self._staged_two_stage_input_buf
        staged[:numel].copy_(tensor.view(-1))

        partition_numel = numel // self.world_size
        partition_words = partition_numel // self._elements_per_word
        kernel_config = self._kernel_config.two_stage
        block_words = kernel_config.block_words(
            world_size=self.world_size,
            subgroup_size=self._kernel_config.subgroup_size,
        )
        num_tiles = triton.cdiv(partition_words, block_words)
        num_programs = min(num_tiles, kernel_config.max_programs)
        iris_reduce_symmetric_two_stage_gluon_kernel[(num_programs,)](
            staged,
            self._staged_two_stage_scratch_buf,
            out,
            self._staged_two_stage_ready_flags,
            *self._heap_base_addresses,
            RANK=self.rank_in_group,
            WORLD_SIZE=self.world_size,
            PARTITION_WORDS=partition_words,
            BLOCK_WORDS=block_words,
            NUM_PROGRAMS=num_programs,
            NUM_TILES=num_tiles,
            NUM_WARPS=kernel_config.num_subgroups,
            SUBGROUP_SIZE=self._kernel_config.subgroup_size,
            WORDS_PER_LANE=kernel_config.words_per_lane,
            ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[self.dtype],
            ELEMENTS_PER_WORD=self._elements_per_word,
            ALIGNED_OUTPUT=out.data_ptr() % self._kernel_config.packed_word_bytes == 0,
            EXIT_BARRIER=True,
            num_warps=kernel_config.num_subgroups,
        )
        return out

    def all_reduce_symmetric(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Reduce consecutive symmetric inputs into caller-owned local storage."""
        assert self.owns_outputs(tensors)
        if not _platform.is_cdna4:
            raise RuntimeError("producer-direct Iris all-reduce requires CDNA4")
        kernel_config = self._kernel_config.producer_direct
        if not kernel_config.supports_world_size(self.world_size):
            raise RuntimeError(
                "producer-direct Iris all-reduce does not support group size "
                f"{self.world_size}"
            )
        if self.dtype not in _PRODUCER_DIRECT_GL_DTYPES:
            raise RuntimeError(
                f"producer-direct Iris all-reduce does not support {self.dtype}"
            )

        shapes = tuple(tuple(tensor.shape) for tensor in tensors)
        total_numel = sum(tensor.numel() for tensor in tensors)
        output = torch.empty(total_numel, dtype=self.dtype, device=self.device)
        if (
            self.enable_lamport
            and _kimi_k3_moe_producer_direct_protocol(
                self.world_size, shapes, self.dtype
            )
            == "lamport"
        ):
            self._all_reduce_symmetric_lamport(output)
        else:
            self._all_reduce_symmetric_pull(output)
        return self._views(output, shapes)

    def _all_reduce_symmetric_lamport(self, output: torch.Tensor) -> None:
        total_numel = output.numel()
        config = self._kernel_config.kimi_k3_moe
        assert total_numel <= self._kimi_k3_moe_lamport_max_numel
        assert total_numel % config.lamport_block_elements == 0
        assert self._kimi_k3_moe_lamport_region is not None
        assert self._kimi_k3_moe_lamport_epochs is not None
        assert self._kimi_k3_moe_lamport_peer_addresses is not None
        num_programs = total_numel // config.lamport_block_elements
        lamport_all_reduce_bf16[(num_programs,)](
            self._input_buf,
            self._kimi_k3_moe_lamport_region,
            output,
            self._kimi_k3_moe_lamport_epochs,
            *self._kimi_k3_moe_lamport_peer_addresses,
            RANK=self.rank_in_group,
            WORLD_SIZE=self.world_size,
            MAX_ELEMENTS=self._kimi_k3_moe_lamport_max_numel,
            NUM_STAGES=config.lamport_stages,
            num_warps=config.lamport_num_subgroups,
        )

    def _all_reduce_symmetric_pull(self, output: torch.Tensor) -> None:
        total_numel = output.numel()
        kernel_config = self._kernel_config.producer_direct
        use_two_stage = _use_two_stage_producer_direct(
            world_size=self.world_size,
            total_numel=total_numel,
            dtype=self.dtype,
        )
        if use_two_stage:
            two_stage_config = self._kernel_config.two_stage
            assert self._producer_direct_scratch_buf is not None
            partition_numel = total_numel // self.world_size
            partition_words = partition_numel // self._elements_per_word
            block_words = two_stage_config.block_words(
                world_size=self.world_size,
                subgroup_size=self._kernel_config.subgroup_size,
            )
            num_tiles = triton.cdiv(partition_words, block_words)
            num_programs = min(num_tiles, two_stage_config.max_programs)
            iris_reduce_symmetric_two_stage_gluon_kernel[(num_programs,)](
                self._input_buf,
                self._producer_direct_scratch_buf,
                output,
                self._producer_direct_ready_flags,
                *self._heap_base_addresses,
                RANK=self.rank_in_group,
                WORLD_SIZE=self.world_size,
                PARTITION_WORDS=partition_words,
                BLOCK_WORDS=block_words,
                NUM_PROGRAMS=num_programs,
                NUM_TILES=num_tiles,
                NUM_WARPS=two_stage_config.num_subgroups,
                SUBGROUP_SIZE=self._kernel_config.subgroup_size,
                WORDS_PER_LANE=two_stage_config.words_per_lane,
                ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[self.dtype],
                ELEMENTS_PER_WORD=self._elements_per_word,
                ALIGNED_OUTPUT=True,
                EXIT_BARRIER=False,
                num_warps=two_stage_config.num_subgroups,
            )
        else:
            block_size = kernel_config.one_stage_block_size
            num_tiles = triton.cdiv(total_numel, block_size)
            num_programs = min(num_tiles, kernel_config.one_stage_max_programs)
            iris_reduce_symmetric_gluon_kernel[(num_programs,)](
                self._input_buf,
                output,
                self._producer_direct_ready_flags,
                *self._heap_base_addresses,
                RANK=self.rank_in_group,
                WORLD_SIZE=self.world_size,
                TOTAL_NUMEL=total_numel,
                BLOCK_SIZE=block_size,
                NUM_PROGRAMS=num_programs,
                NUM_TILES=num_tiles,
                NUM_WARPS=kernel_config.one_stage_num_subgroups,
                SUBGROUP_SIZE=self._kernel_config.subgroup_size,
                WORDS_PER_LANE=kernel_config.one_stage_words_per_lane,
                PUBLISH_READY=kernel_config.publish_ready,
                ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[self.dtype],
                ELEMENTS_PER_WORD=self._elements_per_word,
                num_warps=kernel_config.one_stage_num_subgroups,
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
        *,
        persistent: bool,
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
        group_ranks = dist.get_process_group_ranks(group)
        assert group_ranks[rank_in_group] == dist.get_rank()
        self._group_heap_bases = self._ctx.get_heap_bases()[group_ranks].contiguous()
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris AR+RMSNorm symmetric-heap buffer allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
        )

        self._rank_start = 0
        self._rank_stride = 1
        self._iris_rank = rank_in_group

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
        assert num_tokens <= self.max_token_num, (
            f"num_tokens ({num_tokens}) exceeds max_token_num "
            f"({self.max_token_num})"
        )
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

        self._ctx.device_barrier(group=self.group)

        heap_bases = self._group_heap_bases
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
        self._ctx.device_barrier(group=self.group)
        return norm_out, residual_out


def create_iris_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    staged_max_numel: int,
    producer_direct_max_numel: int,
    attnres_max_numel: int,
    attnres_max_rows: int,
    enable_lamport: bool,
    moe_tail_max_rows: int,
    dtype: torch.dtype,
    heap_size: int | None,
    device: torch.device | None,
) -> "IrisAllReduce":
    """Create an Iris all-reduce state with separate capacities for each path.

    Args:
        group: Process group used by the collectives.
        rank_in_group: This process's rank within ``group``.
        staged_max_numel: Maximum ordinary staged all-reduce payload.
        producer_direct_max_numel: Maximum producer-direct payload.
        attnres_max_numel: Maximum fused attention/AttnRes payload.
        attnres_max_rows: Maximum fused attention/AttnRes rows.
        enable_lamport: Allow Lamport for eligible producer-direct payloads.
        moe_tail_max_rows: Maximum rows in the reusable symmetric result buffer;
            zero skips its allocation.
        dtype: Element type for all payload buffers.
        heap_size: Optional symmetric heap size in bytes.
        device: Device on which buffers are allocated.

    Returns:
        The initialized all-reduce state.
    """
    return IrisAllReduce(
        group=group,
        rank_in_group=rank_in_group,
        staged_max_numel=staged_max_numel,
        producer_direct_max_numel=producer_direct_max_numel,
        attnres_max_numel=attnres_max_numel,
        attnres_max_rows=attnres_max_rows,
        enable_lamport=enable_lamport,
        moe_tail_max_rows=moe_tail_max_rows,
        dtype=dtype,
        heap_size=heap_size,
        device=device,
    )


def get_or_create_iris_state(
    *,
    group: dist.ProcessGroup,
    rank_in_group: int,
    device: torch.device,
    staged_max_numel: int,
    producer_direct_max_numel: int,
    attnres_max_numel: int,
    attnres_max_rows: int,
    enable_lamport: bool,
    moe_tail_max_rows: int,
    dtype: torch.dtype,
) -> IrisAllReduce:
    """Reuse a compatible prepared state, or allocate its explicit capacities.

    Args:
        group: Process group used by the collectives.
        rank_in_group: This process's rank within group.
        device: Device owning the symmetric buffers.
        staged_max_numel: Required ordinary all-reduce capacity.
        producer_direct_max_numel: Required producer-output capacity.
        attnres_max_numel: Required fused AttnRes payload capacity.
        attnres_max_rows: Required fused AttnRes row capacity.
        enable_lamport: Whether eligible producer outputs may use Lamport.
        moe_tail_max_rows: Required row-sharded result row capacity.
        dtype: Element type for all payload buffers.

    Returns:
        A retained state with sufficient capacity and compatible protocols.
        Prepare the largest capacities first; the shared heap cannot grow.
    """
    key = (
        id(group),
        rank_in_group,
        device,
        dtype,
        staged_max_numel,
        producer_direct_max_numel,
        attnres_max_numel,
        attnres_max_rows,
        enable_lamport,
        moe_tail_max_rows,
    )
    state = IRIS_AR_STATES.get(key)
    if state is None:
        state = next(
            (
                candidate
                for candidate in IRIS_AR_STATES.values()
                if candidate.group is group
                and candidate.rank_in_group == rank_in_group
                and candidate.device == device
                and candidate.dtype == dtype
                and candidate.staged_max_numel >= staged_max_numel
                and candidate.producer_direct_max_numel >= producer_direct_max_numel
                and candidate.attnres_max_numel >= attnres_max_numel
                and candidate.attnres_max_rows >= attnres_max_rows
                and candidate.moe_tail_max_rows >= moe_tail_max_rows
                # AttnRes-only users do not select a producer-direct protocol.
                and (
                    producer_direct_max_numel == 0
                    or candidate.enable_lamport == enable_lamport
                )
            ),
            None,
        )
    if state is None:
        state = create_iris_state(
            group=group,
            rank_in_group=rank_in_group,
            device=device,
            staged_max_numel=staged_max_numel,
            producer_direct_max_numel=producer_direct_max_numel,
            attnres_max_numel=attnres_max_numel,
            attnres_max_rows=attnres_max_rows,
            enable_lamport=enable_lamport,
            moe_tail_max_rows=moe_tail_max_rows,
            dtype=dtype,
            heap_size=None,
        )
    IRIS_AR_STATES[key] = state
    return state


def iris_all_reduce(
    state: "IrisAllReduce",
    tensor: torch.Tensor,
    *,
    out: torch.Tensor,
    op,
) -> torch.Tensor:
    """Sum into an explicit destination without allocating or copying a result.

    Args:
        state: Prepared communication state.
        tensor: Contiguous local contribution.
        out: Same-shaped, same-dtype destination. Use tensor for in-place reduction
            or disjoint storage to preserve the input. Element-aligned views are
            supported without changing the collective selected by other ranks.
        op: Reduction operation; only SUM is supported.

    Returns:
        out, containing the group sum on the calling stream.
    """
    return state.all_reduce(tensor, out=out, op=op)


def iris_acquire_outputs(
    state: "IrisAllReduce",
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, ...]:
    """Return consecutive symmetric producer-output views for Iris."""
    return state.acquire_outputs(shapes)


def iris_all_reduce_symmetric(
    state: "IrisAllReduce",
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Return caller-owned reductions of consecutive symmetric producer outputs."""
    return state.all_reduce_symmetric(tensors)


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
    """Finish the exact Kimi-K3 attention reduction and AttnRes mix."""
    if op is None:
        op = dist.ReduceOp.SUM
    assert op == dist.ReduceOp.SUM
    assert state.attnres is not None
    return state.attnres.run(
        partial,
        residual,
        score_weight,
        output_weight,
        scratch,
        eps,
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
    *,
    persistent: bool,
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


def find_iris_state(
    group: dist.ProcessGroup, tensors: tuple[torch.Tensor, ...]
) -> IrisAllReduce | None:
    """Find the prepared owner of consecutive producer outputs, without allocation.

    Args:
        group: Process group that owns the producer's storage.
        tensors: Ordered views returned by symmetric output acquisition.

    Returns:
        The owning state, or None when the tensors are not prepared outputs.
    """
    return next(
        (
            state
            for state in IRIS_AR_STATES.values()
            if state.group is group and state.owns_outputs(tensors)
        ),
        None,
    )


def iris_kimi3_moe_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    prefix: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    prefix_is_sharded: bool,
    norm_weight: torch.Tensor | None,
    eps: float | None,
    group: dist.ProcessGroup,
) -> torch.Tensor | None:
    """Run the row-partition MoE fusion using the prepared producer's workspace."""
    state = find_iris_state(group, (routed_partial, shared_partial))
    if state is None or state.row_sharded is None:
        return None
    return state.row_sharded.moe_tail(
        routed_partial,
        shared_partial,
        prefix,
        projection_weight,
        prefix_is_sharded=prefix_is_sharded,
        norm_weight=norm_weight,
        eps=eps,
    )


def iris_attention_mix(
    partial: torch.Tensor,
    residual: torch.Tensor | None,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float,
    out_norm_weight: torch.Tensor,
    out_norm_eps: float,
    num_valid_blocks: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the row-partition AttnRes fusion using the prepared producer's workspace."""
    state = find_iris_state(group, (partial,))
    if state is None or state.row_sharded is None:
        return None
    return state.row_sharded.attention_mix(
        partial,
        residual,
        block_residual,
        res_weight,
        rms_weight,
        eps=eps,
        out_norm_weight=out_norm_weight,
        out_norm_eps=out_norm_eps,
        num_valid_blocks=num_valid_blocks,
    )


def attnres_combine_supported(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    rank: int,
    group: dist.ProcessGroup,
    local_world_size: int,
    op,
) -> bool:
    """Check the local group and tensor contract before selecting the fusion."""
    if op not in (None, dist.ReduceOp.SUM) or local_world_size <= 0:
        return False
    group_ranks = dist.get_process_group_ranks(group)
    if len({global_rank // local_world_size for global_rank in group_ranks}) != 1:
        return False
    if not 0 <= rank < len(group_ranks):
        return False
    return attnres_supported(
        input_tensor,
        residual,
        score_weight,
        output_weight,
        scratch,
        world_size=group.size(),
        device=input_tensor.device,
    )


def attnres_combine(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    score_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    rank: int,
    group: dist.ProcessGroup,
    local_world_size: int,
    eps: float,
    op,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the prepared AttnRes workspace, allocating its full window if needed."""
    assert attnres_combine_supported(
        input_tensor,
        residual,
        score_weight,
        output_weight,
        scratch,
        rank=rank,
        group=group,
        local_world_size=local_world_size,
        op=op,
    )
    state = next(
        (
            candidate
            for candidate in IRIS_AR_STATES.values()
            if candidate.group is group
            and candidate.rank_in_group == rank
            and candidate.device == input_tensor.device
            and candidate.dtype == input_tensor.dtype
            and candidate.attnres is not None
            and candidate.attnres_max_rows >= input_tensor.shape[0]
            and candidate.attnres_max_numel >= input_tensor.numel()
        ),
        None,
    )
    if state is None:
        state = get_or_create_iris_state(
            group=group,
            rank_in_group=rank,
            device=input_tensor.device,
            staged_max_numel=0,
            producer_direct_max_numel=0,
            attnres_max_numel=ATTNRES_MAX_ROWS * ATTNRES_KERNEL_CONFIG.hidden_size,
            attnres_max_rows=ATTNRES_MAX_ROWS,
            enable_lamport=False,
            moe_tail_max_rows=0,
            dtype=input_tensor.dtype,
        )
    return iris_all_reduce_residual_attnres(
        state,
        input_tensor,
        residual,
        score_weight,
        output_weight,
        scratch,
        eps,
        op=op,
    )
