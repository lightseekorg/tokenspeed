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
from dataclasses import dataclass
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
    "IRIS_ALL_REDUCE_KERNEL_CONFIG",
    "IrisAllReduce",
    "IrisAllReduceKernelConfig",
    "KimiK3AttnResKernelConfig",
    "IrisRSAG",
    "IrisAllReduceResidualRMSNorm",
    "create_iris_state",
    "iris_all_reduce",
    "iris_acquire_outputs",
    "iris_all_reduce_symmetric",
    "iris_all_reduce_residual_attnres",
    "producer_direct_all_reduce_can_run",
    "create_iris_rsag_state",
    "create_iris_ar_rmsnorm_state",
    "iris_allreduce_residual_rmsnorm",
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
class KimiK3AttnResKernelConfig:
    """Kimi-K3 fused attention-TP all-reduce and AttnRes contract.

    Attributes:
        world_size: Required attention tensor-parallel group size.
        hidden_size: Kimi-K3 attention output width.
        num_subgroups: Number of subgroups in each kernel workgroup.
        elements_per_thread: Number of hidden elements processed per thread.
    """

    world_size: int
    hidden_size: int
    num_subgroups: int
    elements_per_thread: int

    def __post_init__(self) -> None:
        if (
            self.world_size <= 1
            or self.hidden_size <= 0
            or self.num_subgroups <= 0
            or self.elements_per_thread <= 0
        ):
            raise ValueError("invalid Kimi-K3 AttnRes Iris kernel configuration")


@dataclass(frozen=True)
class IrisAllReduceKernelConfig:
    """Launch and workspace contract for TokenSpeed's Iris all-reduces.

    ``staged`` and ``producer_direct`` are generic AMD all-reduce paths.
    The producer-direct path is used by Kimi-K3 MoE for both TP/TP and TP/EP;
    it is not EP-specific. ``kimi_k3_attnres`` is model-specific and operates
    only on Kimi-K3's attention tensor-parallel group.

    Attributes:
        subgroup_size: Hardware subgroup width used by the Gluon kernels.
        packed_word_bytes: Packed element width used by the symmetric kernels.
        staged: Launch and workspace parameters for staged all-reduce.
        producer_direct: Launch and eligibility parameters for producer-direct
            all-reduce.
        two_stage: Launch and workspace parameters shared by ordinary staged and
            producer-direct two-stage all-reduce.
        kimi_k3_attnres: Launch and shape parameters for Kimi-K3 AttnRes.
    """

    subgroup_size: int
    packed_word_bytes: int
    staged: _StagedAllReduceKernelConfig
    producer_direct: _ProducerDirectAllReduceKernelConfig
    two_stage: _TwoStageAllReduceKernelConfig
    kimi_k3_attnres: KimiK3AttnResKernelConfig

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
    kimi_k3_attnres=KimiK3AttnResKernelConfig(
        world_size=8,
        hidden_size=7168,
        num_subgroups=4,
        elements_per_thread=1,
    ),
)


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
        staged_max_numel: int,
        producer_direct_max_numel: int,
        attnres_max_numel: int,
        attnres_max_rows: int,
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
        self.dtype = dtype
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        self.world_size = group.size()
        if (
            min(
                staged_max_numel,
                producer_direct_max_numel,
                attnres_max_numel,
                attnres_max_rows,
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
        self._elements_per_word = (
            self._kernel_config.packed_word_bytes // dtype.itemsize
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

        # Whether this state can ever dispatch a two-stage reduction. Platform,
        # group size and element type are all fixed for the life of the state,
        # so deciding once here keeps the buffers, the heap estimate and the
        # dispatch from disagreeing. Only the payload size is per-call, and it
        # stays in _use_two_stage_plain.
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
                + 2 * attnres_max_numel
                + (staged_max_numel if self._staged_two_stage_supported else 0)
                + self._staged_two_stage_scratch_numel
            )
            flag_numel = self.world_size * (
                self._staged_max_programs
                + sum(tuning.num_programs() for tuning in self._staged_tunings)
                + 2 * attnres_max_rows
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
                + (16 << 20),
            )

        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        self._ctx = _get_or_create_iris_context(heap_size)
        group_ranks = dist.get_process_group_ranks(group)
        assert len(group_ranks) == self.world_size
        assert group_ranks[rank_in_group] == dist.get_rank()
        self._input_buf = (
            self._ctx.zeros((producer_direct_max_numel,), dtype=dtype)
            if producer_direct_max_numel
            else None
        )
        self._attnres_input_buf = (
            self._ctx.zeros((2, attnres_max_numel), dtype=dtype)
            if attnres_max_numel
            else None
        )
        self._attnres_ready_flags = (
            self._ctx.zeros((attnres_max_rows, self.world_size), dtype=torch.int32)
            if attnres_max_numel
            else None
        )
        self._attnres_consumed_flags = (
            self._ctx.zeros((attnres_max_rows, self.world_size), dtype=torch.int32)
            if attnres_max_numel
            else None
        )
        self._producer_direct_scratch_buf = (
            self._ctx.zeros((self._producer_direct_scratch_numel,), dtype=dtype)
            if self._producer_direct_scratch_numel
            else None
        )
        output_max_numel = max(
            producer_direct_max_numel,
            staged_max_numel if self._staged_two_stage_supported else 0,
        )
        self._reduced_output_buf = (
            torch.empty(output_max_numel, dtype=dtype, device=self.device)
            if output_max_numel
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
        # Two-stage plain all-reduce. One-shot stages inside its own kernel and
        # picks a slot from its per-block epoch; the two-stage kernel instead
        # reads peers' inputs out of symmetric memory, so the payload has to be
        # staged before launch. That staging cannot share _staged_input_buf --
        # its slot is chosen by one-shot's kernel from a counter we cannot see --
        # nor _input_buf, which acquire_outputs hands to producer-direct.
        #
        # A single buffer, deliberately: rotating slots from the host does not
        # survive graph capture, which records the staging copy's address and
        # replays it unchanged. The kernel's exit barrier is what makes one
        # buffer safe -- see EXIT_BARRIER.
        #
        # Skipped entirely where dispatch could never reach them: a CDNA3 part
        # or a group of some other size would otherwise reserve a payload
        # buffer and a scratch partition it can never use, and an explicit
        # heap_size sized for the previous allocations would fail to fit them.
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
        heap_bases = self._ctx.get_heap_bases()
        self._group_heap_bases = heap_bases[group_ranks].contiguous()
        group_heap_bases = [int(address) for address in self._group_heap_bases.tolist()]
        self._heap_base_addresses = tuple(
            group_heap_bases + [group_heap_bases[-1]] * (8 - self.world_size)
        )
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris all-reduce symmetric-heap buffers allocated: %s GB",
            free_gpu_memory_begin - free_gpu_memory_after,
        )

        self._rank_start = 0
        self._rank_stride = 1
        self._iris_rank = rank_in_group
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
        assert 0 < numel <= self.staged_max_numel, (
            f"tensor numel ({numel}) exceeds iris buffer capacity "
            f"({self.staged_max_numel})"
        )
        kernel_config = self._kernel_config.staged
        tuning, use_two_stage = _select_staged_all_reduce_path(
            numel=numel,
            world_size=self.world_size,
            dtype=self.dtype,
            two_stage_supported=self._staged_two_stage_supported,
        )
        # One-shot has every rank publish the whole payload and read world_size
        # copies of it, so its cost grows with the payload; the two-stage form
        # moves 2x however wide the world is. Measured on gfx950 at world 8,
        # including the staging copy this path pays and one-shot does not:
        # parity below ~16 tokens of hidden 7168, then 1.11x at 16, 1.37x at 32,
        # 1.82x at 64. The predicate is the kernel's partitioning requirement,
        # not a crossover -- see _use_two_stage_plain.
        # The two-stage kernel stores through a CDNA4 buffer intrinsic, so it is
        # gated the same way the producer-direct reduce is; older AMD parts keep
        # the portable one-shot path.
        # Keep an explicitly tuned one-shot shape on that path. On TP4 gfx950,
        # one-shot remains faster for 16x4096 while two-stage wins at 64x4096.
        if use_two_stage:
            return self._all_reduce_two_stage(tensor, numel, safe=safe)
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
            tensor.view(-1),
            ready_flags,
            self._group_heap_bases,
            numel,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            BLOCK_SIZE=block_size,
            SLOT_STRIDE=slot_stride,
            NUM_SLOTS=kernel_config.input_slots,
            num_warps=num_subgroups,
        )

        return tensor.clone() if safe else tensor

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
        self, tensor: torch.Tensor, numel: int, safe: bool
    ) -> torch.Tensor:
        """Reduce ``tensor`` in place via reduce-scatter then all-gather.

        Args:
            tensor: Contiguous local contribution; overwritten with the sum.
            numel: ``tensor.numel()``, already checked against the heap capacity.
            safe: Return a copy rather than the caller's tensor, matching the
                one-shot path -- the reduction lands in place either way, so
                without this the result aliases the input.

        Returns:
            The reduction across the group; a clone of ``tensor`` when ``safe``.
        """
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
        assert self._reduced_output_buf is not None
        output = self._reduced_output_buf[:numel]
        iris_reduce_symmetric_two_stage_gluon_kernel[(num_programs,)](
            staged,
            self._staged_two_stage_scratch_buf,
            output,
            self._staged_two_stage_ready_flags,
            *self._heap_base_addresses,
            RANK=self._iris_rank,
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
            EXIT_BARRIER=True,
            num_warps=kernel_config.num_subgroups,
        )
        tensor.view(-1).copy_(output)
        return tensor.clone() if safe else tensor

    def all_reduce_symmetric(
        self, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Reduce consecutive producer outputs from symmetric memory."""
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

        total_numel = sum(tensor.numel() for tensor in tensors)
        assert self._reduced_output_buf is not None
        outputs = self._views(
            self._reduced_output_buf,
            tuple(tuple(tensor.shape) for tensor in tensors),
        )
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
                self._reduced_output_buf,
                self._producer_direct_ready_flags,
                *self._heap_base_addresses,
                RANK=self._iris_rank,
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
                EXIT_BARRIER=False,
                num_warps=two_stage_config.num_subgroups,
            )
        else:
            block_size = kernel_config.one_stage_block_size
            num_tiles = triton.cdiv(total_numel, block_size)
            num_programs = min(num_tiles, kernel_config.one_stage_max_programs)
            iris_reduce_symmetric_gluon_kernel[(num_programs,)](
                self._input_buf,
                self._reduced_output_buf,
                self._producer_direct_ready_flags,
                *self._heap_base_addresses,
                RANK=self._iris_rank,
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
        return outputs

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
        """Reduce a Kimi-K3 attention partial and finish its AttnRes mix."""
        if op is None:
            op = dist.ReduceOp.SUM
        assert op == dist.ReduceOp.SUM, f"Iris all-reduce only supports SUM, got {op}"
        kernel_config = self._kernel_config.kimi_k3_attnres
        assert _platform.is_cdna4 and self.world_size == kernel_config.world_size
        num_tokens = partial.shape[0]
        assert 0 < num_tokens <= self.attnres_max_rows
        expected_shape = (num_tokens, kernel_config.hidden_size)
        assert partial.shape == residual.shape == expected_shape
        assert partial.dtype == residual.dtype == self.dtype == torch.bfloat16
        assert partial.device == residual.device == self.device
        assert partial.is_contiguous() and residual.is_contiguous()
        assert 0 < partial.numel() <= self.attnres_max_numel, (
            f"tensor numel ({partial.numel()}) exceeds iris buffer capacity "
            f"({self.attnres_max_numel})"
        )
        assert self._attnres_input_buf is not None
        assert score_weight.shape == output_weight.shape == (kernel_config.hidden_size,)
        assert score_weight.dtype == output_weight.dtype == torch.bfloat16
        assert score_weight.device == output_weight.device == self.device
        assert score_weight.is_contiguous() and output_weight.is_contiguous()
        m, s_, acc = scratch
        assert m.shape == s_.shape == (num_tokens,)
        assert acc.shape == expected_shape
        assert m.dtype == s_.dtype == acc.dtype == torch.float32
        assert m.device == s_.device == acc.device == self.device
        assert m.is_contiguous() and s_.is_contiguous() and acc.is_contiguous()
        assert eps > 0.0

        hidden = torch.empty_like(partial)
        residual_out = torch.empty_like(residual)
        iris_stage_one_shot_allreduce_residual_attnres_gluon_kernel[(num_tokens,)](
            partial,
            residual,
            self._attnres_input_buf,
            score_weight,
            output_weight,
            m,
            s_,
            acc,
            hidden,
            residual_out,
            self._attnres_ready_flags,
            self._attnres_consumed_flags,
            *self._heap_base_addresses,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            M=num_tokens,
            HIDDEN=kernel_config.hidden_size,
            BLOCK=triton.next_power_of_2(kernel_config.hidden_size),
            INPUT_SLOT_STRIDE=self.attnres_max_numel,
            EPS=eps,
            ELEMENTS_PER_THREAD=kernel_config.elements_per_thread,
            NUM_WARPS=kernel_config.num_subgroups,
            SUBGROUP_SIZE=self._kernel_config.subgroup_size,
            num_warps=kernel_config.num_subgroups,
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
    SLOT_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    """One-shot all-reduce over rotating staging slots.

    A single staging slot is not safe across calls. Each rank stages, publishes
    a ready epoch, waits for its peers, sums their slots and returns -- but the
    store happens *before* the wait, so a rank called straight back overwrites
    the slot a slower peer is still summing, and that peer silently sums the
    next collective's data. No error, no hang, just wrong numbers on whichever
    ranks lagged.

    Rotating over at least two slots closes it, and the existing entry barrier
    makes two sufficient. Writing epoch E+1 lands on a different slot. Before a
    rank can wrap back to E's slot, it must pass the entry wait at E+1; a peer
    publishes ready(E+1) only after it has finished reading E. By the time a
    slot is reused, every peer is provably done with it, and no consumption flag
    or exit barrier has to be paid for.

    Epochs are per-block, so a grid that shrinks between calls leaves the higher
    blocks' counters where they were. That is consistent across ranks -- every
    rank launches the same grid for the same collective -- and blocks never read
    each other's slots, so they may sit on different slots at once.
    """
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    flag_offset = block_id * WORLD_SIZE
    local_ready = ready_flags + flag_offset + RANK
    epoch = tl.load(local_ready).to(tl.int32) + 1
    slot = (epoch % NUM_SLOTS) * SLOT_STRIDE

    local = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(input_sym_ptr + slot + offsets, local, mask=mask, cache_modifier=".wt")
    tl.debug_barrier()

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
                input_sym_ptr + slot + offsets,
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
def _iris_sync_rank_epoch(
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
    SUBGROUP_SIZE: gl.constexpr,
    PUBLISH: gl.constexpr,
):
    """Synchronize by polling peer epochs or publishing them locally."""
    ready_layout: gl.constexpr = gl.BlockedLayout(
        [1], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    peer_ids = gl.arange(0, WORLD_SIZE, layout=ready_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    flags_heap_offset = tl.cast(ready_flags, gl.uint64) - local_heap
    peer_mask = peer_ids != RANK
    if PUBLISH:
        remote_flags = tl.cast(
            peer_heaps + flags_heap_offset,
            gl.pointer_type(gl.int32),
        )
        remote_flags += block_id * WORLD_SIZE + RANK
        gl.store(
            remote_flags,
            epoch,
            mask=peer_mask,
            cache_modifier=".wt",
        )
        wait_flags = ready_flags + block_id * WORLD_SIZE + peer_ids
    else:
        wait_flags = tl.cast(
            peer_heaps + flags_heap_offset,
            gl.pointer_type(gl.int32),
        )
        wait_flags += block_id * WORLD_SIZE + peer_ids
    seen = gl.full([WORLD_SIZE], 0, gl.int32, layout=ready_layout)
    while gl.min(gl.where(peer_mask, seen, epoch), axis=0) < epoch:
        seen = gl.load(
            wait_flags,
            mask=peer_mask,
            other=epoch,
            cache_modifier=".cv",
            volatile=True,
        )


@gluon.jit
def _unpack_16bitx4(packed, dtype: gl.constexpr):
    value_0 = (packed & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    value_1 = (
        ((packed >> 16) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    value_2 = (
        ((packed >> 32) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    value_3 = (
        ((packed >> 48) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    return value_0, value_1, value_2, value_3


@gluon.jit
def _pack_16bitx4(value_0, value_1, value_2, value_3, dtype: gl.constexpr):
    bits_0 = value_0.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_1 = value_1.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_2 = value_2.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_3 = value_3.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    return bits_0 | (bits_1 << 16) | (bits_2 << 32) | (bits_3 << 48)


@gluon.jit
def _unpack_word(packed, dtype: gl.constexpr, elements_per_word: gl.constexpr):
    # Explicit branches keep Gluon from type-checking the inactive bitcast.
    if elements_per_word == 4:
        return _unpack_16bitx4(packed, dtype)
    else:
        value_0 = (packed & 0xFFFFFFFF).to(gl.uint32).to(dtype, bitcast=True)
        value_1 = ((packed >> 32) & 0xFFFFFFFF).to(gl.uint32).to(dtype, bitcast=True)
        return value_0, value_1, value_0, value_1


@gluon.jit
def _pack_word(
    value_0,
    value_1,
    value_2,
    value_3,
    dtype: gl.constexpr,
    elements_per_word: gl.constexpr,
):
    if elements_per_word == 4:
        return _pack_16bitx4(value_0, value_1, value_2, value_3, dtype)
    else:
        bits_0 = value_0.to(dtype).to(gl.uint32, bitcast=True).to(gl.uint64)
        bits_1 = value_1.to(dtype).to(gl.uint32, bitcast=True).to(gl.uint64)
        return bits_0 | (bits_1 << 32)


@gluon.jit
def iris_reduce_symmetric_gluon_kernel(
    input_sym_ptr,
    output_ptr,
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
    TOTAL_NUMEL: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_TILES: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    PUBLISH_READY: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
):
    """Reduce producer outputs placed consecutively in symmetric memory."""
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
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
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
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=PUBLISH_READY,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    layout: gl.constexpr = gl.BlockedLayout(
        [WORDS_PER_LANE], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    lane = gl.arange(0, BLOCK_SIZE // ELEMENTS_PER_WORD, layout=layout)
    total_packed: gl.constexpr = TOTAL_NUMEL // ELEMENTS_PER_WORD
    tile_id = block_id
    while tile_id < NUM_TILES:
        packed_offset = tile_id * (BLOCK_SIZE // ELEMENTS_PER_WORD) + lane
        mask = packed_offset < total_packed
        local_packed = gl.amd.cdna4.buffer_load(
            tl.cast(input_sym_ptr, gl.pointer_type(gl.uint64)),
            packed_offset.to(gl.int32),
            mask=mask,
            other=0,
        )
        acc_0, acc_1, acc_2, acc_3 = _unpack_word(
            local_packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
        )
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
                peer_0, peer_1, peer_2, peer_3 = _unpack_word(
                    peer_packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
                )
                acc_0 += peer_0
                acc_1 += peer_1
                acc_2 += peer_2
                acc_3 += peer_3

        packed_output = _pack_word(
            acc_0,
            acc_1,
            acc_2,
            acc_3,
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        gl.amd.cdna4.buffer_store(
            packed_output,
            tl.cast(output_ptr, gl.pointer_type(gl.uint64)),
            packed_offset.to(gl.int32),
            mask=mask,
        )
        tile_id += NUM_PROGRAMS

    # Do not return while a peer program can still be reading this rank's
    # input. The next producer reuses the same symmetric buffer.
    completion_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        completion_epoch,
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
        SUBGROUP_SIZE,
        PUBLISH=PUBLISH_READY,
    )


@gluon.jit
def iris_reduce_symmetric_two_stage_gluon_kernel(
    input_sym_ptr,
    scratch_sym_ptr,
    output_ptr,
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
    PARTITION_WORDS: gl.constexpr,
    BLOCK_WORDS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_TILES: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
    EXIT_BARRIER: gl.constexpr,
):
    """Reduce-scatter producer outputs, then all-gather the rank partitions."""
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
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
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
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=True,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    scratch_heap_offset = tl.cast(scratch_sym_ptr, gl.uint64) - local_heap
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, WORDS_PER_LANE],
        [1, SUBGROUP_SIZE],
        [WORLD_SIZE, NUM_WARPS // WORLD_SIZE],
        [1, 0],
    )
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [WORLD_SIZE, 1], [1, SUBGROUP_SIZE], [1, NUM_WARPS], [0, 1]
    )
    peer_layout: gl.constexpr = gl.SliceLayout(1, load_layout)
    word_layout: gl.constexpr = gl.SliceLayout(0, load_layout)
    reduce_word_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    peer_ids = gl.arange(0, WORLD_SIZE, layout=peer_layout)
    words = gl.arange(0, BLOCK_WORDS, layout=word_layout)
    reduce_words = gl.arange(0, BLOCK_WORDS, layout=reduce_word_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    peer_inputs = tl.cast(
        gl.expand_dims(peer_heaps, 1) + input_heap_offset,
        gl.pointer_type(gl.uint64),
    )
    peer_scratch = tl.cast(
        gl.expand_dims(peer_heaps, 1) + scratch_heap_offset,
        gl.pointer_type(gl.uint64),
    )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[32, 4]],
        [WORLD_SIZE, BLOCK_WORDS],
        [1, 0],
    )
    peer_values = gl.allocate_shared_memory(
        gl.uint64,
        [WORLD_SIZE, BLOCK_WORDS],
        shared_layout,
    )
    rank_start: gl.constexpr = RANK * PARTITION_WORDS

    # Reduce only this rank's partition of the full input into symmetric scratch.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        input_offset = rank_start + partition_offset
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_inputs + gl.expand_dims(input_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=".cg",
        )
        peer_values.store(values)
        gl.barrier()

        packed = peer_values.load(reduce_layout)
        value_0, value_1, value_2, value_3 = _unpack_word(
            packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
        )
        reduced = _pack_word(
            gl.sum(value_0, axis=0),
            gl.sum(value_1, axis=0),
            gl.sum(value_2, axis=0),
            gl.sum(value_3, axis=0),
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        gl.amd.cdna4.buffer_store(
            reduced,
            tl.cast(scratch_sym_ptr, gl.pointer_type(gl.uint64)),
            (tile_id * BLOCK_WORDS + reduce_words).to(gl.int32),
            mask=tile_id * BLOCK_WORDS + reduce_words < PARTITION_WORDS,
            cache=".wt",
        )
        gl.barrier()
        tile_id += NUM_PROGRAMS

    partitions_ready = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        partitions_ready,
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
        SUBGROUP_SIZE,
        PUBLISH=True,
    )

    # Gather one reduced partition from every rank into the local output.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_scratch + gl.expand_dims(partition_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=".cg",
        )
        output_offset = gl.expand_dims(peer_ids * PARTITION_WORDS, 1) + gl.expand_dims(
            partition_offset, 0
        )
        gl.store(
            tl.cast(output_ptr, gl.pointer_type(gl.uint64)) + output_offset,
            values,
            mask=gl.expand_dims(mask, 0),
        )
        tile_id += NUM_PROGRAMS

    if EXIT_BARRIER:
        # Callers that stage into the symmetric input before launching cannot
        # rotate buffers safely: under graph capture the staging copy records a
        # fixed address and replays it, so a rank one invocation ahead would
        # overwrite an input its slower peers are still reducing. Holding the
        # kernel until every peer has finished reading makes a single staging
        # buffer correct by construction, at the price of one more rendezvous.
        # Producer-direct callers own their input and pass False.
        reads_done = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
        _iris_sync_rank_epoch(
            ready_flags,
            block_id,
            reads_done,
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
            SUBGROUP_SIZE,
            PUBLISH=True,
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
    consumed_flags,
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
    M: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    INPUT_SLOT_STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """Kimi-K3 attention AR, residual, and split AttnRes combine."""
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    offset = gl.arange(0, BLOCK, layout=layout)
    mask = offset < HIDDEN
    offset_i32 = (row * HIDDEN + offset).to(gl.int32)
    weight_offset_i32 = offset.to(gl.int32)

    local = gl.amd.cdna4.buffer_load(
        partial_ptr,
        offset_i32,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    local_ready = ready_flags + row * WORLD_SIZE + RANK
    epoch = gl.load(local_ready).to(gl.int32) + 1
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
    reuse_epoch = gl.maximum(epoch - 2, 0)
    _iris_sync_rank_epoch(
        consumed_flags,
        row,
        reuse_epoch,
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
        SUBGROUP_SIZE,
        PUBLISH=False,
    )

    input_slot_ptr = input_sym_ptr + (epoch & 1) * INPUT_SLOT_STRIDE
    gl.amd.cdna4.buffer_store(
        local.to(input_slot_ptr.dtype.element_ty),
        input_slot_ptr,
        offset_i32,
        mask=mask,
        cache=".wt",
    )
    gl.barrier()

    gl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")
    _iris_sync_rank_epoch(
        ready_flags,
        row,
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
        SUBGROUP_SIZE,
        PUBLISH=True,
    )

    input_heap_offset = tl.cast(input_slot_ptr, gl.uint64) - local_heap
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

    # Publish consumption without serializing this epilogue. Reuse waits only
    # when a later invocation wraps back to the same staging slot.
    gl.barrier()
    consumed = consumed_flags + row * WORLD_SIZE + RANK
    gl.atomic_xchg(consumed, epoch, sem="release", scope="sys")

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
        weight_offset_i32,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    square_sum = gl.sum(gl.where(mask, prefix * prefix, 0.0), axis=0)
    dot = gl.sum(gl.where(mask, prefix * score_weight, 0.0), axis=0)
    prefix_logit = dot * gl.rsqrt(square_sum / HIDDEN + EPS)

    block_m = gl.load(scratch_m_ptr + row)
    block_s = gl.load(scratch_s_ptr + row)
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
        weight_offset_i32,
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
    staged_max_numel: int,
    producer_direct_max_numel: int,
    attnres_max_numel: int,
    attnres_max_rows: int,
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
    """Reduce consecutive symmetric producer outputs in one launch."""
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
