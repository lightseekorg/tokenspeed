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

# Loaded only after the attention mix has passed its eligibility checks.
_attn_res_mix_gfx950 = None

__all__ = [
    "IRIS_ALL_REDUCE_KERNEL_CONFIG",
    "IrisAllReduce",
    "IrisAllReduceKernelConfig",
    "KimiK3AttnResKernelConfig",
    "KimiK3MoeAllReduceKernelConfig",
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
_MOE_REDUCE_PROGRAMS = 24
_MOE_GATHER_PROGRAMS = 128


@gluon.jit
def _iris_drain_subgroup_vmem():
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r,~{memory}",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


def _peer_addresses(
    tensor: torch.Tensor,
    heap_bases: tuple[int, ...],
    rank: int,
) -> tuple[int, ...]:
    heap_offset = tensor.data_ptr() - heap_bases[rank]
    return tuple(heap_base + heap_offset for heap_base in heap_bases)


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
        kimi_k3_moe: Shape, launch, and mailbox parameters for K3 MoE Lamport.
        kimi_k3_attnres: Launch and shape parameters for Kimi-K3 AttnRes.
    """

    subgroup_size: int
    packed_word_bytes: int
    staged: _StagedAllReduceKernelConfig
    producer_direct: _ProducerDirectAllReduceKernelConfig
    two_stage: _TwoStageAllReduceKernelConfig
    kimi_k3_moe: KimiK3MoeAllReduceKernelConfig
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
    kimi_k3_attnres=KimiK3AttnResKernelConfig(
        world_size=8,
        hidden_size=7168,
        num_subgroups=16,
        elements_per_thread=8,
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
        if moe_tail_max_rows and not (
            _platform.is_cdna4
            and self.world_size == 8
            and dtype == torch.bfloat16
            and moe_tail_max_rows <= 8192
            and moe_tail_max_rows % 8 == 0
            and moe_tail_max_rows * moe_config.row_numel <= producer_direct_max_numel
        ):
            raise ValueError(
                "K3 MoE result requires TP8 BF16 producer capacity on CDNA4"
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
            _MOE_REDUCE_PROGRAMS if moe_tail_max_rows else 0,
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
                + 2 * self.world_size * attnres_max_numel
                + (staged_max_numel if self._staged_two_stage_supported else 0)
                + self._staged_two_stage_scratch_numel
                + moe_config.lamport_stages
                * self.world_size
                * self._kimi_k3_moe_lamport_max_numel
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
            # Preserve the base heap's headroom for other collective states.
            if moe_tail_max_rows:
                heap_size += (
                    moe_tail_max_rows * moe_config.hidden_size * dtype.itemsize
                    + _MOE_GATHER_PROGRAMS * self.world_size * torch.int32.itemsize
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
        # Reserve the reusable result separately from producer and collective
        # scratch before cache sizing and graph capture.
        self._moe_tail_output_buf = (
            self._ctx.empty((moe_tail_max_rows, moe_config.hidden_size), dtype=dtype)
            if moe_tail_max_rows
            else None
        )
        self._moe_tail_ready_flags = (
            self._ctx.zeros((_MOE_GATHER_PROGRAMS, self.world_size), dtype=torch.int32)
            if moe_tail_max_rows
            else None
        )
        self._attnres_push_inbox = (
            self._ctx.zeros((2, self.world_size, attnres_max_numel), dtype=dtype)
            if attnres_max_numel
            else None
        )
        self._attnres_push_epochs = (
            torch.zeros((attnres_max_rows,), dtype=torch.int32, device=self.device)
            if attnres_max_numel
            else None
        )
        self._attnres_push_ready_flags = (
            self._ctx.zeros((2, attnres_max_rows, self.world_size), dtype=torch.int32)
            if attnres_max_numel
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
        if self._attnres_push_inbox is not None:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
        heap_bases = self._ctx.get_heap_bases()
        self._group_heap_bases = heap_bases[group_ranks].contiguous()
        group_heap_bases = [int(address) for address in self._group_heap_bases.tolist()]
        self._heap_base_addresses = tuple(
            group_heap_bases + [group_heap_bases[-1]] * (8 - self.world_size)
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
        self._attnres_push_peer_inboxes = (
            _peer_addresses(
                self._attnres_push_inbox,
                self._heap_base_addresses,
                rank_in_group,
            )
            if self._attnres_push_inbox is not None
            else None
        )
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Iris all-reduce symmetric-heap buffers allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
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

        Unaligned destinations use a temporary and copy-back so every rank
        keeps the same collective protocol while the kernel uses packed stores.

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
        output = tensor.view(-1)
        copy_output = output.data_ptr() % self._kernel_config.packed_word_bytes != 0
        if copy_output:
            output = torch.empty_like(output)
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
        if copy_output:
            tensor.view(-1).copy_(output)
        return tensor.clone() if safe else tensor

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
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            TOTAL_ELEMENTS=total_numel,
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
                output,
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
        assert self._attnres_push_inbox is not None
        assert self._attnres_push_epochs is not None
        assert self._attnres_push_ready_flags is not None
        assert self._attnres_push_peer_inboxes is not None
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
        iris_push_one_shot_allreduce_residual_attnres_gluon_kernel[(num_tokens,)](
            partial,
            residual,
            self._attnres_push_inbox,
            score_weight,
            output_weight,
            m,
            s_,
            acc,
            hidden,
            residual_out,
            self._attnres_push_epochs,
            self._attnres_push_ready_flags,
            *self._attnres_push_peer_inboxes,
            *self._heap_base_addresses,
            RANK=self._iris_rank,
            WORLD_SIZE=self.world_size,
            HIDDEN=kernel_config.hidden_size,
            BLOCK=triton.next_power_of_2(kernel_config.hidden_size),
            MAX_ELEMENTS=self.attnres_max_numel,
            READY_SLOT_STRIDE=self.attnres_max_rows * self.world_size,
            EPS=eps,
            ELEMENTS_PER_THREAD=kernel_config.elements_per_thread,
            NUM_WARPS=kernel_config.num_subgroups,
            SUBGROUP_SIZE=self._kernel_config.subgroup_size,
            num_warps=kernel_config.num_subgroups,
        )
        return hidden, residual_out


# CDNA4 collectives over consecutive token rows assigned by rank.


@gluon.jit
def _row_partition_store_completion(
    flags,
    peer_flags,
    block_id,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # Finish every subgroup's write-through stores before publishing completion.
    _iris_drain_subgroup_vmem()
    gl.barrier()
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    remote = peers != RANK
    gl.store(peer_flags + block_id * 8 + RANK, epoch, mask=remote, cache_modifier=".wt")
    local = flags + block_id * 8 + peers
    seen = gl.load(local, mask=remote, other=epoch, cache_modifier=".cv", volatile=True)
    while gl.sum((remote & ((seen - epoch).to(gl.int32) < 0)).to(gl.int32), 0) != 0:
        seen = gl.load(
            local, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
        )
    gl.atomic_add(local, 0, mask=remote, sem="acquire", scope="sys")


@gluon.jit
def _row_partition_entry_barrier(
    flags,
    peer_flags,
    block_id,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # Prior producers complete on the calling stream before this entry barrier.
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    remote = peers != RANK
    gl.atomic_xchg(
        peer_flags + block_id * 8 + RANK,
        epoch,
        mask=remote,
        sem="release",
        scope="sys",
    )
    local_flags = flags + block_id * 8 + peers
    seen = gl.load(
        local_flags, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
    )
    # Accept newer epochs across uint32 wraparound; peers stay within one call.
    while gl.sum((remote & ((seen - epoch).to(gl.int32) < 0)).to(gl.int32), 0) != 0:
        seen = gl.load(
            local_flags, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
        )
    # Acquire lowering already joins the workgroup after cache invalidation.
    gl.atomic_add(local_flags, 0, mask=remote, sem="acquire", scope="sys")


@gluon.jit
def _peer_buffers(pointer, heaps, RANK: gl.constexpr):
    offset = pointer.to(gl.uint64) - heaps[RANK]
    result = ()
    for peer in gl.static_range(8):
        result += (
            gl.multiple_of((heaps[peer] + offset).to(gl.pointer_type(gl.bfloat16)), 16),
        )
    return result


@gluon.jit
def _peer_flags(pointer, heaps, RANK: gl.constexpr, NUM_WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    bases = gl.full((8,), 0, gl.uint64, layout)
    for peer in gl.static_range(8):
        bases = gl.where(peers == peer, heaps[peer], bases)
    offset = pointer.to(gl.uint64) - heaps[RANK]
    return (bases + offset).to(gl.pointer_type(gl.uint32))


# Rank q owns L = ROWS/8 consecutive rows. For local row u and coordinate j,
# r = q*L + u, reduce that row from all eight producer ranks:
#
#   scratch_routed_q[u,j] = BF16(sum_p FP32(routed_partial_p[r,j]))
#   scratch_shared_q[u,j] = BF16(sum_p FP32(shared_partial_p[r,j]))
#
# The sum uses the kernel's fixed FP32 tree. Each peer's input packs all
# routed rows before all shared rows; scratch packs only q's reduced rows
# in that order.
@gluon.jit(do_not_specialize=["ROWS"])
def iris_moe_reduce_scatter_gluon_kernel(
    input_ptr,
    scratch_ptr,
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
    ROWS,
    FIRST_WIDTH: gl.constexpr,
    SECOND_WIDTH: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    inputs = _peer_buffers(input_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peer_flags = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    block_id = gl.program_id(0)
    epoch = (
        gl.atomic_add(flags + block_id * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    )
    _row_partition_entry_barrier(flags, peer_flags, block_id, epoch, RANK, NUM_WARPS)
    FIRST_ELEMENTS = ROWS // 8 * FIRST_WIDTH
    SECOND_ELEMENTS = ROWS // 8 * SECOND_WIDTH
    PARTITION_ELEMENTS = FIRST_ELEMENTS + SECOND_ELEMENTS
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(
        block_id, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS
    ):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        source = gl.where(
            offsets < FIRST_ELEMENTS,
            RANK * FIRST_ELEMENTS + offsets,
            ROWS * FIRST_WIDTH + RANK * SECOND_ELEMENTS + offsets - FIRST_ELEMENTS,
        )
        values = ()
        for step in gl.static_range(8):
            values += (
                gl.amd.cdna4.buffer_load(
                    inputs[(RANK + step) % 8], source, mask, 0, cache=".cg"
                ),
            )
        sums = ()
        for peer in gl.static_range(2):
            sums += (
                values[(peer - RANK) % 8].to(gl.float32)
                + values[(peer + 2 - RANK) % 8].to(gl.float32),
                values[(peer + 4 - RANK) % 8].to(gl.float32)
                + values[(peer + 6 - RANK) % 8].to(gl.float32),
            )
        reduced = ((sums[0] + sums[1]) + (sums[2] + sums[3])).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(reduced, scratch_ptr, offsets, mask, cache=".wt")
    # The matching gather waits for every rank after these reads complete.
    # Its completion must precede the next producer's symmetric-input writes.


# Rank q owns L = M/8 consecutive rows. For local row u, r = q*L + u,
# set i = r for a replicated prefix or i = u when PREFIX_IS_SHARDED.
# For every destination rank p:
#
#   output_p[r,j] = BF16((FP32(prefix_q[i,j])
#                         + FP32(projected_q[u,j]))
#                         + FP32(shared_reduced_q[u,j]))
#
# Rank q pushes its rows to every peer; other ranks write disjoint rows.
@gluon.jit(do_not_specialize=["LOCAL_ROWS"])
def iris_moe_add_push_gather_gluon_kernel(
    projection_ptr,
    shared_ptr,
    prefix_ptr,
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PREFIX_IS_SHARDED: gl.constexpr,
):
    # Preserve row alignment for vector loads/stores without specializing M.
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
    # In-place prefixes are safe: ranks read then write disjoint rows.
    # Reduce-scatter entry waits for prior prefix consumers.
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    outputs = _peer_buffers(output_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peer_flags = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    block_id = gl.program_id(0)
    epoch = (
        gl.atomic_add(flags + block_id * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    )
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(
        block_id, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS
    ):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        prefix_offsets = offsets
        if not PREFIX_IS_SHARDED:
            prefix_offsets += RANK * PARTITION_ELEMENTS
        a = gl.amd.cdna4.buffer_load(
            prefix_ptr, prefix_offsets, mask, 0, cache=".cg"
        ).to(gl.float32)
        b = gl.amd.cdna4.buffer_load(projection_ptr, offsets, mask, 0, cache=".cg").to(
            gl.float32
        )
        c = gl.amd.cdna4.buffer_load(shared_ptr, offsets, mask, 0, cache=".cg").to(
            gl.float32
        )
        result = (a + b + c).to(gl.bfloat16)
        # Precomputed peer bases keep stores consecutive without VMEM drains.
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                result,
                outputs[(RANK + step) % 8],
                offsets + RANK * PARTITION_ELEMENTS,
                mask,
                cache=".wt",
            )
    _row_partition_store_completion(flags, peer_flags, block_id, epoch, RANK, NUM_WARPS)


def _reduce_metadata(grid, kernel, args):
    elements = args["LOCAL_ROWS"] * 7168
    return {
        "name": kernel.name,
        "bytes": elements * (9 + int(args["HAS_RESIDUAL"])) * 2,
        "flops32": elements * (7 + int(args["HAS_RESIDUAL"])),
    }


def _gather_metadata(grid, kernel, args):
    return {"name": kernel.name, "bytes": args["LOCAL_ROWS"] * 7168 * 9 * 2}


def _mix_gather_metadata(grid, kernel, args):
    return {
        "name": kernel.name,
        "bytes": args["LOCAL_ROWS"]
        * 7168
        * (args["NUM_VALID_BLOCKS"] + 10 + 2 * int(args["NUM_VALID_BLOCKS"] > 0))
        * 2,
    }


# Rank q owns L = M/8 consecutive rows. For local row u, r = q*L + u,
# reduce that row from all eight attention producers:
#
#   reduced_q[u,j] = BF16(sum_p FP32(partial_p[r,j]))
#   prefix_q[u,j] = reduced_q[u,j]                         (no residual)
#                 = BF16(FP32(reduced_q[u,j])
#                      + FP32(residual_q[r,j]))            (with residual)
#
# The sum uses the even/odd FP32 tree. BF16 rounding precedes the residual
# add; only rank q's rows are stored in its local prefix.
@gluon.jit(launch_metadata=_reduce_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_attention_reduce_scatter_gluon_kernel(
    input_ptr,
    residual_ptr,
    prefix_ptr,
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
):
    # Whole rows preserve vector alignment without specializing the row count.
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    inputs = _peer_buffers(input_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    _row_partition_entry_barrier(flags, peers, pid, epoch, RANK, NUM_WARPS)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        source = RANK * PARTITION_ELEMENTS + offsets
        values = ()
        for step in gl.static_range(8):
            values += (
                gl.amd.cdna4.buffer_load(
                    inputs[(RANK + step) % 8], source, mask, 0, cache=".cg"
                ),
            )
        # Preserve Iris's even/odd FP32 tree and the BF16 boundary before add.
        even = (
            values[(0 - RANK) % 8].to(gl.float32)
            + values[(2 - RANK) % 8].to(gl.float32)
        ) + (
            values[(4 - RANK) % 8].to(gl.float32)
            + values[(6 - RANK) % 8].to(gl.float32)
        )
        odd = (
            values[(1 - RANK) % 8].to(gl.float32)
            + values[(3 - RANK) % 8].to(gl.float32)
        ) + (
            values[(5 - RANK) % 8].to(gl.float32)
            + values[(7 - RANK) % 8].to(gl.float32)
        )
        prefix = (even + odd).to(gl.bfloat16)
        if HAS_RESIDUAL:
            residual = gl.amd.cdna4.buffer_load(
                residual_ptr, source, mask, 0, cache=".ca"
            )
            prefix = (prefix.to(gl.float32) + residual.to(gl.float32)).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(prefix, prefix_ptr, offsets, mask, cache=".wb")
    # Only local storage was written. The gather's completion orders every
    # rank's input reads before the next producer reuses the symmetric input.


# Rank q has mixed its L = M/8 local rows. For u in [0,L), r = q*L + u,
# and every destination rank p:
#
#   output_p[r,j] = mixed_q[u,j]
#
# Each rank pushes its rows to every peer; other ranks write disjoint rows.
@gluon.jit(launch_metadata=_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_attention_push_gather_gluon_kernel(
    mixed_ptr,
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    outputs = _peer_buffers(output_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        value = gl.amd.cdna4.buffer_load(mixed_ptr, offsets, mask, 0, cache=".cg")
        # Scalar peer bases let all eight stores issue without intervening drains.
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                value,
                outputs[(RANK + step) % 8],
                RANK * PARTITION_ELEMENTS + offsets,
                mask,
                cache=".wt",
            )
    _row_partition_store_completion(flags, peers, pid, epoch, RANK, NUM_WARPS)


# Rank q owns local prefix row u for global row r = q*L + u, L = M/8.
# Mix that prefix with the first NUM_VALID_BLOCKS history candidates at r,
# round to BF16, apply output RMSNorm, and round to BF16 again. For every peer p:
#
#   output_p[r,j] = mixed_q[u,j]
#
# Each rank pushes its rows to every peer; other ranks write disjoint rows.
@gluon.jit(launch_metadata=_mix_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_attention_mix_push_gluon_kernel(
    prefix_ptr,
    output_ptr,
    block_residual,
    res_weight,
    rms_weight,
    out_norm_weight,
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
    LOCAL_ROWS,
    STRIDE_BLOCK_T: gl.constexpr,
    STRIDE_BLOCK_N,
    NUM_VALID_BLOCKS: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    outputs = _peer_buffers(output_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    hidden = gl.arange(0, 8192, layout=layout)
    mask = hidden < 7168
    for row in range(pid, LOCAL_ROWS, NUM_PROGRAMS):
        token = RANK * LOCAL_ROWS + row
        prefix = gl.amd.cdna4.buffer_load(
            prefix_ptr, row * 7168 + hidden, mask, 0, cache=".ca"
        )
        mixed = _attn_res_mix_gfx950(
            prefix,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight,
            token,
            hidden,
            mask,
            STRIDE_BLOCK_T,
            STRIDE_BLOCK_N,
            7168,
            NUM_VALID_BLOCKS + 1,
            SCORE_EPS,
            OUTPUT_EPS,
        )
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                mixed,
                outputs[(RANK + step) % 8],
                token * 7168 + hidden,
                mask,
                cache=".wt",
            )
    _row_partition_store_completion(flags, peers, pid, epoch, RANK, NUM_WARPS)


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
def _iris_sanitize_lamport_bf16(values):
    bits = values.to(gl.uint16, bitcast=True)
    return gl.where(bits == 0x8000, 0, bits).to(gl.bfloat16, bitcast=True)


@gluon.jit
def _iris_wait_lamport_peers(
    region,
    generation,
    offsets,
    valid,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    values = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        values += (gl.full([64, 8], 0, gl.bfloat16, LAYOUT),)
    active = valid
    while gl.max(active.to(gl.int32), 0) != 0:
        loaded = ()
        # A lane reloads every peer until all seven packs are ready. Issue the
        # independent reads before checking them; retired lanes keep their data.
        # .cv controls hardware caches, not compiler volatility. The compiler
        # regression test checks that these reads remain in the polling cycle.
        for delta in gl.static_range(1, WORLD_SIZE):
            peer = (RANK + delta) % WORLD_SIZE
            pointer = (
                region + generation * WORLD_SIZE * MAX_ELEMENTS + peer * MAX_ELEMENTS
            )
            loaded += (
                gl.amd.cdna4.buffer_load(
                    pointer,
                    offsets,
                    mask=active[:, None],
                    other=values[delta - 1],
                    cache=".cv",
                ),
            )
        active = gl.full([64], False, gl.int1, gl.SliceLayout(1, LAYOUT))
        for delta in gl.static_range(0, WORLD_SIZE - 1):
            active |= valid & (
                gl.max(
                    (loaded[delta].to(gl.uint16, bitcast=True) == 0x8000).to(gl.int32),
                    1,
                )
                != 0
            )
        values = loaded
    return values


@gluon.jit
def lamport_all_reduce_bf16(
    input_sym_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    region_0,
    region_1,
    region_2,
    region_3,
    region_4,
    region_5,
    region_6,
    region_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TOTAL_ELEMENTS: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """Push K3 BF16 tiles and poll all peer packs with one subgroup per tile."""
    gl.static_assert(WORLD_SIZE == 8)
    gl.static_assert(TOTAL_ELEMENTS % 512 == 0)
    gl.static_assert(TOTAL_ELEMENTS <= MAX_ELEMENTS)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * 8 + element[None, :]
    mask = offsets < TOTAL_ELEMENTS
    generation = (
        gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    )
    stride: gl.constexpr = WORLD_SIZE * MAX_ELEMENTS
    local = gl.amd.cdna4.buffer_load(input_sym_ptr, offsets, mask=mask, other=0.0)
    local = _iris_sanitize_lamport_bf16(local)
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = _iris_heap_base(
            peer,
            region_0,
            region_1,
            region_2,
            region_3,
            region_4,
            region_5,
            region_6,
            region_7,
        )
        # Preserve 16-byte publication stores after casting integer addresses.
        destination = gl.multiple_of(destination.to(gl.pointer_type(gl.bfloat16)), 16)
        destination += generation * stride + RANK * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(local, destination, offsets, mask=mask, cache=".wt")

    peers = _iris_wait_lamport_peers(
        region_sym_ptr,
        generation,
        offsets,
        pack * 8 < TOTAL_ELEMENTS,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        layout,
    )
    # All ranks use the same FP32 addition order, then round once to BF16.
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            term = local
        else:
            term = peers[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if peer == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    gl.amd.cdna4.buffer_store(
        accumulator.to(gl.bfloat16), output_ptr, offsets, mask=mask
    )

    # Clear only this tile's consumed generation. Skipped tiles keep their
    # own epochs, so mixed row counts need no global counter or tail clearing.
    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = region_sym_ptr + generation * stride + peer * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(
            sentinel, destination, offsets, mask=mask, cache=".wt"
        )
    gl.barrier()
    gl.store(epochs + gl.program_id(0), ((generation + 1) % NUM_STAGES).to(gl.int32))


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
def _iris_sync_rank_token(
    flags,
    row,
    token,
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
):
    layout: gl.constexpr = gl.BlockedLayout([1], [SUBGROUP_SIZE], [NUM_WARPS], [0])
    peers = gl.arange(0, WORLD_SIZE, layout=layout)
    peer_mask = peers != RANK
    peer_heaps = gl.where(peers == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peers == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peers == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peers == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peers == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peers == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peers == 6, heap_base_6, peer_heaps)
    flags_heap_offset = tl.cast(flags, gl.uint64) - local_heap
    peer_flags = tl.cast(
        peer_heaps + flags_heap_offset,
        gl.pointer_type(gl.int32),
    )
    gl.atomic_xchg(
        peer_flags + row * WORLD_SIZE + RANK,
        token,
        mask=peer_mask,
        sem="release",
        scope="sys",
    )
    local_flags = flags + row * WORLD_SIZE + peers
    seen = gl.load(
        local_flags,
        mask=peer_mask,
        other=token,
        cache_modifier=".cv",
        volatile=True,
    )
    while gl.max(gl.where(peer_mask & (seen != token), 1, 0), axis=0) != 0:
        seen = gl.load(
            local_flags,
            mask=peer_mask,
            other=token,
            cache_modifier=".cv",
            volatile=True,
        )
    gl.atomic_add(
        local_flags,
        0,
        mask=peer_mask,
        sem="acquire",
        scope="sys",
    )
    _iris_drain_subgroup_vmem()
    # The acquire is subgroup-local. Keep every subgroup at the protocol
    # boundary until all of them have observed the peer publications; the
    # caller consumes the peer inbox immediately after this helper returns.
    gl.barrier()


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
    seen = gl.load(
        wait_flags,
        mask=peer_mask,
        other=epoch,
        cache_modifier=".cv",
        volatile=True,
    )
    # Compare modulo 32 bits, including when a peer has passed the wrap before
    # this rank. A zero or negative epoch still requires observing every peer.
    while gl.min((seen - epoch).to(gl.int32), axis=0) < 0:
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
    # The reduced size and the tile/program counts derived from it follow the
    # batch; runtime so every batch shape shares one binary.
    TOTAL_NUMEL,
    BLOCK_SIZE: gl.constexpr,
    NUM_PROGRAMS,
    NUM_TILES,
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
    total_packed = TOTAL_NUMEL // ELEMENTS_PER_WORD
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
    # The partition size and the tile/program counts derived from it follow
    # the batch; runtime so every batch shape shares one binary.
    PARTITION_WORDS,
    BLOCK_WORDS: gl.constexpr,
    NUM_PROGRAMS,
    NUM_TILES,
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
    rank_start = RANK * PARTITION_WORDS

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
def _iris_attnres_epilogue(
    reduced,
    residual_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    row_offsets,
    weight_offsets,
    mask,
    row,
    HIDDEN: gl.constexpr,
    EPS: gl.constexpr,
):
    reduced = reduced.to(gl.bfloat16).to(gl.float32)
    residual = gl.amd.cdna4.buffer_load(
        residual_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    prefix = (reduced + residual).to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        prefix.to(residual_out_ptr.dtype.element_ty),
        residual_out_ptr,
        row_offsets,
        mask=mask,
    )
    score_weight = gl.amd.cdna4.buffer_load(
        score_weight_ptr,
        weight_offsets,
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
        row_offsets,
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
        weight_offsets,
        mask=mask,
        other=0.0,
    ).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        (mixed * inverse_rms * output_weight).to(hidden_ptr.dtype.element_ty),
        hidden_ptr,
        row_offsets,
        mask=mask,
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
    consumed = consumed_flags + row * WORLD_SIZE + RANK
    gl.atomic_xchg(consumed, epoch, sem="release", scope="sys")

    _iris_attnres_epilogue(
        reduced,
        residual_ptr,
        score_weight_ptr,
        output_weight_ptr,
        scratch_m_ptr,
        scratch_s_ptr,
        scratch_acc_ptr,
        hidden_ptr,
        residual_out_ptr,
        offset_i32,
        weight_offset_i32,
        mask,
        row,
        HIDDEN,
        EPS,
    )


@gluon.jit
def iris_push_one_shot_allreduce_residual_attnres_gluon_kernel(
    partial_ptr,
    residual_ptr,
    inbox_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    generations,
    ready_flags,
    inbox_0: gl.pointer_type(gl.bfloat16),
    inbox_1: gl.pointer_type(gl.bfloat16),
    inbox_2: gl.pointer_type(gl.bfloat16),
    inbox_3: gl.pointer_type(gl.bfloat16),
    inbox_4: gl.pointer_type(gl.bfloat16),
    inbox_5: gl.pointer_type(gl.bfloat16),
    inbox_6: gl.pointer_type(gl.bfloat16),
    inbox_7: gl.pointer_type(gl.bfloat16),
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
    MAX_ELEMENTS: gl.constexpr,
    READY_SLOT_STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """Push Kimi-K3 attention rows into two-slot rank-ordered inboxes.

    Peer inboxes are host-computed byte addresses. Explicit BF16 pointer
    annotations preserve their pointer ABI and element-wise device offsets
    without a Python pointer wrapper. The Iris context owns the mappings.
    """
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    element = gl.arange(0, BLOCK, layout=layout)
    mask = element < HIDDEN
    row_offsets = (row * HIDDEN + element).to(gl.int32)
    weight_offsets = element.to(gl.int32)
    local = gl.amd.cdna4.buffer_load(
        partial_ptr,
        row_offsets,
        mask=mask,
        other=0.0,
    )
    generation = gl.load(generations + row).to(gl.int32) + 1
    slot = generation & 1
    inbox_slot_offset = slot * WORLD_SIZE * MAX_ELEMENTS
    sync_ready_flags = ready_flags + slot * READY_SLOT_STRIDE
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
    for peer_delta in gl.static_range(1, WORLD_SIZE):
        destination = (RANK + peer_delta) % WORLD_SIZE
        destination_inbox = _iris_heap_base(
            destination,
            inbox_0,
            inbox_1,
            inbox_2,
            inbox_3,
            inbox_4,
            inbox_5,
            inbox_6,
            inbox_7,
        )
        gl.amd.cdna4.buffer_store(
            local,
            destination_inbox + inbox_slot_offset + RANK * MAX_ELEMENTS,
            row_offsets,
            mask=mask,
            cache=".wt",
        )
    _iris_drain_subgroup_vmem()
    # The drain is subgroup-local. Join all producer subgroups before the
    # control subgroup publishes the generation to peer ranks.
    gl.barrier()

    _iris_sync_rank_token(
        sync_ready_flags,
        row,
        generation,
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
    )
    local_inbox = inbox_sym_ptr + inbox_slot_offset

    if RANK == 0:
        reduced = local.to(gl.float32)
    else:
        reduced = gl.amd.cdna4.buffer_load(
            local_inbox,
            row_offsets,
            mask=mask,
            other=0.0,
            cache=".cg",
        ).to(gl.float32)
    for source in gl.static_range(1, WORLD_SIZE):
        if source == RANK:
            reduced += local.to(gl.float32)
        else:
            reduced += gl.amd.cdna4.buffer_load(
                local_inbox + source * MAX_ELEMENTS,
                row_offsets,
                mask=mask,
                other=0.0,
                cache=".cg",
            ).to(gl.float32)

    gl.store(generations + row, generation)
    _iris_attnres_epilogue(
        reduced,
        residual_ptr,
        score_weight_ptr,
        output_weight_ptr,
        scratch_m_ptr,
        scratch_s_ptr,
        scratch_acc_ptr,
        hidden_ptr,
        residual_out_ptr,
        row_offsets,
        weight_offsets,
        mask,
        row,
        HIDDEN,
        EPS,
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
            "Iris AR+RMSNorm symmetric-heap buffer allocated: "
            f"{free_gpu_memory_begin - free_gpu_memory_after!s} GB",
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


def _find_iris_state(
    group: dist.ProcessGroup, tensors: tuple[torch.Tensor, ...]
) -> IrisAllReduce | None:
    return next(
        (
            state
            for state in IRIS_AR_STATES.values()
            if state.group is group and state.owns_outputs(tensors)
        ),
        None,
    )


def _overlaps(tensor: torch.Tensor, buffer: torch.Tensor) -> bool:
    span = tensor.numel()
    if span == 0:
        return False
    # Collective buffers are contiguous; history can have padding between rows.
    if not tensor.is_contiguous():
        span = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    begin = tensor.data_ptr()
    end = begin + span * tensor.element_size()
    return (
        begin < buffer.data_ptr() + buffer.numel() * buffer.element_size()
        and buffer.data_ptr() < end
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
    """Reduce MoE partials by row, project local rows, and gather the result.

    All eight ranks use identical shapes and call order, each handling M/8 rows.
    Tensor inputs are contiguous BF16; M is positive and divisible by eight.
    The prepared producer inputs are preserved.

    Args:
        routed_partial: Prepared routed partial, ``[M, 3584]``.
        shared_partial: Prepared shared partial, ``[M, 7168]``, immediately
            following ``routed_partial`` in Iris storage.
        prefix: Residual, replicated ``[M, 7168]`` or this rank's
            ``[M/8, 7168]`` rows. A replicated prefix may exactly alias
            the result buffer and is consumed before overwrite; other
            overlaps are rejected.
        projection_weight: Replicated ``[7168, 3584]`` weight.
        prefix_is_sharded: Whether ``prefix`` contains only this rank's rows.
        norm_weight: RMSNorm ``[3584]`` weight, or None.
        eps: Positive RMSNorm epsilon, or None without normalization.
        group: Eight-rank group owning the prepared Iris buffers.

    Returns:
        BF16 ``[M, 7168]`` view of the reusable result buffer, or None
        before launch if unsupported. The next MoE tail or attention mix can
        overwrite it; clone it to retain the value. Calls sharing the state
        must run in order on one stream, including graph capture and replay.
    """
    if not current_platform().is_cdna4 or routed_partial.ndim != 2:
        return None
    rows, latent = routed_partial.shape
    if (
        rows <= 0
        or rows % 8 != 0
        or latent != 3584
        or shared_partial.shape != (rows, 7168)
        or prefix.shape != (rows // 8 if prefix_is_sharded else rows, 7168)
        or projection_weight.shape != (7168, 3584)
        or group.size() != 8
    ):
        return None
    tensors = (routed_partial, shared_partial, prefix, projection_weight)
    if norm_weight is not None:
        if (
            norm_weight.shape != (3584,)
            or eps is None
            or not math.isfinite(eps)
            or eps <= 0
        ):
            return None
        tensors += (norm_weight,)
    elif eps is not None:
        return None
    if any(
        not tensor.is_cuda
        or tensor.device != routed_partial.device
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        for tensor in tensors
    ):
        return None

    # Reuse the state that owns both prepared outputs.
    state = _find_iris_state(group, (routed_partial, shared_partial))
    if state is None:
        return None
    local_rows = rows // 8
    routed_elements = local_rows * 3584
    shared_elements = local_rows * 7168
    scratch = state._producer_direct_scratch_buf
    flags = state._producer_direct_ready_flags
    programs = _MOE_REDUCE_PROGRAMS
    gather_programs = _MOE_GATHER_PROGRAMS
    result_buffer = state._moe_tail_output_buf
    gather_flags = state._moe_tail_ready_flags
    if (
        scratch is None
        or scratch.numel() < routed_elements + shared_elements
        or flags is None
        or flags.shape[0] < programs
        or gather_flags is None
        or gather_flags.shape[0] < gather_programs
        or result_buffer is None
        or result_buffer.shape[0] < rows
    ):
        return None
    # Reject unsafe overlaps before launching either collective.
    for tensor in tensors[2:]:
        for buffer in (state._input_buf, scratch, result_buffer):
            if _overlaps(tensor, buffer):
                # Only exact prefix aliasing preserves row ownership.
                if not (
                    buffer is result_buffer
                    and tensor is prefix
                    and tensor.data_ptr() == buffer.data_ptr()
                    and not prefix_is_sharded
                ):
                    return None

    from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_latent_projection
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

    routed = scratch[:routed_elements].view(local_rows, 3584)
    shared = scratch[routed_elements : routed_elements + shared_elements].view(
        local_rows, 7168
    )
    output = result_buffer[:rows]
    # A sharded prefix is disjoint from the result. Its owner can project into
    # its output rows, then consume them before the gather overwrites them.
    # A replicated prefix can already occupy the result and must be preserved.
    projected = (
        output[
            state.rank_in_group * local_rows : (state.rank_in_group + 1) * local_rows
        ]
        if prefix_is_sharded
        else torch.empty((local_rows, 7168), device=prefix.device, dtype=prefix.dtype)
    )
    iris_moe_reduce_scatter_gluon_kernel[(programs,)](
        state._input_buf,
        scratch,
        flags,
        *state._heap_base_addresses,
        RANK=state.rank_in_group,
        ROWS=rows,
        FIRST_WIDTH=3584,
        SECOND_WIDTH=7168,
        BLOCK_ELEMENTS=2048,
        NUM_PROGRAMS=programs,
        NUM_WARPS=4,
        num_warps=4,
    )
    normalized = (
        rmsnorm(routed, norm_weight, eps, residual=None, out=routed)
        if norm_weight is not None
        else routed
    )
    kimi3_latent_projection(
        normalized, projection_weight, out=projected, solution="auto"
    )
    iris_moe_add_push_gather_gluon_kernel[(gather_programs,)](
        projected,
        shared,
        prefix,
        output,
        gather_flags,
        *state._heap_base_addresses,
        RANK=state.rank_in_group,
        LOCAL_ROWS=local_rows,
        BLOCK_ELEMENTS=2048,
        NUM_PROGRAMS=gather_programs,
        NUM_WARPS=4,
        PREFIX_IS_SHARDED=prefix_is_sharded,
        num_warps=4,
    )
    return output


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
    """Reduce attention partials for this rank's consecutive token rows.

    Gather the normalized AttnRes mix to every rank.

    Args:
        partial: Prepared contiguous CUDA BF16 projection, ``[M, 7168]``.
            M is positive and divisible by eight. The producer is preserved.
        residual: Replicated contiguous BF16 residual, ``[M, 7168]``, or None
            on a block-write layer. A disjoint residual is preserved. An exact
            alias of the prepared MoE result is consumed before that result is
            overwritten; shifted overlaps are unsupported.
        block_residual: Replicated BF16 history, ``[K, M, 7168]``, with a
            contiguous hidden dimension. Padding between tokens/blocks is allowed.
        res_weight: Replicated contiguous BF16 scorer weight, ``[7168]``.
        rms_weight: Replicated contiguous BF16 score RMSNorm weight, ``[7168]``.
        eps: Positive finite score RMSNorm epsilon.
        out_norm_weight: Replicated contiguous BF16 output RMSNorm weight,
            ``[7168]``.
        out_norm_eps: Positive finite output RMSNorm epsilon.
        num_valid_blocks: Number of leading history snapshots to mix, from 0
            through 11. History and weights must not overlap collective storage.
        group: The exact eight-rank group owning the prepared producer. Every
            rank supplies the same shapes, eligibility and operation order.

    Returns:
        This rank's residual rows ``[M/8, 7168]`` in separate storage and
        a replicated activation ``[M, 7168]`` in the reusable result buffer,
        or None before launch if unsupported. Finish using the activation on
        the calling stream before the next attention mix or MoE tail on this
        group; clone it if needed longer. The residual rows remain local until
        the MoE tail consumes them or the fallback gathers them.

    No process groups or symmetric buffers are created. Calls sharing this
    state are serialized on one stream, including capture and replay. Runtime
    policy selects the profitable token window independently of this contract.
    """
    if (
        not current_platform().is_cdna4
        or partial.ndim != 2
        or partial.shape[0] <= 0
        or partial.shape[0] % 8 != 0
        or partial.shape[1] != 7168
        or not partial.is_cuda
        or partial.dtype != torch.bfloat16
        or not partial.is_contiguous()
        or group.size() != 8
        or block_residual.ndim != 3
        or block_residual.shape[1:] != partial.shape
        or block_residual.dtype != partial.dtype
        or block_residual.device != partial.device
        or block_residual.stride(-1) != 1
        or (partial.shape[0] - 1) * block_residual.stride(1) + 7168 >= 1 << 30
        or not isinstance(num_valid_blocks, int)
        or not 0 <= num_valid_blocks <= min(11, block_residual.shape[0])
        or not math.isfinite(eps)
        or eps <= 0
        or not math.isfinite(out_norm_eps)
        or out_norm_eps <= 0
    ):
        return None
    if residual is not None and (
        residual.shape != partial.shape
        or residual.dtype != partial.dtype
        or residual.device != partial.device
        or not residual.is_contiguous()
    ):
        return None
    weights = (res_weight, rms_weight, out_norm_weight)
    if any(
        weight.shape != (7168,)
        or weight.dtype != partial.dtype
        or weight.device != partial.device
        or not weight.is_contiguous()
        for weight in weights
    ):
        return None

    state = _find_iris_state(group, (partial,))
    if state is None:
        return None
    output_buffer = state._moe_tail_output_buf
    flags = state._producer_direct_ready_flags
    gather_flags = state._moe_tail_ready_flags
    if (
        output_buffer is None
        or output_buffer.shape[0] < partial.shape[0]
        or flags is None
        or flags.shape[0] < _MOE_REDUCE_PROGRAMS
        or gather_flags is None
        or gather_flags.shape[0] < _MOE_GATHER_PROGRAMS
    ):
        return None
    protected = (state._input_buf, state._producer_direct_scratch_buf, output_buffer)
    for tensor in (block_residual, *weights):
        if any(
            buffer is not None and _overlaps(tensor, buffer) for buffer in protected
        ):
            return None
    if residual is not None:
        for buffer in protected:
            if (
                buffer is not None
                and _overlaps(residual, buffer)
                and not (
                    buffer is output_buffer and residual.data_ptr() == buffer.data_ptr()
                )
            ):
                return None

    from tokenspeed_kernel.ops.residual import attn_res_fwd, attn_res_fwd_available

    rows = partial.shape[0] // 8
    first_row = state.rank_in_group * rows
    history = block_residual[:, first_row : first_row + rows]
    if not attn_res_fwd_available(
        partial[:rows],
        history,
        res_weight,
        rms_weight,
        eps,
        out_norm_weight=out_norm_weight,
        out_norm_eps=out_norm_eps,
        delta=None,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=-1,
    ):
        return None

    # Resolve the gfx950 helper before either collective publishes an epoch.
    global _attn_res_mix_gfx950
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.attn_res import (
        _attn_res_mix_gfx950,
    )

    prefix = torch.empty_like(partial[:rows])
    output = output_buffer[: partial.shape[0]]
    programs = _MOE_REDUCE_PROGRAMS
    iris_attention_reduce_scatter_gluon_kernel[(programs,)](
        partial,
        residual,
        prefix,
        flags,
        *state._heap_base_addresses,
        RANK=state.rank_in_group,
        LOCAL_ROWS=rows,
        BLOCK_ELEMENTS=2048,
        NUM_PROGRAMS=programs,
        NUM_WARPS=4,
        HAS_RESIDUAL=residual is not None,
        num_warps=4,
    )
    # The ordinary mixer is faster in the middle token range. At larger sizes,
    # longer histories also need enough rows to amortize live peer pointers.
    fuse_mix = (partial.shape[0] < 1024 or partial.shape[0] >= 4096) and (
        num_valid_blocks <= 6 or (partial.shape[0] >= 7680 and num_valid_blocks <= 8)
    )
    if fuse_mix:
        gather_programs = _MOE_GATHER_PROGRAMS
        num_subgroups = 8 if num_valid_blocks <= 7 else 4
        iris_attention_mix_push_gluon_kernel[(gather_programs,)](
            prefix,
            output,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight,
            gather_flags,
            *state._heap_base_addresses,
            RANK=state.rank_in_group,
            LOCAL_ROWS=rows,
            STRIDE_BLOCK_T=block_residual.stride(1),
            STRIDE_BLOCK_N=block_residual.stride(0),
            NUM_VALID_BLOCKS=num_valid_blocks,
            SCORE_EPS=eps,
            OUTPUT_EPS=out_norm_eps,
            NUM_PROGRAMS=gather_programs,
            NUM_WARPS=num_subgroups,
            num_warps=num_subgroups,
        )
    else:
        # Longer histories favor the existing mixer without persistent peer
        # pointers occupying registers throughout the candidate reductions.
        mixed = attn_res_fwd(
            prefix,
            history,
            res_weight,
            rms_weight,
            eps,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
            delta=None,
            num_valid_blocks=num_valid_blocks,
            block_write_idx=-1,
        )
        gather_programs = 32
        iris_attention_push_gather_gluon_kernel[(gather_programs,)](
            mixed,
            output,
            gather_flags,
            *state._heap_base_addresses,
            RANK=state.rank_in_group,
            LOCAL_ROWS=rows,
            BLOCK_ELEMENTS=2048,
            NUM_PROGRAMS=gather_programs,
            NUM_WARPS=4,
            num_warps=4,
        )
    return prefix, output
