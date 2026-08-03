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

import logging
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


class RoutingMethodType(IntEnum):
    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5
    SigmoidRenorm = 6
    MiniMax2 = 7
    Unspecified = 8


class All2AllBackend(Enum):

    NONE = "none"
    DEEPEP = "deepep"
    FLASHINFER_NVLINK_ONE_SIDED = "flashinfer_nvlink_one_sided"
    # AMD-native EP dispatch/combine via MORI (github.com/ROCm/mori). IntraNode
    # (XGMI) + InterNode (RDMA) kernels; real all-to-all, unlike the masked-replicate
    # `none` fallback.
    MORI = "mori"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        for member in cls:
            if value == member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_none(self):
        return self == All2AllBackend.NONE

    def is_deepep(self):
        return self == All2AllBackend.DEEPEP

    def is_flashinfer_nvlink_one_sided(self):
        return self == All2AllBackend.FLASHINFER_NVLINK_ONE_SIDED

    def is_mori(self):
        return self == All2AllBackend.MORI

    def is_all_to_all(self):
        """True for backends that perform a real dispatch/combine all-to-all inside the
        MoE kernel (as opposed to the masked-replicate ``none`` fallback). Such backends
        return the COMPLETE per-token result and must run through the model's all-to-all
        MoE forward (local/dp-sharded tokens, no framework MoE collective)."""
        return self in (All2AllBackend.DEEPEP, All2AllBackend.MORI)


class MoeBackend(Enum):

    AUTO = "auto"
    TRITON = "triton"
    GLUON = "gluon"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"

    DEEP_GEMM = "deep_gemm"
    DEEP_GEMM_MEGA_MOE = "deep_gemm_mega_moe"
    MEGA_MOE = "mega_moe"

    def is_auto(self):
        return self == MoeBackend.AUTO

    def is_triton(self):
        return self == MoeBackend.TRITON

    def is_gluon(self):
        return self == MoeBackend.GLUON

    def is_flashinfer_trtllm(self):
        return self == MoeBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutlass(self):
        return self == MoeBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self):
        return self == MoeBackend.FLASHINFER_CUTEDSL

    def is_deep_gemm(self):
        return self == MoeBackend.DEEP_GEMM

    def is_deep_gemm_mega_moe(self):
        return self in (MoeBackend.DEEP_GEMM_MEGA_MOE, MoeBackend.MEGA_MOE)

    def is_mega_moe(self):
        return self.is_deep_gemm_mega_moe()


class DeepEPMode(Enum):
    """Which DeepEP dispatch/combine legs a deployment may use.

    ``AUTO`` allocates both and lets every forward choose through
    :func:`use_deepep_low_latency`.
    """

    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self) -> bool:
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self) -> bool:
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def is_normal(self) -> bool:
        return self == DeepEPMode.NORMAL

    def is_low_latency(self) -> bool:
        return self == DeepEPMode.LOW_LATENCY

    def is_auto(self) -> bool:
        return self == DeepEPMode.AUTO


def use_deepep_low_latency(ctx: ForwardContext, attn_dp_size: int) -> bool:
    """Whether this forward should take DeepEP's low-latency legs.

    The low-latency dispatch is bounded by a preallocated per-GPU capacity
    (``--low-latency-max-num-tokens-per-gpu``), so only decode-shaped forwards
    fit; extend-shaped ones need the normal legs. The two are separate
    collectives, so every rank of the EP group must reach the same answer.

    Without DP attention the whole EP group forwards one batch, so the local
    ``forward_mode`` already is that shared answer. With DP attention each rank
    has its own batch and its own mode, and the only agreed-upon value is the
    replicated ``all_decode_or_idle`` flag -- so a single extending rank moves
    the entire group onto the normal legs.

    Args:
        ctx: Forward context for the current forward.
        attn_dp_size: Attention data-parallel degree, i.e. whether per-rank
            forward modes can diverge at all.

    Returns:
        True for the low-latency legs, False for the normal legs.
    """
    # A pinned mode leaves no choice: only AUTO allocates both legs, and a
    # low_latency-only deployment has no normal buffers to fall back to.
    mode = get_deepep_mode()
    if mode is DeepEPMode.LOW_LATENCY:
        return True
    if mode is DeepEPMode.NORMAL:
        return False
    if attn_dp_size > 1:
        return ctx.all_decode_or_idle
    return ctx.forward_mode is not None and ctx.forward_mode.is_decode_or_idle()


ALL2ALL_BACKEND: All2AllBackend | None = None
MOE_BACKEND: MoeBackend | None = None
DEEPEP_MODE: DeepEPMode | None = None
DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER: bool | None = None


def initialize_moe_config(server_args: ServerArgs):
    global ALL2ALL_BACKEND
    global MOE_BACKEND
    global DEEPEP_MODE
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER

    ALL2ALL_BACKEND = All2AllBackend(server_args.all2all_backend)
    MOE_BACKEND = MoeBackend(server_args.moe_backend)
    DEEPEP_MODE = DeepEPMode(server_args.deepep_mode)
    DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER = (
        server_args.disable_flashinfer_cutlass_moe_fp4_allgather
    )


def get_all2all_backend() -> All2AllBackend:
    global ALL2ALL_BACKEND
    if ALL2ALL_BACKEND is None:
        logger.warning("ALL2ALL_BACKEND is not initialized, using default backend")
        ALL2ALL_BACKEND = All2AllBackend.NONE
    return ALL2ALL_BACKEND


def get_moe_backend() -> MoeBackend:
    global MOE_BACKEND
    if MOE_BACKEND is None:
        logger.warning("MOE_BACKEND is not initialized, using auto backend")
        MOE_BACKEND = MoeBackend.AUTO
    return MOE_BACKEND


def get_deepep_mode() -> DeepEPMode:
    global DEEPEP_MODE
    if DEEPEP_MODE is None:
        logger.warning("DEEPEP_MODE is not initialized, using auto mode")
        DEEPEP_MODE = DeepEPMode.AUTO
    return DEEPEP_MODE
