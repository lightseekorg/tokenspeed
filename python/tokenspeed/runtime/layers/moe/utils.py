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


class MoeBackend(Enum):

    AUTO = "auto"
    TRITON = "triton"
    TRITON_KERNEL = "triton_kernel"
    MARLIN = "marlin"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    DEEP_GEMM_MEGA_MOE = "deep_gemm_mega_moe"
    MEGA_MOE = "mega_moe"

    def is_auto(self):
        return self == MoeBackend.AUTO

    def is_triton(self):
        return self == MoeBackend.TRITON

    def is_triton_kernel(self):
        return self == MoeBackend.TRITON_KERNEL

    def is_marlin(self):
        return self == MoeBackend.MARLIN

    def is_flashinfer_trtllm(self):
        return self == MoeBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutlass(self):
        return self == MoeBackend.FLASHINFER_CUTLASS

    def is_flashinfer_cutedsl(self):
        return self == MoeBackend.FLASHINFER_CUTEDSL

    def is_flashinfer_mxfp4(self):
        return self == MoeBackend.FLASHINFER_MXFP4

    def is_deep_gemm_mega_moe(self):
        return self in (MoeBackend.DEEP_GEMM_MEGA_MOE, MoeBackend.MEGA_MOE)

    def is_mega_moe(self):
        return self.is_deep_gemm_mega_moe()


class DeepEPMode(Enum):

    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self) -> bool:
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self) -> bool:
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY

    def is_normal(self) -> bool:
        return self == DeepEPMode.NORMAL

    def is_low_latency(self) -> bool:
        return self == DeepEPMode.LOW_LATENCY

    def is_auto(self) -> bool:
        return self == DeepEPMode.AUTO


ALL2ALL_BACKEND: All2AllBackend | None = None
MOE_BACKEND: MoeBackend | None = None
DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER: bool | None = None


def initialize_moe_config(server_args: ServerArgs):
    global ALL2ALL_BACKEND
    global MOE_BACKEND
    global DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER

    ALL2ALL_BACKEND = All2AllBackend(server_args.all2all_backend)
    MOE_BACKEND = MoeBackend(server_args.moe_backend)
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
