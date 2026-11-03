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

"""SM107 prefill Up GEMM plus four-branch Mix, with load-time weight packing."""

import threading
from functools import cache

import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils.hardware_info import HardwareInfo
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_up_mix import HcUpMixKernel

    _AVAILABLE = True
except (ImportError, NotImplementedError):
    _AVAILABLE = False

_PLANS = {}
_PLAN_LOCK = threading.Lock()


def supports_up_mix(device: torch.device) -> bool:
    """Whether CuTe DSL and the SM107 target required by this kernel are available."""
    return (
        _AVAILABLE
        and device.type == "cuda"
        and current_platform().is_nvidia
        and torch.cuda.get_device_capability(device) == (10, 7)
    )


def prepare_up_mix_weight(
    weight: torch.Tensor, hc_count: int, hidden_size: int, lowrank: int
) -> torch.Tensor | None:
    """Pack the Up weight for this Up+Mix kernel, or return None if unsupported.

    The model owns this derived weight and must refresh it after weight updates.
    Forward still requires the canonical weight for small batches and fallbacks.
    """
    if (
        (hc_count, hidden_size, lowrank) != (4, 2560, 320)
        or weight.dtype not in (torch.bfloat16, torch.float16)
        or not supports_up_mix(weight.device)
    ):
        return None
    # Every N256 tile contains four adjacent N64 branch planes.
    return (
        weight.reshape(4, 40, 64, 320)
        .permute(1, 0, 2, 3)
        .reshape(10240, 320)
        .contiguous()
    )


@cache
def _cluster_capacity(index: int, stream: int) -> int:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm HC Up+Mix on this stream before CUDA graph capture")
    with torch.cuda.device(index):
        return HardwareInfo(index).get_max_active_clusters(2, CUstream(stream))


def _operands(activated, packed, normalized, out):
    tensors = tuple(
        from_dlpack(t.detach().unsqueeze(-1), assumed_align=16)
        for t in (activated, packed, normalized, out)
    )
    return tuple(
        t if i == 1 else t.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2))
        for i, t in enumerate(tensors)
    )


if _AVAILABLE:

    @register_kernel(
        "residual",
        "hyperconnection_mix",
        name="cute_dsl_prefill_hyperconnection_mix",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 7),
            max_arch_version=ArchVersion(10, 7),
        ),
        signatures=format_signatures(
            ("normalized", "projection_weight", "up_weight"),
            "dense",
            {torch.bfloat16, torch.float16},
        ),
        traits={
            "prefill_up_mix": frozenset({True}),
            "hc_count": frozenset({4}),
            "hidden_size": frozenset({2560}),
            "lowrank": frozenset({320}),
            "contiguous": frozenset({True}),
            "fused_tma_aligned": frozenset({True}),
            "weights_independent": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_prefill_hyperconnection_mix(
        normalized: torch.Tensor,
        projection_weight: torch.Tensor,
        up_weight: torch.Tensor,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        projection_rows: int,
        projection_scale: float,
        weights_independent: bool,
        packed_up_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Keep down/SiLU unchanged and fuse Up plus Mix for large batches.

        Arguments and returns follow gated_residual_mix. A load-time packed
        copy of up_weight is required. Token counts and tail masks are runtime;
        one plan serves all row counts for a device, dtype, PDL and capacity.
        """
        from tokenspeed_kernel.ops.residual.triton import _launch_projection_epilogue

        rows = normalized.shape[0]
        tensors = (normalized, projection_weight, up_weight, packed_up_weight)
        if (
            not supports_up_mix(normalized.device)
            or packed_up_weight is None
            or (hc_count, hidden_size, lowrank) != (4, 2560, 320)
            or rows < 4096
            or not weights_independent
            or any(not t.is_contiguous() or t.data_ptr() % 16 for t in tensors)
        ):
            raise ValueError(
                "prefill Up+Mix requires SM107, T>=4096, HC4/H2560/R320 and aligned packed weights"
            )
        with torch.cuda.device(normalized.device):
            projected = F.linear(normalized, projection_weight)[:, :projection_rows]
            activated, inject = _launch_projection_epilogue(
                projected, lowrank, hc_count, projection_scale
            )
            out = normalized.new_empty((rows, hidden_size))
            stream = int(torch.cuda.current_stream(normalized.device).cuda_stream)
            capacity = _cluster_capacity(normalized.device.index, stream)
            enable_pdl = pdl_enabled()
            key = (normalized.device.index, normalized.dtype, enable_pdl, capacity)
            operands = _operands(activated, packed_up_weight, normalized, out)
            with _PLAN_LOCK:
                plan = _PLANS.get(key)
                if plan is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError("warm HC Up+Mix before CUDA graph capture")
                    plan = cute.compile(
                        HcUpMixKernel(use_pdl=enable_pdl),
                        *operands,
                        capacity,
                        CUstream(stream),
                        options="--gpu-arch=sm_107a --opt-level 2"
                    )
                    _PLANS[key] = plan
            plan(*operands, CUstream(stream))
            return out, inject
