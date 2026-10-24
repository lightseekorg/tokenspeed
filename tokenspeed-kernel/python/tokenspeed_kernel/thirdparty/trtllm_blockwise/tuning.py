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


"""Startup tactic selection using TokenSpeed's shared FlashInfer tuning cache."""

from __future__ import annotations

import functools
import threading

import torch
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)
from tokenspeed_kernel.compile_monitor import is_serving
from tokenspeed_kernel.thirdparty.trtllm_blockwise import _arguments

_compiled = {}
_lock = threading.Lock()


@functools.cache
def _tactics():
    import cutlass
    from tokenspeed_kernel.thirdparty.trtllm_blockwise._kernel import (
        Sm100BlockwiseGemmKernel,
    )

    return tuple(
        (tile, cm, cn, two)
        for two in (False, True)
        for tile in (64, 128, 256)
        for cm, cn in (
            (1, 1),
            (1, 2),
            (1, 4),
            (2, 1),
            (2, 2),
            (2, 4),
            (4, 1),
            (4, 2),
            (4, 4),
        )
        if Sm100BlockwiseGemmKernel.can_implement(
            cutlass.Float8E4M3FN,
            cutlass.Float32,
            cutlass.BFloat16,
            two,
            (tile, 128),
            (cm, cn),
            256,
            128,
            128,
            1,
            "k",
            "k",
            "n",
        )
    )


def prepare_tactics(device: torch.device) -> None:
    """Compile dynamic-shape tactics before startup tuning or graph capture."""
    if device.index in _compiled:
        return
    with torch.cuda.device(device), _lock:
        if device.index in _compiled:
            return
        if torch.cuda.is_current_stream_capturing() or is_serving():
            raise RuntimeError(
                "Prepare block-FP8 tactics before serving or graph capture"
            )
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.runtime import make_fake_stream
        from tokenspeed_kernel.thirdparty.trtllm_blockwise._kernel import (
            Sm100BlockwiseGemmKernel,
        )

        a = torch.empty((256, 128), device=device, dtype=torch.float8_e4m3fn)
        b = torch.empty((128, 128), device=device, dtype=torch.float8_e4m3fn)
        sa = torch.empty((1, 256), device=device, dtype=torch.float32)
        sb = torch.empty((1, 1), device=device, dtype=torch.float32)
        out = torch.empty((256, 128), device=device, dtype=torch.bfloat16)
        compiled = {}
        for tactic in _tactics():
            tile, cm, cn, two = tactic
            kernel = Sm100BlockwiseGemmKernel(
                cutlass.Float32, two, (tile, 128), (cm, cn)
            )
            active = utils.HardwareInfo().get_max_active_clusters(cm * cn)
            compiled[tactic] = cute.compile(
                kernel.wrapper,
                *_arguments(a, b, sa, sb, out),
                active,
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options="--opt-level 2 --enable-tvm-ffi",
            )
        _compiled[device.index] = compiled


class ExistingBlockFp8Runner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [0]

    def forward(self, inputs, tactic, **kwargs):
        if set(kwargs) - {"do_preparation"}:
            raise TypeError(f"Unsupported runner arguments: {tuple(kwargs)}")
        from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale
        from tokenspeed_kernel.ops.gemm.triton import triton_mm_fp8_blockscale

        a, b, sa, sb, out = inputs
        kernel = (
            triton_mm_fp8_blockscale
            if 17 <= a.shape[0] <= 32
            else flashinfer_mm_fp8_blockscale
        )
        return kernel(
            a, b, sa, sb, torch.bfloat16, alpha=None, block_size=[128, 128], out=out
        )


class CuteBlockFp8Runner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return list(_tactics())

    def forward(self, inputs, tactic, **kwargs):
        if set(kwargs) - {"do_preparation"}:
            raise TypeError(f"Unsupported runner arguments: {tuple(kwargs)}")
        from cuda.bindings.driver import CUstream

        a, b, sa, sb, out = inputs
        prepare_tactics(a.device)
        if tactic == -1:
            m = a.shape[0]
            tactic = (
                (64, 1, 1, False)
                if m <= 64
                else ((128, 1, 1, False) if m <= 128 else (256, 2, 1, True))
            )
        _compiled[a.device.index][tuple(tactic)](
            *_arguments(a, b, sa.T.contiguous(), sb.contiguous(), out),
            CUstream(torch.cuda.current_stream(a.device).cuda_stream),
        )
        return out


@functools.cache
def _configuration(maximum: int):
    buckets = (1, 2, 4, 8, 16, 32, *range(64, ((maximum + 63) // 64) * 64 + 1, 64))
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec((0, 2, 4), (0, 0, 0), buckets, _bucket),
        ),
        use_cold_l2_cache=True,
    )


def _bucket(m: int) -> int:
    if m <= 32:
        return 1 << (max(m, 1) - 1).bit_length()
    return ((m + 63) // 64) * 64


_existing = ExistingBlockFp8Runner()
_cute = CuteBlockFp8Runner()


def tuned_blockwise_gemm(a, b, a_scales, b_scales, out, *, allow_existing: bool):
    """Select a cached startup tactic for aligned FP8 inputs and BF16 output.

    ``allow_existing`` includes the existing CUTLASS/Triton path when true;
    false restricts selection to CuTe DSL. Scales retain their FP32 values.
    """
    if a.shape[0] == 0:
        return out
    from tokenspeed_kernel.ops.tuning import get_autotune_max_rows

    inputs = [a, b, a_scales, b_scales, out]
    if (
        allow_existing
        and a.device.index not in _compiled
        and (torch.cuda.is_current_stream_capturing() or is_serving())
    ):
        return _existing(inputs, tactic=-1)
    prepare_tactics(a.device)
    runners = [_existing, _cute] if allow_existing else [_cute]
    runner, tactic = AutoTuner.get().choose_one(
        "tokenspeed::block_fp8_gemm",
        runners,
        _configuration(get_autotune_max_rows()),
        inputs,
    )
    return runner(inputs, tactic=tactic)
