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

from functools import cache

import deep_gemm
import torch
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.tuning import is_autotuning


@triton.jit
def _reduce_split_k_kernel(
    partial,
    out,
    elements,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // N, offsets % N
    value = tl.full((BLOCK,), 0, tl.float32)
    for group in tl.static_range(GROUPS):
        value += tl.load(
            partial + (row * GROUPS + group) * N + col,
            offsets < elements,
            other=0,
        )
    tl.store(out + offsets, value.to(tl.bfloat16), offsets < elements)


class _SmallMRunner(TunableRunner):
    def __init__(self, *, enable_pdl: bool, num_sms: int):
        self.enable_pdl = enable_pdl
        self.num_sms = num_sms

    def get_valid_tactics(self, inputs, profile):
        # DeepGEMM's SM100 descriptor requires an even SM budget, including
        # when evaluating single-SM layouts. N/128 estimates the narrow O grid.
        tiles = (inputs[1].shape[0] + 127) // 128
        candidates = {self.num_sms, self.num_sms - 2}
        candidates.update(
            min(self.num_sms, (tiles * factor + 1) // 2 * 2) for factor in (1, 2, 4)
        )
        grids = sorted(value for value in candidates if value > 0)
        k = inputs[0].shape[1]
        # Negative tactics select split-K; -1 remains the native default.
        # A scale word must belong to one group so all operand views alias
        # the original FP8 codes and packed scales without repacking.
        splits = [
            -groups
            for groups in (2, 3, 4, 6, 8)
            if k % (groups * 4 * 128) == 0 and k // groups >= 512
        ]
        return grids + splits

    def get_cache_key_extras(self, inputs):
        return (
            "deepgemm-small-m-v4",
            128,
            self.enable_pdl,
            self.num_sms,
            deep_gemm.get_tc_util(),
            torch.cuda.get_device_capability(inputs[0].device),
            tuple(tuple(t.stride()) for t in inputs),
        )

    def forward(self, inputs, tactic=-1, do_preparation=False):
        a, b, a_scale, b_scale, out = inputs
        old_sms = deep_gemm.get_num_sms()
        selected_sms = self.num_sms if tactic < 0 else tactic
        deep_gemm.set_pdl(self.enable_pdl)
        try:
            if selected_sms != old_sms:
                deep_gemm.set_num_sms(selected_sms)
            if tactic < -1:
                groups = -tactic
                m, k = a.shape
                n = b.shape[0]
                scale_columns = a_scale.shape[1] // groups
                # Preserve the MN-major stride even for M=1: reshape can
                # assign a different stride to that singleton dimension.
                sa = a_scale.as_strided(
                    (m, groups, scale_columns),
                    (
                        a_scale.stride(0),
                        scale_columns * a_scale.stride(1),
                        a_scale.stride(1),
                    ),
                )
                sb = b_scale.as_strided(
                    (groups, n, scale_columns),
                    (
                        scale_columns * b_scale.stride(1),
                        b_scale.stride(0),
                        b_scale.stride(1),
                    ),
                )
                partial = torch.empty(
                    (m, groups, n), dtype=torch.float32, device=a.device
                )
                deep_gemm.fp8_einsum(
                    "bhr,hdr->bhd",
                    (a.view(m, groups, k // groups), sa),
                    (b.view(n, groups, k // groups).permute(1, 0, 2), sb),
                    partial,
                    recipe=(1, 1, 128),
                )
                _reduce_split_k_kernel[(triton.cdiv(m * n, 256),)](
                    partial,
                    out,
                    m * n,
                    N=n,
                    GROUPS=groups,
                    ENABLE_PDL=self.enable_pdl,
                    BLOCK=256,
                    num_warps=4,
                    **({"launch_pdl": True} if self.enable_pdl else {}),
                )
                return out
            deep_gemm.fp8_gemm_nt(
                (a, a_scale),
                (b, b_scale),
                out,
                recipe_a=(1, 128),
                recipe_b=(1, 128),
            )
        finally:
            if selected_sms != old_sms:
                deep_gemm.set_num_sms(old_sms)
        return out


_TUNING = TuningConfig(
    use_cold_l2_cache=True, use_cuda_graph=True, cuda_graph_profile_replays=1
)


@cache
def _runner(enable_pdl: bool, num_sms: int):
    return _SmallMRunner(enable_pdl=enable_pdl, num_sms=num_sms)


def _run(inputs, runner):
    tuner = AutoTuner.get()
    chosen, tactic = tuner.choose_one(
        "tokenspeed_deepgemm_small_m", [runner], _TUNING, inputs
    )
    return chosen(inputs, tactic=tactic)


def block_fp8_mm_tuned(a, b, a_scale, b_scale, out, *, enable_pdl):
    """Select native grids or batched split-K for narrow, long-K projections.

    Args:
        a: E4M3 activations [M,K].
        b: E4M3 weights [N,K].
        a_scale: Packed UE8M0 activation scales with MN-major aligned strides.
        b_scale: Packed UE8M0 weight scales.
        out: BF16 output [M,N], written in place.
        enable_pdl: Whether the native launch uses PDL.

    Returns:
        The supplied output. Split-K aliases the original operands and reduces
        FP32 partials directly to BF16. Startup profiles M=1/2/4/8/16/32;
        larger M retains the native heuristic. Capture only queries the
        persistent FlashInfer cache, never profiles tactics.
    """
    runner = _runner(enable_pdl, deep_gemm.get_num_sms())
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing and is_autotuning() and a.shape[0] > 32:
        # Exact shape entries avoid resizing packed scale layouts through a
        # generic dynamic-M tensor specification. Each entry is shared across
        # layers; only a cache miss profiles it.
        blocks = (a.shape[1] // 128 + 3) // 4
        for rows in (1, 2, 4, 8, 16, 32):
            scratch_a = a[:rows]
            padded_rows = (rows + 3) // 4 * 4
            scratch_scale = torch.full(
                (blocks, padded_rows),
                0x7F7F7F7F,
                dtype=torch.int32,
                device=a.device,
            ).t()[:rows]
            scratch_scale.copy_(a_scale[:rows])
            scratch_out = torch.empty(
                (rows, b.shape[0]), dtype=out.dtype, device=a.device
            )
            _run([scratch_a, b, scratch_scale, b_scale, scratch_out], runner)
    if a.shape[0] > 32:
        return runner([a, b, a_scale, b_scale, out], tactic=-1)
    if capturing:
        # FI tuning contexts are reference-counted: autotune(False) cannot
        # disable an outer autotune(True) during graph capture.
        tuner = AutoTuner.get()
        with tuner._lock:
            was_tuning = tuner.is_tuning_mode
            tuner.is_tuning_mode = False
            try:
                return _run([a, b, a_scale, b_scale, out], runner)
            finally:
                tuner.is_tuning_mode = was_tuning
    return _run([a, b, a_scale, b_scale, out], runner)
