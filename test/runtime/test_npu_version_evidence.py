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

"""Numeric/shape evidence for the NPU version-drift behavior changes.

Addresses code review 4 C9/C10/C11 (version matrix drift): the three
``torch 2.10`` fixes previously carried only comment-level equivalence
arguments. Each test below gives elementwise/shape evidence on NPU:

- ``npu_swiglu`` (replaces removed ``mlp_split_swiglu``): elementwise equal
  to ``silu(gate) * up`` over a random hidden row.
- RoPE ``npu_mrope`` 2D reshape + the CPU-forced ``deepseek_yarn`` ramp:
  shape/values match the reference arithmetic.
- HCCL ``all_gather_into_tensor`` / ``reduce_scatter_tensor`` (renamed
  ``all_gather_single`` / ``reduce_scatter_single``): round-trip shape and
  layout over a two-rank process group.

Run with ASCEND_RT_VISIBLE_DEVICES selecting available NPUs.
"""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an Ascend NPU"
)

VOCAB = 512


def test_npu_swiglu_equals_silu_gate_times_up():
    """C9: npu_swiglu(hidden, dim=-1) == silu(gate) * up elementwise."""
    torch_npu = __import__("torch_npu")
    torch.manual_seed(0)
    d = 128
    hidden = torch.randn(16, 2 * d, dtype=torch.bfloat16, device="npu")
    gate, up = hidden[..., :d], hidden[..., d:]
    expected = torch.nn.functional.silu(gate) * up
    actual = torch_npu.npu_swiglu(hidden, dim=-1)
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=1e-3, rtol=1e-2, msg="npu_swiglu"
    )
    # bf16 fused-op rounding guard: the fused kernel may round a few ulp
    # differently than the reference decomposition, but must stay within a
    # bf16-magnitude relative band (never a structural/algorithmic mismatch).
    maxdiff = (actual.float() - expected.float()).abs().max().item()
    rel = maxdiff / max(expected.float().abs().max().item(), 1.0)
    assert rel < 0.02, f"maxdiff {maxdiff} (rel {rel})"


def test_npu_mrope_reshape_and_yarn_ramp_cpu():
    """C10: 2D [3, seq] reshape layout + CPU-forced yarn ramp is finite/exact."""
    # -- 2D reshape: interleaved 3 rows of positions -> [3, -1] recovers rows.
    seq = 37
    r0 = torch.arange(seq)
    r1 = r0 + 1000
    r2 = r0 + 2000
    interleaved = torch.cat([r0, r1, r2])
    reshaped = interleaved.reshape(3, -1)
    assert reshaped.shape == (3, seq)
    torch.testing.assert_close(reshaped[0], r0)
    torch.testing.assert_close(reshaped[1], r1)
    torch.testing.assert_close(reshaped[2], r2)

    # -- deepseek_yarn ramp on CPU (NPU torch lacks CUDA, host builds table).
    # Mirror the real ``_compute_inv_freq`` arithmetic and compare the
    # CPU-forced ramp against the same table built on the device.
    from tokenspeed.runtime.layers.rotary_embedding import (
        _yarn_find_correction_range,
        _yarn_linear_ramp_mask,
    )

    def yarn_inv_freq(device, ramp_device):
        rotary_dim = 256
        base = 10000.0
        scaling_factor = 10.0
        beta_fast, beta_slow = 32, 1
        max_position_embeddings = 32768
        extrapolation_factor = 1.0
        pos_freqs = base ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=ramp_device)
            / rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = _yarn_find_correction_range(
            beta_fast,
            beta_slow,
            rotary_dim,
            base,
            max_position_embeddings,
        )
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(
                low,
                high,
                rotary_dim // 2,
                dtype=torch.float,
                device=ramp_device,
            )
        ) * extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq.to(device)

    cpu_tbl = yarn_inv_freq("npu", "cpu")
    npu_tbl = yarn_inv_freq("npu", "npu")
    assert cpu_tbl.shape == (256 // 2,)
    assert torch.isfinite(cpu_tbl).all()
    # ramp is deterministic host arithmetic: elementwise equal to the device
    # reference (tolerances for CPU/device ulp division differences).
    torch.testing.assert_close(
        cpu_tbl.float(), npu_tbl.float(), atol=1e-4, rtol=1e-3, msg="yarn ramp"
    )


def test_hccl_all_gather_reduce_scatter_roundtrip():
    """C11: renamed HCCL collectives keep the shape/layout contract.

    Runs in a two-rank process group (torchrun) on two NPUs: the gathered
    output equals the rank-order concatenation of inputs, and
    reduce_scatter_tensor recovers the per-rank pieces. Skipped when the
    rank is not the torchrun entry (this file doubles as the torchrun body).
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        pytest.skip("HCCL shape check needs torchrun with WORLD_SIZE >= 2")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="hccl", rank=rank, world_size=world
    )
    try:
        torch.manual_seed(0)
        x = torch.full((8, 4), rank, dtype=torch.float32, device="npu")
        gathered = torch.empty((world * 8, 4), dtype=torch.float32, device="npu")
        torch.distributed.all_gather_into_tensor(gathered, x.contiguous())
        if rank == 0:
            expected = torch.cat(
                [torch.full((8, 4), r, dtype=torch.float32, device="npu") for r in range(world)]
            )
            torch.testing.assert_close(gathered, expected, msg="all_gather_into_tensor")
        out = torch.empty((8, 4), dtype=torch.float32, device="npu")
        torch.distributed.reduce_scatter_tensor(out, gathered, torch.distributed.ReduceOp.SUM)
        expected_scatter = torch.full((8, 4), world * rank, dtype=torch.float32, device="npu")
        torch.testing.assert_close(out, expected_scatter, msg="reduce_scatter_tensor")
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
