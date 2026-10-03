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

"""``--yarn-ramp-mask-device``: where deepseek_yarn computes its inv_freq table."""

from __future__ import annotations

import math

import pytest
import torch

import tokenspeed.runtime.layers.rotary_embedding as rotary_module
from tokenspeed.runtime.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    get_rope,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

ROTARY_DIM = 64
BASE = 10000
ORIGINAL_MAX_POSITION = 4096
SCALING_FACTOR = 40.0


def _cpu_reference_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001
    return torch.clamp(
        (torch.arange(dim, dtype=torch.float) - low) / (high - low), 0, 1
    )


def _cpu_reference_inv_freq() -> torch.Tensor:
    pos_freqs = BASE ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float) / ROTARY_DIM)
    low, high = _yarn_find_correction_range(
        32, 1, ROTARY_DIM, BASE, ORIGINAL_MAX_POSITION
    )
    mask = 1 - _cpu_reference_ramp(low, high, ROTARY_DIM // 2)
    return (1.0 / (SCALING_FACTOR * pos_freqs)) * (1 - mask) + (1.0 / pos_freqs) * mask


def test_cpu_ramp_matches_the_pure_cpu_reference():
    low, high = 3, 29
    ramp = _yarn_linear_ramp_mask(low, high, ROTARY_DIM // 2, torch.float, device="cpu")
    assert torch.equal(ramp, _cpu_reference_ramp(low, high, ROTARY_DIM // 2))
    # The degenerate range is widened, not divided by zero.
    flat = _yarn_linear_ramp_mask(5, 5, 8, torch.float, device=None)
    assert torch.isfinite(flat).all()


def _rope(device: str, ramp_device: str) -> DeepseekScalingRotaryEmbedding:
    return DeepseekScalingRotaryEmbedding(
        ROTARY_DIM,
        ROTARY_DIM,
        ORIGINAL_MAX_POSITION,
        BASE,
        False,
        SCALING_FACTOR,
        torch.bfloat16,
        device=device,
        ramp_device=ramp_device,
    )


def test_deepseek_yarn_cpu_ramp_reproduces_the_cpu_inv_freq():
    rope = _rope(device="cpu", ramp_device="cpu")
    assert rope.ramp_device == "cpu"
    assert torch.equal(
        rope._compute_inv_freq(SCALING_FACTOR), _cpu_reference_inv_freq()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_model_device_takes_the_whole_table_from_the_ramp_device():
    # Under ramp_device=cpu the entire inv_freq (position frequencies, both
    # divisions, ramp) is the pure-CPU table, moved to the model device once:
    # bitwise equal to the CPU reference, not merely close.
    cpu_built = _rope(device="cuda", ramp_device="cpu")._compute_inv_freq(
        SCALING_FACTOR
    )
    assert cpu_built.device.type == "cuda"
    assert torch.equal(cpu_built.cpu(), _cpu_reference_inv_freq())
    cuda_built = _rope(device="cuda", ramp_device="cuda")._compute_inv_freq(
        SCALING_FACTOR
    )
    assert cuda_built.device.type == "cuda"
    assert cuda_built.shape == cpu_built.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_get_rope_reads_the_switch_and_keys_its_cache_on_it(monkeypatch):
    rope_scaling = {
        "rope_type": "deepseek_yarn",
        "factor": SCALING_FACTOR,
        "original_max_position_embeddings": ORIGINAL_MAX_POSITION,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    monkeypatch.setattr(rotary_module, "_ROPE_DICT", {})
    monkeypatch.setitem(global_server_args_dict, "yarn_ramp_mask_device", "cpu")
    cpu_rope = get_rope(
        ROTARY_DIM, ROTARY_DIM, 8192, BASE, False, dict(rope_scaling), torch.bfloat16
    )
    assert isinstance(cpu_rope, DeepseekScalingRotaryEmbedding)
    assert cpu_rope.ramp_device == "cpu"
    assert cpu_rope is get_rope(
        ROTARY_DIM, ROTARY_DIM, 8192, BASE, False, dict(rope_scaling), torch.bfloat16
    )
    monkeypatch.setitem(global_server_args_dict, "yarn_ramp_mask_device", "cuda")
    cuda_rope = get_rope(
        ROTARY_DIM, ROTARY_DIM, 8192, BASE, False, dict(rope_scaling), torch.bfloat16
    )
    assert cuda_rope is not cpu_rope
    assert cuda_rope.ramp_device == "cuda"
    # Same inv_freq up to the ulp-level division difference the switch exists for.
    cpu_freq = cpu_rope._compute_inv_freq(SCALING_FACTOR).cpu()
    cuda_freq = cuda_rope._compute_inv_freq(SCALING_FACTOR).cpu()
    assert torch.allclose(cpu_freq, cuda_freq, rtol=1e-6, atol=0)
    assert math.isclose(float(cpu_freq[0]), 1.0, rel_tol=1e-6)
