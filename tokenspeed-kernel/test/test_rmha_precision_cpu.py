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

"""CPU checks of RMHA selection and the real decode-combine Triton program."""

import os
import subprocess
import sys

import pytest
import torch
from tokenspeed_kernel.ops.attention.rmha import rel_mha_plan
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@pytest.mark.parametrize(
    "mode",
    ["rel_mha_prefill", "rel_mha_extend_with_kvcache", "rel_mha_decode_with_kvcache"],
)
def test_fp32_output_selects_a_kernel_that_retains_precision(mode: str) -> None:
    roles = (
        ("q", "k", "v") if mode == "rel_mha_prefill" else ("q", "k_cache", "v_cache")
    )
    signature = format_signature(
        **{name: dense_tensor_format(torch.bfloat16) for name in roles}
    )
    selected = select_kernel(
        "attention",
        mode,
        signature,
        traits={
            "head_dim": 128,
            "page_size": 128,
            "return_lse": True,
            "sliding_window": False,
            "fp32_output": True,
        },
        solution=None,
        override=None,
    )
    spec = next(
        spec
        for spec in KernelRegistry.get().get_for_operator(
            "attention", mode, platform=current_platform(), format_signature=signature
        )
        if spec.name == selected.name
    )
    assert True in spec.traits["fp32_output"]


def test_plan_rejects_undeclared_output_precision() -> None:
    with pytest.raises(ValueError, match="None or torch.float32"):
        rel_mha_plan(torch.bfloat16, 128, output_dtype=torch.float16)


@pytest.mark.parametrize(
    "output_dtype,return_lse",
    [("float32", True), ("bfloat16", True), ("bfloat16", False)],
)
def test_decode_combine_precision_and_empty_rows_on_cpu(
    output_dtype: str, return_lse: bool
) -> None:
    # TRITON_INTERPRET must be set before import, hence the isolated CPU-only process.
    program = r"""
import torch
from tokenspeed_kernel.ops.attention.rmha.triton import _rel_mha_decode_stage2_kernel
mid = torch.tensor([[[[1.001, 4.002], [6.003, 9.004]],
                     [[2.005, 5.006], [7.007, 8.008]]],
                    [[[float('nan'), float('nan')], [float('nan'), float('nan')]],
                     [[float('nan'), float('nan')], [float('nan'), float('nan')]]]], dtype=torch.float32)
mid_lse = torch.tensor([[[0., 1.], [-1000., -998.]],
                        [[float('nan'), float('nan')], [float('nan'), float('nan')]]], dtype=torch.float32)
out = torch.empty(2, 2, 2, dtype=DTYPE)
lse = torch.empty(2, 2, dtype=torch.float32) if RETURN_LSE else None
seq = torch.tensor([64, 0], dtype=torch.int32)
splits = 2  # the uniform split count
_rel_mha_decode_stage2_kernel[(2, 2)](
    mid, mid_lse, out, lse, seq, splits, *mid.stride()[:3], out.stride(0), out.stride(1),
    MAX_SEQLEN_Q=1, MAX_KV_SPLITS=2, MIN_BLOCK_KV=32, BLOCK_DV=2,
    Lv=2, WINDOW_LEFT=-1, HAS_LSE=RETURN_LSE, ENABLE_PDL=False, num_warps=4,
)
expected = torch.zeros(2, 2, 2, dtype=torch.float32)
expected[0] = (mid[0] * torch.softmax(mid_lse[0], dim=-1)[..., None]).sum(dim=1)
torch.testing.assert_close(out, expected.to(DTYPE), atol=1e-6, rtol=1e-6)
assert torch.count_nonzero(out[1]) == 0
if RETURN_LSE:
    torch.testing.assert_close(lse[0], torch.logsumexp(mid_lse[0], dim=-1), atol=1e-6, rtol=1e-6)
    assert torch.isneginf(lse[1]).all()
if DTYPE == torch.float32:
    assert not torch.equal(out[0], out[0].to(torch.bfloat16).float())
"""
    program = (
        f"DTYPE = __import__('torch').{output_dtype}\nRETURN_LSE = {return_lse!r}\n"
        + program
    )
    environment = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
