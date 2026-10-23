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

"""The Qwen3.8-Flash-Next MTP draft front (norms, linears and the stream add
in one launch) against the model's eager math."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.megakernel import (
    qwen4_exp_mtp_front,
    select_qwen4_exp_mtp_front,
)
from tokenspeed_kernel.ops.megakernel.triton import _qwen4_exp_mtp_front_kernel
from utils import assert_no_triton_compile

H, VOCAB, EPS = 2560, 1000, 1e-6

if not torch.cuda.is_available():
    pytest.skip("a GPU is required", allow_module_level=True)


def _operands(rows: int, seed: int):
    g = torch.Generator().manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).bfloat16().cuda()

    return dict(
        input_ids=torch.randint(0, VOCAB, (rows,), generator=g).int().cuda(),
        embedding=randn(VOCAB, H),
        hidden=randn(rows, 4 * H, scale=3.0),
        norm_embedding=randn(H, scale=0.1),
        norm_hidden=randn(4 * H, scale=0.1),
        fc_embedding=randn(H, H, scale=0.02),
        fc_hidden=randn(H, H, scale=0.02),
    )


def _norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Gemma RMSNorm in the eager reference's order: FP32 rows times rsqrt,
    times the FP32 1 + weight, one BF16 cast."""
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (x * (1.0 + weight.float())).bfloat16()


def _reference(ops) -> torch.Tensor:
    """``Qwen4ExpForCausalLMNextN._fuse_inputs``."""
    embedded = torch.nn.functional.linear(
        _norm(ops["embedding"][ops["input_ids"].long()], ops["norm_embedding"]),
        ops["fc_embedding"],
    )
    hidden = _norm(ops["hidden"], ops["norm_hidden"]).unflatten(-1, (4, H))
    hidden = torch.nn.functional.linear(hidden, ops["fc_hidden"])
    return (hidden + embedded.unsqueeze(-2)).flatten(-2)


def _run(kernel, ops) -> torch.Tensor:
    out = torch.empty_like(ops["hidden"])
    qwen4_exp_mtp_front(kernel, **ops, eps=EPS, out=out)
    return out


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_mtp_front_matches_the_model_math(rows):
    kernel = select_qwen4_exp_mtp_front(4)
    ops = _operands(rows, rows)
    got, ref = _run(kernel, ops), _reference(ops)
    # Only the linears' FP32 summation order differs. A linear's output can
    # then round one BF16 step apart, and the add can cancel, so the bound is
    # one BF16 step (2**-8 relative) of the largest output, and few elements
    # differ at all.
    step = 2**-8 * ref.float().abs().max().item()
    torch.testing.assert_close(got.float(), ref.float(), rtol=0, atol=step)
    assert (got != ref).float().mean().item() < 1e-3


def test_rows_equal_one_row_launches():
    kernel = select_qwen4_exp_mtp_front(4)
    ops = _operands(4, 7)
    together = _run(kernel, ops)
    for r in range(4):
        one = dict(ops, input_ids=ops["input_ids"][r : r + 1])
        one["hidden"] = ops["hidden"][r : r + 1]
        assert torch.equal(_run(kernel, one)[0], together[r])


def test_row_count_does_not_recompile():
    kernel = select_qwen4_exp_mtp_front(4)
    _run(kernel, _operands(1, 0))
    with assert_no_triton_compile(_qwen4_exp_mtp_front_kernel):
        for rows in (2, 3, 4, 1):
            _run(kernel, _operands(rows, rows))
