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

"""GDN chain verification picks its form within the measured geometry."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.attention import gdn
from tokenspeed_kernel.ops.attention.gdn import GdnChainVerifyForm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="runtime-1gpu")

STEP, TREE, CHAIN = (
    GdnChainVerifyForm.STEP,
    GdnChainVerifyForm.TREE,
    GdnChainVerifyForm.CHAIN,
)


@pytest.mark.parametrize("is_blackwell", [False, True])
@pytest.mark.parametrize(
    "num_tokens,num_states,state_dtype,chain_form,form",
    [
        # bf16 state where the caller runs the chain form: 2 to 16 tokens.
        (1, 12, torch.bfloat16, True, STEP),
        (2, 12, torch.bfloat16, True, CHAIN),
        (4, 6144, torch.bfloat16, True, CHAIN),
        (13, 24, torch.bfloat16, True, CHAIN),
        (16, 384, torch.bfloat16, True, CHAIN),
        (17, 6144, torch.bfloat16, True, STEP),
        # Without the chain form (or with fp32 state) the tree form keeps its range.
        (3, 6144, torch.bfloat16, False, STEP),
        (12, 24, torch.bfloat16, False, STEP),
        (13, 23, torch.bfloat16, False, STEP),
        (13, 24, torch.bfloat16, False, TREE),
        (16, 6144, torch.bfloat16, False, TREE),
        (3, 6144, torch.float16, True, STEP),
        (3, 6144, torch.float32, True, STEP),
        (13, 24, torch.float32, True, TREE),
        (17, 24, torch.float32, True, STEP),
    ],
)
def test_gdn_chain_verify_policy(
    monkeypatch, is_blackwell, num_tokens, num_states, state_dtype, chain_form, form
):
    monkeypatch.setattr(
        gdn, "current_platform", lambda: SimpleNamespace(is_blackwell=is_blackwell)
    )
    picked = gdn.gdn_chain_verify_form(
        num_tokens, num_states, state_dtype, chain_form=chain_form
    )
    assert picked is (form if is_blackwell else STEP)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
