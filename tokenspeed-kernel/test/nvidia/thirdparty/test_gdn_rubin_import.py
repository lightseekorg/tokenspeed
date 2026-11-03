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

"""A vendor adapter can execute before the op registry is imported."""

import subprocess
import sys
import textwrap

import pytest
from tokenspeed_kernel.ops.attention.gdn.flashinfer import (
    is_rubin_decode_available as is_available,
)


@pytest.mark.skipif(
    not is_available(), reason="SM107 and FlashInfer PR6604 are required"
)
def test_rubin_adapter_executes_before_registry_import():
    program = textwrap.dedent("""
        from tokenspeed_kernel.thirdparty.flashinfer import gdn_rubin
        import torch

        q = torch.zeros(1, 1, 4, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.zeros(1, 1, 12, 128, device="cuda", dtype=torch.bfloat16)
        gate = torch.zeros(1, 1, 12, device="cuda", dtype=torch.bfloat16)
        parameter = torch.zeros(12, device="cuda")
        state = torch.ones(1, 12, 128, 128, device="cuda", dtype=torch.bfloat16)
        rows = torch.zeros(1, device="cuda", dtype=torch.int32)
        out = gdn_rubin.gated_delta_rule_t1_rubin(
            enable_pdl=False,
            q=q, k=q, v=v,
            a=gate, b=gate, A_log=parameter, dt_bias=parameter,
            initial_state_source=state, initial_state_indices=rows,
            output_state_indices=rows, scale=128**-0.5,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
        torch.testing.assert_close(state, torch.full_like(state, 0.5), atol=0, rtol=0)
        """)
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
