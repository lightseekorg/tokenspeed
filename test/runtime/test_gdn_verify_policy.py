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

"""GDN chain verification selects chunking only within its supported geometry."""

import os
import sys
from types import SimpleNamespace

import pytest
from tokenspeed_kernel.ops.attention import gdn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="runtime-1gpu")


@pytest.mark.parametrize("is_blackwell", [False, True])
@pytest.mark.parametrize(
    "num_tokens,num_states,in_chunked_range",
    [
        (3, 8, False),
        (12, 24, False),
        (13, 23, False),
        (13, 24, True),
        (16, 24, True),
        (17, 24, False),
    ],
)
def test_gdn_chain_verify_policy(
    monkeypatch, is_blackwell, num_tokens, num_states, in_chunked_range
):
    monkeypatch.setattr(
        gdn, "current_platform", lambda: SimpleNamespace(is_blackwell=is_blackwell)
    )
    assert gdn.gdn_chain_verify_is_chunked(num_tokens, num_states) == (
        is_blackwell and in_chunked_range
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
