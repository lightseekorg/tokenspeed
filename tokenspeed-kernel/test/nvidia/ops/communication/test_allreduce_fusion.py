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

"""Public fused-all-reduce input contracts."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

api = importlib.import_module("tokenspeed_kernel.ops.communication.allreduce_fusion")


def inputs():
    backend = SimpleNamespace(device=torch.device("cpu"))
    kernel = SimpleNamespace(
        impl=Mock(return_value=torch.ones(2, 8, dtype=torch.bfloat16))
    )
    workspace = api.AllReduceFusionWorkspace(backend, kernel, 8, 2, 2048)
    x = torch.ones(4, 8, dtype=torch.bfloat16)
    gamma = torch.ones(8, dtype=torch.bfloat16)
    weights = torch.ones(4, dtype=torch.bfloat16)
    indices = torch.arange(4, dtype=torch.int32)
    return workspace, x, gamma, weights, indices


@pytest.mark.parametrize("finalize", [False, True])
def test_input_patterns_preserve_route_metadata(finalize):
    workspace, x, gamma, weights, indices = inputs()
    if not finalize:
        x, weights, indices = x[:2], None, None
    pattern = (
        api.AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM
        if finalize
        else api.AllReduceFusionPattern.ALLREDUCE_RMSNORM
    )
    api.allreduce_fusion(
        x,
        workspace,
        pattern=pattern,
        rms_gamma=gamma,
        num_tokens=2,
        expert_weights=weights,
        expanded_idx_to_permuted_idx=indices,
    )
    args = workspace.kernel.impl.call_args.args
    assert args[2] is pattern
    if finalize:
        torch.testing.assert_close(args[5], weights.view(2, 2), rtol=0, atol=0)
        torch.testing.assert_close(args[6], indices.view(2, 2), rtol=0, atol=0)
    else:
        assert args[5:] == (None, None)


@pytest.mark.parametrize("finalize", [False, True])
def test_route_metadata_must_match_the_input_pattern(finalize):
    workspace, x, gamma, weights, indices = inputs()
    if finalize:
        weights, indices = None, None
    else:
        x = x[:2]
    with pytest.raises(ValueError):
        api.allreduce_fusion(
            x,
            workspace,
            pattern=(
                api.AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM
                if finalize
                else api.AllReduceFusionPattern.ALLREDUCE_RMSNORM
            ),
            rms_gamma=gamma,
            num_tokens=2,
            expert_weights=weights,
            expanded_idx_to_permuted_idx=indices,
        )
    workspace.kernel.impl.assert_not_called()
