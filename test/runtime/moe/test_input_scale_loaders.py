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

"""Per-expert activation scale loading: scalar checkpoints keep the global max."""

import torch

from tokenspeed.runtime.layers.moe.weights.loaders import load_per_expert_input_scale


def test_scalar_per_expert_scales_accumulate_the_global_max() -> None:
    # Checkpoints such as Kimi K2.5 NVFP4 ship one scalar input scale per
    # expert, loaded one call at a time with local_expert_id=None.
    param = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    for scale in (0.5, 0.9, 0.7, 0.8):
        load_per_expert_input_scale(param, torch.tensor(scale), "w2", None)
    assert torch.equal(param.data, torch.tensor([0.9, 0.9, 0.9, 0.9]))


def test_fused_per_expert_scales_copy_per_expert() -> None:
    param = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    load_per_expert_input_scale(param, torch.tensor([0.1, 0.4, 0.2, 0.3]), "w2", None)
    assert torch.equal(param.data, torch.tensor([0.1, 0.4, 0.2, 0.3]))


def test_local_expert_scale_writes_only_that_expert() -> None:
    param = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    load_per_expert_input_scale(param, torch.tensor(0.6), "w2", 2)
    assert torch.equal(param.data, torch.tensor([0.0, 0.0, 0.6, 0.0]))
