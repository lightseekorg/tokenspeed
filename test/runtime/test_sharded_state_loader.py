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


"""Rank-local checkpoint loading preserves every saved parameter."""

import os
import sys
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.model_loader import loader


def test_sharded_state_loads_only_its_rank_and_combines_parts(tmp_path, monkeypatch):
    references = []
    for rank in range(2):
        reference = nn.Linear(4, 2)
        with torch.no_grad():
            reference.weight.copy_(torch.arange(8).view(2, 4) + rank * 10)
            reference.bias.fill_(rank + 1)
        references.append(reference)
        for part, (name, tensor) in enumerate(reference.state_dict().items()):
            save_file(
                {name: tensor},
                str(tmp_path / f"model-rank-{rank}-part-{part}.safetensors"),
            )

    monkeypatch.setattr(loader, "_initialize_model", lambda *args: nn.Linear(4, 2))
    checkpoint_loader = loader.ShardedStateLoader(
        LoadConfig(load_format="sharded_state")
    )
    inputs = torch.arange(12, dtype=torch.float32).view(3, 4)
    for rank, reference in enumerate(references):
        restored = checkpoint_loader.load_model(
            model_config=SimpleNamespace(
                model_path=str(tmp_path),
                revision=None,
                dtype=torch.float32,
                mapping=Mapping(rank=rank, world_size=2),
            ),
            device_config=SimpleNamespace(device="cpu"),
        )
        for name, expected in reference.state_dict().items():
            assert torch.equal(restored.state_dict()[name], expected)
        assert torch.equal(restored(inputs), reference(inputs))
