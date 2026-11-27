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

"""Small CPU weights and runner state shared by disk-reload tests."""

from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.weight_loader import WeightLoader


class _TinyModel(torch.nn.Module):
    """A real module with a real ``load_weights``, small enough for CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4, 3))
        self.b = torch.nn.Parameter(torch.zeros(4))
        self.seen: list[str] = []

    def load_weights(self, weights) -> None:
        params = dict(self.named_parameters())
        for name, tensor in weights:
            self.seen.append(name)
            params[name].data.copy_(tensor)


def _write_checkpoint(directory: Path, scale: float) -> dict[str, torch.Tensor]:
    """A one-shard safetensors checkpoint, the shape a trainer writes."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        "w": torch.arange(12, dtype=torch.float32).reshape(4, 3) * scale,
        "b": torch.arange(4, dtype=torch.float32) * scale,
    }
    save_file(tensors, str(directory / "model.safetensors"))
    return tensors


def _runner(model: torch.nn.Module, booted_from: Path) -> ModelRunner:
    """A ModelRunner with only what the disk reload touches (no GPU load)."""
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = model
    runner.device = "cpu"
    runner.gpu_id = 0
    runner.checkpoint_load_group = None
    # Resident weights (no peer sharing), so disk reloads are allowed.
    runner.mapping = Mapping(rank=0)
    runner.model_config = SimpleNamespace(
        model_path=str(booted_from), mapping=runner.mapping, revision=None
    )
    runner.server_args = SimpleNamespace(
        enable_memory_saver=False,
        weight_memory_policy=None,
        load_format="auto",
        download_dir=None,
        model_loader_extra_config={},
        ext_yaml=None,
        weight_loader_prefetch_checkpoints=False,
        weight_loader_prefetch_num_threads=1,
    )
    # Some CPU fixtures initialize weights in memory, without a startup checkpoint.
    runner._disk_reload_shapes = WeightLoader.checkpoint_shapes(
        runner.model_config, model
    )
    return runner
