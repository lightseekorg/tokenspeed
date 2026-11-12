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

"""Draft loading can override the target format without changing target weights."""

import argparse
import json
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import ModelConfig  # noqa: E402
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.weight_loader import WeightLoader  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402
from tokenspeed.runtime.utils.torch_memory_saver_adapter import (  # noqa: E402
    TorchMemorySaverAdapter,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def checkpoint(tmp_path):
    # A layerless draft exercises the real model and disk loader cheaply.
    config = dict(
        architectures=["DFlashDraftModel"],
        model_type="qwen3",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        block_size=8,
        dflash_config={"target_layer_ids": [0, 1]},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


def load(checkpoint, *, load_format, draft_load_format, is_draft):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    cli = [
        "--model",
        str(checkpoint),
        "--load-format",
        load_format,
        "--dtype",
        "float32",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        str(checkpoint),
    ]
    if draft_load_format is not None:
        cli += ["--speculative-draft-model-load-format", draft_load_format]
    args = ServerArgs.from_cli_args(parser.parse_args(cli))
    args.mapping = Mapping(rank=0, world_size=1)
    config = ModelConfig(
        str(checkpoint), dtype="float32", model_override_args="{}", server_args=args
    )
    model = WeightLoader.load_model(
        model_config=config,
        server_args=args,
        device="cuda",
        gpu_id=0,
        memory_saver_adapter=TorchMemorySaverAdapter.create(enable=False),
        checkpoint_load_group=None,
        is_draft_worker=is_draft,
    )
    assert args.load_format == load_format
    assert args.speculative_draft_model_load_format == draft_load_format
    return model


@pytest.mark.parametrize("file_format", ["pt", "safetensors"])
@pytest.mark.parametrize(
    "load_format,draft_load_format", [("dummy", "auto"), ("auto", None)]
)
def test_draft_reads_checkpoint_weights(
    checkpoint, file_format, load_format, draft_load_format
):
    weights = {
        "fc.weight": torch.full((16, 32), 0.375),
        "norm.weight": torch.full((16,), 0.625),
        "hidden_norm.weight": torch.full((16,), 0.875),
    }
    if file_format == "pt":
        torch.save(weights, checkpoint / "pytorch_model.bin")
    else:
        save_file(weights, checkpoint / "model.safetensors")
    model = load(
        checkpoint,
        load_format=load_format,
        draft_load_format=draft_load_format,
        is_draft=True,
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.cpu(), weights[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "load_format,draft_load_format,is_draft",
    [
        ("dummy", "auto", False),
        ("dummy", None, False),
        ("dummy", None, True),
        ("auto", "dummy", True),
    ],
)
def test_dummy_loading_needs_no_checkpoint(
    checkpoint, load_format, draft_load_format, is_draft
):
    model = load(
        checkpoint,
        load_format=load_format,
        draft_load_format=draft_load_format,
        is_draft=is_draft,
    )
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize(
    "load_format,draft_load_format,is_draft",
    [("dummy", "auto", True), ("auto", "dummy", False)],
)
def test_checkpoint_loading_does_not_fall_back_to_dummy(
    checkpoint, load_format, draft_load_format, is_draft
):
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        load(
            checkpoint,
            load_format=load_format,
            draft_load_format=draft_load_format,
            is_draft=is_draft,
        )


def test_draft_preserves_explicit_file_format(checkpoint):
    torch.save({"fc.weight": torch.ones(16, 32)}, checkpoint / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        load(
            checkpoint,
            load_format="auto",
            draft_load_format="safetensors",
            is_draft=True,
        )


def test_cli_rejects_unknown_draft_format(checkpoint):
    with pytest.raises(SystemExit):
        load(
            checkpoint, load_format="dummy", draft_load_format="unknown", is_draft=True
        )
