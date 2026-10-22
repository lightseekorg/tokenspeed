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

"""Budget boundaries and shared encoder dispatch regression tests."""

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.multimodal.encoder_batching import (  # noqa: E402
    pack_encoder_batches,
)
from tokenspeed.runtime.multimodal.encoder_cudagraph import (  # noqa: E402
    EncoderForwardStepRunner,
    VisionEncoderCudaGraphAdapter,
)


@pytest.mark.parametrize(
    "tokens,sequences,limit,items,metadata,expected",
    [
        ([6, 2, 4], [1, 1, 1], 6, None, None, [[1, 2], [0]]),
        ([2] * 9, [1] * 9, 20, None, None, [list(range(9))]),
        ([2] * 3, [1] * 3, 20, 2, None, [[0, 1], [2]]),
        ([2, 2, 2], [2, 2, 1], 20, None, 3, [[0], [1, 2]]),
        ([21, 2, 3], [1, 1, 1], 20, None, None, [[1, 2], [0]]),
        ([1, 2], [5, 1], 20, None, 3, [[0], [1]]),
        ([], [], 20, None, None, []),
    ],
)
def test_pack(tokens, sequences, limit, items, metadata, expected):
    assert (
        pack_encoder_batches(
            tokens,
            sequences,
            max_tokens=limit,
            max_items=items,
            max_metadata_sequences=metadata,
        )
        == expected
    )


@pytest.mark.parametrize(
    "tokens,sequences,limit,items,metadata",
    [
        ([1], [], 10, None, None),
        ([-1], [1], 10, None, None),
        ([1], [-1], 10, None, None),
        ([1], [1], 0, None, None),
        ([1], [1], 10, 0, None),
        ([1], [1], 10, None, 0),
    ],
)
def test_invalid_budget(tokens, sequences, limit, items, metadata):
    with pytest.raises(ValueError):
        pack_encoder_batches(
            tokens,
            sequences,
            max_tokens=limit,
            max_items=items,
            max_metadata_sequences=metadata,
        )


@dataclass
class Batch:
    encoder_output_tokens: list[int]
    metadata_sequences: list[int]
    indices: list[int]

    def num_items(self):
        return len(self.indices)

    def select(self, indices):
        return Batch(
            [self.encoder_output_tokens[i] for i in indices],
            [self.metadata_sequences[i] for i in indices],
            [self.indices[i] for i in indices],
        )


def test_runner_restores_order_and_preserves_eager_fallback():
    adapter = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        modality_name="image",
        capture_tp_size=1,
        capture_tp_group=None,
    )
    runner = EncoderForwardStepRunner(
        adapter=adapter,
        budget_range=(2, 8),
        max_batch_size=4,
        max_metadata_sequences_per_batch=4,
        metadata_sequence_budget_from_encoder_output_budget=False,
    )
    batch = Batch([9, 2, 4, 2], [1] * 4, [0, 1, 2, 3])

    def execute(part):
        return torch.cat(
            [
                torch.full((n, 1), i, dtype=torch.float32)
                for n, i in zip(part.encoder_output_tokens, part.indices, strict=True)
            ]
        )

    with patch.object(runner, "_run_eager", side_effect=execute) as eager, patch.object(
        runner, "_run_budget_graph", side_effect=lambda part, budget: execute(part)
    ) as graph:
        output = runner._dispatch(batch)
    assert graph.call_count == eager.call_count == 1
    assert graph.call_args.args[1] == 8
    assert eager.call_args.args[0].indices == [0]
    for i, (rows, tensor) in enumerate(
        zip(batch.encoder_output_tokens, output, strict=True)
    ):
        torch.testing.assert_close(
            tensor, torch.full((rows, 1), i, dtype=torch.float32)
        )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.cuda is not None),
    reason="Qwen3 vision graphs require an NVIDIA GPU",
)
def test_budget_graph_ignores_what_an_earlier_replay_left_in_its_padding():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )

    from tokenspeed.runtime.models.qwen3_vision import Qwen3VLMoeVisionModel

    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=128,
        intermediate_size=256,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=128,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0],
    )
    torch.manual_seed(0)
    tower = Qwen3VLMoeVisionModel(
        config, Mapping(rank=0, world_size=1), prefix="visual"
    )
    tower = tower.cuda().bfloat16().eval()
    with torch.no_grad():
        for name, param in tower.named_parameters():
            if "norm" in name:
                param.fill_(1.0 if name.endswith("weight") else 0.0)
            else:
                param.normal_(0, 0.05)

    def pre_encode(items):
        grid = torch.tensor([[1, h, w] for h, w in items], device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(len(items))
        patches = torch.randn(
            int(grid.prod(-1).sum()),
            24,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        if items == [(32, 32)]:
            patches.fill_(float("nan"))
        return tower.prepare_patch_embed(patches, grid), grid

    runner = EncoderForwardStepRunner(
        adapter=VisionEncoderCudaGraphAdapter(
            tower=tower,
            pre_encode=pre_encode,
            post_encode=lambda outs, grid: torch.cat(outs),
            out_div=4,
            merge=2,
            input_feature_shape=(1, 128),
            modality_name="vision",
            capture_tp_size=1,
            capture_tp_group=None,
        ),
        budget_range=(16, 256),
        metadata_sequence_budget_from_encoder_output_budget=True,
    )
    # A NaN batch fills the 256-token graph, then a 150-token batch leaves 424 of its rows as padding.
    items = [(24, 24), (6, 4)]
    with torch.no_grad():
        runner([(32, 32)])
        output = runner(items)
        tokens, grid = pre_encode(items)
        expected = tower.forward_blocks(tokens, tower.prepare_metadata(grid))
    assert sorted(runner.budget_graphs) == [16, 32, 64, 128, 256]
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
