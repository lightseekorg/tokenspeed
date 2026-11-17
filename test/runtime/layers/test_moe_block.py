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

"""MoEBlock adds the shared expert to the routed experts' output."""

import os
import sys

import pytest
import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution import forward_step
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.mlp import GatedMLP
from tokenspeed.runtime.layers.moe.block import MoEBlock, MoEBlockFusion
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

_HIDDEN = 1024
_INTERMEDIATE = 256
_EXPERTS = 8
_TOP_K = 2


def _block(fusions, alt_stream) -> MoEBlock:
    """A softmax-routed block on BF16 weights seeded the same for every call."""
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return (torch.randn(*shape, generator=generator) * 0.05).to(
            "cuda", torch.bfloat16
        )

    dense = Mapping(rank=0, world_size=1).dense
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        gate = nn.Linear(_HIDDEN, _EXPERTS, bias=False)
        experts = MoELayer(
            top_k=_TOP_K,
            num_experts=_EXPERTS,
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            quant_config=None,
            layer_index=0,
            prefix="model.layers.0.mlp",
        )
        shared = GatedMLP.tensor_parallel(
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            activation=SiluAndMul(),
            tp_rank=dense.tp_rank,
            tp_size=dense.tp_size,
            tp_group=dense.tp_group,
            reduce_results=False,
            fusions=(),
            quant_config=None,
            prefix="model.layers.0.mlp.shared_experts",
        )
    topk = TopK(
        top_k=_TOP_K, renormalize=True, output_format=experts.topk_output_format
    )
    gate.weight.data.copy_(randn(_EXPERTS, _HIDDEN))
    for e in range(_EXPERTS):
        for shard_id in ("w1", "w3"):
            w = randn(_INTERMEDIATE, _HIDDEN)
            experts.w13_weight.weight_loader(experts.w13_weight, w, shard_id, e)
        w2 = randn(_HIDDEN, _INTERMEDIATE)
        experts.w2_weight.weight_loader(experts.w2_weight, w2, "w2", e)
    experts.process_weights_after_loading(experts)
    shared.gate_up_proj.weight.data.copy_(randn(2 * _INTERMEDIATE, _HIDDEN))
    shared.down_proj.weight.data.copy_(randn(_HIDDEN, _INTERMEDIATE))
    return MoEBlock(gate, topk, experts, shared, alt_stream, fusions)


def _input(rows: int, seed: int = 0) -> torch.Tensor:
    x = torch.randn(rows, _HIDDEN, generator=torch.Generator().manual_seed(seed))
    return x.to("cuda", torch.bfloat16)


@torch.no_grad()
@pytest.mark.parametrize("rows", [1, 77])
def test_adds_shared_expert_to_routed_experts(rows: int) -> None:
    block = _block((), None)
    x = _input(rows)

    routed = block.experts(
        hidden_states=x,
        topk_output=block.topk(x, block.gate(x)),
        num_global_tokens=rows,
        max_num_tokens_per_gpu=rows,
    )
    assert torch.equal(block(x, rows, rows), routed + block.shared_experts(x))


@torch.no_grad()
@pytest.mark.parametrize("fusions", [(), (MoEBlockFusion.FINALIZE_ADD_SHARED,)])
def test_empty_input(fusions) -> None:
    assert _block(fusions, None)(_input(0), 0, 0).shape == (0, _HIDDEN)


@torch.no_grad()
def test_fused_finalize_matches_separate_kernels() -> None:
    block = _block((MoEBlockFusion.FINALIZE_ADD_SHARED,), None)
    if not block.experts.supports_deferred_cuda_finalize:
        pytest.skip("the MoE backend on this GPU finalizes its own rows")
    x = _input(77)
    expected = _block((), None)(x, 77, 77)

    out = block(x, 77, 77)

    # The fused epilogue rounds routed plus shared once, so only close.
    torch.testing.assert_close(out, expected, atol=2**-6, rtol=2**-7)


@torch.no_grad()
def test_graph_overlaps_shared_expert_bitwise(monkeypatch) -> None:
    # The shared expert runs on the auxiliary stream only under CUDA graphs.
    block = _block((MoEBlockFusion.FINALIZE_ADD_SHARED,), torch.cuda.Stream())
    x = _input(33)
    expected = block(x, 33, 33)

    static = torch.zeros_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    monkeypatch.setattr(forward_step, "_is_cuda_graph_phase", True)
    with torch.cuda.stream(stream):
        block(static, 33, 33)
    torch.cuda.current_stream().wait_stream(stream)
    monkeypatch.setattr(forward_step, "_is_capture_mode", True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = block(static, 33, 33)
    static.copy_(x)
    graph.replay()

    assert torch.equal(out, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
