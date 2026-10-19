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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.numerics.torch.gemm import (
    torch_dsv4_grouped_output_projection,
)
from tokenspeed_kernel.ops.gemm import (
    dsv4_grouped_output_projection,
    dsv4_grouped_output_projection_plan,
    dsv4_grouped_output_projection_preprocessor,
    dsv4_grouped_output_projection_warmup,
)
from tokenspeed_kernel.weights import get_weight_broker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("tokens", [1, 7])
def test_grouped_output_projection_preparation_and_graph_replay(
    device: str, require, tokens: int
) -> None:
    solution = "triton"
    require(
        "gemm", "dsv4_grouped_output_projection", solution, torch.bfloat16, "attention"
    )
    generator = torch.Generator(device=device).manual_seed(42)
    groups, heads_per_group, head_dim, output_dim = 2, 2, 512, 128
    nope_dim, rope_dim = 448, 64
    block_size = (128, 128)
    input_dim = heads_per_group * head_dim
    attention = (
        torch.randn(
            tokens,
            groups * heads_per_group,
            head_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.25
    )
    positions = torch.arange(tokens, device=device) * 3 + 1
    angles = torch.randn(32, rope_dim // 2, generator=generator, device=device)
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    checkpoint_weight = (
        torch.randn(groups * output_dim, input_dim, generator=generator, device=device)
        .mul_(32)
        .to(torch.float8_e4m3fn)
    )
    checkpoint_scale = torch.exp2(
        torch.randint(
            -8,
            -3,
            (groups * output_dim // 128, input_dim // 128),
            generator=generator,
            device=device,
        ).float()
    )
    owner = SimpleNamespace(
        weight=checkpoint_weight.clone(), weight_scale_inv=checkpoint_scale.clone()
    )
    broker = get_weight_broker()
    plan = dsv4_grouped_output_projection_plan(
        input_dtype=torch.bfloat16,
        weight_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=torch.float32,
        num_groups=groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        output_dim=output_dim,
        block_size=block_size,
        scale_format="ue8m0",
    )
    preprocessor = dsv4_grouped_output_projection_preprocessor(plan, solution=solution)
    if preprocessor is None:
        broker.enroll(owner.weight, None)
        broker.enroll(owner.weight_scale_inv, None)
    else:
        broker.preprocess(preprocessor, owner)
    dsv4_grouped_output_projection_warmup(
        plan,
        owner.weight,
        owner.weight_scale_inv,
        max_tokens=tokens,
        allow_unknown_layout=False,
        solution=solution,
    )

    def project() -> torch.Tensor:
        return dsv4_grouped_output_projection(
            plan,
            attention,
            positions,
            cos_sin,
            owner.weight,
            owner.weight_scale_inv,
            allow_unknown_layout=False,
            solution=solution,
        )

    def check(actual: torch.Tensor) -> None:
        expected = torch_dsv4_grouped_output_projection(
            attention=attention,
            positions=positions,
            cos_sin_cache=cos_sin,
            weight=checkpoint_weight,
            weight_scale=checkpoint_scale,
            num_groups=groups,
            heads_per_group=heads_per_group,
            output_dim=output_dim,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            block_size=block_size,
        )
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.float(), expected, atol=1e-3, rtol=5e-3)

    check(project())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        project()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = project()
    attention.mul_(0.75)
    positions.add_(1)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check(captured)
