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

"""Runtime route selection and graph checks for packed FP8 MoE."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from utils import is_cdna4

pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx950.moe.fp16",
    reason="tokenspeed-kernel-amd is required for packed FP8 MoE route tests",
)

from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import (  # noqa: E402
    moe_align_device,
    moe_align_fused,
)

pytestmark = pytest.mark.skipif(not is_cdna4(), reason="requires AMD gfx950 GPU")


class _PackedModule(torch.nn.Module):
    def __init__(
        self,
        packed13: torch.Tensor,
        packed2: torch.Tensor,
        scales13: torch.Tensor,
        scales2: torch.Tensor,
        rank: int,
        num_experts_total: int,
    ) -> None:
        super().__init__()
        from tokenspeed.runtime.layers.activation import SwigluArg

        self.w13_weight = packed13
        self.w2_weight = packed2
        self.w13_weight_scale_inv = scales13
        self.w2_weight_scale_inv = scales2
        self.num_local_experts = packed13.shape[0]
        self.num_experts = num_experts_total
        self.ep_rank = rank
        self.ep_size = num_experts_total // self.num_local_experts
        self.swiglu_arg = SwigluArg(alpha=1.0, limit=7.0)


def _run_packed(
    x: torch.Tensor, module: _PackedModule, weights: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    import tokenspeed_kernel.ops.moe.gluon as fp8

    apply = (
        fp8.gluon_fp8_packed_ep1_precomputed_moe_apply
        if module.ep_size == 1
        else fp8.gluon_fp8_packed_ep4_precomputed_moe_apply
    )
    return apply({"activation": "swiglu"}, x, module, torch.empty(0), weights, ids)


def test_packed_fp8_empty_input_returns_empty_output() -> None:
    hidden, intermediate = 4096, 512
    x = torch.empty((0, hidden), device="cuda", dtype=torch.bfloat16)
    ids = torch.empty((0, 8), device="cuda", dtype=torch.int32)
    weights = torch.empty((0, 8), device="cuda", dtype=torch.float32)
    module = _PackedModule(
        torch.empty(
            (1, 2 * intermediate, hidden), device="cuda", dtype=torch.float8_e4m3fn
        ),
        torch.empty(
            (1, hidden, intermediate), device="cuda", dtype=torch.float8_e4m3fn
        ),
        torch.empty((1, 2 * intermediate // 128, hidden // 128), device="cuda"),
        torch.empty((1, hidden // 128, intermediate // 128), device="cuda"),
        rank=0,
        num_experts_total=1,
    )
    output = _run_packed(x, module, weights, ids)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


def _whole_call_kernels() -> tuple[object, ...]:
    from tokenspeed_kernel.ops.quantization.triton import (
        _fp8_token_group_quantize_kernel,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        packed_large,
        packed_small,
        warp_decode,
    )

    # Guard both alignment backends and all sublaunches for larger route sets.
    return (
        warp_decode._stage1_fp8_warp_gemv,
        warp_decode._stage2_fp8_warp_gemv,
        moe_align_fused._fused_align_kernel,
        moe_align_device._prepare_small_kernel,
        moe_align_device._scatter_kernel,
        moe_align_device._chunk_count_kernel,
        moe_align_device._chunk_scatter_kernel,
        _fp8_token_group_quantize_kernel,
        packed_small.gluon_fp8_packed_small_stage1_gfx950,
        packed_small.gluon_fp8_packed_small_stage2_gfx950,
        packed_small.gluon_fp8_packed_small_activation_gfx950,
        packed_large.gluon_fp8_packed_large_stage1_gfx950,
        packed_large.gluon_fp8_packed_large_stage2_gfx950,
    )


@pytest.mark.parametrize(
    ("rows", "ep_size", "rank", "intermediate", "path", "fully_local"),
    [
        (1, 1, 0, 512, "warp", False),
        (3, 1, 0, 512, "public-warp", False),
        (4, 1, 0, 512, "warp", False),
        (5, 1, 0, 512, "small", False),
        (8, 1, 0, 512, "small", False),
        (1, 4, 0, 2048, "small", False),
        (1, 4, 3, 2048, "small", False),
        (2, 4, 0, 2048, "small", False),
        (2, 4, 3, 2048, "small", False),
        (3, 4, 0, 2048, "small", False),
        (8, 4, 3, 512, "small", False),
        (32, 4, 0, 2048, "small", False),
        (32, 4, 3, 2048, "small", False),
        (128, 4, 3, 512, "large", False),
        (129, 4, 3, 512, "large", False),
        (256, 4, 0, 2048, "large", False),
        (256, 4, 3, 2048, "large", False),
        pytest.param(2, 4, 0, 512, "small", True, id="fully-local"),
    ],
)
def test_packed_fp8_runtime_routes_do_not_recompile(
    rows: int,
    ep_size: int,
    rank: int,
    intermediate: int,
    path: str,
    fully_local: bool,
    record_property: Callable[[str, object], None],
) -> None:
    from test_gluon_fp8_packed_small import (
        _assert_numerics,
        _fixture,
        _reference,
        _routes,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )
    from utils import assert_no_triton_compile

    hidden = 1024 if fully_local else 4096
    x, w13, w2, s13, s2, ids, weights = _fixture(
        rows, 1 if fully_local else ep_size, rank, False, hidden, intermediate
    )
    start = rank * w13.shape[0]
    packed13, packed2 = pack_fp8_weight_n16k32(w13), pack_fp8_weight_n16k32(w2)
    inputs = (x, packed13, packed2, s13, s2, ids, weights)
    module = _PackedModule(
        packed13, packed2, s13, s2, rank, num_experts_total=w13.shape[0] * ep_size
    )
    if path == "public-warp":
        from test_gluon_fp8_packed_integration import _plan

        public_plan = _plan(ep_size=1, ispp=512)
    pointers = tuple(tensor.data_ptr() for tensor in inputs)
    original_ids, original_weights = ids.clone(), weights.clone()
    changed_ids = (
        ids.flip(1)
        if fully_local
        else _routes(rows, w13.shape[0], ep_size, start, variant=1)
    )
    changed_weights = (
        torch.arange(1, 9, device="cuda").float()[None, :].expand_as(weights) / 36.0
    )
    normal, _ = _reference(
        x,
        w13,
        w2,
        s13,
        s2,
        original_ids,
        original_weights,
        start,
        7.0,
        quantized_activations=path not in ("warp", "public-warp"),
    )
    changed, _ = _reference(
        x,
        w13,
        w2,
        s13,
        s2,
        changed_ids,
        changed_weights,
        start,
        7.0,
        quantized_activations=path not in ("warp", "public-warp"),
    )
    assert not torch.equal(normal, changed)
    updated_weight_case = None
    if rows == 1 and ep_size == 1:
        # The captured graph must read the same canonical packed storage after
        # a checkpoint update, including new FP32 block scales.
        expert = int((original_ids[0, 0] - start).item())
        original_s13, original_s2 = s13[expert].clone(), s2[expert].clone()
        w13[expert].fill_(1.0)
        w2[expert].fill_(-0.5)
        s13[expert].mul_(1.25)
        s2[expert].mul_(0.75)
        updated, _ = _reference(
            x,
            w13,
            w2,
            s13,
            s2,
            original_ids,
            original_weights,
            start,
            7.0,
            quantized_activations=path not in ("warp", "public-warp"),
        )
        assert not torch.equal(normal, updated)
        updated_weight_case = (
            expert,
            pack_fp8_weight_n16k32(w13[expert : expert + 1])[0],
            pack_fp8_weight_n16k32(w2[expert : expert + 1])[0],
            s13[expert].clone(),
            s2[expert].clone(),
            updated,
        )
        s13[expert].copy_(original_s13)
        s2[expert].copy_(original_s2)
    del w13, w2  # Captured execution uses only the packed weights.
    kernels = _whole_call_kernels()
    record_property(
        "guarded_kernels", ",".join(kernel.fn.__name__ for kernel in kernels)
    )
    record_property("route_slots", ids.numel())
    record_property("reference_path", path)

    def run() -> torch.Tensor:
        if path == "public-warp":
            from tokenspeed_kernel.ops.moe import moe_apply

            return moe_apply(
                public_plan,
                x,
                module,
                torch.empty(0, device=x.device),
                topk_ids=ids,
                topk_weights=weights,
                do_finalize=True,
            )
        return _run_packed(x, module, weights, ids)

    # Warm only the original routing. Changed/empty routing must not introduce
    # any compilation, even in eager calls before replay of the captured graph.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    cases = [
        ("original", original_ids, original_weights, normal),
        ("changed", changed_ids, changed_weights, changed),
    ]
    if ep_size > 1:
        # Eight distinct remote IDs preserve the router's unique-top-k contract.
        remote_start = packed13.shape[0] if rank == 0 else 0
        empty_ids = (
            torch.arange(8, device="cuda", dtype=torch.int32) + remote_start
        ).expand_as(ids)
        cases.append(("empty", empty_ids, original_weights, None))
    cases.append(("restored", original_ids, original_weights, normal))
    if updated_weight_case is not None:
        cases.append(
            (
                "updated_weight_and_scale",
                original_ids,
                original_weights,
                updated_weight_case[-1],
            )
        )
    with assert_no_triton_compile(*kernels):
        with torch.cuda.graph(graph):
            graph_output = run()
        output_pointer = graph_output.data_ptr()
        original_output: torch.Tensor | None = None
        for label, next_ids, next_weights, expected in cases:
            if label == "updated_weight_and_scale":
                expert, next13, next2, next_s13, next_s2, _ = updated_weight_case
                packed13[expert].copy_(next13)
                packed2[expert].copy_(next2)
                s13[expert].copy_(next_s13)
                s2[expert].copy_(next_s2)
            ids.copy_(next_ids)
            weights.copy_(next_weights)
            eager = run()
            if expected is None:
                # The preceding replay left nonzero data in this same buffer.
                # Poison it again to require the captured preparation to clear it.
                graph_output.fill_(-7)
            graph.replay()
            torch.cuda.synchronize()
            if path == "public-warp":
                torch.testing.assert_close(graph_output, eager, rtol=0, atol=0)
            assert pointers == tuple(tensor.data_ptr() for tensor in inputs)
            assert graph_output.data_ptr() == output_pointer
            if label == "original":
                original_output = graph_output.clone()
            elif label == "changed":
                assert original_output is not None
                assert not torch.equal(eager, original_output)
                assert not torch.equal(graph_output, original_output)
            for execution, output in (("eager", eager), ("graph", graph_output)):
                if expected is None:
                    assert torch.isfinite(output).all()
                    assert torch.count_nonzero(output).item() == 0
                else:
                    _assert_numerics(
                        output,
                        expected,
                        lambda name, value, label=label, execution=execution: (
                            record_property(f"{label}_{execution}_{name}", value)
                        ),
                    )
            record_property(f"{label}_passed", True)


def test_shared_activation_quantizer_row_count_does_not_recompile_gfx950() -> None:
    import tokenspeed_kernel.ops.moe.gluon as fp8
    from tokenspeed_kernel.ops.quantization.triton import (
        _fp8_token_group_quantize_kernel,
    )
    from utils import assert_no_triton_compile

    for rows, width in ((1, 128), (512, 2048)):
        warm = torch.ones((rows, width), device="cuda", dtype=torch.bfloat16)
        fp8._quantize_packed_activation(warm)
    with assert_no_triton_compile(_fp8_token_group_quantize_kernel):
        for rows, width in (
            (1, 128),
            (8, 128),
            (17, 256),
            (511, 2048),
            (512, 2048),
            (8193, 128),
            (1024, 2048),
        ):
            values = torch.ones((rows, width), device="cuda", dtype=torch.bfloat16)
            fp8._quantize_packed_activation(values)


def test_packed_large_row_views_reuse_warmed_capacity_buckets() -> None:
    from test_gluon_fp8_packed_small import _fixture
    from tokenspeed_kernel.ops.moe.gluon import _quantize_packed_activation
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8 import (
        launch_gluon_fp8_packed_moe_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )
    from utils import assert_no_triton_compile

    x, w13, w2, s13, s2, ids, weights = _fixture(67, 1, 0, False, 4096, 512)
    packed13, packed2 = pack_fp8_weight_n16k32(w13), pack_fp8_weight_n16k32(w2)

    def run(m: int):
        return launch_gluon_fp8_packed_moe_gfx950(
            x[:m],
            packed13,
            packed2,
            s13,
            s2,
            ids[:m],
            weights[:m],
            7.0,
            0,
            num_experts_total=packed13.shape[0],
            quantize_activation=_quantize_packed_activation,
        )

    # Changed tensor-view sizes cross explicit capacity buckets: M64 has 512
    # routes, while M65 rounds to 1024. They also cross the schedule switch:
    # through M64 runs 16-row blocks and M65 the 64-row blocks.
    # Warm those two buckets and both schedules, not every checked size.
    run(62)
    run(64)
    run(66)
    with assert_no_triton_compile(*_whole_call_kernels()):
        for m in (63, 64, 65):
            assert torch.isfinite(run(m)).all()


def test_packed_plan_reuses_one_bank_across_warp_boundary(record_property) -> None:
    from test_gluon_fp8_packed_integration import _plan
    from test_gluon_fp8_packed_small import (
        _assert_numerics,
        _fixture,
        _reference,
    )
    from tokenspeed_kernel.ops.moe import moe_apply
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (
        pack_fp8_weight_n16k32,
    )

    x, w13, w2, s13, s2, ids, weights = _fixture(8, 1, 0, False, 4096, 512)
    sizes = (1, 2, 3, 4, 5, 8)
    references = {
        rows: _reference(
            x[:rows],
            w13,
            w2,
            s13,
            s2,
            ids[:rows],
            weights[:rows],
            0,
            7.0,
            quantized_activations=rows > 4,
        )[0]
        for rows in sizes
    }
    packed13, packed2 = pack_fp8_weight_n16k32(w13), pack_fp8_weight_n16k32(w2)
    module = _PackedModule(packed13, packed2, s13, s2, 0, w13.shape[0])
    del w13, w2
    plan = _plan(ep_size=1, ispp=512)
    registration = plan["apply_kernel_name"]
    pointers = packed13.data_ptr(), packed2.data_ptr()
    for rows in sizes:
        actual = moe_apply(
            plan,
            x[:rows],
            module,
            torch.empty(0, device=x.device),
            topk_ids=ids[:rows],
            topk_weights=weights[:rows],
            do_finalize=True,
        )
        _assert_numerics(actual, references[rows], record_property)
        assert plan["apply_kernel_name"] == registration
        assert (module.w13_weight.data_ptr(), module.w2_weight.data_ptr()) == pointers
        assert {key for key in vars(module) if key.startswith("w")} == {
            "w13_weight",
            "w2_weight",
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        }
