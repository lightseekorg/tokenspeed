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

import dataclasses

import pytest
import torch
from tokenspeed_kernel.ops.moe.route_quant import PreparedMoEInput, route_quant_fp8


@pytest.mark.parametrize("tokens", (1, 8))
@pytest.mark.parametrize("experts", (1, 31, 33, 64, 65, 127, 129, 1024))
@pytest.mark.parametrize("topk", (1, 2, 3, 5, 8, 16, 32))
@pytest.mark.parametrize("hidden", (256, 768))
def test_dimensions_reach_device_validation(tokens, experts, topk, hidden):
    x = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
    ids = torch.zeros((tokens, topk), dtype=torch.int32)
    with pytest.raises(ValueError, match="one CUDA device"):
        route_quant_fp8(
            x, ids, experts=experts, expert_offset=experts * 3, solution=None
        )


@pytest.mark.parametrize(
    "experts,offset",
    ((0, 0), (1025, 0), (True, 0), (65, -1), (65, True), (65, 2**31 - 65)),
)
def test_invalid_expert_interval(experts, offset):
    with pytest.raises(ValueError, match="local expert interval"):
        route_quant_fp8(
            torch.zeros((8, 512), dtype=torch.bfloat16),
            torch.zeros((8, 5), dtype=torch.int32),
            experts=experts,
            expert_offset=offset,
            solution=None,
        )


@pytest.mark.parametrize(
    "tokens,width,topk,message",
    (
        (0, 512, 5, "token bucket"),
        (2, 512, 5, "token bucket"),
        (8192, 512, 5, "token bucket"),
        (8, 257, 5, "multiple of 128"),
        (8, 512, 0, "slots"),
        (8, 512, 33, "slots"),
    ),
)
def test_unsupported_shapes(tokens, width, topk, message):
    with pytest.raises(ValueError, match=message):
        route_quant_fp8(
            torch.zeros((tokens, width), dtype=torch.bfloat16),
            torch.zeros((tokens, topk), dtype=torch.int32),
            experts=65,
            expert_offset=0,
            solution=None,
        )


def test_input_dtype_and_contiguity():
    x = torch.zeros((8, 512), dtype=torch.bfloat16)
    ids = torch.zeros((8, 5), dtype=torch.int32)
    with pytest.raises(ValueError, match="BF16 inputs"):
        route_quant_fp8(x.float(), ids, experts=65, expert_offset=0, solution=None)
    with pytest.raises(ValueError, match="int32 route IDs"):
        route_quant_fp8(x, ids.long(), experts=65, expert_offset=0, solution=None)
    with pytest.raises(ValueError, match="contiguous"):
        route_quant_fp8(
            torch.zeros((512, 8), dtype=torch.bfloat16).T,
            ids,
            experts=65,
            expert_offset=0,
            solution=None,
        )


def test_storage_retains_original_input_and_native_map():
    x = torch.zeros((8, 512), dtype=torch.bfloat16)
    ids = torch.zeros((8, 4), dtype=torch.int32)
    native_map = torch.arange(32, dtype=torch.int32).view(4, 8)
    storage = PreparedMoEInput(
        x,
        ids,
        0,
        128,
        torch.empty((32, 512), dtype=torch.float8_e4m3fn),
        torch.empty((4, 4000)),
        torch.zeros(129, dtype=torch.int64),
        torch.empty(32, dtype=torch.int32),
        native_map.T,
        torch.empty(128, dtype=torch.int32),
        torch.empty(128, dtype=torch.int32),
        torch.empty(32, dtype=torch.int32),
    )
    x.fill_(3)
    native_map[2, 5] = 19
    del x, ids, native_map
    assert storage.input[0, 0].item() == 3
    assert storage.inverse[5, 2].item() == 19
    assert storage.inverse.stride() == (1, 8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        storage.experts = 256


@pytest.fixture(params=(128, 256, 768, 1024))
def small_storage(request):
    # Device execution is separate; host layout validation uses real CPU storage.
    tokens, topk, experts = 1, 2, 31
    hidden = request.param
    rows = tokens * topk
    extent = ((rows + experts * 31 + 31) // 32) * 32
    return PreparedMoEInput(
        torch.empty((tokens, hidden), dtype=torch.bfloat16),
        torch.empty((tokens, topk), dtype=torch.int32),
        62,
        experts,
        torch.empty((4, hidden), dtype=torch.float8_e4m3fn),
        torch.empty((hidden // 128, extent), dtype=torch.float32),
        torch.empty(experts + 1, dtype=torch.int64),
        torch.empty(rows, dtype=torch.int32),
        torch.empty((topk, tokens), dtype=torch.int32).T,
        torch.empty(experts, dtype=torch.int32),
        torch.empty(experts, dtype=torch.int32),
        torch.empty(rows, dtype=torch.int32),
    )


def validate_storage(storage, **changes):
    from tokenspeed_kernel.ops.moe.route_quant import _validate_prepared_input

    arguments = dict(
        input=storage.input,
        ids=storage.ids,
        experts=storage.experts,
        expert_offset=storage.expert_offset,
    )
    arguments.update(changes)
    _validate_prepared_input(storage, **arguments)


def test_native_consumer_accepts_padded_capacity(small_storage):
    validate_storage(small_storage)
    assert small_storage.packed.shape[0] > small_storage.ids.numel()
    assert (
        small_storage.inverse.T.reshape(-1).data_ptr()
        == small_storage.inverse.data_ptr()
    )


@pytest.mark.parametrize("field", ("packed", "scales", "inverse", "offsets"))
def test_native_consumer_rejects_truncated_allocation(small_storage, field):
    tensors = {
        "packed": small_storage.packed,
        "scales": small_storage.scales,
        "inverse": small_storage.inverse,
        "offsets": small_storage.offsets,
    }
    tensor = tensors[field]
    malformed = dataclasses.replace(small_storage, **{field: tensor[:-1]})
    with pytest.raises(ValueError, match="layout"):
        validate_storage(malformed)


def test_native_consumer_rejects_overlapping_metadata(small_storage):
    malformed = dataclasses.replace(small_storage, cursors=small_storage.counts)
    with pytest.raises(ValueError, match="disjoint storage"):
        validate_storage(malformed)


@pytest.mark.parametrize("changed", ("input", "ids", "experts", "expert_offset"))
def test_native_consumer_rejects_other_producer(small_storage, changed):
    values = {
        "input": small_storage.input,
        "ids": small_storage.ids,
        "experts": small_storage.experts,
        "expert_offset": small_storage.expert_offset,
    }
    value = values[changed]
    value = value.clone() if isinstance(value, torch.Tensor) else value + 1
    with pytest.raises(ValueError, match="different inputs or experts"):
        validate_storage(small_storage, **{changed: value})


@pytest.fixture
def route_moe_platform(h100_platform):
    from tokenspeed_kernel.platform import Platform

    original = Platform.get()
    Platform.override(h100_platform)
    try:
        yield
    finally:
        Platform.override(original)


def route_moe_plan(**changes):
    from tokenspeed_kernel.ops.moe import moe_plan

    arguments = dict(
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=4,
        ispp=768,
        hidden=256,
        num_experts=132,
        top_k=5,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=False,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
        with_bias=False,
        fast_math=False,
        combine_order="rank",
        features=frozenset(),
        solution="cute_dsl",
    )
    arguments.update(changes)
    return moe_plan("fp8", **arguments)


def test_route_moe_plan_preserves_explicit_output_contract(route_moe_platform):
    plan = route_moe_plan()
    assert plan["supports_precomputed_topk"]
    assert not plan["support_routing"]
    assert not plan["supports_deferred_finalize"]
    assert plan["topk_weights_dtype"] == torch.float32


@pytest.mark.parametrize(
    "changes",
    (
        {"hidden": 257},
        {"top_k": 33},
        {"top_k": None},
        {"input_dtype": torch.float16},
        {"swiglu_form": "generalized"},
        {"activation_clamped": True},
        {"expert_id_repeats": True},
        {"with_bias": True},
        {"requires_deferred_finalize": True},
        {"a2a_backend": "deepep"},
        {"features": frozenset({"batch_invariant"})},
    ),
)
def test_route_moe_plan_rejects_unsupported_contract(route_moe_platform, changes):
    from tokenspeed_kernel.selection import NoKernelFoundError

    with pytest.raises(NoKernelFoundError):
        route_moe_plan(**changes)


@pytest.mark.parametrize(
    "hidden,ispp", ((128, 256), (256, 768), (768, 512), (1024, 2048))
)
def test_route_moe_plan_accepts_block_aligned_dimensions(
    route_moe_platform, hidden, ispp
):
    plan = route_moe_plan(hidden=hidden, ispp=ispp)
    assert plan["supports_precomputed_topk"]
    assert plan["topk_weights_dtype"] == torch.float32


@pytest.mark.parametrize("hidden", (0, -128))
def test_weight_preparation_rejects_nonpositive_width(hidden):
    from tokenspeed_kernel.ops.moe.cute_dsl import cute_dsl_fp8_moe_process_weights

    class Weights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._moe_backend_state = None
            self.hidden_size = hidden
            self.top_k = 1

    with pytest.raises(ValueError, match="positive 128-aligned hidden size"):
        cute_dsl_fp8_moe_process_weights({}, Weights())
