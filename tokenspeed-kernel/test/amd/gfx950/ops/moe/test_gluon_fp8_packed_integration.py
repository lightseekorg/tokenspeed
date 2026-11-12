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

"""CPU planning, load lifecycle, and call-boundary checks for packed FP8 MoE."""

from __future__ import annotations

import pytest
import tokenspeed_kernel.ops.moe.gluon as fp8
import torch

pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess",
    reason="tokenspeed-kernel-amd is required for packed FP8 MoE integration tests",
)

from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess import (  # noqa: E402
    _copy_packed_fp8_shard,
    pack_fp8_weight_n16k32,
)


def _plan(**changes: object) -> dict:
    from tokenspeed_kernel.ops.moe import moe_plan

    arguments = {
        "weight_dtype": "fp8",
        "input_dtype": torch.bfloat16,
        "activation": "swiglu",
        "routing_mode": "precomputed_topk",
        "a2a_backend": "none",
        "ep_size": 4,
        "ispp": 2048,
        "hidden": 4096,
        "swiglu_form": "standard",
        "activation_clamped": True,
        "expert_id_repeats": False,
        "fp8_scale_block_shape": (128, 128),
        "internal_activation_dtype": "input",
        "fast_math": True,
        "combine_order": "rank",
        "features": frozenset(),
    }
    arguments.update(changes)
    return moe_plan(**arguments)


@pytest.mark.parametrize("ep_size,ispp", [(1, 512), (4, 2048)])
def test_packed_fp8_plan_selects_packed_preprocessor(ep_size: int, ispp: int) -> None:
    plan = _plan(ep_size=ep_size, ispp=ispp)
    assert plan["apply_kernel_name"] == (
        f"gluon_fp8_packed_ep{ep_size}_precomputed_moe_apply"
    )
    assert plan["weight_preprocessor"] is fp8.gluon_fp8_packed_moe_weights
    assert plan["weight_preprocessor"].__module__ == (
        "tokenspeed_kernel_amd.ops.gfx950.moe.fp8.weight_preprocess"
    )
    assert plan["supports_precomputed_topk"]
    assert not plan["supports_deferred_finalize"]


@pytest.mark.parametrize(
    "change",
    [
        {"hidden": 512},
        {"ispp": 1024},
        {"ep_size": 4, "ispp": 512},
        {"ep_size": 1, "ispp": 2048},
        {"activation_clamped": False},
        {"swiglu_form": "generalized"},
        {"activation": "silu", "swiglu_form": None},
        {"expert_id_repeats": True},
    ],
)
def test_packed_fp8_plan_preserves_other_gluon_fp8_cases(change: dict) -> None:
    assert _plan(**change)["apply_kernel_name"] == (
        "gluon_fp8_block_precomputed_moe_apply"
    )


def _small_fp8_weight(modulus: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.arange(2 * 32 * 32, dtype=torch.int32)
        .remainder(modulus)
        .to(torch.uint8)
        .reshape(2, 32, 32)
        .view(torch.float8_e4m3fn),
        requires_grad=False,
    )


def test_packed_fp8_reload_rejects_tile_aligned_partial_input_columns() -> None:
    linear = torch.arange(32 * 64).remainder(127).to(torch.uint8).reshape(1, 32, 64)
    packed = pack_fp8_weight_n16k32(linear.view(torch.float8_e4m3fn))
    original = packed.view(torch.uint8).clone()
    source = torch.ones(32, 32, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="complete local FP8 rows"):
        _copy_packed_fp8_shard(packed[0, :, :32], source)
    assert torch.equal(packed.view(torch.uint8), original)


def test_packed_fp8_preprocess_packs_plain_tensor_weights() -> None:
    class PlainTensorModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w13_weight = _small_fp8_weight(128).detach()
            self.w2_weight = _small_fp8_weight(64).detach()
            self.w13_weight_scale_inv = torch.ones(2, 2, 1)
            self.w2_weight_scale_inv = torch.ones(2, 1, 1)
            # Synthetic weights are already loaded, like the benchmark inputs.
            self.w13_weight.weight_loader = lambda *args, **kwargs: None
            self.w2_weight.weight_loader = lambda *args, **kwargs: None

    module = PlainTensorModule()
    original13, original2 = module.w13_weight.clone(), module.w2_weight.clone()
    fp8.gluon_fp8_packed_moe_weights({}, module)
    assert torch.equal(
        module.w13_weight.view(torch.uint8),
        pack_fp8_weight_n16k32(original13).view(torch.uint8),
    )
    assert torch.equal(
        module.w2_weight.view(torch.uint8),
        pack_fp8_weight_n16k32(original2).view(torch.uint8),
    )
    for weight in (module.w13_weight, module.w2_weight):
        packed = weight.view(torch.uint8).clone()
        weight.weight_loader(weight, torch.empty(0))
        assert torch.equal(weight.view(torch.uint8), packed)
