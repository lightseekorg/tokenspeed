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

import pytest
import torch
from tokenspeed_kernel.ops.moe.gluon import (
    GLUON_FP8_CHANNEL_APPLY,
    gluon_fp8_channel_bf16_precomputed_moe_apply,
)
from utils import is_amd


@pytest.fixture
def require_amd_kernel_package():
    # These CPU checks reach a launcher that imports the optional AMD package.
    if not is_amd():
        pytest.importorskip(
            "tokenspeed_kernel_amd", reason="AMD kernel package is optional"
        )
    else:
        import tokenspeed_kernel_amd  # noqa: F401


def test_apply_rejects_unknown_policy():
    with pytest.raises(ValueError, match="precision_policy"):
        gluon_fp8_channel_bf16_precomputed_moe_apply(
            {"precision_policy": "unknown"},
            None,
            None,
            None,
        )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"enable_pdl": True}, "enable_pdl"),
        ({"plan_extra": {"output_scale": 2.0}}, "unit output scale"),
        ({"plan_extra": {"combine_order": "slot"}}, "rank-local"),
    ],
)
def test_apply_rejects_unsupported_launch_before_device_use(options, error):
    options = dict(options)
    plan = {
        "apply_kernel_name": GLUON_FP8_CHANNEL_APPLY,
        "precision_policy": "default",
    }
    plan.update(options.pop("plan_extra", {}))
    with pytest.raises(ValueError, match=error):
        gluon_fp8_channel_bf16_precomputed_moe_apply(
            plan, None, torch.nn.Module(), None, **options
        )


@pytest.mark.parametrize("intermediate", [320, 640])
@pytest.mark.usefixtures("require_amd_kernel_package")
def test_default_policy_still_rejects_non_gpu_tensors(intermediate):
    weights = torch.nn.Module()
    weights.w13_weight = torch.empty(
        256, 2 * intermediate, 5120, device="meta", dtype=torch.float8_e4m3fn
    )
    weights.w2_weight = torch.empty(
        256, 5120, intermediate, device="meta", dtype=torch.float8_e4m3fn
    )
    weights.w13_weight_scale = torch.empty(256, 2 * intermediate, device="meta")
    weights.w2_weight_scale = torch.empty(256, 5120, device="meta")
    with pytest.raises(ValueError, match="contiguous GPU tensors"):
        gluon_fp8_channel_bf16_precomputed_moe_apply(
            {
                "apply_kernel_name": GLUON_FP8_CHANNEL_APPLY,
                "precision_policy": "default",
            },
            torch.empty(1, 5120, device="meta", dtype=torch.bfloat16),
            weights,
            None,
            topk_ids=torch.zeros(1, 8, dtype=torch.int32),
            topk_weights=torch.ones(1, 8),
        )


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"precision_policy": "default"},
    ],
)
@pytest.mark.usefixtures("require_amd_kernel_package")
def test_default_policy_executes_empty_gluon_batch_without_name_pin(
    monkeypatch, policy
):
    # Meta tensors stand in for GPU storage; all dtype, shape and scale checks
    # still run. An empty batch needs no GPU kernel launch.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    weights = torch.nn.Module()
    with torch.device("meta"):
        weights.w13_weight = torch.empty(256, 640, 5120, dtype=torch.float8_e4m3fn)
        weights.w2_weight = torch.empty(256, 5120, 320, dtype=torch.float8_e4m3fn)
        weights.w13_weight_scale = torch.empty(256, 640, dtype=torch.float32)
        weights.w2_weight_scale = torch.empty(256, 5120, dtype=torch.float32)
        x = torch.empty(0, 5120, dtype=torch.bfloat16)
        output = gluon_fp8_channel_bf16_precomputed_moe_apply(
            policy,
            x,
            weights,
            None,
            topk_weights=torch.empty(0, 8, dtype=torch.float32),
            topk_ids=torch.empty(0, 8, dtype=torch.int32),
        )
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


@pytest.mark.parametrize(
    "experts,topk,intermediate", [(128, 8, 320), (256, 4, 320), (256, 8, 352)]
)
def test_default_policy_rejects_unsupported_expert_geometry(
    experts, topk, intermediate
):
    weights = torch.nn.Module()
    with torch.device("meta"):
        weights.w13_weight = torch.empty(
            experts, 2 * intermediate, 5120, dtype=torch.float8_e4m3fn
        )
        weights.w2_weight = torch.empty(
            experts, 5120, intermediate, dtype=torch.float8_e4m3fn
        )
        with pytest.raises(ValueError, match="E256 H5120 I320/I640 top8"):
            gluon_fp8_channel_bf16_precomputed_moe_apply(
                {"precision_policy": "default"},
                torch.empty(1, 5120, dtype=torch.bfloat16),
                weights,
                None,
                topk_weights=torch.empty(1, topk, dtype=torch.float32),
                topk_ids=torch.empty(1, topk, dtype=torch.int32),
            )


@pytest.mark.parametrize("ep_size,ep_rank", [(2, 0), (1, 1)])
def test_default_policy_rejects_expert_parallelism(ep_size, ep_rank):
    weights = torch.nn.Module()
    weights.ep_size = ep_size
    weights.ep_rank = ep_rank
    with pytest.raises(ValueError, match="EP1 rank-local"):
        gluon_fp8_channel_bf16_precomputed_moe_apply(
            {"precision_policy": "default"}, None, weights, None
        )


@pytest.mark.usefixtures("require_amd_kernel_package")
def test_default_policy_rejects_interleaved_weights():
    weights = torch.nn.Module()
    weights.w13_input_layout = "interleaved"
    with torch.device("meta"):
        weights.w13_weight = torch.empty(256, 640, 5120, dtype=torch.float8_e4m3fn)
        weights.w2_weight = torch.empty(256, 5120, 320, dtype=torch.float8_e4m3fn)
        with pytest.raises(ValueError, match="concatenated gate/up weights"):
            gluon_fp8_channel_bf16_precomputed_moe_apply(
                {"precision_policy": "default"},
                torch.empty(0, 5120, dtype=torch.bfloat16),
                weights,
                None,
                topk_weights=torch.empty(0, 8, dtype=torch.float32),
                topk_ids=torch.empty(0, 8, dtype=torch.int32),
            )
