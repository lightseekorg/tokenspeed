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
from tokenspeed_kernel.ops import gemm
from tokenspeed_kernel.ops.gemm import deep_gemm
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.weights import WeightBroker, get_weight_broker


def _plan():
    return gemm.dsv4_grouped_output_projection_plan(
        input_dtype=torch.bfloat16,
        weight_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=torch.float32,
        num_groups=1,
        heads_per_group=1,
        head_dim=128,
        nope_dim=64,
        rope_dim=64,
        output_dim=128,
        block_size=(128, 128),
        scale_format="ue8m0",
    )


def _weights():
    owner = torch.nn.Module()
    owner.weight = torch.nn.Parameter(
        torch.zeros(128, 128).to(torch.float8_e4m3fn), requires_grad=False
    )
    owner.weight_scale_inv = torch.nn.Parameter(torch.ones(1, 1), requires_grad=False)
    return owner


def _register(
    name, layout, calls, *, preprocess=None, priority=10, dtype=torch.float32
):
    def impl(**kwargs):
        calls.append(name)
        return kwargs["attention"]

    def warmup(**kwargs):
        calls.append(f"warmup:{name}")

    impl._tokenspeed_warmup = warmup
    KernelRegistry.get().register(
        KernelSpec(
            name=name,
            family="gemm",
            mode="dsv4_grouped_output_projection",
            format_signatures=frozenset(
                {
                    format_signature(
                        layouts={layout},
                        attention=dense_tensor_format(torch.bfloat16),
                        weight=dense_tensor_format(torch.float8_e4m3fn),
                    )
                }
            ),
            traits={"weight_scale_dtype": frozenset({dtype})},
            priority=priority,
            weight_preprocessor=(
                (("weight", "weight_scale_inv"), preprocess)
                if preprocess is not None
                else None
            ),
        ),
        impl,
    )


def _run(plan, owner, *, allow_unknown_layout=False):
    return gemm.dsv4_grouped_output_projection(
        plan,
        torch.zeros(2, 1, 128, dtype=torch.bfloat16),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, 64),
        owner.weight,
        owner.weight_scale_inv,
        allow_unknown_layout=allow_unknown_layout,
    )


def test_canonical_enrollment_and_dtype_views(fresh_registry):
    calls = []
    _register("float_scales", None, calls)
    _register("packed_scales", None, calls, dtype=torch.int32)
    broker = get_weight_broker()
    plan, owner = _plan(), _weights()
    assert gemm.dsv4_grouped_output_projection_preprocessor(plan) is None
    _run(plan, owner, allow_unknown_layout=True)
    with pytest.raises(ValueError, match="untracked"):
        _run(plan, owner)
    broker.enroll(owner.weight, None)
    broker.enroll(owner.weight_scale_inv, None)
    assert broker.layout(owner.weight) is None
    assert broker.layout(owner.weight_scale_inv) is None
    _run(plan, owner)
    owner.weight_scale_inv.data = owner.weight_scale_inv.data.view(torch.int32)
    _run(plan, owner)
    assert calls == ["float_scales", "float_scales", "packed_scales"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_deep_gemm_installs_scale_output(monkeypatch, dtype):
    monkeypatch.setattr(deep_gemm, "ceil_to_ue8m0", lambda scales: scales)
    monkeypatch.setattr(
        deep_gemm,
        "transform_sf_into_required_layout",
        lambda **kwargs: kwargs["sf"].to(dtype).clone(),
    )
    owner, broker = _weights(), WeightBroker()
    prepare = deep_gemm._deep_gemm_dsv4_grouped_output_projection_weights
    broker.preprocess(
        (("weight", "weight_scale_inv"), prepare),
        owner,
        config={
            "num_groups": 1,
            "output_dim": 128,
            "input_dim": 128,
            "block_size": (128, 128),
        },
    )
    assert owner.weight_scale_inv.dtype == dtype
    assert owner.weight_scale_inv.shape == (1, 1, 1)
    assert broker.layout(owner.weight_scale_inv) is prepare
    assert broker.layout(owner.weight) is prepare
