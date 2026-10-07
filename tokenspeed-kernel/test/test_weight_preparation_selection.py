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

from dataclasses import replace

import pytest
import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, register_kernel
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

pytestmark = pytest.mark.usefixtures("fresh_registry")


def _prepare(weight):
    raise AssertionError("selection must not execute a preprocessor")


SOURCE = format_signature(weight=dense_tensor_format(torch.float32))
PACKED = replace(SOURCE, layouts=frozenset({_prepare}))


def _register(name, signature, **kwargs):
    spec = KernelSpec(
        name=name,
        family="test_weight",
        mode="apply",
        format_signatures=frozenset({signature}),
        **kwargs,
    )
    KernelRegistry.get().register(spec, lambda: name)
    return spec


def test_layout_matching_and_initialization_have_distinct_cache_entries(h100_platform):
    _register("canonical", SOURCE, priority=8)
    _register(
        "packed", PACKED, priority=12, weight_preprocessor=(("weight",), _prepare)
    )
    for _ in range(2):
        for signature, ignore_layout, expected in (
            (SOURCE, True, "packed"),
            (SOURCE, False, "canonical"),
            (PACKED, False, "packed"),
        ):
            assert (
                select_kernel(
                    "test_weight",
                    "apply",
                    signature,
                    platform=h100_platform,
                    ignore_layout=ignore_layout,
                ).name
                == expected
            )

    KernelRegistry.get()._unregister("canonical")
    with pytest.raises(NoKernelFoundError):
        select_kernel("test_weight", "apply", SOURCE, platform=h100_platform)


@pytest.mark.parametrize(
    "incompatibility", ["platform", "feature", "trait", "solution", "dtype"]
)
def test_initialization_ignores_only_layout(h100_platform, incompatibility):
    compatible = _register(
        "compatible",
        PACKED,
        priority=8,
        solution="wanted",
        features=frozenset({"required"}),
        traits={"group_size": frozenset({128})},
    )
    changes = {
        "platform": {"capability": CapabilityRequirement(vendors=frozenset({"amd"}))},
        "feature": {"features": frozenset()},
        "trait": {"traits": {"group_size": frozenset({64})}},
        "solution": {"solution": "different"},
        "dtype": {
            "format_signatures": format_signatures(
                "weight", "dense", {torch.int32}, {_prepare}
            )
        },
    }
    KernelRegistry.get().register(
        replace(
            compatible, name="incompatible", priority=16, **changes[incompatibility]
        ),
        lambda: "incompatible",
    )
    assert (
        select_kernel(
            "test_weight",
            "apply",
            SOURCE,
            platform=h100_platform,
            features=frozenset({"required"}),
            traits={"group_size": 128},
            solution="wanted",
            ignore_layout=True,
        ).name
        == "compatible"
    )


def test_registration_supports_canonical_and_transformed_storage(h100_platform):
    register_kernel(
        "test_weight",
        "apply",
        name="both",
        solution="test",
        signatures=format_signatures(
            "weight", "dense", {torch.float32}, {None, _prepare}
        ),
        weight_preprocessor=(("weight",), _prepare),
    )(lambda: "both")
    for signature in (
        SOURCE,
        PACKED,
        replace(SOURCE, layouts=frozenset({None, _prepare})),
    ):
        assert (
            select_kernel("test_weight", "apply", signature, platform=h100_platform)()
            == "both"
        )
    assert KernelRegistry.get().get_by_name("both").weight_preprocessor == (
        ("weight",),
        _prepare,
    )


def test_shared_layout_does_not_pin_execution_to_initialization_winner(h100_platform):
    _register("small", PACKED, traits={"rows": frozenset({1})})
    _register("large", PACKED, traits={"rows": frozenset({256})})
    assert (
        select_kernel(
            "test_weight",
            "apply",
            SOURCE,
            platform=h100_platform,
            traits={"rows": 1},
            ignore_layout=True,
        ).name
        == "small"
    )
    assert (
        select_kernel(
            "test_weight",
            "apply",
            PACKED,
            platform=h100_platform,
            traits={"rows": 256},
        ).name
        == "large"
    )


def test_explicit_override_keeps_compatibility_bypass(h100_platform):
    _register("forced", PACKED)
    for ignore_layout in (False, True):
        assert (
            select_kernel(
                "test_weight",
                "apply",
                SOURCE,
                platform=h100_platform,
                override="forced",
                ignore_layout=ignore_layout,
            ).name
            == "forced"
        )


def test_legacy_preprocessor_requires_initialization_or_pinned_execution(h100_platform):
    _register("legacy", SOURCE, weight_preprocessor=_prepare)
    assert (
        select_kernel(
            "test_weight", "apply", SOURCE, platform=h100_platform, ignore_layout=True
        ).name
        == "legacy"
    )
    with pytest.raises(NoKernelFoundError):
        select_kernel("test_weight", "apply", SOURCE, platform=h100_platform)
    assert (
        select_kernel(
            "test_weight", "apply", SOURCE, platform=h100_platform, override="legacy"
        ).name
        == "legacy"
    )
