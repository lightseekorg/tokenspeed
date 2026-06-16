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

import pytest
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.preprocessing import (
    WeightPreprocessorConflictError,
    WeightPreprocessorRegistry,
    WeightPreprocessorResolutionError,
    WeightPreprocessorSpec,
    WeightPreprocessorValidationError,
    register_weight_preprocessor,
    resolve_weight_preprocessor_conflict,
    resolve_weight_preprocessor_ref,
    validate_weight_preprocessor_links,
)
from tokenspeed_kernel.registry import (
    KernelRegistry,
    KernelSpec,
    WeightPreprocessorRef,
)

pytestmark = pytest.mark.usefixtures("fresh_registry")


def _register_apply(
    *,
    name: str = "moe_apply",
    family: str = "moe",
    preprocessor: WeightPreprocessorRef | None = None,
    capability: CapabilityRequirement | None = None,
    weight_dtype: str = "mxfp4",
) -> None:
    KernelRegistry.get().register(
        KernelSpec(
            name=name,
            family=family,
            mode="apply",
            capability=capability or CapabilityRequirement(),
            traits={"weight_dtype": frozenset({weight_dtype})},
            weight_preprocessor=preprocessor,
        ),
        lambda **_: None,
    )


def _register_preprocessor(
    *,
    name: str = "moe_weights",
    family: str = "moe",
    capability: CapabilityRequirement | None = None,
    weight_dtype: str = "mxfp4",
) -> None:
    @register_weight_preprocessor(
        family,
        name=name,
        capability=capability,
        traits={"weight_dtype": frozenset({weight_dtype})},
    )
    def _preprocess(**_):
        return None


def test_preprocessor_registry_has_no_solution_field():
    _register_preprocessor(name="layout_a")

    spec = WeightPreprocessorRegistry.get().get_by_name("layout_a")

    assert spec is not None
    assert spec.name == "layout_a"
    assert spec.family == "moe"
    assert spec.traits["weight_dtype"] == frozenset({"mxfp4"})
    assert not hasattr(spec, "solution")


def test_preprocessor_ref_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        WeightPreprocessorRef("")


def test_preprocessor_ref_requires_bool_required():
    with pytest.raises(TypeError, match="required"):
        WeightPreprocessorRef("layout_a", required="yes")


def test_preprocessor_spec_validates_local_metadata():
    with pytest.raises(ValueError, match="family"):
        WeightPreprocessorSpec(name="layout_a", family="")

    with pytest.raises(TypeError, match="frozenset"):
        WeightPreprocessorSpec(
            name="layout_a",
            family="moe",
            traits={"weight_dtype": {"mxfp4"}},
        )


def test_validation_allows_kernel_before_preprocessor_registration():
    _register_apply(preprocessor=WeightPreprocessorRef("layout_a"))
    _register_preprocessor(name="layout_a")

    validate_weight_preprocessor_links(family="moe", mode="apply")


def test_validation_rejects_missing_preprocessor():
    _register_apply(preprocessor=WeightPreprocessorRef("missing"))

    with pytest.raises(WeightPreprocessorValidationError, match="missing"):
        validate_weight_preprocessor_links(family="moe", mode="apply")


def test_validation_rejects_family_mismatch():
    _register_apply(preprocessor=WeightPreprocessorRef("layout_a"))
    _register_preprocessor(name="layout_a", family="gemm")

    with pytest.raises(WeightPreprocessorValidationError, match="belongs to family"):
        validate_weight_preprocessor_links(family="moe", mode="apply")


def test_validation_rejects_trait_mismatch():
    _register_apply(preprocessor=WeightPreprocessorRef("layout_a"))
    _register_preprocessor(name="layout_a", weight_dtype="nvfp4")

    with pytest.raises(WeightPreprocessorValidationError, match="weight_dtype"):
        validate_weight_preprocessor_links(family="moe", mode="apply")


def test_validation_rejects_capability_mismatch():
    _register_apply(
        preprocessor=WeightPreprocessorRef("layout_a"),
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    )
    _register_preprocessor(
        name="layout_a",
        capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    )

    with pytest.raises(WeightPreprocessorValidationError, match="capability"):
        validate_weight_preprocessor_links(family="moe", mode="apply")


def test_required_capability_mismatch_errors_at_resolution(h100_platform):
    _register_preprocessor(
        name="amd_layout",
        capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    )

    with pytest.raises(WeightPreprocessorResolutionError, match="cannot run"):
        resolve_weight_preprocessor_ref(
            WeightPreprocessorRef("amd_layout", required=True),
            family="moe",
            platform=h100_platform,
        )


def test_optional_capability_mismatch_warns_and_skips(h100_platform):
    _register_preprocessor(
        name="amd_layout",
        capability=CapabilityRequirement(vendors=frozenset({"amd"})),
    )

    with pytest.warns(RuntimeWarning, match="skipping optional preprocessing"):
        resolved = resolve_weight_preprocessor_ref(
            WeightPreprocessorRef("amd_layout", required=False),
            family="moe",
            platform=h100_platform,
        )

    assert resolved is None


def test_missing_preprocessor_errors_at_resolution(h100_platform):
    with pytest.raises(WeightPreprocessorResolutionError, match="missing"):
        resolve_weight_preprocessor_ref(
            WeightPreprocessorRef("missing", required=False),
            family="moe",
            platform=h100_platform,
        )


def test_same_preprocessor_requests_dedupe_and_preserve_required():
    ref = resolve_weight_preprocessor_conflict(
        [
            WeightPreprocessorRef("layout_a", required=False),
            WeightPreprocessorRef("layout_a", required=True),
        ]
    )

    assert ref == WeightPreprocessorRef("layout_a", required=True)


def test_optional_conflicting_preprocessors_warn_and_skip():
    with pytest.warns(RuntimeWarning, match="Skipping optional conflicting"):
        ref = resolve_weight_preprocessor_conflict(
            [
                WeightPreprocessorRef("layout_a", required=False),
                WeightPreprocessorRef("layout_b", required=False),
            ]
        )

    assert ref is None


def test_required_conflicting_preprocessors_error():
    with pytest.raises(WeightPreprocessorConflictError, match="Conflicting"):
        resolve_weight_preprocessor_conflict(
            [
                WeightPreprocessorRef("layout_a", required=True),
                WeightPreprocessorRef("layout_b", required=False),
            ]
        )


def test_optional_preprocessor_conflicts_with_no_preprocessing_warns_and_skips():
    with pytest.warns(RuntimeWarning, match="no-preprocessing constraint"):
        ref = resolve_weight_preprocessor_conflict(
            [
                None,
                WeightPreprocessorRef("layout_a", required=False),
            ]
        )

    assert ref is None


def test_required_preprocessor_conflicts_with_no_preprocessing_errors():
    with pytest.raises(WeightPreprocessorConflictError, match="no-preprocessing"):
        resolve_weight_preprocessor_conflict(
            [
                None,
                WeightPreprocessorRef("layout_a", required=True),
            ]
        )


def test_resolution_accepts_matching_capability(mi300_platform):
    _register_preprocessor(
        name="amd_layout",
        capability=CapabilityRequirement(
            vendors=frozenset({"amd"}),
            min_arch_version=ArchVersion(9, 4),
        ),
    )

    spec = resolve_weight_preprocessor_ref(
        WeightPreprocessorRef("amd_layout", required=True),
        family="moe",
        platform=mi300_platform,
    )

    assert spec is not None
    assert spec.name == "amd_layout"
