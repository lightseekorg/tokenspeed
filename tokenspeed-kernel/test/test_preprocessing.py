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
from tokenspeed_kernel.preprocessing import (
    WeightPreprocessorConflictError,
    WeightPreprocessorRequest,
    resolve_weight_preprocessor_conflict,
)


def layout_a(**_):
    return None


def layout_b(**_):
    return None


def layout_base(**_):
    return None


def layout_fast(**_):
    return None


def test_preprocessor_request_rejects_empty_candidate_list():
    with pytest.raises(ValueError, match="at least one"):
        WeightPreprocessorRequest(())


def test_preprocessor_request_normalizes_single_callable():
    request = WeightPreprocessorRequest(layout_a, required=False)

    assert request.preprocessors == (layout_a,)
    assert request.preprocessor is layout_a
    assert request.required is False


def test_preprocessor_request_preserves_ordered_candidates():
    request = WeightPreprocessorRequest((layout_fast, layout_base))

    assert request.preprocessors == (layout_fast, layout_base)
    assert request.preprocessor is layout_fast


def test_preprocessor_request_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        WeightPreprocessorRequest(("layout_a",))


def test_preprocessor_request_rejects_duplicate_candidates():
    with pytest.raises(ValueError, match="unique"):
        WeightPreprocessorRequest((layout_a, layout_a))


def test_preprocessor_request_requires_bool_required():
    with pytest.raises(TypeError, match="required"):
        WeightPreprocessorRequest(layout_a, required="yes")


def test_same_preprocessor_requests_dedupe_and_preserve_required():
    optional_request = WeightPreprocessorRequest(layout_a, required=False)
    required_request = WeightPreprocessorRequest(layout_a, required=True)

    request = resolve_weight_preprocessor_conflict([optional_request, required_request])

    assert request == WeightPreprocessorRequest(layout_a, required=True)


def test_conflicting_requests_choose_highest_common_preprocessor():
    fast_or_base = WeightPreprocessorRequest((layout_fast, layout_base))
    base_only = WeightPreprocessorRequest(layout_base, required=False)

    request = resolve_weight_preprocessor_conflict([fast_or_base, base_only])

    assert request == WeightPreprocessorRequest(layout_base, required=True)


def test_optional_conflicting_preprocessors_warn_and_skip():
    with pytest.warns(
        RuntimeWarning,
        match="skipping conflicting preprocessors.*hurt performance",
    ):
        request_a = WeightPreprocessorRequest(layout_a, required=False)
        request_b = WeightPreprocessorRequest(layout_b, required=False)

        request = resolve_weight_preprocessor_conflict([request_a, request_b])

    assert request is None


def test_required_conflicting_preprocessors_error():
    with pytest.raises(WeightPreprocessorConflictError, match="Conflicting"):
        required_request = WeightPreprocessorRequest(layout_a, required=True)
        optional_request = WeightPreprocessorRequest(layout_b, required=False)

        resolve_weight_preprocessor_conflict([required_request, optional_request])


def test_optional_preprocessor_conflicts_with_no_preprocessing_warns_and_skips():
    with pytest.warns(
        RuntimeWarning,
        match="skipping conflicting preprocessors.*hurt performance",
    ):
        request = resolve_weight_preprocessor_conflict(
            [None, WeightPreprocessorRequest(layout_a, required=False)]
        )

    assert request is None


def test_required_preprocessor_conflicts_with_no_preprocessing_errors():
    with pytest.raises(WeightPreprocessorConflictError, match="no-preprocessing"):
        resolve_weight_preprocessor_conflict(
            [None, WeightPreprocessorRequest(layout_a, required=True)]
        )
