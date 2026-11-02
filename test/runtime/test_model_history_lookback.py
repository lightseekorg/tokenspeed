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

"""History seed bounds are optional model facts, shared by target and draft."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, suite="runtime-1gpu")

import pytest

from tokenspeed.runtime.configs.model_profile import (
    ModelProfile,
    resolve_history_seed_lookback,
)


def _profile(**overrides) -> ModelProfile:
    fields = dict(
        configure_attention=lambda model, args: None,
        cache_family="test_history",
        linear_attention=None,
        default_attention_backend=None,
        default_prefix_granularity=None,
        request_token_history=True,
        is_multimodal=False,
        tokenizer_kwargs={},
        attention_instances_per_layer=1,
        numerics_envelopes=frozenset({"auto"}),
    )
    fields.update(overrides)
    return ModelProfile(**fields)


def test_omission_preserves_unbounded_history() -> None:
    profile = _profile()
    assert profile.request_token_history_lookback is None
    assert resolve_history_seed_lookback(profile) is None


@pytest.mark.parametrize("bound", [-1, 1.5, True])
def test_lookback_requires_a_nonnegative_integer(bound) -> None:
    with pytest.raises(ValueError, match="request_token_history_lookback"):
        _profile(request_token_history_lookback=bound)


def test_only_history_readers_can_declare_a_bound() -> None:
    with pytest.raises(ValueError, match="request_token_history_lookback"):
        _profile(request_token_history=False, request_token_history_lookback=0)


@pytest.mark.parametrize(
    "target,draft,expected",
    [(3, 5, 5), (5, 3, 5), (0, 0, 0), (3, None, None), (None, 3, None)],
)
def test_target_and_draft_requirements_are_both_preserved(target, draft, expected):
    assert (
        resolve_history_seed_lookback(
            _profile(request_token_history_lookback=target),
            _profile(request_token_history_lookback=draft),
        )
        == expected
    )


def test_nonreaders_do_not_force_full_seeds() -> None:
    reader = _profile(request_token_history_lookback=3)
    nonreader = _profile(request_token_history=False)
    assert resolve_history_seed_lookback(reader, nonreader, None) == 3
    assert resolve_history_seed_lookback(nonreader, None) is None
    assert resolve_history_seed_lookback(reader, None) == 3
