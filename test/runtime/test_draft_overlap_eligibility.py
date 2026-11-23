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

"""Which configurations run decode graphs with draft-tail overlap.

The overlap is on wherever the configuration meets every condition, unless
TOKENSPEED_DRAFT_GRAPH_OVERLAP=0; =1 makes a missed condition a startup error.
"""

import logging
from test.ci_system.ci_register import register_cuda_ci
from test.runtime.drafter_prefill_graph_test_utils import make_eagle

import pytest
from torch import nn

from tokenspeed.runtime.execution._draft_overlap import (
    draft_overlap_ineligibility,
    resolve_draft_overlap,
)

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class OverlapDraft(nn.Module):
    """Stand in for model weights when checking the startup capability contract."""

    def __init__(self, *, supports_draft_overlap: bool):
        super().__init__()
        self.supports_draft_overlap = supports_draft_overlap


def _drafter(steps):
    drafter, _ = make_eagle(spec_num_steps=steps)
    drafter.draft_model_runner.model = OverlapDraft(supports_draft_overlap=True)
    return drafter


@pytest.mark.parametrize("steps", [2, 3, 5])
def test_multistep_chain_can_overlap(steps):
    assert _drafter(steps).decode_overlap_ineligibility() is None


def test_one_step_has_no_tail_to_overlap():
    assert "at least two" in _drafter(1).decode_overlap_ineligibility()


@pytest.mark.parametrize("field", ["tree_spec", "draft_sampler"])
def test_tree_and_sampled_drafts_cannot_overlap(field):
    drafter = _drafter(3)
    # These objects stand in for GPU-backed tree/sampler state; only presence matters.
    if field == "tree_spec":
        drafter.tree_spec = object()
    else:
        drafter.draft_sampler = object()
    assert "greedy draft chain" in drafter.decode_overlap_ineligibility()


def test_draft_token_history_cannot_overlap():
    drafter = _drafter(3)
    drafter.draft_reads_token_history = True
    assert "draft token history" in drafter.decode_overlap_ineligibility()


def test_unaudited_model_cannot_overlap():
    drafter, _ = make_eagle()
    assert "does not declare support" in drafter.decode_overlap_ineligibility()


def test_model_can_decline_overlap():
    drafter = _drafter(3)
    drafter.draft_model_runner.model = OverlapDraft(supports_draft_overlap=False)
    assert "does not declare support" in drafter.decode_overlap_ineligibility()


def _eligible(**changes):
    config = dict(
        nvidia_decode_graphs=True,
        pp_size=1,
        dp_size=1,
        spec_algo="MTP",
        drafter=_drafter(3),
        grammar_backend="none",
        shared_moe_row_masks=False,
        cache_layers=lambda: ({0, 1, 2}, {3}),
        separate_metadata=True,
    )
    config.update(changes)
    return config


def test_an_eagle_mtp_decode_configuration_is_eligible():
    assert draft_overlap_ineligibility(**_eligible()) is None


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"nvidia_decode_graphs": False}, "NVIDIA decode graphs"),
        ({"pp_size": 2}, "DP1/PP1"),
        ({"dp_size": 2}, "DP1/PP1"),
        ({"spec_algo": None, "drafter": None}, "Eagle MTP drafter"),
        ({"spec_algo": "DFLASH"}, "Eagle MTP drafter"),
        ({"drafter": None}, "Eagle MTP drafter"),
        ({"drafter": _drafter(1)}, "at least two draft steps"),
        ({"grammar_backend": "xgrammar"}, "does not support grammar"),
        ({"shared_moe_row_masks": True}, "shared MoE row masks"),
        ({"cache_layers": lambda: ({0, 1}, {1})}, "separate target/draft"),
        ({"cache_layers": lambda: ({0, 1}, set())}, "separate target/draft"),
        ({"separate_metadata": False}, "separate target/draft"),
    ],
)
def test_each_unmet_condition_is_named(change, reason):
    assert reason in draft_overlap_ineligibility(**_eligible(**change))


def test_a_configuration_without_an_eagle_drafter_never_reads_its_cache_layers():
    """Other models' KV pools are not asked for draft layers they do not have."""
    reads = []
    config = _eligible(
        spec_algo=None, drafter=None, cache_layers=lambda: reads.append(1)
    )
    assert draft_overlap_ineligibility(**config) is not None
    assert reads == []


def _ineligible():
    return "Draft graph overlap requires NVIDIA decode graphs, DP1/PP1"


@pytest.fixture
def overlap_log(caplog):
    # The runtime's loggers do not propagate to the root logger.
    logger = logging.getLogger("tokenspeed.runtime.execution._draft_overlap")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


def test_unset_turns_the_overlap_on_where_eligible():
    assert resolve_draft_overlap(None, lambda: None) is True


def test_unset_leaves_an_ineligible_configuration_off_and_says_why(overlap_log):
    assert resolve_draft_overlap(None, _ineligible) is False
    assert _ineligible() in overlap_log.text


def test_one_requires_the_overlap():
    assert resolve_draft_overlap("1", lambda: None) is True
    with pytest.raises(ValueError, match="NVIDIA decode graphs"):
        resolve_draft_overlap("1", _ineligible)


def test_zero_turns_the_overlap_off_without_checking():
    def unchecked():
        raise AssertionError("eligibility must not be checked")

    assert resolve_draft_overlap("0", unchecked) is False


@pytest.mark.parametrize("setting", ["", "2", "true", "on"])
def test_other_values_are_startup_errors(setting):
    with pytest.raises(ValueError, match="must be 0 or 1"):
        resolve_draft_overlap(setting, lambda: None)
