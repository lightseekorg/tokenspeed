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
from tokenspeed.runtime.execution.drafter.eagle import Eagle

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class OverlapDraft(nn.Module):
    """Stand in for model weights when checking the startup capability contract."""

    def __init__(self, *, supports_draft_overlap: bool):
        super().__init__()
        self.supports_draft_overlap = supports_draft_overlap


def _drafter(steps: int) -> Eagle:
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


GRAPHS = "Draft graph overlap requires NVIDIA decode graphs, DP1/PP1"
EAGLE = "Draft graph overlap requires the Eagle MTP drafter"
UNAUDITED = "Draft model does not declare support for draft overlap"
STEPS = "Draft graph overlap requires at least two draft steps"
GREEDY = "Draft graph overlap requires a greedy draft chain"
HISTORY = "Draft graph overlap does not support draft token history"
GRAMMAR = "Draft graph overlap does not support grammar"
MASKS = "Draft graph overlap does not support shared MoE row masks"
LAYERS = "Draft graph overlap requires separate target/draft cache layers and metadata"


class EagleSubclass(Eagle):
    """A drafter derived from Eagle; the overlap was audited for Eagle itself."""


def _subclass() -> Eagle:
    drafter = _drafter(3)
    drafter.__class__ = EagleSubclass
    return drafter


def _tree_and_history() -> Eagle:
    drafter = _drafter(3)
    drafter.tree_spec = object()
    drafter.draft_reads_token_history = True
    return drafter


def _history() -> Eagle:
    drafter = _drafter(3)
    drafter.draft_reads_token_history = True
    return drafter


def _unread() -> tuple[set[int], set[int]]:
    raise AssertionError("cache layers read before the drafter conditions hold")


def _eligible(**changes) -> dict:
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
    ("changes", "first"),
    [
        (dict(nvidia_decode_graphs=False, spec_algo=None, drafter=None), GRAPHS),
        (dict(pp_size=2), GRAPHS),
        (dict(dp_size=2), GRAPHS),
        (dict(spec_algo=None, grammar_backend="xgrammar"), EAGLE),
        (dict(spec_algo="DFLASH"), EAGLE),
        (dict(drafter=None), EAGLE),
        (dict(drafter=_subclass()), EAGLE),
        (dict(drafter=_drafter(1)), STEPS),
        (dict(drafter=_tree_and_history()), GREEDY),
        (dict(drafter=_history(), grammar_backend="xgrammar"), HISTORY),
        (dict(grammar_backend="xgrammar", shared_moe_row_masks=True), GRAMMAR),
        (dict(shared_moe_row_masks=True, cache_layers=_unread), MASKS),
        (dict(cache_layers=lambda: ({0, 1}, {1})), LAYERS),
        (dict(cache_layers=lambda: ({0, 1}, set())), LAYERS),
        (dict(cache_layers=lambda: (set(), {3})), LAYERS),
        (dict(separate_metadata=False), LAYERS),
    ],
)
def test_the_first_unmet_condition_is_named_exactly(changes, first):
    """Two broken conditions at once pin the order the startup checks always had."""
    assert draft_overlap_ineligibility(**_eligible(**changes)) == first


def test_the_drafter_is_checked_before_the_cache_layers():
    drafter = _drafter(3)
    drafter.draft_model_runner.model = OverlapDraft(supports_draft_overlap=False)
    config = _eligible(drafter=drafter, cache_layers=_unread)
    assert draft_overlap_ineligibility(**config) == UNAUDITED


def test_a_configuration_without_an_eagle_drafter_never_reads_its_cache_layers():
    """Other models' KV pools are not asked for draft layers they do not have."""
    config = _eligible(spec_algo=None, drafter=None, cache_layers=_unread)
    assert draft_overlap_ineligibility(**config) == EAGLE


def _ineligible() -> str:
    return GRAMMAR


@pytest.fixture
def overlap_log(caplog):
    # The runtime's loggers do not propagate to the root logger.
    logger = logging.getLogger("tokenspeed.runtime.execution._draft_overlap")
    logger.addHandler(caplog.handler)
    with caplog.at_level(logging.INFO, logger=logger.name):
        yield caplog
    logger.removeHandler(caplog.handler)


def test_unset_turns_the_overlap_on_where_eligible_and_says_so(overlap_log):
    assert resolve_draft_overlap(None, lambda: None, log=True) is True
    assert "Draft graph overlap on" in overlap_log.text


def test_unset_leaves_an_ineligible_configuration_off_and_says_why(overlap_log):
    assert resolve_draft_overlap(None, _ineligible, log=True) is False
    assert GRAMMAR in overlap_log.text


def test_only_the_logging_rank_logs(overlap_log):
    assert resolve_draft_overlap(None, _ineligible, log=False) is False
    assert resolve_draft_overlap(None, lambda: None, log=False) is True
    assert "Draft graph overlap" not in overlap_log.text


def test_one_requires_the_overlap():
    assert resolve_draft_overlap("1", lambda: None, log=False) is True
    with pytest.raises(ValueError) as error:
        resolve_draft_overlap("1", _ineligible, log=False)
    assert str(error.value) == GRAMMAR


def test_zero_turns_the_overlap_off_without_checking(overlap_log):
    def unchecked() -> str | None:
        raise AssertionError("eligibility must not be checked")

    assert resolve_draft_overlap("0", unchecked, log=True) is False
    assert "TOKENSPEED_DRAFT_GRAPH_OVERLAP=0" in overlap_log.text


@pytest.mark.parametrize("setting", ["", "2", "true", "on"])
def test_other_values_are_startup_errors(setting):
    with pytest.raises(ValueError, match="must be 0 or 1"):
        resolve_draft_overlap(setting, lambda: None, log=False)
