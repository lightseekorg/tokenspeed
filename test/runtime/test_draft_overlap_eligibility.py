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

"""Check which draft paths can release the execution stream after step one."""

import pytest
from drafter_prefill_graph_test_utils import make_eagle
from torch import nn


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
    _drafter(steps).validate_decode_overlap()


def test_one_step_has_no_tail_to_overlap():
    with pytest.raises(ValueError, match="at least two"):
        _drafter(1).validate_decode_overlap()


@pytest.mark.parametrize("field", ["tree_spec", "draft_sampler"])
def test_tree_and_sampled_drafts_cannot_overlap(field):
    drafter = _drafter(3)
    # These objects stand in for GPU-backed tree/sampler state; only presence matters.
    if field == "tree_spec":
        drafter.tree_spec = object()
    else:
        drafter.draft_sampler = object()
    with pytest.raises(ValueError, match="greedy draft chain"):
        drafter.validate_decode_overlap()


def test_draft_token_history_cannot_overlap():
    drafter = _drafter(3)
    drafter.draft_reads_token_history = True
    with pytest.raises(ValueError, match="draft token history"):
        drafter.validate_decode_overlap()


def test_unaudited_model_cannot_overlap():
    drafter, _ = make_eagle()
    with pytest.raises(ValueError, match="does not declare support"):
        drafter.validate_decode_overlap()


def test_model_can_decline_overlap():
    drafter = _drafter(3)
    drafter.draft_model_runner.model = OverlapDraft(supports_draft_overlap=False)
    with pytest.raises(ValueError, match="does not declare support"):
        drafter.validate_decode_overlap()
