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

"""Accepted-prefix publication and convolution commit for retained KDA history."""

import pytest
import torch
from kda_retained_reference import (
    check_prefix,
    commit,
    endpoint,
    make_case,
    materialize,
    reference,
    replay_state,
    ring_taps,
    verify,
)
from tokenspeed_kernel.ops.metadata import REQUEST_ENDED

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


#: The registry entries behind the probe and the no-store fused verify are
#: vendor-gated; the shared records tests exercise the selected implementation.


# K3 TP8 rank geometry, trimmed to 4 heads to keep the reference loop quick.
HV, K, V, D_FA = 4, 128, 128, 128


@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("cross_page", [False, True])
def test_retained_commit_matches_every_accepted_prefix(width, cross_page):
    case = make_case(width + 2, width, cross_page=cross_page, strided=True)
    expected = reference(case)
    output = verify(case)
    torch.testing.assert_close(output[0].float(), expected[0], atol=2e-4, rtol=1e-2)
    accepted = torch.arange(width + 2, device="cuda", dtype=torch.int32)
    commit(case, accepted, ended_slots=())
    check_prefix(case, accepted, expected)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("cross_page", [False, True])
def test_ended_request_commit_leaves_a_dense_endpoint(width, cross_page):
    """The forward that ends a request folds its records and ring into its live page.

    Rows 0 and 2 end: their pages alone hold the accepted endpoint, recurrent
    and convolution state, and their slots count no record and hold no ring
    rows. Row 1 keeps its records and its window in its ring.
    """
    case = make_case(3, width, cross_page=cross_page, strided=True)
    expected = reference(case)
    verify(case)
    accepted = torch.tensor([width, 1, 0], device="cuda", dtype=torch.int32)
    ends = commit(case, accepted, ended_slots=(0, 2))
    # Both ended requests' states are complete, so their finishes may publish.
    assert ends.tolist() == [REQUEST_ENDED, 0, REQUEST_ENDED]
    history = case["records"].history
    # Before a cross-page verify, the prepare copied row 1's dense base to its
    # new live page; its two accepted records stay valid on top of it.
    assert history.length[:, 0].tolist() == [0, 2 + 1, 0]
    assert replay_state(history)[:, 1].tolist() == [0, (3 + 1) % (width + 3) + 1, 0]
    # Each live page plus its slot's remaining records is the endpoint.
    check_prefix(case, accepted, expected, ended_slots=(0, 2))


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("cross_page", [False, True])
def test_retraction_writes_the_rings_window_into_the_live_page(width, cross_page):
    """Decode commits leave the pages' convolution state; a retraction fold writes it.

    After two rounds, each live page plus its slot's records is the
    endpoint and the ring holds the window. Materialization folds both into
    the live page, which then alone is the endpoint, and clears the slot.
    """
    case = make_case(4, width, cross_page=cross_page, strided=True)
    for round_index, take in enumerate((width, max(1, width - 1))):
        expected = reference(case)
        verify(case)
        accepted = torch.full((4,), take, device="cuda", dtype=torch.int32)
        accepted[round_index] = 1
        commit(case, accepted, ended_slots=())
        check_prefix(case, accepted, expected)
        case["raw"].mul_(0.9)
    history = case["records"].history
    endpoints = [
        endpoint(case["state"][page], history, row)
        for row, page in enumerate(history.live_state_indices.tolist())
    ]
    taps = [
        ring_taps(history, case["records"].preconv, row).clone() for row in range(4)
    ]
    materialize(case)
    assert replay_state(history).tolist() == [[0, 0]] * 4
    for row, page in enumerate(history.live_state_indices.tolist()):
        torch.testing.assert_close(case["conv"][page], taps[row], atol=0, rtol=0)
        torch.testing.assert_close(
            case["state"][page].float(), endpoints[row], atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("accepted_history,width", [(12, 4), (13, 4), (15, 1)])
def test_retained_capacity_boundary_and_rejected_slot_reuse(accepted_history, width):
    case = make_case(3, width, accepted_history=accepted_history)
    for take in (0, 1, width):
        expected = reference(case)
        output = verify(case)
        torch.testing.assert_close(output[0].float(), expected[0], atol=2e-4, rtol=1e-2)
        accepted = torch.full((3,), take, device="cuda", dtype=torch.int32)
        commit(case, accepted, ended_slots=())
        check_prefix(case, accepted, expected)
        case["raw"].mul_(0.9)


def test_retained_commit_rejects_stale_candidate_stamps():
    case = make_case(3, 4)
    verify(case)
    history = case["records"].history
    before_conv, before_length = case["conv"].clone(), history.length.clone()
    case["records"].stamps[..., 1].fill_(-1)
    commit(case, torch.ones(3, device="cuda", dtype=torch.int32), ended_slots=())
    torch.testing.assert_close(case["conv"], before_conv, atol=0, rtol=0)
    # Every layer shares the count, which the commit publishes whatever one
    # layer's stamps say: a stale stamp means that layer's verify did not
    # run, which serving never does.
    torch.testing.assert_close(history.length, before_length + 1, atol=0, rtol=0)


def test_in_place_commit_full_pool_accounting():
    case = make_case(4, 4, ragged=True, strided=True)
    expected = reference(case)
    before_state, before_conv = case["state"].clone(), case["conv"].clone()
    verify(case)
    accepted = torch.tensor([0, 1, 9, 0], device="cuda", dtype=torch.int32)
    commit(case, accepted, ended_slots=())
    check_prefix(case, accepted, expected)
    torch.testing.assert_close(case["state"], before_state, atol=0, rtol=0)
    for page in (0, 4, 5, 6, 7, 8, 9, 10):
        torch.testing.assert_close(
            case["conv"][page], before_conv[page], atol=0, rtol=0
        )
