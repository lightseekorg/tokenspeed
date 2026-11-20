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

"""Accepted-state fusion against an independent double-precision recurrence."""

import pytest
import torch
from test_gdn_wy_cute_dsl import make_case, reference
from tokenspeed_kernel.ops.attention.gdn import GdnPendingReplay, gdn_decode_mtp
from utils import assert_no_cute_compile, move_pages_past_int32


def accepted_state(case, counts):
    k = case["k"].double()
    if case["use_qk_l2norm"]:
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k.repeat_interleave(case["v"].shape[2] // k.shape[2], 2)
    decay = (
        -case["A_log"].double().exp()
        * torch.nn.functional.softplus(case["a"].double() + case["dt_bias"].double())
    ).exp()
    beta = case["b"].double().sigmoid()
    h = case["initial_state"][case["initial_state_indices"].long()].double()
    for t in range(4):
        decayed = h * decay[:, t, :, None, None]
        update = (
            case["v"][:, t].double() - (decayed * k[:, t, :, None, :]).sum(-1)
        ) * beta[:, t, :, None]
        changed = decayed + update[..., None] * k[:, t, :, None, :]
        h = torch.where((counts > t)[:, None, None, None], changed, h)
    return h


@pytest.mark.parametrize(
    ("strided", "state_scale", "strong_gate", "normalize"),
    [
        (False, 0.02, False, True),
        (False, 4.0, False, True),
        (False, 0.02, True, False),
        (True, 0.02, False, True),
        (True, 4.0, False, False),
        (True, 0.02, True, True),
    ],
)
def test_deferred_wy(state_scale, strong_gate, strided, normalize):
    case = make_case(
        batch=5,
        tokens=4,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=False,
        strided=strided,
        heads=1,
        vheads=4,
    )
    case["use_qk_l2norm"] = normalize
    if not normalize:
        case["q"].mul_(128**-0.5)
        case["k"].mul_(128**-0.5)
    case["initial_state"].mul_(state_scale / 0.02)
    if strong_gate:
        case["a"].fill_(10)
    counts = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.int32)
    expected_state = accepted_state(case, counts)
    pending = make_pending(case, counts, capacity=5, bank=0)
    # Previous acceptance and current verification must use different inputs.
    # Otherwise replaying current K/V instead of the retained bank can pass.
    case["q"].neg_()
    case["k"].mul_(-0.75)
    case["v"].add_(0.125)
    case["a"].add_(0.25)
    case["b"].sub_(0.375)
    indices = case["initial_state_indices"]
    expected_case = dict(case)
    pool = case["initial_state"].double().clone()
    pool[indices.long()] = expected_state
    expected_case["initial_state"] = pool
    expected = reference(expected_case)
    actual = gdn_decode_mtp(**case, pending_replay=pending)
    torch.testing.assert_close(actual.double(), expected, rtol=8e-3, atol=5e-4)
    torch.testing.assert_close(
        case["initial_state"][indices.long()],
        expected_state.bfloat16(),
        rtol=8e-3,
        atol=5e-4,
    )


def make_pending(case, counts, *, capacity, bank):
    batch, tokens, heads, dim = case["k"].shape
    vheads = case["v"].shape[2]
    width = heads * dim + vheads * 128 + 2 * vheads
    payload = torch.zeros(
        2, 1, capacity * 4, width, device="cuda", dtype=torch.bfloat16
    )
    payload[bank, 0, : batch * 4].copy_(
        torch.cat(
            [
                case["k"].reshape(batch * tokens, -1),
                case["v"].reshape(batch * tokens, -1),
                case["a"].reshape(batch * tokens, -1),
                case["b"].reshape(batch * tokens, -1),
            ],
            -1,
        )
    )
    steps = torch.zeros(2, capacity, device="cuda", dtype=torch.int32)
    pages = torch.full((2, 1, capacity), -1, device="cuda", dtype=torch.int32)
    steps[bank, :batch].copy_(counts)
    pages[bank, 0, :batch].copy_(case["initial_state_indices"])
    return GdnPendingReplay(
        payload,
        0,
        torch.arange(batch, device="cuda", dtype=torch.int32) + bank * capacity,
        steps,
        pages,
        torch.tensor([1 - bank], device="cuda", dtype=torch.int32),
    )


def test_deferred_wy_runtime_batch_capacity_bank_and_scale_do_not_recompile():
    cases = []
    for batch, capacity, bank in ((1, 2, 0), (3, 17, 1), (17, 48, 0)):
        case = make_case(
            batch=batch,
            tokens=4,
            dtype=torch.bfloat16,
            state_dtype=torch.bfloat16,
            tree=False,
            strided=False,
            heads=1,
            vheads=4,
        )
        counts = torch.arange(batch, device="cuda", dtype=torch.int32) % 5
        pending = make_pending(case, counts, capacity=capacity, bank=bank)
        cases.append((case, pending))
    gdn_decode_mtp(**cases[0][0], pending_replay=cases[0][1])
    with assert_no_cute_compile():
        for case, pending in cases[1:]:
            case["scale"] = 0.23
            result = gdn_decode_mtp(**case, pending_replay=pending)
            assert torch.isfinite(result).all().item()


def test_deferred_wy_state_offsets_exceed_int32():
    case = make_case(
        batch=2,
        tokens=4,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=False,
        strided=False,
        heads=1,
        vheads=4,
    )
    counts = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    expected_state = accepted_state(case, counts)
    expected_case = dict(case)
    pool = case["initial_state"].double().clone()
    pool[case["initial_state_indices"].long()] = expected_state
    expected_case["initial_state"] = pool
    expected = reference(expected_case)
    (large,), indices = move_pages_past_int32(
        (case["initial_state"],), case["initial_state_indices"]
    )
    case["initial_state"], case["initial_state_indices"] = large, indices
    pending = make_pending(case, counts, capacity=3, bank=1)
    output = gdn_decode_mtp(**case, pending_replay=pending)
    torch.testing.assert_close(output.double(), expected, rtol=8e-3, atol=5e-4)
    torch.testing.assert_close(
        large[indices.long()], expected_state.bfloat16(), rtol=8e-3, atol=5e-4
    )


@pytest.mark.parametrize("batch", [1, 16])
def test_deferred_wy_real_model_geometry(batch):
    case = make_case(
        batch=batch,
        tokens=4,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        tree=False,
        strided=False,
        heads=16,
        vheads=48,
    )
    counts = torch.arange(batch, device="cuda", dtype=torch.int32) % 5
    if batch == 1:
        counts.fill_(4)
    expected_state = accepted_state(case, counts)
    pending = make_pending(case, counts, capacity=batch + 1, bank=1)
    case["k"].neg_()
    case["v"].add_(0.125)
    expected_case = dict(case)
    indices = case["initial_state_indices"].long()
    pool = case["initial_state"].double().clone()
    pool[indices] = expected_state
    expected_case["initial_state"] = pool
    actual = gdn_decode_mtp(**case, pending_replay=pending)
    torch.testing.assert_close(
        actual.double(), reference(expected_case), rtol=8e-3, atol=5e-4
    )
    torch.testing.assert_close(
        case["initial_state"][indices], expected_state.bfloat16(), rtol=8e-3, atol=5e-4
    )
