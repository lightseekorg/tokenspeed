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

"""Retained verification keeps pre-convolution rows in the ring and projection storage."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from kda_retained_reference import make_case, prepare_history, reference, verify
from tokenspeed_kernel.ops.attention import kda

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.mark.parametrize("batch,width", [(1, 1), (4, 4), (8, 1)])
@pytest.mark.parametrize("strided", [False, True])
def test_retained_capture_matches_original_rows(batch, width, strided):
    """Each request's window lands in its slot's ring after the seeded taps."""
    case = make_case(batch, width, strided=strided, ragged=True)
    original = case["raw"].clone()
    expected = reference(case)
    output = verify(case)
    records = case["records"]
    ring, starts = records.preconv.rows, records.query_start_loc.tolist()
    for row, page in enumerate(case["reads"].tolist()):
        slot = int(records.history.record_slots[row])
        begin, end = starts[row], starts[row + 1]
        if page <= 0:
            continue
        torch.testing.assert_close(ring[slot, :3].T, case["conv"][page], atol=0, rtol=0)
        torch.testing.assert_close(
            ring[slot, 3 : 3 + end - begin], original[begin:end], atol=0, rtol=0
        )
    torch.testing.assert_close(case["raw"], original, atol=0, rtol=0)
    torch.testing.assert_close(output[0].float(), expected[0], atol=2e-4, rtol=1e-2)


def test_retained_capture_rejects_overlapping_projection_storage():
    case = make_case(2, 4)
    ring = case["records"].preconv.rows
    # A ring that shares the projection rows' storage.
    storage = torch.zeros(
        case["raw"].shape[0] + ring.numel() // ring.shape[-1],
        ring.shape[-1],
        device="cuda",
        dtype=ring.dtype,
    )
    case["raw"] = storage[: case["raw"].shape[0]].copy_(case["raw"])
    overlapping = storage[2 : 2 + ring.numel() // ring.shape[-1]].view_as(ring)
    case["records"] = replace(
        case["records"],
        preconv=kda.KdaPreconvRing(overlapping, case["records"].preconv.heads),
    )
    # The round's prepare seeds the (overlapping) ring rows before any verify;
    # running it again writes the same seeds. The verify itself must refuse
    # before it writes a row.
    records = case["records"]
    prepare_history(
        case["state"],
        records.history,
        case["reads"],
        records.query_start_loc,
        draft_token_num=case["width"],
        conv=case["conv"],
        preconv=records.preconv,
    )
    before = case["raw"].clone()
    with pytest.raises(ValueError, match="overlap|separate"):
        verify(case)
    torch.testing.assert_close(case["raw"], before, atol=0, rtol=0)


def test_retained_verify_requires_the_ring_to_convolve():
    """A verify that convolves takes its history from the ring: none is an error."""
    case = make_case(2, 4)
    records = replace(case["records"], preconv=None)
    with pytest.raises(ValueError, match="pre-convolution ring"):
        kda.kda_fused_paged_verify(
            case["raw"],
            case["weights"],
            None,
            None,
            case["beta"],
            case["a"],
            case["bias"],
            state_pool=case["state"],
            read_indices=case["reads"],
            num_heads=case["heads"],
            head_dim=128,
            draft_token_num=case["width"],
            lower_bound=case["lower_bound"],
            recurrent_layout="v_major",
            retained_records=records,
            output_norm=None,
            g_raw=case["gate"],
            override=None,
            solution=None,
        )
