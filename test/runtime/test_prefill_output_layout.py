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

import pytest

from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout


@pytest.mark.parametrize("width", [1, 6])
def test_mixed_output_offsets_preserve_original_request_rows(width):
    layout = ForwardOutputLayout.from_prefill(
        prefix_lengths=[64, 0],
        input_lengths=[128, 128],
        prompt_lengths=[192, 1024],
        num_decodes=2,
        decode_width=width,
    )
    assert layout.num_prefill_outputs == 1
    assert layout.num_extends == 2
    assert layout.num_output_tokens == 1 + 2 * width
    assert [layout.token_offset(i) for i in range(4)] == [0, 1, 1, 1 + width]


def test_open_chunk_has_no_token_storage():
    layout = ForwardOutputLayout.from_prefill(
        prefix_lengths=[0],
        input_lengths=[128],
        prompt_lengths=[1024],
        num_decodes=0,
        decode_width=6,
    )
    assert layout.num_prefill_outputs == 0
    assert layout.num_output_tokens == 0
    assert layout.token_offset(0) == 0


def test_non_prefix_completions_are_rejected():
    with pytest.raises(ValueError, match="prefix"):
        ForwardOutputLayout.from_prefill(
            prefix_lengths=[0, 0],
            input_lengths=[128, 64],
            prompt_lengths=[1024, 64],
            num_decodes=1,
            decode_width=1,
        )


def test_finishing_replayed_window_uses_current_prefill_target():
    layout = ForwardOutputLayout.from_prefill(
        prefix_lengths=[128],
        input_lengths=[128],
        prompt_lengths=[256],
        num_decodes=1,
        decode_width=6,
    )
    assert layout.num_prefill_outputs == layout.num_extends == 1
    assert [layout.token_offset(i) for i in range(2)] == [0, 1]


@pytest.mark.parametrize(
    "layout,request_names,token_slots,prefills,decodes,widths,per_request",
    [
        (
            ForwardOutputLayout(2, 1, 2, 3),
            ["A", "B", "C", "D"],
            [11, 21, 22, 23, 31, 32, 33],
            (["A"], [11]),
            (["C", "D"], [21, 22, 23, 31, 32, 33]),
            [1, 0, 3, 3],
            [[11], [], [21, 22, 23], [31, 32, 33]],
        ),
        (ForwardOutputLayout(1, 0, 0, 3), ["B"], [], ([], []), ([], []), [0], [[]]),
        (
            ForwardOutputLayout(1, 1, 1, 1),
            ["A", "C"],
            [11, 21],
            (["A"], [11]),
            (["C"], [21]),
            [1, 1],
            [[11], [21]],
        ),
    ],
)
def test_output_ranges_preserve_request_identity_and_fixed_capacity(
    layout, request_names, token_slots, prefills, decodes, widths, per_request
):
    assert (
        request_names[layout.prefill_slice],
        token_slots[layout.prefill_slice],
    ) == prefills
    assert (
        request_names[layout.decode_request_slice],
        token_slots[layout.decode_output_slice],
    ) == decodes
    assert [layout.output_width(i) for i in range(len(request_names))] == widths
    assert [
        token_slots[
            layout.token_offset(i) : layout.token_offset(i) + layout.output_width(i)
        ]
        for i in range(len(request_names))
    ] == per_request
