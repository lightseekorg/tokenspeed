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

"""Streaming collection must preserve repeated token deltas."""

import pytest

from tokenspeed.runtime.engine.collector import RequestOutputCollector


@pytest.mark.parametrize("deltas", [([7, 7], [7, 7]), ([7], [7, 8])])
def test_repeated_token_deltas_are_not_cumulative_snapshots(deltas):
    collector = RequestOutputCollector()
    total = 0
    for ids in deltas:
        total += len(ids)
        collector.put(
            {"output_ids": ids, "meta_info": {"completion_tokens": total}},
            stream=True,
            token_ids_are_delta=True,
        )
    result = collector.take()
    assert result["output_ids"] == [token for ids in deltas for token in ids]
    assert len(result["output_ids"]) == result["meta_info"]["completion_tokens"]


def test_cumulative_token_frames_merge_without_duplicate_prefixes():
    collector = RequestOutputCollector()
    for ids in ([7, 7], [7, 7, 8]):
        collector.put({"output_ids": ids}, stream=True, token_ids_are_delta=False)
    assert collector.take()["output_ids"] == [7, 7, 8]


def test_delta_merges_own_their_token_lists_and_reset_after_take():
    collector = RequestOutputCollector()
    frame = {"output_ids": [7], "output_multi_ids": [[7, 8]]}
    for _ in range(3):
        collector.put(frame, stream=True, token_ids_are_delta=True)
    result = collector.take()
    assert result["output_ids"] == [7, 7, 7]
    assert result["output_multi_ids"] == [[7, 8], [7, 8], [7, 8]]
    assert frame == {"output_ids": [7], "output_multi_ids": [[7, 8]]}
    collector.put(frame, stream=True, token_ids_are_delta=True)
    assert collector.take() == frame
