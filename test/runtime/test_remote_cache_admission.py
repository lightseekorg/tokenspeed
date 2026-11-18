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

import numpy as np
import pytest

from tokenspeed.runtime.layers.attention.remote_cache import (
    copy_remote_cache_admissions,
)


def test_slot_reuse_preserves_the_original_destination_rows():
    inputs = dict(
        slots=[7, 2],
        request_ids=["first", "second", "inactive"],
        prefill_lengths=[9, 12, 0],
        block_tables={
            "history": np.array([[0, 8, -1], [1, 2, 3], [4, 5, 6]]),
            "state": np.array([[0, 11], [13, 0], [0, 0]]),
        },
    )
    original = copy_remote_cache_admissions(**inputs)
    inputs["request_ids"][0] = "replacement"
    inputs["prefill_lengths"][0] = 4
    inputs["block_tables"]["state"][0, 1] = 17
    replacement = copy_remote_cache_admissions(**inputs)

    assert list(original) == [7, 2]
    assert (original[7].request_id, original[7].position) == ("first", 9)
    assert original[7].block_tables == {"history": (0, 8, -1), "state": (0, 11)}
    assert original[2].block_tables["state"] == (13, 0)
    assert (replacement[7].request_id, replacement[7].position) == ("replacement", 4)
    assert replacement[7].block_tables["state"] == (0, 17)
    with pytest.raises(TypeError):
        original[7].block_tables["state"] = (0, 17)


@pytest.mark.parametrize("slots", [[1, 1], [1, -1]])
def test_invalid_slot_cannot_bind_a_destination(slots):
    with pytest.raises(ValueError, match="remote cache slot"):
        copy_remote_cache_admissions(
            slots=slots, request_ids=["a", "b"], prefill_lengths=[1, 2], block_tables={}
        )
