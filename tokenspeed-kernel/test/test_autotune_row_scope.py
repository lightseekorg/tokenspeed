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


from __future__ import annotations

import pytest
from tokenspeed_kernel.ops.tuning import (
    autotune,
    autotune_max_rows,
    get_autotune_max_num_tokens,
    get_autotune_max_rows,
)


def test_row_scope_is_local_and_restores_after_error():
    base = get_autotune_max_num_tokens()
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        with autotune_max_rows(base * 4):
            assert get_autotune_max_rows() == base * 4
            assert get_autotune_max_num_tokens() == base
            with autotune_max_rows(base * 2):
                assert get_autotune_max_rows() == base * 4
            with pytest.raises(RuntimeError, match="stop"):
                with autotune_max_rows(base * 8):
                    assert get_autotune_max_rows() == base * 8
                    raise RuntimeError("stop")
            assert get_autotune_max_rows() == base * 4
        assert get_autotune_max_rows() == base
    with autotune_max_rows(base * 4):
        assert get_autotune_max_rows() == base


@pytest.mark.parametrize("rows", [0, -1])
def test_row_scope_rejects_nonpositive_capacity(rows):
    with pytest.raises(ValueError, match="positive"):
        with autotune_max_rows(rows):
            pass
