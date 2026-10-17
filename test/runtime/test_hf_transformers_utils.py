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

import os
import sys
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.utils.hf_transformers_utils import reconcile_special_tokens

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize(
    "added_vocab,stop_ids",
    [
        (
            {"<|eom_id|>": 128_009, "<|eom|>": 200_007, "<|eot|>": 200_008},
            {128_009, 200_008},
        ),
        ({"<|eom|>": 200_007}, None),
    ],
)
def test_reconcile_special_tokens_stops_on_eot_but_not_eom(
    added_vocab, stop_ids
) -> None:
    """<|eom|> ends a channel, not the turn, so only turn-final markers stop."""
    tokenizer = SimpleNamespace(
        added_tokens_decoder=None,
        additional_stop_token_ids=None,
        get_added_vocab=lambda: added_vocab,
    )
    reconcile_special_tokens(tokenizer)
    assert tokenizer.additional_stop_token_ids == stop_ids


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
