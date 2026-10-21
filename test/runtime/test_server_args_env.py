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

from tokenspeed.runtime.utils.env import (
    global_server_args_dict,
    global_server_args_dict_update,
)
from tokenspeed.runtime.utils.server_args import prepare_server_args


@pytest.mark.parametrize(
    "roles", [("prefill", "decode", "null"), ("decode", "prefill", "null")]
)
def test_resolved_worker_role_reaches_model_initialization(roles):
    original = global_server_args_dict.copy()
    try:
        for role in roles:
            args = prepare_server_args(["--model", "x", "--disaggregation-mode", role])
            global_server_args_dict_update(args)
            assert global_server_args_dict["disaggregation_mode"] == role
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(original)
