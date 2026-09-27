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

"""Dense GEMM CLI selection is independent of the routed-expert backend."""

import argparse
from unittest import mock

import pytest

from tokenspeed.runtime.utils.env import (
    global_server_args_dict,
    global_server_args_dict_update,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def test_dense_gemm_backend_choices_and_worker_config():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    assert parser.parse_args(["--model", "x"]).dense_gemm_backend == "auto"
    for backend in ("auto", "trtllm_cutedsl"):
        parsed = parser.parse_args(["--model", "x", "--dense-gemm-backend", backend])
        assert parsed.dense_gemm_backend == backend
        assert parsed.moe_backend == "auto"
        with mock.patch.object(ServerArgs, "__post_init__", return_value=None):
            args = ServerArgs(model="x", dense_gemm_backend=backend)
        with mock.patch.dict(global_server_args_dict), mock.patch(
            "tokenspeed.runtime.utils.env.pdl_enabled"
        ):
            global_server_args_dict_update(args)
            assert global_server_args_dict["dense_gemm_backend"] == backend
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "x", "--dense-gemm-backend", "typo"])
    args = ServerArgs.__new__(ServerArgs)
    args.dense_gemm_backend = "typo"
    with pytest.raises(ValueError, match="dense-gemm-backend"):
        args.resolve_kernel_backends()
