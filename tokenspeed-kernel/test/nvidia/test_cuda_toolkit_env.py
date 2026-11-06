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
from pathlib import Path

import pytest
from tokenspeed_kernel.platform import prepare_cuda_toolkit_env


@pytest.mark.parametrize("symlink_include", [False, True], ids=["directory", "symlink"])
def test_cuda_toolkit_include_path_preserves_user_environment(
    tmp_path, monkeypatch, symlink_include
):
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    (toolkit / "bin" / "nvcc").touch()
    include = tmp_path / "headers" if symlink_include else toolkit / "include"
    include.mkdir()
    (include / "cuda_runtime.h").write_text("// CUDA header fixture\n")
    if symlink_include:
        (toolkit / "include").symlink_to(include, target_is_directory=True)
    user_include = tmp_path / "user-headers"
    user_include.mkdir()
    user_alias = tmp_path / "user-include"
    user_alias.symlink_to(user_include, target_is_directory=True)
    user_cpath = os.pathsep.join([str(user_alias), "relative/include"])
    user_path = os.pathsep.join(["relative/bin", "/user/bin"])
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    monkeypatch.setenv("CPATH", user_cpath)
    monkeypatch.setenv("PATH", user_path)

    prepare_cuda_toolkit_env()

    expected_cpath = os.pathsep.join([str(include.resolve()), user_cpath])
    expected_path = os.pathsep.join([str(toolkit.resolve() / "bin"), user_path])
    assert os.environ["CUDA_HOME"] == str(toolkit.resolve())
    assert os.environ["CPATH"] == expected_cpath
    assert os.environ["PATH"] == expected_path
    selected_include = Path(os.environ["CPATH"].split(os.pathsep)[0])
    assert selected_include.samefile(toolkit / "include")
    assert (selected_include / "cuda_runtime.h").read_bytes() == (
        toolkit / "include" / "cuda_runtime.h"
    ).read_bytes()

    prepare_cuda_toolkit_env()

    assert os.environ["CUDA_HOME"] == str(toolkit.resolve())
    assert os.environ["CPATH"] == expected_cpath
    assert os.environ["PATH"] == expected_path
