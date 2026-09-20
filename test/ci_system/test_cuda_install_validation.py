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

import importlib.metadata
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def installed_stack(monkeypatch, tmp_path):
    requirements = tmp_path / "cuda.txt"
    requirements.write_text("torch==2.14.0\n")
    thirdparty = tmp_path / "cuda-thirdparty.txt"
    thirdparty.write_text(
        "tokenspeed-trtllm-kernel==1.3.0.post20260919\n"
        "tokenspeed-cutedsl-kda==0.1.0.post20260919\n"
    )
    versions = {
        "tokenspeed-trtllm-kernel": "1.3.0.post20260919",
        "tokenspeed-cutedsl-kda": "0.1.0.post20260919",
    }
    torch = SimpleNamespace(
        __version__="2.14.0+cu130", version=SimpleNamespace(cuda="13.0")
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(sys, "argv", ["-", str(requirements), str(thirdparty), "130"])
    installer = Path(__file__).with_name("install_deps.sh").read_text()
    marker = 'python3 - "${CUDA_REQ}" "${THIRDPARTY_REQ}" "${CUINDEX}" <<\'PY\'\n'
    script = installer.split(marker, 1)[1].split("\nPY", 1)[0]
    return compile(script, "install_deps.sh:validation", "exec"), torch, versions


def test_cuda_install_logs_and_accepts_matching_versions(installed_stack, capsys):
    script, _, _ = installed_stack
    exec(script, {})
    output = capsys.readouterr().out
    assert "Installed torch==2.14.0+cu130" in output
    assert "Installed tokenspeed-trtllm-kernel==1.3.0.post20260919" in output
    assert "Installed tokenspeed-cutedsl-kda==0.1.0.post20260919" in output
    assert "Torch CUDA runtime: 13.0" in output


def test_cuda_install_rejects_torch_downgrade(installed_stack):
    script, torch, _ = installed_stack
    torch.__version__ = "2.13.0+cu130"
    with pytest.raises(SystemExit, match="does not satisfy"):
        exec(script, {})


@pytest.mark.parametrize("name", ["tokenspeed-trtllm-kernel", "tokenspeed-cutedsl-kda"])
def test_cuda_install_rejects_stale_native_dependency(installed_stack, name):
    script, _, versions = installed_stack
    versions[name] = "0.0.0"
    with pytest.raises(SystemExit, match="does not satisfy"):
        exec(script, {})


@pytest.mark.parametrize("cuda", ["12.6", None])
def test_cuda_install_rejects_wrong_cuda_runtime(installed_stack, cuda):
    script, torch, _ = installed_stack
    torch.version.cuda = cuda
    with pytest.raises(SystemExit, match="Expected Torch CUDA 13.0"):
        exec(script, {})
