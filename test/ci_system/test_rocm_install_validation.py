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
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from check_rocm_torch import check_stack


@pytest.fixture
def stack(monkeypatch):
    torch = SimpleNamespace(
        __version__="2.14.0+rocm7.2",
        version=SimpleNamespace(hip="7.2", cuda=None),
    )
    vision = SimpleNamespace(__version__="0.29.0+rocm7.2")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", vision)
    return torch, vision


def test_rocm_stack_accepts_local_build_without_visible_gpu(stack):
    check_stack("2.14.0", "0.29.0", None)


@pytest.mark.parametrize("version", ["2.13.0+rocm7.2", "2.14.0a0+rocm7.2", "2.140.0"])
def test_rocm_stack_rejects_wrong_torch(stack, version):
    stack[0].__version__ = version
    with pytest.raises(SystemExit, match="does not satisfy"):
        check_stack("2.14.0", "0.29.0", None)


@pytest.mark.parametrize("hip,cuda", [(None, None), (None, "13.0"), ("7.2", "13.0")])
def test_rocm_stack_rejects_non_rocm_build(stack, hip, cuda):
    stack[0].version = SimpleNamespace(hip=hip, cuda=cuda)
    with pytest.raises(SystemExit, match="ROCm PyTorch"):
        check_stack("2.14.0", "0.29.0", None)


def test_rocm_stack_rejects_old_torchvision(stack):
    stack[1].__version__ = "0.28.0+rocm7.2"
    with pytest.raises(SystemExit, match="torchvision"):
        check_stack("2.14.0", "0.29.0", None)


def test_rocm_stack_checks_exact_simulator_build_and_device(stack, monkeypatch):
    torch, vision = stack
    torch.__version__ = "2.14.0+rocm10.1.0a20260822"
    vision.__version__ = "0.29.0a0+rocm10.1.0a20260822"
    requirement = f"amd-torch-device-gfx1250=={torch.__version__}"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: torch.__version__)
    check_stack(torch.__version__, vision.__version__, requirement)
    with pytest.raises(SystemExit, match="does not satisfy"):
        check_stack("2.14.0+rocm10.1.0a20260823", vision.__version__, requirement)
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "2.13.0+rocm10.1.0a20260822"
    )
    with pytest.raises(SystemExit, match="amd-torch-device-gfx1250"):
        check_stack(torch.__version__, vision.__version__, requirement)


def test_rocm_stack_rejects_missing_device_package(stack, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        check_stack("2.14.0", "0.29.0", "amd-torch-device-gfx1250==2.14.0")


@pytest.mark.parametrize(
    "script,scenario",
    [(script, scenario)
     for script in ("install_deps_rocm.sh", "install_kernel_benchmark_rocm.sh")
     for scenario in ("stale", "current", "override")]
    + [("install_deps_rocm.sh", "dependency-downgrade")],
)
def test_installer_upgrades_and_validates_torch(tmp_path, script, scenario):
    old_version = "2.14.0+rocm7.2" if scenario == "current" else "2.13.0+rocm7.2"
    target = "2.14.0+rocm10.1.0a20260822" if scenario == "override" else "2.14.0"
    vision = "0.29.0a0+rocm10.1.0a20260822" if scenario == "override" else "0.29.0"
    index = "https://rocm.nightlies.amd.com/whl-multi-arch/" if scenario == "override" else "https://download.pytorch.org/whl/rocm7.2"
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"torch": old_version, "torchvision": "0.29.0+rocm7.2"})
    )
    calls = tmp_path / "pip-calls.jsonl"
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "torch.py").write_text(
        "import json, os\nfrom types import SimpleNamespace\n"
        "with open(os.environ['TEST_STATE']) as f: __version__ = json.load(f)['torch']\n"
        "version = SimpleNamespace(hip='7.2', cuda=None)\n"
        "cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda n: 'test')\n"
    )
    (modules / "torchvision.py").write_text(
        "import json, os\n"
        "with open(os.environ['TEST_STATE']) as f: __version__ = json.load(f)['torchvision']\n"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake_pip = f"#!{sys.executable}\n" + """import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['TEST_CALLS'],'a') as f:f.write(json.dumps(args)+'\\n')
p=Path(os.environ['TEST_STATE']); state=json.loads(p.read_text())
for arg in args:
    if arg.startswith(('torch==','torchvision==')):
        name,version=arg.split('==');state[name]=version if '+' in version else version+'+rocm7.2'
if os.environ.get('TEST_DOWNGRADE')=='1' and './python' in args:
    state['torch']='2.13.0+rocm7.2'
p.write_text(json.dumps(state))
"""
    for name, body in {
        "pip": fake_pip,
        "pip3": fake_pip,
        "sudo": "#!/bin/sh\nexit 0\n",
        "python3": f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
    }.items():
        path = binaries / name
        path.write_text(body)
        path.chmod(0o755)
    # The benchmark uses python -m pip; provide the same recording stub there.
    (modules / "pip.py").write_text(fake_pip.split("\n", 1)[1])
    env = os.environ.copy()
    for key in (
        "TORCH_VERSION",
        "TORCHVISION_VERSION",
        "TORCH_INDEX_URL",
        "TORCH_DEVICE_PACKAGE",
    ):
        env.pop(key, None)
    env.update(
        PATH=f"{binaries}:{env['PATH']}",
        PYTHONPATH=str(modules),
        TEST_STATE=str(state),
        TEST_CALLS=str(calls),
        WORKSPACE=str(tmp_path),
    )
    if scenario == "override":
        env.update(TORCH_VERSION=target, TORCHVISION_VERSION=vision, TORCH_INDEX_URL=index)
    if scenario == "dependency-downgrade":
        env['TEST_DOWNGRADE'] = '1'
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name(script))],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if scenario == "dependency-downgrade":
        assert result.returncode != 0
        assert "does not satisfy torch==2.14.0" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(state.read_text())["torch"] == (target if '+' in target else target+'+rocm7.2')
    commands = (
        [json.loads(line) for line in calls.read_text().splitlines()]
        if calls.exists()
        else []
    )
    installs = [args for args in commands if f"torch=={target}" in args]
    assert len(installs) == (0 if scenario == "current" else 1)
    if installs:
        assert index in installs[0]
        assert "--force-reinstall" in installs[0]
        if script == "install_deps_rocm.sh":
            assert f"torchvision=={vision}" in installs[0]
