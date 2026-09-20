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

from types import SimpleNamespace
from unittest.mock import Mock

import gpu_visibility
import pytest
from gpu_visibility import prepare_environment, resolve_gpu_groups


@pytest.mark.parametrize(
    "parent,groups,expected",
    [
        (None, ["0,1", "2,3"], ["0,1", "2,3"]),
        ("4,6,5,7", ["0,1", "2,3"], ["4,6", "5,7"]),
        ("GPU-a,GPU-b,GPU-c", ["2,0", "GPU-b"], ["GPU-c,GPU-a", "GPU-b"]),
    ],
)
def test_role_selection(parent, groups, expected):
    assert resolve_gpu_groups(parent, groups) == expected


@pytest.mark.parametrize(
    "parent,groups",
    [
        ("", ["0"]),
        ("-1", ["0"]),
        ("4,5", ["2"]),
        ("4,5", ["0", "0"]),
        ("4,5", ["0,0"]),
        ("GPU-a,GPU-b", ["0", "GPU-a"]),
        ("GPU-a", ["GPU-b"]),
        ("4,4", ["0"]),
        ("all", ["0"]),
        ("0,1", [""]),
        ("0,-1,2", ["0"]),
        (None, ["0", "0"]),
        (None, ["01", "1"]),
        ("MIG-abcd", ["0"]),
        ("GPU-abc,GPU-abcdef", ["0", "1"]),
    ],
)
def test_invalid_role_selection(parent, groups):
    with pytest.raises(ValueError):
        resolve_gpu_groups(parent, groups)


@pytest.mark.parametrize("cuda", ["", "-1", "3,1", "GPU-c,GPU-a"])
def test_explicit_cuda_mask_is_preserved(cuda):
    env = {"CUDA_VISIBLE_DEVICES": cuda, "NVIDIA_VISIBLE_DEVICES": "GPU-a,GPU-c"}
    assert prepare_environment(env) == env


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("all", None),
        ("none", ""),
        ("void", None),
        ("", None),
        ("GPU-c,GPU-a", "GPU-c,GPU-a"),
    ],
)
def test_container_allocation_fills_only_missing_cuda_mask(value, expected):
    env = {} if value is None else {"NVIDIA_VISIBLE_DEVICES": value}
    actual = prepare_environment(env)
    assert actual.get("CUDA_VISIBLE_DEVICES") == expected
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_numeric_container_allocation_uses_nvml_uuid_lookup(monkeypatch):
    query = Mock(
        side_effect=[
            SimpleNamespace(stdout="GPU-c\n"),
            SimpleNamespace(stdout="GPU-a\n"),
        ]
    )
    monkeypatch.setattr(gpu_visibility.subprocess, "run", query)
    actual = prepare_environment({"NVIDIA_VISIBLE_DEVICES": "6,4"})
    assert actual["CUDA_VISIBLE_DEVICES"] == "GPU-c,GPU-a"
    assert [call.args[0][1] for call in query.call_args_list] == ["--id=6", "--id=4"]


def test_diagnostic_rejects_cuda_devices_outside_container_allocation(monkeypatch):
    import sys

    cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(uuid="b"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match="outside NVIDIA_VISIBLE_DEVICES"):
        gpu_visibility.diagnose(
            {"NVIDIA_VISIBLE_DEVICES": "GPU-a", "CUDA_VISIBLE_DEVICES": "0"}, 1
        )


def test_diagnostic_reports_actual_cuda_uuid_and_memory(monkeypatch, capsys):
    import sys

    cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(uuid="a"),
        mem_get_info=lambda _: (100, 200),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        gpu_visibility.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout="no processes", stderr="")),
    )
    gpu_visibility.diagnose(
        {"NVIDIA_VISIBLE_DEVICES": "GPU-a", "CUDA_VISIBLE_DEVICES": "0"}, 1
    )
    assert '"free_bytes": 100' in capsys.readouterr().out


@pytest.mark.parametrize(
    "script,selections",
    [
        (
            "serve_qwen35_397b_nvfp4_pd_1p1d.sh",
            {"PREFILL_GPUS": "0,1", "DECODE_GPUS": "1,2"},
        ),
        (
            "serve_qwen35_122b_nvfp4_epd_1e1p2d.sh",
            {"ENCODE_GPUS": "0", "PREFILL_GPUS": "0"},
        ),
    ],
)
def test_qwen_launchers_reject_overlap_before_launch(script, selections, tmp_path):
    import os
    import subprocess
    from pathlib import Path

    result = subprocess.run(
        ["bash", str(Path(__file__).with_name(script))],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "4,6,5,7",
            "MODEL_PATH": str(tmp_path),
            **selections,
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert "selected more than once" in result.stderr
    assert "starting" not in result.stdout


def test_diagnostic_rejects_cuda_truncating_duplicate_aliases(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)),
    )
    with pytest.raises(RuntimeError, match="enumerated 1 devices for 2"):
        gpu_visibility.diagnose({"CUDA_VISIBLE_DEVICES": "0,GPU-a"}, 2)


def test_diagnostic_rejects_duplicate_cuda_uuid(monkeypatch):
    import sys

    cuda = SimpleNamespace(
        device_count=lambda: 2,
        get_device_properties=lambda _: SimpleNamespace(uuid="a"),
        mem_get_info=lambda _: (100, 200),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match="more than once"):
        gpu_visibility.diagnose({"CUDA_VISIBLE_DEVICES": "0,1"}, 2)


def test_numeric_container_allocation_rejects_duplicate_uuid(monkeypatch):
    monkeypatch.setattr(
        gpu_visibility.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout="GPU-a\n")),
    )
    with pytest.raises(ValueError, match="duplicate"):
        prepare_environment({"NVIDIA_VISIBLE_DEVICES": "0,1"})


@pytest.mark.parametrize(
    "parent,groups",
    [
        (None, ["GPU-abc", "GPU-abcdef"]),
        (None, ["0", "GPU-a"]),
        ("0,GPU-a", ["0", "1"]),
    ],
)
def test_cross_role_aliases_are_rejected(parent, groups):
    with pytest.raises(ValueError):
        resolve_gpu_groups(parent, groups)


def test_diagnostic_rejects_ambiguous_allocation_prefix(monkeypatch):
    import sys

    cuda = SimpleNamespace(
        device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(uuid=f"a{index}"),
        mem_get_info=lambda _: (100, 200),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match="ambiguous allocation"):
        gpu_visibility.diagnose(
            {"CUDA_VISIBLE_DEVICES": "0,1", "NVIDIA_VISIBLE_DEVICES": "GPU-a"}, 2
        )


def test_diagnose_cli_does_not_add_cuda_mask(monkeypatch):
    import sys

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "GPU-a")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu_visibility.py", "diagnose", "--runner", "slurm-gb200-2node-4gpu"],
    )
    prepare = Mock(side_effect=AssertionError("diagnostics must preserve environment"))
    probe = Mock()
    monkeypatch.setattr(gpu_visibility, "prepare_environment", prepare)
    monkeypatch.setattr(gpu_visibility, "diagnose", probe)
    gpu_visibility.main()
    prepare.assert_not_called()
    assert "CUDA_VISIBLE_DEVICES" not in probe.call_args.args[0]

    assert probe.call_args.args[1] == 4


@pytest.mark.parametrize("value", ["void", ""])
@pytest.mark.parametrize("count", [0, 2, 4, 8])
def test_runtime_injected_devices_must_match_runner_request(monkeypatch, value, count):
    import sys

    cuda = SimpleNamespace(
        device_count=lambda: count,
        get_device_properties=lambda index: SimpleNamespace(uuid=f"a{index}"),
        mem_get_info=lambda _: (100, 200),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(
        gpu_visibility.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout="no processes", stderr="")),
    )
    env = prepare_environment({"NVIDIA_VISIBLE_DEVICES": value})
    assert "CUDA_VISIBLE_DEVICES" not in env
    if count == 4:
        gpu_visibility.diagnose(env, 4)
    else:
        with pytest.raises(RuntimeError, match="no visible CUDA|runner requests 4"):
            gpu_visibility.diagnose(env, 4)


@pytest.mark.parametrize("mask", ["", "-1"])
def test_explicit_empty_cuda_mask_is_never_broadened(monkeypatch, mask):
    import sys

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(device_count=lambda: 0),
        ),
    )
    env = {"CUDA_VISIBLE_DEVICES": mask, "NVIDIA_VISIBLE_DEVICES": "void"}
    assert prepare_environment(env) == env
    with pytest.raises(RuntimeError, match="no visible CUDA"):
        gpu_visibility.diagnose(env, 4)


@pytest.mark.parametrize("runner", ["b200", "b200-0gpu", "b200-4gpu-8gpu"])
def test_runner_count_must_be_unambiguous(runner):
    with pytest.raises(ValueError, match="one '<N>gpu' segment"):
        gpu_visibility.gpu_count(runner)
