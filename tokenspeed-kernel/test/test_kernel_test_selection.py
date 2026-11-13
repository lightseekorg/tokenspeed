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

"""Kernel test files run when a pull request's changes name their families."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from kernel_test_selection import (
    CHANGED_PATHS_ENV,
    KernelTestSelection,
    selection_from_env,
)

KERNEL = "tokenspeed-kernel/python/tokenspeed_kernel"
AMD = "tokenspeed-kernel-amd/python/tokenspeed_kernel_amd"
TESTS = "tokenspeed-kernel/test"

FILES = {
    f"{KERNEL}/__init__.py": "",
    f"{KERNEL}/registry.py": "",
    f"{KERNEL}/ops/__init__.py": "",
    f"{KERNEL}/ops/gemm/__init__.py": "",
    f"{KERNEL}/ops/gemm/triton.py": "",
    f"{KERNEL}/ops/gemm/_cute_dsl/swiglu.py": "",
    f"{KERNEL}/ops/attention/__init__.py": "",
    f"{KERNEL}/ops/attention/mla/__init__.py": "",
    f"{KERNEL}/ops/attention/mla/triton.py": "",
    f"{KERNEL}/ops/attention/qsa/triton.py": "",
    f"{KERNEL}/ops/moe/cute_dsl.py": "def f():\n    from tokenspeed_kernel.thirdparty.cute_dsl import k\n",
    f"{KERNEL}/thirdparty/__init__.py": "",
    f"{KERNEL}/thirdparty/cute_dsl/__init__.py": "",
    f"{KERNEL}/thirdparty/cute_dsl/k.py": "",
    f"{KERNEL}/numerics/torch/sampling.py": "",
    f"{AMD}/ops/gfx950/moe/fp8.py": "",
    f"{AMD}/ops/gfx950/_xcd.py": "",
    f"{TESTS}/conftest.py": "",
    f"{TESTS}/utils.py": "",
    f"{TESTS}/asm_checks.py": "",
    # A helper that reaches a family and another helper.
    f"{TESTS}/gemm_helpers.py": "import asm_checks\nfrom tokenspeed_kernel.ops.gemm import mm\n",
    f"{TESTS}/ops/test_gemm.py": "from tokenspeed_kernel.ops.gemm import mm\n",
    f"{TESTS}/ops/test_mla.py": "from tokenspeed_kernel.ops.attention.mla import f\n",
    f"{TESTS}/ops/test_attention.py": "from tokenspeed_kernel.ops.attention import f\n",
    f"{TESTS}/ops/test_sampling.py": (
        "from tokenspeed_kernel.numerics.torch.sampling import reference\n"
    ),
    f"{TESTS}/ops/test_asm.py": "import asm_checks\n",
    f"{TESTS}/nvidia/ops/moe/test_moe.py": "",
    f"{TESTS}/nvidia/ops/test_gemm_capacity.py": "import gemm_helpers\n",
    f"{TESTS}/test_all.py": (
        "def test():\n"
        "    from tokenspeed_kernel.registry import load_builtin_kernels\n"
    ),
}
EVERY = {
    "test_gemm",
    "test_mla",
    "test_attention",
    "test_sampling",
    "test_asm",
    "test_moe",
    "test_gemm_capacity",
    "test_all",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name, text in FILES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (
            f"{KERNEL}/ops/gemm/triton.py",
            {"test_gemm", "test_gemm_capacity", "test_all"},
        ),
        (
            f"{KERNEL}/ops/gemm/_cute_dsl/swiglu.py",
            {"test_gemm", "test_gemm_capacity", "test_all"},
        ),
        # A reviewer widens the tests with a directory (pr_test_scope.py).
        (f"{KERNEL}/ops/gemm", {"test_gemm", "test_gemm_capacity", "test_all"}),
        (f"{KERNEL}/ops/attention/mla/", {"test_mla", "test_attention", "test_all"}),
        (
            f"{KERNEL}/ops/attention/mla/triton.py",
            {"test_mla", "test_attention", "test_all"},
        ),
        (f"{KERNEL}/ops/attention/qsa/triton.py", {"test_attention", "test_all"}),
        (
            f"{KERNEL}/ops/attention/__init__.py",
            {"test_mla", "test_attention", "test_all"},
        ),
        # The MoE sources import the changed third-party package.
        (f"{KERNEL}/thirdparty/cute_dsl/k.py", {"test_moe", "test_all"}),
        (f"{AMD}/ops/gfx950/moe/fp8.py", {"test_moe", "test_all"}),
        (f"{KERNEL}/numerics/torch/sampling.py", {"test_sampling", "test_all"}),
        (f"{TESTS}/asm_checks.py", {"test_asm", "test_gemm_capacity", "test_all"}),
        (f"{TESTS}/ops/test_gemm.py", {"test_gemm", "test_all"}),
        (f"{TESTS}/ops/bench_gemm.py", {"test_all"}),
        ("python/tokenspeed/runtime/models/qwen3.py\ndocs/x.md", {"test_all"}),
        (f"{KERNEL}/registry.py", EVERY),
        (f"{AMD}/ops/gfx950/_xcd.py", EVERY),
        (f"{AMD}/_intrinsics/_wave.ll", EVERY),
        (f"{TESTS}/conftest.py", EVERY),
        ("tokenspeed-kernel/python/setup.py", EVERY),
        (".github/workflows/nvidia-b200-tests.yml", EVERY),
    ],
)
def test_changes_select_the_test_files_of_their_families(repo, changed, expected):
    selection = KernelTestSelection(repo, changed.split("\n"))

    tests = sorted((repo / TESTS).rglob("test_*.py"))
    assert {test.stem for test in tests if selection.selects(test)} == expected


def test_selection_comes_from_the_changed_paths_variable(repo, monkeypatch):
    monkeypatch.delenv(CHANGED_PATHS_ENV, raising=False)
    assert selection_from_env(repo) is None
    monkeypatch.setenv(CHANGED_PATHS_ENV, "\n")
    assert selection_from_env(repo) is None

    monkeypatch.setenv(CHANGED_PATHS_ENV, f"{KERNEL}/ops/gemm/triton.py\n")
    selection = selection_from_env(repo)

    assert selection.selects(repo / TESTS / "ops/test_gemm.py")
    assert not selection.selects(repo / TESTS / "ops/test_mla.py")


def test_a_parallel_run_the_selection_empties_passes():
    """A parallel run that the selection empties passes.

    Under pytest-xdist the workers collect and select, so the controller
    learns from them that the selection emptied the run: like a
    single-process run, it passes (exit 0) instead of failing with "no tests
    collected" (exit 5). This file names no op family, so a change to the
    gemm sources deselects all of it.
    """
    pytest.importorskip("xdist")
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env[CHANGED_PATHS_ENV] = f"{KERNEL}/ops/gemm/__init__.py"
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-n", "2", "-p", "no:cacheprovider", __file__],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    # The controller doesn't see the workers' deselections, only that nothing ran.
    assert "no tests ran" in run.stdout


def test_a_nested_run_with_an_inherited_xdist_variable_is_not_a_worker():
    """A nested single-process run inside an xdist worker selects as a run.

    It inherits ``PYTEST_XDIST_WORKER`` but is no worker: its config has no
    ``workeroutput``, and an emptied selection still passes it.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTEST_XDIST_WORKER"] = "gw0"
    env[CHANGED_PATHS_ENV] = f"{KERNEL}/ops/gemm/__init__.py"
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", __file__],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "0 selected" in run.stdout
