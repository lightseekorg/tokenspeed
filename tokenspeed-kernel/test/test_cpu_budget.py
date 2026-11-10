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

"""cpu_budget: the threads a test process runs, from its cores and quota."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from cpu_budget import apply_cpu_budget, cgroup_cpu_quota, cpu_budget


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.mark.parametrize(
    "text, want",
    [("max 100000\n", None), ("800000 100000\n", 8), ("150000 100000\n", 2)],
)
def test_a_cgroup_v2_quota_rounds_up_to_whole_cpus(tmp_path, text, want):
    _write(tmp_path, "cpu.max", text)
    assert cgroup_cpu_quota(tmp_path) == want


@pytest.mark.parametrize("quota, want", [("-1", None), ("400000", 4), ("50000", 1)])
def test_a_cgroup_v1_quota_rounds_up_to_whole_cpus(tmp_path, quota, want):
    _write(tmp_path, "cpu/cpu.cfs_quota_us", quota + "\n")
    _write(tmp_path, "cpu/cpu.cfs_period_us", "100000\n")
    assert cgroup_cpu_quota(tmp_path) == want


def test_no_cgroup_files_mean_no_quota(tmp_path):
    assert cgroup_cpu_quota(tmp_path) is None


def test_the_budget_is_the_allowed_cores_capped_by_the_quota():
    cores = set(range(384))
    assert cpu_budget(cores, None) == 384
    assert cpu_budget(cores, 8) == 8
    assert cpu_budget({0, 1}, 8) == 2
    assert cpu_budget(set(), None) == 1


def test_applying_the_budget_caps_torch_and_children_but_keeps_explicit_settings(
    tmp_path,
):
    _write(tmp_path, "cpu.max", "100000 100000\n")  # one CPU
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        environ = {}
        assert apply_cpu_budget(torch, environ, tmp_path) == 1
        assert torch.get_num_threads() == 1
        assert environ == {"OMP_NUM_THREADS": "1"}
        # A caller's OMP_NUM_THREADS stays.
        environ = {"OMP_NUM_THREADS": "3"}
        apply_cpu_budget(torch, environ, tmp_path)
        assert environ["OMP_NUM_THREADS"] == "3"
    finally:
        torch.set_num_threads(before)
