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

"""TunableOp BLAS solutions: loading, stack check, numerics and online tuning."""

from __future__ import annotations

import glob
import os
import pathlib
import subprocess
import sys

import pytest
import torch
from utils import is_amd

if not is_amd() or torch.version.hip is None:
    pytest.skip("ROCm GPU required for TunableOp", allow_module_level=True)

# Loading goes through the hipblaslt.py entry point that serving uses;
# tuning and the file stamp are the _hipblaslt implementation's.
from tokenspeed_kernel.ops.gemm._hipblaslt.tuning import (  # noqa: E402
    TUNED_CONFIG_DIR,
    find_hipblaslt_tuned_files,
    hipblaslt_file_validators,
    tune_hipblaslt_solutions,
)
from tokenspeed_kernel.ops.gemm.hipblaslt import (  # noqa: E402
    load_hipblaslt_tuned_solutions,
)

# A verify-like skinny projection, small enough to tune in a few seconds.
N, K = 2048, 1024
TUNED_M = 64
UNTUNED_M = 48


@pytest.fixture(autouse=True)
def _restore_tunableop():
    yield
    tunable = torch.cuda.tunable
    tunable.enable(False)
    tunable.tuning_enable(True)
    tunable.record_untuned_enable(False)


@pytest.fixture(scope="module")
def tuned_file(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("hipblaslt") / "tuned.csv")
    count = tune_hipblaslt_solutions(
        shapes=[(N, K)],
        rows=[TUNED_M],
        dtype=torch.bfloat16,
        path=path,
        cold_bytes=64 * 2**20,
    )
    assert count == 1
    return path


def _operands(m: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
    return x, w


def test_replays_tuned_solution_within_bf16_tolerance(tuned_file):
    x, w = _operands(TUNED_M)
    default = torch.mm(x, w.t())
    assert load_hipblaslt_tuned_solutions(tuned_file)
    assert torch.cuda.tunable.is_enabled()
    assert not torch.cuda.tunable.tuning_is_enabled()
    keys = [result[1] for result in torch.cuda.tunable.get_results()]
    assert f"tn_{N}_{TUNED_M}_{K}_ld_{K}_{K}_{N}" in keys
    tuned = torch.mm(x, w.t())
    ref = x.float() @ w.float().t()
    # Another solution may reduce K in another order; both round to BF16.
    torch.testing.assert_close(tuned.float(), ref, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(tuned, default, atol=2e-2, rtol=1e-2)


def test_unlisted_shape_keeps_default_solution(tuned_file):
    x, w = _operands(UNTUNED_M)
    default = torch.mm(x, w.t())
    assert load_hipblaslt_tuned_solutions(tuned_file)
    assert torch.equal(torch.mm(x, w.t()), default)


def test_rejects_file_from_another_stack(tuned_file, tmp_path):
    stale = tmp_path / "stale.csv"
    lines = open(tuned_file).read().splitlines()
    stale.write_text(
        "\n".join(
            (
                "Validator,HIPBLASLT_VERSION,0-0"
                if line.startswith("Validator,HIPBLASLT_VERSION,")
                else line
            )
            for line in lines
        )
        + "\n"
    )
    assert hipblaslt_file_validators(str(stale))["HIPBLASLT_VERSION"] == "0-0"
    x, w = _operands(TUNED_M)
    default = torch.mm(x, w.t())
    assert not load_hipblaslt_tuned_solutions(str(stale))
    assert not torch.cuda.tunable.is_enabled()
    assert torch.equal(torch.mm(x, w.t()), default)


def _restamp(src: str, dst, key: str, value: str) -> None:
    lines = open(src).read().splitlines()
    prefix = f"Validator,{key},"
    dst.write_text(
        "\n".join(prefix + value if line.startswith(prefix) else line for line in lines)
        + "\n"
    )


def test_tuning_extends_existing_file(tuned_file, tmp_path):
    path = tmp_path / "extended.csv"
    path.write_bytes(open(tuned_file, "rb").read())
    before = {line for line in open(path) if not line.startswith("Validator,")}
    count = tune_hipblaslt_solutions(
        shapes=[(N, K)],
        rows=[TUNED_M, UNTUNED_M],
        dtype=torch.bfloat16,
        path=str(path),
        cold_bytes=64 * 2**20,
    )
    assert count == 2
    after = {line for line in open(path) if not line.startswith("Validator,")}
    # The listed GEMM keeps its entry, not re-timed; the new one is added.
    assert before < after
    assert any(f"tn_{N}_{UNTUNED_M}_{K}_" in line for line in after)


def test_tuning_refuses_file_from_another_stack(tuned_file, tmp_path):
    stale = tmp_path / "stale.csv"
    _restamp(tuned_file, stale, "HIPBLASLT_VERSION", "0-0")
    before = stale.read_bytes()
    with pytest.raises(ValueError, match="another stack"):
        tune_hipblaslt_solutions(
            shapes=[(N, K)],
            rows=[UNTUNED_M],
            dtype=torch.bfloat16,
            path=str(stale),
            cold_bytes=64 * 2**20,
        )
    assert not torch.cuda.tunable.is_enabled()
    assert stale.read_bytes() == before


def test_this_gpu_finds_its_architectures_committed_file():
    arch_name = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName
    committed = [
        path
        for path in glob.glob(os.path.join(TUNED_CONFIG_DIR, "gcn_arch=*.csv"))
        if hipblaslt_file_validators(path)["GCN_ARCH_NAME"] == arch_name
    ]
    if not committed:
        pytest.skip(f"no committed gcn_arch= file for {arch_name}")
    # Whatever name the driver reports, the architecture's file comes first.
    paths = find_hipblaslt_tuned_files(torch.cuda.get_device_name(), arch_name)
    assert paths[:1] == committed


def _committed_pair(tmp_path, tuned_file, device_file):
    """Lay out architecture and device files for a test."""
    arch_name = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "gcn_arch=test.csv").write_text(pathlib.Path(tuned_file).read_text())
    name = torch.cuda.get_device_name().replace(" ", "_")
    (configs / f"device_name={name}.csv").write_text(
        pathlib.Path(device_file).read_text()
    )
    assert hipblaslt_file_validators(tuned_file)["GCN_ARCH_NAME"] == arch_name
    return configs


def _tuned_keys(path: str) -> set[tuple[str, str]]:
    with open(path) as f:
        return {tuple(line.split(",")[:2]) for line in f if line.startswith("Gemm")}


def test_architecture_and_device_files_load_together(tuned_file, tmp_path):
    device_file = str(tmp_path / "device.csv")
    assert tune_hipblaslt_solutions(
        shapes=[(N, K)],
        rows=[UNTUNED_M],
        dtype=torch.bfloat16,
        path=device_file,
        cold_bytes=64 * 2**20,
    )
    configs = _committed_pair(tmp_path, tuned_file, device_file)
    expected = sorted(_tuned_keys(tuned_file) | _tuned_keys(device_file))
    assert len(expected) == 2
    _run_fresh_process(
        "import torch\n"
        "from unittest.mock import patch\n"
        "from tokenspeed_kernel.ops.gemm._hipblaslt import tuning\n"
        f"with patch.object(tuning, 'TUNED_CONFIG_DIR', {str(configs)!r}):\n"
        "    assert tuning.load_hipblaslt_tuned_solutions()\n"
        "keys = sorted(tuple(r[:2]) for r in torch.cuda.tunable.get_results())\n"
        f"assert keys == {expected!r}, keys\n",
        tmp_path,
    )


def test_stale_device_file_leaves_the_architecture_file_loaded(tuned_file, tmp_path):
    stale = tmp_path / "stale-device.csv"
    stale.write_text(
        pathlib.Path(tuned_file).read_text().replace(f"_{TUNED_M}_", f"_{UNTUNED_M}_")
    )
    _restamp(str(stale), stale, "HIPBLASLT_VERSION", "0-0")
    configs = _committed_pair(tmp_path, tuned_file, str(stale))
    expected = sorted(_tuned_keys(tuned_file))
    _run_fresh_process(
        "import torch\n"
        "from unittest.mock import patch\n"
        "from tokenspeed_kernel.ops.gemm._hipblaslt import tuning\n"
        f"with patch.object(tuning, 'TUNED_CONFIG_DIR', {str(configs)!r}):\n"
        "    assert tuning.load_hipblaslt_tuned_solutions()\n"
        "keys = sorted(tuple(r[:2]) for r in torch.cuda.tunable.get_results())\n"
        f"assert keys == {expected!r}, keys\n",
        tmp_path,
    )


def test_tuning_keeps_architecture_rows_out_of_the_device_file(tuned_file, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "gcn_arch=test.csv").write_text(pathlib.Path(tuned_file).read_text())
    device_file = tmp_path / "device.csv"
    # A fresh process, so TunableOp's own writes at exit are covered too.
    _run_fresh_process(
        "import torch\n"
        "from unittest.mock import patch\n"
        "from tokenspeed_kernel.ops.gemm._hipblaslt import tuning\n"
        f"with patch.object(tuning, 'TUNED_CONFIG_DIR', {str(configs)!r}):\n"
        "    tuning.tune_hipblaslt_solutions(\n"
        f"        shapes=[({N}, {K})], rows=[{TUNED_M}, {UNTUNED_M}],\n"
        f"        dtype=torch.bfloat16, path={str(device_file)!r},\n"
        "        cold_bytes=64 * 2**20)\n",
        tmp_path,
    )
    written = _tuned_keys(str(device_file))
    assert len(written) == 1
    assert not written & _tuned_keys(tuned_file)


def test_regenerating_the_architecture_file_through_a_symlink_keeps_its_rows(
    tuned_file, tmp_path
):
    configs = tmp_path / "configs"
    configs.mkdir()
    architecture_file = configs / "gcn_arch=test.csv"
    architecture_file.write_text(pathlib.Path(tuned_file).read_text())
    link = tmp_path / "link.csv"
    link.symlink_to(architecture_file)
    # A fresh process: TunableOp results already in this one would be written too.
    _run_fresh_process(
        "import torch\n"
        "from unittest.mock import patch\n"
        "from tokenspeed_kernel.ops.gemm._hipblaslt import tuning\n"
        f"with patch.object(tuning, 'TUNED_CONFIG_DIR', {str(configs)!r}):\n"
        "    written = tuning.tune_hipblaslt_solutions(\n"
        f"        shapes=[({N}, {K})], rows=[{TUNED_M}],\n"
        f"        dtype=torch.bfloat16, path={str(link)!r},\n"
        "        cold_bytes=64 * 2**20)\n"
        # Its own rows are not filtered out as another architecture file's.
        "assert written == 1, written\n",
        tmp_path,
    )
    assert _tuned_keys(str(architecture_file)) == _tuned_keys(tuned_file)


def test_missing_file_keeps_tunableop_disabled(tmp_path):
    assert not load_hipblaslt_tuned_solutions(str(tmp_path / "absent.csv"))
    assert not torch.cuda.tunable.is_enabled()


def _run_fresh_process(code: str, cwd) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTORCH_TUNABLEOP")}
    subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, check=True)


def test_off_unless_loaded(tmp_path):
    # A process that imports the module and runs GEMMs but loads no file keeps
    # TunableOp off and writes no results file.
    _run_fresh_process(
        "import torch, tokenspeed_kernel.ops.gemm.hipblaslt\n"
        f"a = torch.randn({TUNED_M}, {K}, device='cuda', dtype=torch.bfloat16)\n"
        "torch.mm(a, a.t())\n"
        "assert not torch.cuda.tunable.is_enabled()\n",
        cwd=tmp_path,
    )
    assert list(tmp_path.iterdir()) == []


def test_serving_leaves_file_unchanged(tuned_file, tmp_path):
    before = open(tuned_file, "rb").read()
    _run_fresh_process(
        "import torch\n"
        "from tokenspeed_kernel.ops.gemm.hipblaslt import load_hipblaslt_tuned_solutions\n"
        f"assert load_hipblaslt_tuned_solutions({tuned_file!r})\n"
        f"for m in ({TUNED_M}, {UNTUNED_M}):\n"
        f"    x = torch.randn(m, {K}, device='cuda', dtype=torch.bfloat16)\n"
        f"    w = torch.randn({N}, {K}, device='cuda', dtype=torch.bfloat16)\n"
        "    torch.mm(x, w.t())\n"
        "torch.cuda.synchronize()\n",
        cwd=tmp_path,
    )
    assert open(tuned_file, "rb").read() == before
    assert list(tmp_path.iterdir()) == []


def test_online_tuning_adds_unlisted_gemms_in_memory(tuned_file, tmp_path):
    before = open(tuned_file, "rb").read()
    listed = next(
        line.split(",")[2] for line in before.decode().splitlines() if "tn_" in line
    )
    _run_fresh_process(
        "import torch\n"
        "from tokenspeed_kernel.ops.gemm.hipblaslt import (\n"
        "    hipblaslt_online_tuning, load_hipblaslt_tuned_solutions)\n"
        "tunable = torch.cuda.tunable\n"
        f"assert load_hipblaslt_tuned_solutions({tuned_file!r})\n"
        f"w = torch.randn({N}, {K}, device='cuda', dtype=torch.bfloat16)\n"
        "def mm(m):\n"
        f"    torch.mm(torch.randn(m, {K}, device='cuda', dtype=torch.bfloat16), w.t())\n"
        "with hipblaslt_online_tuning():\n"
        f"    mm({TUNED_M}); mm({UNTUNED_M})\n"
        "assert tunable.is_enabled() and not tunable.tuning_is_enabled()\n"
        "mm(80)\n"
        "results = {r[1]: r[2] for r in tunable.get_results()}\n"
        # The listed GEMM keeps its solution; the unlisted one in the block is
        # tuned; one after the block is not.
        f"assert results.pop('tn_{N}_{TUNED_M}_{K}_ld_{K}_{K}_{N}') == {listed!r}\n"
        f"assert list(results) == ['tn_{N}_{UNTUNED_M}_{K}_ld_{K}_{K}_{N}'], results\n",
        cwd=tmp_path,
    )
    assert open(tuned_file, "rb").read() == before
    assert list(tmp_path.iterdir()) == []


def test_online_tuning_without_file(tmp_path):
    _run_fresh_process(
        "import torch\n"
        "from tokenspeed_kernel.ops.gemm.hipblaslt import hipblaslt_online_tuning\n"
        "with hipblaslt_online_tuning():\n"
        f"    x = torch.randn({UNTUNED_M}, {K}, device='cuda', dtype=torch.bfloat16)\n"
        f"    w = torch.randn({N}, {K}, device='cuda', dtype=torch.bfloat16)\n"
        "    torch.mm(x, w.t())\n"
        "assert len(torch.cuda.tunable.get_results()) == 1\n",
        cwd=tmp_path,
    )
    # Nothing written, not even TunableOp's default file in the working dir.
    assert list(tmp_path.iterdir()) == []


def test_online_tuning_cache_serves_the_next_start(tmp_path):
    cache = tmp_path / "cache" / "env.hipblaslt.csv"
    workdir = tmp_path / "cwd"
    workdir.mkdir()

    def start(loaded: int) -> None:
        # The pick saved by an earlier start is loaded before the block, so
        # the GEMM is not timed again.
        _run_fresh_process(
            "import torch\n"
            "from tokenspeed_kernel.ops.gemm.hipblaslt import (\n"
            "    hipblaslt_online_tuning, hipblaslt_online_tuning_cache)\n"
            "tunable = torch.cuda.tunable\n"
            f"w = torch.randn({N}, {K}, device='cuda', dtype=torch.bfloat16)\n"
            f"x = torch.randn({UNTUNED_M}, {K}, device='cuda', dtype=torch.bfloat16)\n"
            f"with hipblaslt_online_tuning_cache({str(cache)!r}, write=True):\n"
            f"    assert len(tunable.get_results()) == {loaded}\n"
            "    with hipblaslt_online_tuning():\n"
            "        torch.mm(x, w.t())\n"
            "assert len(tunable.get_results()) == 1\n",
            cwd=workdir,
        )

    start(loaded=0)
    rows = cache.read_text().splitlines()
    key = f"tn_{N}_{UNTUNED_M}_{K}_ld_{K}_{K}_{N}"
    assert [row.split(",")[1] for row in rows if not row.startswith("Validator,")] == [
        key
    ]
    start(loaded=1)
    assert cache.read_text().splitlines() == rows
    assert list(workdir.iterdir()) == []
