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

import os
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
    hipblaslt_file_validators,
    hipblaslt_tuned_file,
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


def test_default_file_is_this_gpus():
    name = torch.cuda.get_device_name().replace(" ", "_")
    assert hipblaslt_tuned_file().endswith(f"/configs/device_name={name}.csv")


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


# The committed MI355X rows: each listed solution is the one TunableOp runs for
# its GEMM, writes every output element, and gives the bits the GEMM has with
# TunableOp off.
MI355X = "AMD Instinct MI355X"
# (N, K, rows) of the rows below whose solutions were chosen to keep the
# default solution's bits; TunableOp keys them as tn_{N}_{M}_{K}_ld_{K}_{K}_{N}.
PINNED = [
    *[(8192, 2048, m) for m in (*range(49, 97), 320)],
    *[(8192, 4096, m) for m in range(74, 97)],
    (4096, 8192, 2048),
    (18432, 8192, 4096),
    (512, 8192, 16384),
]
INPUTS = 3


def _row_key(n: int, k: int, m: int) -> str:
    return f"tn_{n}_{m}_{k}_ld_{k}_{k}_{n}"


def _file_rows(path: str) -> dict[str, str]:
    rows = {}
    with open(path) as f:
        for line in f:
            op, params, solution, *_ = line.rstrip("\n").split(",")
            if op == "GemmTunableOp_BFloat16_TN":
                rows[params] = solution
    return rows


@pytest.fixture
def mi355x_rows():
    """The MI355X file, loaded as serving loads it; skipped, naming the
    difference, on another GPU or software stack."""
    if torch.cuda.get_device_name() != MI355X:
        pytest.skip(f"the rows are for {MI355X}, not {torch.cuda.get_device_name()}")
    path = hipblaslt_tuned_file()
    tunable = torch.cuda.tunable
    tunable.enable(True)
    running = dict(tunable.get_validators())
    tunable.enable(False)
    mismatch = {
        key: (value, running.get(key))
        for key, value in hipblaslt_file_validators(path).items()
        if running.get(key) != value
    }
    if mismatch:
        pytest.skip(f"the file's stack is not this one (file, running): {mismatch}")
    assert load_hipblaslt_tuned_solutions(path)
    # _restore_tunableop turns TunableOp off after each test.
    return _file_rows(path)


def test_every_pinned_gemm_is_listed_once(mi355x_rows):
    keys = [_row_key(n, k, m) for n, k, m in PINNED]
    assert len(keys) == len(set(keys))
    assert set(keys) <= set(mi355x_rows)
    # TunableOp holds the file's solution for each key.
    held = {
        params: solution for _, params, solution, _ in torch.cuda.tunable.get_results()
    }
    for key in keys:
        assert held.get(key) == mi355x_rows[key], key


@pytest.mark.parametrize("n,k,m", PINNED, ids=lambda v: str(v))
def test_pinned_solution_writes_the_default_bits(mi355x_rows, n, k, m):
    """For several inputs, the pinned GEMM writes every element of a
    NaN-filled output, with the bits TunableOp off gives."""
    tunable = torch.cuda.tunable
    for seed in range(INPUTS):
        gen = torch.Generator(device="cuda").manual_seed(1000 * seed + m)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
        tunable.enable(False)
        default = torch.mm(x, w.t())
        tunable.enable(True)
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        torch.mm(x, w.t(), out=out)
        assert not out.isnan().any(), (n, k, m, seed)
        assert torch.equal(out, default), (n, k, m, seed)
