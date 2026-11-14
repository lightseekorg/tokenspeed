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

"""Offline-tuned TunableOp BLAS solutions: loading, stack check and numerics."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
from utils import is_amd

if not is_amd() or torch.version.hip is None:
    pytest.skip("ROCm GPU required for TunableOp", allow_module_level=True)

# Loading goes through the tokenspeed_kernel entry point that serving uses;
# tuning and the file stamp are the AMD package's.
from tokenspeed_kernel.ops.gemm.hipblaslt import (  # noqa: E402
    load_hipblaslt_tuned_solutions,
)
from tokenspeed_kernel_amd.ops.gemm.hipblaslt import (  # noqa: E402
    hipblaslt_file_validators,
    hipblaslt_tuned_file,
    tune_hipblaslt_solutions,
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
        "import tokenspeed_kernel_amd.ops.gemm.hipblaslt\n"
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
