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


"""Real compiler-cache behavior of the commit experiment's parameter audit."""

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="NVIDIA commit parameter audit"
)


@pytest.fixture(scope="module")
def analysis_module():
    root = Path(__file__).resolve().parents[2]
    directory = root / "benchmarks/nvidia/gdn_replay_cta16"
    # The analysis command imports its sibling command adapter by absolute name.
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "gdn_parameter_analysis", directory / "parameter_analysis.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(directory))


def make_copy_kernel(*, specialize: bool):
    if specialize:

        @gluon.jit
        def copy(out, n: gl.constexpr):
            index = gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [4], [0]))
            gl.store(out + index, index + 1, index < n)

    else:

        @gluon.jit(do_not_specialize=["n"])
        def copy(out, n):
            index = gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [4], [0]))
            gl.store(out + index, index + 1, index < n)

    return copy


@pytest.mark.parametrize("specialize", [False, True])
def test_audit_measures_real_recompilation(analysis_module, specialize):
    kernel = make_copy_kernel(specialize=specialize)
    output = torch.empty(64, device="cuda", dtype=torch.int32)

    def launch(n):
        output.fill_(-1)
        kernel[(1,)](output, n, num_warps=4)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output[:n], torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
        )
        assert (output[n:] == -1).all()

    result = analysis_module.audit_jit_reuse(
        kernel, [partial(launch, n) for n in (3, 8, 32)]
    )
    assert result == {
        "warm_variants": 1,
        "variant_count": 3 if specialize else 1,
        "new_variants": 2 if specialize else 0,
    }


def test_real_gdn_runtime_shapes_share_a_variant(analysis_module, tmp_path):
    root = Path(__file__).resolve().parents[2]
    saved = root / "benchmarks/nvidia/gdn_replay_cta16"
    (tmp_path / "kernel.py").write_bytes((saved / "kernel_stream.py").read_bytes())
    (tmp_path / "candidate.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "settings": {
                        "max_ctas": 16,
                        "num_buffers": 3,
                        "num_warps": 16,
                        "max_registers": 96,
                    }
                }
            }
        )
    )
    config = json.loads((saved / "task.json").read_text())
    base = config["clusters"][0]["workloads"][0]["params"]
    workloads = [
        {
            "params": {
                **base,
                "geometry": [2, 6, 128, 128],
                "batch": batch,
                "layers": layers,
            }
        }
        for batch, layers in [(3, 1), (8, 2)]
    ]
    result = analysis_module.analyze(
        {
            "candidate": {"path": str(tmp_path)},
            "parameter_contract": config["parameter_contract"],
            "workloads": workloads,
        }
    )
    assert result["status"] == "completed"
    assert result["violations"] == []
    assert result["kernel_analysis"]["variant_count"] == 1
