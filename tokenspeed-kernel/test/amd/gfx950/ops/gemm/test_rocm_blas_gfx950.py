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

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the pinned rocBLAS / hipBLASLt GEMMs",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.gemm import rocm_blas  # noqa: E402
from tokenspeed_kernel.ops.gemm import _select_mv, mm  # noqa: E402

PINS = rocm_blas._STATE.resolve()
pytestmark = pytest.mark.skipif(
    not PINS, reason="no pins for this device and library build"
)
CASES = [
    (m, n, k, solution)
    for (n, k), ranges in PINS.items()
    for first, last, solution in ranges
    for m in range(first, last + 1)
]


def default_gemm(*operands: torch.Tensor, op=torch.mm) -> torch.Tensor:
    """Today's GEMM: the same call with TunableOp off."""
    tunable = torch.cuda.tunable
    enabled = tunable.is_enabled()
    tunable.enable(False)
    try:
        return op(*operands)
    finally:
        tunable.enable(enabled)


def nan_buffer(m: int, n: int) -> torch.Tensor:
    # A GEMM that TunableOp skips leaves its output unwritten; NaN never
    # matches, whatever the memory held before.
    return torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)


def held_solutions() -> dict[str, str]:
    return {key: solution for _, key, solution, *_ in torch.cuda.tunable.get_results()}


def test_pinned_rows_select_the_pin_and_keep_default_bits():
    generator = torch.Generator(device="cuda").manual_seed(0)
    weights = {}
    for m, n, k, solution in CASES:
        selected = _select_mv(m, n, k, True, torch.bfloat16, torch.bfloat16, False)
        assert selected is rocm_blas.rocm_blas_pinned_mm, (m, n, k)
        assert held_solutions()[rocm_blas._tunableop_key(m, n, k)] == solution
        if (n, k) not in weights:
            weights[n, k] = (
                torch.randn(
                    n, k, device="cuda", dtype=torch.bfloat16, generator=generator
                )
                * k**-0.5
            )
        weight = weights[n, k]
        for _ in range(3):
            x = torch.randn(
                m, k, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            expected = default_gemm(x, weight.t())
            out = nan_buffer(m, n)
            assert torch.equal(mm(x, weight), expected), (m, n, k)
            assert torch.equal(mm(x, weight, out=out), expected), (m, n, k)


def test_direct_torch_mm_of_pinned_rows_runs_the_pin_with_default_bits():
    # Owned-row prefill projections call torch.mm(x, w.T, out=) directly; once the pins are active, that call takes the pin too.
    generator = torch.Generator(device="cuda").manual_seed(3)
    for m, n, k, solution in CASES:
        assert held_solutions()[rocm_blas._tunableop_key(m, n, k)] == solution
        weight = torch.randn(
            n, k, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
        out = nan_buffer(m, n)
        torch.mm(x, weight.T, out=out)
        assert torch.equal(out, default_gemm(x, weight.t())), (m, n, k)


@pytest.mark.parametrize(
    "op,shapes",
    [
        (torch.mm, ((97, 2048), (2048, 8192))),
        (torch.mm, ((64, 4096), (4096, 8192))),
        (torch.mm, ((80, 8192), (8192, 4608))),
        (torch.mm, ((2048, 8192), (8192, 4096))),
        (torch.addmm, ((4608,), (80, 8192), (8192, 4608))),
        (torch.bmm, ((8, 64, 512), (8, 512, 128))),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_unpinned_gemms_keep_default_bits_with_pins_active(op, shapes, dtype):
    # Active pins leave TunableOp enabled for the whole process.
    assert torch.cuda.tunable.is_enabled()
    generator = torch.Generator(device="cuda").manual_seed(1)
    operands = [
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
        for shape in shapes
    ]
    assert torch.equal(op(*operands), default_gemm(*operands, op=op))


def test_pinned_rows_replay_in_cuda_graphs():
    generator = torch.Generator(device="cuda").manual_seed(2)
    for (n, k), ranges in PINS.items():
        m = ranges[-1][0]
        weight = torch.randn(
            n, k, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
        out = nan_buffer(m, n)
        mm(x, weight, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mm(x, weight, out=out)
        for _ in range(2):
            x.copy_(
                torch.randn(
                    m, k, device="cuda", dtype=torch.bfloat16, generator=generator
                )
            )
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, default_gemm(x, weight.t())), (m, n, k)


def test_a_stale_results_file_does_not_override_the_pins(tmp_path):
    m, n, k, solution = CASES[-1]
    key = rocm_blas._tunableop_key(m, n, k)
    other_key = rocm_blas._tunableop_key(m + 1000, n, k)
    stale = tmp_path / "tunableop_results0.csv"
    stale.write_text(
        "\n".join(
            [f"Validator,{name},{value}" for name, value in rocm_blas._PINNED_BUILD]
            + [
                f"GemmTunableOp_BFloat16_TN,{key},Gemm_Hipblaslt_1,0.0",
                f"GemmTunableOp_BFloat16_TN,{other_key},Gemm_Hipblaslt_1,0.0",
            ]
        )
        + "\n"
    )
    # A fresh process in that directory: TunableOp reads its results file
    # (by default tunableop_results<device>.csv there) once, when first enabled.
    script = textwrap.dedent(f"""
        import json, logging, torch
        logging.basicConfig(level=logging.INFO)
        from tokenspeed_kernel.ops.gemm import mm, rocm_blas
        x = torch.randn({m}, {k}, device="cuda", dtype=torch.bfloat16)
        w = torch.randn({n}, {k}, device="cuda", dtype=torch.bfloat16)
        expected = torch.mm(x, w.t())
        bitwise = torch.equal(mm(x, w), expected)
        held = {{r[1]: r[2] for r in torch.cuda.tunable.get_results()}}
        active = bool(rocm_blas._STATE.resolve())
        print(json.dumps({{"bitwise": bitwise, "held": held, "active": active}}))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["active"] and report["bitwise"]
    # One startup line names the device and the active pins.
    assert result.stderr.count("rocm_blas_pinned_mm:") == 1
    assert "active" in result.stderr
    assert report["held"][key] == solution
    assert other_key not in report["held"]


def test_a_pin_that_does_not_reproduce_the_default_keeps_every_default():
    # Gemm_Hipblaslt_431442 cannot run (8192, 4096) on gfx950 (TunableOp's
    # tuning lists it unsupported), so TunableOp would skip that GEMM. A fresh
    # process, since the pins of this one already enabled TunableOp.
    script = textwrap.dedent("""
        import json, logging, torch
        logging.basicConfig(level=logging.INFO)
        from tokenspeed_kernel.ops.gemm import rocm_blas
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        pins = {(8192, 4096): ((72, 72, "Gemm_Hipblaslt_431442"),)}
        state = rocm_blas._PinState({rocm_blas.device_key(props): pins}, rocm_blas._PINNED_BUILD)
        active = bool(state.resolve())
        x = torch.randn(72, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16)
        out = torch.full((72, 8192), float("nan"), device="cuda", dtype=torch.bfloat16)
        torch.mm(x, w.t(), out=out)
        print(json.dumps({"active": active, "enabled": torch.cuda.tunable.is_enabled(),
                          "written": not bool(out.isnan().any())}))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"active": False, "enabled": False, "written": True}
    lines = [l for l in result.stderr.splitlines() if "rocm_blas_pinned_mm:" in l]
    assert len(lines) == 1 and lines[0].startswith("WARNING")
    assert "Gemm_Hipblaslt_431442 at (N, K, rows) (8192, 4096, 72)" in lines[0]


def test_a_first_call_during_graph_capture_leaves_the_pins_for_the_next_call():
    # A fresh process, so this call is the first to consult the pins.
    script = textwrap.dedent("""
        import json, torch
        from tokenspeed_kernel.ops.gemm import rocm_blas
        state = rocm_blas._PinState(rocm_blas._PINS, rocm_blas._PINNED_BUILD)
        x = torch.randn(8, 8, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            capturing = bool(state.resolve())
            y = x + 1
        print(json.dumps({"capturing": capturing, "after": bool(state.resolve())}))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"capturing": False, "after": True}


def test_another_build_or_enabled_tunableop_keeps_default_gemms(caplog):
    results = torch.cuda.tunable.get_results()
    other_build = rocm_blas._PinState(rocm_blas._PINS, (("PT_VERSION", "0.0.0"),))
    # TunableOp is already on here (the active pins enabled it), as a user's
    # own TunableOp run would leave it.
    enabled = rocm_blas._PinState(rocm_blas._PINS, rocm_blas._PINNED_BUILD)
    with caplog.at_level(logging.INFO, logger=rocm_blas.__name__):
        for state in (other_build, other_build, enabled):
            assert state.resolve() == {}
    assert len(caplog.records) == 2
    assert torch.cuda.tunable.get_results() == results


def test_the_device_key_ignores_the_reported_name():
    # Some MI355X hosts report "AMD Radeon Graphics"; the pins must still apply.
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    generic = SimpleNamespace(
        name="AMD Radeon Graphics",
        gcnArchName=props.gcnArchName,
        pci_domain_id=props.pci_domain_id,
        pci_bus_id=props.pci_bus_id,
        pci_device_id=props.pci_device_id,
    )
    assert rocm_blas.device_key(generic) == rocm_blas.device_key(props)
    assert rocm_blas.device_key(generic) in rocm_blas._PINS


def test_a_device_without_pins_says_so_once(caplog):
    state = rocm_blas._PinState({}, rocm_blas._PINNED_BUILD)
    with caplog.at_level(logging.INFO, logger=rocm_blas.__name__):
        assert state.resolve() == {}
        assert state.resolve() == {}
    assert len(caplog.records) == 1
    assert "no pins are listed" in caplog.records[0].getMessage()
