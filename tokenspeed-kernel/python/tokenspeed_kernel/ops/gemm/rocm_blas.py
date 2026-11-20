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

"""Pinned rocBLAS / hipBLASLt solutions for measured BF16 GEMMs.

For some decode and prefill shapes the default ``torch.mm`` heuristic picks a
slower solution than another that produces the same bits. ``rocm_blas_pinned_mm``
runs the faster one through PyTorch TunableOp, the only public way to run a
given solution index. A pin is listed only for a device and library build
where it was checked bitwise against the default at every pinned row count;
anywhere else the registry keeps today's GEMM and logs once.

The first GEMM selection that consults the pins enables TunableOp for the
process with tuning off and the pins as its only results. A GEMM whose key has no result runs
TunableOp's default solution, the one ``torch.mm`` runs with TunableOp off, so
every other GEMM keeps its bits, in any thread.

TunableOp skips a GEMM whose solution cannot run the problem and leaves its
output unwritten, without an error. So activation then runs every pin once
against the default and keeps the default GEMMs for every shape, with a
warning, if any pin does not reproduce it.
"""

from __future__ import annotations

import logging
import os
import tempfile

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["rocm_blas_pinned_mm", "rocm_blas_pinned_solution"]

logger = logging.getLogger(__name__)

# The TunableOp validators (PyTorch, HIP, hipBLASLt, rocBLAS, arch) of the build
# the pins below were checked on. Solution indices are specific to it.
_PINNED_BUILD = (
    ("PT_VERSION", "2.14.0"),
    ("HIP_VERSION", "702"),
    ("HIPBLASLT_VERSION", "100202-dabb6df2b9"),
    ("GCN_ARCH_NAME", "gfx950:sramecc+:xnack-"),
    ("ROCBLAS_VERSION", "5.2.0.dabb6df2b9"),
)

# Per device: (N, K) of ``x @ weight.T`` -> (first row, last row, solution).
# Each solution reproduced the default output bit for bit at every row count
# in its range (three inputs each) and ran faster in CUDA-graph replays with
# cold weights (at least 1 GiB of weight copies), on MI355X with the build in
# ``_PINNED_BUILD``. Other devices keep the default GEMMs until their pins are
# checked the same way.
#
# (8192, 2048) is a TP4 attention output projection of a speculative verify
# (8 rows per request): 49-63 rows (7 requests) and 65-96 rows run hipBLASLt solutions,
# 64 rows a rocBLAS one, and 320 rows (40 requests) another hipBLASLt one. At
# 72-73 rows the default is a split-K solution with another summation order,
# which the pins reproduce; 33-48 rows have no faster solution with the
# default's bits. (8192, 4096) is the unpooler at 74-96 rows. The prefill pins
# hold single row counts, since near them the default's bits follow the row
# count: (4096, 8192) is the pre-expert projection of a rank's 2048 owned rows
# (an 8192-token chunk), (18432, 8192) the owned-row Q / K / V / gate
# projection at 4096 rows (a 16384-token chunk), and (512, 8192) the DFlash
# draft's K / V projection of a 16384-token chunk's context rows. They were
# checked at both 1900 and 2000 MHz memory clocks, whose default solutions
# differ for some shapes.
#
# Devices are keyed by (GFX arch, PCI device id), read from the hardware:
# some MI355X hosts report the generic name "AMD Radeon Graphics".
_MI355X = ("gfx950", 0x75A3)
_PINS: dict[
    tuple[str, int], dict[tuple[int, int], tuple[tuple[int, int, str], ...]]
] = {
    _MI355X: {
        (8192, 2048): (
            (49, 63, "Gemm_Hipblaslt_439845"),
            (64, 64, "Gemm_Rocblas_439846"),
            (65, 96, "Gemm_Hipblaslt_441404"),
            (320, 320, "Gemm_Hipblaslt_440127"),
        ),
        (8192, 4096): ((74, 96, "Gemm_Rocblas_441444"),),
        (4096, 8192): ((2048, 2048, "Gemm_Hipblaslt_440334"),),
        (18432, 8192): ((4096, 4096, "Gemm_Rocblas_440625"),),
        (512, 8192): ((16384, 16384, "Gemm_Hipblaslt_440535"),),
    },
}

_GFX950_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(9, 5),
    max_arch_version=ArchVersion(9, 5),
    vendors=frozenset({"amd"}),
)


def device_key(props) -> tuple[str, int] | None:
    """(GFX arch, PCI device id) of a device's properties, e.g. ("gfx950",
    0x75A3) for an MI355X; None when the PCI id cannot be read.

    The id comes from sysfs at the device's PCI address, so it holds whatever
    name the driver reports.
    """
    path = (
        f"/sys/bus/pci/devices/{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:"
        f"{props.pci_device_id:02x}.0/device"
    )
    try:
        with open(path) as device:
            pci_id = int(device.read(), 16)
    except (OSError, ValueError):
        return None
    return props.gcnArchName.split(":")[0], pci_id


def _tunableop_key(m: int, n: int, k: int) -> str:
    # torch.mm(x, weight.T) with contiguous operands and output is a TN GEMM.
    return f"tn_{n}_{m}_{k}_ld_{k}_{k}_{n}"


class _PinState:
    """One device's pins, active in TunableOp once checked against the build."""

    def __init__(
        self,
        table: dict[
            tuple[str, int], dict[tuple[int, int], tuple[tuple[int, int, str], ...]]
        ],
        build: tuple[tuple[str, str], ...],
    ) -> None:
        self.table = table
        self.build = build
        self.pins: dict[tuple[int, int], tuple[tuple[int, int, str], ...]] | None = None

    def resolve(self) -> dict[tuple[int, int], tuple[tuple[int, int, str], ...]]:
        """This device's pins, activating them on the first call; {} if none apply."""
        if self.pins is None:
            if torch.cuda.is_current_stream_capturing():
                # Activation compares GEMM outputs on the host, which a
                # capturing stream does not allow: keep the defaults for this
                # call and let the next eager call activate the pins.
                return {}
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            key = device_key(props)
            pins = self.table.get(key, {})
            if key is None:
                reason = "its PCI device id is unreadable"
            elif not pins:
                reason = "no pins are listed for it"
            else:
                reason = self._activate(pins)
            device = f"{props.name} {key}"
            mismatch = self._check(pins) if reason is None else None
            if mismatch is not None:
                torch.cuda.tunable.enable(False)
                logger.warning(
                    f"rocm_blas_pinned_mm: {device}: keeping the default GEMM "
                    f"solutions for every shape: {mismatch}"
                )
                pins = {}
            elif reason is None:
                count = sum(
                    last - first + 1 for r in pins.values() for first, last, _ in r
                )
                logger.info(
                    f"rocm_blas_pinned_mm: {device}: {count} pinned (N, K, rows) "
                    f"of {len(pins)} shapes active"
                )
            else:
                logger.info(
                    f"rocm_blas_pinned_mm: {device}: keeping the default GEMM "
                    f"solutions: {reason}"
                )
                pins = {}
            self.pins = pins
        return self.pins

    def _activate(
        self, pins: dict[tuple[int, int], tuple[tuple[int, int, str], ...]]
    ) -> str | None:
        """Make the pins TunableOp's only results; the reason if they cannot be."""
        tunable = torch.cuda.tunable
        validators = dict(tunable.get_validators())
        build = dict(self.build)
        if {name: validators.get(name) for name in build} != build:
            return f"build {validators} differs from the pinned build {build}"
        if tunable.is_enabled():
            return "TunableOp is already enabled, so its own results decide GEMMs"
        expected = {
            ("GemmTunableOp_BFloat16_TN", _tunableop_key(m, n, k), solution)
            for (n, k), ranges in pins.items()
            for first, last, solution in ranges
            for m in range(first, last + 1)
        }
        rows = [f"Validator,{name},{value}" for name, value in self.build]
        rows += [f"{op},{key},{solution},0.0" for op, key, solution in expected]
        handle, path = tempfile.mkstemp(prefix="tokenspeed-rocm-blas-", suffix=".csv")
        try:
            with os.fdopen(handle, "w") as results:
                results.write("\n".join(rows) + "\n")
            # Enabling reads the results file once; naming this one keeps a
            # stale file in the working directory (or PYTORCH_TUNABLEOP_FILENAME)
            # from adding other solutions.
            tunable.set_filename(path)
            tunable.tuning_enable(False)
            if not tunable.read_file(path):
                return "TunableOp rejected the pinned solutions"
            tunable.enable(True)
        finally:
            os.unlink(path)
        held = {tuple(result[:3]) for result in tunable.get_results()}
        if held != expected:
            tunable.enable(False)
            return f"TunableOp holds other results: {sorted(held - expected)[:4]}"
        return None

    def _check(
        self, pins: dict[tuple[int, int], tuple[tuple[int, int, str], ...]]
    ) -> str | None:
        """Run every active pin once against the default GEMM; the first
        (N, K, rows) whose bits differ, or None.

        A solution that cannot run a problem (unsupported here, or needing more
        workspace than PyTorch gives hipBLASLt) makes TunableOp skip the GEMM
        without an error, leaving the output unwritten. The pinned GEMM writes
        into a NaN-filled buffer, so a skipped GEMM never matches.
        """
        tunable = torch.cuda.tunable
        generator = torch.Generator(device="cuda").manual_seed(0)
        for (n, k), ranges in pins.items():
            weight = torch.randn(
                n, k, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            rows = max(last for _, last, _ in ranges)
            x = torch.randn(
                rows, k, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            for first, last, solution in ranges:
                for m in range(first, last + 1):
                    tunable.enable(False)
                    expected = torch.mm(x[:m], weight.t())
                    tunable.enable(True)
                    out = torch.full(
                        (m, n), float("nan"), device="cuda", dtype=torch.bfloat16
                    )
                    torch.mm(x[:m], weight.t(), out=out)
                    if not torch.equal(out, expected):
                        return (
                            f"{solution} at (N, K, rows) ({n}, {k}, {m}) does not "
                            "reproduce the default GEMM bit for bit"
                        )
        return None


_STATE = _PinState(_PINS, _PINNED_BUILD)


def rocm_blas_pinned_solution(m: int, n: int, k: int) -> str | None:
    """The pinned solution for ``[m, k] x [n, k]^T`` on this device, or None.

    Args:
        m: Rows of ``x``.
        n: Rows of ``weight`` (output columns).
        k: Shared inner dimension.

    Returns:
        The TunableOp solution name, or None where the default GEMM stays.
    """
    for first, last, solution in _STATE.resolve().get((n, k), ()):
        if first <= m <= last:
            return solution
    return None


def _supports_pinned_problem(m: int, n: int, k: int) -> bool:
    return rocm_blas_pinned_solution(m, n, k) is not None


# The gluon decode tiles cover these (N, K) only up to 32 rows. At equal
# priority the registry keeps import order, so a gluon tile added for pinned
# rows would be selected first.
@register_kernel(
    "gemm",
    "mv",
    name="rocm_blas_pinned_mm",
    solution="rocm_blas",
    capability=_GFX950_CAPABILITY,
    signatures=frozenset(
        {
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    priority=Priority.SPECIALIZED,
    traits={"mnk_problem_filter": frozenset({_supports_pinned_problem})},
)
def rocm_blas_pinned_mm(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` through the solution pinned for this shape.

    Bitwise equal to ``torch.mm`` with TunableOp off, on the devices and builds
    listed in this module. Selecting this kernel activated the pins, so the
    call is ``torch.mm`` itself.

    Args:
        x: ``[M, K]`` contiguous BF16 activations.
        weight: ``[N, K]`` contiguous BF16 weight.
        out: Optional contiguous ``[M, N]`` BF16 destination.

    Returns:
        ``[M, N]`` BF16 output.
    """
    if out is not None and not out.is_contiguous():
        # TunableOp keys the solution on the output's leading dimension.
        raise ValueError("rocm_blas_pinned_mm needs a contiguous out")
    if out is None:
        return torch.mm(x, weight.t())
    return torch.mm(x, weight.t(), out=out)
