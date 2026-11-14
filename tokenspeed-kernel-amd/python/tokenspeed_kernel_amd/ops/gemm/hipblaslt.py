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

"""Offline-tuned BLAS solutions (TunableOp) for dense GEMMs on ROCm.

On ROCm, ``torch.mm`` takes the hipBLASLt heuristic's solution for each GEMM
shape. For skinny shapes, such as speculative-decoding verify projections, a
measured solution can be much faster. PyTorch's TunableOp times both
hipBLASLt and rocBLAS solutions, and can replay the fastest one from a results
file, so a tuned file can list solutions from either library.

Tuned files live in this package, one per GPU, named like the Triton kernel
configs: ``configs/device_name=<torch.cuda.get_device_name()>.csv`` with
spaces replaced by underscores (``device_name=AMD_Instinct_MI355X.csv``).
Serving loads the file of its GPU read-only:

* lookups only, so serving never tunes;
* GEMM shapes the file does not list keep PyTorch's default solution;
* no file for the GPU, or a file from a different PyTorch, HIP, hipBLASLt,
  rocBLAS or GPU architecture, is skipped as a whole (logged once), and every
  GEMM keeps the defaults.

Generate or extend the file with
``python -m tokenspeed_kernel_amd.ops.gemm.hipblaslt`` (see ``--help``) on the
GPU and software stack that serves.
"""

from __future__ import annotations

import argparse
import logging
import math
import os

import torch

__all__ = [
    "TUNED_CONFIG_DIR",
    "hipblaslt_file_validators",
    "hipblaslt_tuned_file",
    "load_hipblaslt_tuned_solutions",
    "tune_hipblaslt_solutions",
]

logger = logging.getLogger(__name__)

TUNED_CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


def hipblaslt_tuned_file(device_name: str | None = None) -> str:
    """Path of the in-repo tuned file for a GPU.

    Args:
        device_name: ``torch.cuda.get_device_name()`` of the GPU; the current
            device when omitted.

    Returns:
        ``<TUNED_CONFIG_DIR>/device_name=<name>.csv``, spaces in the name
        replaced by underscores. The file may not exist.
    """
    if device_name is None:
        device_name = torch.cuda.get_device_name()
    name = device_name.replace(" ", "_")
    return os.path.join(TUNED_CONFIG_DIR, f"device_name={name}.csv")


def hipblaslt_file_validators(path: str) -> dict[str, str]:
    """Read the software and hardware stamp of a TunableOp results file.

    Args:
        path: TunableOp results CSV.

    Returns:
        ``Validator`` rows as ``{key: value}`` (such as ``PT_VERSION``,
        ``HIPBLASLT_VERSION``, ``GCN_ARCH_NAME``).

    Raises:
        OSError: The file cannot be read.
    """
    validators = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split(",")
            if len(fields) == 3 and fields[0] == "Validator":
                validators[fields[1]] = fields[2]
    return validators


def _current_validators() -> dict[str, str]:
    # TunableOp registers the BLAS library stamps when its first GEMM runs.
    a = torch.zeros(16, 16, device="cuda", dtype=torch.bfloat16)
    torch.mm(a, a)
    return dict(torch.cuda.tunable.get_validators())


def load_hipblaslt_tuned_solutions(path: str | None = None) -> bool:
    """Replay the BLAS solutions (hipBLASLt or rocBLAS) in a TunableOp results file.

    Process-wide: every later ``torch.mm``-family GEMM whose shape, layout
    and dtype the file lists uses the recorded solution; other GEMMs keep
    PyTorch's default. Call before CUDA graph capture, on every process that
    runs GEMMs. Nothing is tuned or written while serving.

    Args:
        path: TunableOp results CSV made on the same software stack and GPU
            architecture. Defaults to the in-repo file of the current GPU,
            ``hipblaslt_tuned_file()``.

    Returns:
        Whether the file was loaded. False when this is not a ROCm build,
        the GPU has no tuned file, the file cannot be read, or its stamp
        differs from the running stack (logged); TunableOp then stays
        disabled.
    """
    tunable = torch.cuda.tunable
    if torch.version.hip is None:
        logger.warning(f"Ignoring tuned BLAS solutions {path}: not a ROCm build")
        return False
    if path is None:
        path = hipblaslt_tuned_file()
        if not os.path.exists(path):
            logger.info(
                f"No tuned BLAS solutions for {torch.cuda.get_device_name()!r} "
                f"({path}); using default solutions"
            )
            return False
    try:
        expected = hipblaslt_file_validators(path)
    except OSError as e:
        logger.warning(f"Ignoring tuned BLAS solutions {path}: {e}")
        return False
    # Lookups only: no tuning on a miss and no untuned-shape log. TunableOp
    # rewrites its file at exit only when it tuned something new, so the file
    # stays untouched.
    tunable.tuning_enable(False)
    tunable.record_untuned_enable(False)
    # TunableOp reads its file lazily on first use; point it at this one so it
    # never picks up a stray default-named file from the working directory.
    tunable.set_filename(path, insert_device_ordinal=False)
    tunable.enable(True)
    current = _current_validators()
    mismatch = {
        key: (value, current.get(key))
        for key, value in expected.items()
        if current.get(key) != value
    }
    missing = sorted(set(current) - set(expected))
    if not expected or mismatch or missing:
        tunable.enable(False)
        logger.warning(
            f"Ignoring tuned BLAS solutions {path}: made for a different "
            f"stack (file vs running: {mismatch}, unstamped: {missing}); "
            "using default solutions"
        )
        return False
    if not tunable.read_file(path):
        tunable.enable(False)
        logger.warning(f"Could not load tuned BLAS solutions {path}")
        return False
    logger.info(
        f"Loaded {len(tunable.get_results())} tuned BLAS solutions (TunableOp) "
        f"from {path}"
    )
    return True


def tune_hipblaslt_solutions(
    *,
    shapes: list[tuple[int, int]],
    rows: list[int],
    dtype: torch.dtype,
    path: str | None = None,
    cold_bytes: int,
) -> int:
    """Measure the fastest BLAS solution for ``x @ weight.T`` and save them.

    Times every solution TunableOp offers for each ``[m, k] @ [n, k].T`` GEMM,
    with weights rotated through ``cold_bytes`` of copies so each measurement
    reads its weight from HBM, as a decode step does.

    Args:
        shapes: ``(n, k)`` weight shapes, as in ``nn.Linear.weight``.
        rows: Activation row counts ``m`` to tune, such as the token counts
            of captured verify or decode graphs.
        dtype: Activation and weight dtype.
        path: Output TunableOp results CSV. Defaults to the in-repo file of
            the current GPU, ``hipblaslt_tuned_file()``. An existing file
            must carry the running stack's stamp; its GEMMs are kept and not
            re-timed, and the new ones are added. Delete it to re-tune all.
        cold_bytes: Rotating buffer size in bytes.

    Returns:
        Number of tuned GEMMs written, existing ones included.

    Raises:
        ValueError: ``path`` exists and was made for another stack.
    """
    if path is None:
        path = hipblaslt_tuned_file()
    tunable = torch.cuda.tunable
    tunable.tuning_enable(False)
    tunable.record_untuned_enable(False)
    # TunableOp reads this file before its first GEMM, so the GEMMs it lists
    # are not re-timed, and also writes its results here at process exit;
    # point it here so neither uses a file in the working directory.
    tunable.set_filename(path, insert_device_ordinal=False)
    tunable.set_rotating_buffer_size(math.ceil(cold_bytes / 2**20))
    tunable.enable(True)
    if os.path.exists(path):
        # Stamp check with tuning still off, so its probe GEMM is not tuned.
        stamp = hipblaslt_file_validators(path)
        current = _current_validators()
        if stamp != current:
            tunable.enable(False)
            raise ValueError(
                f"{path} was tuned on another stack (file {stamp}, running "
                f"{current}); delete it to re-tune"
            )
        if not tunable.read_file(path):
            tunable.enable(False)
            raise ValueError(f"Could not read tuned BLAS solutions {path}")
    tunable.tuning_enable(True)
    for n, k in shapes:
        weight = torch.randn(n, k, device="cuda", dtype=dtype)
        for m in rows:
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            torch.mm(x, weight.t())
    torch.cuda.synchronize()
    tunable.enable(False)
    results = tunable.get_results()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for key, value in tunable.get_validators():
            f.write(f"Validator,{key},{value}\n")
        for op, params, solution, ms in results:
            f.write(f"{op},{params},{solution},{ms}\n")
    return len(results)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune BLAS solutions (hipBLASLt and rocBLAS, via TunableOp) "
        "for dense projections "
        "(writes or extends the file load_hipblaslt_tuned_solutions reads)."
    )
    parser.add_argument(
        "--shapes",
        required=True,
        help="comma-separated NxK weight shapes, e.g. 14336x5120,5120x6144",
    )
    parser.add_argument(
        "--rows", required=True, help="comma-separated row counts, e.g. 64,128,256"
    )
    parser.add_argument("--dtype", required=True, choices=("bfloat16", "float16"))
    parser.add_argument(
        "--out",
        default=None,
        help="output results CSV (default: this GPU's in-repo file, "
        "configs/device_name=<GPU name>.csv)",
    )
    parser.add_argument(
        "--cold-bytes",
        type=int,
        default=512 * 2**20,
        help="rotating weight buffer, larger than the last-level cache",
    )
    args = parser.parse_args()
    shapes = [tuple(map(int, s.split("x"))) for s in args.shapes.split(",")]
    rows = [int(m) for m in args.rows.split(",")]
    count = tune_hipblaslt_solutions(
        shapes=shapes,
        rows=rows,
        dtype=getattr(torch, args.dtype),
        path=args.out,
        cold_bytes=args.cold_bytes,
    )
    print(f"wrote {count} tuned GEMMs to {args.out or hipblaslt_tuned_file()}")


if __name__ == "__main__":
    _main()
