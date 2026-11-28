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

Tuned files live next to this module, in two kinds. TunableOp checks a
file's solutions against the GPU architecture and library builds stamped in
it, not the GPU model, and some drivers report a generic device name (such as
``AMD Radeon Graphics``), so:

* ``configs/gcn_arch=<architecture>.csv`` serves every GPU whose
  ``gcnArchName`` equals the file's ``GCN_ARCH_NAME`` stamp. It holds only
  rows verified on every GPU model of the architecture to write the default
  solution's bits and not be slower.
* ``configs/device_name=<torch.cuda.get_device_name()>.csv``, spaces replaced
  by underscores (``device_name=AMD_Instinct_MI355X.csv``), adds rows for one
  GPU model. The solutions in these rows may be chosen for speed alone. It
  never lists a GEMM the architecture file lists.

Serving loads the architecture file and then the device file, read-only:

* lookups only, so serving never tunes;
* GEMM shapes the files do not list keep PyTorch's default solution;
* a file from a different PyTorch, HIP, hipBLASLt or rocBLAS build, or an
  unreadable one, is skipped as a whole with a warning; the GPU's other file
  still loads;
* with no file left to load, a warning says the GEMMs run untuned, and every
  GEMM keeps the defaults.

Opt-in online tuning, ``hipblaslt_online_tuning``, times the GEMMs no loaded
file lists while a block runs (serving runs its warmup decode steps there,
before graph capture) and keeps the picks in the process. Around its blocks,
``hipblaslt_online_tuning_cache`` replays the picks a previous run saved in a
cache file outside the repository (when its stamp matches) and saves the
blocks' picks there, so a later start times only the GEMMs it adds; the
committed files are never written.

Generate or extend the file with
``python -m tokenspeed_kernel.ops.gemm._hipblaslt.tuning`` (see ``--help``) on the
GPU and software stack that serves.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import logging
import math
import os
import tempfile
import time
from typing import Iterator

import torch

__all__ = [
    "GENERIC_DEVICE_NAMES",
    "TUNED_CONFIG_DIR",
    "find_hipblaslt_tuned_files",
    "hipblaslt_file_validators",
    "hipblaslt_online_tuning",
    "hipblaslt_online_tuning_cache",
    "hipblaslt_tuned_file",
    "load_hipblaslt_tuned_solutions",
    "tune_hipblaslt_solutions",
]

logger = logging.getLogger(__name__)

TUNED_CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")

# Device names drivers report for several GPU models of one architecture. A
# device file named after one would load on all of them, although its rows
# may be chosen for one model's speed, so none is committed (a test checks)
# and the tuning tool does not write one by default.
GENERIC_DEVICE_NAMES = frozenset({"AMD Radeon Graphics"})


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


def find_hipblaslt_tuned_files(device_name: str, arch_name: str) -> list[str]:
    """The committed tuned files for a GPU, in load order.

    Args:
        device_name: ``torch.cuda.get_device_name()`` of the GPU.
        arch_name: ``gcnArchName`` of the GPU's properties, such as
            ``gfx950:sramecc+:xnack-``.

    Returns:
        The ``gcn_arch=`` file stamped with ``arch_name``, if one is
        committed, then the GPU's ``device_name=`` file, if one is committed
        and stamped with ``arch_name``. Empty when neither is. A file that
        cannot be read is left out with a warning.

    Raises:
        ValueError: Several committed ``gcn_arch=`` files are stamped with
            ``arch_name``.
    """
    stamps = {}
    for path in sorted(glob.glob(os.path.join(TUNED_CONFIG_DIR, "*.csv"))):
        try:
            stamps[path] = hipblaslt_file_validators(path)
        except OSError as e:
            logger.warning(f"Ignoring tuned BLAS solutions {path}: {e}")
    stamped = [
        path
        for path, stamp in stamps.items()
        if stamp.get("GCN_ARCH_NAME") == arch_name
    ]
    architecture = [
        path for path in stamped if os.path.basename(path).startswith("gcn_arch=")
    ]
    if len(architecture) > 1:
        raise ValueError(
            f"several committed tuned files for {arch_name}: {architecture}"
        )
    named = hipblaslt_tuned_file(device_name)
    if named in stamped:
        return architecture + [named]
    if named in stamps:
        logger.warning(
            f"Ignoring tuned BLAS solutions {named}: stamped for "
            f"{stamps[named].get('GCN_ARCH_NAME')!r}, not this GPU's "
            f"{arch_name!r}"
        )
    return architecture


def _current_validators() -> dict[str, str]:
    # TunableOp registers the BLAS library stamps when its first GEMM runs.
    a = torch.zeros(16, 16, device="cuda", dtype=torch.bfloat16)
    torch.mm(a, a)
    return dict(torch.cuda.tunable.get_validators())


def load_hipblaslt_tuned_solutions(path: str | None = None) -> bool:
    """Replay the BLAS solutions (hipBLASLt or rocBLAS) in TunableOp results files.

    Process-wide: every later ``torch.mm``-family GEMM whose shape, layout
    and dtype the file lists uses the recorded solution; other GEMMs keep
    PyTorch's default. Call before CUDA graph capture, on every process that
    runs GEMMs. Nothing is tuned or written while serving.

    Args:
        path: TunableOp results CSV made on the same software stack and GPU
            architecture. Defaults to the committed files of the current GPU,
            ``find_hipblaslt_tuned_files``, loaded together.

    Returns:
        Whether any file was loaded. Each file is loaded only when its stamp
        matches the running stack; one that does not is skipped with a
        warning, and the others still load. False when this is not a ROCm
        build, the GPU has no tuned file, a file cannot be read, or no file
        matches the stack (logged); TunableOp then stays disabled.

    Raises:
        ValueError: ``path`` is omitted and several committed ``gcn_arch=``
            files carry the current GPU's architecture stamp.
    """
    tunable = torch.cuda.tunable
    if torch.version.hip is None:
        logger.warning(f"Ignoring tuned BLAS solutions {path}: not a ROCm build")
        return False
    if path is None:
        device_name = torch.cuda.get_device_name()
        arch_name = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName
        paths = find_hipblaslt_tuned_files(device_name, arch_name)
        if not paths:
            logger.warning(
                f"No tuned BLAS solutions for {device_name!r} ({arch_name}) in "
                f"{TUNED_CONFIG_DIR}: neither a gcn_arch= file nor "
                f"{os.path.basename(hipblaslt_tuned_file(device_name))} is "
                "committed with this architecture's stamp; GEMMs run untuned, "
                "with the BLAS libraries' default solutions"
            )
            return False
    else:
        paths = [path]
    # Lookups only: no tuning on a miss and no untuned-shape log. TunableOp
    # rewrites its file at exit only when it tuned something new; it also
    # reads that file lazily before its first GEMM. Point it at os.devnull, so
    # it reads nothing on its own (only the files whose stamps match below are
    # read) and never picks up a stray default-named file from the working
    # directory.
    tunable.tuning_enable(False)
    tunable.record_untuned_enable(False)
    tunable.set_filename(os.devnull, insert_device_ordinal=False)
    tunable.enable(True)
    current = _current_validators()
    loaded = []
    for file in paths:
        try:
            stamp = hipblaslt_file_validators(file)
        except OSError as e:
            logger.warning(f"Ignoring tuned BLAS solutions {file}: {e}")
            continue
        mismatch = {
            key: (value, current.get(key))
            for key, value in stamp.items()
            if current.get(key) != value
        }
        missing = sorted(set(current) - set(stamp))
        if not stamp or mismatch or missing:
            logger.warning(
                f"Ignoring tuned BLAS solutions {file}: made for a different "
                f"stack (file vs running: {mismatch}, unstamped: {missing}); "
                "its GEMMs use default solutions"
            )
            continue
        # Each read adds the file's rows to those already loaded.
        if not tunable.read_file(file):
            logger.warning(
                f"Could not load tuned BLAS solutions {file}; its GEMMs use "
                "default solutions"
            )
            continue
        loaded.append(file)
    if not loaded:
        tunable.enable(False)
        return False
    logger.info(
        f"Loaded {len(tunable.get_results())} tuned BLAS solutions (TunableOp) "
        f"from {', '.join(loaded)}"
    )
    return True


def _result_keys() -> set[tuple[str, str]]:
    return {tuple(result[:2]) for result in torch.cuda.tunable.get_results()}


def _read_online_cache(path: str, loaded: set[tuple[str, str]]) -> None:
    # A missing, unreadable or stale cache is re-tuned and rewritten.
    if not os.path.exists(path):
        return
    try:
        stamp = hipblaslt_file_validators(path)
        with open(path) as f:
            keys = {
                tuple(line.split(",")[:2])
                for line in f
                if not line.startswith("Validator")
            }
    except OSError as e:
        logger.warning(f"Ignoring online-tuned BLAS solutions {path}: {e}")
        return
    current = _current_validators()
    if stamp != current:
        logger.warning(
            f"Ignoring online-tuned BLAS solutions {path}: made for a different "
            f"stack (file {stamp}, running {current}); re-tuning"
        )
        return
    if keys & loaded:
        # A committed file now lists some of its GEMMs; the committed rows win.
        logger.warning(
            f"Ignoring online-tuned BLAS solutions {path}: a loaded file lists "
            "some of its GEMMs; re-tuning"
        )
        return
    if not torch.cuda.tunable.read_file(path):
        logger.warning(f"Could not load online-tuned BLAS solutions {path}")
        return
    logger.info(f"Loaded {len(keys)} online-tuned BLAS solutions from {path}")


def _write_online_cache(path: str, loaded: set[tuple[str, str]]) -> None:
    tunable = torch.cuda.tunable
    lines = [f"Validator,{key},{value}\n" for key, value in tunable.get_validators()]
    lines += [
        f"{op},{params},{solution},{ms}\n"
        for op, params, solution, ms in tunable.get_results()
        if (op, params) not in loaded
    ]
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # Replaced atomically: other processes may be reading it.
    with tempfile.NamedTemporaryFile(
        "w", dir=directory, suffix=".tmp", delete=False
    ) as tmp:
        tmp.writelines(lines)
    os.replace(tmp.name, path)


def _enable_lookups() -> None:
    tunable = torch.cuda.tunable
    if tunable.is_enabled():
        return
    # No file loaded. Point TunableOp away from its default file in the
    # working directory before its first lookup reads it. Tuning stays off
    # until a tuning block: enable() turns it on, and a GEMM run before the
    # block (checking a results file's stamp runs one) would be tuned and
    # recorded.
    tunable.tuning_enable(False)
    tunable.set_filename(os.devnull, insert_device_ordinal=False)
    tunable.record_untuned_enable(False)
    tunable.enable(True)


@contextlib.contextmanager
def hipblaslt_online_tuning(cold_bytes: int = 512 * 2**20) -> Iterator[None]:
    """Tune, in this process, the GEMMs no loaded file lists.

    Inside the block, the first ``torch.mm``-family GEMM of each new shape,
    layout and dtype times every hipBLASLt and rocBLAS solution TunableOp
    offers and keeps the fastest; GEMMs already loaded (such as from
    ``load_hipblaslt_tuned_solutions`` or ``hipblaslt_online_tuning_cache``)
    are not re-timed. As in the offline tool, operands rotate through
    ``cold_bytes`` of copies, so each timing reads the weight from HBM. On
    exit tuning stops and TunableOp stays on, so later GEMMs replay the loaded
    and the new solutions and other GEMMs keep the default. The committed
    files are never written. Use it outside graph capture.

    Args:
        cold_bytes: Rotating buffer size in bytes while tuning.

    Yields:
        None.
    """
    tunable = torch.cuda.tunable
    _enable_lookups()
    loaded = len(tunable.get_results())
    # TunableOp appends each new result to its file as it tunes; os.devnull
    # keeps them in memory and the loaded file unchanged. The loaded entries
    # stay in memory, so the file name is not restored.
    tunable.set_filename(os.devnull, insert_device_ordinal=False)
    rotating_mb = tunable.get_rotating_buffer_size()
    tunable.set_rotating_buffer_size(math.ceil(cold_bytes / 2**20))
    tunable.tuning_enable(True)
    tic = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        tunable.tuning_enable(False)
        tunable.set_rotating_buffer_size(rotating_mb)
        logger.info(
            f"Tuned {len(tunable.get_results()) - loaded} BLAS solutions "
            f"(TunableOp) online in {time.perf_counter() - tic:.1f}s"
        )


@contextlib.contextmanager
def hipblaslt_online_tuning_cache(path: str, *, write: bool) -> Iterator[None]:
    """Keep the picks of the online tuning blocks inside across starts.

    On entry, loads ``path``, a TunableOp results file of earlier online
    tuning on this stack, so the ``hipblaslt_online_tuning`` blocks inside do
    not re-time its GEMMs. The file is ignored (and its GEMMs re-tuned) when
    its stamp differs from the running stack's or a loaded committed file
    lists one of its GEMMs. Load the committed files before entering.

    Args:
        path: The results file, such as ``hipblaslt_online_cache_path()``.
        write: On a normal exit, write ``path`` with every solution not loaded
            before the block: those read from it and the newly tuned ones.
            One process of a group sharing the file writes it.

    Yields:
        None.
    """
    _enable_lookups()
    committed = _result_keys()
    _read_online_cache(path, committed)
    yield
    if write:
        _write_online_cache(path, committed)


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
        path: Output TunableOp results CSV. Defaults to the current GPU's
            device file, ``hipblaslt_tuned_file()``. An existing file must
            carry the running stack's stamp; its GEMMs are kept and not
            re-timed, and the new ones are added. Deleting it re-tunes that
            file's GEMMs only. Unless ``path`` is the architecture file
            itself, GEMMs the GPU's committed architecture file lists (when
            its stamp matches) are neither re-timed nor written, also for a
            scratch ``path``; they are logged. Re-tuning the architecture
            file's rows means regenerating that file as ``path``.
        cold_bytes: Rotating buffer size in bytes.

    Returns:
        Number of tuned GEMMs written to ``path``: those already in it and
        the new ones. For any ``path`` but the architecture file this leaves
        out the architecture file's GEMMs; regenerating the architecture
        file keeps and counts its existing rows.

    Raises:
        ValueError: ``path`` exists and was made for another stack, ``path``
            or the GPU's architecture file cannot be read, ``path`` is
            omitted on a GPU reporting a name in ``GENERIC_DEVICE_NAMES``, or
            several committed ``gcn_arch=`` files carry the GPU's
            architecture stamp.
    """
    if path is None:
        device_name = torch.cuda.get_device_name()
        if device_name in GENERIC_DEVICE_NAMES:
            raise ValueError(
                f"this GPU reports the generic device name {device_name!r}, "
                "shared by several GPU models; pass path explicitly instead "
                "of writing a device file for it"
            )
        path = hipblaslt_tuned_file(device_name)
    # Resolve the committed files before enabling TunableOp, so a conflict
    # among them raises with TunableOp still off.
    committed = find_hipblaslt_tuned_files(
        torch.cuda.get_device_name(),
        torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName,
    )
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
        try:
            stamp = hipblaslt_file_validators(path)
        except OSError as e:
            tunable.enable(False)
            raise ValueError(f"Could not read tuned BLAS solutions {path}") from e
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
    # The GPU's architecture file already lists its GEMMs: read it, so they are
    # not re-timed, and keep them out of the output, since a device file never
    # lists a GEMM of its architecture file.
    architecture_keys = set()
    for other in committed:
        if not os.path.basename(other).startswith("gcn_arch=") or (
            os.path.realpath(other) == os.path.realpath(path)
        ):
            continue
        try:
            stamp = hipblaslt_file_validators(other)
            if stamp != _current_validators():
                logger.warning(
                    f"{other} was made for another stack: its GEMMs are tuned "
                    f"and written to {path} too; regenerate it by its own rule"
                )
                continue
            readable = tunable.read_file(other)
            with open(other) as f:
                architecture_keys |= {
                    tuple(line.split(",")[:2])
                    for line in f
                    if not line.startswith("Validator")
                }
        except OSError as e:
            tunable.enable(False)
            raise ValueError(f"Could not read tuned BLAS solutions {other}") from e
        if not readable:
            tunable.enable(False)
            raise ValueError(f"Could not read tuned BLAS solutions {other}")
    # Everything to keep is read; from here TunableOp writes nowhere on its
    # own, so the architecture rows it holds cannot reach ``path`` at exit.
    # The file is written below, without them.
    tunable.set_filename(os.devnull, insert_device_ordinal=False)
    tunable.tuning_enable(True)
    for n, k in shapes:
        weight = torch.randn(n, k, device="cuda", dtype=dtype)
        for m in rows:
            x = torch.randn(m, k, device="cuda", dtype=dtype)
            torch.mm(x, weight.t())
    torch.cuda.synchronize()
    tunable.enable(False)
    results = [
        result
        for result in tunable.get_results()
        if tuple(result[:2]) not in architecture_keys
    ]
    # TunableOp keys a TN GEMM ``tn_<n>_<m>_<k>_...``.
    skipped = []
    for _, params in architecture_keys:
        n, m, k = (int(field) for field in params.split("_")[1:4])
        if (n, k) in shapes and m in rows:
            skipped.append(f"{n}x{k} at {m} rows")
    skipped.sort()
    if skipped:
        logger.info(
            f"Not re-timed or written to {path}, the architecture file lists "
            f"them: {', '.join(skipped)}"
        )
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
