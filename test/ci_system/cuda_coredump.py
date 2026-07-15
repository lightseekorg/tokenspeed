"""Explicit CUDA coredump helpers for the runtime CI suite.

``run_ci_suite.py --cuda-coredump-dir PATH`` calls :func:`configure` before it
launches test subprocesses. The resulting ``CUDA_*`` variables are an NVIDIA
driver protocol projected from that explicit CLI option; this module has no
import-time or TokenSpeed environment-variable switch.
"""

import glob
import os
from collections.abc import Mapping, MutableMapping

_CUDA_COREDUMP_FLAGS = (
    "skip_nonrelocated_elf_images,skip_global_memory,"
    "skip_shared_memory,skip_local_memory,skip_constbank_memory"
)


def _driver_environment(dump_dir: str) -> dict[str, str]:
    return {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
        "CUDA_COREDUMP_GENERATION_FLAGS": _CUDA_COREDUMP_FLAGS,
        "CUDA_COREDUMP_FILE": f"{dump_dir}/cuda_coredump_%h.%p.%t",
    }


def configure(
    dump_dir: str,
    *,
    env: MutableMapping[str, str],
) -> None:
    """Project an explicit coredump directory into NVIDIA driver variables.

    Existing conflicting values fail closed rather than silently overriding
    the requested CI configuration. Matching values are harmless and make the
    function idempotent for callers that prepare more than one test batch.
    """

    if not dump_dir:
        raise ValueError("CUDA coredump directory must be non-empty")
    env_vars = _driver_environment(dump_dir)
    # Validate the complete projection before creating a directory or mutating
    # the child environment. A conflict must leave both inputs untouched.
    for key, value in env_vars.items():
        existing = env.get(key)
        if existing is not None and existing != value:
            raise ValueError(
                f"explicit CUDA coredump setting for {key} conflicts with "
                f"an existing value"
            )
    os.makedirs(dump_dir, exist_ok=True)
    env.update(env_vars)


def disable(env: MutableMapping[str, str]) -> None:
    """Remove inherited NVIDIA coredump controls from a child environment."""

    for key in _driver_environment(""):
        env.pop(key, None)


def cleanup_dump_dir(dump_dir: str) -> None:
    """Remove stale coredump files from the dump directory."""
    for dump_file in glob.glob(os.path.join(dump_dir, "cuda_coredump_*")):
        os.remove(dump_file)


def report(dump_dir: str, *, env: Mapping[str, str] | None = None) -> None:
    """Log any CUDA coredump files found after a test failure."""
    inherited = os.environ if env is None else env
    coredump_files = glob.glob(os.path.join(dump_dir, "cuda_coredump_*"))
    if not coredump_files:
        return

    separator = "=" * 60
    print(f"\n{separator}")
    print(f"CUDA coredump(s) detected ({len(coredump_files)} file(s)):")
    for dump_file in coredump_files:
        size_mb = os.path.getsize(dump_file) / (1024 * 1024)
        print(f"  {dump_file} ({size_mb:.1f} MB)")
    print("Use cuda-gdb to analyze: cuda-gdb -c <coredump_file>")

    run_id = inherited.get("GITHUB_RUN_ID")
    if run_id:
        repo = inherited.get("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
        print(f"Download from CI: gh run download {run_id} --repo {repo}")

    print(f"{separator}\n")
