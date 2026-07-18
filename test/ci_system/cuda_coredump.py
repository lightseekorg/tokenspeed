"""CUDA coredump helpers.

When ``TOKENSPEED_CUDA_COREDUMP=1``, this module injects CUDA coredump
environment variables into the current process so GPU exceptions, such as
illegal memory access, produce lightweight coredump files for post-mortem
analysis with ``cuda-gdb``.

The injection happens at module import time via ``_inject_env()`` on a
best-effort basis. If a ``CUDA_*`` variable is already present in the
environment, for example when set by the user in the shell, injection is
skipped for that variable and a warning is emitted. For strict guarantees, set
the ``CUDA_*`` environment variables in the shell before launching Python.
"""

import glob
import os
import warnings

from tokenspeed.runtime.utils.env import envs

_CUDA_COREDUMP_FLAGS = (
    "skip_nonrelocated_elf_images,skip_global_memory,"
    "skip_shared_memory,skip_local_memory,skip_constbank_memory"
)


def is_enabled() -> bool:
    return envs.TOKENSPEED_CUDA_COREDUMP.get()


def get_dump_dir() -> str:
    return envs.TOKENSPEED_CUDA_COREDUMP_DIR.get()


def _inject_env():
    """Inject CUDA coredump environment variables into the current process.

    If a ``CUDA_*`` variable is already present, skip it and emit a warning.
    """
    dump_dir = get_dump_dir()
    os.makedirs(dump_dir, exist_ok=True)

    env_vars = {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
        "CUDA_COREDUMP_GENERATION_FLAGS": _CUDA_COREDUMP_FLAGS,
        "CUDA_COREDUMP_FILE": f"{dump_dir}/cuda_coredump_%h.%p.%t",
    }
    for key, value in env_vars.items():
        if key in os.environ:
            warnings.warn(
                f"CUDA coredump env var {key} is already set to "
                f"'{os.environ[key]}', skipping injection of '{value}'.",
                stacklevel=2,
            )
        else:
            os.environ[key] = value


def cleanup_dump_dir():
    """Remove stale coredump files from the dump directory."""
    dump_dir = get_dump_dir()
    for dump_file in glob.glob(os.path.join(dump_dir, "cuda_coredump_*")):
        os.remove(dump_file)


def report():
    """Log any CUDA coredump files found after a test failure."""
    dump_dir = get_dump_dir()
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

    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        repo = os.environ.get("GITHUB_REPOSITORY", "lightseekorg/tokenspeed")
        print(f"Download from CI: gh run download {run_id} --repo {repo}")

    print(f"{separator}\n")


# Auto-inject CUDA coredump env vars at import time.
# The sentinel env var is inherited by child processes, so injection only
# happens once in the top-level process.
_SENTINEL = "_TOKENSPEED_CUDA_COREDUMP_INJECTED"

if is_enabled() and _SENTINEL not in os.environ:
    os.environ[_SENTINEL] = "1"
    print(f"Injecting CUDA coredump env vars (pid={os.getpid()})")
    _inject_env()
