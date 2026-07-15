#!/usr/bin/env python3
"""Fail closed on inherited environment configuration for release workloads.

The release acceptance jobs intentionally configure runtime behavior through
CLI arguments.  This helper detects environment variables and the default
TokenSpeed Kernel override file that could silently change those arguments or
the selected kernels.  It records names only; environment values are never
written to logs or artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "ENABLE_CP",
        "FLASHINFER_WORKSPACE_SIZE",
        "HIP_VISIBLE_DEVICES",
        "NVIDIA_TF32_OVERRIDE",
        "ROCR_VISIBLE_DEVICES",
        "TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
        "TOKENSPEED_BLOCK_NONZERO_RANK_CHILDREN",
        "TOKENSPEED_CI_SMALL_KV_SIZE",
        "TOKENSPEED_ENABLE_TORCH_INFERENCE_MODE",
        "TOKENSPEED_FLAT_DEBUG",
        "TOKENSPEED_FORCE_FAKE_FULL_NVLINK",
        "TOKENSPEED_KERNEL_OVERRIDES_FILE",
        "TOKENSPEED_KERNEL_VERBOSE",
        "TOKENSPEED_LOG_MM_TIMING",
        "TOKENSPEED_MAMBA_SSM_DTYPE",
        "TOKENSPEED_MM_SKIP_COMPUTE_HASH",
        "TOKENSPEED_NUMA_AWARE_WORKER_AFFINITY",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "TS_DEBUG_CACHE_SYNC",
        "TS_SERVE_ENGINE_MODULE",
    }
)
FORBIDDEN_PREFIXES = (
    "TOKENSPEED_KERNEL_CAPTURE_SHAPES",
    "TOKENSPEED_KERNEL_OVERRIDE_",
    "TOKENSPEED_KERNEL_PROFILE",
)
DEFAULT_OVERRIDE_RELATIVE_PATH = Path(".config/tokenspeed-kernel/overrides.yaml")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit_environment(
    env: Mapping[str, str],
    *,
    home: Path,
) -> dict[str, Any]:
    """Return a name-only audit of inherited runtime configuration.

    Presence is the contract: an explicitly empty forbidden variable still
    fails because it leaves deployment behavior ambiguous.
    """

    forbidden_keys = sorted(
        key
        for key in env
        if key in FORBIDDEN_EXACT_KEYS
        or any(key.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    )
    override_path = home / DEFAULT_OVERRIDE_RELATIVE_PATH
    override_present = override_path.exists() or override_path.is_symlink()
    failures = [
        f"forbidden environment variable is present: {key}" for key in forbidden_keys
    ]
    if override_present:
        failures.append(f"kernel override path is present: {override_path}")

    return {
        "schema_version": 1,
        "check": "runtime_environment_preflight",
        "ok": not failures,
        "forbidden_environment_keys": forbidden_keys,
        "kernel_override_file": {
            "path": str(override_path),
            "present": override_present,
        },
        "failures": failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject inherited product/debug environment configuration."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the name-only JSON audit artifact.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    args = parse_args(argv)
    inherited = os.environ if env is None else env
    home_value = inherited.get("HOME")
    home = Path(home_value) if home_value else Path.home()
    result = audit_environment(inherited, home=home)
    _write_json_atomic(args.output, result)

    if result["ok"]:
        print(f"Runtime environment preflight passed: {args.output}")
        return 0

    print("Runtime environment preflight failed:")
    for failure in result["failures"]:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
