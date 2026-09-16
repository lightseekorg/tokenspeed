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

import importlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenspeed_kernel.ops.tuning import set_autotune_max_num_tokens
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import WarmupBehavior
from tokenspeed_kernel.warmup.discovery import LoadedWarmupProfile
from tokenspeed_kernel.warmup.runner import validate_profile


def _validate_platform(loaded: LoadedWarmupProfile, platform: PlatformInfo) -> None:
    expected = loaded.profile.platform
    capability = platform.arch_version.major * 10 + platform.arch_version.minor
    if platform.vendor != expected.vendor:
        raise ValueError(
            f"warmup profile requires vendor {expected.vendor!r}, got {platform.vendor!r}"
        )
    if not (
        expected.minimum_compute_capability
        <= capability
        <= expected.maximum_compute_capability
    ):
        raise ValueError(
            "warmup profile requires compute capability in "
            f"[{expected.minimum_compute_capability}, "
            f"{expected.maximum_compute_capability}], got {capability}"
        )


def _publish_directory(temporary: Path, output: Path, force: bool) -> None:
    if not output.exists():
        os.replace(temporary, output)
        return
    if not force:
        raise FileExistsError(f"warmup bundle already exists: {output}")

    backup = output.with_name(f".{output.name}.previous-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(output, backup)
    try:
        os.replace(temporary, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def generate_bundle(
    loaded: LoadedWarmupProfile,
    output_dir: str,
    force: bool,
    platform: PlatformInfo,
    command: tuple[str, ...],
) -> Path:
    _validate_platform(loaded, platform)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"warmup bundle path is not a directory: {output}")
    if output.exists() and not force:
        raise FileExistsError(f"warmup bundle already exists: {output}")

    autotuner = importlib.import_module("flashinfer.autotuner")
    validated = validate_profile(loaded.profile)
    set_autotune_max_num_tokens(
        max(target.config.maximum_num_tokens for target in validated)
    )
    prepared = tuple(
        target.config.prepare(target.solution, platform) for target in validated
    )
    for item in prepared:
        if item.registration.warmup_behavior is not WarmupBehavior.FLASHINFER_AUTOTUNE:
            raise ValueError(
                f"selected registration {item.registration.name!r} does not use "
                "FlashInfer autotuning"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    flashinfer_path = temporary / "flashinfer.json"

    tuner = autotuner.AutoTuner.get()
    tuner.clear_cache()
    try:
        with autotuner.autotune(tune_mode=True, cache=str(flashinfer_path)):
            for item in prepared:
                item.run()
        torch.cuda.synchronize()

        if not flashinfer_path.is_file():
            raise RuntimeError("FlashInfer did not write an autotuning cache")
        with flashinfer_path.open() as file:
            native_cache = json.load(file)
        if not isinstance(native_cache, dict):
            raise RuntimeError("FlashInfer wrote a non-object autotuning cache")
        profile_keys = sorted(
            key for key in native_cache if not str(key).startswith("_")
        )
        if not profile_keys:
            raise RuntimeError("FlashInfer autotuning produced no tactic profiles")

        tuner.clear_cache()
        if not tuner.load_configs(str(flashinfer_path)):
            raise RuntimeError("FlashInfer rejected the generated autotuning cache")
        tuner.clear_cache()

        manifest = {
            "schema_version": 1,
            "complete": True,
            "source": {
                "id": loaded.profile.id,
                "sha256": loaded.sha256,
            },
            "provider": loaded.profile.provider,
            "maximum_num_tokens": max(
                target.config.maximum_num_tokens for target in validated
            ),
            "environment": native_cache.get("_metadata", {}),
            "targets": [
                {
                    "api": target.api_spec.api,
                    "solution": target.solution,
                    "registration": item.registration.name,
                }
                for target, item in zip(validated, prepared, strict=True)
            ],
            "flashinfer": {
                "path": "flashinfer.json",
                "profile_count": len(profile_keys),
                "profile_keys": profile_keys,
            },
            "command": list(command),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with (temporary / "manifest.json").open("w") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")

        _publish_directory(temporary, output, force)
    except BaseException:
        tuner.clear_cache()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output
