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

"""Keep CI workers inside their allocated CUDA devices."""

import argparse
import json
import os
import re
import subprocess
import sys


def gpu_count(runner: str) -> int:
    """Read the requested per-node GPU count from a resolved runner label."""
    matches = re.findall(r"(?:^|-)([1-9]\d*)gpu(?=-|$)", runner)
    if len(matches) != 1:
        raise ValueError(f"runner {runner!r} must contain one '<N>gpu' segment")
    return int(matches[0])


def parse_mask(value: str) -> list[str]:
    if value in ("", "-1"):
        return []
    ids = value.split(",")
    if any(not re.fullmatch(r"[0-9]+|GPU-[a-zA-Z0-9-]+", x) for x in ids):
        raise ValueError(f"invalid GPU mask: {value!r}")
    ids = [str(int(device)) if device.isdigit() else device for device in ids]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate devices in GPU mask: {value!r}")
    uuids = [device for device in ids if not device.isdigit()]
    for i, device in enumerate(uuids):
        if any(
            device.startswith(other) or other.startswith(device)
            for other in uuids[i + 1 :]
        ):
            raise ValueError(f"ambiguous UUID aliases in GPU mask: {value!r}")
    return ids


def container_allocation(env: dict[str, str]) -> list[str] | None:
    value = env.get("NVIDIA_VISIBLE_DEVICES")
    if value in (None, "all", "void", ""):
        return None
    if value == "none":
        return []
    devices = parse_mask(value)
    resolved = []
    for device in devices:
        if device.isdigit():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device}",
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ],
                env=env,
                text=True,
                capture_output=True,
                check=True,
                timeout=15,
            )
            device = result.stdout.strip()
            if not device.startswith("GPU-") or len(device.splitlines()) != 1:
                raise ValueError("cannot resolve NVIDIA allocation to a GPU UUID")
        resolved.append(device)
    return parse_mask(",".join(resolved))


def prepare_environment(env: dict[str, str]) -> dict[str, str]:
    result = dict(env)
    if "CUDA_VISIBLE_DEVICES" not in result:
        allocation = container_allocation(result)
        if allocation is not None:
            result["CUDA_VISIBLE_DEVICES"] = ",".join(allocation)
    return result


def resolve_gpu_groups(parent: str | None, groups: list[str]) -> list[str]:
    """Resolve disjoint role selections as ordinals within the parent mask."""
    available = None if parent is None else parse_mask(parent)
    used = set()
    result = []
    for group in groups:
        selected = []
        for device in parse_mask(group):
            if available is not None:
                if device.isdigit():
                    index = int(device)
                    if index >= len(available):
                        raise ValueError(
                            f"GPU ordinal {index} is outside CUDA_VISIBLE_DEVICES={parent!r}"
                        )
                    device = available[index]
                elif device not in available:
                    raise ValueError(
                        f"GPU {device} is outside CUDA_VISIBLE_DEVICES={parent!r}"
                    )
            if device in used:
                raise ValueError(
                    f"GPU {device} is selected more than once on this node"
                )
            used.add(device)
            selected.append(device)
        if not selected:
            raise ValueError("each worker role requires at least one visible GPU")
        result.append(",".join(selected))
    combined = parse_mask(",".join(result))
    if any(device.isdigit() for device in combined) and not all(
        device.isdigit() for device in combined
    ):
        raise ValueError("role GPU masks must use only ordinals or only UUIDs")
    return result


def diagnose(env: dict[str, str], expected_count: int) -> None:
    # This CLI runs in a child process so CUDA contexts die before model loading.
    import torch

    print(
        "[gpu-allocation] "
        + json.dumps(
            {
                key: env.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                )
            }
        ),
        flush=True,
    )
    allocation = container_allocation(env)
    expected = (
        parse_mask(env["CUDA_VISIBLE_DEVICES"])
        if "CUDA_VISIBLE_DEVICES" in env
        else None
    )
    count = torch.cuda.device_count()
    if expected is not None and count != len(expected):
        raise RuntimeError(
            f"CUDA enumerated {count} devices for {len(expected)} requested GPU identifiers"
        )
    if not count:
        raise RuntimeError("GPU task has no visible CUDA devices")
    if count != expected_count:
        raise RuntimeError(
            f"CUDA enumerated {count} devices but the runner requests {expected_count} GPUs per node"
        )
    devices = []
    seen = set()
    matched_allocation = set()
    for index in range(count):
        uuid = "GPU-" + str(torch.cuda.get_device_properties(index).uuid).removeprefix(
            "GPU-"
        )
        if uuid in seen:
            raise RuntimeError(f"CUDA enumerated GPU {uuid} more than once")
        seen.add(uuid)
        if allocation is not None:
            matches = [device for device in allocation if uuid.startswith(device)]
            if len(matches) != 1 or matches[0] in matched_allocation:
                raise RuntimeError(
                    f"CUDA device {index} ({uuid}) is outside NVIDIA_VISIBLE_DEVICES or has an ambiguous allocation"
                )
            matched_allocation.add(matches[0])
        free, total = torch.cuda.mem_get_info(index)
        devices.append(
            dict(ordinal=index, uuid=uuid, free_bytes=free, total_bytes=total)
        )
    print(
        "[gpu-allocation] " + json.dumps(dict(count=count, devices=devices)), flush=True
    )
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv"],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    print(result.stdout or result.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    groups = commands.add_parser("groups")
    groups.add_argument("groups", nargs="+")
    diagnostic = commands.add_parser("diagnose")
    diagnostic.add_argument("--runner", required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    if args.command == "diagnose":
        diagnose(env, gpu_count(args.runner))
    else:
        env = prepare_environment(env)
        resolved = resolve_gpu_groups(env.get("CUDA_VISIBLE_DEVICES"), args.groups)
        print(
            f"[gpu-allocation] parent={env.get('CUDA_VISIBLE_DEVICES')!r} roles={resolved!r}",
            file=sys.stderr,
        )
        print(" ".join(resolved))


if __name__ == "__main__":
    main()
