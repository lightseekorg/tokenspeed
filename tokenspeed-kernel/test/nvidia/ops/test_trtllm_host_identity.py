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

"""CPU process-group checks for the TRT-LLM workspace topology guard."""

from __future__ import annotations

import json
import socket
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication import fabric


def _host_worker(rank: int, directory: str) -> None:
    root = Path(directory)
    assert not fabric.group_spans_hosts(None)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    results = []
    try:
        # Real processes on this machine must retain the single-node path.
        results.append(fabric.group_spans_hosts(None))
        read_text = Path.read_text
        cases = (
            ("same-host", "same-name"),
            ("same-host", "different-name"),
            ("different-host", "same-name"),
            ("different-host", "different-name"),
        )
        for host_mode, name_mode in cases:
            boot_file = root / f"boot-{rank if host_mode == 'different-host' else 0}"

            def read_boot(path, *args, **kwargs):
                if str(path) == "/proc/sys/kernel/random/boot_id":
                    path = boot_file
                return read_text(path, *args, **kwargs)

            # Only unavailable machine/container identities are simulated.
            # Gloo exchanges the real host hash computed from each boot file.
            with (
                patch.object(Path, "read_text", read_boot),
                patch.object(
                    socket,
                    "gethostname",
                    return_value=f"worker-{rank if name_mode == 'different-name' else 0}",
                ),
            ):
                results.append(fabric.group_spans_hosts(dist.group.WORLD))

        # A transport failure must never authorize a CUDA IPC workspace.
        with (
            patch.object(
                dist, "all_gather_object", side_effect=RuntimeError("transport failed")
            ),
            pytest.raises(RuntimeError, match="transport failed"),
        ):
            fabric.group_spans_hosts(dist.group.WORLD)
    finally:
        dist.destroy_process_group()
    assert not fabric.group_spans_hosts(None)
    (root / f"result-{rank}.json").write_text(json.dumps(results))


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU Gloo")
def test_host_identity_across_processes(tmp_path: Path) -> None:
    for rank in range(2):
        (tmp_path / f"boot-{rank}").write_text(f"physical-machine-{rank}\n")
    mp.spawn(_host_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        results = json.loads((tmp_path / f"result-{rank}.json").read_text())
        assert results == [False, False, False, True, True]
