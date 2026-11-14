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

"""Host memory bookkeeping that needs no GPU; tests/nvidia covers the driver."""

from __future__ import annotations

import pytest
from tokenspeed_kernel import host_memory


def _meminfo(tmp_path, node: int, lines: list[str]) -> None:
    directory = tmp_path / f"node{node}"
    directory.mkdir()
    (directory / "meminfo").write_text(
        "".join(f"Node {node} {line}\n" for line in lines)
    )


def test_node_memory_counts_reclaimable_page_cache_as_available(tmp_path, monkeypatch):
    monkeypatch.setattr(host_memory, "_NODE_ROOT", tmp_path)
    _meminfo(
        tmp_path,
        1,
        [
            "MemTotal:       503316480 kB",
            "MemFree:          1048576 kB",
            "MemUsed:        502267904 kB",
            "Active(file):   100663296 kB",
            "Inactive(file): 201326592 kB",
            "Shmem:           52428800 kB",
            "KReclaimable:     4194304 kB",
            "SReclaimable:     3145728 kB",
            "HugePages_Total:     0",
        ],
    )
    memory = host_memory.node_memory(1)
    gib = 1 << 30
    assert memory.total == 480 * gib
    # A checkpoint just read leaves MemFree near the watermark; its clean page
    # cache and reclaimable kernel memory are available to pinned memory too.
    # tmpfs pages (Shmem) are not.
    assert memory.available == (1 + 96 + 192 + 4) * gib
    assert "293.0 GiB available" in memory.describe()


def test_node_memory_uses_slab_reclaimable_on_older_kernels(tmp_path, monkeypatch):
    monkeypatch.setattr(host_memory, "_NODE_ROOT", tmp_path)
    _meminfo(
        tmp_path,
        0,
        [
            "MemTotal:  4096 kB",
            "MemFree:   1024 kB",
            "Active(file):   512 kB",
            "Inactive(file): 256 kB",
            "SReclaimable:   128 kB",
        ],
    )
    assert host_memory.node_memory(0).available == (1024 + 512 + 256 + 128) * 1024


def test_without_a_driver_the_probe_says_why_and_creation_raises(monkeypatch):
    monkeypatch.setattr(host_memory, "_driver", lambda: None)
    host_memory.probe_host_memory.cache_clear()
    try:
        assert "NVIDIA" in host_memory.probe_host_memory(0, True)
        assert host_memory.host_numa_node(0) is None
        with pytest.raises(RuntimeError, match="NVIDIA"):
            host_memory.create_host_memory(1 << 20, (0,), 0, False, 1 << 20, 0.05)
    finally:
        host_memory.probe_host_memory.cache_clear()
