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

"""Contract tests for the Ascend libascendcl SDMA ctypes wrapper."""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

import pytest


def _load_ascend_memcpy_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "tokenspeed_kernel"
        / "thirdparty"
        / "ascend"
        / "memcpy.py"
    )
    spec = importlib.util.spec_from_file_location("_test_ascend_memcpy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_async_chunks_over_4096_and_sets_locations(monkeypatch):
    mod = _load_ascend_memcpy_module()
    calls = []

    def fake_batch(
        dsts,
        dest_maxs,
        srcs,
        sizes,
        num_batches,
        attrs,
        attrs_indexes,
        num_attrs,
        fail_index,
        stream,
    ):
        attr = attrs[0]
        calls.append(
            {
                "dsts": [dsts[i] for i in range(num_batches)],
                "dest_maxs": [dest_maxs[i] for i in range(num_batches)],
                "srcs": [srcs[i] for i in range(num_batches)],
                "sizes": [sizes[i] for i in range(num_batches)],
                "dst_loc": (attr.dstLoc.id, attr.dstLoc.type),
                "src_loc": (attr.srcLoc.id, attr.srcLoc.type),
                "attrs_indexes": [attrs_indexes[i] for i in range(num_attrs)],
                "stream": stream.value,
            }
        )
        return 0

    monkeypatch.setattr(mod, "_load_ascendcl", lambda: (object(), fake_batch, None))

    total = mod.ACL_MEMCPY_BATCH_MAX_COUNT * 2 + 17
    dsts = [0x100000 + i * 64 for i in range(total)]
    srcs = [0x200000 + i * 64 for i in range(total)]
    sizes = [64 + (i % 16) for i in range(total)]

    mod.ascend_memcpy_batch_async(
        dst_addrs=dsts,
        src_addrs=srcs,
        sizes=sizes,
        direction="d2h",
        device_id=3,
        stream_ptr=0xDEAD,
    )

    assert [len(c["sizes"]) for c in calls] == [
        mod.ACL_MEMCPY_BATCH_MAX_COUNT,
        mod.ACL_MEMCPY_BATCH_MAX_COUNT,
        17,
    ]
    assert [addr for c in calls for addr in c["dsts"]] == dsts
    assert [addr for c in calls for addr in c["srcs"]] == srcs
    assert [size for c in calls for size in c["sizes"]] == sizes
    assert [size for c in calls for size in c["dest_maxs"]] == sizes
    for c in calls:
        assert c["src_loc"] == (3, mod.ACL_MEM_LOCATION_TYPE_DEVICE)
        assert c["dst_loc"] == (0, mod.ACL_MEM_LOCATION_TYPE_HOST)
        assert c["attrs_indexes"] == [0]
        assert c["stream"] == 0xDEAD

    calls.clear()
    mod.ascend_memcpy_batch_async(
        dst_addrs=dsts[:5],
        src_addrs=srcs[:5],
        sizes=sizes[:5],
        direction="h2d",
        device_id=2,
        stream_ptr=0xBEEF,
    )
    assert len(calls) == 1
    assert calls[0]["src_loc"] == (0, mod.ACL_MEM_LOCATION_TYPE_HOST)
    assert calls[0]["dst_loc"] == (2, mod.ACL_MEM_LOCATION_TYPE_DEVICE)
    assert calls[0]["stream"] == 0xBEEF


def test_batch_async_propagates_error_with_fail_index(monkeypatch):
    mod = _load_ascend_memcpy_module()

    def failing_batch(
        dsts,
        dest_maxs,
        srcs,
        sizes,
        num_batches,
        attrs,
        attrs_indexes,
        num_attrs,
        fail_index,
        stream,
    ):
        del dsts, dest_maxs, srcs, sizes, num_batches, attrs, attrs_indexes
        del num_attrs, stream
        ctypes.cast(fail_index, ctypes.POINTER(ctypes.c_size_t))[0] = 7
        return 100000

    monkeypatch.setattr(mod, "_load_ascendcl", lambda: (object(), failing_batch, None))

    with pytest.raises(
        RuntimeError, match=r"aclrtMemcpyBatchAsync failed.*100000.*fail_index=7"
    ):
        mod.ascend_memcpy_batch_async(
            dst_addrs=[0x10] * 10,
            src_addrs=[0x20] * 10,
            sizes=[16] * 10,
            direction="d2h",
            device_id=0,
            stream_ptr=0x1,
        )


def test_falls_back_to_aclrt_memcpy_async_when_batch_symbol_missing(monkeypatch):
    mod = _load_ascend_memcpy_module()
    calls = []

    def fake_single(dst, dest_max, src, count, kind, stream):
        calls.append((dst.value, dest_max, src.value, count, kind, stream.value))
        return 0

    monkeypatch.setattr(mod, "_load_ascendcl", lambda: (object(), None, fake_single))

    mod.ascend_memcpy_batch_async(
        dst_addrs=[0x100, 0x200],
        src_addrs=[0x300, 0x400],
        sizes=[32, 64],
        direction="d2h",
        device_id=0,
        stream_ptr=0x55,
    )
    mod.ascend_memcpy_batch_async(
        dst_addrs=[0x500],
        src_addrs=[0x600],
        sizes=[128],
        direction="h2d",
        device_id=0,
        stream_ptr=0x66,
    )

    assert calls == [
        (0x100, 32, 0x300, 32, mod.ACL_MEMCPY_DEVICE_TO_HOST, 0x55),
        (0x200, 64, 0x400, 64, mod.ACL_MEMCPY_DEVICE_TO_HOST, 0x55),
        (0x500, 128, 0x600, 128, mod.ACL_MEMCPY_HOST_TO_DEVICE, 0x66),
    ]


def test_ascend_memcpy_validates_inputs(monkeypatch):
    mod = _load_ascend_memcpy_module()
    monkeypatch.setattr(
        mod,
        "_load_ascendcl",
        lambda: (_ for _ in ()).throw(AssertionError("empty copy must not load acl")),
    )
    mod.ascend_memcpy_batch_async(
        dst_addrs=(),
        src_addrs=(),
        sizes=(),
        direction="d2h",
        device_id=0,
        stream_ptr=0,
    )

    with pytest.raises(ValueError, match="equal length"):
        mod.ascend_memcpy_batch_async(
            dst_addrs=[1],
            src_addrs=[1, 2],
            sizes=[4],
            direction="d2h",
            device_id=0,
            stream_ptr=0,
        )
    with pytest.raises(ValueError, match="direction"):
        mod.ascend_memcpy_batch_async(
            dst_addrs=[1],
            src_addrs=[2],
            sizes=[4],
            direction="d2d",
            device_id=0,
            stream_ptr=0,
        )
    with pytest.raises(ValueError, match="device_id"):
        mod.ascend_memcpy_batch_async(
            dst_addrs=[1],
            src_addrs=[2],
            sizes=[4],
            direction="h2d",
            device_id=-1,
            stream_ptr=0,
        )
