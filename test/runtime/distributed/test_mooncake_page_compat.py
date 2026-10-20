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

import ctypes
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from tokenspeed.runtime.pd.base.mooncake_engine import MooncakeTransferEngine


class HostWriteEngine:
    """Network substitute that actually writes bytes into owned host buffers."""

    def __init__(self):
        self.batch_sizes = []
        self.fail_on = None
        self.failure = -7
        self.raise_failure = False

    def initialize(self, *args):
        return 0

    def get_rpc_port(self):
        return 4321

    def batch_transfer_sync_write(self, session, sources, destinations, lengths):
        self.batch_sizes.append(len(sources))
        if len(self.batch_sizes) == self.fail_on:
            if self.raise_failure:
                raise RuntimeError("injected transport exception")
            return self.failure
        for src, dst, size in zip(sources, destinations, lengths):
            ctypes.memmove(dst, src, size)
        return 0


class NativeHostWriteEngine(HostWriteEngine):
    def __init__(self):
        super().__init__()
        self.native_used = False

    def batch_transfer_sync_write_pages(self, session, src_pages, dst_pages, fields, n):
        self.native_used = True
        # Independent byte-copy oracle, not the adapter's expansion helper.
        for src_base, src_stride, dst_base, dst_stride, length in fields:
            for src_page, dst_page in zip(src_pages, dst_pages):
                ctypes.memmove(
                    int(dst_base) + int(dst_page) * int(dst_stride),
                    int(src_base) + int(src_page) * int(src_stride),
                    int(length),
                )
        return 0


class UnsupportedEngine(HostWriteEngine):
    batch_transfer_sync_write = None


class PageCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.src = (ctypes.c_uint8 * 4096)(*[(i * 13 + 5) % 251 for i in range(4096)])
        self.dst = (ctypes.c_uint8 * 4096)(*[0xCC] * 4096)
        self.source = ctypes.addressof(self.src)
        self.destination = ctypes.addressof(self.dst)
        self.src_pages = np.array([4, 0, 7], dtype=np.int64)
        self.dst_pages = np.array([2, 6, 1], dtype=np.int64)
        self.fields = np.array(
            [
                [self.source + 8, 32, self.destination + 16, 48, 13],
                [self.source + 1024, 64, self.destination + 2048, 72, 17],
            ],
            dtype=np.int64,
        )

    def make_engine(self, engine_class=HostWriteEngine):
        package = types.ModuleType("mooncake")
        module = types.ModuleType("mooncake.engine")
        module.TransferEngine = engine_class
        with patch.dict(sys.modules, {"mooncake": package, "mooncake.engine": module}):
            return MooncakeTransferEngine("127.0.0.1", 0)

    def transfer(self, wrapper, *, limit=4, src=None, dst=None, fields=None):
        return wrapper.batch_transfer_sync_pages(
            "peer",
            self.src_pages if src is None else src,
            self.dst_pages if dst is None else dst,
            self.fields if fields is None else fields,
            max_batch_size=limit,
        )

    def expected(self, descriptors=None):
        result = bytearray([0xCC] * len(self.dst))
        done = 0
        for src_base, src_stride, dst_base, dst_stride, length in self.fields:
            for sp, dp in zip(self.src_pages, self.dst_pages):
                if descriptors is not None and done == descriptors:
                    return bytes(result)
                start = int(src_base) - self.source + int(sp) * int(src_stride)
                target = int(dst_base) - self.destination + int(dp) * int(dst_stride)
                result[target : target + int(length)] = bytes(self.src)[
                    start : start + int(length)
                ]
                done += 1
        return bytes(result)

    def test_noncontiguous_pages_strides_tails_guards_and_source(self):
        before = bytes(self.src)
        for limit, sizes in ((1, [1] * 6), (4, [4, 2]), (6, [6]), (100, [6])):
            with self.subTest(limit=limit):
                ctypes.memset(self.destination, 0xCC, len(self.dst))
                wrapper = self.make_engine()
                self.assertEqual(self.transfer(wrapper, limit=limit), 0)
                self.assertEqual(bytes(self.dst), self.expected())
                self.assertEqual(bytes(self.src), before)
                self.assertEqual(wrapper.engine.batch_sizes, sizes)

    def test_native_page_api_is_preferred_when_both_apis_are_available(self):
        wrapper = self.make_engine(NativeHostWriteEngine)
        self.assertEqual(self.transfer(wrapper), 0)
        self.assertEqual(bytes(self.dst), self.expected())
        self.assertTrue(wrapper.engine.native_used)
        self.assertEqual(wrapper.engine.batch_sizes, [])

    def test_missing_both_transfer_apis_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "batch_transfer_sync_write"):
            self.make_engine(UnsupportedEngine)

    def test_empty_grid_submits_nothing(self):
        for empty_fields in (True, False):
            wrapper = self.make_engine()
            if empty_fields:
                result = self.transfer(wrapper, fields=np.empty((0, 5), dtype=np.int64))
            else:
                result = self.transfer(
                    wrapper,
                    src=np.array([], dtype=np.int64),
                    dst=np.array([], dtype=np.int64),
                )
            self.assertEqual(result, 0)
            self.assertEqual(wrapper.engine.batch_sizes, [])
            self.assertEqual(bytes(self.dst), bytes([0xCC] * len(self.dst)))

    def test_failure_stops_remaining_batches_and_does_not_retry(self):
        for code, raises in ((-7, False), (7, False), (-1, True)):
            with self.subTest(code=code, raises=raises):
                ctypes.memset(self.destination, 0xCC, len(self.dst))
                wrapper = self.make_engine()
                wrapper.engine.fail_on = 2
                wrapper.engine.failure = code
                wrapper.engine.raise_failure = raises
                self.assertEqual(self.transfer(wrapper, limit=2), code)
                self.assertEqual(wrapper.engine.batch_sizes, [2, 2])
                self.assertEqual(bytes(self.dst), self.expected(descriptors=2))

    def test_invalid_metadata_is_rejected_before_any_write(self):
        bad_fields = self.fields.copy()
        bad_fields[-1, -1] = -1
        overflow = self.fields.copy()
        overflow[-1, 1] = np.iinfo(np.int64).max
        invalid = [
            {"limit": 0},
            {"limit": True},
            {"limit": 1.5},
            {"src": self.src_pages.astype(np.int32)},
            {"src": np.array([1, -1, 3], dtype=np.int64)},
            {"dst": self.dst_pages[:2]},
            {"fields": self.fields[:, :4].copy()},
            {"fields": bad_fields},
            {"fields": overflow},
            {"src": np.arange(6, dtype=np.int64)[::2]},
        ]
        for args in invalid:
            with self.subTest(args=args):
                wrapper = self.make_engine()
                self.assertEqual(self.transfer(wrapper, **args), -1)
                self.assertEqual(wrapper.engine.batch_sizes, [])
                self.assertEqual(bytes(self.dst), bytes([0xCC] * len(self.dst)))


if __name__ == "__main__":
    unittest.main()
