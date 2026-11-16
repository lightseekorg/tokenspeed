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

import unittest
from unittest.mock import Mock, call

from tokenspeed.runtime.engine.memory_occupation import memory_regions


class MemoryRegionTests(unittest.TestCase):
    def test_engine_exposes_and_uses_only_its_graph_tag(self):
        for graph_tag, nccl_suspendable, expected in (
            ("acl_graph", False, ["kv_cache", "weights", "acl_graph"]),
            ("cuda_graph", False, ["kv_cache", "weights", "cuda_graph"]),
            ("cuda_graph", True, ["kv_cache", "weights", "cuda_graph", "nccl"]),
        ):
            with self.subTest(graph_tag=graph_tag, nccl_suspendable=nccl_suspendable):
                operations = Mock()
                regions = memory_regions(
                    operations.device,
                    operations.adapter,
                    graph_tag=graph_tag,
                    nccl_suspendable=nccl_suspendable,
                    replica_memory_transitions=True,
                )
                self.assertEqual(list(regions), expected)
                regions[graph_tag].release()
                regions[graph_tag].resume()
                self.assertEqual(
                    operations.mock_calls,
                    [
                        call.device.synchronize(),
                        call.adapter.pause(tag=graph_tag),
                        call.adapter.resume(tag=graph_tag),
                    ],
                )
