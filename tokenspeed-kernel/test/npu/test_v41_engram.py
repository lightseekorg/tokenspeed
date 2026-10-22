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

"""Real NPU host/device Engram row gathering without FP8 device tensors."""

import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.embedding.host_gather import uint8_row_gather

from tokenspeed.runtime.models.deepseek_v41_engram import RowShardedEngramEmbedding


class EngramRowsTest(unittest.TestCase):
    def test_host_and_device_rows_match_checkpoint(self):
        torch.npu.set_device(0)
        mapping = SimpleNamespace(
            attn=SimpleNamespace(tp_size=1, tp_rank=0, tp_group=(0,))
        )
        codes = ((torch.arange(8 * 256).reshape(8, 256) % 17 - 8) / 4).to(
            torch.float8_e4m3fn
        )
        scales = (torch.arange(64).reshape(8, 8) % 7 + 120).to(torch.uint8)
        ids = torch.tensor([[3, 1, -1], [6, 8, 0]], dtype=torch.int64)
        valid = (ids >= 0) & (ids < 8)
        values = codes.float() * scales.view(
            torch.float8_e8m0fnu
        ).float().repeat_interleave(32, -1)
        expected = (
            values[ids.clamp(0, 7)].masked_fill(~valid.unsqueeze(-1), 0).bfloat16()
        )
        for host in [True, False]:
            with self.subTest(host=host):
                table = RowShardedEngramEmbedding(
                    8, 256, mapping, "npu", host, "sharded" if host else "gpu", 1
                )
                table.weight_loader(table.weight, codes)
                table.weight_loader(table.scale, scales)
                actual = table(ids.npu())
                self.assertEqual(actual.device.type, "npu")
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                if not host:
                    self.assertEqual(table.weight.dtype, torch.uint8)
        empty = uint8_row_gather(
            torch.empty((0, 256), dtype=torch.uint8), ids.npu(), None
        )
        self.assertEqual(int(empty.count_nonzero().cpu()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
