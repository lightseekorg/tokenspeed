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

"""Ascend Full/Reindex selection and SWA insertion against CPU arithmetic."""

import unittest

import torch
import torch_npu  # noqa: F401 - registers the NPU device
from tokenspeed_kernel.ops.attention import dsv41


class IndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        torch.npu.matmul.allow_hf32 = False
        torch.manual_seed(41)

    def test_full_and_reindex(self):
        q = torch.randn(3, 4, 128).bfloat16().npu()
        w = torch.randn(3, 4).bfloat16().npu()
        bf16_cache = torch.empty(4, 64, 128, dtype=torch.bfloat16, device="npu")
        dsv41.store_index_bf16(
            torch.randn(256, 128).bfloat16().npu(),
            bf16_cache,
            torch.arange(256, device="npu"),
            quantize=True,
        )
        q_ref = dsv41.index_q_quantize(q, "index", None).cpu()
        keys = bf16_cache.reshape(-1, 128).cpu()
        table = torch.tensor(
            [[1, 3, -1, 2], [2, 1, 3, -1], [1, 2, 3, 0]], dtype=torch.int32
        )
        visible = torch.tensor([253, 97, 0], dtype=torch.int32)
        padded = torch.full((3, 2048), -1, dtype=torch.int32)
        padded[:, :5] = torch.tensor([[0, 2, 24, 30, -1], [1, 4, 10, 15, -1], [-1] * 5])
        holes = torch.full((3, 67), -1, dtype=torch.int32)
        holes[0, 65] = 24
        holes[1, 2] = 1
        holes[1, 64] = 10
        for candidates in [
            padded,
            holes,
            torch.full((3, 2048), -1, dtype=torch.int32),
            None,
            torch.tensor(
                [[0, 2, 24, 30, -1], [1, 4, 10, 15, -1], [-1] * 5], dtype=torch.int32
            ),
        ]:
            logical = torch.arange(256).expand(3, -1)
            pages = table[:, logical[0] // 64]
            valid = (pages > 0) & (pages < 4) & (logical < visible[:, None])
            if candidates is not None:
                valid &= ((logical // 8)[:, :, None] == candidates[:, None, :]).any(-1)
            slots = (pages.long() * 64 + logical % 64).masked_fill(~valid, -1)
            selected = keys[slots.clamp_min(0)]
            dots = torch.einsum("thd,tkd->thk", q_ref.float(), selected.float())
            if candidates is None:
                scores = (dots.bfloat16().relu() * w.cpu()[:, :, None]).sum(1).float()
            else:
                scores = (dots.relu() * w.cpu().float()[:, :, None]).sum(1)
            scores.masked_fill_(~valid, -torch.inf)
            for tile in (16, 64, 2048):
                output = dsv41.index_topk_bf16(
                    q,
                    w,
                    bf16_cache,
                    table.npu(),
                    visible.npu(),
                    None if candidates is None else candidates.npu(),
                    9,
                    3 if candidates is None else 0,
                    8,
                    2,
                    tile,
                    None,
                    None,
                    None,
                    quantize_query=True,
                )
                rows, lens, blocks, block_lens = [x.cpu() for x in output]
                for i in range(3):
                    count = min(9, int(valid[i].sum()))
                    self.assertEqual(int(lens[i]), count)
                    self.assertTrue(bool((rows[i, count:] == -1).all()))
                    chosen = rows[i, :count].long()
                    self.assertTrue(bool((chosen[1:] > chosen[:-1]).all()))
                    torch.testing.assert_close(
                        scores[i, chosen].sort().values,
                        scores[i].topk(count).values.sort().values,
                        atol=0.25,
                        rtol=0.02,
                    )
                    if candidates is None:
                        bs = scores[i].reshape(32, 8).amax(-1)
                        good = torch.where(bs > -torch.inf)[0]
                        count = min(3, good.numel())
                        self.assertEqual(int(block_lens[i]), count)
                        picked = blocks[i, :count].long()
                        if count:
                            self.assertIn(int(good[-1]), picked.tolist())
                            self.assertTrue(bool((picked[1:] > picked[:-1]).all()))
                            rest = picked[picked != good[-1]]
                            bs[good[-1]] = -torch.inf
                            torch.testing.assert_close(
                                bs[rest].sort().values,
                                bs.topk(count - 1).values.sort().values,
                                atol=0.25,
                                rtol=0.02,
                            )

    def test_swa_rope_store_and_returned_rows(self):
        x = torch.randn(4, 512, dtype=torch.bfloat16, device="npu")
        original = x.clone()
        p = torch.tensor([1, 7, -1, 3], device="npu")
        angles = torch.randn(8, 32, device="npu")
        cs = torch.cat([angles.cos(), angles.sin()], -1)
        cache = torch.zeros(2, 64, 512, dtype=torch.bfloat16, device="npu")
        slots = torch.tensor([64, 66, -1, 300], device="npu")
        out = torch.empty_like(x)
        rotated = dsv41.rope_inplace(x.clone(), p, cs, None)
        expected = dsv41.cache_unpack(
            dsv41.cache_pack(rotated, "swa", None), "swa", None
        )
        returned = dsv41.swa_rope_scatter(
            x, p, cs, cache, slots, "swa_bf16_qdq", out, None, 1e-6
        )
        self.assertIs(returned, out)
        torch.testing.assert_close(out.cpu(), expected.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(x.cpu(), original.cpu(), atol=0, rtol=0)
        got = dsv41.cache_gather(cache, slots[:2], "swa_bf16_qdq", None)
        torch.testing.assert_close(got.cpu(), expected[:2].cpu(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class SelectAllTest(unittest.TestCase):
    def test_live_graph_matches_cpu_with_page_holes(self):
        torch.npu.set_device(0)
        for dtype, columns in ((torch.int32, 0), (torch.int32, 8), (torch.int64, 8)):
            table = torch.ones((3, columns), dtype=dtype, device="npu")
            visible = torch.tensor([0, 31, 512], dtype=torch.int32, device="npu")
            for capacity in (0, 64, 2048):

                def buffers(device):
                    return tuple(
                        torch.empty(shape, device=device, dtype=torch.int32)
                        for shape in ((3, 512), (3,), (3, capacity), (3,))
                    )

                out = buffers("npu")

                def run():
                    return dsv41.index_select_all(table, visible, 9, 8, out)

                run()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    run()
                for shift in (0, 1):
                    if columns:
                        table.copy_((torch.arange(24).reshape(3, 8) + shift) % 12 - 1)
                    visible.copy_(torch.tensor([512, 77, 0] if shift else [0, 31, 512]))
                    graph.replay()
                    torch.npu.synchronize()
                    expected = dsv41.index_select_all(
                        table.cpu(), visible.cpu(), 9, 8, buffers("cpu")
                    )
                    for actual, reference in zip(out, expected):
                        torch.testing.assert_close(
                            actual.cpu(), reference, atol=0, rtol=0
                        )
