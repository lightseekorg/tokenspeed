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

"""NPU native sparse attention versus independent CPU sink softmax."""

import unittest

import torch
import torch_npu  # noqa: F401 - registers the NPU device
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel_npu.ops.dsv41_attention import compact_attention


def oracle(q, kv, ids, sink):
    if kv.shape[0] == 0 or ids.shape[1] == 0:
        return torch.zeros_like(q)
    valid = (ids >= 0) & (ids < kv.shape[0])
    selected = kv[ids.clamp(0, kv.shape[0] - 1).long()].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * (512**-0.5)
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    scores = torch.cat([scores, sink[None, :, None].expand(q.shape[0], -1, -1)], -1)
    return torch.einsum(
        "thk,tkd->thd", scores.softmax(-1)[..., :-1], selected
    ).bfloat16()


class AttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        torch.npu.matmul.allow_hf32 = False
        torch.manual_seed(41)

    def test_leading_padding_and_interior_holes(self):
        # The real backend left-pads each SWA window to 128 entries. The vendor
        # SFA needs valid-prefix indices; isolated right-padded tests miss this.
        q = torch.randn(5, 2, 512).bfloat16()
        kv = torch.randn(5, 512).bfloat16()
        sink = torch.zeros(2, dtype=torch.float32)
        for layout in ("right", "left", "holes"):
            with self.subTest(layout=layout):
                ids = torch.full((5, 128), -1, dtype=torch.int32)
                for row in range(5):
                    live = torch.arange(row + 1, dtype=torch.int32).flip(0)
                    if layout == "right":
                        ids[row, : row + 1] = live
                    elif layout == "left":
                        ids[row, 127 - row :] = live
                    else:
                        ids[row, 1 : 2 * row + 2 : 2] = live
                expected = oracle(q, kv, ids, sink)
                actual = compact_attention(
                    q.npu(), kv.npu(), ids.npu(), sink.npu(), 512**-0.5
                ).cpu()
                torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.04)

    def test_compact_sink_and_invalid_selections(self):
        for tokens, heads, width in [(1, 1, 17), (5, 8, 64), (32, 64, 128)]:
            with self.subTest(tokens=tokens, heads=heads):
                q = torch.randn(tokens, heads, 512).bfloat16()
                kv = torch.randn(128, 512).bfloat16()
                ids = torch.stack(
                    [torch.randperm(128)[:width].sort().values for _ in range(tokens)]
                ).int()
                ids[0] = -1
                if tokens > 1:
                    ids[1, width // 2 :] = 999
                    ids[-1, :3] = 1  # Repeated selections retain multiplicity.
                sink = torch.linspace(-4, 6, heads)
                expected = oracle(q, kv, ids, sink)
                actual = compact_attention(
                    q.npu(), kv.npu(), ids.npu(), sink.npu(), 512**-0.5
                ).cpu()
                torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.04)
                self.assertEqual(int(actual[0].count_nonzero()), 0)
                for k, i in [(kv[:0], ids), (kv, ids[:, :0])]:
                    zero = compact_attention(
                        q.npu(), k.npu(), i.npu(), sink.npu(), 512**-0.5
                    )
                    self.assertEqual(int(zero.count_nonzero().cpu()), 0)

    def test_public_paged_and_compact_paths(self):
        tokens, heads = 5, 8
        q = torch.randn(tokens, heads, 512).bfloat16()
        sink = torch.linspace(-2, 4, heads)
        caches, cpu_rows = {}, {}
        for fmt in ("swa", "global"):
            width = 512
            mode = fmt + "_bf16_qdq"
            # Extra page stride exercises the shared arena's field view.
            backing = torch.zeros(
                2, 64 * width + 64, dtype=torch.bfloat16, device="npu"
            )
            cache = backing[:, : 64 * width].view(2, 64, width)
            rows = torch.randn(128, 512).bfloat16().npu()
            dsv41.cache_scatter(rows, cache, torch.arange(128, device="npu"), mode)
            cpu_rows[fmt] = dsv41.cache_gather(
                cache, torch.arange(128, device="npu"), mode, None
            ).cpu()
            caches[fmt] = cache
        slots = torch.tensor(
            [[1, 2, -1, 200], [3, 4, 5, 6], [0, 127, 0, 4], [-1] * 4, [64, 65, 66, 67]],
            dtype=torch.int32,
        )
        lens = torch.tensor([4, 2, 3, 4, 0], dtype=torch.int32)
        ids = slots.masked_fill(torch.arange(4)[None, :] >= lens[:, None], -1)
        valid = (ids >= 0) & (ids < 128)
        global_ids = (ids + 128).masked_fill(~valid, -1)
        joint_ids = torch.cat([ids.masked_fill(~valid, -1), global_ids], -1)
        rows = torch.cat([cpu_rows["swa"], cpu_rows["global"]])
        expected = oracle(q, rows, joint_ids, sink)
        out = torch.empty_like(q, device="npu")
        actual = dsv41.selected_attention(
            q.npu(),
            caches["swa"],
            slots.npu(),
            lens.npu(),
            caches["global"],
            slots.npu(),
            lens.npu(),
            sink.npu(),
            512**-0.5,
            out,
            2,
            None,
            None,
            None,
            lse=None,
            pa_mode="compact",
        )
        self.assertIs(actual, out)
        torch.testing.assert_close(actual.cpu(), expected, atol=0.025, rtol=0.04)
        compact = dsv41.selected_attention(
            q.npu(),
            caches["swa"],
            None,
            None,
            None,
            None,
            None,
            sink.npu(),
            512**-0.5,
            None,
            2,
            None,
            rows[:, None].npu(),
            joint_ids.npu(),
            lse=None,
            pa_mode="compact",
        )
        torch.testing.assert_close(compact.cpu(), expected, atol=0.025, rtol=0.04)
        with self.assertRaisesRegex(ValueError, "schedule"):
            dsv41.selected_attention(
                q.npu(),
                caches["swa"],
                None,
                None,
                None,
                None,
                None,
                sink.npu(),
                512**-0.5,
                None,
                2,
                object(),
                rows[:, None].npu(),
                joint_ids.npu(),
                lse=None,
                pa_mode="compact",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
