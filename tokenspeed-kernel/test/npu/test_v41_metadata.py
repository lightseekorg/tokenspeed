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

"""Ascend metadata, compressor and rotary kernels against CPU contracts."""

import unittest

import torch
import torch_npu
from tokenspeed_kernel.ops.attention import dsv41


class MetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        torch.manual_seed(41)

    def test_request_metadata_refresh_and_paged_addresses(self):
        for count in (0, 1, 33, 257):
            base = torch.arange(count)
            p = base % 17 + 3
            req = (base // 17).int()
            p[11::31] = -1
            req[19::43] = -1
            table = torch.ones(max(1, count // 17 + 1), 16, dtype=torch.int32)
            table[:, 3::4] = 0
            table[:, 5::7] = 9

            def execute(device):
                pp, rr, tt = p.to(device), req.to(device), table.to(device)
                out = [
                    torch.empty(count, dtype=t, device=device)
                    for t in (
                        torch.bool,
                        torch.int64,
                        torch.int32,
                        torch.int64,
                        torch.int64,
                        torch.int64,
                    )
                ]
                dsv41.compressor_metadata(pp, rr, tt, 4, *out)
                write = torch.empty(count, dtype=torch.int64, device=device)
                read = torch.empty(count, 128, dtype=torch.int64, device=device)
                lens = torch.empty(count, dtype=torch.int32, device=device)
                dsv41.decode_window(pp, rr, write, read, lens, tt, 4)
                selected = torch.arange(8, device=device).expand(count, -1).contiguous()
                slots = dsv41.global_slots(selected, pp, rr, tt, 2, 4)
                st, sl = dsv41.selection_table(pp, rr, tt, 2)
                return [*out, write, read, lens, slots, st, sl]

            for step in range(2):
                with self.subTest(count=count, step=step):
                    expected, actual = execute("cpu"), execute("npu")
                    for x, y in zip(actual, expected):
                        torch.testing.assert_close(x.cpu(), y, atol=0, rtol=0)
                    p.add_(1)
                    table[:, :3] = 0

    def test_decode_rows(self):
        for width in (1, 6):

            def execute(device):
                seq = torch.tensor([5, 200, 13], dtype=torch.int32, device=device)
                pool = torch.tensor([7, 9, 2], dtype=torch.int64, device=device)
                p = torch.empty(5 * width, dtype=torch.int64, device=device)
                r = torch.empty_like(p)
                lengths = torch.empty(5, dtype=torch.int32, device=device)
                pools = torch.empty(5, dtype=torch.int64, device=device)
                dsv41.decode_rows(seq, pool, p, r, lengths, pools, 3, 1, width)
                return p, r, lengths, pools

            for actual, expected in zip(execute("npu"), execute("cpu")):
                torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    def test_decode_window_long_context_and_padding(self):
        # Real serving uses int32 read slots and a table sized for 1M context.
        p = torch.tensor([-1, 0, 63, 64, 127, 128, 51325, 1048557])
        requests = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 99])
        table = torch.arange(8 * 16385, dtype=torch.int32).reshape(8, 16385) % 1481
        table[:, 0] = 1

        def execute(device):
            pp, rr, tt = p.to(device), requests.to(device), table.to(device)
            output = (
                torch.empty(8, dtype=torch.int64, device=device),
                torch.empty(8, 128, dtype=torch.int32, device=device),
                torch.empty(8, dtype=torch.int32, device=device),
            )
            for _ in range(3):
                dsv41.decode_window(pp, rr, *output, tt, 1480)
            return output

        for actual, expected in zip(execute("npu"), execute("cpu")):
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    def test_idle_capture_then_live_metadata(self):
        # Exercise the same persistent buffers across idle capture and live rows.
        def execute(device):
            table = torch.ones(32, 524280, dtype=torch.int32, device=device)
            seq = torch.tensor([1, 51325, 1048557], dtype=torch.int32, device=device)
            pool = torch.tensor([9, 5, 17], dtype=torch.int64, device=device)
            pos = torch.empty(32, dtype=torch.int64, device=device)
            req = torch.empty_like(pos)
            lengths = torch.empty(32, dtype=torch.int32, device=device)
            pools = torch.empty_like(pos)
            plan = [
                torch.empty(32, dtype=t, device=device)
                for t in (
                    torch.bool,
                    torch.int64,
                    torch.int32,
                    torch.int64,
                    torch.int64,
                    torch.int64,
                )
            ]
            results = []
            for actual, extends in ((0, 0), (3, 1), (0, 0), (3, 0)):
                dsv41.decode_rows(
                    seq, pool, pos, req, lengths, pools, actual, extends, 1
                )
                dsv41.compressor_metadata(pos, req, table, 1480, *plan)
                results.extend(
                    x.cpu().clone() for x in [pos, req, lengths, pools, *plan]
                )
            return results

        for actual, expected in zip(execute("npu"), execute("cpu")):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_compressor_pool_and_tail(self):
        c = torch.randn(6, 512)
        s = torch.randn_like(c)
        tail = torch.randn(4, 2, 2, 512)
        prev = torch.tensor([-1, 0, -1, 2, -1, -1])
        slots = torch.tensor([2, 2, 4, 4, -1, 99])
        active = torch.tensor([True, True, True, True, False, False])
        weight = torch.randn(512).bfloat16()
        old_c = torch.zeros_like(c)
        old_s = torch.zeros_like(s)
        for i in range(6):
            if active[i] and prev[i] >= 0:
                old_c[i], old_s[i] = c[prev[i]], s[prev[i]]
            elif active[i] and 2 <= slots[i] < 8:
                old_c[i], old_s[i] = tail[slots[i] // 2, slots[i] % 2]
        maximum = torch.maximum(old_s, s)
        e0 = (old_s - maximum).exp()
        e1 = (s - maximum).exp()
        expected = (old_c * (e0 / (e0 + e1)) + c * (e1 / (e0 + e1))).masked_fill(
            ~active[:, None], 0
        )
        device_tail = tail.npu()
        raw = dsv41.compressor_pool(
            c.npu(),
            s.npu(),
            prev.npu(),
            device_tail,
            slots.npu(),
            active.npu(),
            None,
            None,
            0.0,
        )
        torch.testing.assert_close(raw.cpu(), expected, atol=2e-6, rtol=2e-6)
        actual = dsv41.compressor_pool(
            c.npu(),
            s.npu(),
            prev.npu(),
            device_tail,
            slots.npu(),
            active.npu(),
            None,
            weight.npu(),
            1e-6,
        )
        rounded = expected.bfloat16().float()
        norm = (
            rounded
            * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        ).bfloat16()
        torch.testing.assert_close(actual.cpu(), norm, atol=0.01, rtol=0.01)
        write = torch.tensor([-1, 2, 3, 7, 99, 4])
        dsv41.compressor_tail_scatter(c.npu(), s.npu(), device_tail, write.npu())
        for i, slot in enumerate(write.tolist()):
            if 0 <= slot < 8:
                tail[slot // 2, slot % 2, 0] = c[i]
                tail[slot // 2, slot % 2, 1] = s[i]
        torch.testing.assert_close(device_tail.cpu(), tail, atol=0, rtol=0)

    def test_rope_and_padded_query(self):
        for heads in (1, 8):
            q = torch.randn(5, heads, 512).bfloat16()
            p = torch.tensor([0, 3, 7, -1, 15])
            angles = torch.randn(16, 32)
            cs = torch.cat([angles.cos(), angles.sin()], -1)
            expected = q.float()
            v = q[..., -64:].float()
            c = cs[p.clamp_min(0), :32][:, None]
            s = cs[p.clamp_min(0), 32:][:, None]
            expected[..., -64::2] = v[..., ::2] * c - v[..., 1::2] * s
            expected[..., -63::2] = v[..., 1::2] * c + v[..., ::2] * s
            actual = dsv41.rope_inplace(q.npu(), p.npu(), cs.npu(), None)
            torch.testing.assert_close(
                actual.cpu(), expected.bfloat16(), atol=0, rtol=0
            )
            padded = dsv41.rope_pad_query(q.npu(), p.npu(), cs.npu(), None).cpu()
            torch.testing.assert_close(
                padded[:, :heads], expected.bfloat16(), atol=0, rtol=0
            )
            self.assertEqual(int(padded[:, heads:].count_nonzero()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
