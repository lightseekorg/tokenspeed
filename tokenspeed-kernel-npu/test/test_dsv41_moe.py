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

"""NPU correctness tests; synthetic transforms are test fixtures, not calibration."""

import unittest

import torch
import torch.nn.functional as F
import torch_npu
from tokenspeed_kernel_npu.ops.dsv41_moe import (
    AscendGroupedLinear,
    AscendV41GroupedMoE,
    FlatQuantParameters,
    routed_moe_native,
)


def pack(weights):
    scale = weights.float().abs().amax(-1).clamp_min(1e-12) / 7
    quantized = (weights.float() / scale[:, :, None]).round().clamp(-7, 7).int()
    packed = torch.stack(
        [torch_npu.npu_convert_weight_to_int4pack(q.npu()) for q in quantized]
    )
    packed = torch_npu.npu_format_cast(packed, 29, customize_dtype=3)
    scale_words = (
        scale.contiguous()
        .view(torch.int32)
        .to(torch.int64)
        .view(torch.uint64)
        .unsqueeze(1)
        .npu()
    )
    return packed, scale_words, quantized, scale


class GroupedMoETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)
        torch.npu.set_device(0)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.config.allow_internal_format = True

    def test_full_w4a4_ffn_matches_quantized_reference(self):
        torch.manual_seed(43)
        e, h, i = 3, 128, 64
        w13 = torch.randn(e, 2 * i, h).bfloat16()
        w2 = torch.randn(e, h, i).bfloat16()
        packed13, scale13, q13, s13 = pack(w13)
        packed2, scale2, q2, s2 = pack(w2)
        t13 = FlatQuantParameters(
            torch.eye(16, dtype=torch.bfloat16, device="npu"),
            torch.eye(8, dtype=torch.bfloat16, device="npu"),
            0.9,
        )
        t2 = FlatQuantParameters(
            torch.eye(8, dtype=torch.bfloat16, device="npu"),
            torch.eye(8, dtype=torch.bfloat16, device="npu"),
            0.85,
        )
        moe = AscendV41GroupedMoE(
            AscendGroupedLinear("w4a4", packed13, scale13, t13, h, 2 * i),
            AscendGroupedLinear("w4a4", packed2, scale2, t2, i, h),
            10.0,
        )
        x = (torch.randn(5, h) * 4).bfloat16()
        route = torch.tensor([0.1, 0.7, 0.03, 0.5, 0.0])
        counts = torch.tensor([2, 0, 3], dtype=torch.int64)
        actual = moe(x.npu(), counts.npu(), route.npu()).float().cpu()

        def quant(a, clip):
            scale = a.float().abs().amax(-1) * clip / 7
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            return (a.float() / scale[:, None]).round().clamp(-8, 7), scale

        a, sa = quant(x, 0.9)
        expected = []
        start = 0
        for expert, count in enumerate(counts.tolist()):
            end = start + count
            gu = (
                (
                    (a[start:end] @ q13[expert].float().t())
                    * sa[start:end, None]
                    * s13[expert][None, :]
                )
                .bfloat16()
                .float()
            )
            gate, up = gu.chunk(2, -1)
            hidden = (
                F.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * route[start:end, None]
            ).bfloat16()
            a2, sa2 = quant(hidden, 0.85)
            expected.append(
                (
                    (a2 @ q2[expert].float().t()) * sa2[:, None] * s2[expert][None, :]
                ).bfloat16()
            )
            start = end
        expected = torch.cat(expected).float()
        self.assertLess(float((actual - expected).norm() / expected.norm()), 0.012)
        self.assertTrue(bool((actual[-1] == 0).all()))

    def test_bf16_clamp_and_route_weight_placement(self):
        torch.manual_seed(11)
        e, h, i = 3, 128, 64
        weights = torch.randn(e, 2 * i, h).bfloat16()
        down = torch.randn(e, h, i).bfloat16()
        gate_up = AscendGroupedLinear(
            "bf16", weights.npu().transpose(1, 2).contiguous(), None, None, h, 2 * i
        )
        output = AscendGroupedLinear(
            "bf16", down.npu().transpose(1, 2).contiguous(), None, None, i, h
        )
        moe = AscendV41GroupedMoE(gate_up, output, 10.0)
        x = torch.randn(5, h).bfloat16()
        counts = torch.tensor([0, 2, 3], dtype=torch.int64)
        route = torch.tensor([0.01, 0.2, 0.3, 0.9, 0.0], dtype=torch.float32)
        actual = moe(x.npu(), counts.npu(), route.npu()).float().cpu()
        expected = []
        start = 0
        for expert, count in enumerate(counts.tolist()):
            gu = (
                (x[start : start + count].float() @ weights[expert].float().t())
                .bfloat16()
                .float()
            )
            g, u = gu.chunk(2, -1)
            hidden = (
                F.silu(g.clamp(max=10))
                * u.clamp(-10, 10)
                * route[start : start + count, None]
            ).bfloat16()
            expected.append((hidden.float() @ down[expert].float().t()).bfloat16())
            start += count
        expected = torch.cat(expected).float()
        self.assertLess(float((actual - expected).norm() / expected.norm()), 0.008)
        self.assertTrue(bool((actual[-1] == 0).all()))
        empty = moe(x[:0].npu(), torch.zeros_like(counts).npu(), route[:0].npu())
        self.assertEqual(tuple(empty.shape), (0, h))

    def test_flatquant_real_v41_dimensions(self):
        import flash_ops

        for left, right in [(64, 80), (48, 48)]:
            for rows in [0, 5, 32]:
                with self.subTest(left=left, right=right, rows=rows):
                    torch.manual_seed(16)
                    p1 = torch.linalg.qr(torch.randn(left, left)).Q.bfloat16()
                    p2 = torch.linalg.qr(torch.randn(right, right)).Q.bfloat16()
                    x = torch.randn(rows, left, right).bfloat16()
                    clip = 0.9
                    counts = torch.tensor([0, rows, 0], dtype=torch.int64, device="npu")
                    q, s = torch.ops.custom.flat_quant(
                        x.npu(),
                        p1.npu(),
                        p2.npu(),
                        clip_ratio=clip,
                        group_list=counts,
                        group_list_type=1,
                    )
                    self.assertEqual(tuple(q.shape), (rows, left, right // 8))
                    if rows == 0:
                        continue
                    reference = (p1.float() @ (x.float() @ p2.float())).flatten(1)
                    scale = reference.abs().amax(-1) * clip / 7
                    expected = (reference / scale[:, None]).round().clamp(-8, 7)
                    words = q.cpu().to(torch.int64)
                    ints = (
                        (words.unsqueeze(-1) >> (torch.arange(8) * 4)) & 15
                    ).flatten(1)
                    ints = torch.where(ints >= 8, ints - 16, ints)
                    self.assertLessEqual(int((ints - expected).abs().max()), 1)
                    self.assertLess(float((ints != expected).float().mean()), 0.035)
                    torch.testing.assert_close(s.cpu(), scale, rtol=0.015, atol=0.002)

    def test_grouped_int4_with_flatquant_matches_independent_oracle(self):
        torch.manual_seed(21)
        e, h, n = 3, 5120, 128
        w = torch.randn(e, n, h).bfloat16()
        packed, scales, qw, sw = pack(w)
        p1 = torch.linalg.qr(torch.randn(64, 64)).Q.bfloat16()
        p2 = torch.linalg.qr(torch.randn(80, 80)).Q.bfloat16()
        transform = FlatQuantParameters(p1.npu(), p2.npu(), 0.9)
        linear = AscendGroupedLinear("w4a4", packed, scales, transform, h, n)
        x = torch.randn(5, h).bfloat16()
        groups = torch.tensor([2, 0, 3], dtype=torch.int64)
        actual = linear(x.npu(), groups.npu()).float().cpu()
        transformed = (
            p1.float() @ (x.float().reshape(5, 64, 80) @ p2.float())
        ).flatten(1)
        sa = transformed.abs().amax(-1) * 0.9 / 7
        qa = (transformed / sa[:, None]).round().clamp(-8, 7)
        expected = []
        pos = 0
        for i, c in enumerate(groups.tolist()):
            expected.append(
                (qa[pos : pos + c] @ qw[i].float().t())
                * sa[pos : pos + c, None]
                * sw[i][None, :]
            )
            pos += c
        expected = torch.cat(expected)
        self.assertLess(float((actual - expected).norm() / expected.norm()), 0.04)
        self.assertEqual(
            tuple(linear(x[:0].npu(), torch.zeros_like(groups).npu()).shape), (0, n)
        )

    def test_native_routing_matches_reference_through_graph(self):
        # The reference filters routes explicitly. The candidate must handle
        # repeated experts, remote routes and an entirely empty local shard
        # without changing graph shapes or the pre-down route-weight contract.
        for mode in ("bf16", "w4a4"):
            torch.manual_seed(20261015)
            linears = []
            for n, k, left, right in ((128, 128, 16, 8), (128, 64, 8, 8)):
                weights = (torch.randn(3, n, k) * 0.25).bfloat16()
                if mode == "bf16":
                    weight = torch_npu.npu_format_cast(weights.npu(), 29).transpose(
                        1, 2
                    )
                    linear = AscendGroupedLinear(mode, weight, None, None, k, n)
                else:
                    weight, scale, _, _ = pack(weights)
                    transform = FlatQuantParameters(
                        torch.eye(left, dtype=torch.bfloat16, device="npu"),
                        torch.eye(right, dtype=torch.bfloat16, device="npu"),
                        0.9,
                    )
                    linear = AscendGroupedLinear(mode, weight, scale, transform, k, n)
                linears.append(linear)
            moe = AscendV41GroupedMoE(*linears, 10.0)
            x = (torch.randn(8, 128, device="npu") * 4).bfloat16()
            ids = torch.randint(0, 6, (8, 4), device="npu", dtype=torch.int32)
            weights = torch.rand(8, 4, device="npu")

            def run():
                return routed_moe_native(
                    x, ids, weights, moe, expert_start=3, num_experts=6
                )

            def reference():
                local = ids.flatten().long() - 3
                slots = torch.nonzero((local >= 0) & (local < 3)).flatten()
                slots = slots[torch.argsort(local[slots].float())]
                rows = slots // 4
                counts = torch.bincount(local[slots], minlength=3).long()
                expanded = moe(x[rows], counts, weights.flatten()[slots])
                output = torch.zeros_like(x, dtype=torch.float32)
                output.index_add_(0, rows, expanded.float())
                return output.to(x.dtype)

            with torch.inference_mode():
                for _ in range(3):
                    run()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, auto_dispatch_capture=True):
                    actual = run()
                for trial in range(4):
                    with self.subTest(mode=mode, trial=trial):
                        x.copy_(torch.randn_like(x) * 4)
                        weights.copy_(torch.rand_like(weights))
                        if trial == 0:
                            ids.zero_()
                        elif trial == 1:
                            ids.fill_(4)
                        else:
                            ids.copy_(
                                torch.randint(
                                    0, 6, ids.shape, device="npu", dtype=torch.int32
                                )
                            )
                        graph.replay()
                        torch.npu.synchronize()
                        torch.testing.assert_close(actual, reference(), rtol=0, atol=0)
                empty = routed_moe_native(
                    x[:0], ids[:0], weights[:0], moe, expert_start=3, num_experts=6
                )
                self.assertEqual(empty.shape, x[:0].shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
