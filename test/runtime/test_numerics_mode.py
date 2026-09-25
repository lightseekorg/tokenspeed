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

"""The --numerics envelope folds into the individual determinism switches."""

import unittest

from tokenspeed.runtime.utils.server_args import ServerArgs


class TestNumericsMode(unittest.TestCase):
    def test_auto_keeps_performance_defaults(self):
        args = ServerArgs(model="x")
        self.assertEqual(args.numerics, "auto")
        self.assertFalse(args.force_deterministic_rsag)
        self.assertFalse(args.disable_autotune)
        self.assertFalse(args.disable_tf32)

    def test_rl_bitwise_tightens_every_switch(self):
        args = ServerArgs(model="x", numerics="rl-bitwise")
        self.assertTrue(args.force_deterministic_rsag)
        self.assertTrue(args.disable_autotune)
        self.assertTrue(args.disable_tf32)
        self.assertTrue(args.disable_pdl)
        self.assertFalse(args.enable_allreduce_fusion)
        self.assertEqual(args.comm_fusion_max_num_tokens, -1)
        self.assertEqual(args.moe_backend, "aok")
        self.assertTrue(args.batch_invariant_collectives)

    def test_rl_bitwise_keeps_an_explicit_moe_backend(self):
        args = ServerArgs(model="x", numerics="rl-bitwise", moe_backend="triton")
        self.assertEqual(args.moe_backend, "triton")

    def test_auto_keeps_the_moe_backend_auto(self):
        args = ServerArgs(model="x")
        self.assertEqual(args.moe_backend, "auto")

    def test_rl_bitwise_overrides_the_fusion_auto_enable(self):
        # resolve_communication auto-enables allreduce fusion on capable
        # topologies; the envelope must win regardless.
        args = ServerArgs(model="x", numerics="rl-bitwise", world_size=1)
        self.assertFalse(args.enable_allreduce_fusion)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rl-bitwise"):
            ServerArgs(model="x", numerics="bitwise")

    def test_ordered_fold_matches_the_sum_and_only_depends_on_rank_order(self):
        import torch

        from tokenspeed.runtime.distributed.comm_backend.auto import ordered_fold_sum

        torch.manual_seed(7)
        parts = torch.randn(8, 5, 64, dtype=torch.float32)
        out = torch.empty(5, 64, dtype=torch.bfloat16)
        ordered_fold_sum(parts, out)
        expected = parts[0].clone()
        for rank in range(1, 8):
            expected = expected + parts[rank]
        self.assertTrue(torch.equal(out, expected.to(torch.bfloat16)))
        # The same row folds to the same bits inside a larger payload: the
        # batch-invariance claim a ring all-reduce cannot make.
        wide = torch.cat((torch.randn(8, 300, 64), parts.narrow(1, 2, 1)), dim=1)
        wide_out = torch.empty(301, 64, dtype=torch.bfloat16)
        ordered_fold_sum(wide, wide_out)
        self.assertTrue(torch.equal(wide_out[300], out[2]))


if __name__ == "__main__":
    unittest.main()
