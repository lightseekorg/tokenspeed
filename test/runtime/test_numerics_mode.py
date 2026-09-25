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


if __name__ == "__main__":
    unittest.main()
