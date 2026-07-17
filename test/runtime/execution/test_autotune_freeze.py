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

"""The kernel autotune-lifecycle switch the executor flips at end of startup."""

import unittest

from tokenspeed_kernel.ops import tuning


class TestAutotuneFreeze(unittest.TestCase):
    def tearDown(self):
        tuning._frozen = False  # global switch; restore for other tests

    def test_freeze_is_one_way_and_idempotent(self):
        self.assertFalse(tuning.autotune_frozen())
        tuning.freeze_autotuning()
        self.assertTrue(tuning.autotune_frozen())
        tuning.freeze_autotuning()
        self.assertTrue(tuning.autotune_frozen())


if __name__ == "__main__":
    unittest.main()
