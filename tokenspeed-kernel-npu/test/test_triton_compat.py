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

"""Ascend import boundary and fail-closed unsupported CUDA PDL."""

import unittest

import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel_npu._triton import libdevice, tl, triton


@triton.jit
def _store(x, ENABLE_PDL: tl.constexpr):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    tl.store(x, 7.0)


@triton.jit
def _exp(x, out):
    i = tl.arange(0, 32)
    tl.store(out + i, libdevice.exp(tl.load(x + i)))


class TritonCompatibilityTest(unittest.TestCase):
    def test_libdevice_uses_cann_implementation(self):
        torch.npu.set_device(0)
        x = torch.linspace(-5, 5, 32)
        out = torch.empty(32, device="npu")
        _exp[(1,)](x.npu(), out)
        torch.testing.assert_close(out.cpu(), x.exp(), atol=1e-6, rtol=1e-6)

    def test_disabled_pdl_executes_and_enabled_pdl_is_rejected(self):
        torch.npu.set_device(0)
        x = torch.zeros(1, device="npu")
        _store[(1,)](x, ENABLE_PDL=False)
        torch.npu.synchronize()
        self.assertEqual(float(x.cpu()), 7.0)
        with self.assertRaisesRegex(Exception, "CUDA PDL is not supported on Ascend"):
            _store[(1,)](x, ENABLE_PDL=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
