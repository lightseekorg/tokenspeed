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

"""Optional Iris imports, bound to TokenSpeed's Triton distribution."""

import importlib
import pkgutil

from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

# iris does plain ``import triton`` at module load time; route those bindings
# to the vendored ``tokenspeed_triton`` so iris and tokenspeed-kernel share a
# single triton distribution. See
# :func:`redirect_triton_to_tokenspeed_triton` for details.
with redirect_triton_to_tokenspeed_triton():
    import iris

    # Pre-import every iris kernel module that does ``import triton`` at module
    # load time (the CCL APIs above lazy-import them at call time, when the
    # redirect is no longer active).
    import iris.ccl.triton
    from iris.ccl import Config as _IrisConfig
    from iris.ccl.all_gather import all_gather as _iris_all_gather
    from iris.ccl.reduce_scatter import reduce_scatter as _iris_reduce_scatter

    for _info in pkgutil.walk_packages(
        iris.ccl.triton.__path__, prefix="iris.ccl.triton."
    ):
        importlib.import_module(_info.name)
