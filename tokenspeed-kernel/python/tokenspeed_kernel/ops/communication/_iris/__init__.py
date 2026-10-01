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

"""Private Iris implementation behind the public communication API.

The runtime supplies groups and tensors; Iris owns transport and storage.
The host adapter in ../iris.py manages the shared heap, peer maps, and dispatch,
and composes the workspaces in this package:

* all_reduce.py: collectives and shared device synchronization.
* attnres.py: K3 attn reduction and mixing of precomputed history partials.
* row_sharded.py: K3 attn/MoE reduce-scatter, local compute, and all-gather.

The row-sharded fusions share producer buffers and one replicated result.
Each workspace owns its completion state. Capacity is prepared before capture;
callers order producers and consumers before reusing borrowed buffers.

This module binds optional Iris imports to TokenSpeed's Triton distribution.
"""

import importlib
import pkgutil

from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

# Bind Iris's plain Triton imports to the same distribution as TokenSpeed.
with redirect_triton_to_tokenspeed_triton():
    import iris

    # Resolve lazy CCL kernel imports while the redirect is active.
    import iris.ccl.triton
    from iris.ccl import Config as _IrisConfig
    from iris.ccl.all_gather import all_gather as _iris_all_gather
    from iris.ccl.reduce_scatter import reduce_scatter as _iris_reduce_scatter

    for _info in pkgutil.walk_packages(
        iris.ccl.triton.__path__, prefix="iris.ccl.triton."
    ):
        importlib.import_module(_info.name)
