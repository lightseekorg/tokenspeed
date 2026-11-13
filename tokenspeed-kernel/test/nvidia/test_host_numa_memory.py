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

import gc

import pytest
import torch
from tokenspeed_kernel.host_memory import create_host_allocation, import_host_allocation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="requires NVIDIA host-NUMA CUDA VMM",
)


def test_host_numa_export_import_and_alias_lifetime():
    from tokenspeed_kernel.host_memory.cuda_vmm import _probe
    from tokenspeed_kernel.ops.ple import ple_host_gather

    try:
        torch.cuda.set_device(torch.cuda.current_device())
        _probe(torch.cuda.current_device())
    except (RuntimeError, AttributeError, ImportError) as exc:
        pytest.skip(f"Host-NUMA FABRIC handles are unavailable: {exc!s}")
    creator = create_host_allocation(8192 * 64 * 4)
    peer = None
    try:
        peer = import_host_allocation(creator.export())
        owner_views = creator.map()
        peer_views = peer.map()
    finally:
        creator.close()
        if peer is not None:
            peer.close()
    host = owner_views.host.view(torch.float32).view(8192, 64)
    reference = torch.arange(8192 * 64).reshape(8192, 64).float()
    host.copy_(reference)
    gpu_view = peer_views.device.view(torch.float32).view(8192, 64)
    del creator, peer, owner_views, peer_views
    gc.collect()
    ids = torch.tensor([0, 31, 8191], dtype=torch.int64, device="cuda")
    output = torch.empty((3, 64), dtype=torch.float32, device="cuda")
    ple_host_gather(gpu_view, ids, output, 0, 8192, None, None)
    torch.testing.assert_close(output.cpu(), reference[ids.cpu()], rtol=0, atol=0)
