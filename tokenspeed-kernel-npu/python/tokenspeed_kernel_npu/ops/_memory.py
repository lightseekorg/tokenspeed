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

"""Explicitly owned ACL memory for persistent, borrowed tensor views."""

import weakref

import acl
import torch
import torch_npu


def _release(pointer, device):
    # Only teardown/rebinding pays this synchronization. Layer reuse is ordered
    # by the executor stream and HCCL; it never frees or synchronizes this buffer.
    with torch.npu.device(device):
        torch.npu.synchronize(device)
        status = acl.rt.free(pointer)
    if status:
        raise RuntimeError(f"acl.rt.free failed with status {status}")


class NativeBuffer:
    """Own raw HBM outside the Torch caching allocator.

    Views are borrowed for this owner's lifetime. The backend closes the owner
    only after finishing its consumers, when rebinding or releasing the cache.
    """

    def __init__(self, nbytes, device):
        self.nbytes = nbytes
        self.device = torch.device(device)
        with torch.npu.device(self.device):
            # Sparse attention reads distant rows. Prefer huge pages to avoid
            # the small-page TLB pressure of ACL_MEM_MALLOC_NORMAL_ONLY.
            pointer, status = acl.rt.malloc(nbytes, 0)  # ACL_MEM_MALLOC_HUGE_FIRST
            if status:
                raise RuntimeError(
                    f"acl.rt.malloc({nbytes}) failed with status {status}"
                )
            self._release = weakref.finalize(self, _release, pointer, self.device)
            storage = torch_npu._C._construct_storage_from_data_pointer(
                pointer, self.device, nbytes
            )
            self.bytes = torch.empty(0, device=self.device, dtype=torch.uint8).set_(
                storage, 0, (nbytes,), (1,)
            )

    def close(self):
        self.bytes = None
        self._release()
