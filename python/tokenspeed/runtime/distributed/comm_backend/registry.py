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

"""Global communication backend management.

Provides global singleton instances for CommBackend and TritonRSAGBackend.
"""

import threading

from tokenspeed.runtime.distributed.comm_backend.base import CommBackend

_global_backend: CommBackend | None = None
_backend_lock = threading.Lock()


def initialize_comm_backend(
    use_pynccl: bool = False,
    use_custom_allreduce: bool = False,
) -> CommBackend:
    """Create and configure the global communication backend."""
    global _global_backend

    with _backend_lock:
        if _global_backend is not None:
            return _global_backend

        from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend

        _global_backend = AutoBackend()
        _global_backend.configure(
            use_pynccl=use_pynccl,
            use_custom_allreduce=use_custom_allreduce,
        )
        return _global_backend


def get_global_backend() -> CommBackend:
    """Get the global CommBackend, creating AutoBackend if not initialized."""
    global _global_backend
    if _global_backend is None:
        initialize_comm_backend()
    return _global_backend
