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

"""Guards against the trainer weight-update NCCL group silently splitting off
this engine's own communicator.

Kept dependency-free (pure ``torch``, no ``tokenspeed`` imports) so it can be
unit tested on hosts without the compiled ``tokenspeed_kernel`` /
``tokenspeed_triton`` extensions that ``model_runner`` otherwise pulls in.
"""

from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist


@contextlib.contextmanager
def _no_default_group_split():
    """Temporarily clear the default group's ``bound_device_id``.

    Torch >= 2.4 has ``_new_process_group_helper`` build a brand-new NCCL
    communicator via ``ncclCommSplit`` off of the default process group's own
    communicator -- instead of ``ncclCommInitRank`` against the unique id from
    the rendezvous -- whenever the default group was itself initialized with a
    bound device (``init_process_group(..., device_id=...)``). For a group
    that must actually rendezvous with an external peer (the trainer's
    weight-update group), that silently yields a communicator containing only
    this engine's own ranks, so collectives on it complete locally instead of
    reaching the trainer. Clearing the binding for the duration of group
    creation forces the normal ``ncclCommInitRank`` path; the previous value
    is restored afterwards no matter how the block exits.
    """
    default_pg = None
    saved_bound_device_id = None
    if dist.is_initialized():
        default_pg = dist.distributed_c10d._get_default_group()
        bound_device_id = getattr(default_pg, "bound_device_id", None)
        if bound_device_id:
            saved_bound_device_id = bound_device_id
            default_pg.bound_device_id = None
    try:
        yield
    finally:
        if saved_bound_device_id is not None:
            default_pg.bound_device_id = saved_bound_device_id


def _assert_not_split(pg: "dist.ProcessGroup", device: "torch.device") -> None:
    """Raise if ``pg`` ended up as a split of another NCCL communicator.

    ``ProcessGroupNCCL`` creates its underlying communicator lazily, so the
    only reliable way to tell whether ``_new_process_group_helper`` took the
    ``ncclCommSplit`` path guarded against by ``_no_default_group_split``
    above is to inspect the backend's own ``options.split_from`` once the
    group object exists. When it is set, ``pg`` is a private view of another
    communicator rather than one that rendezvoused with the trainer, and
    using it would silently load garbage weights instead of failing loudly.
    """
    try:
        backend = pg._get_backend(device)
    except RuntimeError:
        return  # No backend registered for this device on this pg; can't check.
    options = getattr(backend, "options", None)
    if options is not None and getattr(options, "split_from", None) is not None:
        raise RuntimeError(
            "weight-update group would split from the engine's own NCCL "
            "communicator; refusing to join"
        )
