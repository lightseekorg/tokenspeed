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

"""HCCL communication backend for Ascend NPU."""

from __future__ import annotations

from tokenspeed.runtime.distributed.comm_backend.base import Group
from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend


class HcclBackend(NcclBackend):
    """Backend using torch.distributed HCCL process groups on Ascend NPU."""

    def _get_or_create_resources(self, group: Group):
        if group in self._resources:
            return self._resources[group]

        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        world_size = len(group)
        device_group = None
        if world_size > 1:
            device_group = pg_manager.get_process_group("hccl", group)

        self._resources[group] = {
            "pynccl_comm": None,
            "device_group": device_group,
            "world_size": world_size,
        }
        return self._resources[group]
