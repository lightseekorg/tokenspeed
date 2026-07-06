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

"""Value types for the KV (prefill->decode) disaggregation path.

The role-specific buffer/engine args and bootstrap handshake record. The
transfer-status FSM is the shared :class:`...base.poll.TransferPoll`; the transport
mechanics are the shared :class:`...base.manager.DisaggManagerBase` /
:class:`...base.bootstrap.DisaggBootstrapServer`.
"""

from dataclasses import dataclass, field


@dataclass
class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    offsets: list[tuple[int]]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str
    gpu_id: int
    target_layer_num: int
    draft_layer_num: int
    kv_layer_ids: list[int] = field(default_factory=list)
    kv_unit_lens: list[int] = field(default_factory=list)
    state_data_ptrs: list[int] = field(default_factory=list)
    state_data_lens: list[int] = field(default_factory=list)
    state_item_lens: list[int] = field(default_factory=list)
    state_unit_lens: list[int] = field(default_factory=list)
    state_type: str = "none"
    state_layer_ids: list[int] = field(default_factory=list)
    mamba_offsets: list[int] | None = None


@dataclass
class BootstrapInfo:
    bootstrap_host: str
    bootstrap_port: int
    bootstrap_room: int
