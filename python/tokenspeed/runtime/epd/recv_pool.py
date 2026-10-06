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

"""The EPD receive pool's size, shared by the device builder and the prefill admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs


def is_epd_prefill_node(server_args: ServerArgs, is_multimodal_active: bool) -> bool:
    """Whether this node receives encode->prefill embeddings: a multimodal prefill node."""
    return server_args.disaggregation_mode == "prefill" and is_multimodal_active


def recv_pool_geometry() -> tuple[int, int]:
    """The receive pool's (slot count, slot MB); (0, 0) when the env disables it."""
    n_slots = envs.TOKENSPEED_EPD_RECV_POOL_SLOTS.get()
    slot_mb = envs.TOKENSPEED_EPD_RECV_POOL_SLOT_MB.get()
    if n_slots <= 0 or slot_mb <= 0:
        return 0, 0
    return n_slots, slot_mb


def recv_pool_bytes(server_args: ServerArgs, is_multimodal_active: bool) -> int:
    """Device bytes the receive pool takes on this node, 0 where none is built.

    The prefill admission allocates the pool after the KV cache is sized, so the
    cache budget leaves these bytes out.
    """
    if not is_epd_prefill_node(server_args, is_multimodal_active):
        return 0
    n_slots, slot_mb = recv_pool_geometry()
    return n_slots * (slot_mb << 20)
