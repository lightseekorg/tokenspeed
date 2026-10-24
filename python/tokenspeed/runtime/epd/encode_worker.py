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

"""The request an EPD encode server's ranks take: one multimodal item to
embed and ship to the prefill ranks that wait for it.

The encode server's frontend (``AsyncLLM.submit_encode``, driven by the
gateway's encode servicer) pushes it to the encode ranks, whose host
(``ts-host`` in encode mode, ``epd/encode_host.py``) relays, batches and
ships it; this module only defines the wire struct, which the servicer
imports by this path.
"""

from __future__ import annotations

from tokenspeed.runtime.engine.io_struct import BaseReq
from tokenspeed.runtime.multimodal.inputs import MultimodalDataItem


class EncodeRequest(BaseReq, kw_only=True):
    """One encode request: its prefill peers' rendezvous and its item.

    ``bootstrap_host``/``port`` name the encode server's bootstrap server
    and ``bootstrap_room`` the room the prefill ranks receive the item's
    embedding under (one room per item, so one item per request).

    A tagged IPC struct (``BaseReq``): it rides the request channel as
    msgpack, its item's pixels as ext-encoded tensor buffers or SHM handles
    (``ts-wire``'s ``EncodeRequest`` decodes it on the host).
    """

    request_id: str
    bootstrap_host: str
    bootstrap_port: int
    bootstrap_room: int
    items: list[MultimodalDataItem]
