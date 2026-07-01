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

"""Role-neutral transport substrate shared by the disaggregation transfer roles.

Both the KV (prefill->decode) and the embedding (encode->prefill) paths ship
bytes over the same Mooncake RDMA engine and rendezvous through the same HTTP
bootstrap protocol, so the mechanical (semantics-free) plumbing lives here once
and both roles compose it:

    :mod:`poll`       the :class:`TransferPoll` transfer-status FSM constants
    :mod:`manager`    :class:`DisaggManagerBase` -- engine + ZMQ control socket
                      + room-keyed monotonic, Failed-sticky status tracking
    :mod:`bootstrap`  :class:`DisaggBootstrapServer` -- the aiohttp rendezvous
                      server (PUT register / GET lookup + dp-group sharding)

The role-specific *semantics* (the buffer/arg shapes, the wire frames, the
senders/receivers) stay in :mod:`...kv` and :mod:`...embedding`; this layer
carries no KV/MLA/embedding fields, so the two paths still evolve independently.
Import entry points from the submodules directly.
"""
