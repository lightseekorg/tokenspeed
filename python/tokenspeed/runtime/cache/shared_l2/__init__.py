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

"""Shared L2: one paged Host L2 pool that every prefill rank uses.

``owner`` serves the pool's cache index (``tokenspeed_scheduler.SharedL2Pool``)
and holds no bytes; ``provider`` hosts one node's pool memory; ``client`` is a
rank's control side, which carries the scheduler's lookups, writes and notices
to the owner; ``wire`` is the request-reply wire between them.

On a rank, ``layout`` says where its Device blocks and the pool's pages are,
``transfer`` copies between them by RDMA, ``executor`` loads pool pages into
the Device cache layer by layer in the Host cache executor's place, and
``rank`` joins all of it to the pool at device construction. See the "Shared
L2 pool" section of ``docs/configuration/server.md``.
"""
