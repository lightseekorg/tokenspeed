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

"""Shared L2: one paged Host L2 pool that every rank joined to it uses.

``owner`` serves the pool's cache index (the scheduler's ``SharedL2Pool``)
and holds no bytes; ``provider`` hosts one node's pool memory; ``wire`` is the
request-reply wire to the owner.

On a rank, ``client`` holds the rank's config and its hello,
``layout`` says where its Device blocks are and how a pool page holds one,
``executor`` is the forward thread's part of a load in the Host cache
executor's place, and ``rank`` joins all of it to the pool at device
construction. The rank's client of the owner (lookups, writes, notices) and
the RDMA copies of its loads and writes run in the engine host
(``shared_l2`` of ``ts-control``). See the "Shared L2 pool" section of
``docs/configuration/server.md``.
"""
