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

"""EPD embedding (encode->prefill) disaggregation.

The encode role runs the vision tower only and ships image embeddings over the
Mooncake RDMA engine; the prefill role receives them and skips the tower. The
role-neutral transport substrate (the :class:`...base.poll.TransferPoll` status FSM, the
manager base, the bootstrap server) is shared from :mod:`...disaggregation.base`;
:mod:`conn` layers the embedding buffer args onto it and :mod:`embedding_transfer`
adds the wire frames and senders/receivers. The encode side lives in
:mod:`encode_loop` (entry point), :mod:`encode_worker`, :mod:`encode_scheduler`,
and :mod:`encode_executor`; the prefill side in :mod:`prefill_receiver`. Import
entry points from those submodules directly.
"""
