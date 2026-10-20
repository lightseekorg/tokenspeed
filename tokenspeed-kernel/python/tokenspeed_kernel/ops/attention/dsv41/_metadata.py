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

"""Invocation-owned index geometry, never persistent request or KV state."""

import torch


class IndexGeometry:
    def __init__(self, tensors):
        self.tensors = tensors
        self.stream = None
        self.ready = None
        if tensors[0].is_cuda:
            self.stream = torch.cuda.current_stream(tensors[0].device)
            self.ready = torch.cuda.Event()
            self.ready.record(self.stream)

    def acquire(self):
        if self.stream is not None:
            stream = torch.cuda.current_stream(self.tensors[0].device)
            if stream != self.stream:
                stream.wait_event(self.ready)
                for tensor in self.tensors:
                    tensor.record_stream(stream)
        return self.tensors


class IndexMetadata:
    def __init__(self, table, lengths):
        self.source = IndexGeometry((table, lengths))
        self.paged = {}
        self.sparse = {}

    def validate(self, table, lengths):
        if self.source.tensors[0] is not table or self.source.tensors[1] is not lengths:
            raise ValueError("index metadata belongs to different table/length tensors")
        self.source.acquire()
