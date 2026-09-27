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

from contextlib import contextmanager

import torch


class StreamFork:
    def __init__(self, aux_stream: torch.cuda.Stream | None):
        self.aux_stream = aux_stream
        self.fork_event = torch.cuda.Event() if aux_stream is not None else None
        self.join_event = torch.cuda.Event() if aux_stream is not None else None
        self.checkpoint_event = torch.cuda.Event() if aux_stream is not None else None
        self._active = False
        self._overlap = True
        self._current: torch.cuda.Stream | None = None

    @contextmanager
    def scope(self, *, enable: bool, overlap: bool = True):
        """Configure auxiliary-stream execution for work in this scope.

        Args:
            enable: Whether calls to ``branch()`` use the auxiliary stream.
            overlap: Whether the main stream may overlap with auxiliary-stream
                work. If false, the main stream waits after each branch.

        Yields:
            This stream fork, ready to create branches with ``branch()``.
        """
        self._active = enable and self.aux_stream is not None
        self._overlap = overlap
        if self._active:
            self._current = torch.cuda.current_stream()
            self.fork_event.record(self._current)
        try:
            yield self
        finally:
            if self._active:
                self.join()
                self._active = False
                self._overlap = True
                self._current = None

    def join(self):
        """Order the main stream after the latest branch without closing the scope."""
        if self._active:
            self.join_event.wait(self._current)

    def record_checkpoint(self):
        """Mark an intermediate point inside a branch on the auxiliary stream.

        There is one reusable checkpoint per fork. Record it before calling
        join_checkpoint in each scope; later branch work is not part of it.
        """
        if self._active:
            self.checkpoint_event.record(self.aux_stream)

    def join_checkpoint(self):
        """Order main after the recorded checkpoint, not the whole branch."""
        if self._active:
            self.checkpoint_event.wait(self._current)

    @contextmanager
    def branch_after_main(self):
        """Continue auxiliary work after everything currently enqueued on main.

        Re-record the fork event before creating the branch. Earlier branch
        waits keep their original event generation, including under capture.
        A subsequent join waits for this branch's new completion generation.
        """
        if self._active:
            self.fork_event.record(self._current)
        with self.branch():
            yield

    @contextmanager
    def branch(self):
        if not self._active:
            yield
            return
        with torch.cuda.stream(self.aux_stream):
            self.fork_event.wait(self.aux_stream)
            yield
            self.join_event.record(self.aux_stream)
        if not self._overlap:
            self.join_event.wait(self._current)
