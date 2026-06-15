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

from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    _primary_memory_saver = torch_memory_saver.TorchMemorySaver()
except ImportError:
    pass


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def pause(self, tag: str | None = None):
        raise NotImplementedError

    def resume(self, tag: str | None = None):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    def configure_subprocess(self):
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        # tag defaults to "default" in the library; pass it through explicitly so
        # weights vs kv_cache regions can be paused/resumed independently.
        # enable_cpu_backup=True offloads (and restores) contents to CPU on
        # pause (weights); False discards them (kv_cache).
        return _primary_memory_saver.region(
            tag=tag or "default", enable_cpu_backup=enable_cpu_backup
        )

    def pause(self, tag: str | None = None):
        return _primary_memory_saver.pause(tag=tag)

    def resume(self, tag: str | None = None):
        return _primary_memory_saver.resume(tag=tag)


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str | None = None, enable_cpu_backup: bool = False):
        yield

    def pause(self, tag: str | None = None):
        pass

    def resume(self, tag: str | None = None):
        pass
