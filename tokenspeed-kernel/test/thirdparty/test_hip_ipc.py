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

import ctypes

from tokenspeed_kernel.thirdparty.hip import hip_ipc


class _FakeFunction:
    def __init__(self, implementation):
        self._implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._implementation(*args)


def test_hip_ipc_wrapper_loads_lazily_and_preserves_abi(monkeypatch):
    raw_handle = b"Z" * 64
    closed: list[int] = []

    def get_address_range(base_ptr, size_ptr, ptr):
        assert ptr.value == 0x1010
        ctypes.cast(base_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1000
        ctypes.cast(size_ptr, ctypes.POINTER(ctypes.c_size_t))[0] = 0x400
        return 0

    def get_mem_handle(handle_ptr, ptr):
        assert ptr.value == 0x1000
        ctypes.memmove(handle_ptr, raw_handle, len(raw_handle))
        return 0

    def open_mem_handle(out_ptr, handle, flags):
        assert ctypes.string_at(ctypes.byref(handle), 64) == raw_handle
        assert flags == 1
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = 0x2000
        return 0

    def close_mem_handle(ptr):
        closed.append(ptr.value)
        return 0

    class FakeHipRuntime:
        hipMemGetAddressRange = _FakeFunction(get_address_range)
        hipIpcGetMemHandle = _FakeFunction(get_mem_handle)
        hipIpcOpenMemHandle = _FakeFunction(open_mem_handle)
        hipIpcCloseMemHandle = _FakeFunction(close_mem_handle)

    loaded: list[str] = []

    def load_library(so_file):
        loaded.append(so_file)
        return FakeHipRuntime()

    hip_ipc.get_hip_ipc_library.cache_clear()
    monkeypatch.setattr(hip_ipc.ctypes, "CDLL", load_library)
    assert loaded == []

    runtime = hip_ipc.get_hip_ipc_library()
    assert loaded == ["libamdhip64.so"]
    assert hip_ipc.get_hip_ipc_library() is runtime
    assert runtime.hipMemGetAddressRange(0x1010) == (0x1000, 0x400)
    assert runtime.hipIpcGetMemHandle(0x1000) == raw_handle
    assert runtime.hipIpcOpenMemHandle(raw_handle) == 0x2000
    runtime.hipIpcCloseMemHandle(0x2000)
    assert closed == [0x2000]

    hip_ipc.get_hip_ipc_library.cache_clear()
