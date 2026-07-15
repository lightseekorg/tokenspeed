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

from types import SimpleNamespace

import pytest
from tokenspeed_kernel.ops.communication import nccl


@pytest.mark.parametrize(
    ("is_nvidia", "is_amd", "expected"),
    [
        (True, False, "libnccl.so.2"),
        (False, True, "librccl.so.1"),
    ],
)
def test_find_nccl_library_uses_platform_default(
    monkeypatch, is_nvidia, is_amd, expected
):
    monkeypatch.setattr(
        nccl,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=is_nvidia, is_amd=is_amd),
    )

    assert nccl.find_nccl_library() == expected


def test_find_nccl_library_rejects_cpu(monkeypatch):
    monkeypatch.setattr(
        nccl,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=False, is_amd=False),
    )

    with pytest.raises(ValueError, match="only supports CUDA and ROCm"):
        nccl.find_nccl_library()


def test_nccl_library_uses_explicit_path(monkeypatch):
    loaded_paths: list[str] = []

    class _FakeFunction:
        restype = None
        argtypes = None

    class _FakeLibrary:
        def __init__(self):
            self.functions: dict[str, _FakeFunction] = {}

        def __getattr__(self, name: str) -> _FakeFunction:
            return self.functions.setdefault(name, _FakeFunction())

    def _load(path: str) -> _FakeLibrary:
        loaded_paths.append(path)
        return _FakeLibrary()

    monkeypatch.setattr(nccl.ctypes, "CDLL", _load)
    monkeypatch.setattr(
        nccl,
        "find_nccl_library",
        lambda: pytest.fail("default library resolution must not run"),
    )
    nccl.NCCLLibrary.path_to_library_cache.clear()
    nccl.NCCLLibrary.path_to_dict_mapping.clear()

    library = nccl.NCCLLibrary(so_file="/opt/nccl/lib/libnccl.so.2")

    assert loaded_paths == ["/opt/nccl/lib/libnccl.so.2"]
    assert library.lib is nccl.NCCLLibrary.path_to_library_cache[loaded_paths[0]]
