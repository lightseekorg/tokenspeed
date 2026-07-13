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

"""Host-function CUDA graph node.

Wraps ``cudaLaunchHostFunc`` so Python callables can be recorded as nodes
inside a CUDA graph. The C++ stub is JIT-compiled on first use via
``torch.utils.cpp_extension.load_inline``.
"""

from __future__ import annotations

import atexit
import threading
from collections.abc import Callable
from typing import Any

import torch

_ext = None
_ext_lock = threading.Lock()


_CPP_SRC = r"""
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

struct HostFuncUserData {
    bool free_user_data;  // trampoline deletes self when true
    py::function fn;
    py::tuple args;
    py::dict kwargs;

    HostFuncUserData(bool fud, py::function f, py::tuple a, py::dict k)
        : free_user_data(fud), fn(std::move(f)),
          args(std::move(a)), kwargs(std::move(k)) {}
};

static void CUDART_CB host_func_trampoline(void* user_data) {
    // cudaLaunchHostFunc runs on a driver thread without the GIL.
    py::gil_scoped_acquire gil;
    auto* data = static_cast<HostFuncUserData*>(user_data);
    try {
        data->fn(*data->args, **data->kwargs);
    } catch (const py::error_already_set& e) {
        // Can't throw across the CUDA callback boundary; log + swallow.
        PyErr_Print();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        PyErr_Print();
    }
    if (data->free_user_data) {
        delete data;
    }
    // else: caller keeps the handle and frees it explicitly.
}

// stream_ptr: CUstream as uintptr_t (torch.cuda.Stream.cuda_stream)
// free_user_data: True for non-capturing stream (safe to free after call);
//                 False during capture (callback replays from the same
//                 HostFuncUserData, so caller must keep it alive).
// Returns the user-data pointer as int when free_user_data=False, else None.
// Called with the GIL held (pybind11 default). We only release GIL for
// the actual cudaLaunchHostFunc call.
static std::optional<uintptr_t> launch_hostfunc(
    uintptr_t stream_ptr,
    bool free_user_data,
    py::function fn,
    py::args args,
    py::kwargs kwargs)
{
    auto* data = new HostFuncUserData(
        free_user_data, std::move(fn), py::tuple(args), py::dict(kwargs));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaError_t err;
    {
        py::gil_scoped_release release;
        err = cudaLaunchHostFunc(stream, host_func_trampoline, data);
    }
    if (err != cudaSuccess) {
        delete data;
        throw std::runtime_error(
            std::string("cudaLaunchHostFunc failed: ") + cudaGetErrorString(err));
    }
    if (free_user_data) {
        return std::nullopt;
    }
    return reinterpret_cast<uintptr_t>(data);
}

static void free_hostfunc_user_data(uintptr_t handle) {
    auto* data = reinterpret_cast<HostFuncUserData*>(handle);
    delete data;
}

PYBIND11_MODULE(tokenspeed_hostfunc_ext, m) {
    m.def("launch_hostfunc", &launch_hostfunc,
          "Schedule a Python callable on a CUDA stream via cudaLaunchHostFunc.");
    m.def("free_hostfunc_user_data", &free_hostfunc_user_data,
          "Free user data reserved for a captured host function.");
}
"""


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    with _ext_lock:
        if _ext is not None:
            return _ext
        import os

        from torch.utils.cpp_extension import CUDA_HOME, load_inline

        # Resolve CUDA headers explicitly — PyTorch's CUDA_HOME discovery
        # sometimes misses conda-forge's ``$PREFIX/targets/.../include`` layout.
        cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME")
        include_dirs = []
        library_dirs = []
        if cuda_home:
            inc = os.path.join(cuda_home, "include")
            lib = os.path.join(cuda_home, "lib64")
            if os.path.isdir(inc):
                include_dirs.append(inc)
            if os.path.isdir(lib):
                library_dirs.append(lib)
        for cand in (
            "/home/jue/miniforge3/targets/x86_64-linux/include",
            "/opt/conda/targets/x86_64-linux/include",
        ):
            if os.path.isdir(cand) and cand not in include_dirs:
                include_dirs.append(cand)

        _ext = load_inline(
            name="tokenspeed_hostfunc_ext",
            cpp_sources=[_CPP_SRC],
            extra_cflags=["-O2"] + [f"-I{d}" for d in include_dirs],
            extra_ldflags=[f"-L{d}" for d in library_dirs] + ["-lcudart"],
            verbose=False,
        )
        return _ext


# Handles kept alive for the lifetime of a captured graph — freeing them
# before the graph is destroyed would dangle-ref inside the trampoline.
_USER_DATA_HANDLES: set[int] = set()


def launch_hostfunc(fn: Callable, *args: Any, **kwargs: Any) -> int | None:
    """Run ``fn(*args, **kwargs)`` on the current CUDA stream.

    When capturing, returns a handle to the user-data the caller must keep
    alive; otherwise executes eagerly and returns None.
    """
    stream = torch.cuda.current_stream()
    is_capturing = torch.cuda.is_current_stream_capturing()
    ext = _load_ext()
    handle = ext.launch_hostfunc(
        stream.cuda_stream, not is_capturing, fn, *args, **kwargs
    )
    if is_capturing:
        if handle is None:
            raise RuntimeError("hostfunc launch did not return a capture handle.")
        _USER_DATA_HANDLES.add(handle)
    else:
        if handle is not None:
            raise RuntimeError("hostfunc eager launch unexpectedly returned a handle.")
    return handle


def hostfunc(fn: Callable) -> Callable:
    """Decorator: turn ``fn`` into a host-function graph node when called."""

    def wrapper(*args: Any, **kwargs: Any):
        return launch_hostfunc(fn, *args, **kwargs)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


def free_hostfunc_user_data(handle: int) -> None:
    if handle not in _USER_DATA_HANDLES:
        raise ValueError(f"hostfunc user-data handle {handle} not tracked")
    _load_ext().free_hostfunc_user_data(handle)
    _USER_DATA_HANDLES.remove(handle)


def free_all_hostfunc_user_data() -> None:
    if _ext is None:
        return
    for handle in list(_USER_DATA_HANDLES):
        _ext.free_hostfunc_user_data(handle)
    _USER_DATA_HANDLES.clear()


atexit.register(free_all_hostfunc_user_data)
