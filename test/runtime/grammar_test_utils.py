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

"""Real xgrammar matchers behind the engine host's grammar entries.

In serving, the Rust engine host owns the matchers (``ts-grammar``) and hands
the data plane C entries that reach them by handle (``HostGrammarEntries``).
Tests have no engine host, so this JIT-compiles the same entries over
xgrammar's Python matchers: a handle table with a hold count per handle, as
``ts_grammar::table`` keeps, whose entries call the matcher's methods through
tvm-ffi's C ABI, without the GIL, as the engine host's do.
"""

from __future__ import annotations

import threading
from typing import Any

import xgrammar as xgr

from tokenspeed.runtime.grammar.host_grammar import HostGrammar, HostGrammarEntries

_CPP_SRC = r"""
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tvm/ffi/c_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

struct Entry {
    TVMFFIObject* matcher;
    size_t holds;
};

std::mutex table_mutex;
std::unordered_map<uint64_t, Entry> table;
uint64_t next_handle = 1;

// xgrammar's GrammarMatcher methods, held for the process.
std::map<std::string, TVMFFIObjectHandle> methods;

// A reference to a handle's matcher for one call, or none for an unknown handle.
class Ref {
  public:
    explicit Ref(uint64_t handle) {
        std::lock_guard<std::mutex> lock(table_mutex);
        auto found = table.find(handle);
        if (found != table.end()) {
            matcher_ = found->second.matcher;
            TVMFFIObjectIncRef(matcher_);
        }
    }
    ~Ref() {
        if (matcher_ != nullptr) {
            TVMFFIObjectDecRef(matcher_);
        }
    }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    TVMFFIObject* get() const { return matcher_; }

  private:
    TVMFFIObject* matcher_ = nullptr;
};

TVMFFIAny ObjectArg(TVMFFIObject* object) {
    TVMFFIAny any{};
    any.type_index = object->type_index;
    any.v_obj = object;
    return any;
}

TVMFFIAny IntArg(int64_t value) {
    TVMFFIAny any{};
    any.type_index = kTVMFFIInt;
    any.v_int64 = value;
    return any;
}

TVMFFIAny BoolArg(bool value) {
    TVMFFIAny any{};
    any.type_index = kTVMFFIBool;
    any.v_int64 = value;
    return any;
}

TVMFFIAny FloatArg(double value) {
    TVMFFIAny any{};
    any.type_index = kTVMFFIFloat;
    any.v_float64 = value;
    return any;
}

TVMFFIAny NoneArg() {
    TVMFFIAny any{};
    any.type_index = kTVMFFINone;
    return any;
}

TVMFFIAny TensorArg(DLTensor* tensor) {
    TVMFFIAny any{};
    any.type_index = kTVMFFIDLTensorPtr;
    any.v_ptr = tensor;
    return any;
}

DLTensor HostTensor(size_t data, int64_t* shape, int32_t ndim, uint8_t bits) {
    DLTensor tensor{};
    tensor.data = reinterpret_cast<void*>(data);
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = ndim;
    tensor.dtype = DLDataType{kDLInt, bits, 1};
    tensor.shape = shape;
    return tensor;
}

// The method's result on the handle's matcher: 1 true, 0 false, -1 an
// unknown handle or a raised error.
int32_t Call(const char* method, uint64_t handle, TVMFFIAny* rest, int32_t num_rest) {
    Ref matcher(handle);
    if (matcher.get() == nullptr) {
        return -1;
    }
    TVMFFIAny args[8];
    args[0] = ObjectArg(matcher.get());
    for (int32_t i = 0; i < num_rest; ++i) {
        args[i + 1] = rest[i];
    }
    TVMFFIAny result{};
    result.type_index = kTVMFFINone;
    if (TVMFFIFunctionCall(methods.at(method), args, num_rest + 1, &result) != 0) {
        TVMFFIObjectHandle error = nullptr;
        TVMFFIErrorMoveFromRaised(&error);
        if (error != nullptr) {
            TVMFFIObjectDecRef(error);
        }
        return -1;
    }
    return result.type_index == kTVMFFIBool && result.v_int64 != 0 ? 1 : 0;
}

int32_t IsTerminated(uint64_t handle) { return Call("is_terminated", handle, nullptr, 0); }

int32_t AcceptToken(uint64_t handle, int32_t token) {
    TVMFFIAny rest[2] = {IntArg(token), BoolArg(false)};
    return Call("accept_token", handle, rest, 2);
}

int32_t Rollback(uint64_t handle, int32_t num_tokens) {
    TVMFFIAny rest[1] = {IntArg(num_tokens)};
    return Call("rollback", handle, rest, 1) < 0 ? -1 : 0;
}

int32_t FillBitmask(uint64_t handle, size_t data, int64_t rows, int64_t width, int32_t row) {
    int64_t shape[2] = {rows, width};
    DLTensor bitmask = HostTensor(data, shape, 2, 32);
    TVMFFIAny rest[3] = {TensorArg(&bitmask), IntArg(row), BoolArg(false)};
    return Call("fill_next_token_bitmask", handle, rest, 3) < 0 ? -1 : 0;
}

int32_t TraverseDraftTree(
    uint64_t handle, size_t next_node, size_t next_sibling, size_t draft_tokens,
    int64_t num_nodes, size_t bitmask, int64_t width) {
    int64_t column[1] = {num_nodes};
    int64_t mask_shape[2] = {num_nodes, width};
    DLTensor next = HostTensor(next_node, column, 1, 64);
    DLTensor sibling = HostTensor(next_sibling, column, 1, 64);
    DLTensor drafts = HostTensor(draft_tokens, column, 1, 64);
    DLTensor masks = HostTensor(bitmask, mask_shape, 2, 32);
    TVMFFIAny rest[6] = {TensorArg(&next), TensorArg(&sibling), TensorArg(&drafts),
                         TensorArg(&masks), FloatArg(-1.0), NoneArg()};
    return Call("traverse_draft_tree", handle, rest, 6) < 0 ? -1 : 0;
}

int32_t Retain(uint64_t handle) {
    std::lock_guard<std::mutex> lock(table_mutex);
    auto found = table.find(handle);
    if (found == table.end()) {
        return -1;
    }
    ++found->second.holds;
    return 0;
}

void Free(uint64_t handle) {
    TVMFFIObject* forgotten = nullptr;
    {
        std::lock_guard<std::mutex> lock(table_mutex);
        auto found = table.find(handle);
        if (found == table.end()) {
            return;
        }
        if (--found->second.holds == 0) {
            forgotten = found->second.matcher;
            table.erase(found);
        }
    }
    if (forgotten != nullptr) {
        TVMFFIObjectDecRef(forgotten);
    }
}

}  // namespace

// Binds xgrammar's matcher methods (name -> tvm-ffi function); the entries' addresses.
std::map<std::string, uintptr_t> Bind(const std::map<std::string, uintptr_t>& functions) {
    for (const auto& [name, function] : functions) {
        auto handle = reinterpret_cast<TVMFFIObjectHandle>(function);
        TVMFFIObjectIncRef(handle);
        methods[name] = handle;
    }
    return {
        {"is_terminated", reinterpret_cast<uintptr_t>(&IsTerminated)},
        {"accept_token", reinterpret_cast<uintptr_t>(&AcceptToken)},
        {"rollback", reinterpret_cast<uintptr_t>(&Rollback)},
        {"fill_bitmask", reinterpret_cast<uintptr_t>(&FillBitmask)},
        {"traverse_draft_tree", reinterpret_cast<uintptr_t>(&TraverseDraftTree)},
        {"retain", reinterpret_cast<uintptr_t>(&Retain)},
        {"free", reinterpret_cast<uintptr_t>(&Free)},
    };
}

// Registers a matcher (its tvm-ffi object), held once; its handle.
uint64_t Register(uintptr_t matcher) {
    auto* object = reinterpret_cast<TVMFFIObject*>(matcher);
    TVMFFIObjectIncRef(object);
    std::lock_guard<std::mutex> lock(table_mutex);
    const uint64_t handle = next_handle++;
    table.emplace(handle, Entry{object, 1});
    return handle;
}

// Whether the table still knows the handle.
bool Known(uint64_t handle) {
    std::lock_guard<std::mutex> lock(table_mutex);
    return table.count(handle) != 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bind", &Bind);
    m.def("register", &Register);
    m.def("known", &Known);
}
"""

_METHODS = (
    "is_terminated",
    "accept_token",
    "rollback",
    "fill_next_token_bitmask",
    "traverse_draft_tree",
)

_lock = threading.Lock()
_ext: Any | None = None
_entries: HostGrammarEntries | None = None


def _load() -> tuple[Any, HostGrammarEntries]:
    global _ext, _entries
    with _lock:
        if _ext is None:
            import os

            from torch.utils.cpp_extension import load_inline
            from tvm_ffi import libinfo
            from xgrammar.base import _core

            # Link the libtvm_ffi xgrammar loaded: the error state is per library.
            tvm_ffi_dir = os.path.dirname(libinfo.find_libtvm_ffi())
            ext = load_inline(
                name="tokenspeed_test_grammar_entries",
                cpp_sources=[_CPP_SRC],
                extra_cflags=["-O2"] + [f"-I{d}" for d in libinfo.include_paths()],
                extra_ldflags=[
                    f"-L{tvm_ffi_dir}",
                    "-ltvm_ffi",
                    f"-Wl,-rpath,{tvm_ffi_dir}",
                ],
                verbose=False,
            )
            functions = {
                method.name: method.func.__chandle__()
                for method in _core.GrammarMatcher.__tvm_ffi_type_info__.methods
            }
            _entries = HostGrammarEntries(
                ext.bind({name: functions[name] for name in _METHODS})
            )
            _ext = ext
        return _ext, _entries


def host_grammar(compiled: xgr.CompiledGrammar, accepted=()) -> HostGrammar:
    """A new matcher for ``compiled`` as the engine host hands it to the data
    plane: registered under a handle its request holds once, having accepted
    ``accepted``."""
    ext, entries = _load()
    matcher = xgr.GrammarMatcher(compiled)
    grammar = HostGrammar(
        entries,
        ext.register(matcher._handle.__chandle__()),
        frozenset(matcher.stop_token_ids),
    )
    for token in accepted:
        grammar.accept_token(token)
    return grammar


def handle_known(handle: int) -> bool:
    """Whether some hold still keeps ``handle``."""
    return bool(_load()[0].known(handle))
