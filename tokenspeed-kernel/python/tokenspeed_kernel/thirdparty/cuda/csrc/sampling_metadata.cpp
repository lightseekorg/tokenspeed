// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <cuda.h>
#include <dlfcn.h>
#include <stdexcept>

namespace py = pybind11;

static bool floating(at::ScalarType dtype) {
  return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
}

class MetadataValidator {
  using Key = std::array<int64_t, 17>;
  std::map<Key, py::tuple> tuples_;
 public:
py::object validate_metadata(py::handle logits, py::handle bias, py::handle out,
                            py::handle values, py::handle indices,
                            int current_device, int bound_device,
                            py::handle global_offset) {
  const std::array<py::handle, 5> objects = {logits, bias, out, values, indices};
  std::array<const at::Tensor*, 5> tensors;
  for (int i = 0; i < 5; ++i) {
    if (!THPVariable_Check(objects[i].ptr())) return py::none();
    const auto& tensor = THPVariable_Unpack(objects[i].ptr());
    if (tensor.is_nested() || tensor.layout() != at::kStrided || tensor.is_neg() || tensor.is_conj())
      return py::none();
    tensors[i] = &tensor;
  }
  const auto& x = *tensors[0]; const auto& b = *tensors[1];
  const auto& o = *tensors[2]; const auto& v = *tensors[3]; const auto& ix = *tensors[4];
  if (x.dim() != 2 || b.sizes() != x.sizes()) return py::none();
  const int64_t rows = x.size(0), vocab = x.size(1), tiles = (vocab + 4095) / 4096;
  if (rows <= 0 || vocab <= 0 || vocab > 262144 || !floating(x.scalar_type()) || !floating(b.scalar_type())
      || (o.scalar_type() != at::kInt && o.scalar_type() != at::kLong)
      || o.dim() != 1 || o.size(0) != rows || !x.is_cuda()
      || x.get_device() != current_device || x.get_device() != bound_device
      || v.dim() != 2 || ix.sizes() != v.sizes() || v.size(0) < rows || v.size(1) < tiles
      || v.scalar_type() != at::kFloat || ix.scalar_type() != at::kInt
      || !v.is_contiguous() || !ix.is_contiguous()
      || x.stride(0) <= 0 || x.stride(1) <= 0 || b.stride(0) <= 0 || b.stride(1) <= 0 || o.stride(0) <= 0
      || !PyLong_CheckExact(global_offset.ptr())) return py::none();
  for (int i = 1; i < 5; ++i) if (tensors[i]->device() != x.device()) return py::none();
  int overflow = 0;
  const int64_t offset = PyLong_AsLongLongAndOverflow(global_offset.ptr(), &overflow);
  if (PyErr_Occurred()) { PyErr_Clear(); return py::none(); }
  const int64_t max_id = o.scalar_type() == at::kInt ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int64_t>::max();
  if (overflow || offset < 0 || offset > max_id - (vocab - 1)) return py::none();
  // Use wider intermediates so conservative byte intervals retain Python's
  // non-wrapping arithmetic even for unusual positive-strided descriptors.
  std::array<__int128, 5> starts, ends;
  for (int i = 0; i < 5; ++i) {
    const auto& t = *tensors[i];
    starts[i] = reinterpret_cast<uintptr_t>(t.const_data_ptr());
    __int128 elements = 1;
    for (int dim = 0; dim < t.dim(); ++dim) elements += static_cast<__int128>(t.size(dim) - 1) * t.stride(dim);
    ends[i] = starts[i] + elements * t.element_size();
  }
  for (int write = 2; write < 5; ++write)
    for (int read = 0; read < write; ++read)
      if (starts[write] < ends[read] && starts[read] < ends[write]) return py::none();
  // Cache only immutable scalar tuples after every live eligibility/alias
  // check above has passed. Never retain full addresses or Tensor objects.
  Key key = {rows, vocab, x.get_device(), static_cast<int>(x.scalar_type()),
             static_cast<int>(b.scalar_type()), static_cast<int>(o.scalar_type()),
             x.stride(0), x.stride(1), b.stride(0), b.stride(1), o.stride(0), v.stride(0),
             static_cast<int64_t>(starts[0] % 16), static_cast<int64_t>(starts[1] % 16),
             static_cast<int64_t>(starts[2] % 16), static_cast<int64_t>(starts[3] % 16),
             static_cast<int64_t>(starts[4] % 16)};
  auto found = tuples_.find(key);
  if (found != tuples_.end()) return found->second;
  py::tuple alignments(5);
  for (int i = 0; i < 5; ++i) alignments[i] = py::int_(key[12 + i]);
  auto result = py::make_tuple(rows, vocab, x.get_device(), key[3], key[4], key[5],
                              py::make_tuple(x.stride(0), x.stride(1)), py::make_tuple(b.stride(0), b.stride(1)),
                              o.stride(0), v.stride(0), alignments);
  if (tuples_.size() >= 128) tuples_.clear();
  tuples_.emplace(key, result);
  return result;
}
};

// Keep only executable handles and immutable launch dimensions. The Python
// cache owns both original CompiledKernel runners, keeping these handles live.
// Every call receives the current tensors and current stream. Two trailing
// null pointers preserve Triton 3.8.10's zero-sized scratch ABI.
class NativeLaunchPair {
  using Launch = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, unsigned,
                              CUstream, void**, void**);
  Launch launch_;
  CUfunction partial_, final_;
  unsigned partial_threads_, final_threads_, partial_shared_, final_shared_;
  static Launch resolve_launch() {
    static void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error("CUDA driver is unavailable for native launch");
    static auto function = reinterpret_cast<Launch>(dlsym(library, "cuLaunchKernel"));
    if (!function) throw std::runtime_error("CUDA driver lacks cuLaunchKernel");
    return function;
  }
  static void check(CUresult result) {
    if (result != CUDA_SUCCESS)
      throw std::runtime_error("Native bias-argmax CUDA launch failed with code " + std::to_string(result));
  }
 public:
  NativeLaunchPair(uintptr_t partial, uintptr_t final,
                   unsigned partial_warps, unsigned final_warps,
                   unsigned partial_shared, unsigned final_shared)
      : launch_(resolve_launch()), partial_(reinterpret_cast<CUfunction>(partial)),
        final_(reinterpret_cast<CUfunction>(final)), partial_threads_(partial_warps * 32),
        final_threads_(final_warps * 32), partial_shared_(partial_shared), final_shared_(final_shared) {
    if (!partial_ || !final_ || partial_warps != 4 || final_warps != 4)
      throw std::runtime_error("Unexpected bias-argmax executable geometry");
  }
  void run(py::handle logits, py::handle bias, py::handle out,
           py::handle values, py::handle indices, unsigned rows, unsigned tiles,
           uintptr_t stream) const {
    if (!rows || !tiles) throw std::runtime_error("Empty native launch geometry");
    std::array<py::handle, 5> objects = {logits, bias, out, values, indices};
    std::array<void*, 5> addresses;
    for (int i = 0; i < 5; ++i) {
      if (!THPVariable_Check(objects[i].ptr())) throw py::type_error("Expected a current Tensor");
      const auto& tensor = THPVariable_Unpack(objects[i].ptr());
      if (!tensor.is_cuda()) throw py::type_error("Expected a current CUDA Tensor");
      addresses[i] = const_cast<void*>(tensor.const_data_ptr());
    }
    void* scratch = nullptr;
    void* partial_args[] = {&addresses[0], &addresses[1], &addresses[3], &addresses[4], &scratch, &scratch};
    void* final_args[] = {&addresses[3], &addresses[4], &addresses[2], &scratch, &scratch};
    auto current_stream = reinterpret_cast<CUstream>(stream);
    // Match the vendor launcher's GIL contract. All Python/Tensor metadata
    // has been read above; only driver calls and C++ error checks run here.
    // The argument tuple keeps tensors alive, and RAII reacquires the GIL
    // before pybind11 returns or translates a CUDA failure to Python.
    {
      py::gil_scoped_release release;
      check(launch_(partial_, rows, tiles, 1, partial_threads_, 1, 1,
                    partial_shared_, current_stream, partial_args, nullptr));
      check(launch_(final_, rows, 1, 1, final_threads_, 1, 1,
                    final_shared_, current_stream, final_args, nullptr));
    }
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<MetadataValidator>(module, "MetadataValidator")
      .def(py::init<>())
      .def("validate_metadata", &MetadataValidator::validate_metadata,
           "Check all current tensors before reusing an immutable scalar tuple");
  py::class_<NativeLaunchPair>(module, "NativeLaunchPair")
      .def(py::init<uintptr_t, uintptr_t, unsigned, unsigned, unsigned, unsigned>())
      .def("run", &NativeLaunchPair::run,
           "Launch the unchanged checked two-stage kernels on this invocation's current stream");
}
