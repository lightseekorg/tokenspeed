#include "pybind.h"
namespace py = pybind11;
using namespace causalflow::petit::pybind;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<VmmSymmetricHeap::Layout>(m, "Layout", py::module_local())
        .def(py::init<>())
        .def_readwrite("barrier_record_bytes", &VmmSymmetricHeap::Layout::barrier_record_bytes)
        .def_readwrite("rank_sym_buffer_base", &VmmSymmetricHeap::Layout::rank_sym_buffer_base)
        .def_readwrite("rank_slot_bytes", &VmmSymmetricHeap::Layout::rank_slot_bytes)
        .def_readwrite("local_offset", &VmmSymmetricHeap::Layout::local_offset)
        .def_readwrite("local_bytes", &VmmSymmetricHeap::Layout::local_bytes);
    py::class_<VmmSymmetricHeap>(m, "VmmSymmetricHeap", py::module_local())
        .def(py::init<int>())
        .def("allocate", &VmmSymmetricHeap::Allocate)
        .def("local_tensor", &VmmSymmetricHeap::LocalTensor, py::keep_alive<0, 1>())
        .def_property_readonly("world_size", &VmmSymmetricHeap::world_size)
        .def_property_readonly("rank", &VmmSymmetricHeap::rank)
        .def_property_readonly("device_index", &VmmSymmetricHeap::device_index);
}
