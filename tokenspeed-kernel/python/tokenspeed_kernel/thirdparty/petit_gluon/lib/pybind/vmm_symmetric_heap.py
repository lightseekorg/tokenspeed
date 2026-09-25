"""Standalone, host-only HIP VMM allocator; contains no device kernels."""

from functools import cache
from pathlib import Path


@cache
def runtime():
    from torch.utils.cpp_extension import ROCM_HOME, load

    root = Path(__file__).resolve().parent
    return load(
        name="gluon_petit_vmm",
        sources=[str(root / "vmm_symmetric_heap.cc"), str(root / "bindings.cc")],
        extra_include_paths=[str(Path(ROCM_HOME) / "include")],
        extra_cflags=["-O2", "-std=c++17", "-D__HIP_PLATFORM_AMD__"],
        extra_ldflags=["-L" + str(Path(ROCM_HOME) / "lib"), "-lamdhip64"],
        with_cuda=False,
    )


def create_vmm_symmetric_heap(world_size):
    return runtime().VmmSymmetricHeap(world_size)
