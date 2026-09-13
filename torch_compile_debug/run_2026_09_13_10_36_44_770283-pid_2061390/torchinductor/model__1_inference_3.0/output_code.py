# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import torch_npu
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
import torch_npu
has_initialized = False
from torch_npu._inductor import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/tv/ctvrjaj2ne3ulfnht5jmlmlfves3yh4yiwo5algmxariuluypywh.py
# Topologically Sorted Source Nodes: [float_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
# Graph fragment:
#   %arg1_1 : Tensor "bf16[1, s3][s3, 1]npu:1" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "f32[1, s3][s3, 1]npu:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   return %convert_element_type
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_unk_fused__to_copy_0 = async_compile.triton('triton_unk_fused__to_copy_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.pointwise_npu_index(
    size_hints=[128256], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'x0_numel': 'i32'}, 'device': NPUDeviceProperties(type='npu', index=1, multi_processor_count=40, cc='Ascend910B3', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__to_copy_0', 'mutated_arg_names': [], 'backend_hash': '394969db20204ff7063154b39b7e222d2c0904403002a3228e6ad67c8c1380e3', 'split_axis': [0], 'tiling_axis': [0], 'axis_names': ['x0'], 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_unk_fused__to_copy_0(in_ptr0, out_ptr0, x0_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp1, x0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        s3 = arg0_1
        buf0 = empty_strided((1, s3), (s3, 1), device='npu', dtype=torch.float32)
        # Topologically Sorted Source Nodes: [float_1], Original ATen: [aten._to_copy]
        stream1 = get_raw_stream(1)
        triton_unk_fused__to_copy_0.run(arg1_1, buf0, 128256, stream=stream1)
        del arg1_1
        # Topologically Sorted Source Nodes: [float_1, raw_logprobs], Original ATen: [aten._to_copy, aten._log_softmax]
        buf1 = torch.ops.aten._log_softmax.default(buf0, -1, False)
        del buf0
        buf2 = buf1
        assert_size_stride(buf2, (1, s3), (s3, 1), 'torch.ops.aten._log_softmax.default')
        assert_alignment(buf2, 16, 'torch.ops.aten._log_softmax.default')
        del buf1
        # Topologically Sorted Source Nodes: [unsqueeze, gather], Original ATen: [aten.unsqueeze, aten.gather]
        buf3 = torch.ops.aten.gather.default(buf2, -1, reinterpret_tensor(arg2_1, (1, 1), (1, 1), 0))
        del arg2_1
        del buf2
        buf4 = buf3
        assert_size_stride(buf4, (1, 1), (1, 1), 'torch.ops.aten.gather.default')
        assert_alignment(buf4, 16, 'torch.ops.aten.gather.default')
        return (reinterpret_tensor(buf4, (1, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 128256
    arg1_1 = rand_strided((1, 128256), (128256, 1), device='npu:1', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, ), (1, ), device='npu:1', dtype=torch.int32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
