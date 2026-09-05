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

import triton
import triton.language as tl

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.utils import get_available_gpu_memory


@triton.jit
def create_flashinfer_kv_indices_triton(
    page_table_ptr,  # [bs, max_context_len] per-token slot table, row i == batch position i
    page_kernel_lens_ptr,
    kv_indptr,
    kv_indices_ptr,
    page_table_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    kv_indices_offset = tl.load(kv_indptr + pid)
    kv_len = tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < kv_len
        data = tl.load(
            page_table_ptr + pid * page_table_stride + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


# --- Page-based memory profiling ---


def profile_available_cache_memory_bytes(
    attn_config: AttnConfig,
    gpu_id: int,
    tp_size: int,
    gpu_memory_utilization: float,
    total_gpu_memory: int,
    world_group=None,
) -> int:
    cpu_group = (
        pg_manager.get_process_group("gloo", world_group)
        if world_group is not None
        else None
    )
    available_gpu_memory = get_available_gpu_memory(
        attn_config.device,
        gpu_id,
        distributed=tp_size > 1,
        cpu_group=cpu_group,
    )
    cache_memory = available_gpu_memory - total_gpu_memory * (
        1 - gpu_memory_utilization
    )
    return int(cache_memory * (1 << 30))
