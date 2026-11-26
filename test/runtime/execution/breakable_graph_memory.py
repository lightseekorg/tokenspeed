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

"""Exercise breakable-graph storage growth and real allocator sleep/wake."""

import gc
from dataclasses import dataclass

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    break_point,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


@dataclass
class Scale:
    value: float


@torch.inference_mode()
def main():
    saver = TorchMemorySaverAdapter.create(True)
    storage, captures, pool = {}, [], None
    with saver.region(tag="weights", enable_cpu_backup=True):
        weight = torch.ones(16, device="cuda")
    for rows in (8, 32):
        with saver.named_region("executor_inputs"):
            inputs = torch.ones((rows, 16), device="cuda")
        scale = Scale(2.0)

        @break_point
        def transform(value, scale):
            return value * scale.value

        def forward(inputs=inputs, scale=scale):
            return transform(inputs + weight, scale) + 3

        forward()
        torch.cuda.synchronize()
        cap = BreakableCapture(
            pool=pool, handoff_storage=storage, memory_saver_adapter=saver
        )
        with cap:
            output = forward()
        pool = cap.pool
        captures.append((cap, inputs, output, scale))
    addresses = [
        tensor.data_ptr()
        for _, inputs, output, _ in captures
        for tensor in (inputs, output)
    ] + [weight.data_ptr()]
    for cycle in range(3):
        weight.fill_(cycle + 1)
        for cap, inputs, output, scale in captures:
            inputs.fill_(cycle + 2)
            scale.value = cycle + 3.0
            cap.replay(valid_rows=inputs.shape[0])
            expected = torch.full(inputs.shape, (2 * cycle + 3) * scale.value + 3)
            assert torch.equal(output.cpu(), expected)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        saver.pause(tag="cuda_graph")
        saver.pause(tag="weights")
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ordinary = sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if tuple(segment["segment_pool_id"]) == (0, 0)
            )
            assert ordinary == 0, ordinary
        finally:
            saver.resume(tag="weights")
            saver.resume(tag="cuda_graph")
        assert addresses == [
            tensor.data_ptr()
            for _, inputs, output, _ in captures
            for tensor in (inputs, output)
        ] + [weight.data_ptr()]
        for cap, inputs, output, scale in captures:
            cap.replay(valid_rows=inputs.shape[0])
            expected = torch.full(inputs.shape, (2 * cycle + 3) * scale.value + 3)
            assert torch.equal(output.cpu(), expected)
    print(
        "PASS breakable graph: growth, live values, three sleep/wake cycles, zero ordinary allocator bytes"
    )


if __name__ == "__main__":
    main()
