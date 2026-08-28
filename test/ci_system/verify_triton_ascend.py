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

"""Compile and execute a minimal Triton kernel on Ascend NPU."""

from __future__ import annotations

import importlib.metadata

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from tokenspeed_kernel._triton import triton as kernel_triton
from tokenspeed_kernel.ops.kvcache.triton import zero_byte_ranges
from tokenspeed_kernel_npu._triton import triton as ascend_kernel_triton
from triton.runtime import driver


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def main() -> None:
    target = driver.active.get_current_target()
    if target.backend != "npu":
        raise RuntimeError(f"expected the Triton NPU backend, got {target!r}")
    if kernel_triton is not triton:
        raise RuntimeError("tokenspeed-kernel did not select Triton-Ascend")
    if ascend_kernel_triton is not triton:
        raise RuntimeError("tokenspeed-kernel-npu did not select Triton-Ascend")

    size = 4097
    block_size = 256
    x = torch.randn(size, device="npu", dtype=torch.float32)
    y = torch.randn(size, device="npu", dtype=torch.float32)
    output = torch.empty_like(x)
    _add_kernel[(triton.cdiv(size, block_size),)](
        x,
        y,
        output,
        size=size,
        block_size=block_size,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), (x + y).cpu(), rtol=0, atol=0)

    backing = torch.ones(4096, device="npu", dtype=torch.uint8)
    zero_byte_ranges(backing, [(128, 1000), (2048, 512)])
    torch.npu.synchronize()
    if backing[128:1128].count_nonzero() or backing[2048:2560].count_nonzero():
        raise AssertionError("TokenSpeed Triton KV-cache kernel produced bad output")

    print(
        "Triton-Ascend verification passed:",
        {
            "annotated_doc": importlib.metadata.version("annotated-doc"),
            "apache_tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
            "hf_xet": importlib.metadata.version("hf-xet"),
            "torch": torch.__version__,
            "torch_npu": importlib.metadata.version("torch-npu"),
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "shellingham": importlib.metadata.version("shellingham"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "typer": importlib.metadata.version("typer"),
            "triton": triton.__version__,
            "triton_ascend": importlib.metadata.version("triton-ascend"),
            "tokenspeed_kernel_npu": importlib.metadata.version(
                "tokenspeed-kernel-npu"
            ),
            "target": str(target),
            "device": torch.npu.get_device_name(torch.npu.current_device()),
        },
    )


if __name__ == "__main__":
    main()
