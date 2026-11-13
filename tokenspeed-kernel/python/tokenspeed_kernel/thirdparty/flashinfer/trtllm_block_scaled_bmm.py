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

"""JIT module of the stand-alone TRT-LLM-gen block-scaled batched GEMM.

It reuses the stock TRT-LLM MoE JIT spec (flags, includes, the batched-GEMM
runner and its support sources) and replaces the fused-MoE launcher with
``trtllm_block_scaled_bmm.cu``. The GEMM kernels are FlashInfer's published
cubins, fetched through its cubin loader.
"""

from dataclasses import replace
from functools import cache
from pathlib import Path

# The stock spec's sources this module links: the runner and its support code.
_RUNNER_SOURCES = frozenset(
    {
        "trtllm_batched_gemm_runner.cu",
        "envUtils.cpp",
        "logger.cpp",
        "stringUtils.cpp",
        "tllmException.cpp",
    }
)


@cache
def load_block_scaled_bmm_module():
    """Compile and load on first use, before CUDA graph capture."""
    import torch
    from flashinfer.jit.cubin_loader import setup_cubin_loader
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    major, minor = torch.cuda.get_device_capability()
    spec = gen_trtllm_gen_fused_moe_sm100_module(enable_rubin=(major, minor) == (10, 7))
    sources = [path for path in spec.sources if Path(path).name in _RUNNER_SOURCES]
    if len(sources) != len(_RUNNER_SOURCES):
        raise RuntimeError("Unsupported FlashInfer TRT-LLM JIT source list")
    spec = replace(
        spec,
        name="tokenspeed_trtllm_block_scaled_bmm_v1",
        sources=[*sources, Path(__file__).resolve().with_suffix(".cu")],
    )
    module = spec.build_and_load()
    for path in spec.get_library_paths():
        setup_cubin_loader(str(path))
    return module
