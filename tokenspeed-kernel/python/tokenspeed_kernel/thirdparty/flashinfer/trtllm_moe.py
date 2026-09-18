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

"""Private FlashInfer TRT-LLM launcher with initialized routing-map padding.

Keep the upstream routing, tuning and GEMM implementations. Only the native
routing workspace allocation gains a stream-ordered initialization, recorded
during capture and executed on every replay. The installed package, upstream
Python globals and upstream JIT artifacts remain untouched.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import re
import types
from dataclasses import replace
from pathlib import Path

# Match the named allocation, not a token count or an allocator callback.
_ROUTE_ALLOCATION = re.compile(
    r"(?m)^(?P<indent>[ \t]*)permuted_idx_to_token_idx\s*=\s*"
    r"alloc_tensor\(\{max_num_padded_tokens(?:\s*\+\s*1)?\},\s*"
    r"dl_int32,\s*hidden_states\.device\(\)\);"
)


def _initialize_routing_map(source: str) -> str:
    matches = list(_ROUTE_ALLOCATION.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            "Unsupported FlashInfer TRT-LLM routing workspace: expected exactly "
            "one permuted_idx_to_token_idx allocation; review the native adapter."
        )
    match = matches[0]
    indent = match["indent"]
    lines = (
        "// Initialize tile padding and any guard entry before routing writes live rows.",
        "// Graph-pool reuse can overwrite this storage: initialization must replay.",
        "CHECK_CUDA_ERROR(cudaMemsetAsync(",
        "    permuted_idx_to_token_idx.data_ptr(), 0xff,",
        "    static_cast<size_t>(permuted_idx_to_token_idx.numel()) * sizeof(int32_t),",
        "    get_stream(hidden_states.device())));",
    )
    initialization = "\n" + "\n".join(indent + line for line in lines)
    return source[: match.end()] + initialization + source[match.end() :]


def _routing_initialized_spec(*args, **kwargs):
    from filelock import FileLock
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    spec = gen_trtllm_gen_fused_moe_sm100_module(*args, **kwargs)
    launchers = [
        Path(path)
        for path in spec.sources
        if Path(path).name == "trtllm_fused_moe_kernel_launcher.cu"
    ]
    if len(launchers) != 1:
        raise RuntimeError("Unsupported FlashInfer TRT-LLM JIT source list")
    source = _initialize_routing_map(launchers[0].read_text())
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    name = f"tokenspeed_{spec.name}_route_init_{digest}"
    directory = jit_env.FLASHINFER_GEN_SRC_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    launcher = directory / launchers[0].name
    # Concurrent TP workers produce identical content; never rewrite a source
    # another worker's compiler may currently be reading.
    with FileLock(str(directory / "source.lock")):
        if not launcher.exists():
            launcher.write_text(source)
        elif launcher.read_text() != source:
            raise RuntimeError("FlashInfer routing adapter source-cache mismatch")
    return replace(
        spec,
        name=name,
        sources=[
            launcher if Path(path) == launchers[0] else path for path in spec.sources
        ],
    )


def _clone(function, namespace):
    raw = inspect.unwrap(function)
    clone = types.FunctionType(
        raw.__code__, namespace, raw.__name__, raw.__defaults__, raw.__closure__
    )
    clone.__kwdefaults__ = raw.__kwdefaults__
    clone.__annotations__ = raw.__annotations__
    clone.__qualname__ = raw.__qualname__
    return clone


def _register_private(register, name, *args, **kwargs):
    namespace, separator, operator = name.partition("::")
    if namespace != "flashinfer" or not separator:
        raise RuntimeError(f"Unexpected FlashInfer operator name: {name}")
    return register(f"tokenspeed_flashinfer_route_init::{operator}", *args, **kwargs)


@functools.cache
def _entrypoints():
    from flashinfer.fused_moe import core

    namespace = dict(vars(core))
    namespace["gen_trtllm_gen_fused_moe_sm100_module"] = _routing_initialized_spec
    for name in ("register_custom_op", "register_fake_op"):
        namespace[name] = functools.partial(_register_private, getattr(core, name))
    # Rebind the public dispatch and cached module factory, retaining the
    # upstream operator signatures, fake implementations and tactic selection.
    factory = "_get_trtllm_moe_sm100_module_impl"
    if hasattr(core, factory):
        namespace[factory] = functools.cache(_clone(getattr(core, factory), namespace))
        namespace["get_trtllm_moe_sm100_module"] = _clone(
            core.get_trtllm_moe_sm100_module, namespace
        )
    else:
        namespace["get_trtllm_moe_sm100_module"] = functools.cache(
            _clone(core.get_trtllm_moe_sm100_module, namespace)
        )
    for name in ("trtllm_fp4_block_scale_moe", "trtllm_fp4_block_scale_routed_moe"):
        namespace[name] = _clone(getattr(core, name), namespace)
    return namespace


def trtllm_fp4_block_scale_moe(*args, **kwargs):
    """Run FlashInfer FP4 MoE from logits with initialized routing padding.

    Arguments and returned tensors follow FlashInfer's same-named API.
    """
    return _entrypoints()["trtllm_fp4_block_scale_moe"](*args, **kwargs)


def trtllm_fp4_block_scale_routed_moe(*args, **kwargs):
    """Run FlashInfer FP4 MoE from precomputed routing with initialized padding.

    Arguments and returned tensors follow FlashInfer's same-named API.
    """
    return _entrypoints()["trtllm_fp4_block_scale_routed_moe"](*args, **kwargs)
