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

"""Private FlashInfer TRT-LLM module with producer-owned routing-map padding.

Keep upstream routing decisions, tuning and GEMM implementations. The routing
kernels also write invalid rows, including tile padding and the allocation
guard, on every invocation/replay. The installed package, upstream Python
globals and upstream JIT artifacts remain untouched.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import posixpath
import re
import types
from dataclasses import replace
from pathlib import Path

from tokenspeed_kernel.thirdparty.flashinfer._routing_padding import (
    patch_routing_sources,
)


def _relocate_header(source: str) -> str:
    # Copied siblings retain their local includes. Includes climbing out of the
    # copied directory must instead resolve from FlashInfer's installed root.
    return re.sub(
        r'(?m)^#include "(?P<path>\.\./[^"\n]+)"',
        lambda match: '#include "'
        + posixpath.normpath("flashinfer/trtllm/fused_moe/" + match["path"])
        + '"',
        source,
    )


def _prepare_routing_sources(
    sources: dict[str, str], headers: set[str]
) -> dict[str, str]:
    patched = patch_routing_sources(sources)
    for name, source in patched.items():
        if name in headers:
            source = _relocate_header(source)
        if source != sources[name]:
            # Also mark headers changed only by include relocation. Retain the
            # complete upstream copyright/license header after this notice.
            source = (
                "// Modified by TokenSpeed (LightSeek Foundation) for its routing-map\n"
                "// padding adapter: padding initialization and/or private-JIT include relocation.\n"
                "// Original copyright and license notices are retained below.\n"
                + source
            )
        patched[name] = source
    return patched


def _routing_initialized_spec(*args, **kwargs):
    from filelock import FileLock
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    spec = gen_trtllm_gen_fused_moe_sm100_module(*args, **kwargs)
    native_sources = {Path(path).name: Path(path) for path in spec.sources}
    header_root = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/trtllm/fused_moe"
    # Include the unchanged sibling headers too: their quoted relative includes
    # must resolve to our private RoutingKernel/runner rather than installed ones.
    headers = {path.name: path for path in header_root.iterdir() if path.is_file()}
    inputs = {key: path.read_text() for key, path in (native_sources | headers).items()}
    patched = _prepare_routing_sources(inputs, set(headers))
    digest = hashlib.sha256()
    for key, source in sorted(patched.items()):
        digest.update(key.encode() + b"\0" + source.encode() + b"\0")
    name = f"tokenspeed_{spec.name}_route_padding_{digest.hexdigest()[:16]}"
    directory = jit_env.FLASHINFER_GEN_SRC_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    # Concurrent TP workers produce identical content; never rewrite a source
    # another worker's compiler may currently be reading.
    with FileLock(str(directory / "source.lock")):
        private_sources = {}
        for key, source in patched.items():
            if key in headers:
                target = directory / "include/flashinfer/trtllm/fused_moe" / key
            elif source != inputs[key]:
                target = directory / key
                private_sources[key] = target
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(source)
            elif target.read_text() != source:
                raise RuntimeError("FlashInfer routing adapter source-cache mismatch")
    return replace(
        spec,
        name=name,
        sources=[private_sources.get(Path(path).name, path) for path in spec.sources],
        extra_include_dirs=[directory / "include", *(spec.extra_include_dirs or [])],
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
