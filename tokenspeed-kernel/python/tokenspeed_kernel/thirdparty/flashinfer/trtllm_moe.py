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

Keep upstream routing decisions and GEMM implementations. The routing kernels
also write invalid rows, including tile padding and allocation guard, on every
invocation/replay. One Qwen3.8 decode shape narrows valid MoE tactics to tile 32.
The installed package, upstream Python globals and JIT artifacts remain untouched.
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
    if len(native_sources) != len(spec.sources):
        raise RuntimeError("Unsupported FlashInfer duplicate native source names")
    header_root = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/trtllm/fused_moe"
    # Include the unchanged sibling headers too: their quoted relative includes
    # must resolve to our private RoutingKernel/runner rather than installed ones.
    headers = {path.name: path for path in header_root.iterdir() if path.is_file()}
    routing_root = native_sources["trtllm_fused_moe_routing_runner.cu"].parent
    custom_header = "trtllm_fused_moe_routing_custom.cuh"
    source_headers = {custom_header: routing_root / custom_header}
    if native_sources.keys() & headers.keys() or source_headers.keys() & (
        native_sources.keys() | headers.keys()
    ):
        raise RuntimeError("Unsupported FlashInfer overlapping source/header names")
    inputs = {
        key: path.read_text()
        for key, path in (native_sources | headers | source_headers).items()
    }
    # These translation units include the shared producer next to themselves.
    # Copy them with that header so quoted includes cannot find the stock copy.
    custom_sources = {
        f"trtllm_fused_moe_routing_custom_{family}.cu"
        for family in ("block", "cluster", "cluster_large", "entry")
    }
    for key in custom_sources:
        if inputs.get(key, "").count(f'#include "{custom_header}"') != 1:
            raise RuntimeError(f"Unsupported FlashInfer custom routing include: {key}")
    routing_sources = {
        key for key, path in native_sources.items() if path.parent == routing_root
    }
    for key in routing_sources | source_headers.keys():
        if re.search(r'(?m)^#include "\.\./', inputs[key]):
            raise RuntimeError(
                f"Unsupported FlashInfer relative routing include: {key}"
            )
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
            elif (
                source != inputs[key] or key in routing_sources or key in source_headers
            ):
                target = directory / key
                if key in native_sources:
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


def _is_qwen38_decode_shape(
    *,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    nvfp4: bool,
) -> bool:
    return (
        nvfp4
        and 1 <= num_tokens <= 32
        and top_k == 10
        and num_experts == 512
        and num_local_experts == 128
        and hidden_size == 2560
        and intermediate_size == 640
    )


def _prefer_qwen38_decode_tile_32(tactics):
    """Keep valid tile-32 tactics for the matching Qwen3.8 decode shape."""
    # FlashInfer 0.7 returns tvm_ffi.Array tactics; element 0 is tile_tokens_dim.
    selected = [tactic for tactic in tactics if int(tactic[0]) == 32]
    return selected or tactics


def _require_tactic_hooks(runner_type: type) -> None:
    for name in ("get_cache_key_extras", "get_valid_tactics"):
        if not any(name in vars(base) for base in runner_type.__mro__):
            raise RuntimeError(
                f"FlashInfer MoE runner no longer defines {name}; "
                "review the private tile-32 adapter."
            )


def _require_runner_rebinding(namespace: dict) -> None:
    names = (
        "trtllm_fp4_block_scale_moe",
        "trtllm_fp4_block_scale_routed_moe",
        "get_trtllm_moe_sm100_module",
        "_get_trtllm_moe_sm100_module_impl",
    )
    for name in names:
        if name in namespace:
            function = inspect.unwrap(namespace[name])
            if (
                "TrtllmMoERunner" in function.__code__.co_names
                and function.__globals__ is namespace
            ):
                return
    raise RuntimeError(
        "FlashInfer MoE entrypoints no longer construct TrtllmMoERunner "
        "through cloned globals; review the private tile-32 adapter."
    )


@functools.cache
def _entrypoints():
    from flashinfer.fused_moe import core
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.tllm_enums import DtypeTrtllmGen

    _require_tactic_hooks(core.TrtllmMoERunner)

    class TokenSpeedFP4MoERunner(core.TrtllmMoERunner):
        """Keep low-M MoE tile selection separate from upstream tuning caches."""

        def _matches_qwen38_decode_shape(self, inputs):
            return _is_qwen38_decode_shape(
                num_tokens=MoeRunnerInputs.from_list(inputs).hidden_states.shape[0],
                top_k=self.top_k,
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                nvfp4=self.dtype_weights == DtypeTrtllmGen.E2m1,
            )

        def get_cache_key_extras(self, inputs):
            extras = super().get_cache_key_extras(inputs)
            if self._matches_qwen38_decode_shape(inputs):
                return (*extras, "tokenspeed-qwen38-tile32-v1")
            return extras

        def get_valid_tactics(self, inputs, profile):
            tactics = super().get_valid_tactics(inputs, profile)
            if self._matches_qwen38_decode_shape(inputs):
                return _prefer_qwen38_decode_tile_32(tactics)
            return tactics

    namespace = dict(vars(core))
    namespace["TrtllmMoERunner"] = TokenSpeedFP4MoERunner
    namespace["gen_trtllm_gen_fused_moe_sm100_module"] = _routing_initialized_spec
    for name in ("register_custom_op", "register_fake_op"):
        namespace[name] = functools.partial(_register_private, getattr(core, name))
    # Rebind the public dispatch and cached module factory, retaining the
    # upstream operator signatures and fake implementations.
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
    _require_runner_rebinding(namespace)
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
