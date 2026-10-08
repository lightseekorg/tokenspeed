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

"""Private FlashInfer TRT-LLM BF16 MoE launcher for 64-aligned gated sizes.

FlashInfer's ``Bf16MoeLauncher::check_moe`` requires an intermediate size that
is a multiple of 128. Its BF16 cubins tile GEMM1's rows (twice the intermediate
size for gated activations) by 128 and read GEMM2's K (the intermediate size)
in 128-byte blocks of 64 elements, so gated activations only need a multiple
of 64; non-gated activations still need 128. This adapter compiles a copy of
the installed launcher whose BF16 check states exactly that, and exposes the
BF16 entry points through a cloned namespace with private operator names.
Routing, tuning and GEMM code stay FlashInfer's. The installed package,
upstream Python globals and stock JIT artifacts are unchanged.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import re
import shutil

from tokenspeed_kernel.thirdparty.flashinfer.trtllm_moe import (
    _clone,
    _patched_launcher_spec,
    _register_private,
)

logger = logging.getLogger(__name__)

# Intermediate-size multiples the stock launcher requires and the private one
# requires for gated activations.
STOCK_ISPP_ALIGNMENT = 128
GATED_ISPP_ALIGNMENT = 64

_LAUNCHER = "trtllm_fused_moe_kernel_launcher.cu"
_OPERATORS = "tokenspeed_flashinfer_bf16_ispp64"
_GATED_FACTOR = re.compile(
    r"\bintermediate_size_factor = isGatedActivation\(activation_type\) \? 2 : 1;"
)
_BF16_LAUNCHER = re.compile(r"(?m)^class Bf16MoeLauncher : public FusedMoeLauncher \{$")
_NEXT_CLASS = re.compile(r"(?m)^class \w+")
_CHECK_MOE = re.compile(r"void check_moe\(\) const override \{")
_METHOD_END = re.compile(r"(?m)^  \}$")
_ISPP_CHECK = re.compile(
    r"(?m)^(?P<indent>[ \t]*)TVM_FFI_ICHECK_EQ\(args->intermediate_size % 128, 0\)"
    r"\s*<< \"the second dimension of weights must be a multiple of 128\.\";"
)
_RELAXED_CHECK = (
    "// Gated activations tile 2 * intermediate_size GEMM1 rows by 128 and read",
    "// GEMM2's K in 64-element blocks; non-gated activations keep 128.",
    "int64_t const intermediate_size_alignment = intermediate_size_factor == 2 ? 64 : 128;",
    "TVM_FFI_ICHECK_EQ(args->intermediate_size % intermediate_size_alignment, 0)",
    '    << "BF16 MoE: intermediate_size must be a multiple of "',
    '    << intermediate_size_alignment << ", got " << args->intermediate_size << ".";',
)
# Each cloned function must reach the next through the cloned globals.
_DISPATCH = {
    "trtllm_bf16_moe": {"get_trtllm_moe_sm100_module"},
    "trtllm_bf16_routed_moe": {"get_trtllm_moe_sm100_module"},
    "get_trtllm_moe_sm100_module": {"_get_trtllm_moe_sm100_module_impl"},
    "_get_trtllm_moe_sm100_module_impl": {
        "gen_trtllm_gen_fused_moe_sm100_module",
        "register_custom_op",
    },
}


def _one(matches: list, what: str) -> re.Match:
    if len(matches) != 1:
        raise RuntimeError(
            f"Unsupported FlashInfer BF16 MoE launcher: expected exactly one {what}, "
            f"found {len(matches)}; review the BF16 MoE adapter."
        )
    return matches[0]


def _relax_bf16_intermediate_check(source: str) -> str:
    """Relax only Bf16MoeLauncher::check_moe to 64 for gated activations."""
    _one(list(_GATED_FACTOR.finditer(source)), "gated intermediate_size_factor")
    launcher = _one(list(_BF16_LAUNCHER.finditer(source)), "Bf16MoeLauncher")
    following = _NEXT_CLASS.search(source, launcher.end())
    class_end = following.start() if following else len(source)
    method = _one(
        list(_CHECK_MOE.finditer(source, launcher.end(), class_end)),
        "Bf16MoeLauncher::check_moe",
    )
    method_end = _METHOD_END.search(source, method.end(), class_end)
    if method_end is None:
        raise RuntimeError("Unsupported FlashInfer BF16 MoE launcher: no check_moe end")
    check = _one(
        list(_ISPP_CHECK.finditer(source, method.end(), method_end.start())),
        "intermediate_size % 128 check in Bf16MoeLauncher::check_moe",
    )
    relaxed = "\n".join(check["indent"] + line for line in _RELAXED_CHECK)
    return source[: check.start()] + relaxed + source[check.end() :]


def _relaxed_spec(*args, **kwargs):
    return _patched_launcher_spec(
        _relax_bf16_intermediate_check, "bf16_ispp64", *args, **kwargs
    )


def _require_jit_compiler() -> None:
    """Raise unless FlashInfer's JIT can compile the private module.

    The private module is not part of FlashInfer's AOT jit-cache, and FlashInfer
    reuses a module from its JIT workspace only when ninja finds it up to date
    with the current build, so a module built earlier does not count. This
    checks what that build needs: JIT enabled, FlashInfer's CUDA home and its
    nvcc. Nothing is generated or compiled here.
    """
    from flashinfer.jit import cpp_ext

    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        raise RuntimeError("FLASHINFER_DISABLE_JIT is set")
    # The compiler FlashInfer's generated build.ninja runs.
    cuda_home = cpp_ext.get_cuda_path()
    nvcc = os.environ.get("FLASHINFER_NVCC", f"{cuda_home}/bin/nvcc")
    if shutil.which(nvcc) is None:
        raise RuntimeError(f"FlashInfer's CUDA compiler {nvcc} is missing")


@functools.cache
def _entrypoints() -> dict:
    from flashinfer.fused_moe import core

    missing = sorted(name for name in _DISPATCH if not hasattr(core, name))
    if missing:
        raise RuntimeError(
            f"FlashInfer no longer defines {missing}; review the BF16 MoE adapter."
        )
    namespace = dict(vars(core))
    namespace["gen_trtllm_gen_fused_moe_sm100_module"] = _relaxed_spec
    for name in ("register_custom_op", "register_fake_op"):
        namespace[name] = functools.partial(
            _register_private, getattr(core, name), prefix=_OPERATORS
        )
    for name in _DISPATCH:
        namespace[name] = _clone(getattr(core, name), namespace)
    factory = "_get_trtllm_moe_sm100_module_impl"
    namespace[factory] = functools.cache(namespace[factory])
    for name, callees in _DISPATCH.items():
        missing = callees - set(inspect.unwrap(namespace[name]).__code__.co_names)
        if missing:
            raise RuntimeError(
                f"FlashInfer {name} no longer uses {sorted(missing)}; "
                "review the BF16 MoE adapter."
            )
    return namespace


@functools.cache
def gated_ispp_alignment() -> int:
    """Intermediate-size multiple gated BF16 MoE needs on the installed FlashInfer.

    ``GATED_ISPP_ALIGNMENT`` when the private launcher applies to the installed
    sources and FlashInfer's JIT can compile it, otherwise the stock launcher's
    ``STOCK_ISPP_ALIGNMENT``. Decided once per process; only reads the
    installed sources and FlashInfer's JIT settings, nothing is compiled.
    """
    try:
        from flashinfer.jit import env as jit_env

        _relax_bf16_intermediate_check(
            (jit_env.FLASHINFER_CSRC_DIR / _LAUNCHER).read_text()
        )
        _entrypoints()
        _require_jit_compiler()
    # Called at kernel registration: any FlashInfer layout this adapter does
    # not recognize, or a private module FlashInfer cannot compile, keeps the
    # stock launcher instead of failing the import or warmup.
    except Exception as error:
        logger.warning(
            "Gated BF16 TRT-LLM MoE keeps FlashInfer's multiple of %d: %s",
            STOCK_ISPP_ALIGNMENT,
            error,
        )
        return STOCK_ISPP_ALIGNMENT
    return GATED_ISPP_ALIGNMENT


def trtllm_bf16_moe(*args, **kwargs):
    """Run FlashInfer BF16 MoE from logits on the 64-aligned launcher.

    Arguments and returned tensors follow FlashInfer's same-named API.
    """
    return _entrypoints()["trtllm_bf16_moe"](*args, **kwargs)


def trtllm_bf16_routed_moe(*args, **kwargs):
    """Run FlashInfer BF16 MoE from precomputed routing on the 64-aligned launcher.

    Arguments and returned tensors follow FlashInfer's same-named API.
    """
    return _entrypoints()["trtllm_bf16_routed_moe"](*args, **kwargs)
