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

"""One process of test_cute_compile_cache: compile, run, report.

Usage: python cute_compile_cache_child.py <case> <output.pt> [--kernel-file F]
[--options S] [--stream explicit|env] [--capability M,N] [--compiles N]
[--pretend CHANGE]. Prints one
JSON line: whether the cache installed, its keyed compilations and hits, and
the CuTe DSL kernels the monitor saw compile. ``--pretend`` stands in for a
cutlass-dsl release the cache does not know.
"""

import argparse
import importlib.util
import inspect
import json
import logging
import math

import tokenspeed_kernel.compile_monitor as compile_monitor
import tokenspeed_kernel.cute_compile_cache as cute_compile_cache
import torch
from cutlass.base_dsl.dsl import BaseDSL
from tokenspeed_kernel.cute_compile_cache import (
    cute_compile_cache_stats,
    install_cute_compile_cache,
)

# Extra fields a case adds to the printed report.
_REPORT: dict = {}

_CHANGES = ("untested-version", "renamed-parameters", "no-cache-shape", "keying-error")


def _pretend_before_install(change: str) -> None:
    if change == "untested-version":
        cute_compile_cache._cutlass_dsl_version = lambda: "4.9.0"
    elif change == "renamed-parameters":
        original = BaseDSL.compile_and_cache

        def compile_and_cache(self, *args, **kwargs):
            return original(self, *args, **kwargs)

        BaseDSL.compile_and_cache = compile_and_cache
    elif change == "keying-error":

        def gpu_arch(dsl):
            raise AttributeError("'CompileOptions' object has no attribute 'gpu_arch'")

        cute_compile_cache._gpu_arch = gpu_arch


def _pretend_after_install(change: str) -> None:
    if change != "no-cache-shape":
        return
    # A caller that passes no_cache in another form; cutlass-dsl reads it as
    # "not cached" (``not no_cache`` is False).
    installed = BaseDSL.compile_and_cache
    signature = inspect.signature(installed)

    def compile_and_cache(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        if bound.arguments["no_cache"] is True:
            bound.arguments["no_cache"] = "always"
        return installed(*bound.args, **bound.kwargs)

    BaseDSL.compile_and_cache = compile_and_cache


def _add(kernel_file: str, options: str | None, compiles: int) -> list[torch.Tensor]:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    spec = importlib.util.spec_from_file_location("cute_cache_case_kernel", kernel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    kwargs = {} if options is None else {"options": options}
    for _ in range(compiles):
        compiled = cute.compile(
            module.add, from_dlpack(x), cutlass.Float32(1.5), 32, **kwargs
        )
        compiled(from_dlpack(x), cutlass.Float32(1.5))
    return [x]


def _add_dropped(kernel_file: str) -> list[torch.Tensor]:
    import gc
    import weakref

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    spec = importlib.util.spec_from_file_location("cute_cache_case_kernel", kernel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    alive = []
    for threads in (8, 16, 32):
        compiled = cute.compile(
            module.add, from_dlpack(x), cutlass.Float32(1.5), threads
        )
        compiled(from_dlpack(x), cutlass.Float32(1.5))
        alive.append(weakref.ref(compiled))
        del compiled
    gc.collect()
    _REPORT["alive"] = sum(ref() is not None for ref in alive)
    return [x]


def _add_tvm_ffi(kernel_file: str, stream: str) -> list[torch.Tensor]:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_stream

    spec = importlib.util.spec_from_file_location("cute_cache_case_kernel", kernel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    # The same trace either way; only the TVM-FFI entry attached after
    # compiling takes the stream argument or reads the environment stream.
    env_stream = stream == "env"
    compiled = cute.compile(
        module.add,
        from_dlpack(x),
        cutlass.Float32(1.5),
        32,
        make_fake_stream(use_tvm_ffi_env_stream=env_stream),
        options="--enable-tvm-ffi",
    )
    if env_stream:
        compiled(x, 1.5)
    else:
        compiled(x, 1.5, torch.cuda.current_stream())
    return [x]


def _gated_residual() -> list[torch.Tensor]:
    from tokenspeed_kernel.ops.residual import (
        gated_residual_mix,
        pad_gated_residual_projection_weight,
    )

    g, h, r, rows = 4, 2560, 320, 17
    generator = torch.Generator(device="cuda").manual_seed(1729)
    x = torch.randn(
        rows, g * h, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    projection = torch.randn(
        r + g, g * h, device="cuda", dtype=torch.bfloat16, generator=generator
    ) / math.sqrt(g * h)
    up = torch.randn(g * h, r, device="cuda", dtype=torch.bfloat16, generator=generator)
    projection = pad_gated_residual_projection_weight(projection, projection_rows=r + g)
    outputs = gated_residual_mix(
        x,
        projection,
        up / math.sqrt(r),
        g,
        h,
        r,
        projection_rows=r + g,
        projection_scale=1.0 / g,
        weights_independent=True,
        override="cute_fused_hyperconnection_mix",
        solution=None,
    )
    return [value for value in outputs if value is not None]


def _stream_topk() -> list[torch.Tensor]:
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_stream_topk import (
        kernel as cute_dsl_qsa_stream_topk,
    )

    torch.manual_seed(83)
    # 128 pages of 64 keys: an 8192-key capacity, the guarded (keep) small path.
    rows, heads, head_dim, pages = 8, 4, 128, 128
    page_size, block_topk = 64, 512
    query = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key_cache = torch.randn(
        pages * page_size, 1, head_dim, device="cuda", dtype=torch.bfloat16
    )
    page_table = torch.randperm(pages, device="cuda").to(torch.int32).unsqueeze(0)
    request_indices = torch.zeros(rows, device="cuda", dtype=torch.int64)
    complete_blocks = torch.full((rows,), pages, device="cuda", dtype=torch.int32)
    selected = cute_dsl_qsa_stream_topk(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=rows,
        max_partial_bytes=32 * 1024 * 1024,
        enable_pdl=False,
    )
    return [selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=("add", "add_dropped", "add_tvm_ffi", "gated_residual", "stream_topk"),
    )
    parser.add_argument("output")
    parser.add_argument("--kernel-file")
    parser.add_argument("--options")
    parser.add_argument("--stream", choices=("explicit", "env"))
    parser.add_argument("--capability")
    parser.add_argument("--compiles", type=int, default=1)
    parser.add_argument("--pretend", choices=_CHANGES)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("tokenspeed_kernel.cute_compile_cache").setLevel(logging.DEBUG)
    if args.capability is not None:
        major, minor = (int(part) for part in args.capability.split(","))
        torch.cuda.get_device_capability = lambda device=None: (major, minor)
    if args.pretend is not None:
        _pretend_before_install(args.pretend)
    installed = install_cute_compile_cache()
    if args.pretend is not None:
        _pretend_after_install(args.pretend)
    compile_monitor.install_compile_monitor("warn")
    if args.case == "add":
        outputs = _add(args.kernel_file, args.options, args.compiles)
    elif args.case == "add_dropped":
        outputs = _add_dropped(args.kernel_file)
    elif args.case == "add_tvm_ffi":
        outputs = _add_tvm_ffi(args.kernel_file, args.stream)
    elif args.case == "gated_residual":
        outputs = _gated_residual()
    else:
        outputs = _stream_topk()
    torch.cuda.synchronize()
    torch.save([value.cpu() for value in outputs], args.output)
    stats = cute_compile_cache_stats()
    kernels = sorted(
        name
        for name in compile_monitor._hooks.monitor._kernels
        if not name.startswith("cutlass.")
    )
    print(
        json.dumps(
            {
                "installed": installed,
                "keyed": None if stats is None else stats.keyed,
                "hits": None if stats is None else stats.hits,
                "kernels": kernels,
                **_REPORT,
            }
        )
    )


if __name__ == "__main__":
    main()
