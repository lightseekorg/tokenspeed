"""Compile native per-thread control flow from Gluon into an inline device body.

The outer Gluon kernel supplies one invocation per physical thread. The body is
parsed and lowered by Gluon with scalar parameters, so ordinary if/while retain
per-thread semantics. No HIP/C++ device implementation is used. Scalar memory
operations bypass Gluon's redundant-lane masking; allocation and launch remain
in the outer kernel. LLVM optimization runs after this body is linked there.
"""

import re
from hashlib import sha256
from types import ModuleType

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import _native_call
from lib.tal.device import _CompileTimeView, source_key
from triton._C.libtriton import amd, ir, llvm, passes
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget
from triton.compiler.code_generator import ASTFunction, CodeGenerator
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language._core import (
    _unwrap_if_constexpr,
    builtin,
    distributed_type,
)
from triton.language.core import constexpr_type
from triton.runtime.cache import get_cache_manager


def _memory_code(dtype):
    return (
        "f32"
        if dtype == l.float32
        else "f16" if dtype == l.float16 else "i" + str(dtype.primitive_bitwidth)
    )


@builtin
def _load(
    ptr,
    mask=None,
    other=None,
    boundary_check=(),
    padding_option="",
    cache_modifier="",
    eviction_policy="",
    volatile=False,
    _semantic=None,
):
    if (
        ptr.type.is_block()
        or mask is not None
        or boundary_check
        or padding_option
        or cache_modifier
        or eviction_policy
        or volatile
    ):
        raise ValueError(
            "thread_jit load requires a scalar pointer; spell conditional accesses with if"
        )
    dtype = ptr.dtype.element_ty
    code = _memory_code(dtype)
    value = _native_call(
        "memory.load." + code,
        code,
        ("p" + str(_unwrap_if_constexpr(ptr.dtype.address_space)),),
        (ptr,),
        False,
        _semantic=_semantic,
    )
    return _semantic.bitcast(value, dtype)


@builtin
def _store(
    ptr,
    value,
    mask=None,
    boundary_check=(),
    cache_modifier="",
    eviction_policy="",
    _semantic=None,
):
    if ptr.type.is_block() or boundary_check or cache_modifier or eviction_policy:
        raise ValueError("thread_jit store requires a scalar pointer")
    dtype = ptr.dtype.element_ty
    value = _semantic.cast(_semantic.to_tensor(value), dtype)
    code = _memory_code(dtype)
    if dtype == l.bfloat16:
        value = _semantic.bitcast(value, l.uint16)
    signature = ("p" + str(_unwrap_if_constexpr(ptr.dtype.address_space)), code)
    args = (ptr, value)
    name = "memory.store." + code
    if mask is not None:
        name = "when:" + name
        signature += ("i1",)
        args += (mask,)
    return _native_call(name, "void", signature, args, False, _semantic=_semantic)


@builtin
def _atomic_add(ptr, val, mask=None, sem=None, scope=None, _semantic=None):
    if (
        ptr.type.is_block()
        or mask is not None
        or sem not in (None, "relaxed")
        or scope not in (None, "gpu")
    ):
        raise ValueError(
            "thread_jit atomic_add supports scalar native relaxed agent atomics"
        )
    dtype = ptr.dtype.element_ty
    code = _memory_code(dtype)
    if code not in ("i32", "i64"):
        raise ValueError(
            "thread_jit atomic_add currently supports native integer counters"
        )
    value = _semantic.cast(_semantic.to_tensor(val), dtype)
    return _native_call(
        "memory.atomic.add." + code,
        code,
        ("p" + str(_unwrap_if_constexpr(ptr.dtype.address_space)), code),
        (ptr, value),
        False,
        _semantic=_semantic,
    )


_language = ModuleType(l.__name__)
_language.__dict__.update(l.__dict__)
_language.load, _language.store = _load, _store
_language.atomic_add = _atomic_add


class thread_device_method:
    """Keep a native device method's per-thread branches at the Gluon boundary."""

    def __init__(self, fn):
        fn.__annotations__["self"] = l.constexpr
        self.jit = thread_jit(fn)

    def __get__(self, obj, owner=None):
        return self if obj is None else _BoundThreadDeviceMethod(self, obj)


class _BoundThreadDeviceMethod:
    __triton_builtin__ = True

    def __init__(self, method, obj):
        self.method, self.obj = method, obj

    @property
    def cache_key(self):
        return self.obj.cache_key + self.method.jit.cache_key

    def __call__(self, *args, _semantic=None, _generator=None, **kwargs):
        return self.method.jit(
            l.constexpr(_CompileTimeView(self.obj)),
            *args,
            _semantic=_semantic,
            _generator=_generator,
            **kwargs,
        )


def _scalar_type(value, tensors):
    if isinstance(value, l.tensor):
        tensors.append(value)
        return value.dtype
    if isinstance(value, l.tuple):
        return l.tuple_type(
            [_scalar_type(v, tensors) for v in value.values], value.type.fields
        )
    return l.constexpr(_unwrap_if_constexpr(value)).type


def _lower_device(mod, backend, options):
    """Triton 3.8's LLVM lowering without assigning kernel ABI or optimizing."""
    mod = backend.gluon_to_ttgir(mod, {"num_ctas": options.num_ctas}, options)
    libraries = dict(
        re.findall(
            r'libname = "(petit_native_[0-9a-f]+)", libpath = "([^"]+)"', str(mod)
        )
    )
    pm = ir.pass_manager(mod.context)
    llvm23 = hasattr(amd.passes.ttgpuir, "add_warp_pipeline_conversion")
    amd.passes.ttgpuir.add_update_async_wait_count(pm, options.arch)
    if llvm23:
        amd.passes.ttgpuir.add_warp_pipeline_conversion(pm, options.arch)
    passes.convert.add_scf_to_cf(pm)
    passes.gluon.add_inliner(pm)
    passes.convert.add_index_to_llvmir(pm)
    if llvm23:
        amd.passes.ttgpuir.add_allocate_shared_memory(pm, options.arch)
    else:
        amd.passes.ttgpuir.add_allocate_shared_memory(pm)
    passes.ttgpuir.add_allocate_global_scratch_memory(pm)
    amd.passes.ttgpuir.add_to_llvmir(pm, options.arch, True)
    if llvm23:
        amd.passes.ttgpuir.add_warp_specialize_to_llvm(pm, options.arch)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.convert.add_cf_to_llvmir(pm)
    passes.convert.add_arith_to_llvmir(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    if llvm23:
        amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm, options.arch, True)
        passes.convert.add_reconcile_unrealized_casts(pm)
    else:
        amd.passes.ttgpuir.add_builtin_func_to_llvmir(pm, True)
    pm.run(mod, "petit_thread_device_lowering")
    llvm.init_targets()
    context = llvm.context()
    result = llvm.to_module(mod, context)
    amd.attach_target_triple(result)
    llvm.attach_datalayout(result, amd.TARGET_TRIPLE, options.arch, "")
    for fn in result.get_functions():
        if not fn.is_declaration():
            fn.set_calling_conv(0)
    if libraries:
        llvm.link_extern_libs(result, list(libraries.values()))
    return str(result)


def _disable_atomic_or_loop_unrolling(source):
    """Keep native runtime publication loops as loops through LLVM optimize.

    Triton's LLVM pipeline otherwise unrolls the small dynamic loop containing
    BufferResource::AtomicOrU32 eight times. Clang retains the corresponding
    native Petit loop. Attach the standard LLVM loop annotation only to a
    backedge block that calls the raw-buffer atomic-or adapter.
    """
    atomic_adapters = set()
    for match in re.finditer(
        r"define[^@]*@([^(]+)\([^\n]*\)[^{]*\{\n(.*?)\n\}", source, re.S
    ):
        if "@llvm.amdgcn.raw.buffer.atomic.or.i32" in match.group(2):
            atomic_adapters.add(match.group(1))
    if not atomic_adapters:
        return source

    lines = source.splitlines()
    annotated = []
    for index, line in enumerate(lines):
        if not any(
            ("call " in line and "@" + symbol + "(" in line)
            for symbol in atomic_adapters
        ):
            continue
        block_begin = index
        while block_begin >= 0 and not re.fullmatch(
            r"[A-Za-z$._][\w$.-]*:", lines[block_begin].strip()
        ):
            block_begin -= 1
        block_end = index
        while block_end < len(lines) and not lines[block_end].lstrip().startswith(
            "br "
        ):
            block_end += 1
        if block_begin < 0 or block_end == len(lines):
            continue
        branch = lines[block_end]
        if re.fullmatch(r"\s*br label %[A-Za-z$._][\w$.-]*\s*", branch):
            annotated.append(block_end)
    if not annotated:
        return source

    metadata_ids = [int(value) for value in re.findall(r"!(\d+)", source)]
    loop_id = max(metadata_ids, default=-1) + 1
    property_id = loop_id + 1
    for index in annotated:
        lines[index] = lines[index].rstrip() + f", !llvm.loop !{loop_id}"
    lines.extend(
        [
            f"!{loop_id} = distinct !{{!{loop_id}, !{property_id}}}",
            f'!{property_id} = !{{!"llvm.loop.unroll.disable"}}',
        ]
    )
    return "\n".join(lines) + "\n"


class thread_jit:
    __triton_builtin__ = True

    def __init__(self, fn):
        self.jit = g.jit(fn)
        self._signature = self.jit.signature
        self._compiled = {}

    @property
    def cache_key(self):
        return self.jit.cache_key + source_key()

    def _compile(self, arg_types, num_warps, arch):
        key = sha256(
            (
                self.cache_key
                + str([t.mangle() for t in arg_types])
                + str(num_warps)
                + arch
            ).encode()
        ).hexdigest()
        if key in self._compiled:
            return self._compiled[key]
        library = "petit_native_" + key
        name, wrapper = library + "_body", library + "_call"
        backend = HIPBackend(GPUTarget("hip", arch, 64))
        options = backend.parse_options(
            dict(num_warps=num_warps, enable_fp_fusion=False, sanitize_overflow=False)
        )
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        module_map = backend.get_module_map()
        module_map[l.__name__] = _language
        try:
            prototype = ASTFunction([], list(arg_types), {})
        except TypeError:
            # LLVM 22's Triton frontend still takes a constants map.
            prototype = ASTFunction([], list(arg_types), {}, {})
        file_name = getattr(self.jit, "file_name", self.jit.fn.__code__.co_filename)
        begin_line = getattr(
            self.jit, "def_file_line_number", self.jit.starting_line_number + 1
        )
        generator = CodeGenerator(
            context,
            prototype,
            self.jit.get_capture_scope(),
            name,
            self.jit,
            options=options,
            codegen_fns=backend.get_codegen_implementation(options),
            module_map=module_map,
            is_gluon=True,
            is_kernel=False,
            file_name=file_name,
            begin_line=begin_line,
        )
        for attr, value in [
            ("ttg.num-warps", num_warps),
            ("ttg.num-ctas", 1),
            ("ttg.threads-per-warp", 64),
        ]:
            generator.module.set_attr(attr, generator.builder.get_int32_attr(value))
        generator.module.set_attr(
            "ttg.target", generator.builder.get_string_attr("hip:" + arch)
        )
        generator.visit(self.jit.parse())
        generator.fn.set_attr(
            "sym_visibility", generator.builder.get_string_attr("public")
        )
        generator.module.context = context
        result_type = generator.ret_type
        if not isinstance(result_type, (l.dtype, constexpr_type)):
            raise TypeError("thread_jit device boundary returns one scalar or None")
        source = _lower_device(generator.module, backend, options)
        lines = source.splitlines()
        start = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("define ") and "@" + name + "(" in line
        )
        end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
        head, params_text = lines[start].split("@" + name + "(", 1)
        params_text = params_text.rsplit(")", 1)[0]
        params = params_text.split(", ") if params_text else []
        # The device lowering appends global/profile scratch pointers. This
        # ABI boundary permits neither allocation, so both must be unused.
        for param in params[-2:]:
            variable = param.split()[-1]
            if re.search(
                re.escape(variable) + r"\b", "\n".join(lines[start + 1 : end])
            ):
                raise ValueError(
                    "thread_jit scratch allocation must remain in the outer kernel"
                )
        entry_number = len(params)
        params = params[:-2]
        params = [re.sub(r"%(\d+)", r"%v\1", param) for param in params]
        for i in range(start + 1, end):
            lines[i] = re.sub(r"%(\d+)", r"%v\1", lines[i])
            lines[i] = re.sub(r"^(\d+):", r"v\1:", lines[i])
        lines.insert(start + 1, f"v{entry_number}:")
        lines[start] = (
            head + "@" + name + "(" + ", ".join(params) + ") alwaysinline convergent {"
        )
        is_void = isinstance(result_type, constexpr_type) or result_type == l.void
        returned = "void" if is_void else head.removeprefix("define ").strip()
        if returned == "void":
            wrapper_ir = (
                "define i32 @"
                + wrapper
                + "("
                + ", ".join(params)
                + ") alwaysinline convergent {\n"
            )
            wrapper_ir += (
                "call void @" + name + "(" + ", ".join(params) + ")\nret i32 0\n}\n"
            )
        else:
            wrapper_ir = (
                "define "
                + returned
                + " @"
                + wrapper
                + "("
                + ", ".join(params)
                + ") alwaysinline convergent {\n"
            )
            wrapper_ir += (
                "%value = call "
                + returned
                + " @"
                + name
                + "("
                + ", ".join(params)
                + ")\nret "
                + returned
                + " %value\n}\n"
            )
        source = "\n".join(lines) + "\n" + wrapper_ir
        source = _disable_atomic_or_loop_unrolling(source)
        manager = get_cache_manager(key)
        path = manager.put(source, "thread_device.ll", binary=False)
        self._compiled[key] = (library, path, wrapper, result_type)
        return self._compiled[key]

    def __call__(self, *args, _semantic=None, _generator=None, **kwargs):
        if _generator is None:
            raise TypeError(
                "thread_jit functions are called from an outer @gluon.jit kernel"
            )
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = []
        arg_types = tuple(
            _scalar_type(bound.arguments[n], values) for n in self.jit.arg_names
        )
        library, path, symbol, result_type = self._compile(
            arg_types,
            _semantic.builder.options.num_warps,
            _semantic.builder.options.arch,
        )
        for i in range(1, len(values)):
            values[0], values[i] = _semantic.broadcast_impl_value(values[0], values[i])
        for i in range(1, len(values)):
            values[i], _ = _semantic.broadcast_impl_value(values[i], values[0])
        is_void = isinstance(result_type, constexpr_type) or result_type == l.void
        dtype = l.int32 if is_void else result_type
        ty = (
            distributed_type(dtype, values[0].type.shape, values[0].type.layout)
            if values[0].type.is_block()
            else dtype
        )
        handle = _semantic.builder.create_extern_elementwise(
            library,
            path,
            symbol,
            [v.handle for v in values],
            ty.to_ir(_semantic.builder),
            False,
        )
        return None if is_void else l.tensor(handle, ty)
