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

import os
import sys
import textwrap
import time
import uuid

import pytest
import torch
from tokenspeed_kernel.hot_reload import (
    ReloadError,
    SourceTracker,
    dependent_modules,
    reload_modules,
)
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.warmup import warmup_registrations

KERNELS = """
import functools

def f(x):
    return x + 1

@functools.cache
def scaled(x):
    return x * 10

_CACHE = {}

def cached(key):
    return _CACHE.setdefault(key, f(key))

class Base:
    def who(self):
        return "base"

class Layer(Base):
    scale = 2

    def forward(self, x):
        return f(x) * self.scale

    def who(self):
        return "layer"
"""

USER = """
from {package}.kernels import f, Layer

def use(x):
    return f(x)
"""


class Package:
    """A throwaway importable package whose module sources a test edits."""

    def __init__(self, root, monkeypatch) -> None:
        self.name = f"hot_reload_{uuid.uuid4().hex[:8]}"
        self.dir = root / self.name
        self.dir.mkdir()
        (self.dir / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str(root))

    def write(self, module: str, source: str) -> None:
        path = self.dir / f"{module}.py"
        path.write_text(textwrap.dedent(source).replace("{package}", self.name))
        # Older than any tracker made after it, whatever the filesystem's clock granularity.
        self._stamp(path, time.time_ns() - 10_000_000_000)

    def edit(self, module: str, old: str, new: str) -> None:
        path = self.dir / f"{module}.py"
        source = path.read_text()
        assert old in source
        path.write_text(source.replace(old, new))
        # Newer than any tracker made before it.
        self._stamp(path, time.time_ns() + 2_000_000_000)

    def _stamp(self, path, stamp: int) -> None:
        os.utime(path, ns=(stamp, stamp))

    def module(self, module: str):
        __import__(f"{self.name}.{module}")
        return sys.modules[f"{self.name}.{module}"]

    def tracker(self) -> SourceTracker:
        tracker = SourceTracker([self.name], loaded_at=time.time())
        tracker.remember_bindings([self.name])
        return tracker

    def reload_stale(self, tracker: SourceTracker):
        names = dependent_modules(tracker.stale(), [self.name])
        return reload_modules(tracker.read(names))

    def forget(self) -> None:
        for name in [name for name in sys.modules if name.startswith(self.name)]:
            del sys.modules[name]


@pytest.fixture
def package(tmp_path, monkeypatch):
    package = Package(tmp_path, monkeypatch)
    registered = KernelRegistry.get().snapshot()
    yield package
    KernelRegistry.get().restore(registered)
    package.forget()


def test_every_earlier_reference_runs_the_edited_code(package):
    package.write("kernels", KERNELS)
    package.write("user", USER)
    kernels, user = package.module("kernels"), package.module("user")
    f, layer = kernels.f, kernels.Layer()
    forward = layer.forward
    assert (f(1), kernels.scaled(1), forward(1), kernels.cached(5)) == (2, 10, 4, 6)
    tracker = package.tracker()
    package.edit("kernels", "return x + 1", "return x + 100")
    package.edit("kernels", "return x * 10", "return x * 20")
    package.edit("kernels", "scale = 2", "scale = 3")

    report = package.reload_stale(tracker)

    assert report.modules == [kernels.__name__, user.__name__]
    # The imported name, the bound method, the live instance and the cleared caches all see it.
    assert (f(1), user.use(1), forward(1), kernels.scaled(1)) == (101, 101, 303, 20)
    assert kernels.cached(5) == 105
    assert user.f is f and isinstance(layer, kernels.Layer)
    tracker.mark_loaded(tracker.read([kernels.__name__, user.__name__]))
    assert tracker.stale() == []


def test_a_method_that_starts_calling_super_reaches_live_instances(package):
    package.write("kernels", KERNELS)
    kernels = package.module("kernels")
    layer = kernels.Layer()
    tracker = package.tracker()
    package.edit("kernels", 'return "layer"', 'return "layer:" + super().who()')

    report = package.reload_stale(tracker)

    assert layer.who() == "layer:base"
    assert report.redirected == [f"{kernels.__name__}.Layer.who"]


def test_a_function_whose_closure_changes_shape_serves_its_new_body(package):
    package.write(
        "kernels",
        KERNELS + """
def traced(fn):
    def wrapper(x):
        return fn(x)
    return wrapper

def plain(x):
    return x
""",
    )
    kernels = package.module("kernels")
    plain = kernels.plain
    tracker = package.tracker()
    package.edit(
        "kernels",
        "def plain(x):\n    return x",
        "@traced\ndef plain(x):\n    return -x",
    )

    report = package.reload_stale(tracker)

    assert plain(3) == -3
    assert report.redirected == [f"{kernels.__name__}.plain"]


def test_a_failing_module_restores_every_module_and_registration(package):
    package.write(
        "kernels",
        KERNELS + """
import torch
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures

@register_kernel(
    "hot_reload_test",
    "op",
    solution="triton",
    name="{package}_op",
    signatures=format_signatures(("x",), "dense", {torch.float32}),
)
def op(x):
    return "old"
""",
    )
    kernels = package.module("kernels")
    registry = KernelRegistry.get()
    tracker = package.tracker()
    package.edit("kernels", 'return "old"', 'return "new"\n\nundefined_name')
    package.edit("kernels", "return x + 1", "return x + 100")

    with pytest.raises(ReloadError, match="undefined_name"):
        package.reload_stale(tracker)

    assert kernels.f(1) == 2
    assert registry.get_impl(f"{package.name}_op") is kernels.op
    assert kernels.op(0) == "old"


def test_a_class_attribute_naming_another_modules_class_never_modifies_it(package):
    package.write(
        "backends", "class Fast:\n    name = 'fast'\n\nclass Slow:\n    name = 'slow'\n"
    )
    package.write(
        "layer",
        "from {package}.backends import Fast, Slow\n\nclass Layer:\n    backend = Fast\n    base = object\n",
    )
    backends, layer = package.module("backends"), package.module("layer")
    tracker = package.tracker()
    package.edit("layer", "backend = Fast", "backend = Slow")

    report = package.reload_stale(tracker)

    assert layer.Layer.backend is backends.Slow
    assert (backends.Fast.name, backends.Slow.name) == ("fast", "slow")
    assert report.stale == []


def test_an_attribute_switched_between_two_classes_of_one_module_modifies_neither(
    package,
):
    package.write(
        "layer",
        "class Fast:\n    name = 'fast'\n\nclass Slow:\n    name = 'slow'\n\nclass Layer:\n    backend = Fast\n",
    )
    layer = package.module("layer")
    tracker = package.tracker()
    package.edit("layer", "backend = Fast", "backend = Slow")

    package.reload_stale(tracker)

    assert layer.Layer.backend is layer.Slow
    assert (layer.Fast.name, layer.Slow.name) == ("fast", "slow")


def test_a_deleted_override_and_a_new_base_reach_live_instances(package):
    package.write(
        "layer",
        "class A:\n    def who(self):\n        return 'a'\n\n"
        "class B:\n    def who(self):\n        return 'b'\n\n"
        "class C(A):\n    def who(self):\n        return 'c'\n",
    )
    layer = package.module("layer")
    instance = layer.C()
    tracker = package.tracker()
    package.edit(
        "layer",
        "class C(A):\n    def who(self):\n        return 'c'\n",
        "class C(B):\n    pass\n",
    )

    report = package.reload_stale(tracker)

    assert instance.who() == "b" and isinstance(instance, layer.B)
    report.revert()
    assert instance.who() == "c" and isinstance(instance, layer.A)


def test_a_provider_reloads_before_the_module_that_copied_its_constant(package):
    package.write("provider", "N = 1\n")
    package.write(
        "consumer", "from {package}.provider import N\n\ndef f():\n    return N\n"
    )
    # Imported first, the consumer precedes its provider in sys.modules.
    consumer = package.module("consumer")
    tracker = package.tracker()
    package.edit("provider", "N = 1", "N = 2")

    package.reload_stale(tracker)

    assert consumer.f() == 2


def test_a_revert_restores_an_earlier_redirect(package):
    package.write(
        "kernels",
        "def traced(fn):\n    def wrapper(x):\n        return fn(x)\n    return wrapper\n\n"
        "def plain(x):\n    return x + 1\n",
    )
    kernels = package.module("kernels")
    plain = kernels.plain
    tracker = package.tracker()
    package.edit(
        "kernels",
        "def plain(x):\n    return x + 1",
        "@traced\ndef plain(x):\n    return x + 100",
    )
    tracker.mark_loaded(tracker.read([kernels.__name__]))
    package.reload_stale(package.tracker())
    tracker = package.tracker()
    package.edit("kernels", "return x + 100", "return x + 200")

    report = package.reload_stale(tracker)

    assert plain(1) == 201
    report.revert()
    assert plain(1) == 101


def test_a_failure_while_patching_undoes_what_was_already_patched(package):
    package.write(
        "meta",
        "class Strict(type):\n    def __setattr__(cls, name, value):\n"
        "        if name == 'added':\n            raise RuntimeError('refused')\n"
        "        super().__setattr__(name, value)\n",
    )
    package.write(
        "kernels",
        "from {package}.meta import Strict\n\ndef f():\n    return 1\n\nclass C(metaclass=Strict):\n    pass\n",
    )
    kernels = package.module("kernels")
    f = kernels.f
    tracker = package.tracker()
    package.edit("kernels", "return 1", "return 2")
    package.edit(
        "kernels",
        "class C(metaclass=Strict):\n    pass",
        "class C(metaclass=Strict):\n    added = 1",
    )

    with pytest.raises(ReloadError, match="refused"):
        package.reload_stale(tracker)

    assert f() == 1 and kernels.f is f


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_warmup_variants_move_onto_the_old_kernel_object(package):
    package.write(
        "warm",
        "from tokenspeed_kernel._triton import tl, triton\n"
        "from tokenspeed_kernel.warmup import register_triton_warmup\n\n"
        "@triton.jit\ndef _kernel(x_ptr, BLOCK: tl.constexpr):\n    pass\n\n"
        "register_triton_warmup(_kernel, variants=({'BLOCK': (64,)},))\n",
    )
    warm = package.module("warm")
    kernel = warm._kernel
    tracker = package.tracker()
    package.edit("warm", "(64,)", "(128,)")

    report = package.reload_stale(tracker)

    mine = {
        k: v for k, v in warmup_registrations().items() if _defining(k) == warm.__name__
    }
    assert mine == {kernel: ({"BLOCK": (128,)},)}
    report.revert()
    mine = {
        k: v for k, v in warmup_registrations().items() if _defining(k) == warm.__name__
    }
    assert mine == {kernel: ({"BLOCK": (64,)},)}


def _defining(kernel) -> str:
    return kernel.fn.__module__


def test_closures_from_one_factory_swapped_under_an_alias_modify_neither(package):
    package.write(
        "factory",
        "def make(n):\n    def wrapper():\n        return n\n    return wrapper\n\nFast = make(1)\nSlow = make(2)\n",
    )
    package.write(
        "layer", "from {package}.factory import Fast, Slow\n\nbackend = Fast\n"
    )
    factory, layer = package.module("factory"), package.module("layer")
    tracker = package.tracker()
    package.edit("layer", "backend = Fast", "backend = Slow")

    package.reload_stale(tracker)

    assert layer.backend is factory.Slow
    assert (factory.Fast(), factory.Slow()) == (1, 2)


def test_a_new_classmethod_override_can_call_super(package):
    package.write(
        "layer",
        "class B:\n    @classmethod\n    def f(cls):\n        return 1\n\nclass C(B):\n    pass\n",
    )
    layer = package.module("layer")
    cls = layer.C
    tracker = package.tracker()
    package.edit(
        "layer",
        "class C(B):\n    pass",
        "class C(B):\n    @classmethod\n    def f(cls):\n        return super().f() + 1",
    )

    package.reload_stale(tracker)

    assert cls.f() == 2


def test_deleted_overrides_data_and_functions_stop_serving(package):
    package.write(
        "layer",
        "class B:\n    def f(self):\n        return 1\n\n"
        "class C(B):\n    n = 2\n    f = lambda self: 2\n\n"
        "def helper():\n    return 3\n",
    )
    layer = package.module("layer")
    instance = layer.C()
    tracker = package.tracker()
    package.edit("layer", "    f = lambda self: 2\n", "")
    package.edit("layer", "def helper():\n    return 3\n", "")

    package.reload_stale(tracker)

    assert instance.f() == 1 and "helper" not in vars(layer)
    tracker = package.tracker()
    package.edit("layer", "    n = 2\n", "    pass\n")
    package.reload_stale(tracker)
    assert "n" not in vars(layer.C)


def test_class_data_deleted_from_the_source_stops_shadowing_its_base(package):
    package.write("layer", "class B:\n    n = 1\n\nclass C(B):\n    n = 2\n")
    layer = package.module("layer")
    instance = layer.C()
    tracker = package.tracker()
    package.edit("layer", "class C(B):\n    n = 2\n", "class C(B):\n    pass\n")

    package.reload_stale(tracker)

    assert instance.n == 1


def test_attributes_installed_at_run_time_survive_a_reload(package):
    package.write(
        "layer",
        "class B:\n    def __init_subclass__(cls):\n        cls.cache = {}\n\n"
        "class C(B):\n    pass\n\ncache = globals().setdefault('cache', {})\n\n"
        "def read():\n    return cache\n",
    )
    layer = package.module("layer")
    layer.C.installed = 1
    tracker = package.tracker()
    package.edit("layer", "def read():", "def read() -> dict:")

    package.reload_stale(tracker)

    assert layer.read() == {} and layer.C.installed == 1 and "cache" in vars(layer.C)


def test_a_tuple_of_classes_holds_the_patched_classes(package):
    package.write(
        "methods",
        "class Fp8:\n    def name(self):\n        return 'old'\n\n"
        "METHODS = (Fp8,)\n\ndef supported(method):\n"
        "    return isinstance(method, METHODS)\n",
    )
    methods = package.module("methods")
    method = methods.Fp8()
    tracker = package.tracker()
    package.edit("methods", "return 'old'", "return 'new'")

    package.reload_stale(tracker)

    # A live instance still passes the check its module makes against the tuple.
    assert methods.supported(method) and method.name() == "new"
    assert methods.METHODS[0] is type(method)


def test_a_dict_keyed_by_a_class_finds_the_patched_class(package):
    package.write(
        "dispatch",
        "class Handler:\n    def name(self):\n        return 'old'\n\n"
        "DISPATCH = {Handler: 'fast'}\n\ndef route(handler):\n"
        "    return DISPATCH[type(handler)]\n",
    )
    dispatch = package.module("dispatch")
    handler = dispatch.Handler()
    tracker = package.tracker()
    package.edit("dispatch", "return 'old'", "return 'new'")

    package.reload_stale(tracker)

    assert dispatch.route(handler) == "fast" and handler.name() == "new"


def test_a_removed_enum_member_stays_and_the_rest_of_the_edit_lands(package):
    package.write(
        "modes",
        "import enum\n\nclass Mode(enum.Enum):\n    FAST = 1\n    SLOW = 2\n\n"
        "def label():\n    return 'old'\n",
    )
    modes = package.module("modes")
    tracker = package.tracker()
    package.edit("modes", "    SLOW = 2\n", "")
    package.edit("modes", "return 'old'", "return 'new'")

    report = package.reload_stale(tracker)

    assert modes.label() == "new"
    assert any(entry.endswith("Mode.SLOW") for entry in report.stale)


def test_the_report_names_run_time_state_the_reload_reset(package):
    package.write(
        "config",
        "BACKEND = None\nKEPT = globals().setdefault('KEPT', {})\n\n"
        "def init(backend):\n    global BACKEND\n    BACKEND = backend\n"
        "    KEPT['backend'] = backend\n\ndef get():\n    return BACKEND\n",
    )
    config = package.module("config")
    config.init("fast")
    tracker = package.tracker()
    package.edit("config", "def get():", "def get() -> str | None:")

    report = package.reload_stale(tracker)

    # Plain state returns to its import-time value and is reported; setdefault state survives.
    assert config.get() is None and config.KEPT == {"backend": "fast"}
    assert report.reset == [f"{config.__name__}.BACKEND"]


def test_a_consumer_still_importing_a_deleted_definition_fails_the_reload(package):
    package.write("provider", "def helper():\n    return 1\n")
    package.write("consumer", "from {package}.provider import helper\n")
    provider = package.module("provider")
    package.module("consumer")
    tracker = package.tracker()
    package.edit("provider", "def helper():\n    return 1\n", "VALUE = 1\n")

    with pytest.raises(ReloadError, match="helper"):
        package.reload_stale(tracker)

    assert provider.helper() == 1


def test_a_type_checking_import_does_not_reorder_the_reload(package):
    package.write(
        "b",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import {package}.a\n\nN = 1\n",
    )
    package.write("a", "from {package}.b import N\n\ndef f():\n    return N\n")
    a = package.module("a")
    tracker = package.tracker()
    package.edit("b", "N = 1", "N = 2")

    package.reload_stale(tracker)

    assert a.f() == 2


def test_an_import_inside_a_function_does_not_reorder_the_reload(package):
    package.write(
        "b", "N = 1\n\ndef later():\n    import {package}.a\n    return {package}.a\n"
    )
    package.write("a", "from {package}.b import N\n\ndef f():\n    return N\n")
    a = package.module("a")
    tracker = package.tracker()
    package.edit("b", "N = 1", "N = 2")

    package.reload_stale(tracker)

    assert a.f() == 2


def test_registrations_a_reload_drops_are_not_left_selectable(package):
    package.write(
        "ops",
        "import torch\n"
        "from tokenspeed_kernel.registry import register_kernel\n"
        "from tokenspeed_kernel.signature import format_signatures\n\n"
        "def _skip(*args, **kwargs):\n    return lambda fn: fn\n\n"
        "@register_kernel(\n    'hot_reload_test', 'op', solution='triton', name='{package}_op',\n"
        "    signatures=format_signatures(('x',), 'dense', {torch.float32}),\n)\n"
        "def op(x):\n    return x\n",
    )
    package.module("ops")
    tracker = package.tracker()
    package.edit("ops", "@register_kernel(", "@_skip(")

    package.reload_stale(tracker)

    assert KernelRegistry.get().get_impl(f"{package.name}_op") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_wrapped_kernel_keeps_its_inner_identity_and_its_warmup(package):
    package.write(
        "warm",
        "from tokenspeed_kernel._triton import tl, triton\n"
        "from tokenspeed_kernel.warmup import register_triton_warmup\n\n"
        "@triton.heuristics({'EVEN': lambda args: True})\n"
        "@triton.jit\ndef _kernel(x_ptr, EVEN: tl.constexpr, BLOCK: tl.constexpr):\n    pass\n\n"
        "register_triton_warmup(_kernel.fn, variants=({'BLOCK': (64,)},))\n",
    )
    warm = package.module("warm")
    inner = warm._kernel.fn
    tracker = package.tracker()
    package.edit("warm", "(64,)", "(128,)")

    package.reload_stale(tracker)

    assert warm._kernel.fn is inner
    assert _registered(warm.__name__) == {inner: ({"BLOCK": (128,)},)}
    tracker = package.tracker()
    package.edit("warm", "register_triton_warmup(", "dict(_kernel=")
    package.reload_stale(tracker)
    assert _registered(warm.__name__) == {}


def _registered(module: str) -> dict:
    return {
        kernel: variants
        for kernel, variants in warmup_registrations().items()
        if kernel.fn.__module__ == module
    }


def test_a_revert_puts_back_every_patched_object(package):
    package.write("kernels", KERNELS)
    package.write("user", USER)
    kernels, user = package.module("kernels"), package.module("user")
    layer = kernels.Layer()
    tracker = package.tracker()
    package.edit("kernels", "return x + 1", "return x + 100")
    package.edit("kernels", 'return "layer"', 'return "layer:" + super().who()')
    report = package.reload_stale(tracker)
    assert (user.use(1), layer.who()) == (101, "layer:base")

    report.revert()

    assert (user.use(1), layer.forward(1), layer.who()) == (2, 4, "layer")
    assert kernels.scaled(1) == 10


def test_an_unchanged_rewrite_is_not_stale_once_loaded(package):
    package.write("kernels", KERNELS)
    kernels = package.module("kernels")
    tracker = package.tracker()
    # A rewrite of the same text is newer, so it counts until the tracker knows its content.
    package.edit("kernels", "def f", "def f")
    assert tracker.stale() == [kernels.__name__]
    tracker.mark_loaded(tracker.read([kernels.__name__]))
    package.edit("kernels", "def f", "def f")
    assert tracker.stale() == []


TRITON = """
import torch
from tokenspeed_kernel._triton import tl, triton

@triton.jit
def _scale(x, ALPHA: tl.constexpr):
    return x * ALPHA

@triton.autotune(configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})], key=["n"])
@triton.jit
def _kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(y_ptr + offsets, _scale(tl.load(x_ptr + offsets, mask=mask), 2.0), mask=mask)

def run(x):
    y = torch.empty_like(x)
    _kernel[lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]),)](x, y, x.numel())
    return y
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_an_edited_triton_callee_recompiles_its_autotuned_caller(package):
    package.write("tk", TRITON)
    tk = package.module("tk")
    run, kernel = tk.run, tk._kernel
    x = torch.arange(256, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(run(x), x * 2)
    tracker = package.tracker()
    package.edit("tk", "return x * ALPHA", "return x * ALPHA + 1000.0")

    package.reload_stale(tracker)

    torch.testing.assert_close(run(x), x * 2 + 1000)
    assert tk._kernel is kernel


BARE_TRITON = """
from tokenspeed_kernel._triton import tl, triton

@triton.jit
def _add(x_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(x_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + 1.0, mask=mask)

def run(x):
    _add[(1,)](x, x.numel(), BLOCK=256)
    return x
"""

BARE_GLUON = """
from tokenspeed_kernel._triton import gl, gluon, triton

@gluon.jit
def _add(x_ptr, n, BLOCK: gl.constexpr, WARP: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [WARP], [4], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    mask = offsets < n
    gl.store(x_ptr + offsets, gl.load(x_ptr + offsets, mask=mask) + 1.0, mask=mask)

def run(x):
    # The layout spans one warp, 64 threads on AMD and 32 on NVIDIA.
    warp = triton.runtime.driver.active.get_current_target().warp_size
    _add[(1,)](x, x.numel(), BLOCK=256, WARP=warp, num_warps=4)
    return x
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("source", [BARE_TRITON, BARE_GLUON], ids=["triton", "gluon"])
def test_an_edited_bare_kernel_compiles_on_its_old_object_and_reverts(package, source):
    package.write("bare", source)
    bare = package.module("bare")
    kernel = bare._add
    x = torch.zeros(256, device="cuda")
    torch.testing.assert_close(bare.run(x.clone()), x + 1)
    tracker = package.tracker()
    package.edit("bare", "+ 1.0, mask=mask)", "+ 7.0, mask=mask)")

    report = package.reload_stale(tracker)

    # The old kernel object compiles the new source itself, here through a direct launch.
    assert bare._add is kernel
    torch.testing.assert_close(bare.run(x.clone()), x + 7)
    report.revert()
    torch.testing.assert_close(bare.run(x.clone()), x + 1)


CUTE_DSL = """
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def _add_kernel(a: cute.Tensor, b: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    b[tid] = a[tid] + cutlass.Float32(1.0)

@cute.jit
def _launch(a: cute.Tensor, b: cute.Tensor):
    _add_kernel(a, b).launch(grid=(1, 1, 1), block=(a.shape[0], 1, 1))

_COMPILED = {}

def run(a):
    b = torch.empty_like(a)
    ta, tb = from_dlpack(a), from_dlpack(b)
    if a.shape not in _COMPILED:
        _COMPILED[a.shape] = cute.compile(_launch, ta, tb)
    _COMPILED[a.shape](ta, tb)
    return b
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_an_edited_cute_dsl_kernel_recompiles_through_its_module_cache(package):
    pytest.importorskip("cutlass")
    package.write("ck", CUTE_DSL)
    package.write("caller", "from {package}.ck import run\n")
    ck, caller = package.module("ck"), package.module("caller")
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(caller.run(x), x + 1)
    tracker = package.tracker()
    package.edit("ck", "cutlass.Float32(1.0)", "cutlass.Float32(500.0)")

    report = package.reload_stale(tracker)

    torch.testing.assert_close(caller.run(x), x + 500)
    report.revert()
    torch.testing.assert_close(ck.run(x), x + 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
