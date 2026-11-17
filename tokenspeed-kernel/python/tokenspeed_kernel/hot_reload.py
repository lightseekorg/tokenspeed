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

"""Reload edited Python modules into a running process, patching old objects in place so every reference runs the new code."""

from __future__ import annotations

import ast
import collections
import enum
import functools
import hashlib
import linecache
import os
import sys
import time
import types
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from tokenspeed_kernel._triton import _IS_NPU, triton
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.warmup import (
    rekey_warmup_registrations,
    remove_warmup_registrations,
    restore_warmup_registrations,
    warmup_registrations,
)

__all__ = [
    "ModuleSource",
    "ReloadError",
    "ReloadReport",
    "SourceTracker",
    "dependent_modules",
    "reload_modules",
]

_MISSING = object()
# Attributes the import system owns; re-execution leaves them in place.
_LOADER_KEYS = frozenset(
    {
        "__name__",
        "__file__",
        "__spec__",
        "__loader__",
        "__package__",
        "__path__",
        "__cached__",
        "__builtins__",
    }
)
# Class attributes that describe the class object rather than its definition.
_CLASS_OWN_KEYS = frozenset(
    {"__dict__", "__weakref__", "__module__", "__qualname__", "_abc_impl"}
)
_LRU_WRAPPER = type(functools.lru_cache()(lambda: None))
_KERNEL_TYPES: tuple[type, ...] = (
    () if _IS_NPU else (triton.runtime.jit.KernelInterface,)
)
# Containers whose elements a reload points back at the old objects.
_CONTAINERS = (tuple, list, dict, set, frozenset)
# Module global through which redirected functions reach their new definitions.
_TARGETS = "__hot_reload_targets__"


class ReloadError(RuntimeError):
    """A module failed to compile or execute; every module was restored."""


@dataclass(frozen=True)
class SourceBindings:
    """The names a module's source binds at top level (None behind a star import) and in each class body, by qualname."""

    module: frozenset[str] | None
    classes: dict[str, frozenset[str]]


@dataclass(frozen=True)
class ModuleSource:
    """The exact source one module is re-executed from, with the bindings of the source it replaces when known."""

    name: str
    path: str
    text: bytes
    mtime_ns: int
    previous: SourceBindings | None


@dataclass
class ReloadReport:
    """What a reload changed: modules re-executed, objects patched, objects left stale."""

    modules: list[str]
    patched: int = 0
    redirected: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    # Module data re-execution changed: state set at run time returns to its import-time value unless kept with globals().setdefault.
    reset: list[str] = field(default_factory=list)
    seconds: float = 0.0
    # The reverse of every change, applied last first by revert().
    undo: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def revert(self) -> None:
        """Undo the reload exactly: patched objects, module namespaces and kernel registrations return to their earlier state."""
        while self.undo:
            self.undo.pop()()


def _in_packages(name: str, packages: Sequence[str]) -> bool:
    return any(
        name == package or name.startswith(f"{package}.") for package in packages
    )


def _source_file(module: types.ModuleType) -> str | None:
    path = module.__dict__.get("__file__")
    return path if isinstance(path, str) and path.endswith(".py") else None


def _digest(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


class SourceTracker:
    """Tells which loaded modules under ``packages`` run older code than their source file."""

    def __init__(self, packages: Sequence[str], *, loaded_at: float) -> None:
        self.packages = tuple(packages)
        # A module never reloaded runs what its file held when the process started.
        self._started_ns = int(loaded_at * 1e9)
        self._mtime_ns: dict[str, int] = {}
        self._digest: dict[str, str] = {}
        self._bindings: dict[str, SourceBindings] = {}

    def remember_bindings(self, packages: Sequence[str]) -> None:
        """Record what the running sources of modules under ``packages`` bind, so a later edit that drops a name is seen."""
        for name, path in self.modules():
            if _in_packages(name, packages) and name not in self._bindings:
                try:
                    bindings = source_bindings(_read(path))
                except OSError:
                    continue
                if bindings is not None:
                    self._bindings[name] = bindings

    def modules(self) -> list[tuple[str, str]]:
        """``(name, path)`` of every loaded source module under the tracked packages, in import order."""
        tracked = []
        for name, module in list(sys.modules.items()):
            if module is None or not _in_packages(name, self.packages):
                continue
            path = _source_file(module)
            if path is not None:
                tracked.append((name, path))
        return tracked

    def stale(self) -> list[str]:
        """Names of tracked modules whose file changed since this process last executed it."""
        names = []
        for name, path in self.modules():
            try:
                mtime_ns = os.stat(path).st_mtime_ns
            except OSError:
                continue
            if mtime_ns <= self._mtime_ns.get(name, self._started_ns):
                continue
            known = self._digest.get(name)
            if known is not None and known == _digest(_read(path)):
                continue
            names.append(name)
        return names

    def read(self, names: Iterable[str]) -> list[ModuleSource]:
        """The current source of each named module, in the order given."""
        paths = dict(self.modules())
        return [
            _read_source(name, paths[name], self._bindings.get(name))
            for name in names
            if name in paths
        ]

    def mark_loaded(self, sources: Iterable[ModuleSource]) -> None:
        """Record that ``sources`` now run, so only later edits count as stale."""
        for source in sources:
            self._mtime_ns[source.name] = source.mtime_ns
            self._digest[source.name] = _digest(source.text)
            bindings = source_bindings(source.text)
            if bindings is not None:
                self._bindings[source.name] = bindings


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _read_source(name: str, path: str, previous: SourceBindings | None) -> ModuleSource:
    mtime_ns = os.stat(path).st_mtime_ns
    return ModuleSource(
        name=name, path=path, text=_read(path), mtime_ns=mtime_ns, previous=previous
    )


def _defining_module(value: Any) -> str | None:
    """The module that defined a function, class, cache wrapper or Triton kernel."""
    if isinstance(value, (types.FunctionType, type, _LRU_WRAPPER)):
        return value.__module__
    if _KERNEL_TYPES and isinstance(value, _KERNEL_TYPES):
        inner = _innermost(value)
        return inner.__module__ if isinstance(inner, types.FunctionType) else None
    return None


# path -> (mtime_ns, modules its source imports).
_IMPORTS: dict[str, tuple[int, frozenset[str]]] = {}


def _imported_modules(name: str, module: types.ModuleType) -> frozenset[str]:
    """The modules ``module``'s source imports, so a copied constant (``from p import N``) counts as a use."""
    path = _source_file(module)
    if path is None:
        return frozenset()
    try:
        mtime_ns = os.stat(path).st_mtime_ns
        cached = _IMPORTS.get(path)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]
        tree = ast.parse(_read(path))
    except (OSError, SyntaxError, ValueError):
        return frozenset()
    package = module.__dict__.get("__package__") or name.rpartition(".")[0]
    imported = set()
    for node in _executed_at_import(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = (
                    package.rsplit(".", node.level - 1)[0]
                    if node.level > 1
                    else package
                )
                base = f"{parent}.{base}" if base else parent
            imported.add(base)
            # ``from package import submodule`` imports the submodule.
            imported.update(f"{base}.{alias.name}" for alias in node.names)
    result = frozenset(imported)
    _IMPORTS[path] = (mtime_ns, result)
    return result


def _never_runs(test: ast.expr) -> bool:
    """Whether an ``if`` test is ``TYPE_CHECKING`` or ``False``, whose body module execution skips."""
    if isinstance(test, ast.Constant):
        return test.value is False
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _executed_at_import(tree: ast.AST) -> Iterator[ast.AST]:
    """The nodes a module's execution reaches; a function body's or a TYPE_CHECKING block's imports run later, if ever."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.If) and _never_runs(node.test):
            stack.extend(node.orelse)
            continue
        for child in ast.iter_child_nodes(node):
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                stack.append(child)


def source_bindings(text: bytes) -> SourceBindings | None:
    """What ``text`` binds at top level and in each class body; None when it does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    classes: dict[str, frozenset[str]] = {}
    return SourceBindings(module=_bound_names(tree.body, "", classes), classes=classes)


def _bound_names(
    body: list[ast.stmt], prefix: str, classes: dict[str, frozenset[str]]
) -> frozenset[str] | None:
    """The names ``body`` binds in its own scope, recording nested class bodies under ``prefix``; None behind a star import."""
    bound = set()
    star = False
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef):
            bound.add(node.name)
            qualname = f"{prefix}{node.name}"
            members = _bound_names(node.body, f"{qualname}.", classes)
            if members is not None:
                classes[qualname] = members
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    star = True
                else:
                    bound.add(alias.asname or alias.name.partition(".")[0])
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.Lambda):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return None if star else frozenset(bound)


def _used_modules(name: str, module: types.ModuleType) -> set[str]:
    """The modules whose objects or submodules ``module``'s namespace refers to, or whose names its source imports."""
    used = set(_imported_modules(name, module))
    for value in list(module.__dict__.values()):
        if isinstance(value, types.ModuleType):
            # A package holds its submodules as attributes without using them.
            if value.__name__.rpartition(".")[0] != name:
                used.add(value.__name__)
        else:
            defining = _defining_module(value)
            if defining is not None:
                used.add(defining)
    used.discard(name)
    return used


def dependent_modules(names: Iterable[str], packages: Sequence[str]) -> list[str]:
    """``names`` plus every loaded module under ``packages`` that uses them, transitively, providers before their users."""
    targets = set(names)
    uses = {
        name: _used_modules(name, module)
        for name, module in list(sys.modules.items())
        if module is not None
        and _in_packages(name, packages)
        and _source_file(module) is not None
    }
    grew = True
    while grew:
        grew = False
        for name, used in uses.items():
            if name not in targets and used & targets:
                targets.add(name)
                grew = True
    # sys.modules lists an importer before the modules its body imports, so order by use.
    pending = [name for name in list(sys.modules) if name in targets]
    ordered: list[str] = []
    while pending:
        ready = [
            name
            for name in pending
            if not (uses.get(name, set()) & (set(pending) - {name}))
        ]
        # A cycle keeps its import order.
        ordered.extend(ready or pending[:1])
        pending = [name for name in pending if name not in ordered]
    return ordered


def reload_modules(sources: Sequence[ModuleSource]) -> ReloadReport:
    """Re-execute ``sources`` in order and patch what they defined in place; on failure restore all of them."""
    started = time.perf_counter()
    modules = []
    for source in sources:
        module = sys.modules.get(source.name)
        if module is None:
            raise ReloadError(f"{source.name} is not loaded")
        modules.append(module)
    try:
        codes = [
            compile(source.text, source.path, "exec", dont_inherit=True)
            for source in sources
        ]
    except SyntaxError as error:
        raise ReloadError(f"{error.filename}:{error.lineno}: {error.msg}") from error
    saved = [dict(module.__dict__) for module in modules]
    reloading = {source.name for source in sources}
    registry = KernelRegistry.get()
    registered = registry.snapshot()
    warmups = warmup_registrations()
    jit_functions = {} if _IS_NPU else triton.runtime.jit._triton_jit_function_registry
    named_kernels = dict(jit_functions)

    def restore_namespaces() -> None:
        for module, namespace in zip(modules, saved):
            module.__dict__.clear()
            module.__dict__.update(namespace)
        registry.restore(registered)
        restore_warmup_registrations(warmups)
        jit_functions.clear()
        jit_functions.update(named_kernels)
        linecache.checkcache()

    imported = set(sys.modules)
    current = ""
    try:
        for source, module, code in zip(sources, modules, codes):
            current = source.name
            exec(code, module.__dict__)
    except BaseException as error:
        restore_namespaces()
        # A module the failed source imported first is dropped, so a retry imports it afresh.
        for name in set(sys.modules) - imported:
            del sys.modules[name]
        raise ReloadError(f"{current}: {type(error).__name__}: {error}") from error
    report = ReloadReport(modules=[source.name for source in sources])
    report.undo.append(restore_namespaces)
    try:
        # A registration the edited source no longer makes is not left selectable.
        dropped = [
            name
            for name, impl in registered.impls.items()
            if _defining_module(impl) in reloading and registry.get_impl(name) is impl
        ]
        patcher = _Patcher(
            report,
            {
                source.name: (source.previous, source_bindings(source.text))
                for source in sources
            },
        )
        for module, namespace in zip(modules, saved):
            patcher.pair(module, namespace)
        patcher.apply()
        for module, namespace in zip(modules, saved):
            patcher.remove_deleted(module, namespace)
        for module in modules:
            patcher.rebind(module.__dict__)
        for module, namespace in zip(modules, saved):
            report.reset.extend(_reset_data(module, namespace))
        # A fresh import of a module importing a deleted definition fails; so does this reload.
        for module in modules:
            for name, value in module.__dict__.items():
                if id(value) in patcher.removed:
                    raise ReloadError(
                        f"{module.__name__} still binds {name}, which "
                        f"{patcher.removed[id(value)]} no longer defines"
                    )
        registry.replace_impls(patcher.replacements)
        for name in dropped:
            registry.unregister(name)
        moved = rekey_warmup_registrations(patcher.replacements)
        remove_warmup_registrations(
            kernel
            for kernel in warmups
            if _defining_module(kernel) in reloading and id(kernel) not in moved
        )
    except BaseException as error:
        report.revert()
        raise ReloadError(
            f"patching failed: {type(error).__name__}: {error}"
        ) from error
    linecache.checkcache()
    report.seconds = time.perf_counter() - started
    return report


class _Patcher:
    """Updates old objects in place from their new definitions, recording how to undo each change in ``report``."""

    def __init__(
        self,
        report: ReloadReport,
        bindings: dict[str, tuple[SourceBindings | None, SourceBindings | None]],
    ) -> None:
        self.report = report
        # Per module: what the replaced source and the new one bind.
        self.bindings = bindings
        # id(definition the edit deleted) -> its module.
        self.removed: dict[int, str] = {}
        self.pairs: list[tuple[Any, Any]] = []
        # id(new object) -> the old object that now stands for it.
        self.replacements: dict[int, Any] = {}
        # Keeps the new objects alive so their ids stay unique.
        self._new: list[Any] = []

    def pair(self, module: types.ModuleType, old_namespace: dict[str, Any]) -> None:
        """Pair each object ``module`` defines anew under a name with the one that name defined before."""
        replaced = []
        for name, new in list(module.__dict__.items()):
            old = old_namespace.get(name, _MISSING)
            if name in _LOADER_KEYS or old is _MISSING or old is new:
                continue
            if id(new) in self.replacements or any(
                old is done for done, _ in self.pairs
            ):
                continue
            # An alias (``backend = Fast``) is rebound, never patched: only a definition pairs.
            if _defined_as(old, module.__name__, name):
                if type(old) is type(new) and _defined_as(new, module.__name__, name):
                    self.pairs.append((old, new))
                    self.replacements[id(new)] = old
                    self._new.append(new)
                else:
                    replaced.append(old)
        self.report.stale.extend(_describe(old) for old in replaced)

    def apply(self) -> None:
        for old, new in self.pairs:
            if not self._patch(old, new):
                # Names keep the new object; only references taken earlier stay stale.
                del self.replacements[id(new)]
                self.report.stale.append(_describe(old))

    def remove_deleted(
        self, module: types.ModuleType, old_namespace: dict[str, Any]
    ) -> None:
        """Drop the definitions the edited source stopped binding, and report its dropped data, which may be bound at run time."""
        previous, current = self.bindings[module.__name__]
        if current is None or current.module is None:
            return
        for name, old in old_namespace.items():
            if name in _LOADER_KEYS or name in current.module:
                continue
            if module.__dict__.get(name, _MISSING) is not old:
                continue
            if _defined_as(old, module.__name__, name):
                del module.__dict__[name]
                self.removed[id(old)] = module.__name__
                self.report.patched += 1
            elif previous is not None and previous.module and name in previous.module:
                self.report.stale.append(
                    f"{module.__name__}.{name} (deleted from the source, kept)"
                )

    def rebind(self, namespace: dict[str, Any]) -> None:
        """Point a namespace's names, and the containers it binds, at the old objects that now hold the new definitions."""
        for name, value in list(namespace.items()):
            old = self.replacements.get(id(value), _MISSING)
            if old is not _MISSING:
                namespace[name] = old
            elif name != _TARGETS and type(value) in _CONTAINERS:
                # Redirect targets stay the new definitions the trampolines call.
                namespace[name] = self._rebind_container(value, depth=2)

    def _rebind_container(self, value: Any, *, depth: int) -> Any:
        """``value`` holding the old objects in place of new definitions: lists, dicts and sets in place, tuples rebuilt only when one changes."""

        def item(element: Any) -> Any:
            if depth > 1 and type(element) in _CONTAINERS:
                return self._rebind_container(element, depth=depth - 1)
            return self._translate(element)

        if type(value) is list:
            value[:] = [item(element) for element in value]
        elif type(value) is dict:
            # Keys too: a dict keyed by the module's class looks up the old object.
            items = [(self._translate(k), item(e)) for k, e in value.items()]
            value.clear()
            value.update(items)
        elif type(value) is set:
            elements = {item(element) for element in value}
            value.clear()
            value.update(elements)
        else:
            elements = [item(element) for element in value]
            if any(new is not old for new, old in zip(elements, value)):
                return type(value)(elements)
        return value

    def _translate(self, value: Any) -> Any:
        return self.replacements.get(id(value), value)

    def _patch(self, old: Any, new: Any) -> bool:
        if isinstance(new, types.FunctionType):
            return self._patch_function(old, new)
        if isinstance(new, type):
            self._patch_class(old, new)
            return True
        if isinstance(new, _LRU_WRAPPER):
            old.cache_clear()
            # Results cached under the new code must not outlive a revert either.
            self.report.undo.append(old.cache_clear)
            return self._patch(old.__wrapped__, new.__wrapped__)
        if isinstance(new, (staticmethod, classmethod)):
            return self._patch(old.__func__, new.__func__)
        if isinstance(new, functools.cached_property):
            return self._patch(old.func, new.func)
        if isinstance(new, property):
            return all(
                self._patch(old_part, new_part)
                for old_part, new_part in (
                    (old.fget, new.fget),
                    (old.fset, new.fset),
                    (old.fdel, new.fdel),
                )
                if old_part is not None or new_part is not None
            )
        if _KERNEL_TYPES and isinstance(new, _KERNEL_TYPES):
            self._patch_kernel(old, new)
            return True
        return False

    def _patch_function(self, old: Any, new: Any) -> bool:
        if type(old) is not type(new):
            return False
        self._retarget_class_cell(new)
        self._save_function(old)
        if old.__code__.co_freevars != new.__code__.co_freevars:
            self._redirect(old, new)
            return True
        old.__code__ = new.__code__
        old.__defaults__ = (
            None
            if new.__defaults__ is None
            else tuple(self._translate(value) for value in new.__defaults__)
        )
        old.__kwdefaults__ = new.__kwdefaults__
        old.__annotations__ = new.__annotations__
        old.__name__ = new.__name__
        old.__qualname__ = new.__qualname__
        state = {key: self._translate(value) for key, value in new.__dict__.items()}
        old.__dict__.clear()
        old.__dict__.update(state)
        for old_cell, new_cell in zip(old.__closure__ or (), new.__closure__ or ()):
            try:
                value = new_cell.cell_contents
            except ValueError:
                continue
            old_cell.cell_contents = self._translate(value)
        self.report.patched += 1
        return True

    def _save_function(self, function: Any) -> None:
        """Record how to put ``function`` back as it is now."""
        code, defaults, kwdefaults = (
            function.__code__,
            function.__defaults__,
            function.__kwdefaults__,
        )
        annotations, name, qualname = (
            function.__annotations__,
            function.__name__,
            function.__qualname__,
        )
        state = dict(function.__dict__)
        cells = list(function.__closure__ or ())
        values = [_cell_value(cell) for cell in cells]

        def restore() -> None:
            function.__code__ = code
            function.__defaults__ = defaults
            function.__kwdefaults__ = kwdefaults
            function.__annotations__ = annotations
            function.__name__ = name
            function.__qualname__ = qualname
            function.__dict__.clear()
            function.__dict__.update(state)
            for cell, value in zip(cells, values):
                if value is not _MISSING:
                    cell.cell_contents = value

        self.report.undo.append(restore)

    def _redirect(self, old: Any, new: Any) -> None:
        """Make an old function call its new definition, whose closure differs in shape."""
        targets = old.__globals__.setdefault(_TARGETS, {})
        key = f"{old.__qualname__}@{id(old):x}"
        self.report.undo.append(
            functools.partial(_restore_item, targets, key, targets.get(key, _MISSING))
        )
        targets[key] = new
        # The trampoline keeps the old closure's length, which its code must match, but never reads it.
        cells = [f"_cell{index}" for index in range(len(old.__closure__ or ()))]
        source = "def _factory():\n"
        source += "".join(f"    {cell} = None\n" for cell in cells)
        source += "    def _(*args, **kwargs):\n"
        if cells:
            source += f"        if False:\n            ({', '.join(cells)},)\n"
        source += f"        return {_TARGETS}[{key!r}](*args, **kwargs)\n    return _\n"
        namespace: dict[str, Any] = {}
        exec(
            compile(source, f"<reload {old.__qualname__}>", "exec"),
            old.__globals__,
            namespace,
        )
        old.__code__ = namespace["_factory"]().__code__
        old.__defaults__ = None
        old.__kwdefaults__ = None
        old.__dict__.update(new.__dict__)
        self.report.redirected.append(_describe(old))

    def _retarget_class_cell(self, value: Any) -> None:
        """Point a new method's ``super()`` cell, through any descriptor wrapping it, at the old class its instances still have."""
        if isinstance(value, (staticmethod, classmethod)):
            self._retarget_class_cell(value.__func__)
            return
        if isinstance(value, property):
            for accessor in (value.fget, value.fset, value.fdel):
                if accessor is not None:
                    self._retarget_class_cell(accessor)
            return
        if isinstance(value, functools.cached_property):
            self._retarget_class_cell(value.func)
            return
        if isinstance(value, _LRU_WRAPPER):
            self._retarget_class_cell(value.__wrapped__)
            return
        if not isinstance(value, types.FunctionType):
            return
        function = value
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or ()
        ):
            if name != "__class__":
                continue
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            cell.cell_contents = self._translate(value)

    def _patch_class(self, old: type, new: type) -> None:
        members_only = issubclass(new, enum.Enum)
        before, after = self.bindings.get(new.__module__, (None, None))
        bound_before = (
            before.classes.get(old.__qualname__, frozenset())
            if before is not None
            else frozenset()
        )
        bound_now = after.classes.get(new.__qualname__) if after is not None else None
        bases = tuple(self._translate(base) for base in new.__bases__)
        if bases != old.__bases__:
            previous = old.__bases__
            try:
                old.__bases__ = bases
            except TypeError:
                self.report.stale.append(f"{_describe(old)} (base classes)")
            else:
                self.report.undo.append(
                    functools.partial(_restore_attribute, old, "__bases__", previous)
                )
        for name, member in list(old.__dict__.items()):
            if name in new.__dict__ or name in _CLASS_OWN_KEYS:
                continue
            # A member the class body stopped binding no longer shadows the base class's.
            deleted = (
                bound_now is not None and name in bound_before and name not in bound_now
            )
            if deleted or _defined_in_body(member, old, name):
                try:
                    delattr(old, name)
                except (AttributeError, TypeError):
                    # An enum member, for one, cannot be deleted.
                    self.report.stale.append(f"{_describe(old)}.{name}")
                    continue
                self.report.undo.append(
                    functools.partial(_restore_attribute, old, name, member)
                )
                self.report.patched += 1
        for name, value in list(new.__dict__.items()):
            if name in _CLASS_OWN_KEYS:
                continue
            if isinstance(
                value, (types.MemberDescriptorType, types.GetSetDescriptorType)
            ):
                continue
            current = old.__dict__.get(name, _MISSING)
            if current is value:
                continue
            # Only a member defined in the class body is patched; a reference is rebound.
            member = f"{new.__qualname__}.{name}"
            if (
                current is not _MISSING
                and type(current) is type(value)
                and _defined_as(current, new.__module__, member)
                and _defined_as(value, new.__module__, member)
                and self._patch(current, value)
            ):
                continue
            if members_only and not isinstance(value, types.FunctionType):
                continue
            replacement = self._translate(value)
            self._retarget_class_cell(replacement)
            try:
                setattr(old, name, replacement)
            except (AttributeError, TypeError):
                self.report.stale.append(f"{old.__module__}.{old.__qualname__}.{name}")
                continue
            self.report.undo.append(
                functools.partial(_restore_attribute, old, name, current)
            )
            self.report.patched += 1

    def _patch_kernel(self, old: Any, new: Any) -> None:
        # A wrapper's inner kernel (heuristics around a JIT function) keeps its identity too.
        if (
            isinstance(old.fn, _KERNEL_TYPES)
            and type(old.fn) is type(new.fn)
            and id(new.fn) not in self.replacements
        ):
            self.replacements[id(new.fn)] = old.fn
            self._new.append(new.fn)
            self._patch_kernel(old.fn, new.fn)
        saved = dict(old.__dict__)
        self.report.undo.append(functools.partial(_restore_state, old, saved))
        state = {
            key: self._translate(_adopt(value, new, old))
            for key, value in new.__dict__.items()
        }
        old.__dict__.clear()
        old.__dict__.update(state)
        if isinstance(new, triton.runtime.jit.JITFunction):
            # Triton resolves kernels by name through this table; keep it on the old object.
            jit_functions = triton.runtime.jit._triton_jit_function_registry
            key = f"{new.module}:{new.fn.__qualname__}"
            if jit_functions.get(key) is new:
                jit_functions[key] = old
        self.report.patched += 1


def _cell_value(cell: types.CellType) -> Any:
    try:
        return cell.cell_contents
    except ValueError:
        return _MISSING


def _restore_item(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is _MISSING:
        mapping.pop(key, None)
    else:
        mapping[key] = value


def _restore_attribute(owner: type, name: str, value: Any) -> None:
    if value is _MISSING:
        delattr(owner, name)
    else:
        setattr(owner, name, value)


def _restore_state(owner: Any, state: dict[str, Any]) -> None:
    owner.__dict__.clear()
    owner.__dict__.update(state)


def _adopt(value: Any, new: Any, old: Any) -> Any:
    """Rebind state ``new`` built around itself, such as Triton's binder factory, to ``old``."""
    if isinstance(value, types.MethodType) and value.__self__ is new:
        return types.MethodType(value.__func__, old)
    if (
        isinstance(value, collections.defaultdict)
        and isinstance(value.default_factory, types.MethodType)
        and value.default_factory.__self__ is new
    ):
        value.default_factory = types.MethodType(value.default_factory.__func__, old)
    return value


def _innermost(value: Any) -> Any:
    """The Python function a cache wrapper, method wrapper, property or Triton kernel finally wraps."""
    while True:
        if isinstance(value, _LRU_WRAPPER):
            value = value.__wrapped__
        elif isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        elif isinstance(value, functools.cached_property):
            value = value.func
        elif isinstance(value, property) and value.fget is not None:
            value = value.fget
        elif _KERNEL_TYPES and isinstance(value, _KERNEL_TYPES):
            value = value.fn
        else:
            return value


def _reset_data(module: types.ModuleType, old_namespace: dict[str, Any]) -> list[str]:
    """The data names whose value re-executing ``module`` changed; objects compared only by identity are left out."""
    reset = []
    for name, old in old_namespace.items():
        new = module.__dict__.get(name, _MISSING)
        if name in _LOADER_KEYS or new is old or new is _MISSING:
            continue
        if isinstance(old, (types.ModuleType, type)) or callable(old) and callable(new):
            continue
        if type(old) is type(new):
            if type(old).__eq__ is object.__eq__:
                continue
            try:
                if bool(old == new):
                    continue
            except Exception:
                pass
        reset.append(f"{module.__name__}.{name}")
    return reset


def _defined_as(value: Any, module: str, qualname: str) -> bool:
    """Whether ``value`` is the function, class or kernel ``module`` defines as ``qualname``, possibly through a decorator's wrapper."""
    return _owns(_innermost(value), module, qualname, depth=0)


def _owns(inner: Any, module: str, qualname: str, *, depth: int) -> bool:
    if not isinstance(inner, (types.FunctionType, type)):
        return False
    if inner.__module__ == module and inner.__qualname__ == qualname:
        return True
    # A wrapper a decorator built without functools.wraps owns the definition it closes over.
    if (
        depth < 2
        and isinstance(inner, types.FunctionType)
        and "<locals>" in inner.__qualname__
    ):
        return any(
            _owns(_innermost(cell), module, qualname, depth=depth + 1)
            for cell in map(_cell_value, inner.__closure__ or ())
            if isinstance(cell, types.FunctionType)
        )
    return False


def _defined_in_body(value: Any, owner: type, name: str) -> bool:
    """Whether ``value`` is the member ``owner``'s class body defines as ``name``, a lambda included."""
    return any(
        _defined_as(value, owner.__module__, f"{owner.__qualname__}.{member}")
        for member in (name, "<lambda>")
    )


def _describe(value: Any) -> str:
    inner = _innermost(value)
    if isinstance(inner, (types.FunctionType, type)):
        return f"{inner.__module__}.{inner.__qualname__}"
    return type(value).__qualname__
