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

"""Every in-tree kernel is registered under its Python function's name, and
that name starts with the kernel's solution.

The registry name is what ``override=`` strings, ``describe_kernel`` output
and profiler scopes show, so it must lead straight back to the function that
implements it, and the backend must be readable off the name alone. The one
twist is ``solution="reference"``: that marks the PyTorch ground-truth role
the selector and numerics CLI special-case, and those kernels are named after
their implementation, ``torch_``. The check is static so it covers every
vendor's registrations regardless of the host platform; a runtime pass over
whatever the host did load double-checks the registered callables.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

_PACKAGE_ROOT = Path(__file__).parents[1] / "python" / "tokenspeed_kernel"
_SCANNED_TREES = ("ops", "numerics/reference")
# Kernels registered with this solution are named after their implementation.
_NAME_PREFIXES = {"reference": "torch"}


@dataclass(frozen=True)
class _Registration:
    path: Path
    lineno: int
    name: str | None
    solution: str | None
    function: str | None

    @property
    def location(self) -> str:
        return f"{self.path.relative_to(_PACKAGE_ROOT)}:{self.lineno}"


@dataclass(frozen=True)
class _Helper:
    """A module function whose body registers through ``register_kernel``.

    ``name`` and ``solution`` are either a string literal or the helper
    parameter that supplies them. ``decorated_parameter`` is set when the
    helper applies the registration to one of its own parameters (used as a
    bare ``@helper``); otherwise it returns the decorator (used as
    ``@helper(...)``).
    """

    parameters: tuple[str, ...]
    name: str | None
    name_parameter: str | None
    solution: str | None
    solution_parameter: str | None
    decorated_parameter: str | None


def _expected_prefix(solution: str) -> str:
    return _NAME_PREFIXES.get(solution, solution) + "_"


def _callee_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_register_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _callee_name(node.func) == "register_kernel"


def _keyword(call: ast.Call, keyword: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


def _literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _parameter(node: ast.expr | None, parameters: tuple[str, ...]) -> str | None:
    if isinstance(node, ast.Name) and node.id in parameters:
        return node.id
    return None


def _function_parameters(function: ast.FunctionDef) -> tuple[str, ...]:
    args = function.args
    return tuple(
        argument.arg for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs)
    )


def _helpers(tree: ast.Module) -> tuple[dict[str, _Helper], set[ast.Call]]:
    """Find registration helpers and the ``register_kernel`` calls they own."""
    helpers: dict[str, _Helper] = {}
    owned: set[ast.Call] = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        parameters = _function_parameters(function)
        for node in ast.walk(function):
            if isinstance(node, ast.Return) and _is_register_call(node.value):
                call = node.value
                decorated = None
            elif (
                isinstance(node, ast.Call)
                and _is_register_call(node.func)
                and len(node.args) == 1
                and _parameter(node.args[0], parameters) is not None
            ):
                call = node.func
                decorated = node.args[0].id
            else:
                continue
            owned.add(call)
            name_node = _keyword(call, "name")
            solution_node = _keyword(call, "solution")
            helpers[function.name] = _Helper(
                parameters=parameters,
                name=_literal(name_node),
                name_parameter=_parameter(name_node, parameters),
                solution=_literal(solution_node),
                solution_parameter=_parameter(solution_node, parameters),
                decorated_parameter=decorated,
            )
    return helpers, owned


def _argument(call: ast.Call, parameter: str, parameters: tuple[str, ...]) -> str | None:
    """The string literal a call site passes for ``parameter``."""
    keyword = _keyword(call, parameter)
    if keyword is not None:
        return _literal(keyword)
    index = parameters.index(parameter)
    if index < len(call.args):
        return _literal(call.args[index])
    return None


def _resolve_helper(
    helper: _Helper, call: ast.Call | None
) -> tuple[str | None, str | None]:
    name = helper.name
    solution = helper.solution
    if call is not None:
        if helper.name_parameter is not None:
            name = _argument(call, helper.name_parameter, helper.parameters)
        if helper.solution_parameter is not None:
            solution = _argument(call, helper.solution_parameter, helper.parameters)
    return name, solution


def _registrations(path: Path) -> list[_Registration]:
    tree = ast.parse(path.read_text(), filename=str(path))
    helpers, owned = _helpers(tree)
    found: list[_Registration] = []
    attributed: set[ast.Call] = set()

    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for decorator in function.decorator_list:
            if _is_register_call(decorator):
                attributed.add(decorator)
                found.append(
                    _Registration(
                        path,
                        decorator.lineno,
                        _literal(_keyword(decorator, "name")),
                        _literal(_keyword(decorator, "solution")),
                        function.name,
                    )
                )
                continue
            if isinstance(decorator, ast.Call):
                helper = helpers.get(_callee_name(decorator.func) or "")
                if helper is None or helper.decorated_parameter is not None:
                    continue
                name, solution = _resolve_helper(helper, decorator)
            elif isinstance(decorator, ast.Name):
                helper = helpers.get(decorator.id)
                if helper is None or helper.decorated_parameter is None:
                    continue
                name, solution = _resolve_helper(helper, None)
            else:
                continue
            found.append(
                _Registration(path, decorator.lineno, name, solution, function.name)
            )

    for node in ast.walk(tree):
        # ``register_kernel(...)(fn)`` applied to a module-level function.
        if not (
            isinstance(node, ast.Call)
            and _is_register_call(node.func)
            and node.func not in owned
        ):
            continue
        attributed.add(node.func)
        target = node.args[0] if len(node.args) == 1 else None
        found.append(
            _Registration(
                path,
                node.lineno,
                _literal(_keyword(node.func, "name")),
                _literal(_keyword(node.func, "solution")),
                target.id if isinstance(target, ast.Name) else None,
            )
        )

    for node in ast.walk(tree):
        if _is_register_call(node) and node not in attributed and node not in owned:
            found.append(_Registration(path, node.lineno, None, None, None))
    return found


def _scanned_files() -> list[Path]:
    return sorted(
        path
        for tree in _SCANNED_TREES
        for path in (_PACKAGE_ROOT / tree).rglob("*.py")
    )


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: str(p.relative_to(_PACKAGE_ROOT)))
def test_registered_kernel_names_match_functions_and_solutions(path: Path) -> None:
    problems = []
    for registration in _registrations(path):
        if registration.function is None:
            problems.append(
                f"{registration.location}: cannot tell which function this "
                "registration applies to; register a named function directly"
            )
            continue
        if registration.name is None or registration.solution is None:
            problems.append(
                f"{registration.location}: name= and solution= must be string "
                f"literals (or literals passed through a registration helper) "
                f"for {registration.function}"
            )
            continue
        if registration.name != registration.function:
            problems.append(
                f"{registration.location}: kernel {registration.name!r} is "
                f"implemented by {registration.function!r}; the two must match"
            )
        prefix = _expected_prefix(registration.solution)
        if not registration.name.startswith(prefix):
            problems.append(
                f"{registration.location}: kernel {registration.name!r} must "
                f"start with {prefix!r} (solution {registration.solution!r})"
            )
    assert not problems, "\n".join(problems)


def test_scan_covers_the_registration_forms_in_tree() -> None:
    """The static scan must actually see the shapes registrations take."""
    seen_forms = set()
    for path in _scanned_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        helpers, _ = _helpers(tree)
        if any(helper.decorated_parameter is not None for helper in helpers.values()):
            seen_forms.add("decorator-helper")
        if any(helper.decorated_parameter is None for helper in helpers.values()):
            seen_forms.add("factory-helper")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_register_call(node.func):
                seen_forms.add("functional")
            if isinstance(node, ast.FunctionDef) and any(
                _is_register_call(decorator) for decorator in node.decorator_list
            ):
                seen_forms.add("decorator")
    assert seen_forms == {
        "decorator",
        "decorator-helper",
        "factory-helper",
        "functional",
    }


def test_loaded_kernels_are_registered_under_their_callable_names() -> None:
    load_builtin_kernels()
    registry = KernelRegistry.get()
    mismatched = []
    for spec in registry.list_kernels():
        impl = registry.get_impl(spec.name)
        module = getattr(impl, "__module__", "") or ""
        # Tests register throwaway kernels into the singleton; only judge the
        # in-tree ones.
        if not module.startswith("tokenspeed_kernel."):
            continue
        if impl.__name__ != spec.name:
            mismatched.append(f"{spec.name} -> {module}.{impl.__name__}")
        prefix = _expected_prefix(spec.solution)
        if not spec.name.startswith(prefix):
            mismatched.append(f"{spec.name} lacks prefix {prefix!r}")
    assert not mismatched, "\n".join(mismatched)
