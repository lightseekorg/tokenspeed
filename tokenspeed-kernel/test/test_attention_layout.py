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

import ast
from pathlib import Path

_OPERATOR_VARIANTS = {
    "mha_": "mha",
    "rel_mha_": "rmha",
    "mla_": "mla",
    "dsa_": "dsa",
    "dsv4_": "dsv4",
    "kda_": "kda",
    "kpool_": "kpool",
    "msa_": "msa",
    "qsa_": "qsa",
    "gdn_": "gdn",
}


def _literal_attention_registrations(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    helpers = {}
    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        parameters = [argument.arg for argument in function.args.args]
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            callee = node.func
            callee_name = (
                callee.id
                if isinstance(callee, ast.Name)
                else callee.attr if isinstance(callee, ast.Attribute) else None
            )
            namespace, operator = node.args[:2]
            if (
                callee_name == "register_kernel"
                and isinstance(namespace, ast.Constant)
                and namespace.value == "attention"
                and isinstance(operator, ast.Name)
                and operator.id in parameters
            ):
                helpers[function.name] = parameters.index(operator.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            function_name = function.id
        elif isinstance(function, ast.Attribute):
            function_name = function.attr
        else:
            continue
        if function_name in helpers:
            operator_index = helpers[function_name]
            if len(node.args) > operator_index:
                operator = node.args[operator_index]
                if isinstance(operator, ast.Constant) and isinstance(
                    operator.value, str
                ):
                    yield operator.value, node.lineno
            continue
        if function_name != "register_kernel" or len(node.args) < 2:
            continue
        namespace, operator = node.args[:2]
        if (
            isinstance(namespace, ast.Constant)
            and namespace.value == "attention"
            and isinstance(operator, ast.Constant)
            and isinstance(operator.value, str)
        ):
            yield operator.value, node.lineno


def _operator_variant(operator: str) -> str | None:
    return next(
        (
            variant
            for prefix, variant in _OPERATOR_VARIANTS.items()
            if operator.startswith(prefix)
        ),
        None,
    )


def test_attention_implementations_are_grouped_by_variant():
    attention_dir = (
        Path(__file__).parents[1] / "python" / "tokenspeed_kernel" / "ops" / "attention"
    )
    package_dirs = {
        path.name
        for path in attention_dir.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    assert package_dirs == {
        "dsa",
        "dsv4",
        "gdn",
        "kda",
        "kpool",
        "mha",
        "mla",
        "msa",
        "qsa",
        "rmha",
    }

    implementations = {
        "cuda",
        "cute_dsl",
        "deep_gemm",
        "flashinfer",
        "gluon",
        "tokenspeed_mla",
        "triton",
    }
    assert {path.stem for path in attention_dir.glob("*.py")} == {
        "__init__",
        "cuda",
        "triton",
    }

    for variant in package_dirs:
        variant_dir = attention_dir / variant
        implementation_modules = {
            path.stem for path in variant_dir.glob("*.py") if path.name != "__init__.py"
        }
        assert implementation_modules <= implementations
        assert all(
            path.name.startswith("_")
            for path in variant_dir.iterdir()
            if path.is_dir() and (path / "__init__.py").is_file()
        )

        tree = ast.parse((variant_dir / "__init__.py").read_text())
        direct_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert {
            f"tokenspeed_kernel.ops.attention.{variant}.{implementation}"
            for implementation in implementation_modules
        } <= direct_imports

    assert not (attention_dir / "dsa" / "_cuda" / "__init__.py").exists()
    assert not (attention_dir / "gdn" / "_triton" / "linear" / "__init__.py").exists()
    assert not (attention_dir / "mla" / "_tokenspeed_mla" / "__init__.py").exists()


def test_attention_registrations_are_owned_by_their_variant():
    attention_dir = (
        Path(__file__).parents[1] / "python" / "tokenspeed_kernel" / "ops" / "attention"
    )
    for variant_dir in attention_dir.iterdir():
        if not variant_dir.is_dir() or not (variant_dir / "__init__.py").is_file():
            continue
        for path in variant_dir.rglob("*.py"):
            if path.name == "__init__.py":
                continue
            for operator, lineno in _literal_attention_registrations(path):
                assert _operator_variant(operator) == variant_dir.name, (
                    f"{path.relative_to(attention_dir)}:{lineno} registers "
                    f"{operator!r} outside its variant directory"
                )
