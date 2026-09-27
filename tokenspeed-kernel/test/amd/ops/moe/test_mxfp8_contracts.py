# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU-only MXFP8 shape and layout contracts, without GPU imports."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_AMD = Path(__file__).resolve().parents[5] / "tokenspeed-kernel-amd"
_PACKAGE = "tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4"
_DIRECTORY = _AMD / "python" / Path(*_PACKAGE.split("."))


def _source_function(filename, name):
    module = ast.parse((_DIRECTORY / filename).read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@pytest.mark.parametrize("sorted_rows", [2**28 - 1, 2**28, 2**28 + 1])
def test_fused_quantizer_checks_rounded_sorted_extent_before_allocation(sorted_rows):
    function = _source_function("mxfp8_quantize.py", "quantize_mxfp8")
    # Run only the host wrapper with metadata stand-ins, never import torch/JIT.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )

    class AllocationReached(Exception):
        pass

    def allocate(*args, **kwargs):
        raise AllocationReached

    env = {
        "torch": SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8", empty=allocate),
        "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
    }
    exec(compile(ast.fix_missing_locations(module), "<quantizer-host>", "exec"), env)
    expected = AllocationReached if sorted_rows <= 2**28 else ValueError
    with pytest.raises(expected) as error:
        env["quantize_mxfp8"](
            SimpleNamespace(shape=(1, 256), dtype="bf16", device="unused"),
            SimpleNamespace(numel=lambda: sorted_rows),
            None,
            tokens=2**24,
            topk=1,
            slot_major=False,
            block_m=32,
        )
    if expected is ValueError:
        assert "sorted quantization" in str(error.value)
    else:
        assert ((sorted_rows * 8 + 127) // 128) * 128 - 1 <= 2**31 - 1


@pytest.mark.parametrize("k", [256, 3072, 3584])
def test_fused_scale_destinations_are_bijective(k):
    function = _source_function("mxfp8_quantize.py", "_quantize_sorted_mxfp8")
    assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "offset"
            for target in node.targets
        )
    )

    class RemoveScalarCast(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                return self.visit(node.func.value)
            return self.generic_visit(node)

    offset = ast.fix_missing_locations(
        RemoveScalarCast().visit(ast.Expression(assignment.value))
    )
    code = compile(offset, "<sorted-scale-offset>", "eval")
    destinations = [
        eval(code, {"row": row, "kg": kg, "K": k})
        for row in range(96)
        for kg in range(k // 32)
    ]
    assert sorted(destinations) == list(range(96 * (k // 32)))
    # The physical axes are [M32, K256, group%4, M16, K-half, M-half].
    canonical = [
        (row // 32, kg // 8, kg % 4, row % 16, kg % 8 // 4, row % 32 // 16)
        for row in range(96)
        for kg in range(k // 32)
    ]
    assert [
        canonical[i]
        for i in sorted(range(len(destinations)), key=destinations.__getitem__)
    ] == sorted(canonical)
    # q is stored only by value owners; sorted-route duplicates cannot race it.
    role = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "values_role"
        and any(
            isinstance(child, ast.Call) and ast.unparse(child.func) == "gl.store"
            for statement in node.body
            for child in ast.walk(statement)
        )
    )
    stores = [
        [
            ast.unparse(node.args[0])
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "gl.store"
        ]
        for body in (role.body, role.orelse)
    ]
    assert len(stores[0]) == len(stores[1]) == 1
    assert stores[0][0].startswith("q + ") and stores[1][0] == "output + offset"


@pytest.mark.parametrize("k", [256, 3072, 3584])
def test_fused_quantizer_uniform_scale_exit_preserves_partial_groups(k):
    function = _source_function("mxfp8_quantize.py", "_quantize_sorted_mxfp8")
    guard = next(
        node
        for node in function.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not values_role"
    )
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.If)
    condition = guard.body[0]
    assert len(condition.body) == 1 and isinstance(condition.body[0], ast.Return)
    assert (
        ast.unparse(condition.test)
        == "(pid - VALUE_BLOCKS) * 128 // (K // 32) >= valid_rows"
    )
    assert guard.lineno < next(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "values"
            for target in node.targets
        )
    )
    expression = compile(ast.Expression(condition.test), "<scale-cta-exit>", "eval")
    for block in (0, 1, 2, 3, 7, 2**24 - 1):
        first_group = block * 128
        assert first_group + 127 <= 2**31 - 1
        for valid_rows in (0, 1, 16, 17, 32, 96, 2**31 - 1):
            exit_taken = eval(
                expression,
                {
                    "pid": block + 13,
                    "VALUE_BLOCKS": 13,
                    "K": k,
                    "valid_rows": valid_rows,
                },
            )
            assert exit_taken == all(
                (first_group + lane) // (k // 32) >= valid_rows for lane in range(128)
            )


def test_bm32_bucket_and_runtime_shape_contract():
    function = _source_function("prefill_mxfp8.py", "mxfp8_situ_prefill")
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "block_m"
            for target in node.targets
        )
    )
    expression = compile(ast.Expression(assignment.value), "<tile-bucket>", "eval")
    assert [
        eval(expression, {"m": m})
        for m in (1, 4, 8, 15, 16, 32, 64, 65, 896, 1024, 1025, 8192)
    ] == [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 128, 128]
    for filename, name, runtime in (
        ("mxfp8_quantize.py", "_quantize_sorted_mxfp8", ("ROWS", "M", "VALUE_BLOCKS")),
        ("mxfp8_gemm.py", "_mxfp8_stage1", ("M",)),
        ("mxfp8_gemm.py", "_mxfp8_stage2", ("M",)),
        ("expert_mesh.py", "_scatter_mesh", ("M", "PITCH", "OUT_SIZE")),
    ):
        kernel = _source_function(filename, name)
        assert all(
            arg.annotation is None for arg in kernel.args.args if arg.arg in runtime
        )
        decorator = kernel.decorator_list[0]
        assert (
            ast.literal_eval(
                next(
                    keyword.value
                    for keyword in decorator.keywords
                    if keyword.arg == "do_not_specialize"
                )
            )
            == runtime
        )
