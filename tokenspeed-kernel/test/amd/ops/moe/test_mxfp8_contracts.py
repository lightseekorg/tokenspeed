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

"""CPU-only MXFP8 host-wrapper contracts, without GPU imports.

The kernel layout and compile-reuse contracts run natively in
``test_gluon_mxfp8_prefill_gfx950.py``.
"""

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
