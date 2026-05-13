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

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch
import torch.library
from torch.library import Library

tokenspeed_lib = Library("tokenspeed", "FRAGMENT")


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str],
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
) -> None:
    """Register a low-overhead torch custom op in the TokenSpeed namespace."""

    target = target_lib or tokenspeed_lib
    lib_name = getattr(getattr(target, "m", None), "name", "tokenspeed")
    try:
        if hasattr(torch.ops, lib_name) and hasattr(
            getattr(torch.ops, lib_name), op_name
        ):
            return
    except (AttributeError, RuntimeError):
        pass

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        custom_op_impl = importlib.import_module("torch._custom_op.impl")
        schema_str = custom_op_impl.infer_schema(op_func, mutates_args)

    target.define(op_name + schema_str)
    target.impl(op_name, op_func, "CUDA")
    if fake_impl is not None:
        target._register_fake(op_name, fake_impl)
