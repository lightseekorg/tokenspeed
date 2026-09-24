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

"""Isolation checks for the private FlashInfer routing adapter."""

import functools
import inspect

import pytest
from tokenspeed_kernel.thirdparty.flashinfer.trtllm_moe import (
    _clone,
    _entrypoints,
    _register_private,
)

_CLONE_VALUE = object()


def test_function_rebinding_does_not_mutate_upstream():
    sentinel = object()

    def original(x, *, value):
        return x, value, _CLONE_VALUE

    clone = _clone(original, {**original.__globals__, "_CLONE_VALUE": sentinel})
    assert clone(3, value=4) == (3, 4, sentinel)
    assert inspect.signature(clone) == inspect.signature(original)
    assert original(3, value=4) == (3, 4, _CLONE_VALUE)
    assert original.__globals__["_CLONE_VALUE"] is not sentinel


def test_operator_names_are_private():
    def register(name, *, mutates_args):
        return name, mutates_args

    assert _register_private(register, "flashinfer::moe", mutates_args=("out",)) == (
        "tokenspeed_flashinfer_route_init::moe",
        ("out",),
    )
    with pytest.raises(RuntimeError, match="Unexpected FlashInfer operator"):
        _register_private(register, "another::moe", mutates_args=())


def test_upstream_dispatch_and_caches_are_unchanged():
    core = pytest.importorskip("flashinfer.fused_moe.core")
    before = dict(vars(core))
    private = _entrypoints()
    assert vars(core) == before
    assert (
        private["get_trtllm_moe_sm100_module"] is not core.get_trtllm_moe_sm100_module
    )
    for name in ("trtllm_fp4_block_scale_moe", "trtllm_fp4_block_scale_routed_moe"):
        assert private[name].__globals__ is private
        assert inspect.signature(private[name]) == inspect.signature(
            getattr(core, name)
        )
    factory = private.get("_get_trtllm_moe_sm100_module_impl")
    if factory is not None:
        assert isinstance(factory, functools._lru_cache_wrapper)
        assert factory is not core._get_trtllm_moe_sm100_module_impl
