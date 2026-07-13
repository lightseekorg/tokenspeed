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

"""Structural guards for single-read ExecutionPlan operation views."""

from __future__ import annotations

import ast
import pathlib


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    event_loop = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EventLoop"
    )
    return next(
        node
        for node in event_loop.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _self_calls(method: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == name
    ]


def test_event_loops_materialize_forward_once_and_reuse_it_for_cache_submit():
    root = pathlib.Path(__file__).resolve().parents[2]
    source = (root / "python/tokenspeed/runtime/engine/event_loop.py").read_text()
    tree = ast.parse(source)

    for method_name in ("event_loop", "event_loop_overlap"):
        method = _method(tree, method_name)
        get_forward_calls = _self_calls(method, "_get_forward_op")
        submit_calls = _self_calls(method, "_submit_cache_ops")

        assert len(get_forward_calls) == 1
        assert len(submit_calls) == 1
        assert len(submit_calls[0].args) == 2
        assert isinstance(submit_calls[0].args[1], ast.Name)
        assert submit_calls[0].args[1].id == "forward_op"


def test_cache_submit_does_not_rematerialize_forward_view():
    root = pathlib.Path(__file__).resolve().parents[2]
    source = (root / "python/tokenspeed/runtime/engine/event_loop.py").read_text()
    method = _method(ast.parse(source), "_submit_cache_ops")

    assert not _self_calls(method, "_get_forward_op")
