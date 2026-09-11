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

"""The boot path's capture step: where it is called and what it covers.

The checks read source text instead of importing the runtime: what moved is
boot sequencing, and a sequencing test should not need the model to load.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys

# Executed as a script by run_ci_suite: the test dir must be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

_RUNTIME = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "tokenspeed"
    / "runtime"
    / "execution"
)
_STEP_OPERATIONS = frozenset({"_autotune", "freeze", "capture", "capture_graphs"})


def _tree(name: str) -> ast.Module:
    return ast.parse((_RUNTIME / name).read_text())


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _attribute_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    ]


def test_the_boot_path_runs_the_capture_step():
    """The one line that keeps boot behaviour unchanged has to be asserted.

    Construction no longer tunes, freezes or captures, so ``build_device_side``
    calling ``capture_graphs`` is the whole of the change's boot contract. With
    that call gone the engine boots untuned, with an unfrozen workspace, no
    graphs and no post-startup seed -- and the first decode dies inside
    ``ForwardStepRunner`` on a graph key that was never captured, because
    ``_can_use_graph`` answers from ``bs <= max_capture_bs``, not from
    membership in ``self.graphs``.
    """
    builder = _function(_tree("device.py"), "build_device_side")
    calls = [
        call
        for call in _attribute_calls(builder)
        if call.func.attr == "capture_graphs"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "executor"
    ]
    assert len(calls) == 1, calls


def test_the_capture_step_precedes_the_per_rank_memory_summary():
    """The summary reads ``memory_allocated`` at the instant it runs.

    Graph memory only appears in the ``activations + graphs`` row if the
    graphs are already captured, so the step has to sit above the summary.
    """
    builder = _function(_tree("device.py"), "build_device_side")
    capture = [
        call.lineno
        for call in _attribute_calls(builder)
        if call.func.attr == "capture_graphs"
    ]
    summary = [
        call.lineno
        for call in ast.walk(builder)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "log_gpu_memory_summary"
    ]
    assert capture and summary
    assert max(capture) < min(summary), (capture, summary)


def test_nothing_the_constructor_reaches_tunes_freezes_or_captures():
    """Transitive, so the seam cannot be closed behind one helper.

    Checking ``__init__``'s own call names only rules out the literal move
    back; a ``self._install_graphs()`` added to the constructor reads as an
    unrelated attribute call and would pass. Follow every ``self.<method>``
    the constructor can reach instead.
    """
    tree = _tree("model_executor.py")
    executor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelExecutor"
    )
    methods = {
        node.name: node for node in executor.body if isinstance(node, ast.FunctionDef)
    }

    reached: set[str] = set()
    offenders: list[str] = []
    frontier = ["__init__"]
    while frontier:
        name = frontier.pop()
        if name in reached:
            continue
        reached.add(name)
        for call in _attribute_calls(methods[name]):
            attr = call.func.attr
            if attr in _STEP_OPERATIONS:
                offenders.append(f"{name} -> {attr} (line {call.lineno})")
            if (
                isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
                and attr in methods
            ):
                frontier.append(attr)

    assert "__init__" in reached
    assert not offenders, offenders


def test_the_step_names_every_operation_the_constructor_gave_up():
    """The move must be complete, not partial: all four land in the step."""
    step = _function(_tree("model_executor.py"), "capture_graphs")
    names = {call.func.attr for call in _attribute_calls(step)}
    assert _STEP_OPERATIONS - {"capture_graphs"} <= names, names


def test_the_step_runs_its_operations_in_the_order_the_constructor_did():
    """Tune, freeze, capture decode, capture prefill; the caller seeds after.

    Tuning and capture draw from the generator, so the boot path seeds once
    they are done, exactly where the constructor used to.
    """
    step = _function(_tree("model_executor.py"), "capture_graphs")
    sequence = [
        call.func.attr
        for call in _attribute_calls(step)
        if call.func.attr in _STEP_OPERATIONS
    ]
    assert sequence == ["_autotune", "freeze", "capture", "capture"]

    builder = _function(_tree("device.py"), "build_device_side")
    seed = [
        call.lineno
        for call in ast.walk(builder)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "set_random_seed"
    ]
    capture = [
        call.lineno
        for call in _attribute_calls(builder)
        if call.func.attr == "capture_graphs"
    ]
    assert seed and capture and max(capture) < min(seed), (capture, seed)
