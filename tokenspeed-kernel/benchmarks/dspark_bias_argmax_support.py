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

"""Execute source-identical DSpark methods with explicit boundary fixtures.

The baseline is a verified snapshot of the original full files, never a
candidate with its optimization disabled. AST selection removes unrelated
model imports/initialization, but every selected method, including decorators,
is compiled unchanged. This is a consumer-boundary replay, not model E2E.
"""

from __future__ import annotations

import ast
import functools
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

BASELINE_SHA = "7978c48b5c56445c600880f0ff275a611e584cef"
BASELINE_DIGESTS = {
    "dflash.py": "b026d3ca1ceea8f2d500e7f2f3ccf410e638722e77505080f61013d2d0f770fa",
    "dspark.py": "a2f52f4d8dd815bbfbfacfcbba66e41a605c671fcd5e89ca8a1bf178c3483234",
    "nvtx.py": "b8a257790b149053f1b0264987d8132120d58d4453b5c9abb8bed2d024644482",
    "model_dspark.py": "aa6daae58b55846296740ff3deca2869c515f2efb32792e1ec481f85314041c2",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _forbidden(*args, **kwargs):
    raise AssertionError("fixture crossed its TP1 original-vocabulary boundary")


def _compile_nodes(path, names, namespace, methods):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if getattr(node, "name", None) not in names:
            continue
        if isinstance(node, ast.ClassDef) and node.name in methods:
            keep = methods[node.name]
            node.body = [
                item for item in node.body if getattr(item, "name", None) in keep
            ]
            # Heavy base construction is outside this source-method replay.
            node.bases = (
                [ast.Name(id="DFlash", ctx=ast.Load())] if node.name == "DSpark" else []
            )
        selected.append(node)
    assert {node.name for node in selected} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def load_consumers(repository, baseline_dir, create_workspace, try_argmax):
    repository, baseline_dir = Path(repository), Path(baseline_dir)
    for name, expected in BASELINE_DIGESTS.items():
        assert digest(baseline_dir / name) == expected, f"baseline changed: {name}"
    runtime = repository / "python/tokenspeed/runtime"
    classes, provenance = {}, {}
    for arm in ("A", "P"):
        paths = {
            "dflash": (
                baseline_dir / "dflash.py"
                if arm == "A"
                else runtime / "execution/drafter/dflash.py"
            ),
            "dspark": (
                baseline_dir / "dspark.py"
                if arm == "A"
                else runtime / "execution/drafter/dspark.py"
            ),
            "nvtx": (
                baseline_dir / "nvtx.py" if arm == "A" else runtime / "utils/nvtx.py"
            ),
            "markov": (
                baseline_dir / "model_dspark.py"
                if arm == "A"
                else runtime / "models/dspark.py"
            ),
        }
        namespace = {
            "__name__": f"_dspark_source_boundary_{arm}",
            "torch": torch,
            "nn": torch.nn,
            "functools": functools,
            "_enabled": False,
            "_VALID_COLORS": frozenset({"blue", "purple"}),
            "try_bias_argmax": try_argmax,
            "create_bias_argmax_workspace": create_workspace,
            "_UNSET": object(),
            "_dist_argmax": _forbidden,
            "all_gather_into_tensor": _forbidden,
        }
        _compile_nodes(paths["nvtx"], {"_Range", "nvtx_range"}, namespace, {})
        _compile_nodes(
            paths["dflash"],
            {"DFlash"},
            namespace,
            {
                "DFlash": {
                    "_greedy_argmax_vocab_parallel",
                    "_ensure_dist_argmax_state",
                    "wire_target",
                }
            },
        )
        _compile_nodes(
            paths["dspark"],
            {"DSpark"},
            namespace,
            {
                "DSpark": {
                    "_sample_block",
                    "_block_base_logits",
                    "_make_step_bias_fn",
                }
            },
        )
        _compile_nodes(paths["markov"], {"VanillaMarkov"}, namespace, {})
        classes[arm] = (namespace["DSpark"], namespace["VanillaMarkov"])
        provenance[arm] = {
            key: {"path": str(value.resolve()), "sha256": digest(value)}
            for key, value in paths.items()
        }
    return classes, provenance


def make_consumer(classes, arm, rows, steps, vocab, hidden, rank, dtype, device):
    consumer_class, markov_class = classes[arm]
    consumer = consumer_class()
    consumer.spec_algorithm = "DSPARK"
    consumer.spec_num_tokens = steps
    consumer._dist_argmax_state = None
    create_workspace = consumer._greedy_argmax_vocab_parallel.__func__.__globals__[
        "create_bias_argmax_workspace"
    ]
    consumer._bias_argmax_workspace = (
        create_workspace(rows, vocab, device) if arm == "P" else None
    )
    shard = SimpleNamespace(
        num_org_elements=vocab,
        num_org_elements_padded=vocab,
        num_added_elements=0,
        org_vocab_start_index=0,
        added_vocab_start_index=vocab,
    )
    consumer.lm_head = SimpleNamespace(
        weight=torch.randn((vocab, hidden), device=device, dtype=dtype) / hidden**0.5,
        shard_indices=shard,
    )
    consumer.logits_processor = SimpleNamespace(tp_size=1)
    consumer.markov_head = markov_class(vocab_size=vocab, markov_rank=rank).to(
        device=device, dtype=dtype
    )
    consumer.markov_head.requires_grad_(False)
    with torch.no_grad():
        consumer.markov_head.markov_w1.weight.normal_(0, 1)
        consumer.markov_head.markov_w2.weight.normal_(0, rank**-0.5)
    return consumer


def copy_consumer_parameters(source, target):
    target.lm_head.weight.copy_(source.lm_head.weight)
    target.markov_head.load_state_dict(source.markov_head.state_dict())
