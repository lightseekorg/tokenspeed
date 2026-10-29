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

"""Compile an Iris collective without launching it to check its cache policy."""

import re

import pytest
from tokenspeed_kernel.platform import ArchVersion, current_platform

_ARCH_VERSIONS = {"gfx950": ArchVersion(9, 5), "gfx1250": ArchVersion(12, 5)}

_ARCHS = (
    pytest.param(
        "gfx950",
        marks=pytest.mark.skipif(
            not current_platform().is_cdna4, reason="gfx950 codegen"
        ),
    ),
    pytest.param(
        "gfx1250",
        marks=pytest.mark.skipif(
            not current_platform().is_cdna5, reason="gfx1250 codegen"
        ),
    ),
)


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _compile_allgather_columns(arch_name):
    """Compile the eight-rank column all-gather for ``arch_name`` without launching.

    The input is already published, so the only stores into peer memory are
    the barrier epochs.
    """
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris import all_gather as ag
    from tokenspeed_kernel.ops.communication._iris.all_reduce import (
        iris_arch_policy,
    )

    policy = iris_arch_policy(_ARCH_VERSIONS[arch_name])
    subgroup_size = triton.backends.amd.compiler.HIPOptions(arch=arch_name).warp_size
    fn = ag.iris_pull_allgather_columns
    pointers = ("input_ptr", "input_sym_ptr", "output_ptr", "ready_flags")
    signature = {name: "*bf16" for name in pointers[:3]}
    signature.update({"ready_flags": "*i32", "heaps": ("i64",) * 8})
    signature["LOCAL_WORDS"] = "i32"
    constexprs = {
        "RANK": 0,
        "WORLD_SIZE": 8,
        "LOCAL_HIDDEN_WORDS": 256,
        "TOTAL_HIDDEN_WORDS": 8 * 256,
        "BLOCK_WORDS": 4 * subgroup_size,
        "NUM_WARPS": 1,
        "WORDS_PER_LANE": 4,
        "INPUT_PUBLISHED": True,
        "PEER_PAYLOAD_LOAD_CACHE": policy.peer_payload_load,
        "PEER_SIGNAL_STORE_CACHE": policy.peer_signal_store,
    }
    signature.update({name: "constexpr" for name in constexprs})
    attrs = {
        (index,): [["tt.divisibility", 16]]
        for index, name in enumerate(fn.arg_names)
        if name in pointers
    }
    source = gluon._runtime.GluonASTSource(
        fn, signature, constexprs=constexprs, attrs=attrs
    )
    return triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", arch_name, subgroup_size),
        options={"num_warps": 1},
    )


_LOAD = re.compile(r"\b(?:global|buffer)_load_(\w+)")
# Flags are 32-bit; the BF16 payload is always loaded in wider words.
_FLAG_WIDTHS = ("dword", "b32")


def _policy(line):
    """Return the cache-policy fields of a memory instruction."""
    return " ".join(re.findall(r"\b(?:sc0|sc1|nt|th:\w+|scope:\w+)\b", line))


def _seeds_atomic_expansion(lines, index):
    """Return whether the load at ``index`` seeds an expanded atomic.

    LLVM may expand an atomic RMW into a compare-and-swap loop, which opens
    with a plain load of the current value; that load falls through into the
    loop header LLVM names ``%atomicrmw.start``.
    """
    for line in lines[index + 1 :]:
        if re.match(r"^\.LBB\w+:", line):
            return "%atomicrmw.start" in line
    return False


def _backedge_cycles(lines):
    """Return the line spans that a backward branch closes."""
    labels = {
        match[1]: index
        for index, line in enumerate(lines)
        if (match := re.match(r"^(\.LBB\w+):", line))
    }
    cycles = []
    for index, line in enumerate(lines):
        match = re.search(r"\bs_(?:cbranch_\w+|branch)\s+(\.LBB\w+)", line)
        if match and labels.get(match[1], index) < index:
            cycles.append((labels[match[1]], index))
    return cycles


# gfx12 spells the cache policy as th:/scope: fields, gfx9 as sc0/sc1/nt bits.
_ISA = {
    "gfx950": {
        "flag_load": "sc0 sc1",
        "payload_loads": {"", "nt"},
        "flag_store": (r"\bglobal_store_dword\s", "sc0 sc1"),
        "system": r"\bsc1\b",
    },
    "gfx1250": {
        "flag_load": "th:TH_LOAD_NT scope:SCOPE_SYS",
        "payload_loads": {""},
        "flag_store": (r"\bglobal_store_b32\b", ""),
        "system": r"\bSCOPE_SYS\b",
    },
}


@pytest.mark.parametrize("arch", _ARCHS)
def test_iris_arch_policy_codegen(arch, tmp_path):
    """Peer accesses carry their architecture's IrisArchPolicy.

    gfx950 reads peer data non-temporally and writes peer barrier epochs
    write-through. On gfx1250 both carry no cache policy. Flag polls and epoch
    atomics are system-scoped everywhere, and each poll stays on its retry loop.
    The pull all-reduce reads the same table and shares the barrier helper, so
    one kernel demonstrates the policy.
    """
    isa = _ISA[arch]
    assembly = _compile_allgather_columns(arch).asm["amdgcn"]
    (tmp_path / f"{arch}-allgather-columns.amdgcn").write_text(assembly)
    lines = assembly.splitlines()

    loads = [
        (index, match[1])
        for index, line in enumerate(lines)
        if (match := _LOAD.search(line)) and not _seeds_atomic_expansion(lines, index)
    ]
    polls = [index for index, width in loads if width in _FLAG_WIDTHS]
    payload = [lines[index] for index, width in loads if width not in _FLAG_WIDTHS]
    assert len(polls) == 2, [lines[index] for index in polls]
    assert all(_policy(lines[index]) == isa["flag_load"] for index in polls)
    assert all(_policy(line) in isa["payload_loads"] for line in payload), payload
    if arch == "gfx950":
        assert any(_policy(line) == "nt" for line in payload), payload

    cycles = _backedge_cycles(lines)
    assert all(
        any(start <= poll <= end for start, end in cycles) for poll in polls
    ), "a peer flag poll was hoisted out of its retry loop"

    store, store_policy = isa["flag_store"]
    stores = [line for line in lines if re.search(store, line)]
    assert stores
    assert all(_policy(line) == store_policy for line in stores), stores
    atomics = [line for line in lines if re.search(r"\bglobal_atomic_", line)]
    assert atomics
    assert all(re.search(isa["system"], line) for line in atomics), atomics

    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)
