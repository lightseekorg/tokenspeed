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

"""Compile the Iris pull all-reduce protocols without launching them."""

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


def _aligned(fn, names):
    """Mark the named pointers and every heap base 16-byte aligned, as at launch."""
    return {
        (index,): [["tt.divisibility", 16]]
        for index, name in enumerate(fn.arg_names)
        if name in names or name.startswith("heap_base_")
    }


def _compile(arch_name, kind, world_size):
    """Compile one all-reduce protocol for ``arch_name`` without launching it.

    Args:
        arch_name: Target architecture, "gfx950" or "gfx1250".
        kind: "staged_oneshot", "pull_oneshot", "pull_twoshot", or
            "pull_twoshot_exit" (two-shot with the exit barrier).
        world_size: Number of ranks the protocol is specialized for.

    Returns:
        The compiled kernel, with its assembly in ``asm["amdgcn"]``.
    """
    from tokenspeed_kernel._triton import gl, gluon, triton
    from tokenspeed_kernel.ops.communication._iris import all_reduce as ar

    arch = ar.pull_all_reduce_arch_config(_ARCH_VERSIONS[arch_name])
    if kind == "staged_oneshot":
        fn = ar.iris_pull_oneshot_staged
        signature = {
            "input_ptr": "*bf16",
            "input_sym_ptr": "*bf16",
            "output_ptr": "*bf16",
            "ready_flags": "*i32",
            "heap_bases": "*i64",
            "NUMEL": "i32",
        }
        constexprs = {
            "RANK": 0,
            "WORLD_SIZE": world_size,
            "BLOCK_SIZE": ar.StagedPullOneShot.BLOCK_SIZE,
            "SLOT_STRIDE": 16 * 7168,
            "NUM_SLOTS": ar.StagedPullOneShot.INPUT_SLOTS,
            "PEER_LOAD_CACHE": arch.peer_load_cache,
        }
        signature.update({name: "constexpr" for name in constexprs})
        source = triton.compiler.ASTSource(
            fn,
            signature,
            constexprs=constexprs,
            attrs=_aligned(fn, set(signature) - set(constexprs)),
        )
        num_warps = ar.StagedPullOneShot.NUM_SUBGROUPS
    else:
        two_shot = kind.startswith("pull_twoshot")
        fn = ar.iris_pull_twoshot if two_shot else ar.iris_pull_oneshot
        pointers = (
            ("input_sym_ptr", "scratch_sym_ptr", "output_ptr")
            if two_shot
            else ("input_sym_ptr", "output_ptr")
        )
        signature = {name: "*bf16" for name in pointers}
        signature["ready_flags"] = "*i32"
        signature.update(
            {name: "i64" for name in fn.arg_names if name.startswith("heap_base_")}
        )
        sizes = ("PARTITION_WORDS" if two_shot else "TOTAL_NUMEL", "NUM_PROGRAMS")
        signature.update({name: "i32" for name in (*sizes, "NUM_TILES")})
        num_warps = (
            ar._TWO_SHOT_NUM_SUBGROUPS if two_shot else ar.ProducerPull.NUM_SUBGROUPS
        )
        constexprs = {
            "RANK": 0,
            "WORLD_SIZE": world_size,
            "NUM_WARPS": num_warps,
            "SUBGROUP_SIZE": arch.subgroup_size,
            "WORDS_PER_LANE": ar._PULL_WORDS_PER_LANE,
            "ELEMENT_DTYPE": gl.bfloat16,
            "ELEMENTS_PER_WORD": ar._PACKED_WORD_BYTES // 2,
            "PEER_LOAD_CACHE": arch.peer_load_cache,
            "REMOTE_STORE_CACHE": arch.remote_store_cache,
            "residual": None,
            "post": None,
            "comb": None,
            "HIDDEN_WORDS": 0,
            "HC_POST": False,
        }
        if two_shot:
            constexprs["BLOCK_WORDS"] = (
                num_warps * arch.subgroup_size * ar._PULL_WORDS_PER_LANE // world_size
            )
            constexprs["EXIT_BARRIER"] = kind == "pull_twoshot_exit"
        else:
            constexprs["BLOCK_SIZE"] = ar.ProducerPull.BLOCK_SIZE
            constexprs["PUBLISH_READY"] = False
        source = gluon._runtime.GluonASTSource(
            fn,
            signature,
            constexprs=constexprs,
            attrs=_aligned(fn, (*pointers, "ready_flags")),
        )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", arch_name, arch.subgroup_size),
        options={"num_warps": num_warps},
    )
    return kernel


_LOAD = re.compile(r"\b(?:global|buffer)_load_(\w+)")
# Flags are 32-bit; the aligned BF16 payload is always loaded in wider words.
_FLAG_WIDTHS = ("dword", "b32")


def _policy(line):
    """Return the cache-policy fields of a memory instruction."""
    return " ".join(re.findall(r"\b(?:sc0|sc1|nt|th:\w+|scope:\w+)\b", line))


def _backedge_cycles(lines):
    """Return the line spans that a backward branch closes.

    A heuristic for retry loops: it catches a poll hoisted above its loop header,
    but not every block reordering the compiler could produce.
    """
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
        # Buffer loads spell the .cg peer loads "sc0 nt"; local loads are plain.
        "payload_loads": {"", "nt", "sc0 nt"},
        "system": r"\bsc1\b",
        "flag_store": r"\bglobal_store_dword\s",
    },
    "gfx1250": {
        "flag_load": "th:TH_LOAD_NT scope:SCOPE_SYS",
        "payload_loads": {""},
        "system": r"\bSCOPE_SYS\b",
        "flag_store": r"\bglobal_store_b32\b",
    },
}

_KINDS = ("staged_oneshot", "pull_oneshot", "pull_twoshot", "pull_twoshot_exit")


def _expected_polls(kind, world_size):
    """The staged gather polls once per peer, the pull protocols once per barrier."""
    if kind == "staged_oneshot":
        return world_size - 1
    return 3 if kind == "pull_twoshot_exit" else 2


@pytest.mark.parametrize("world_size", (2, 4, 8))
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("arch", _ARCHS)
def test_iris_all_reduce_codegen(arch, kind, world_size, tmp_path):
    """Peer accesses keep their architecture's cache policy.

    Flag polls and epoch atomics are system-scoped on every architecture, and
    each poll must stay on its retry backedge. gfx950 reads peer data
    non-temporally and writes peer flags write-through. On gfx1250 a
    non-temporal or scoped data load refetches every partially read sector over
    the fabric, so payload loads carry no cache policy at all.
    """
    isa = _ISA[arch]
    kernel = _compile(arch, kind, world_size)
    assembly = kernel.asm["amdgcn"]
    (tmp_path / f"{arch}-{kind}-ws{world_size}.amdgcn").write_text(assembly)
    lines = assembly.splitlines()

    loads = [
        (index, match[1])
        for index, line in enumerate(lines)
        if (match := _LOAD.search(line))
    ]
    polls = [index for index, width in loads if width in _FLAG_WIDTHS]
    payload = [lines[index] for index, width in loads if width not in _FLAG_WIDTHS]
    assert len(polls) == _expected_polls(kind, world_size), [lines[i] for i in polls]
    assert all(_policy(lines[index]) == isa["flag_load"] for index in polls), [
        lines[index] for index in polls
    ]
    assert payload
    assert all(_policy(line) in isa["payload_loads"] for line in payload), payload
    if arch == "gfx950":
        assert any(
            "nt" in _policy(line).split() for line in payload
        ), "peer data loads lost their non-temporal hint"

    cycles = _backedge_cycles(lines)
    assert all(
        any(start <= poll <= end for start, end in cycles) for poll in polls
    ), "a peer flag poll was hoisted out of its retry loop"

    if kind.startswith("pull_twoshot"):
        # The 32-bit global stores are the epochs written into peer flag rows.
        stores = [line for line in lines if re.search(isa["flag_store"], line)]
        assert stores
        if arch == "gfx1250":
            assert not any("th:" in line or "scope:" in line for line in stores), stores
        else:
            assert all(re.search(r"\bsc0 sc1\b", line) for line in stores), stores
    atomics = [line for line in lines if re.search(r"\bglobal_atomic_", line)]
    assert atomics
    assert all(re.search(isa["system"], line) for line in atomics), atomics

    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)
