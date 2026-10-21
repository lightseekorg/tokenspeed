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

"""Compile the Iris K3 fusions for gfx1250 without launching them."""

import re

import pytest
from tokenspeed_kernel.platform import ArchVersion, current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd, reason="Iris requires AMD ROCm"
)

_GFX1250 = ("gfx1250", 32)


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


def _backedge_cycles(lines):
    """Return the (label, branch) line spans of every backward branch."""
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


_K3_KERNELS = {
    "moe_pull_scatter": {"FIRST_WIDTH": 3584, "SECOND_WIDTH": 7168},
    "moe_push_gather": {"PREFIX_IS_SHARDED": True},
    "attn_pull_scatter": {"HAS_RESIDUAL": True},
    "attn_push_gather": {},
    "attnres_push_oneshot": {},
}


def _compile_k3(kind, rank):
    """Compile one K3 kernel for gfx1250 as ``rank`` of eight."""
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel.ops.communication._iris import all_reduce as ar
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        allreduce_sharded as sharded,
    )
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
        attnres,
    )

    policy = ar.iris_arch_policy(ArchVersion(12, 5))
    num_warps = 4
    constexprs = {"RANK": rank, **_K3_KERNELS[kind]}
    if kind == "attnres_push_oneshot":
        fn = attnres.iris_k3attn_push_oneshot
        num_warps = 16
        constexprs.update(
            WORLD_SIZE=8,
            HIDDEN=7168,
            BLOCK=8192,
            MAX_ELEMENTS=16 * 7168,
            READY_SLOT_STRIDE=16 * 8,
            EPS=1e-6,
            ELEMENTS_PER_THREAD=8,
            PEER_PAYLOAD_STORE_CACHE=policy.peer_payload_store,
        )
    else:
        fn = getattr(sharded, f"iris_k3{kind}")
        constexprs["BLOCK_ELEMENTS"] = 2048
        if "pull" in kind:
            constexprs.update(
                NUM_PROGRAMS=sharded.REDUCE_PROGRAMS,
                PEER_PAYLOAD_LOAD_CACHE=policy.peer_payload_load,
            )
        else:
            constexprs.update(
                NUM_PROGRAMS=sharded.GATHER_PROGRAMS,
                PEER_PAYLOAD_STORE_CACHE=policy.peer_payload_store,
            )
    constexprs["NUM_WARPS"] = num_warps
    for wait in ("wait_vmem", "wait_stores"):
        if wait.upper() in fn.arg_names:
            constexprs[wait.upper()] = getattr(policy, wait)
    signature = {}
    for name in fn.arg_names:
        if name in constexprs:
            continue
        if name.startswith("heap_base_"):
            signature[name] = "i64"
        elif name in ("ROWS", "LOCAL_ROWS"):
            signature[name] = "i32"
        elif name in ("ready_flags", "generations"):
            signature[name] = "*i32"
        elif name.startswith("scratch_") and name != "scratch_ptr":
            signature[name] = "*fp32"
        else:
            signature[name] = "*bf16"
    source = gluon._runtime.GluonASTSource(
        fn,
        signature,
        constexprs=constexprs,
        attrs=_aligned(
            fn, {name for name, kind in signature.items() if kind.startswith("*")}
        ),
    )
    return triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", *_GFX1250),
        options={"num_warps": num_warps},
    )


@pytest.mark.parametrize("rank", (0, 7))
@pytest.mark.parametrize("kind", tuple(_K3_KERNELS))
def test_iris_k3_codegen_gfx1250(kind, rank, tmp_path):
    """The K3 fusions build for wave32 and publish with system-scope releases.

    Peer payloads move as plain 16-byte accesses, the only non-temporal loads
    are flag polls on their retry loops, and gathers drain their peer stores
    before the release that publishes them.
    """
    assembly = _compile_k3(kind, rank).asm["amdgcn"]
    (tmp_path / f"{kind}-rank{rank}.amdgcn").write_text(assembly)
    lines = assembly.splitlines()
    assert "vmcnt" not in assembly
    assert re.search(r"\.amdhsa_private_segment_fixed_size\s+0\b", assembly)

    loads = [line for line in lines if re.search(r"\b(global|buffer)_load_", line)]
    non_temporal = [line for line in loads if "TH_LOAD_NT" in line]
    assert all("SCOPE_SYS" in line for line in non_temporal), non_temporal
    polls = [
        index
        for index, line in enumerate(lines)
        if re.search(r"\b(global|buffer)_load_\w+\s.*SCOPE_SYS", line)
    ]
    assert polls
    cycles = _backedge_cycles(lines)
    assert all(
        any(start <= poll <= end for start, end in cycles) for poll in polls
    ), "a peer flag poll was hoisted out of its retry loop"

    payload_stores = [line for line in lines if re.search(r"\bbuffer_store_", line)]
    assert all("buffer_store_b128" in line for line in payload_stores)
    if "push" in kind:
        peer_stores = [line for line in payload_stores if "th:" not in line]
        assert len(peer_stores) >= 7, payload_stores
    if "pull" in kind:
        assert sum("buffer_load_b128" in line for line in loads) >= 8
    releases = [
        index
        for index, line in enumerate(lines)
        if re.search(r"\bglobal_atomic_\w+.*SCOPE_SYS", line)
    ]
    assert releases, "epochs must be published with system-scope atomics"
    if "push" in kind:
        stores = [i for i, line in enumerate(lines) if "buffer_store_b128" in line]
        publish = next(index for index in releases if index > stores[0])
        last_store = max(index for index in stores if index < publish)
        assert any(
            "s_wait_storecnt 0x0" in line for line in lines[last_store:publish]
        ), "peer stores must drain before their epoch is released"
