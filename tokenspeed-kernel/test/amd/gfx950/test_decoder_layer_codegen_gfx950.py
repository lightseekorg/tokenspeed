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

"""Compile gate of the fused Muse Spark decoder layer on gfx950.

Every plan variant is compiled without a GPU, with the argument attributes a
launch gives it, and its assembly is checked against the rules the layer's
schedule depends on: one workgroup of 8 waves per CU within 256 VGPR+AGPR
and no spills; no L2 write-back or invalidate; the persistent entry and poll
rules; weight streams that never drain at a boundary; weights issued before
a wait that stay in flight through it.

Instructions are tied to the functions they came from through the ``.loc``
lines and their inlining chains, so each rule names the role it applies to,
not an assembly offset.
"""

import re
from functools import lru_cache

import pytest
import torch

pytest.importorskip("tokenspeed_kernel_amd")
layer = pytest.importorskip(
    "tokenspeed_kernel.ops.decoder_layer._gluon.muse_spark.layer"
)

from asm_checks import (  # noqa: E402
    _innermost_loop_lines,
    banned_instruction_problems,
    drains_of,
    frame_functions,
    isa_report,
    persistent_codegen_problems,
    register_problems,
    serialized_coherent_loads,
    source_lines,
    split_poll_problems,
    store_loop_preheader_drains,
    stream_loop_problems,
    write_through_store_drains,
)
from tokenspeed_kernel._triton import gluon, triton  # noqa: E402

# Any rank: the variants differ in their peers' order only.
_RANK = 3
# Functions whose code only runs on the failure path.
_FAILURE_PATHS = {"fail_stop", "iris_fail_stop", "trap"}


def _variant_id(key):
    return f"{key.requests}req-{key.layer_kind}-next_{key.next_kind}"


def _variants(layer_kind=None):
    for key in layer.PLAN_VARIANTS:
        if layer_kind in (None, key.layer_kind):
            yield pytest.param(key, id=_variant_id(key))


def _compile(key):
    """The variant's assembly and constexprs; a compile error is raised
    again on every call without compiling twice."""
    result = _compile_once(key)
    if isinstance(result, Exception):
        raise result
    return result[:2]


def lds_bytes(key):
    """The variant's LDS (dynamic shared memory) per workgroup."""
    _compile(key)
    return _compile_once(key)[2]


@lru_cache(maxsize=None)
def _compile_once(key):
    """Compile the variant as a launch would (``offline_attrs``), or return
    its compile error."""
    fn, signature, constexprs, options = layer.offline_compile_args(key, _RANK)
    full = {name: signature[name] for name in fn.arg_names if name in signature}
    full.update({name: "constexpr" for name in constexprs})
    source = gluon._runtime.GluonASTSource(
        fn, full, constexprs=constexprs, attrs=layer.offline_attrs(signature)
    )
    try:
        kernel = triton.compile(
            source, target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
            options=options,
        )  # fmt: skip
    except Exception as error:  # noqa: BLE001 - re-raised by _compile
        return error
    return kernel.asm["amdgcn"], constexprs, kernel.metadata.shared


@lru_cache(maxsize=None)
def attributed(key):
    """``(texts, chains)``: the variant's instruction lines and, for each,
    the names of the functions it came from, innermost first."""
    assembly, _ = _compile(key)
    lines = source_lines(assembly)
    names = frame_functions(assembly)
    return [line.text for line in lines], [tuple(names(line.frames)) for line in lines]


def _in(chains, function):
    return [i for i, chain in enumerate(chains) if function in chain]


def _loads(texts, indices):
    return [i for i in indices if "_load_" in texts[i]]


def _next_in(chains, function, after):
    return next(i for i in _in(chains, function) if i > after)


@pytest.mark.parametrize("key", _variants())
def test_registers_fit_8_waves_without_spills(key):
    assembly, _ = _compile(key)
    assert register_problems(isa_report(assembly)) == []


@pytest.mark.parametrize("key", _variants())
def test_no_l2_maintenance_wave_sync_or_sleep(key):
    assembly, _ = _compile(key)
    assert banned_instruction_problems(assembly) == []


@pytest.mark.parametrize("key", _variants())
def test_entry_and_poll_rules(key):
    # Entry order and claim, joins, and every poll: one vote, a deadline or
    # abort read on its retry path, no barrier or LDS exchange in it, and a
    # barrier after each vector flag poll.
    assembly, _ = _compile(key)
    assert persistent_codegen_problems(assembly, entry="vector") == []
    assert split_poll_problems(assembly) == []


@pytest.mark.parametrize("key", _variants())
def test_no_volatile_drains_on_the_main_path(key):
    assembly, _ = _compile(key)
    texts, chains = attributed(key)
    main = [
        text
        for text, chain in zip(texts, chains)
        if not _FAILURE_PATHS.intersection(chain)
    ]
    assert serialized_coherent_loads(main) == []
    assert write_through_store_drains(main) == []
    assert store_loop_preheader_drains(assembly) == []


@pytest.mark.parametrize("key", _variants())
def test_moe_loops_keep_the_ring_and_have_no_barrier(key):
    assembly, constexprs = _compile(key)
    raw = assembly.splitlines()
    by_index = {line.index: line for line in source_lines(assembly)}
    moe_loops = [
        indices
        for indices in _innermost_loop_lines(assembly).values()
        if any(
            "native_stream.py" in (name for name, _ in by_index[i].frames)
            for i in indices
            if i in by_index
        )
    ]
    assert moe_loops
    for indices in moe_loops:
        loop = "\n".join(raw[i] for i in indices)
        assert "s_barrier" not in loop
        if "v_mfma" in loop:
            assert stream_loop_problems(loop, constexprs["RING"]) == []


def _entry_waits(texts, chains):
    """Indices of the waits inside the launch entry (the vector form's claim
    exchange waits for the role's loads issued before it, by design)."""
    return {i for i in _in(chains, "enter_check") if "s_waitcnt" in texts[i]}


@pytest.mark.parametrize("key", _variants())
def test_entry_is_an_epoch_store_and_a_claim_exchange(key):
    # The vector entry: each inlined copy stores the epoch and exchanges the
    # claim, and runs no other vector memory operation (its epoch read is a
    # scalar load), so the role's weights issued before it stay in flight
    # up to the exchange's own wait.
    texts, chains = attributed(key)
    entry = [
        texts[i]
        for i, chain in enumerate(chains)
        if {"enter_issue", "enter_check"}.intersection(chain)
        and not _FAILURE_PATHS.intersection(chain)
        and re.match(r"(global|buffer|flat)_", texts[i])
    ]
    stores = [t for t in entry if t.startswith("global_store_dwordx2")]
    claims = [t for t in entry if t.startswith("global_atomic_swap_x2")]
    assert claims and len(stores) == len(claims) == len(entry) // 2
    assert len(stores) + len(claims) == len(entry)


@pytest.mark.parametrize("key", _variants())
def test_router_weights_stay_in_flight_until_the_all_gather(key):
    # The router's tiles are issued after the sender's push (slot 0: after
    # OWN1); nothing may retire them before the router's x build waits for
    # the all-gather.
    texts, chains = attributed(key)
    issued = _loads(texts, _in(chains, "_router_weights"))
    assert issued
    poll = _next_in(chains, "_kpart_state", issued[-1])
    assert [texts[i] for i in drains_of(texts, issued[-1], poll)] == []


@pytest.mark.parametrize("key", _variants())
def test_pooler_weights_stay_in_flight_until_the_all_gather(key):
    # Past the entry: a sliding layer issues them before the entry check,
    # whose claim exchange may wait for them.
    texts, chains = attributed(key)
    name = (
        "_pooler_weights_sliding" if key.layer_kind == "sliding" else "_pooler_weights"
    )
    issued = _loads(texts, _in(chains, name))
    assert issued
    poll = _next_in(chains, "_kpart_state", issued[-1])
    drains = set(drains_of(texts, issued[-1], poll)) - _entry_waits(texts, chains)
    assert [texts[i] for i in sorted(drains)] == []


@pytest.mark.parametrize("key", _variants())
def test_sender_clears_drain_before_its_push(key):
    texts, chains = attributed(key)
    send = set(_in(chains, "_send"))
    clears = [
        i for i in _in(chains, "iris_clear_share") if i in send and "_store" in texts[i]
    ]
    pushes = [
        i for i in _in(chains, "h1_rs_sink") if i in send and "_store" in texts[i]
    ]
    assert clears and pushes
    assert drains_of(texts, clears[-1], pushes[0])


_TORCH_TYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8e4nv": torch.float8_e4m3fn,
    "i64": torch.int64,
    "i32": torch.int32,
    "u32": torch.uint32,
    "u8": torch.uint8,
}


def _launch_args(constexprs, **dtypes):
    """CPU stand-ins for a launch's arguments, typed as the offline
    signature says unless ``dtypes`` overrides a tensor's dtype."""
    args = []
    for name, kind in layer.offline_signature(constexprs).items():
        if isinstance(kind, tuple):
            args.append((0,) * len(kind))
        elif kind.startswith("*"):
            dtype = dtypes.get(name, _TORCH_TYPES[kind.lstrip("*k")])
            args.append(torch.empty(1, dtype=dtype))
        else:
            args.append(1)
    return tuple(args)


def test_launch_types_are_checked_against_the_offline_signature():
    # The Iris areas are allocated as int32 words; launched like that, the
    # kernel compiles another binary than the one prepare checked (and a
    # BF16 store into them fails to verify).
    _, _, constexprs, _ = layer.offline_compile_args(layer.PLAN_VARIANTS[0], 3)
    assert layer.signature_problems(_launch_args(constexprs), constexprs) == []
    problems = layer.signature_problems(
        _launch_args(constexprs, h1=torch.int32, box=torch.int32), constexprs
    )
    assert problems == [
        "box: launched *i32, offline *u32",
        "h1: launched *i32, offline *bf16",
    ]
