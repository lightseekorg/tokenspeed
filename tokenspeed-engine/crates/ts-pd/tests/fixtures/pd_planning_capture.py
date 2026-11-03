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

"""Record the Python PD planner's answers as ts-pd's planning fixtures.

A pytest plugin: run the Python planner's own tests with it loaded and every
call into the functions ts-pd's ``layout`` module ports is recorded, with its
inputs and its answer (or that it refused), into ``planning/``: the contracts
the records name, and the records by concern (each file stays small). It
needs the Python planner ts-pd replaced: run it on a tree from before the
planner's removal::

    TS_PD_CAPTURE_OUT=tokenspeed-engine/crates/ts-pd/tests/fixtures/planning \\
    PYTHONPATH=tokenspeed-engine/crates/ts-pd/tests/fixtures:$PYTHONPATH \\
    python -m pytest -p pd_planning_capture test/runtime/distributed/test_cache_pd_*.py ...
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import sys
import threading

import numpy as np

from tokenspeed.runtime.pd import cache_protocol, host_peer, transfer_plan
from tokenspeed.runtime.pd.mooncake import entities
from tokenspeed.runtime.pd.mooncake import prefill as mooncake_prefill

CONTRACTS: dict[str, str] = {}
RECORDS: list[dict] = []


def _contract(contract) -> str | None:
    """A contract's id in the fixture's contract table (its wire form)."""
    if contract is None:
        return None
    wire = contract.to_wire_bytes().decode("ascii")
    key = hashlib.sha1(wire.encode()).hexdigest()[:16]
    CONTRACTS[key] = wire
    return key


def _ints(values) -> list[int]:
    return [int(value) for value in np.asarray(values).reshape(-1).tolist()]


def _manifest(manifest) -> dict | None:
    if manifest is None:
        return None
    return {
        "groups": [
            {"group_id": group.group_id, "block_ids": _ints(group.block_ids)}
            for group in manifest.groups
        ],
        "prefix_len": int(manifest.prefix_len),
        "prompt_len": int(manifest.prompt_len),
        "state_prefix_cached": bool(manifest.state_prefix_cached),
    }


def _selection(selection) -> dict | None:
    if selection is None:
        return None
    return {
        "groups": [
            {
                "source_block_ids": _ints(group.source_block_ids),
                "destination_positions": _ints(group.destination_positions),
            }
            for group in selection.groups
        ]
    }


def _fragment(fragment) -> dict:
    return {key: value for key, value in dataclasses.asdict(fragment).items()}


def _filters(filters) -> dict | None:
    if filters is None:
        return None
    return {
        group_id: (
            None
            if value is None
            else {"owner_rank": value.owner_rank, "owner_count": value.owner_count}
        )
        for group_id, value in filters.items()
    }


def _plan(plan) -> dict:
    return {
        "fragments_by_prefill_rank": [
            [rank, [_fragment(fragment) for fragment in fragments]]
            for rank, fragments in plan.fragments_by_prefill_rank.items()
        ],
        "owner_filters_by_prefill_rank": [
            [rank, _filters(filters)]
            for rank, filters in plan.owner_filters_by_prefill_rank.items()
        ],
    }


def _tables(forward_op) -> dict:
    return {
        group_id: np.asarray(table).astype(np.int64).tolist()
        for group_id, table in forward_op.block_tables_arrays().items()
    }


def _error(error: BaseException) -> dict:
    return {"error": type(error).__name__, "message": str(error)}


_DEPTH = threading.local()


def _record(kind: str, args, run):
    """Run ``run``, record its answer or refusal, and pass either on. Only
    the outermost wrapped call records: what it calls is part of its answer.
    ``args`` is a thunk, evaluated only for a recorded call."""
    depth = getattr(_DEPTH, "value", 0)
    if depth:
        return run()[0]
    record = {"kind": kind, "args": args()}
    _DEPTH.value = depth + 1
    try:
        answer = run()
    except Exception as error:  # noqa: BLE001 -- recorded, then re-raised
        record["result"] = _error(error)
        RECORDS.append(record)
        raise
    finally:
        _DEPTH.value = depth
    record["result"] = {"ok": answer[1]}
    RECORDS.append(record)
    return answer[0]


def _contract_facts(contract) -> dict:
    """What ts-pd reads off a parsed contract."""
    groups = []
    for spec in contract.group_specs:
        groups.append(
            {
                "group_id": spec.group_id,
                "shard_count": spec.shard_count,
                "virtual_block_count": contract.virtual_block_count(spec.group_id),
                "fields": [
                    field.field_id for field in contract.fields_for_group(spec.group_id)
                ],
            }
        )
    fields = []
    for field in contract.plan.fields:
        try:
            address = contract.field_address(0x100000, field.field_id)
        except (KeyError, IndexError):
            address = None
        partition = contract.transfer_schema.partition_for(field.field_id)
        fields.append(
            {
                "field_id": field.field_id,
                "address": address,
                "payload_bytes": field.payload_bytes,
                "element_size": field.element_size,
                "partition": (
                    None
                    if partition is None
                    else {
                        "axis": partition.axis,
                        "global_extent": partition.global_extent,
                        "global_parts": list(partition.global_parts),
                    }
                ),
            }
        )
    return {"groups": groups, "fields": fields}


def _kv_args(kv_args) -> dict:
    return {
        "cache_layout": _contract(kv_args.cache_layout),
        "wire_layout": _contract(kv_args.wire_layout),
        "kv_data_ptr": int(kv_args.kv_data_ptr),
        "cache_fields_by_stage": [
            list(stage) for stage in kv_args.cache_fields_by_stage
        ],
    }


def _host_copies(blocks) -> list[dict]:
    """One transfer's copies as the host receives them (``_copy_items``)."""
    items = []
    for kind, a, b, c in host_peer._copy_items(list(blocks)):
        if kind == "pages":
            fields = np.frombuffer(c, dtype="<i8").reshape(-1, 5).tolist()
            items.append(
                {
                    "pages": {
                        "src_pages": np.frombuffer(a, dtype="<i8").tolist(),
                        "dst_pages": np.frombuffer(b, dtype="<i8").tolist(),
                        "fields": fields,
                    }
                }
            )
        else:
            items.append(
                {
                    "spans": {
                        "sources": np.frombuffer(a, dtype="<u8").tolist(),
                        "destinations": np.frombuffer(b, dtype="<u8").tolist(),
                        "lengths": np.frombuffer(c, dtype="<u8").tolist(),
                    }
                }
            )
    return items


# ---------------------------------------------------------------------------
# The wrapped entry points. Each keeps the original's signature and behavior.

_ORIGINALS: dict[str, object] = {}


def _contract_from_wire(cls, raw):
    def run():
        contract = _ORIGINALS["contract_from_wire"](cls, raw)
        return contract, _contract_facts(contract)

    return _record("contract_from_wire", lambda: {"raw": bytes(raw).hex()}, run)


def _manifest_from_wire(cls, raw):
    def run():
        manifest = _ORIGINALS["manifest_from_wire"](cls, raw)
        return manifest, _manifest(manifest)

    return _record("manifest_from_wire", lambda: {"raw": bytes(raw).hex()}, run)


def _validate_manifest(manifest, *, layout, peer):
    def args():
        return {
            "manifest": _manifest(manifest),
            "layout": _contract(layout),
            "peer": peer,
        }

    def run():
        _ORIGINALS["validate_manifest"](manifest, layout=layout, peer=peer)
        return None, True

    return _record("validate_manifest", args, run)


def _build_manifest(
    forward_op, *, layout, request_row, prefix_len, prompt_len, state_prefix_cached
):
    def args():
        return {
            "tables": _tables(forward_op),
            "layout": _contract(layout),
            "request_row": request_row,
            "prefix_len": prefix_len,
            "prompt_len": prompt_len,
            "state_prefix_cached": state_prefix_cached,
        }

    def run():
        manifest = _ORIGINALS["build_manifest"](
            forward_op,
            layout=layout,
            request_row=request_row,
            prefix_len=prefix_len,
            prompt_len=prompt_len,
            state_prefix_cached=state_prefix_cached,
        )
        return manifest, _manifest(manifest)

    return _record("build_manifest", args, run)


def _build_selection(
    forward_op,
    *,
    layout,
    request_row,
    prefix_len,
    prompt_len,
    chunk_start,
    chunk_end,
    state_prefix_cached,
):
    def args():
        return {
            "tables": _tables(forward_op),
            "layout": _contract(layout),
            "request_row": request_row,
            "prefix_len": prefix_len,
            "prompt_len": prompt_len,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "state_prefix_cached": state_prefix_cached,
        }

    def run():
        selection = _ORIGINALS["build_selection"](
            forward_op,
            layout=layout,
            request_row=request_row,
            prefix_len=prefix_len,
            prompt_len=prompt_len,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            state_prefix_cached=state_prefix_cached,
        )
        return selection, _selection(selection)

    return _record("build_selection", args, run)


def _validate_peer_layout(layout, peer_layout):
    def args():
        return {"layout": _contract(layout), "peer_layout": _contract(peer_layout)}

    def run():
        _ORIGINALS["validate_peer_layout"](layout, peer_layout)
        return None, True

    return _record("validate_peer_layout", args, run)


def _pipeline_plan(
    *,
    prefill_tp_size,
    decode_tp_size,
    decode_tp_rank,
    prefill_layout,
    decode_layout,
    cache_fields_by_stage,
):
    def args():
        return {
            "prefill_tp_size": prefill_tp_size,
            "decode_tp_size": decode_tp_size,
            "decode_tp_rank": decode_tp_rank,
            "prefill_layout": _contract(prefill_layout),
            "decode_layout": _contract(decode_layout),
            "cache_fields_by_stage": [list(stage) for stage in cache_fields_by_stage],
        }

    def run():
        plan, dummy = _ORIGINALS["pipeline_plan"](
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=decode_tp_size,
            decode_tp_rank=decode_tp_rank,
            prefill_layout=prefill_layout,
            decode_layout=decode_layout,
            cache_fields_by_stage=cache_fields_by_stage,
        )
        return (plan, dummy), {"plan": _plan(plan), "dummy": list(dummy)}

    return _record("pipeline_plan", args, run)


def _planner_init(
    self,
    *,
    prefill_tp_size,
    decode_tp_size,
    prefill_layout,
    decode_layout,
    prefill_field_ids,
):
    def args():
        return {
            "prefill_tp_size": prefill_tp_size,
            "decode_tp_size": decode_tp_size,
            "prefill_layout": _contract(prefill_layout),
            "decode_layout": _contract(decode_layout),
            "prefill_field_ids": (
                None if prefill_field_ids is None else sorted(prefill_field_ids)
            ),
        }

    self._capture_args = args

    def run():
        _ORIGINALS["planner_init"](
            self,
            prefill_tp_size=prefill_tp_size,
            decode_tp_size=decode_tp_size,
            prefill_layout=prefill_layout,
            decode_layout=decode_layout,
            prefill_field_ids=prefill_field_ids,
        )
        return None, {
            "decode_ranks_by_prefill_rank": [
                [rank, sorted(ranks)]
                for rank, ranks in self.decode_ranks_by_prefill_rank.items()
            ],
            "sharded_group_ids": list(self.sharded_group_ids),
        }

    return _record("planner", args, run)


def _planner_plan(self, decode_tp_rank):
    def args():
        return {"planner": self._capture_args(), "decode_tp_rank": decode_tp_rank}

    def run():
        plan = _ORIGINALS["planner_plan"](self, decode_tp_rank)
        return plan, _plan(plan)

    return _record("planner_plan", args, run)


def _validate_owner_filters(*, group_specs, fragments, owner_filters):
    def args():
        return {
            "group_specs": [dataclasses.asdict(spec) for spec in group_specs],
            "fragments": [_fragment(fragment) for fragment in fragments],
            "owner_filters": _filters(owner_filters),
        }

    def run():
        _ORIGINALS["validate_owner_filters"](
            group_specs=group_specs, fragments=fragments, owner_filters=owner_filters
        )
        return None, True

    return _record("validate_owner_filters", args, run)


def _local_transfer_pages(
    source_blocks,
    destination_blocks,
    *,
    group_id,
    source_layout,
    destination_layout,
    source_owner_rank,
    destination_owner_rank,
):
    def args():
        return {
            "source_blocks": _ints(source_blocks),
            "destination_blocks": _ints(destination_blocks),
            "group_id": group_id,
            "source_layout": _contract(source_layout),
            "destination_layout": _contract(destination_layout),
            "source_owner_rank": source_owner_rank,
            "destination_owner_rank": destination_owner_rank,
        }

    def run():
        pages = _ORIGINALS["local_transfer_pages"](
            source_blocks,
            destination_blocks,
            group_id=group_id,
            source_layout=source_layout,
            destination_layout=destination_layout,
            source_owner_rank=source_owner_rank,
            destination_owner_rank=destination_owner_rank,
        )
        return pages, {"source": _ints(pages[0]), "destination": _ints(pages[1])}

    return _record("local_transfer_pages", args, run)


def _prepare_registration(kv_args, topology, registration):
    def args():
        return {
            "kv_args": _kv_args(kv_args),
            "topology": {
                "tp_size": topology.tp_size,
                "tp_rank": topology.tp_rank,
                "pp_rank": topology.pp_rank,
            },
            "registration": {
                "peer_layout": _contract(registration.peer_cache_layout),
                "decode_tp_size": registration.decode_tp_size,
                "decode_tp_rank": registration.decode_tp_rank,
            },
        }

    def run():
        planned = _ORIGINALS["prepare_registration"](kv_args, topology, registration)
        return planned, {
            "fragments": [
                _fragment(fragment) for fragment in planned.transfer_fragments
            ],
            "owner_filters": _filters(planned.transfer_owner_filters),
            "is_dummy": planned.is_dummy,
            "expected_decode_ranks": sorted(planned.expected_decode_ranks),
        }

    return _record("prepare_registration", args, run)


def _transfer_blocks(
    kv_args,
    *,
    dst_ptr,
    dst_tp_rank,
    src_block_manifest,
    dst_block_manifest,
    transfer_fragments=(),
    owner_filters,
    dst_cache_layout,
    block_selection=None,
    step_field_ids,
):
    def args():
        return {
            "cache_layout": _contract(kv_args.cache_layout),
            "kv_data_ptr": int(kv_args.kv_data_ptr),
            "dst_ptr": int(dst_ptr),
            "dst_tp_rank": int(dst_tp_rank),
            "src_manifest": _manifest(src_block_manifest),
            "dst_manifest": _manifest(dst_block_manifest),
            "fragments": [_fragment(fragment) for fragment in transfer_fragments],
            "owner_filters": _filters(owner_filters),
            "dst_layout": _contract(dst_cache_layout),
            "selection": _selection(block_selection),
            "step_field_ids": [
                None if fields is None else sorted(fields) for fields in step_field_ids
            ],
        }

    def run():
        steps = [
            list(step)
            for step in _ORIGINALS["transfer_blocks"](
                kv_args,
                dst_ptr=dst_ptr,
                dst_tp_rank=dst_tp_rank,
                src_block_manifest=src_block_manifest,
                dst_block_manifest=dst_block_manifest,
                transfer_fragments=transfer_fragments,
                owner_filters=owner_filters,
                dst_cache_layout=dst_cache_layout,
                block_selection=block_selection,
                step_field_ids=step_field_ids,
            )
        ]
        return iter(steps), [_host_copies(step) for step in steps]

    return _record("transfer_blocks", args, run)


def _registration_from_zmq(cls, msg):
    def args():
        return {"frames": [bytes(frame).hex() for frame in msg]}

    def run():
        parsed = _ORIGINALS["registration_from_zmq"](cls, msg)
        return parsed, {
            "endpoint": parsed.endpoint,
            "dst_port": parsed.dst_port,
            "session_id": parsed.mooncake_session_id,
            "dst_kv_ptr": parsed.dst_kv_ptr,
            "peer_layout": _contract(parsed.peer_cache_layout),
            "decode_tp_size": parsed.decode_tp_size,
            "decode_tp_rank": parsed.decode_tp_rank,
        }

    return _record("registration_from_zmq", args, run)


def _transfer_info_from_zmq(cls, msg):
    def args():
        return {"frames": [bytes(frame).hex() for frame in msg]}

    def run():
        parsed = _ORIGINALS["transfer_info_from_zmq"](cls, msg)
        return parsed, {
            "room": parsed.room,
            "session_id": parsed.mooncake_session_id,
            "manifest": _manifest(parsed.block_manifest),
        }

    return _record("transfer_info_from_zmq", args, run)


def _plan_route(self, prefill_info):
    def args():
        return {
            "decode_layout": _contract(self.kv_args.cache_layout),
            "decode_tp_size": self.topology.tp_size,
            "decode_tp_rank": self.topology.tp_rank,
            "prefill_info": prefill_info,
        }

    def run():
        targets, dummy, expected, dp_size = _ORIGINALS["plan_route"](self, prefill_info)
        return (targets, dummy, expected, dp_size), {
            "target_tp_ranks": list(targets),
            "dummy_tp_ranks": list(dummy),
            "expected_prefill_ranks": list(expected),
            "prefill_dp_size": dp_size,
        }

    return _record("plan_route", args, run)


# (key, owner, attribute, wrapper, is_classmethod)
_TARGETS = (
    (
        "contract_from_wire",
        cache_protocol.CacheTransferContract,
        "from_wire_bytes",
        _contract_from_wire,
        True,
    ),
    (
        "manifest_from_wire",
        cache_protocol.CachePDBlockManifest,
        "from_wire_bytes",
        _manifest_from_wire,
        True,
    ),
    (
        "validate_manifest",
        cache_protocol,
        "validate_cache_manifest",
        _validate_manifest,
        False,
    ),
    (
        "build_manifest",
        cache_protocol,
        "build_cache_block_manifest",
        _build_manifest,
        False,
    ),
    (
        "build_selection",
        cache_protocol,
        "build_cache_layerwise_block_selection",
        _build_selection,
        False,
    ),
    (
        "validate_peer_layout",
        cache_protocol,
        "validate_cache_peer_layout",
        _validate_peer_layout,
        False,
    ),
    (
        "pipeline_plan",
        transfer_plan,
        "build_pipeline_transfer_plan",
        _pipeline_plan,
        False,
    ),
    (
        "planner_init",
        transfer_plan.CacheTransferPlanner,
        "__init__",
        _planner_init,
        False,
    ),
    (
        "planner_plan",
        transfer_plan.CacheTransferPlanner,
        "plan_for_decode_rank",
        _planner_plan,
        False,
    ),
    (
        "validate_owner_filters",
        transfer_plan,
        "validate_rank_owner_filters",
        _validate_owner_filters,
        False,
    ),
    (
        "local_transfer_pages",
        transfer_plan,
        "local_transfer_pages",
        _local_transfer_pages,
        False,
    ),
    (
        "prepare_registration",
        mooncake_prefill,
        "prepare_decode_registration",
        _prepare_registration,
        False,
    ),
    (
        "transfer_blocks",
        mooncake_prefill,
        "cache_transfer_blocks",
        _transfer_blocks,
        False,
    ),
    (
        "registration_from_zmq",
        entities.KVArgsRegisterInfo,
        "from_zmq",
        _registration_from_zmq,
        True,
    ),
    (
        "transfer_info_from_zmq",
        entities.TransferInfo,
        "from_zmq",
        _transfer_info_from_zmq,
        True,
    ),
    ("plan_route", host_peer.HostDecodePeer, "plan_route", _plan_route, False),
)

_WRAPPED: dict[int, object] = {}


def _install() -> None:
    for key, owner, attribute, wrapper, is_classmethod in _TARGETS:
        original = owner.__dict__[attribute]
        if is_classmethod:
            _ORIGINALS[key] = original.__func__
            replacement = classmethod(wrapper)
        else:
            _ORIGINALS[key] = original
            replacement = functools.wraps(original)(wrapper)
        setattr(owner, attribute, replacement)
        if not is_classmethod and not isinstance(owner, type):
            _WRAPPED[id(original)] = replacement
    _rebind()


def _rebind() -> None:
    """Point every module's imported copy of a wrapped function at its
    wrapper (modules that imported it by name before the plugin ran)."""
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not namespace:
            continue
        for name, value in list(namespace.items()):
            replacement = _WRAPPED.get(id(value))
            if replacement is not None and replacement is not value:
                namespace[name] = replacement


_install()


def pytest_runtest_setup(item):
    _rebind()


_FRAGMENT_FIELDS = (
    "group_id",
    "field_id",
    "src_byte_offset",
    "dst_byte_offset",
    "src_row_stride_bytes",
    "dst_row_stride_bytes",
    "bytes_per_row",
    "rows_per_page",
)


def _compact(value):
    """Fragments as arrays in ``_FRAGMENT_FIELDS`` order."""
    if isinstance(value, dict):
        if set(value) == set(_FRAGMENT_FIELDS):
            return [value[key] for key in _FRAGMENT_FIELDS]
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def _transfer_signature(record) -> str:
    """What distinguishes one transfer's planning from another's: its two
    layouts, its route, its source kind, its destination rank, its step
    pattern and whether it was refused; the manifests' block ids aside."""
    args = record["args"]
    return json.dumps(
        [
            args["cache_layout"],
            args["dst_layout"],
            args["fragments"],
            args["owner_filters"],
            args["selection"] is None,
            args["dst_tp_rank"],
            [fields is None for fields in args["step_field_ids"]],
            "error" in record["result"],
        ],
        sort_keys=True,
    )


def _fixture() -> dict:
    """The records, once each; one transfer per signature, the smallest."""
    seen = set()
    records = []
    transfers: dict[str, dict] = {}
    for record in RECORDS:
        key = json.dumps([record["kind"], record["args"]], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if record["kind"] in ("contract_from_wire", "manifest_from_wire"):
            raw = bytes.fromhex(record["args"]["raw"])
            if raw.isascii():
                record = {**record, "args": {"text": raw.decode("ascii")}}
        if record["kind"] == "transfer_blocks":
            signature = _transfer_signature(record)
            kept = transfers.get(signature)
            size = len(json.dumps(record))
            if kept is None or size < len(json.dumps(kept)):
                transfers[signature] = record
            continue
        records.append(record)
    records.extend(transfers.values())
    used = set()

    def collect(value) -> None:
        if isinstance(value, str) and value in CONTRACTS:
            used.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect([record["args"] for record in records])
    collect([record["result"] for record in records])
    contracts = {key: wire for key, wire in CONTRACTS.items() if key in used}
    return {"contracts": contracts, "records": _compact(records)}


# The records' files, by the kinds each holds.
_FILES = {
    "manifests": (
        "build_manifest",
        "build_selection",
        "contract_from_wire",
        "manifest_from_wire",
        "registration_from_zmq",
        "transfer_info_from_zmq",
        "validate_manifest",
        "validate_peer_layout",
    ),
    "routes": (
        "local_transfer_pages",
        "pipeline_plan",
        "plan_route",
        "planner",
        "planner_plan",
        "prepare_registration",
        "validate_owner_filters",
    ),
    "transfers": ("transfer_blocks",),
}


def _write(path: str, value) -> None:
    with open(path, "w") as handle:
        json.dump(value, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("TS_PD_CAPTURE_OUT")
    if not out:
        return
    fixture = _fixture()
    os.makedirs(out, exist_ok=True)
    _write(os.path.join(out, "contracts.json"), fixture["contracts"])
    for name, kinds in _FILES.items():
        records = [record for record in fixture["records"] if record["kind"] in kinds]
        _write(os.path.join(out, f"{name}.json"), records)
