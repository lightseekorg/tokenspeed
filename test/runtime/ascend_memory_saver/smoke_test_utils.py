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

"""Shared launch and evidence helpers for direct Ascend smoke tests."""

import hashlib
import importlib.machinery
import json
import os
import socket
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _trace_file(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def trace_host_environment():
    """Bind rank-only tracing to the Python and packages this smoke test uses."""
    # Query the existing package path; importing _host in the launcher would
    # register its embedded scheduler beside the launcher's standalone one.
    package = sys.modules["tokenspeed_engine"]
    modules = {
        f"tokenspeed_engine.{name}": _trace_file(
            importlib.machinery.PathFinder.find_spec(name, package.__path__).origin
        )
        for name in ("host", "_host")
    }
    runtime = sys.modules["tokenspeed.runtime"]
    for parent, name in (
        ("engine", "rust_host"),
        ("utils", "ascend_memory_saver_adapter"),
    ):
        package = importlib.machinery.PathFinder.find_spec(parent, runtime.__path__)
        module = importlib.machinery.PathFinder.find_spec(
            name, package.submodule_search_locations
        )
        modules[f"tokenspeed.runtime.{parent}.{name}"] = _trace_file(module.origin)
    binding = json.dumps(
        {
            "python": sys.executable,
            "executable": _trace_file(sys.executable),
            "prefix": sys.prefix,
            "modules": modules,
        },
        sort_keys=True,
    )
    return {
        "GSM8K_TRACE_HOST_BINDING": binding,
        "GSM8K_TRACE_HOST_BINDING_SHA256": hashlib.sha256(binding.encode()).hexdigest(),
    }


def trace_environment(path):
    trace_dir = Path(__file__).resolve().parent / "trace"
    return {
        "GSM8K_GRAPH_TRACE": str(path),
        **trace_host_environment(),
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                (str(trace_dir), str(trace_dir.parent), os.environ.get("PYTHONPATH")),
            )
        ),
    }


def _module_host_argv(argv):
    return argv[1:3] == ["-m", "tokenspeed_engine.host"]


def trace_host_binding():
    """The launcher and ordinary Python must never acquire trace wrappers."""
    if not _module_host_argv(sys.orig_argv):
        return None
    raw = os.environ["GSM8K_TRACE_HOST_BINDING"]
    require(
        hashlib.sha256(raw.encode()).hexdigest()
        == os.environ["GSM8K_TRACE_HOST_BINDING_SHA256"],
        "Trace binding changed",
    )
    binding = json.loads(raw)
    require(
        sys.executable == binding["python"] and sys.orig_argv[0] == binding["python"],
        "Trace Python differs from selected interpreter",
    )
    require(
        _trace_file(sys.executable) == binding["executable"]
        and Path("/proc/self/exe").resolve(strict=True)
        == Path(binding["executable"]["path"]),
        "Trace process executable changed",
    )
    require(sys.prefix == binding["prefix"], "Trace Python prefix changed")
    return binding


def _trace_loaded_module(name, expected, *, native):
    module = sys.modules[name]
    require(
        _trace_file(module.__file__) == expected
        and Path(module.__spec__.origin).resolve(strict=True) == Path(expected["path"]),
        f"Loaded trace module differs from selected source: {name}",
    )
    if native:
        require(
            isinstance(module.__spec__.loader, importlib.machinery.ExtensionFileLoader),
            "Trace host is not the native extension",
        )
    return Path(expected["path"])


def verify_trace_host(binding, target):
    """Verify already loaded modules before installing any trace wrapper."""
    modules = binding["modules"]
    main = sys.modules["__main__"]
    require(
        main.__spec__ is not None and main.__spec__.name == "tokenspeed_engine.host",
        "Trace is outside the real module host",
    )
    host = _trace_loaded_module(
        "__main__", modules["tokenspeed_engine.host"], native=False
    )
    native = _trace_loaded_module(
        "tokenspeed_engine._host", modules["tokenspeed_engine._host"], native=True
    )
    runtime = _trace_loaded_module(
        "tokenspeed.runtime.engine.rust_host",
        modules["tokenspeed.runtime.engine.rust_host"],
        native=False,
    )
    require(
        host.parent == native.parent, "Trace host Python and native packages differ"
    )
    if target == "tokenspeed.runtime.utils.ascend_memory_saver_adapter":
        adapter = _trace_loaded_module(target, modules[target], native=False)
        require(
            adapter.is_relative_to(runtime.parents[1]), "Trace runtime packages differ"
        )
    return {
        "python": binding["python"],
        "host": str(host),
        "native": modules["tokenspeed_engine._host"],
        "runtime": str(runtime),
    }


def weight_ownership_bytes(witness, blocks, *, has_draft, retain_draft):
    """Join actual PTA segment owners to native blocks and parameter storage."""
    require(
        set(witness["roles"]) == ({"target", "draft"} if has_draft else {"target"}),
        "Unexpected model roles",
    )
    pools = {tuple(pool["id"]): pool for pool in witness["pools"]}
    require(len(pools) == len(witness["pools"]), "Ambiguous owned PTA pool")
    keys = {(pool["tag"], pool["policy"], pool["scope"]) for pool in pools.values()}
    require(len(keys) == len(pools), "Duplicate owned PoolKey")
    native_count = len(blocks)
    blocks = {int(block["ptr"], 16): block for block in blocks}
    require(len(blocks) == native_count, "Duplicate native weight address")
    owners = {}
    segments = []
    totals = dict(discard_parameters=0, retained_draft=0, persistent_buffers=0)
    for segment in witness["segments"]:
        if segment["device"] != witness["device"]:
            continue
        start, size = segment["address"], segment["total_size"]
        pool = pools.get(tuple(segment["segment_pool_id"]))
        if pool is None or pool["tag"] != "weights":
            require(
                all(
                    start + size <= base or base + block["aligned_bytes"] <= start
                    for base, block in blocks.items()
                ),
                "Foreign PTA segment overlaps native weights",
            )
            continue
        matches = [
            base
            for base, block in blocks.items()
            if size > 0
            and base <= start
            and start + size <= base + block["aligned_bytes"]
        ]
        require(len(matches) == 1, "PTA segment lacks one complete native weight span")
        base = matches[0]
        require(
            all(start + size <= low or high <= start for low, high, _ in segments),
            "Duplicate or overlapping PTA weight segment",
        )
        if (pool["scope"], pool["policy"]) == ("persistent_buffers", "cpu_backup"):
            kind = "persistent_buffers"
        elif (pool["scope"], pool["policy"]) == ("weights", "discard"):
            kind = "discard_parameters"
        else:
            require(
                retain_draft
                and has_draft
                and (pool["scope"], pool["policy"]) == ("weights", "cpu_backup"),
                "Unexpected backup weight owner",
            )
            kind = "retained_draft"
        require(
            blocks[base]["policy"]
            == ("discard" if kind == "discard_parameters" else "backup"),
            "Native and PTA owner policies differ",
        )
        owner = kind, tuple(pool["id"])
        require(
            owners.get(base, owner) == owner, "Native block has conflicting PTA owners"
        )
        if base not in owners:
            totals[kind] += blocks[base]["aligned_bytes"]
        owners[base] = owner
        segments.append((start, start + size, kind))
    require(
        owners.keys() == blocks.keys(), "Native weight block has no owned PTA segment"
    )
    storage_bytes = dict(target=0, independent_draft=0)
    seen = set()
    parameter_roles = set()
    # The target owns exact shared storage, regardless of the draft policy.
    for parameter in sorted(
        witness["parameters"], key=lambda item: item["role"] != "target"
    ):
        role = parameter["role"]
        require(
            role in witness["roles"] and parameter["bytes"] > 0,
            "Unexpected parameter storage",
        )
        parameter_roles.add(role)
        extent = parameter["ptr"], parameter["ptr"] + parameter["bytes"]
        if extent in seen:
            continue
        require(
            all(extent[1] <= start or end <= extent[0] for start, end in seen),
            "Overlapping parameter storage",
        )
        seen.add(extent)
        kinds = [
            kind
            for start, end, kind in segments
            if start <= extent[0] and extent[1] <= end
        ]
        expected = (
            "retained_draft"
            if role == "draft" and retain_draft
            else "discard_parameters"
        )
        require(kinds == [expected], f"Wrong pool for {role}.{parameter['name']}")
        storage_bytes[
            "target" if role == "target" else "independent_draft"
        ] += parameter["bytes"]
    require(
        parameter_roles == set(witness["roles"])
        and storage_bytes["target"] > 0
        and totals["discard_parameters"] > 0
        and totals["persistent_buffers"] > 0,
        "Missing target or persistent buffer ownership",
    )
    require(
        bool(totals["retained_draft"])
        == bool(retain_draft and storage_bytes["independent_draft"]),
        "Retained draft bytes lack independent draft parameters",
    )
    return {"native_aligned_bytes": totals, "parameter_storage_bytes": storage_bytes}


def make_server_args(args):
    caps = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= args.concurrency]
    if caps[-1] != args.concurrency:
        caps.append(args.concurrency)
    return [
        "--model",
        str(args.model),
        "--device",
        "npu",
        "--tensor-parallel-size",
        str(args.tp_size),
        "--dist-init-addr",
        args.dist_init_addr,
        "--dtype",
        "bfloat16",
        "--seed",
        str(args.seed),
        "--attention-backend",
        "mha",
        "--sampling-backend",
        "greedy",
        "--enable-memory-saver",
        "--weight-memory-policy",
        "cpu_backup",
        "--disable-prefill-graph",
        "--disable-cudagraph-memory-reserve",
        "--disable-pdl",
        "--disable-autotune",
        "--disable-prefix-caching",
        "--disable-kvstore",
        "--enable-log-request-stats",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(args.concurrency),
        "--max-total-tokens",
        str(max(16384, args.concurrency * 1024)),
        "--chunked-prefill-size",
        "4096",
        "--max-cudagraph-capture-size",
        str(args.concurrency),
        "--stream-interval",
        "1",
        "--cudagraph-capture-sizes",
        *map(str, caps),
    ]


def ephemeral_loopback_address():
    """Choose a local port and leave space for TokenSpeed's derived port cluster."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        if port <= 65529:
            return f"127.0.0.1:{port}"


def read_trace(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(
        not any(row["event"] == "host_identity_error" for row in rows),
        "Host identity verification failed",
    )
    require(
        any(row["event"] == "host_identity" for row in rows),
        "Missing verified module-host trace",
    )
    return rows


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def run_and_report(args, run, report):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report["status"] = "failed"
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        write_json(args.output_dir / "report.json", report)
