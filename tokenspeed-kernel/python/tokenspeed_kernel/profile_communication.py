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

"""Bind rank/link evidence once at profile startup; annotate selected Op routes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from tokenspeed_kernel.op_profile import _profile_process_identity, current_op_metadata

_binding: dict | None = None
_group_observations: dict[tuple, dict] = {}
_workspace_allocations: list[dict] = []


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _cuda_uuid(device) -> str:
    cuda_uuid = torch.cuda.get_device_properties(device).uuid
    if isinstance(cuda_uuid, bytes) and len(cuda_uuid) == 16:
        cuda_uuid = str(uuid.UUID(bytes=cuda_uuid))
    cuda_uuid = _text(cuda_uuid)
    return cuda_uuid if cuda_uuid.startswith("GPU-") else f"GPU-{cuda_uuid}"


def _probe(device, *, include_topology: bool) -> dict:
    """Capture raw state on this rank, before any profiling capture/replay."""
    result = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        "process_identity": _profile_process_identity(),
    }
    result["uuid"] = _cuda_uuid(device)
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(result["uuid"])
        result.update(
            nvml_uuid=_text(pynvml.nvmlDeviceGetUUID(handle)),
            physical_index=pynvml.nvmlDeviceGetIndex(handle),
            driver=_text(pynvml.nvmlSystemGetDriverVersion()),
            nvml=_text(pynvml.nvmlSystemGetNVMLVersion()),
            switch_device_type=int(pynvml.NVML_NVLINK_DEVICE_TYPE_SWITCH),
            links=[],
        )
        if result["nvml_uuid"].lower() != result["uuid"].lower():
            raise ValueError("CUDA and NVML UUID differ")
        try:
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            current_mig_mode, pending_mig_mode = pynvml.nvmlDeviceGetMigMode(handle)
            result.update(
                pci_device_id=int(pci.pciDeviceId),
                pci_subsystem_id=int(pci.pciSubSystemId),
                mig_mode=int(current_mig_mode),
                pending_mig_mode=int(pending_mig_mode),
                full_device=result["nvml_uuid"].startswith("GPU-")
                and current_mig_mode == pynvml.NVML_DEVICE_MIG_DISABLE,
            )
        except (AttributeError, pynvml.NVMLError) as error:
            result["board_identity_unavailable"] = f"{type(error).__name__}: {error}"
        for link in range(pynvml.NVML_NVLINK_MAX_LINKS):
            try:
                active = bool(pynvml.nvmlDeviceGetNvLinkState(handle, link))
            except pynvml.NVMLError as error:
                result["links"].append({"link": link, "unavailable": str(error)})
                continue
            row = {"link": link, "active": active}
            result["links"].append(row)
            if active:
                try:
                    row["version"] = int(
                        pynvml.nvmlDeviceGetNvLinkVersion(handle, link)
                    )
                except pynvml.NVMLError as error:
                    row["version_unavailable"] = str(error)
                row.update(
                    remote_device_type=int(
                        pynvml.nvmlDeviceGetNvLinkRemoteDeviceType(handle, link)
                    ),
                    remote_pci=_text(
                        pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link).busId
                    ),
                )
        # Driver status is retained verbatim. Its displayed rate is only a
        # matching predicate; the published profile supplies the denominator.
        commands = [
            ("link_status", ["nvidia-smi", "nvlink", "-s", "-i", result["uuid"]])
        ]
        if include_topology:
            commands.append(("topology", ["nvidia-smi", "topo", "-m"]))
        for key, arguments in commands:
            completed = subprocess.run(
                arguments, capture_output=True, text=True, timeout=20, check=False
            )
            result[key] = {
                "argv": arguments,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        rates = {}
        if result["link_status"]["returncode"] == 0:
            for link, rate in re.findall(
                r"Link\s+(\d+):\s+([0-9.]+)\s+GB/s", result["link_status"]["stdout"]
            ):
                rates[int(link)] = float(rate) * 1e9
        for row in result["links"]:
            if row.get("active"):
                row["raw_status_bytes_per_second"] = rates.get(row["link"])
    finally:
        pynvml.nvmlShutdown()
    return result


def bind_communication_profile(
    *, run_id: str, config: dict, rank: int, device, cpu_group
) -> None:
    """All runtime ranks call once, including ranks excluded from collection.

    A failed local probe still participates in the same CPU gather. The bound
    snapshot is immutable for this run and never refreshed on the replay path.
    """
    global _binding
    config_sha = _sha(_json(config))
    if _binding is not None:
        if (_binding["run_id"], _binding["config_sha256"], _binding["rank"]) != (
            run_id,
            config_sha,
            rank,
        ):
            raise ValueError(
                "Communication profiling already bound to another run/config/rank"
            )
        return
    local = dict(run_id=run_id, config_sha256=config_sha, rank=rank)
    try:
        local.update(_probe(device, include_topology=dist.get_rank(cpu_group) == 0))
    except Exception as error:
        local["unavailable"] = f"{type(error).__name__}: {error}"
    gathered = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(gathered, local, group=cpu_group)
    ranks = tuple(dist.get_process_group_ranks(cpu_group))
    valid = all(
        isinstance(item, dict)
        and item.get("run_id") == run_id
        and item.get("config_sha256") == config_sha
        and item.get("rank") == expected
        for expected, item in zip(ranks, gathered, strict=True)
    )
    if not valid:
        raise ValueError("Communication evidence run/config/rank mismatch")
    kernel = Path(__file__).parent
    source_paths = (
        Path(__file__),
        kernel / "ops/communication/trtllm.py",
        kernel / "thirdparty/cuda/trtllm.py",
        kernel / "thirdparty/cuda/csrc/trtllm_mnnvl_allreduce_fusion.cu",
        kernel
        / "thirdparty/cuda/csrc/include/flashinfer/comm/trtllm_mnnvl_allreduce_fusion.cuh",
        kernel / "thirdparty/cuda/csrc/include/flashinfer/vec_dtypes.cuh",
    )
    sources = {
        str(path.relative_to(kernel)): (
            _sha(path.read_bytes()) if path.is_file() else None
        )
        for path in source_paths
    }
    snapshot = dict(
        link_version_definition={
            "api": "nvmlDeviceGetNvLinkVersion",
            "value_kind": "driver_api_raw",
        },
        schema="tokenspeed.communication_evidence/v1",
        run_id=run_id,
        config_sha256=config_sha,
        rank=rank,
        rank_evidence=gathered,
        route_sources=sources,
    )
    contents = _json(snapshot)
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank-{rank}.communication.json"
    with path.open("xb") as stream:
        stream.write(contents)
    _binding = dict(snapshot, sha256=_sha(contents), path=str(path.resolve()))
    # Model construction can allocate before the runtime creates its profiler.
    # These are host-only facts, retained without extending a GPU buffer's life.
    for allocation in _workspace_allocations:
        _freeze_workspace_allocation(allocation)


def bind_mnnvl_workspace(workspace, *, group) -> dict | None:
    """Record the actual symmetric allocator/mapping once, outside capture.

    Returns host-only allocation evidence for the manager to retain, or None
    when profiling is disabled. Unknown APIs/backends do not affect execution;
    they leave communication normalization unavailable. No rendezvous, device
    tensor read, allocation or new collective is performed here.
    """
    if _binding is None and not os.environ.get("TOKENSPEED_TIMELINE_CONFIG"):
        return None
    allocation = {
        "workspace_identity": id(workspace),
        "ranks": tuple(dist.get_process_group_ranks(group)),
        "group_name": group.group_name,
        "mapping_verified": False,
    }
    try:
        import torch.distributed._symmetric_memory as symm_mem

        buffer, handle = workspace._refs
        peers = tuple(int(pointer) for pointer in handle.buffer_ptrs)
        local_rank = dist.get_rank(group)
        allocation.update(
            allocator_backend=symm_mem.get_backend(buffer.device),
            torch_version=torch.__version__,
            torch_git_version=torch.version.git_version,
            allocator_python_path=symm_mem.__file__,
            allocator_python_sha256=_sha(Path(symm_mem.__file__).read_bytes()),
            device=str(buffer.device),
            uuid=_cuda_uuid(buffer.device),
            rank=local_rank,
            world_size=handle.world_size,
            multicast_ptr=int(handle.multicast_ptr or 0),
            local_ptr=int(workspace.local_ptr),
            peer_count=len(peers),
            buffer_size=handle.buffer_size,
        )
        allocation["mapping_verified"] = (
            allocation["allocator_backend"] == "CUDA"
            and buffer.is_cuda
            and handle.device == buffer.device == workspace.peer_ptrs.device
            and handle.rank == local_rank == workspace.tp_rank
            and handle.world_size == len(allocation["ranks"]) == workspace.tp_size
            and allocation["multicast_ptr"] > 0
            and allocation["multicast_ptr"] == workspace.multicast_ptr
            and len(peers) == len(set(peers)) == handle.world_size
            and all(pointer > 0 for pointer in peers)
            and peers[local_rank] == workspace.local_ptr
            and workspace.peer_ptrs.numel() == handle.world_size
            and workspace.buffer_size_bytes > 0
            and 3 * workspace.buffer_size_bytes <= handle.buffer_size
        )
        if not allocation["mapping_verified"]:
            allocation["unavailable"] = (
                "symmetric allocator/mapping/rank/geometry predicate mismatch"
            )
    except Exception as error:
        allocation["unavailable"] = f"{type(error).__name__}: {error}"
    _workspace_allocations.append(allocation)
    if _binding is not None:
        _freeze_workspace_allocation(allocation)
    return allocation


def _freeze_workspace_allocation(allocation: dict) -> None:
    if "source" in allocation:
        return
    local = next(
        row for row in _binding["rank_evidence"] if row["rank"] == _binding["rank"]
    )
    if allocation.get("uuid", "").lower() != local.get("uuid", "").lower():
        allocation["mapping_verified"] = False
        allocation["unavailable"] = (
            "workspace device UUID differs from communication evidence"
        )
    snapshot = {
        "schema": "tokenspeed.mnnvl_allocation_evidence/v1",
        "run_id": _binding["run_id"],
        "rank": _binding["rank"],
        "config_sha256": _binding["config_sha256"],
        "communication_snapshot_sha256": _binding["sha256"],
        "allocation": allocation.copy(),
    }
    contents = _json(snapshot)
    digest = _sha(contents)
    path = Path(_binding["path"]).with_name(
        f"rank-{_binding['rank']}.mnnvl-allocation-{digest}.json"
    )
    with path.open("xb") as stream:
        stream.write(contents)
    allocation["source"] = f"{path}#sha256={digest}"
    allocation["communication_snapshot_sha256"] = _binding["sha256"]


def _nvlink_group(
    rows: list[dict], topology_evidence: dict
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    """Prove a switched peer route and retain switch addresses and link counts."""
    host = topology_evidence.get("host")
    topology = topology_evidence.get("topology", {})
    if (
        not host
        or topology.get("returncode") != 0
        or any(row.get("host") != host or row.get("unavailable") for row in rows)
    ):
        return None
    indices = [row.get("physical_index") for row in rows]
    if any(type(index) is not int or index < 0 for index in indices) or len(
        set(indices)
    ) != len(indices):
        return None
    # nvidia-smi can underline its header with ANSI SGR formatting even when
    # captured to a pipe. Keep raw evidence and parse plain tokens.
    text = re.sub(r"\x1b\[[0-9;]*m", "", topology["stdout"])
    lines = [line.split() for line in text.splitlines()]
    header = next((line for line in lines if line and line[0] == "GPU0"), None)
    if header is None:
        return None
    table = {
        line[0]: line[1:]
        for line in lines
        if line and re.fullmatch(r"GPU\d+", line[0]) and line is not header
    }
    switches = []
    peer_link_counts = set()
    for row in rows:
        active = [link for link in row.get("links", ()) if link.get("active")]
        switch_type = row.get("switch_device_type")
        if (
            not active
            or type(switch_type) is not int
            or any(
                link.get("remote_device_type") != switch_type
                or not isinstance(link.get("remote_pci"), str)
                or not link["remote_pci"]
                for link in active
            )
        ):
            return None
        switches.append({link["remote_pci"] for link in active})
        own = table.get(f"GPU{row['physical_index']}")
        if own is None:
            return None
        for index in indices:
            if index == row["physical_index"]:
                continue
            label = f"GPU{index}"
            if (
                header.count(label) != 1
                or header.index(label) >= len(own)
                or not re.fullmatch(r"NV[1-9]\d*", own[header.index(label)])
            ):
                return None
            peer_link_counts.add(int(own[header.index(label)][2:]))
    if not switches or any(ports != switches[0] for ports in switches):
        return None
    return tuple(sorted(switches[0])), tuple(sorted(peer_link_counts))


def _group_observation(ranks: tuple[int, ...], group_name) -> dict:
    key = (ranks, group_name)
    if _binding is None:
        return {}
    if key not in _group_observations:
        _group_observations[key] = _group_observation_from_snapshot(
            _binding, ranks, group_name
        )
    return _group_observations[key]


def _group_observation_from_snapshot(
    binding: dict, ranks: tuple[int, ...], group_name
) -> dict:
    evidence = {row["rank"]: row for row in binding["rank_evidence"]}
    if any(rank not in evidence for rank in ranks):
        return {}
    rows = [evidence[rank] for rank in ranks]
    source = f"{binding['path']}#sha256={binding['sha256']}"
    result = dict(
        comm_group=f"nccl:{group_name!r}:{ranks}",
        comm_world_size=len(ranks),
        comm_ranks=ranks,
        comm_rank=binding["rank"],
        comm_scope="per_rank",
        comm_direction="one_direction",
        comm_normalization="nccl_tests_bus_bytes",
        comm_placement_source=source + "#rank_evidence",
        comm_route_source=source
        + "#route_sources; selected decision in this Op scalar_args",
    )
    if all(row.get("host") and row.get("uuid") for row in rows):
        result.update(
            comm_rank_hosts=tuple(row["host"] for row in rows),
            comm_rank_device_uuids=tuple(row["uuid"] for row in rows),
            comm_placement=(
                "single_node"
                if len({row["host"] for row in rows}) == 1
                else "multi_node"
            ),
        )
    active = [
        [link for link in row.get("links", ()) if link.get("active")] for row in rows
    ]
    if all(
        links
        and all(
            type(link.get("raw_status_bytes_per_second")) in (int, float)
            for link in links
        )
        for links in active
    ):
        result.update(
            comm_rank_active_link_counts=tuple(len(links) for links in active),
            comm_rank_min_link_rate_bytes_per_second=tuple(
                min(link["raw_status_bytes_per_second"] for link in links)
                for links in active
            ),
            comm_link_source=source + "#rank_evidence.links+link_status",
        )
        if all(
            all(type(link.get("version")) is int for link in links)
            and len({link["version"] for link in links}) == 1
            for links in active
        ):
            result["comm_rank_link_versions"] = tuple(
                links[0]["version"] for links in active
            )
    topology = _nvlink_group(rows, binding["rank_evidence"][0])
    if topology is not None:
        result.update(
            comm_topology="single_node_nvswitch",
            comm_topology_source=source + "#rank_evidence.links+topology",
        )
        if "comm_link_source" in result and all(
            type(row.get("pci_device_id")) is int
            and type(row.get("pci_subsystem_id")) is int
            and type(row.get("full_device")) is bool
            and isinstance(row.get("nvml_uuid"), str)
            and row["nvml_uuid"].lower() == row.get("uuid", "").lower()
            and all(
                type(link.get("link")) is int and link["link"] >= 0 for link in links
            )
            and len({link["link"] for link in links}) == len(links)
            for row, links in zip(rows, active, strict=True)
        ):
            switch_pci, peer_link_counts = topology
            result.update(
                comm_rank_pci_device_ids=tuple(row["pci_device_id"] for row in rows),
                comm_rank_pci_subsystem_ids=tuple(
                    row["pci_subsystem_id"] for row in rows
                ),
                comm_rank_full_devices=tuple(row["full_device"] for row in rows),
                comm_switch_pci_addresses=switch_pci,
                comm_peer_nvlink_counts=peer_link_counts,
                comm_rank_max_link_rate_bytes_per_second=tuple(
                    max(link["raw_status_bytes_per_second"] for link in links)
                    for links in active
                ),
                comm_nvlink_board_source=source
                + "#rank_evidence.pci+mig+links+topology",
            )
    return result


def _mnnvl_route_observation(
    metadata: dict,
    binding: dict | None,
    allocation: dict | None,
    ranks: tuple[int, ...],
    group_name,
    *,
    twoshot_geometry: dict | None = None,
) -> dict:
    algorithm = metadata.get("comm_algorithm")
    if not (
        metadata.get("comm_backend") == "trtllm"
        and metadata.get("comm_workspace") == "mnnvl"
        and metadata.get("comm_protocol") == "mnnvl"
        and (algorithm == "oneshot" or (algorithm == "twoshot" and twoshot_geometry))
        and metadata.get("comm_pattern") in (0, 1)
        and metadata.get("comm_topology") == "single_node_nvswitch"
        and metadata.get("comm_placement") == "single_node"
        and binding is not None
        and allocation is not None
        and allocation.get("mapping_verified") is True
        and allocation.get("source")
        and allocation.get("communication_snapshot_sha256") == binding["sha256"]
        and tuple(allocation["ranks"]) == ranks
        and allocation["group_name"] == group_name
        and ranks[allocation["rank"]] == binding["rank"]
        and all(binding["route_sources"].get(path) for path in _MNNVL_ROUTE_SOURCES)
    ):
        return {}
    if twoshot_geometry is None:
        model = "trtllm_mnnvl_oneshot_remote_payload"
        addressing = "nvls_multicast"
        model_source = "one complete input MC store; other-rank replica payload delivery; local poll/reduce"
    else:
        model = "trtllm_mnnvl_twoshot_remote_payload"
        addressing = "nvlink_peer_unicast_then_nvls_multicast"
        model_source = "native two-shot kernel; unicast scatter to token owner; owner multicasts reduced vector"
    return dict(
        comm_transport="nvlink",
        comm_addressing=addressing,
        comm_allocator_backend="CUDA",
        comm_allocation_source=allocation["source"],
        comm_normalization=model,
        comm_payload_model=model,
        comm_payload_model_source=metadata["comm_route_source"]
        + "#trtllm_mnnvl_allreduce_fusion.cuh: "
        + model_source,
        comm_transport_source=allocation["source"]
        + "; "
        + metadata["comm_route_source"]
        + "; "
        + metadata["comm_topology_source"],
        **(twoshot_geometry or {}),
    )


_MNNVL_FUSION_HEADER = (
    "thirdparty/cuda/csrc/include/flashinfer/comm/trtllm_mnnvl_allreduce_fusion.cuh"
)
_MNNVL_TWOSHOT_SOURCE_SHA256 = (
    "e18ef6e6482ddc4d9660e4ee05bd0d1fc5b5162593b1191442783aa5695388bf"
)
_MNNVL_ROUTE_SOURCES = (
    "thirdparty/cuda/trtllm.py",
    "thirdparty/cuda/csrc/trtllm_mnnvl_allreduce_fusion.cu",
    _MNNVL_FUSION_HEADER,
    "thirdparty/cuda/csrc/include/flashinfer/vec_dtypes.cuh",
)


def _mnnvl_twoshot_geometry(op, activities, binding, ranks) -> dict:
    """Prove the reviewed scatter/multicast launch and recover its token geometry."""
    observation = op["scalar_args"]
    pattern = observation.get("comm_pattern")
    spec = {
        0: ("communication.all_reduce", "trtllm_all_reduce", "tensor"),
        1: (
            "communication.all_reduce_residual_rmsnorm",
            "trtllm_allreduce_residual_rmsnorm",
            "input_tensor",
        ),
    }
    if (
        type(pattern) is not int
        or pattern not in spec
        or observation.get("comm_use_oneshot") is not False
        or len(ranks) not in (2, 4, 8, 16)
        or binding["route_sources"].get(_MNNVL_FUSION_HEADER)
        != _MNNVL_TWOSHOT_SOURCE_SHA256
        or type(observation.get("comm_payload_element_size")) is not int
        or observation["comm_payload_element_size"] != 2
        or len(activities) != 1
    ):
        raise ValueError(
            "reviewed MNNVL two-shot source/selection/activity unavailable"
        )
    op_name, implementation, input_name = spec[pattern]
    tensor = op["inputs"][input_name]
    shape = tensor["shape"]
    native_type = {"torch.bfloat16": "__nv_bfloat16", "torch.float16": "__half"}.get(
        tensor["dtype"]
    )
    if (
        (op.get("op_name"), op.get("implementation")) != (op_name, implementation)
        or native_type is None
        or not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        or shape[0] > 2048
        or shape[1] % 8
        or tuple(tensor["stride"]) != (shape[1], 1)
        or tensor["logical_bytes"] != shape[0] * shape[1] * 2
        or tensor["logical_bytes"] != observation.get("comm_payload_bytes")
    ):
        raise ValueError("MNNVL two-shot payload geometry differs from the Op input")
    activity = activities[0]
    kernel = activity.get("kernel") or {}
    name = kernel.get("demangled_name") or ""
    kernel_name = "mnnvl_allreduce_fusion_kernel_twoshot"
    prefix = (
        "void flashinfer::trtllm_mnnvl_allreduce_fusion::"
        + kernel_name
        + "<(flashinfer::trtllm_allreduce_fusion::AllReduceFusionPattern)"
        + f"{pattern}, {native_type}, (int){len(ranks)}, "
    )
    grid, block, cluster = (kernel.get(key) for key in ("grid", "block", "cluster"))
    if (
        activity.get("kind") != "kernel"
        or activity.get("table") != "CUPTI_ACTIVITY_KIND_KERNEL"
        or type(activity.get("source_rowid")) is not int
        or activity["source_rowid"] <= 0
        or kernel.get("short_name") != kernel_name
        or not isinstance(kernel.get("name"), str)
        or not kernel["name"]
        or not name.startswith(prefix)
        or not all(
            isinstance(dims, (list, tuple))
            and len(dims) == 3
            and all(type(dim) is int and dim > 0 for dim in dims)
            for dims in (grid, block, cluster)
        )
        or tuple(grid) != (shape[0], cluster[1], 1)
        or tuple(cluster) != (1, cluster[1], 1)
        or tuple(block[1:]) != (1, 1)
        or block[0] * cluster[1] * 8 != shape[1]
    ):
        raise ValueError("MNNVL two-shot native kernel/launch geometry differs")
    return {
        "comm_payload_tokens": shape[0],
        "comm_payload_bytes_per_token": shape[1] * 2,
        "comm_native_kernel": kernel["name"],
        "comm_native_kernel_rowid": activity["source_rowid"],
    }


def record_communication_route(
    *,
    ranks: tuple[int, ...],
    group_name,
    backend: str,
    workspace: str | None,
    algorithm: str | None,
    protocol: str | None,
    mnnvl_allocation: dict | None = None,
) -> None:
    """Copy cached evidence and this call's selected host decision; no probes."""
    metadata = current_op_metadata()
    if metadata is None:
        return
    metadata.update(ranks=ranks, group_name=group_name)
    metadata.update(_group_observation(ranks, group_name))
    # A selected IPC peer-pointer Lamport kernel on a proven switched NVLink
    # group uses its NVLink peer route. Allocation or topology alone is not the
    # selection evidence. MNNVL additionally requires its actual CUDA symmetric
    # allocation and NVLS multicast mapping, never just a workspace name.
    metadata.update(comm_backend=backend, comm_workspace=workspace)
    for key in (
        "comm_addressing",
        "comm_allocator_backend",
        "comm_allocation_source",
        "comm_payload_model",
        "comm_payload_model_source",
    ):
        metadata.pop(key, None)
    for key, value in (("comm_algorithm", algorithm), ("comm_protocol", protocol)):
        if value is not None:
            metadata[key] = value
        else:
            metadata.pop(key, None)
    if (
        backend == "trtllm"
        and workspace == "ipc"
        and protocol == "lamport"
        and metadata.get("comm_topology") == "single_node_nvswitch"
    ):
        metadata.update(
            comm_transport="nvlink",
            comm_transport_source=metadata["comm_route_source"]
            + "; "
            + metadata["comm_topology_source"],
        )
    else:
        route = _mnnvl_route_observation(
            metadata, _binding, mnnvl_allocation, ranks, group_name
        )
        if route:
            metadata.update(route)
        else:
            metadata.pop("comm_transport", None)
            metadata.pop("comm_transport_source", None)


def bind_capture_communication(
    capture: dict,
    directory: Path,
    *,
    sealed_artifacts: dict[str, dict] | None = None,
    native_op_activities: dict[str, list[dict]] | None = None,
) -> dict[str, dict]:
    """Rebind missing static route evidence from same-run hashed receipts.

    No raw artifact is changed, no transport is inferred from names, and no
    machine/device query runs offline. Unknown or ambiguous mappings stay absent.
    """
    pending = [
        op
        for op in capture["ops"]
        if op["scalar_args"].get("comm_backend") == "trtllm"
        and op["scalar_args"].get("comm_workspace") == "mnnvl"
        and not op["scalar_args"].get("comm_transport")
    ]

    def read(path, expected=None):
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("communication receipt byte budget exceeded")
        contents = path.read_bytes()
        digest = _sha(contents)
        if sealed_artifacts is not None:
            source = sealed_artifacts.get(str(path.resolve()))
            if (
                source is None
                or source["bytes"] != len(contents)
                or source["sha256"] != digest
            ):
                raise ValueError(
                    "Communication source differs from sealed run inventory"
                )
        if expected is not None and digest != expected:
            raise ValueError("communication receipt SHA256 mismatch")
        return json.loads(contents), digest

    if sealed_artifacts is not None:
        verified_references = set()
        for op in capture["ops"]:
            observation = op["scalar_args"]
            if observation.get("comm_addressing") != "nvls_multicast":
                continue
            for field in ("comm_placement_source", "comm_allocation_source"):
                original, separator, suffix = observation[field].partition("#sha256=")
                digest = suffix[:64]
                if not separator or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError(
                        "Recorded multicast route lacks sealed receipt identity"
                    )
                path = directory / Path(original).name
                if (path, digest) not in verified_references:
                    read(path, digest)
                    verified_references.add((path, digest))

    result = {}
    cache = {}
    for op in pending:
        observation = op["scalar_args"].copy()
        try:
            ranks = tuple(observation["ranks"])
            group = observation["group_name"]
            reference = observation["comm_placement_source"]
            original, separator, suffix = reference.partition("#sha256=")
            digest = suffix[:64]
            if not separator or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("hashed communication snapshot reference unavailable")
            key = (original, digest, ranks, group)
            if key not in cache:
                snapshot, _ = read(directory / Path(original).name, digest)
                config, _ = read(directory / f"rank-{capture['rank']}.config.json")
                if (
                    snapshot.get("schema") != "tokenspeed.communication_evidence/v1"
                    or (snapshot.get("run_id"), snapshot.get("rank"))
                    != (capture["run_id"], capture["rank"])
                    or config.get("run_id") != capture["run_id"]
                    or config.get("rank") != capture["rank"]
                    or snapshot.get("config_sha256") != config.get("sha256")
                    or _sha(_json(config["config"])) != config["sha256"]
                ):
                    raise ValueError("communication run/rank/config identity mismatch")
                peers = {row["rank"]: row for row in snapshot["rank_evidence"]}
                if len(peers) != len(snapshot["rank_evidence"]) or any(
                    rank not in peers for rank in ranks
                ):
                    raise ValueError("ordered communication peer identity unavailable")
                if peers[capture["rank"]].get("pid") != capture["process_id"]:
                    raise ValueError(
                        "communication worker process differs from capture"
                    )
                source_paths = ("ops/communication/trtllm.py", *_MNNVL_ROUTE_SOURCES)
                trusted_sources = {
                    path: _sha((Path(__file__).parent / path).read_bytes())
                    for path in source_paths
                }
                allocations, allocation_sources = {}, []
                for rank in ranks:
                    peer, peer_sha = read(directory / f"rank-{rank}.communication.json")
                    if (
                        peer.get("schema") != snapshot["schema"]
                        or peer.get("run_id") != capture["run_id"]
                        or peer.get("rank") != rank
                        or peer.get("config_sha256") != snapshot["config_sha256"]
                        or peer.get("rank_evidence") != snapshot["rank_evidence"]
                        or any(
                            peer.get("route_sources", {}).get(path) != sha
                            for path, sha in trusted_sources.items()
                        )
                    ):
                        raise ValueError("communication peer/source contract differs")
                    matches = []
                    for path in directory.glob(f"rank-{rank}.mnnvl-allocation-*.json"):
                        if (
                            sealed_artifacts is not None
                            and str(path.resolve()) not in sealed_artifacts
                        ):
                            continue
                        item, item_sha = read(path)
                        allocation = item.get("allocation", {})
                        if (
                            item.get("run_id") != capture["run_id"]
                            or item.get("rank") != rank
                            or allocation.get("group_name") != group
                            or tuple(allocation.get("ranks", ())) != ranks
                        ):
                            continue
                        if (
                            path.stem.rsplit("-", 1)[-1] != item_sha
                            or item.get("schema")
                            != "tokenspeed.mnnvl_allocation_evidence/v1"
                            or item.get("config_sha256") != snapshot["config_sha256"]
                            or item.get("communication_snapshot_sha256") != peer_sha
                            or allocation.get("mapping_verified") is not True
                            or allocation.get("allocator_backend") != "CUDA"
                            or allocation.get("world_size") != len(ranks)
                            or allocation.get("peer_count") != len(ranks)
                            or allocation.get("rank") != ranks.index(rank)
                            or allocation.get("uuid", "").lower()
                            != peers[rank].get("uuid", "").lower()
                            or any(
                                type(allocation.get(field)) is not int
                                or allocation[field] <= 0
                                for field in (
                                    "local_ptr",
                                    "multicast_ptr",
                                    "buffer_size",
                                )
                            )
                        ):
                            raise ValueError(
                                "CUDA multicast allocation identity/mapping unavailable"
                            )
                        locator = (
                            f"{Path(original).parent / path.name}#sha256={item_sha}"
                        )
                        matches.append(
                            dict(
                                allocation,
                                source=locator,
                                communication_snapshot_sha256=peer_sha,
                            )
                        )
                    if len(matches) != 1:
                        raise ValueError(
                            "CUDA multicast allocation missing or ambiguous for peer group"
                        )
                    allocations[rank] = matches[0]
                    allocation_sources.append(matches[0]["source"])
                binding = dict(snapshot, path=original, sha256=digest)
                base = _group_observation_from_snapshot(binding, ranks, group)
                cache[key] = (
                    binding,
                    base,
                    allocations[capture["rank"]],
                    tuple(allocation_sources),
                )
            binding, base, allocation, sources = cache[key]
            if any(
                _json(observation[field]) != _json(value)
                for field, value in base.items()
                if field in observation
            ):
                raise ValueError(
                    "recorded communication placement differs from receipt"
                )
            observation.update(base)
            twoshot_geometry = None
            if observation.get("comm_algorithm") == "twoshot":
                if sealed_artifacts is None or native_op_activities is None:
                    raise ValueError(
                        "MNNVL two-shot requires sealed routes and native Op activities"
                    )
                twoshot_geometry = _mnnvl_twoshot_geometry(
                    op, native_op_activities.get(op["trace_label"], ()), binding, ranks
                )
            route = _mnnvl_route_observation(
                observation,
                binding,
                allocation,
                ranks,
                group,
                twoshot_geometry=twoshot_geometry,
            )
            if not route:
                raise ValueError(
                    "selected branch/topology does not prove the multicast route"
                )
            result[op["trace_label"]] = {
                **base,
                **route,
                "comm_all_rank_allocation_sources": sources,
                "comm_all_rank_mapping_verified": True,
                "comm_offline_binding_verified": True,
            }
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            result[op["trace_label"]] = {
                "comm_offline_binding_missing_reason": f"{type(error).__name__}: {error}"
            }
    return result
