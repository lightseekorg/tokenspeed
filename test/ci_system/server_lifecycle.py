"""Strict server lifecycle management for release CI tasks.

The managed path deliberately keeps the launched command as the process-group
root.  A successful shutdown sends SIGTERM to that root only; process-group
signals are reserved for best-effort cleanup after validation has already
failed.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

GpuProbe = Callable[[Sequence[int]], dict[str, Any]]

MANAGED_SERVER_REGISTRY_ROOT = Path("/tmp/tokenspeed-ci-managed-servers")
_MANAGED_SERVER_SIGNALS = (signal.SIGTERM, signal.SIGINT)
_ACTIVE_SERVERS: dict[int, "ManagedServer"] = {}
_ACTIVE_SERVERS_LOCK = threading.RLock()
_PREVIOUS_SIGNAL_HANDLERS: dict[int, Any] = {}
_STARTING_MANAGED_SERVERS = 0
_DEFERRED_MANAGED_SIGNALS: list[tuple[int, Any]] = []

# The main thread blocks SIGTERM/SIGINT across Popen and durable registration.
# This exec trampoline restores the caller's original mask in the child before
# replacing itself with bash and then the real command, so the server keeps the
# Popen PID without inheriting the parent's temporary protection.
_SIGNAL_MASK_EXEC_TRAMPOLINE = (
    "import os,signal,sys\n"
    "mask={int(item) for item in sys.argv[2].split(',') if item}\n"
    "signal.pthread_sigmask(signal.SIG_SETMASK, mask)\n"
    'os.execvpe("bash", ["bash", "-c", "exec " + sys.argv[1]], os.environ)\n'
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _format_exception(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _tee_output(stream: TextIO, log_handle: TextIO, output_errors: list[str]) -> None:
    try:
        for line in stream:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
    except BaseException as error:
        output_errors.append(_format_exception(error))
    finally:
        handles = (("server output stream", stream), ("server log", log_handle))
        for name, handle in handles:
            try:
                handle.close()
            except BaseException as error:
                output_errors.append(f"cannot close {name}: {_format_exception(error)}")


def _read_process(pid: int) -> dict[str, Any] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, ValueError):
        return None
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = stat[closing_parenthesis + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return {
            "pid": pid,
            "name": stat[stat.find("(") + 1 : closing_parenthesis],
            "state": fields[0],
            "parent_pid": int(fields[1]),
            "process_group": int(fields[2]),
            "start_time_ticks": int(fields[19]),
        }
    except ValueError:
        return None


def _process_table() -> dict[int, dict[str, Any]]:
    table: dict[int, dict[str, Any]] = {}
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return table
    for entry in entries:
        if not entry.name.isdigit():
            continue
        identity = _read_process(int(entry.name))
        if identity is not None:
            table[identity["pid"]] = identity
    return table


def _descendants(
    root_pid: int, table: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_pids: set[int] = set()
    parents = {root_pid}
    while parents:
        children = [
            dict(identity)
            for identity in table.values()
            if identity["parent_pid"] in parents
            and identity["pid"] not in selected_pids
        ]
        if not children:
            break
        selected.extend(children)
        selected_pids.update(child["pid"] for child in children)
        parents = {child["pid"] for child in children}
    return sorted(selected, key=lambda item: item["pid"])


def _identity_is_alive(identity: Mapping[str, Any]) -> bool:
    current = _read_process(int(identity["pid"]))
    return (
        current is not None
        and current["state"] != "Z"
        and current["start_time_ticks"] == identity.get("start_time_ticks")
    )


def _process_group_members(process_group: int) -> list[dict[str, Any]]:
    members = [
        identity
        for identity in _process_table().values()
        if identity["process_group"] == process_group
    ]
    return sorted(members, key=lambda item: item["pid"])


def _deduplicate_identities(
    identities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for identity in identities:
        key = (int(identity["pid"]), int(identity["start_time_ticks"]))
        selected[key] = dict(identity)
    return sorted(selected.values(), key=lambda item: item["pid"])


def _runner_identity(env: Mapping[str, str]) -> str:
    for key in ("RUNNER_NAME", "CI_RUNNER_NAME", "HOSTNAME", "CI_RUNNER_LABEL"):
        value = env.get(key)
        if value:
            return value
    return socket.gethostname()


def _registry_directory(
    env: Mapping[str, str], registry_root: Path = MANAGED_SERVER_REGISTRY_ROOT
) -> Path:
    safe_runner = re.sub(r"[^A-Za-z0-9_.-]", "_", _runner_identity(env))
    return registry_root / f"uid-{os.getuid()}" / safe_runner


def _registry_record(
    *,
    command: str,
    cwd: Path,
    env: Mapping[str, str],
    root_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "owner_uid": os.getuid(),
        "runner_identity": _runner_identity(env),
        "task_name": env.get("CI_TASK_NAME"),
        "command": command,
        "cwd": str(cwd),
        "created_at_unix": time.time(),
        "process_group": int(root_identity["process_group"]),
        "root": {
            "pid": int(root_identity["pid"]),
            "start_time_ticks": int(root_identity["start_time_ticks"]),
            "process_group": int(root_identity["process_group"]),
        },
    }


def _parse_registry_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError("unsupported or missing schema_version")
        if record.get("owner_uid") != os.getuid():
            raise ValueError("registry owner UID does not match the current user")
        root = record.get("root")
        if not isinstance(root, dict):
            raise ValueError("root identity is missing")
        pid = root.get("pid")
        started = root.get("start_time_ticks")
        process_group = record.get("process_group")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 1
            or isinstance(started, bool)
            or not isinstance(started, int)
            or started <= 0
            or isinstance(process_group, bool)
            or not isinstance(process_group, int)
            or process_group <= 1
        ):
            raise ValueError("root identity fields are invalid")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"invalid managed-server registry {path}: {error}"
        ) from error
    return record


def _remove_registry_path(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _kill_identity(identity: Mapping[str, Any], signal_number: int) -> bool:
    if _identity_is_alive(identity):
        try:
            os.kill(int(identity["pid"]), signal_number)
        except (ProcessLookupError, PermissionError):
            return False
        return True
    return False


def _refresh_known_group_members(
    process_group: int,
    known: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current = _process_group_members(process_group)
    known_keys = {
        (int(identity["pid"]), int(identity["start_time_ticks"])) for identity in known
    }
    if not any(
        (member["pid"], member["start_time_ticks"]) in known_keys for member in current
    ):
        return known
    return _deduplicate_identities([*known, *current])


def _signal_known_group(
    process_group: int,
    identities: Sequence[Mapping[str, Any]],
    signal_number: int,
) -> bool:
    current = _process_group_members(process_group)
    known = {
        (int(identity["pid"]), int(identity["start_time_ticks"]))
        for identity in identities
    }
    if not any(
        (member["pid"], member["start_time_ticks"]) in known for member in current
    ):
        return False
    try:
        os.killpg(process_group, signal_number)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _wait_for_identities(
    identities: Sequence[Mapping[str, Any]],
    *,
    process_group: int,
    timeout: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    known = _deduplicate_identities(identities)
    while time.monotonic() < deadline:
        known = _refresh_known_group_members(process_group, known)
        survivors = [identity for identity in known if _identity_is_alive(identity)]
        if not survivors:
            return []
        time.sleep(0.05)
    return [identity for identity in known if _identity_is_alive(identity)]


def _terminate_identities(
    process_group: int,
    identities: Sequence[Mapping[str, Any]],
    *,
    term_timeout: float = 3,
    kill_timeout: float = 3,
) -> list[dict[str, Any]]:
    """Terminate captured process identities without trusting a bare PID/PGID."""

    known = _refresh_known_group_members(
        process_group, _deduplicate_identities(identities)
    )
    _signal_known_group(process_group, known, signal.SIGTERM)
    for identity in known:
        _kill_identity(identity, signal.SIGTERM)
    survivors = _wait_for_identities(
        known, process_group=process_group, timeout=term_timeout
    )
    if not survivors:
        return []

    known = _refresh_known_group_members(
        process_group, _deduplicate_identities([*known, *survivors])
    )
    _signal_known_group(process_group, known, signal.SIGKILL)
    for identity in known:
        _kill_identity(identity, signal.SIGKILL)
    return _wait_for_identities(
        known, process_group=process_group, timeout=kill_timeout
    )


def cleanup_stale_managed_servers(
    env: Mapping[str, str],
    *,
    registry_root: Path = MANAGED_SERVER_REGISTRY_ROOT,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Recover identity-matched managed servers recorded by an earlier run."""

    registry_directory = _registry_directory(env, registry_root)
    if not registry_directory.exists():
        return []

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in sorted(registry_directory.glob("*.json")):
        record = _parse_registry_record(path)
        root = dict(record["root"])
        process_group = int(record["process_group"])
        if not _identity_is_alive(root):
            if not dry_run:
                path.unlink(missing_ok=True)
            results.append(
                {"registry": str(path), "status": "stale_identity", "pid": root["pid"]}
            )
            continue
        if dry_run:
            print(
                f"[dry-run] recover managed server PID {root['pid']} from {path}",
                flush=True,
            )
            results.append(
                {"registry": str(path), "status": "would_recover", "pid": root["pid"]}
            )
            continue

        table = _process_table()
        current_root = table.get(int(root["pid"]))
        if (
            current_root is None
            or current_root["start_time_ticks"] != root["start_time_ticks"]
        ):
            path.unlink(missing_ok=True)
            results.append(
                {"registry": str(path), "status": "stale_identity", "pid": root["pid"]}
            )
            continue
        identities = _deduplicate_identities(
            [
                root,
                *_descendants(int(root["pid"]), table),
                *(
                    identity
                    for identity in table.values()
                    if identity["process_group"] == process_group
                ),
            ]
        )
        survivors = _terminate_identities(process_group, identities)
        if survivors:
            failures.append(
                f"{path}: managed server identities survived: "
                + ", ".join(str(item["pid"]) for item in survivors)
            )
            results.append(
                {
                    "registry": str(path),
                    "status": "failed",
                    "pid": root["pid"],
                    "survivors": survivors,
                }
            )
            continue
        path.unlink(missing_ok=True)
        results.append(
            {"registry": str(path), "status": "recovered", "pid": root["pid"]}
        )

    if failures:
        raise RuntimeError("managed-server recovery failed: " + "; ".join(failures))
    return results


def _zombie_identities() -> set[tuple[int, int]]:
    return {
        (identity["pid"], identity["start_time_ticks"])
        for identity in _process_table().values()
        if identity["state"] == "Z"
    }


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.25)
        return client.connect_ex(("127.0.0.1", port)) == 0


def probe_nvidia_gpus(gpu_indices: Sequence[int]) -> dict[str, Any]:
    """Return selected GPU memory and compute-process state from nvidia-smi."""

    wanted = set(gpu_indices)
    try:
        gpu_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        process_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"ok": False, "error": str(error), "gpus": {}}

    if gpu_query.returncode != 0 or process_query.returncode != 0:
        return {
            "ok": False,
            "error": (
                f"nvidia-smi query failed: gpu={gpu_query.returncode}, "
                f"compute={process_query.returncode}"
            ),
            "gpus": {},
        }

    gpus: dict[int, dict[str, Any]] = {}
    uuid_to_index: dict[str, int] = {}
    try:
        for raw_line in gpu_query.stdout.splitlines():
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",")]
            if len(fields) != 4:
                raise ValueError(f"unexpected GPU query row: {raw_line!r}")
            index = int(fields[0])
            if index not in wanted:
                continue
            uuid = fields[1]
            gpus[index] = {
                "uuid": uuid,
                "memory_used_mib": int(fields[2]),
                "utilization_percent": int(fields[3]),
                "compute_pids": [],
            }
            uuid_to_index[uuid] = index

        missing = sorted(wanted - set(gpus))
        if missing:
            raise ValueError(f"selected GPU indices are missing: {missing}")

        for raw_line in process_query.stdout.splitlines():
            if not raw_line.strip():
                continue
            fields = [field.strip() for field in raw_line.split(",")]
            if len(fields) != 2:
                raise ValueError(f"unexpected compute-process row: {raw_line!r}")
            pid, uuid = int(fields[0]), fields[1]
            index = uuid_to_index.get(uuid)
            if index is not None:
                gpus[index]["compute_pids"].append(pid)
    except ValueError as error:
        return {"ok": False, "error": str(error), "gpus": {}}

    return {
        "ok": True,
        "gpus": {str(index): gpus[index] for index in sorted(gpus)},
    }


@dataclass
class ManagedServer:
    """A server whose Popen PID is the actual command root."""

    process: subprocess.Popen[str]
    process_group: int
    root_identity: dict[str, Any]
    output_thread: threading.Thread
    output_stream: TextIO = field(repr=False)
    log_handle: TextIO = field(repr=False)
    baseline_zombies: set[tuple[int, int]] = field(repr=False)
    baseline_gpu: dict[str, Any]
    gpu_probe: GpuProbe = field(repr=False)
    registry_path: Path | None = field(default=None, repr=False)
    output_errors: list[str] = field(default_factory=list, repr=False)
    output_thread_started: bool = field(default=False, repr=False)


def _invoke_previous_signal_handler(signum: int, frame: Any, previous: Any) -> None:
    if previous == signal.SIG_IGN:
        return
    if callable(previous):
        previous(signum, frame)
        return
    raise SystemExit(128 + signum)


def _managed_server_signal_handler(signum: int, frame: Any) -> None:
    with _ACTIVE_SERVERS_LOCK:
        if _STARTING_MANAGED_SERVERS:
            _DEFERRED_MANAGED_SIGNALS.append((signum, frame))
            return
        servers = list(_ACTIVE_SERVERS.values())
        previous = _PREVIOUS_SIGNAL_HANDLERS.get(signum, signal.SIG_DFL)
    for server in servers:
        abort_managed_server(server)

    _invoke_previous_signal_handler(signum, frame, previous)


def _install_managed_server_signal_handlers_locked() -> None:
    if _PREVIOUS_SIGNAL_HANDLERS:
        return

    installed: list[int] = []
    try:
        for signum in _MANAGED_SERVER_SIGNALS:
            _PREVIOUS_SIGNAL_HANDLERS[signum] = signal.getsignal(signum)
            signal.signal(signum, _managed_server_signal_handler)
            installed.append(signum)
    except BaseException:
        for signum in installed:
            signal.signal(signum, _PREVIOUS_SIGNAL_HANDLERS.pop(signum))
        raise


def _restore_managed_server_signal_handlers_locked() -> None:
    if _ACTIVE_SERVERS or _STARTING_MANAGED_SERVERS or not _PREVIOUS_SIGNAL_HANDLERS:
        return
    for signum, previous in list(_PREVIOUS_SIGNAL_HANDLERS.items()):
        if signal.getsignal(signum) is _managed_server_signal_handler:
            signal.signal(signum, previous)
    _PREVIOUS_SIGNAL_HANDLERS.clear()


def _begin_managed_server_start() -> set[signal.Signals]:
    """Defer termination until a new root is registered or aborted."""

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "managed servers must start on the main thread so SIGTERM/SIGINT "
            "cleanup can be installed"
        )

    global _STARTING_MANAGED_SERVERS
    with _ACTIVE_SERVERS_LOCK:
        _install_managed_server_signal_handlers_locked()
        _STARTING_MANAGED_SERVERS += 1
    try:
        return signal.pthread_sigmask(signal.SIG_BLOCK, _MANAGED_SERVER_SIGNALS)
    except BaseException:
        with _ACTIVE_SERVERS_LOCK:
            _STARTING_MANAGED_SERVERS -= 1
            _restore_managed_server_signal_handlers_locked()
        raise


def _finish_managed_server_start(previous_mask: set[signal.Signals]) -> None:
    """Restore the caller mask and replay signals deferred by another thread."""

    global _STARTING_MANAGED_SERVERS
    with _ACTIVE_SERVERS_LOCK:
        if _STARTING_MANAGED_SERVERS <= 0:
            raise RuntimeError("managed-server start guard is not active")
        _STARTING_MANAGED_SERVERS -= 1
        deferred = (
            list(_DEFERRED_MANAGED_SIGNALS) if _STARTING_MANAGED_SERVERS == 0 else []
        )
        if deferred:
            _DEFERRED_MANAGED_SIGNALS.clear()
        previous_handlers = {
            signum: _PREVIOUS_SIGNAL_HANDLERS.get(signum, signal.SIG_DFL)
            for signum, _ in deferred
        }
        _restore_managed_server_signal_handlers_locked()

    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    for signum, frame in deferred:
        with _ACTIVE_SERVERS_LOCK:
            has_active_server = bool(_ACTIVE_SERVERS)
        if has_active_server:
            _managed_server_signal_handler(signum, frame)
        else:
            _invoke_previous_signal_handler(signum, frame, previous_handlers[signum])


def _register_active_server(server: ManagedServer) -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "managed servers must start on the main thread so SIGTERM/SIGINT "
            "cleanup can be installed"
        )
    with _ACTIVE_SERVERS_LOCK:
        _install_managed_server_signal_handlers_locked()
        _ACTIVE_SERVERS[server.process.pid] = server


def _unregister_active_server(server: ManagedServer) -> None:
    with _ACTIVE_SERVERS_LOCK:
        _ACTIVE_SERVERS.pop(server.process.pid, None)
        if threading.current_thread() is not threading.main_thread():
            return
        _restore_managed_server_signal_handlers_locked()


def _join_output_thread(server: ManagedServer, timeout: float) -> bool:
    if not server.output_thread_started:
        return False
    try:
        server.output_thread.join(timeout=timeout)
    except RuntimeError:
        return False
    return server.output_thread.is_alive()


def _close_server_streams(server: ManagedServer) -> None:
    for handle in (server.output_stream, server.log_handle):
        try:
            handle.close()
        except BaseException:
            pass


def _capture_server_identities(server: ManagedServer) -> list[dict[str, Any]]:
    identities = [server.root_identity]
    table = _process_table()
    root = table.get(server.process.pid)
    if (
        root is not None
        and root["start_time_ticks"] == server.root_identity["start_time_ticks"]
    ):
        identities.extend(_descendants(server.process.pid, table))
        identities.extend(
            identity
            for identity in table.values()
            if identity["process_group"] == server.process_group
        )
    return _deduplicate_identities(identities)


def abort_managed_server(server: ManagedServer | None) -> dict[str, Any]:
    """Best-effort, identity-safe emergency cleanup that never raises."""

    if server is None:
        return {"passed": True, "failures": [], "survivors": []}

    failures: list[str] = []
    survivors: list[dict[str, Any]] = []
    try:
        identities = _capture_server_identities(server)
        survivors = _terminate_identities(server.process_group, identities)
    except BaseException as error:
        failures.append(
            f"cannot terminate server identities: {_format_exception(error)}"
        )

    try:
        if server.process.poll() is None:
            server.process.wait(timeout=1)
    except BaseException as error:
        failures.append(f"cannot reap server root: {_format_exception(error)}")

    _close_server_streams(server)
    try:
        if _join_output_thread(server, 2):
            failures.append("server output thread did not exit during abort")
    except BaseException as error:
        failures.append(f"cannot join server output thread: {_format_exception(error)}")

    try:
        survivors = [
            identity
            for identity in _deduplicate_identities(survivors)
            if _identity_is_alive(identity)
        ]
        if _identity_is_alive(server.root_identity):
            survivors = _deduplicate_identities([*survivors, server.root_identity])
    except BaseException as error:
        failures.append(f"cannot verify server cleanup: {_format_exception(error)}")

    try:
        cleanup_complete = not survivors and server.process.poll() is not None
    except BaseException as error:
        cleanup_complete = False
        failures.append(f"cannot poll server root: {_format_exception(error)}")
    if cleanup_complete:
        _remove_registry_path(server.registry_path)
    else:
        failures.append(
            "managed server cleanup is incomplete"
            + (
                ": " + ", ".join(str(item["pid"]) for item in survivors)
                if survivors
                else ""
            )
        )
    try:
        _unregister_active_server(server)
    except BaseException as error:
        failures.append(
            f"cannot restore managed-server signal handlers: {_format_exception(error)}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "survivors": survivors,
        "output_errors": list(server.output_errors),
    }


def _abort_unregistered_process(
    process: subprocess.Popen[str], log_handle: TextIO
) -> None:
    """Reap a just-created session before a ManagedServer can be registered."""

    try:
        root = _read_process(process.pid)
        if root is not None and root["process_group"] == process.pid:
            identities = _deduplicate_identities(
                [root, *_process_group_members(process.pid)]
            )
            _terminate_identities(
                process.pid, identities, term_timeout=1, kill_timeout=1
            )
        elif process.poll() is None:
            process.kill()
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    except BaseException:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
        except BaseException:
            pass
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except BaseException:
                pass
        try:
            log_handle.close()
        except BaseException:
            pass


def start_managed_server(
    command: str,
    *,
    env: Mapping[str, str],
    cwd: Path,
    log_path: Path,
    gpu_indices: Sequence[int],
    max_baseline_memory_mib: int = 16,
    dry_run: bool = False,
    gpu_probe: GpuProbe = probe_nvidia_gpus,
    registry_root: Path = MANAGED_SERVER_REGISTRY_ROOT,
) -> ManagedServer | None:
    """Launch ``command`` with ``exec`` so Popen owns the real root PID."""

    print(f"$ {command}", flush=True)
    if dry_run:
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    try:
        baseline_gpu = gpu_probe(gpu_indices)
    except BaseException:
        log_handle.close()
        raise
    if not baseline_gpu.get("ok"):
        log_handle.close()
        raise RuntimeError(
            f"cannot capture pre-start GPU baseline: {baseline_gpu.get('error')}"
        )
    dirty_gpus: list[str] = []
    for index in gpu_indices:
        details = baseline_gpu.get("gpus", {}).get(str(index))
        if details is None:
            dirty_gpus.append(f"GPU {index} is missing from the baseline")
            continue
        compute_pids = details.get("compute_pids", [])
        if compute_pids:
            dirty_gpus.append(f"GPU {index} has compute PIDs {compute_pids}")
        memory_used = int(details.get("memory_used_mib", 0))
        if memory_used > max_baseline_memory_mib:
            dirty_gpus.append(
                f"GPU {index} uses {memory_used} MiB; maximum clean baseline is "
                f"{max_baseline_memory_mib} MiB"
            )
    if dirty_gpus:
        log_handle.close()
        raise RuntimeError(
            "refusing to start server on a dirty GPU baseline: " + "; ".join(dirty_gpus)
        )
    baseline_zombies = _zombie_identities()

    try:
        previous_signal_mask = _begin_managed_server_start()
    except BaseException:
        log_handle.close()
        raise
    process: subprocess.Popen[str] | None = None
    server: ManagedServer | None = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _SIGNAL_MASK_EXEC_TRAMPOLINE,
                command,
                ",".join(str(int(signum)) for signum in sorted(previous_signal_mask)),
            ],
            cwd=cwd,
            env=dict(env),
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        root_identity: dict[str, Any] | None = None
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            root_identity = _read_process(process.pid)
            if root_identity is not None:
                break
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if root_identity is None:
            raise RuntimeError("cannot capture managed server root identity")
        if root_identity["process_group"] != process.pid:
            raise RuntimeError(
                "managed server did not become its own process-group root: "
                f"pid={process.pid}, pgid={root_identity['process_group']}"
            )

        output_errors: list[str] = []
        output_thread = threading.Thread(
            target=_tee_output,
            args=(process.stdout, log_handle, output_errors),
            name=f"server-log-{process.pid}",
            daemon=True,
        )
        registry_path = _registry_directory(env, registry_root) / (
            f"{process.pid}-{root_identity['start_time_ticks']}.json"
        )
        server = ManagedServer(
            process=process,
            process_group=process.pid,
            root_identity=root_identity,
            output_thread=output_thread,
            output_stream=process.stdout,
            log_handle=log_handle,
            baseline_zombies=baseline_zombies,
            baseline_gpu=baseline_gpu,
            gpu_probe=gpu_probe,
            registry_path=registry_path,
            output_errors=output_errors,
        )
        _write_json_atomic(
            registry_path,
            _registry_record(
                command=command,
                cwd=cwd,
                env=env,
                root_identity=root_identity,
            ),
        )
        server.output_thread_started = True
        output_thread.start()
        _register_active_server(server)
    except BaseException:
        if server is not None:
            abort_managed_server(server)
        elif process is not None:
            _abort_unregistered_process(process, log_handle)
        else:
            log_handle.close()
        raise
    finally:
        _finish_managed_server_start(previous_signal_mask)
    return server


def _fallback_cleanup(
    server: ManagedServer, identities: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    # A failed validation is already irrevocably failed.  Group and per-PID
    # signals below are hygiene only, including escaped descendants.
    survivors = _terminate_identities(server.process_group, identities)
    try:
        if server.process.poll() is None:
            server.process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    return [identity for identity in survivors if _identity_is_alive(identity)]


def _stop_managed_server(
    server: ManagedServer | None,
    shutdown: Mapping[str, Any],
    *,
    cwd: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate a root-only graceful shutdown and emit its durable artifact."""

    output_path = cwd / str(shutdown["output"])
    if dry_run:
        result = {
            "schema_version": 1,
            "check": "server_shutdown",
            "passed": True,
            "skipped": True,
            "failures": [],
            "fallback_used": False,
        }
        return result
    if server is None:
        result = {
            "schema_version": 1,
            "check": "server_shutdown",
            "passed": False,
            "failures": ["managed server was not started"],
            "fallback_used": False,
        }
        _write_json_atomic(output_path, result)
        return result

    failures: list[str] = []
    started = time.monotonic()
    root: dict[str, Any] | None = None
    descendants: list[dict[str, Any]] = []
    group_before: list[dict[str, Any]] = []
    returncode: int | None = server.process.poll()

    process_table = _process_table()
    current_root = process_table.get(server.process.pid)
    if current_root is None:
        failures.append("cannot identify live server root before shutdown")
    elif current_root["start_time_ticks"] != server.root_identity["start_time_ticks"]:
        failures.append(
            "server root PID identity changed before shutdown; refusing to signal it"
        )
    else:
        root = current_root
        descendants = _descendants(server.process.pid, process_table)
        group_before = sorted(
            (
                identity
                for identity in process_table.values()
                if identity["process_group"] == server.process_group
            ),
            key=lambda item: item["pid"],
        )

    if returncode is not None:
        failures.append(f"server root exited before shutdown with code {returncode}")
    elif not _identity_is_alive(server.root_identity):
        failures.append("registered server root identity is no longer alive")
        returncode = server.process.poll()
    else:
        try:
            # Normal path: signal only the true command root.
            if not _kill_identity(server.root_identity, signal.SIGTERM):
                raise ProcessLookupError
            returncode = server.process.wait(timeout=float(shutdown["timeout_seconds"]))
        except ProcessLookupError:
            failures.append("server root disappeared before SIGTERM was delivered")
            returncode = server.process.poll()
        except subprocess.TimeoutExpired:
            failures.append(
                "server root did not exit within "
                f"{shutdown['timeout_seconds']} seconds after SIGTERM"
            )
            returncode = server.process.poll()

    if returncode != shutdown["expected_exit_code"]:
        failures.append(
            f"server root exit code {returncode!r} does not match expected "
            f"{shutdown['expected_exit_code']}"
        )

    output_stream_open = _join_output_thread(server, 2)
    if output_stream_open:
        failures.append("server descendants kept the output stream open")
    identities = _deduplicate_identities(
        [server.root_identity, *descendants, *group_before]
    )
    survivors = [identity for identity in identities if _identity_is_alive(identity)]
    if survivors:
        failures.append(
            "server process tree survived graceful shutdown: "
            + ", ".join(str(identity["pid"]) for identity in survivors)
        )

    group_after = _process_group_members(server.process_group)
    if group_after:
        failures.append(
            "server process group is not empty: "
            + ", ".join(str(identity["pid"]) for identity in group_after)
        )

    open_ports = [port for port in shutdown["ports"] if _port_is_open(port)]
    if open_ports:
        failures.append(f"server ports remain open: {open_ports}")

    final_gpu = server.gpu_probe(shutdown["gpu_indices"])
    if not final_gpu.get("ok"):
        failures.append(f"cannot validate final GPU state: {final_gpu.get('error')}")
    else:
        for index in shutdown["gpu_indices"]:
            details = final_gpu["gpus"].get(str(index))
            if details is None:
                failures.append(f"final GPU state is missing index {index}")
                continue
            if details["compute_pids"]:
                failures.append(
                    f"GPU {index} retains compute PIDs: {details['compute_pids']}"
                )
            if details["memory_used_mib"] > shutdown["max_memory_mib"]:
                failures.append(
                    f"GPU {index} retains {details['memory_used_mib']} MiB; "
                    f"maximum is {shutdown['max_memory_mib']} MiB"
                )

    new_zombies = sorted(_zombie_identities() - server.baseline_zombies)
    relevant_identities = {
        (int(identity["pid"]), int(identity["start_time_ticks"]))
        for identity in [*identities, *group_before, *group_after]
    }
    relevant_zombies = [
        {"pid": pid, "start_time_ticks": created}
        for pid, created in new_zombies
        if (pid, created) in relevant_identities
    ]
    if relevant_zombies:
        failures.append(
            "server shutdown left new zombie processes: "
            + ", ".join(str(item["pid"]) for item in relevant_zombies)
        )

    output_capture_errors = list(server.output_errors)
    for error in output_capture_errors:
        failures.append(f"server log capture failed: {error}")

    fallback_used = bool(failures)
    fallback_survivors: list[dict[str, Any]] = []
    if fallback_used:
        # Only identities captured while the registered root identity was
        # valid may authorize cleanup.  Never promote an arbitrary later PGID
        # member into the trusted set: the numeric PGID may have been reused.
        fallback_survivors = _fallback_cleanup(server, identities)
        if fallback_survivors:
            failures.append(
                "server identities survived fallback cleanup: "
                + ", ".join(str(item["pid"]) for item in fallback_survivors)
            )
        if _join_output_thread(server, 2):
            _close_server_streams(server)
            _join_output_thread(server, 1)
        for error in server.output_errors[len(output_capture_errors) :]:
            output_capture_errors.append(error)
            failures.append(f"server log capture failed: {error}")

    output_thread_alive_after_cleanup = _join_output_thread(server, 0)
    cleanup_complete = (
        not _identity_is_alive(server.root_identity)
        and not fallback_survivors
        and server.process.poll() is not None
    )
    if cleanup_complete:
        _remove_registry_path(server.registry_path)
        _unregister_active_server(server)

    result = {
        "schema_version": 1,
        "check": "server_shutdown",
        "passed": not failures,
        "target": shutdown["target"],
        "signal": shutdown["signal"],
        "timeout_seconds": shutdown["timeout_seconds"],
        "expected_exit_code": shutdown["expected_exit_code"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "root": root,
        "registered_root": server.root_identity,
        "descendants_before_signal": descendants,
        "process_group": server.process_group,
        "process_group_before_signal": group_before,
        "process_group_after_shutdown": group_after,
        "returncode": returncode,
        "output_stream_open": output_stream_open,
        "output_thread_alive_after_cleanup": output_thread_alive_after_cleanup,
        "output_capture_errors": output_capture_errors,
        "survivors": survivors,
        "fallback_survivors": fallback_survivors,
        "open_ports": open_ports,
        "gpu_baseline": server.baseline_gpu,
        "gpu_final": final_gpu,
        "new_relevant_zombies": relevant_zombies,
        "fallback_used": fallback_used,
        "failures": failures,
    }
    _write_json_atomic(output_path, result)
    return result


def stop_managed_server(
    server: ManagedServer | None,
    shutdown: Mapping[str, Any],
    *,
    cwd: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate shutdown, aborting the registered process tree on exceptions."""

    try:
        return _stop_managed_server(server, shutdown, cwd=cwd, dry_run=dry_run)
    except BaseException:
        abort_managed_server(server)
        raise
