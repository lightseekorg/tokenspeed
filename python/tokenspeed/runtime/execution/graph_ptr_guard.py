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

"""Capture-time tensor-identity snapshots for CUDA-graph decode metadata.

A captured graph holds the addresses of every tensor its kernels read
forever; the unified refresh path must therefore write the exact tensors it
bound at capture (docs/design/unified_path.md, "Pointer-stable per-bs
views"). A refresh that reallocates a buffer — or binds a metadata view over
fresh storage — silently feeds the replayed kernels stale data. Unit parity
tests cannot see this class of bug, so this module makes it assertable:
snapshot the tensor identities reachable from the graph-visible metadata
slots right after capture, and re-verify them on every replay when
``TOKENSPEED_GRAPH_DEBUG=1`` (production replays pay only a bool check).
"""

from __future__ import annotations

import os
import types

import torch

GRAPH_DEBUG_ENV = "TOKENSPEED_GRAPH_DEBUG"

# Decode-graph-visible metadata slots. Prefill slots stay out: extend
# metadata is rebuilt fresh each round by design (init_forward_metadata) and
# the decode graph never reads it. Slots are looked up through ``vars()``,
# not ``getattr``, so attribute-mirroring wrappers (Inkling's ``__getattr__``)
# and delegating properties (DSA) do not double-report their child's slots —
# children are reached through ``child_backends()`` instead.
_GRAPH_METADATA_SLOTS = (
    "forward_decode_metadata",
    "forward_metadata",
    "conv_decode_metadata",
    # The router's per-group decode write-slot views (recorded through the
    # leaves' KV writes).
    "decode_write_locations",
)

_MAX_WALK_DEPTH = 8

_SKIPPED_CALLABLE_TYPES = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    type,
)


def graph_debug_enabled() -> bool:
    """Whether replay-time pointer-stability verification is enabled."""
    return os.environ.get(GRAPH_DEBUG_ENV) == "1"


def _tensor_identity(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _is_backend_like(obj) -> bool:
    # Duck-typed so this module needs no backend import: anything carrying
    # the unified decode contract is a backend and is walked only through
    # the explicit child_backends() recursion, never from inside metadata.
    return callable(getattr(obj, "refresh_decode_metadata", None))


def _walk(obj, path: str, out: dict, seen: set, depth: int):
    if obj is None or depth <= 0:
        return
    if isinstance(obj, torch.Tensor):
        # Aliased tensors are recorded once per path on purpose: each path is
        # an address the captured graph may have recorded.
        out[path] = _tensor_identity(obj)
        return
    if isinstance(obj, (str, bytes, int, float, bool, complex)):
        return
    if isinstance(obj, _SKIPPED_CALLABLE_TYPES) or isinstance(obj, torch.nn.Module):
        return
    if _is_backend_like(obj):
        return
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        for key, value in obj.items():
            _walk(value, f"{path}[{key!r}]", out, seen, depth - 1)
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _walk(value, f"{path}[{index}]", out, seen, depth - 1)
        return
    attrs = getattr(obj, "__dict__", None)
    if attrs is None:
        return
    for name, value in attrs.items():
        _walk(value, f"{path}.{name}", out, seen, depth - 1)


def _walk_backend(backend, prefix: str, out: dict, seen: set) -> None:
    if id(backend) in seen:
        return
    seen.add(id(backend))
    attrs = getattr(backend, "__dict__", {})
    for slot in _GRAPH_METADATA_SLOTS:
        metadata = attrs.get(slot)
        if metadata is not None:
            _walk(metadata, f"{prefix}.{slot}", out, seen, _MAX_WALK_DEPTH)
    for child in backend.child_backends():
        _walk_backend(child, f"{prefix}.{type(child).__name__}", out, seen)


def snapshot_graph_metadata(backend) -> dict[str, tuple]:
    """Record the identity of every tensor reachable from ``backend``'s
    decode-graph-visible metadata slots (recursing into ``child_backends()``).

    Every tensor a slot reaches is pinned: per-step-mutable objects the
    kernels own (FlashMLA's tile schedule) live on the backend, outside the
    slots, rather than being exempted here.

    Args:
        backend: An attention backend whose decode metadata a captured graph
            reads.

    Returns:
        ``{path: (data_ptr, shape, stride, dtype, device)}`` — one entry per
        tensor, keyed by its attribute path from the backend.
    """
    out: dict[str, tuple] = {}
    _walk_backend(backend, type(backend).__name__, out, set())
    return out


def verify_graph_metadata(backend, snapshot: dict[str, tuple], *, context: str):
    """Re-walk ``backend``'s decode metadata and compare with ``snapshot``.

    Args:
        backend: The backend that was snapshotted at capture time.
        snapshot: The mapping returned by :func:`snapshot_graph_metadata`.
        context: Replay context for the error message (variant, batch size).

    Raises:
        RuntimeError: A tensor moved, changed geometry, disappeared, or
            appeared — the refresh no longer writes what the graph reads.
    """
    current = snapshot_graph_metadata(backend)
    if current == snapshot:
        return
    lines = []
    for path, recorded in snapshot.items():
        got = current.get(path)
        if got is None:
            lines.append(f"  {path}: recorded {recorded}, now missing")
        elif got != recorded:
            lines.append(f"  {path}: recorded {recorded}, now {got}")
    for path, got in current.items():
        if path not in snapshot:
            lines.append(f"  {path}: {got} was not present at capture")
    raise RuntimeError(
        f"CUDA-graph metadata pointer-stability breach ({context}): the "
        "captured graph reads the addresses recorded at capture, so "
        "refresh_decode_metadata must write the same tensors it bound then "
        "(docs/design/unified_path.md).\n" + "\n".join(lines)
    )
