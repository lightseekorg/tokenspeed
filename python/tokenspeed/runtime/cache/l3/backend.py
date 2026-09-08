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

"""Storage-backend interface for Host CacheBlocks under flat KV."""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

_HF_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}")
_HF_HUB_REPO_DIR_RE = re.compile(r"(?:models|datasets|spaces)--.+")
_CHECKPOINT_METADATA_FILES = (
    "config.json",
    "hf_quant_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "consolidated.safetensors.index.json",
)
# First matching group wins, matching DefaultModelLoader._prepare_weights.
_LOAD_FORMAT_WEIGHT_PATTERN_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "auto": (("*.safetensors",), ("*.bin",), ("*.pt",)),
    "safetensors": (("*.safetensors",),),
    "instanttensor": (("*.safetensors",),),
    "mistral": (("consolidated*.safetensors",),),
    "pt": (("*.pt",),),
    "npcache": (("*.bin",),),
    "sharded_state": (("model-rank-*-part-*.safetensors",), ("*.safetensors",)),
    "dummy": (),
}


def resolve_l3_weight_version(
    current: str,
    requested: str | None,
    *,
    flush_cache: bool,
    storage_backend: str | None,
) -> str | None:
    """Choose the namespace to publish after a successful weight load.

    An explicit ``requested`` version always wins. ``None`` keeps the
    current namespace. Flushed L3 updates must pass a caller-supplied
    identity; minting ``{current}-uN`` would let independent checkpoints
    collide under the same successor.
    """

    del current, flush_cache, storage_backend
    if requested is not None:
        return str(requested)
    return None


L3_FLUSH_REQUIRES_WEIGHT_VERSION = (
    "L3 flushed updates require weight_version so independent replicas "
    "cannot restore another checkpoint's objects"
)


def storage_object_key(
    content_hash: str,
    group_id: int,
    page_offset: int,
    *,
    prefix: str,
    rank: int,
    cp_rank: int,
) -> str:
    """Return the L3 object key for one packed Host CacheBlock.

    TokenSpeed's Host pool is one compact byte buffer (flat KV). One Mooncake
    object stores the packed bytes of a single CacheBlock, keyed by the
    scheduler content hash plus the group/offset/rank that uniquely identify
    the shard. Attention TP and context-parallel ranks each own a different
    physical KV slice; ``ENABLE_CP`` folds requested TP into CP and leaves
    every worker at ``attn_tp_rank == 0``, so the CP rank must be in the key.
    """

    if not content_hash:
        raise ValueError("content_hash must be non-empty")
    tagged = f"{prefix}_{content_hash}" if prefix else content_hash
    return (
        f"{tagged}|g{int(group_id)}|o{int(page_offset)}"
        f"|r{int(rank)}|c{int(cp_rank)}"
    )


def cache_layout_signature(layout: Any, *, cache_dtype: str) -> str:
    """Return a stable fingerprint of the bytes stored in one L3 page.

    The signature is the packed Host CacheBlock: dtype, group packing, and
    each field's payload geometry. Device arena offsets
    (``device_block_zero_offset_bytes``, buffer index) are omitted; later
    planes sit at ``(num_lcm_blocks + 1) * bytes_per_lcm_block``, so GPU
    capacity would otherwise split otherwise identical Mooncake objects.
    """

    groups = []
    for group in layout.groups:
        fields = [
            {
                "id": field.field_id,
                "stride": int(field.block_stride_bytes),
                "payload": int(field.payload_bytes),
            }
            for field in group.fields
        ]
        groups.append(
            {
                "id": group.group_id,
                "blocks_per_lcm": int(group.cache_blocks_per_lcm_block),
                "fields": fields,
            }
        )
    payload = json.dumps(
        {"cache_dtype": str(cache_dtype), "groups": groups},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def l3_cache_quantization_id(
    *,
    quantization: str,
    quantization_param_path: str,
    draft_quantization: str,
) -> str:
    """Return the cache-quantization identity that shapes packed KV bytes.

    FP8 deployments that share ``kv_cache_dtype`` can still load different
    ``quantization_param_path`` scale files. A speculative draft pool packs
    its fields into the same Host CacheBlocks, so ``draft_quantization``
    (``--speculative-draft-model-quantization``) is required: two
    deployments that share a draft checkpoint but quantize it differently
    must not share Mooncake keys. Callers pass empty strings when
    quantization, the scale file, or the draft pool is unset.
    """

    scale_id = ""
    if quantization_param_path:
        if os.path.isfile(quantization_param_path):
            scale_id = _file_digest(quantization_param_path)
        else:
            scale_id = str(quantization_param_path)
    return json.dumps(
        {
            "quantization": str(quantization),
            "scale_id": scale_id,
            "draft_quantization": str(draft_quantization),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def l3_checkpoint_id(
    model_path: str,
    *,
    hf_config: Any,
    revision: str,
    load_format: str,
) -> str:
    """Return an immutable identity for the loaded checkpoint bytes.

    ``--revision`` may be a moving branch or omitted. Two instances that
    resolve different commits (or local trees) must not share Mooncake
    keys. A local directory is identified from a Hugging Face hub cache
    snapshot (``.../(models|datasets|spaces)--<repo>/snapshots/<commit>``
    with a sibling ``refs`` directory) or a fingerprint of the weight
    files ``--load-format`` actually selects — never from
    ``hf_config._commit_hash``, which a copied or fine-tuned tree can
    inherit from its source, and never from a 40-character hex basename
    or a folder merely named ``snapshots``. Local fingerprints also hash
    ``hf_quant_config.json`` so ModelOpt mixed-precision maps and KV
    quantization cannot collide under identical weight bytes, and local
    ``*.py`` (top-level modules and imported package subdirectories) so
    ``--trust-remote-code`` configuration/modeling helpers that derive
    architecture fields cannot share a namespace with identical
    JSON/weights. Hugging Face hub ids still prefer the
    loaded config commit, then a cached snapshot directory, then a
    pinned ``--revision``. The returned id always includes the
    normalized load format so two deployments that share a directory
    (or commit) but select different ``.safetensors`` / ``.bin`` /
    ``.pt`` sets cannot restore each other's KV.
    """

    fmt = _normalize_load_format(load_format)
    if os.path.isdir(model_path):
        snapshot = _snapshot_commit_hash(model_path)
        if snapshot is not None:
            return _checkpoint_id_with_load_format(snapshot, load_format=fmt)
        return _checkpoint_id_with_load_format(
            "local-" + _local_checkpoint_fingerprint(model_path, fmt),
            load_format=fmt,
        )
    commit = getattr(hf_config, "_commit_hash", None)
    if isinstance(commit, str) and _HF_COMMIT_HASH_RE.fullmatch(commit):
        return _checkpoint_id_with_load_format(commit, load_format=fmt)
    model_dir = _resolved_model_dir(model_path, revision=revision)
    if model_dir is not None:
        snapshot = _snapshot_commit_hash(model_dir)
        if snapshot is not None:
            return _checkpoint_id_with_load_format(snapshot, load_format=fmt)
        return _checkpoint_id_with_load_format(
            "local-" + _local_checkpoint_fingerprint(model_dir, fmt),
            load_format=fmt,
        )
    if isinstance(revision, str) and _HF_COMMIT_HASH_RE.fullmatch(revision):
        return _checkpoint_id_with_load_format(revision, load_format=fmt)
    raise ValueError(
        "L3 namespace needs an immutable checkpoint id; pin --revision to a "
        "commit or load from a local snapshot"
    )


def share_l3_checkpoint_ids(
    ids: list[str],
    *,
    rank: int,
    world_size: int,
    gather: Callable[[list], list],
) -> list[str]:
    """Return one replica-wide identity from every rank's local checkpoint ids.

    Each rank fingerprints the files it can read. Rank-local
    ``--load-format sharded_state`` directories hold only
    ``model-rank-{rank}-part-*``, so replacing every rank with rank 0's
    id would keep a Mooncake namespace after another shard changed.
    ``gather`` must implement a replica-wide object all-gather that
    returns one payload list per rank, in rank order. When
    ``world_size`` is 1 the ids are returned unchanged and ``gather`` is
    not called. When every rank reports the same ids, those ids are used
    unchanged (a shared snapshot, or a directory that contains every
    shard).
    """

    if world_size <= 1:
        return list(ids)
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    gathered = gather(list(ids))
    if not isinstance(gathered, list) or len(gathered) != world_size:
        raise ValueError("L3 checkpoint gather must return one payload per rank")
    slot_count = len(ids)
    combined: list[str] = []
    for slot in range(slot_count):
        column: list[str] = []
        for row in gathered:
            if not isinstance(row, list) or len(row) != slot_count:
                raise ValueError(
                    "L3 checkpoint gather payload must match the local id list"
                )
            column.append(str(row[slot]))
        combined.append(_combined_checkpoint_id(column))
    return combined


def _combined_checkpoint_id(column: list[str]) -> str:
    if all(item == column[0] for item in column):
        return column[0]
    payload = json.dumps(column, separators=(",", ":"))
    return "local-" + hashlib.sha256(payload.encode()).hexdigest()


def _snapshot_commit_hash(snapshot_path: str) -> str | None:
    """Return the commit only for a Hugging Face hub cache snapshot path.

    The layout is ``.../(models|datasets|spaces)--<repo>/snapshots/<40-hex>``
    with a sibling ``refs`` directory. A 40-character hex basename is not
    enough, and neither is a folder merely named ``snapshots``: a copied
    or fine-tuned tree such as ``/models/snapshots/<hash>`` can keep
    those names while holding different bytes.
    """

    normalized = os.path.normpath(snapshot_path)
    candidate = os.path.basename(normalized)
    if not _HF_COMMIT_HASH_RE.fullmatch(candidate):
        return None
    snapshot_dir = os.path.dirname(normalized)
    if os.path.basename(snapshot_dir) != "snapshots":
        return None
    repo_dir = os.path.dirname(snapshot_dir)
    if not _HF_HUB_REPO_DIR_RE.fullmatch(os.path.basename(repo_dir)):
        return None
    if not os.path.isdir(os.path.join(repo_dir, "refs")):
        return None
    return candidate


def _resolved_model_dir(model_path: str, *, revision: str) -> str | None:
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model_path,
            revision=revision or None,
            local_files_only=True,
            ignore_patterns=["*.pt", "*.safetensors", "*.bin"],
        )
    except Exception:
        return None


def _normalize_load_format(load_format: str) -> str:
    if not isinstance(load_format, str):
        raise TypeError("load_format must be a str")
    normalized = load_format.strip().lower()
    if not normalized:
        raise ValueError("load_format must be a non-empty str")
    return normalized


def _checkpoint_id_with_load_format(identity: str, *, load_format: str) -> str:
    return f"{identity}:{load_format}"


def _selected_weight_names(names: Sequence[str], *, load_format: str) -> frozenset[str]:
    """Return the weight files ``--load-format`` would load from ``names``.

    Pattern groups match ``DefaultModelLoader._prepare_weights``: the first
    group that matches any file wins, so ``auto`` hashes ``*.safetensors``
    when those exist and does not mix in leftover ``*.bin`` / ``*.pt``.
    ``sharded_state`` hashes ``model-rank-*-part-*.safetensors`` (every
    rank's local files; replica gather combines those digests) and falls
    back to ``*.safetensors``.
    Unknown loaders raise rather than hashing metadata alone.
    """

    groups = _LOAD_FORMAT_WEIGHT_PATTERN_GROUPS.get(load_format)
    if groups is None:
        raise ValueError(
            "L3 cannot fingerprint load-format "
            f"{load_format!r}; unsupported loaders cannot share a Mooncake "
            "namespace"
        )
    for patterns in groups:
        matched = frozenset(
            name
            for name in names
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
        )
        if matched:
            return matched
    return frozenset()


def _is_local_checkpoint_code(name: str) -> bool:
    """Return whether ``name`` is custom HF code loaded with trust_remote_code.

    Configuration and modeling modules, including helpers imported from
    package subdirectories, can derive rope, layout, and other fields
    that change KV without touching ``config.json`` or the weight
    tensors. Hugging Face snapshot commits already cover those files;
    local fingerprints must hash them too.
    """

    return name.endswith(".py")


def _local_checkpoint_code_files(model_dir: str) -> tuple[tuple[str, str], ...]:
    """Return ``(relative posix path, absolute path)`` for local custom code.

    Walks package subdirectories so an imported helper such as
    ``model_helpers/attention.py`` cannot keep the L3 checkpoint id after
    changing KV computation. ``__pycache__`` and hidden directories are
    skipped; bytecode and VCS metadata are not part of the loaded model.
    """

    found: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(
        model_dir, topdown=True, onerror=None, followlinks=False
    ):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name != "__pycache__" and not name.startswith(".")
        )
        rel_dir = os.path.relpath(dirpath, model_dir)
        for name in sorted(filenames):
            if not _is_local_checkpoint_code(name):
                continue
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):
                continue
            if rel_dir == os.curdir:
                rel = name
            else:
                rel = "/".join((*rel_dir.split(os.sep), name))
            found.append((rel, path))
    return tuple(found)


@functools.cache
def _local_checkpoint_fingerprint(model_dir: str, load_format: str) -> str:
    """Hash config/index/quant-config bytes and the selected weight files.

    Cached by directory path and load format so a process that resolves
    the same local checkpoint more than once (target plus draft, or a
    repeated prefix rebuild) does not re-read every shard.
    ``hf_quant_config.json`` is hashed with ``config.json``: ModelOpt
    mixed-precision maps, group sizes, and KV quantization live there,
    not in the weight tensors. ``consolidated.safetensors.index.json`` is
    hashed so two Mistral dumps with the same ``consolidated*.safetensors``
    candidates but different shard maps cannot share a namespace.
    Local ``*.py`` is hashed, including files imported from package
    subdirectories, so two trees with identical JSON/weights but
    different ``--trust-remote-code`` configuration modules cannot share
    a namespace. ``--load-format`` selects so a directory that contains
    more than one checkpoint encoding cannot share a namespace across
    loaders.
    """
    hasher = hashlib.sha256()
    try:
        names = tuple(sorted(os.listdir(model_dir)))
    except OSError:
        return hasher.hexdigest()
    selected_weights = _selected_weight_names(names, load_format=load_format)
    for rel, path in _local_checkpoint_code_files(model_dir):
        hasher.update(rel.encode())
        _update_file_digest(hasher, path)
    for name in names:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            continue
        if name in _CHECKPOINT_METADATA_FILES:
            hasher.update(name.encode())
            _update_file_digest(hasher, path)
            continue
        if name in selected_weights:
            hasher.update(name.encode())
            _update_file_digest(hasher, path)
    return hasher.hexdigest()


def _update_file_digest(hasher, path: str) -> None:
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)


def _file_digest(path: str) -> str:
    hasher = hashlib.sha256()
    _update_file_digest(hasher, path)
    return hasher.hexdigest()


def storage_key_prefix(
    model_name: str,
    *,
    revision: str,
    weight_version: str,
    model_overrides: dict,
    cache_signature: str,
    pipeline_rank: int,
    attn_tp_size: int,
    cp_size: int,
    draft_model: str,
    draft_revision: str,
    draft_weight_version: str,
    cache_quantization: str,
) -> str:
    """Return a collision-resistant namespace for compatible L3 objects.

    Every component is required so a new caller cannot omit the checkpoint
    identity, cache layout, pipeline stage, attention-TP width,
    context-parallel width, draft pool, cache-quantization config (target
    and draft), or runtime HF overrides and silently collide with an
    incompatible deployment.
    ``revision`` is the resolved immutable checkpoint (Hugging Face commit
    or local fingerprint), not a moving branch name. ``model_overrides``
    is the ``--hf-overrides`` dict applied to the HF text config
    (rope_theta, rope_scaling, and the rest of the effective architecture).
    Empty strings and an empty override dict are valid and mean "unset"
    (no draft pool, no extra cache scales, no HF overrides). ``cp_size``
    belongs here rather than only in the per-object ``c{cp_rank}`` shard
    id: zigzag CP assigns different token blocks to the same rank under
    different widths. ``attn_tp_size`` belongs here rather than only in
    the per-object ``r{tp_rank}`` shard id: GQA with TP above the KV-head
    count keeps one local KV head per rank, so packed Host geometry is
    unchanged, while ``tp_rank // num_kv_head_replicas`` assigns different
    heads to the same rank.
    """

    if not isinstance(model_overrides, dict):
        raise TypeError("model_overrides must be a dict")
    payload = json.dumps(
        {
            "model": str(model_name),
            "revision": str(revision),
            "weight_version": str(weight_version),
            "model_overrides": model_overrides,
            "cache_signature": str(cache_signature),
            "pipeline_rank": int(pipeline_rank),
            "attn_tp_size": int(attn_tp_size),
            "cp_size": int(cp_size),
            "draft_model": str(draft_model),
            "draft_revision": str(draft_revision),
            "draft_weight_version": str(draft_weight_version),
            "cache_quantization": str(cache_quantization),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "tsl3v1-" + hashlib.sha256(payload.encode()).hexdigest()


def host_buffer_ptr(host_buffer: Any) -> int:
    data_ptr = getattr(host_buffer, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    raise TypeError(f"host buffer {type(host_buffer)!r} has no data_ptr()")


def copy_host_bytes(host_buffer: Any, offset: int, size: int) -> bytes:
    view = host_buffer[offset : offset + size]
    tobytes = getattr(view, "tobytes", None)
    if callable(tobytes):
        return bytes(tobytes())
    numpy = getattr(view, "numpy", None)
    if callable(numpy):
        return bytes(numpy())
    return bytes(view)


def l3_unread_key_capacity(
    *, num_host_pages: int, cache_blocks_per_lcm_block: Sequence[int]
) -> int:
    """Return the unread-set bound in per-group CacheBlocks, not LCM parents.

    Each Host LCM parent packs ``cache_blocks_per_lcm_block`` CacheBlocks
    per group. Unread keys are those CacheBlocks, so a single multi-group
    prefetch can insert more entries than ``num_host_pages``. Matching the
    scheduler L3 shadow, capacity is ``num_host_pages`` times the sum of
    each group's packing.
    """

    packed = 0
    for count in cache_blocks_per_lcm_block:
        if int(count) <= 0:
            raise ValueError("cache_blocks_per_lcm_block must be positive")
        packed += int(count)
    if packed <= 0:
        return max(int(num_host_pages), 1)
    return max(int(num_host_pages) * packed, 1)


def l3_pages_newly_published(
    pages: Sequence[tuple], existed: Sequence[bool]
) -> list[tuple]:
    """Return pages that were absent before backup and may leave the unread set.

    Mooncake puts are create-only. An object that ``batch_exists`` already
    reports cannot be overwritten, so a failed ``batch_get_into`` of that
    object must stay unread. Length mismatch returns no pages so a
    truncated existence probe cannot clear the blacklist.
    """

    if len(existed) != len(pages):
        return []
    return [page for page, present in zip(pages, existed) if not present]


class L3UnreadKeySet:
    """Failed L3 gets that must not be re-admitted from ``batch_exists``.

    A vanished or unreadable object can stay visible to ``batch_exists``.
    Those keys stay unread until a Host backup creates a replacement
    object (not a create-only skip of the existing one), a namespace
    delete succeeds, or the set exceeds Host CacheBlock
    capacity (oldest first) so a long-lived process cannot accumulate
    every historical failure. Replica admission MIN-reduces local
    readability so one rank cannot forget earlier than its peers.
    """

    def __init__(self, *, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("L3 unread capacity must be positive")
        self._capacity = int(capacity)
        self._keys: OrderedDict[tuple[int, str, int], None] = OrderedDict()
        self._lock = threading.Lock()

    def mark(
        self,
        groups: Sequence[int],
        hashes: Sequence[str],
        offsets: Sequence[int],
    ) -> None:
        with self._lock:
            for group_id, content_hash, page_offset in zip(groups, hashes, offsets):
                key = (int(group_id), str(content_hash), int(page_offset))
                self._keys.pop(key, None)
                self._keys[key] = None
            while len(self._keys) > self._capacity:
                self._keys.popitem(last=False)

    def contains(self, group_id: int, content_hash: str, page_offset: int) -> bool:
        with self._lock:
            return (
                int(group_id),
                str(content_hash),
                int(page_offset),
            ) in self._keys

    def forget(
        self,
        groups: Sequence[int],
        hashes: Sequence[str],
        offsets: Sequence[int],
    ) -> None:
        with self._lock:
            for group_id, content_hash, page_offset in zip(groups, hashes, offsets):
                self._keys.pop(
                    (int(group_id), str(content_hash), int(page_offset)),
                    None,
                )

    def forget_pages(self, pages: Sequence[tuple]) -> None:
        groups = []
        hashes = []
        offsets = []
        for group_id, _host_block, content_hash, page_offset in pages:
            groups.append(int(group_id))
            hashes.append(str(content_hash))
            offsets.append(int(page_offset))
        self.forget(groups=groups, hashes=hashes, offsets=offsets)

    def clear(self) -> None:
        with self._lock:
            self._keys.clear()


def write_host_bytes(host_buffer: Any, offset: int, payload: bytes) -> None:
    size = len(payload)
    dest = host_buffer[offset : offset + size]
    copy_ = getattr(dest, "copy_", None)
    if callable(copy_):
        import torch

        copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
        return
    host_buffer[offset : offset + size] = payload


class KvStoreStorage(Protocol):
    """Byte store for packed Host CacheBlocks.

    Implementations must be safe to call from the runtime thread that owns
    the Host buffer. ``batch_get_into`` / ``batch_put_from`` operate on
    offsets into that registered buffer (SGLang HiCacheStorage v1).
    """

    def batch_exists(self, keys: Sequence[str]) -> list[bool]:
        """Return per-key existence, aligned with ``keys``."""

    def batch_get_into(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        """Read objects into Host buffer slices. True means the copy succeeded."""

    def batch_put_from(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        """Write Host buffer slices into the store. True means the put succeeded."""

    def remove_by_prefix(self, prefix: str) -> bool:
        """Remove every object whose key starts with ``prefix``.

        Returns True when matching objects are gone. False must not be
        followed by an irreversible Device/Host ``ClearCache``.
        """

    def close(self) -> None:
        """Release backend resources. Idempotent."""


class MemoryKvStore:
    """In-process dict store used by tests and as a Mooncake-free reference."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def batch_exists(self, keys: Sequence[str]) -> list[bool]:
        return [key in self._objects for key in keys]

    def batch_get_into(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 get")
        results = []
        for key, offset, size in zip(keys, offsets, sizes):
            payload = self._objects.get(key)
            if payload is None or len(payload) != size:
                results.append(False)
                continue
            write_host_bytes(host_buffer, int(offset), payload)
            results.append(True)
        return results

    def batch_put_from(
        self,
        keys: Sequence[str],
        host_buffer: Any,
        offsets: Sequence[int],
        sizes: Sequence[int],
    ) -> list[bool]:
        if not (len(keys) == len(offsets) == len(sizes)):
            raise ValueError("ragged L3 put")
        results = []
        for key, offset, size in zip(keys, offsets, sizes):
            if key in self._objects:
                results.append(True)
                continue
            self._objects[key] = copy_host_bytes(host_buffer, int(offset), int(size))
            results.append(True)
        return results

    def remove_by_prefix(self, prefix: str) -> bool:
        self._objects = {
            key: payload
            for key, payload in self._objects.items()
            if not key.startswith(prefix)
        }
        return True

    def close(self) -> None:
        self._objects.clear()
