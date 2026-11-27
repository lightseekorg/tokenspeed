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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Object-store (GCS / S3) metadata sync and byte-range safetensors helpers."""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import stat
import struct
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

_OBJECT_STORE_SCHEMES = ("gs://", "s3://")
_METADATA_FILE_SUFFIXES = (
    ".json",
    ".py",
    ".jinja",
    ".tiktoken",
    ".model",
    ".spm",
    ".txt",
    ".yaml",
    ".yml",
)
_NON_METADATA_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
    ".h5",
    ".hdf5",
    ".ckpt",
    ".onnx",
    ".msgpack",
    ".tar",
    ".gz",
    ".tgz",
    ".zip",
    ".zst",
    ".bz2",
    ".xz",
    ".parquet",
    ".arrow",
    ".npz",
    ".npy",
    ".ot",
    ".tflite",
)
_METADATA_WARN_BYTES = 64 * 1024 * 1024
_METADATA_MARKER_FILENAME = ".tokenspeed_metadata_synced"
_METADATA_MANIFEST_VERSION = 1
_SOURCE_URI_FILENAME = ".tokenspeed_source_uri"
_MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024

# Match safetensors' PyTorch dtype decoding (float8_e8m0fnu / float4_e2m1fn_x2
# when present in the installed PyTorch build, falling back to uint8 on older builds).
_F8_E8M0_DTYPE: torch.dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
_F4_E2M1_DTYPE: torch.dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)

# Mapping from safetensors dtype tags to (torch.dtype, element_bytes_or_none).
# Sub-byte packed dtypes (F4) have element_bytes=None (2 values per packed byte
# along the last dimension), matching safetensors' own PyTorch representation.
_SAFETENSORS_DTYPE_MAP: dict[str, tuple[torch.dtype, int | None]] = {
    "BOOL": (torch.bool, 1),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "F8_E8M0": (_F8_E8M0_DTYPE, 1),
    "U16": (torch.uint16, 2),
    "I16": (torch.int16, 2),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "U32": (torch.uint32, 4),
    "I32": (torch.int32, 4),
    "F32": (torch.float32, 4),
    "U64": (torch.uint64, 8),
    "I64": (torch.int64, 8),
    "F64": (torch.float64, 8),
    "F4": (_F4_E2M1_DTYPE, None),
    "F4_E2M1": (_F4_E2M1_DTYPE, None),
}


def is_object_store_uri(path: str) -> bool:
    """Return whether ``path`` is a supported object-store URI (``gs://`` or ``s3://``)."""
    return path.startswith(_OBJECT_STORE_SCHEMES)


def parse_object_store_uri(uri: str) -> tuple[str, str, str]:
    """Split ``gs://bucket/prefix`` or ``s3://bucket/prefix`` into ``(scheme, bucket, prefix)``."""
    if not is_object_store_uri(uri):
        raise ValueError(f"Unsupported object-store URI: {uri!r}")
    scheme, _, rest = uri.partition("://")
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"Object-store URI is missing bucket name: {uri!r}")
    return scheme, bucket, prefix.strip("/")


def join_object_store_uri(base_uri: str, relative_path: str) -> str:
    """Join ``base_uri`` and ``relative_path`` with a single slash."""
    return f"{base_uri.rstrip('/')}/{relative_path.lstrip('/')}"


def resolve_metadata_cache_dir(cache_dir: str | None = None) -> str:
    """Resolve the base directory for synced object-store checkpoint metadata.

    When ``cache_dir`` is omitted, uses a user-owned cache location
    (``$HF_HOME/tokenspeed-object-store-meta``, ``$XDG_CACHE_HOME/tokenspeed/object-store-meta``,
    or ``~/.cache/tokenspeed/object-store-meta``) rather than a world-writable
    system temporary directory.
    """
    if cache_dir:
        return os.path.join(cache_dir, "tokenspeed-object-store-meta")
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "tokenspeed-object-store-meta")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return os.path.join(xdg_cache, "tokenspeed", "object-store-meta")
    return os.path.join(
        os.path.expanduser("~"), ".cache", "tokenspeed", "object-store-meta"
    )


def _verify_owned_path(path: str, *, expect_dir: bool) -> None:
    """Verify ``path`` is not a symlink, matches expected type, and is user-owned."""
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise PermissionError(f"Refusing to trust symlinked cache path: {path!r}")
    if expect_dir and not stat.S_ISDIR(st.st_mode):
        raise PermissionError(f"Expected cache directory at {path!r}")
    if not expect_dir and not stat.S_ISREG(st.st_mode):
        raise PermissionError(f"Expected regular cache file at {path!r}")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise PermissionError(
            f"Refusing to trust cache path {path!r} owned by uid {st.st_uid} "
            f"(expected uid {os.getuid()})"
        )


def read_source_uri(local_dir: str) -> str | None:
    """Return the remote object-store URI recorded for a synced metadata directory, if any."""
    marker = os.path.join(local_dir, _SOURCE_URI_FILENAME)
    if not os.path.isfile(marker):
        return None
    uri = Path(marker).read_text(encoding="utf-8").strip()
    return uri if is_object_store_uri(uri) else None


def _is_metadata_object(relative_name: str) -> bool:
    """Return whether ``relative_name`` matches the checkpoint metadata allowlist."""
    clean_parts = [
        part for part in relative_name.split("/") if part and part not in (".", "..")
    ]
    if not clean_parts:
        return False
    if any(part == "__pycache__" or part.startswith(".") for part in clean_parts):
        return False
    lower = relative_name.lower()
    if lower.endswith(_NON_METADATA_SUFFIXES):
        return False
    if clean_parts[0] == "encoding":
        return True
    return lower.endswith(_METADATA_FILE_SUFFIXES)


class _ClientLocal(threading.local):
    """Thread-local cache for GCS / S3 SDK clients."""

    def __init__(self) -> None:
        super().__init__()
        self.gcs_client: Any = None
        self.s3_client: Any = None


_CLIENT_LOCAL = _ClientLocal()

_RANGE_READ_EXECUTOR_LOCK = threading.Lock()
_RANGE_READ_EXECUTOR: ThreadPoolExecutor | None = None


def _shutdown_range_read_executor() -> None:
    global _RANGE_READ_EXECUTOR
    with _RANGE_READ_EXECUTOR_LOCK:
        executor = _RANGE_READ_EXECUTOR
        _RANGE_READ_EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=False)


atexit.register(_shutdown_range_read_executor)


def _get_range_read_executor() -> ThreadPoolExecutor:
    """Return a shared thread pool so worker threads keep warm GCS/S3 clients."""
    global _RANGE_READ_EXECUTOR
    with _RANGE_READ_EXECUTOR_LOCK:
        if _RANGE_READ_EXECUTOR is None:
            _RANGE_READ_EXECUTOR = ThreadPoolExecutor(
                max_workers=RemoteSafetensorsSlice._PREFETCH_WORKERS,
                thread_name_prefix="tokenspeed-objstore",
            )
        return _RANGE_READ_EXECUTOR


def _gcs_client():
    if _CLIENT_LOCAL.gcs_client is not None:
        return _CLIENT_LOCAL.gcs_client
    try:
        from google.cloud import storage
    except ImportError as e:
        raise ImportError(
            "Reading gs:// URIs requires google-cloud-storage; install it in your environment."
        ) from e
    _CLIENT_LOCAL.gcs_client = storage.Client()
    return _CLIENT_LOCAL.gcs_client


def _s3_client():
    if _CLIENT_LOCAL.s3_client is not None:
        return _CLIENT_LOCAL.s3_client
    try:
        import boto3
    except ImportError as e:
        raise ImportError(
            "Reading s3:// URIs requires boto3; install it in your environment."
        ) from e
    _CLIENT_LOCAL.s3_client = boto3.client("s3")
    return _CLIENT_LOCAL.s3_client


def _list_object_store_objects(uri: str) -> list[tuple[str, str]]:
    """List ``(relative_path, version_token)`` for non-directory objects under ``uri``.

    ``version_token`` records GCS ``generation`` or S3 ``ETag`` so cached
    metadata directories detect updated remote checkpoints.
    """
    scheme, bucket_name, prefix = parse_object_store_uri(uri)
    list_prefix = f"{prefix}/" if prefix else ""
    entries: list[tuple[str, str]] = []
    if scheme == "gs":
        client = _gcs_client()
        # Use bucket() rather than get_bucket() so callers without bucket-get IAM work.
        bucket = client.bucket(bucket_name)
        for blob in bucket.list_blobs(prefix=list_prefix):
            rel = blob.name[len(list_prefix) :]
            if rel and not rel.endswith("/"):
                entries.append((rel, f"gen:{blob.generation}"))
    elif scheme == "s3":
        client = _s3_client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=list_prefix):
            for obj in page.get("Contents", ()):
                rel = obj["Key"][len(list_prefix) :]
                if rel and not rel.endswith("/"):
                    etag = str(obj.get("ETag", "")).strip('"')
                    size = obj.get("Size")
                    version = (
                        f"etag:{etag}:size:{size}"
                        if size is not None
                        else f"etag:{etag}"
                    )
                    entries.append((rel, version))
    else:
        raise ValueError(f"Unsupported object-store scheme: {scheme!r}")
    entries.sort(key=lambda item: item[0])
    return entries


def _download_object_to_file(uri: str, dst_path: str) -> None:
    """Download a single object-store URI to ``dst_path`` atomically."""
    scheme, bucket_name, key = parse_object_store_uri(uri)
    os.makedirs(os.path.dirname(dst_path), mode=0o700, exist_ok=True)
    tmp_path = f"{dst_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        if scheme == "gs":
            client = _gcs_client()
            blob = client.bucket(bucket_name).blob(key)
            blob.download_to_filename(tmp_path)
        elif scheme == "s3":
            client = _s3_client()
            client.download_file(bucket_name, key, tmp_path)
        else:
            raise ValueError(f"Unsupported object-store scheme: {scheme!r}")
        os.replace(tmp_path, dst_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _write_text_atomic(path: str, content: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        Path(tmp_path).write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _read_metadata_manifest(marker_path: str) -> dict[str, Any] | None:
    if not os.path.lexists(marker_path):
        return None
    _verify_owned_path(marker_path, expect_dir=False)
    try:
        parsed = json.loads(Path(marker_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("version") != _METADATA_MANIFEST_VERSION:
        return None
    if not isinstance(parsed.get("uri"), str) or not isinstance(
        parsed.get("objects"), dict
    ):
        return None
    return parsed


def _resolve_safe_target(local_dir: str, rel: str, source_uri: str) -> str:
    """Resolve ``rel`` inside ``local_dir`` and reject path traversal."""
    target = os.path.realpath(os.path.join(local_dir, rel))
    root_real = os.path.realpath(local_dir)
    if not (target == root_real or target.startswith(root_real + os.sep)):
        raise ValueError(f"Unsafe object key {rel!r} under {source_uri!r}")
    return target


def _manifest_matches_local(
    local_dir: str,
    marker_path: str,
    normalized_uri: str,
    expected_objects: dict[str, str],
) -> bool:
    manifest = _read_metadata_manifest(marker_path)
    if manifest is None:
        return False
    if manifest["uri"] != normalized_uri or manifest["objects"] != expected_objects:
        return False
    for rel in expected_objects:
        target = _resolve_safe_target(local_dir, rel, normalized_uri)
        if not os.path.lexists(target):
            return False
        _verify_owned_path(target, expect_dir=False)
    return True


def read_object_store_range(uri: str, start: int, end_exclusive: int) -> bytes:
    """Read byte range ``[start, end_exclusive)`` from a ``gs://`` or ``s3://`` object."""
    if start < 0 or end_exclusive < start:
        raise ValueError(f"Invalid byte range [{start}, {end_exclusive}) for {uri!r}")
    if start == end_exclusive:
        return b""
    scheme, bucket_name, key = parse_object_store_uri(uri)
    end_inclusive = end_exclusive - 1
    if scheme == "gs":
        client = _gcs_client()
        blob = client.bucket(bucket_name).blob(key)
        return blob.download_as_bytes(start=start, end=end_inclusive)
    if scheme == "s3":
        client = _s3_client()
        resp = client.get_object(
            Bucket=bucket_name, Key=key, Range=f"bytes={start}-{end_inclusive}"
        )
        return resp["Body"].read()
    raise ValueError(f"Unsupported object-store scheme: {scheme!r}")


def _metadata_cache_paths(
    uri: str, cache_dir: str | None = None
) -> tuple[str, str, str]:
    """Return ``(normalized_uri, base_cache, local_dir)`` for ``uri``."""
    normalized = uri.rstrip("/")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    base_cache = resolve_metadata_cache_dir(cache_dir)
    return normalized, base_cache, os.path.join(base_cache, digest)


def sync_object_store_metadata(
    uri: str,
    cache_dir: str | None = None,
    *,
    list_objects: Callable[[str], list[tuple[str, str]]] | None = None,
    download_object: Callable[[str, str], None] | None = None,
) -> str:
    """Sync non-weight checkpoint metadata from ``gs://`` or ``s3://`` to a local cache.

    Downloads small metadata files (``config.json``, tokenizer assets,
    ``model.safetensors.index.json``, custom ``*.py`` modules, ``encoding/*``)
    while skipping weight files (``*.safetensors``, ``*.bin``, ``*.pt``).
    Concurrent ranks synchronize on a cross-process file lock, verify cache
    ownership, and compare remote object versions (GCS ``generation`` / S3
    ``ETag``) against the local manifest so stale metadata is refreshed
    automatically when remote objects change.
    """
    from tokenspeed.runtime.model_loader.weight_utils import get_lock

    normalized, base_cache, local_dir = _metadata_cache_paths(uri, cache_dir)
    marker_path = os.path.join(local_dir, _METADATA_MARKER_FILENAME)

    list_fn = _list_object_store_objects if list_objects is None else list_objects
    download_fn = (
        _download_object_to_file if download_object is None else download_object
    )

    # Re-list remote objects on each call so updated remote metadata versions
    # invalidate the local cache; checkpoint prefixes typically hold ~O(100)
    # objects so a metadata listing is inexpensive compared to model loading.
    remote_entries = list_fn(normalized)
    expected_objects = {
        rel: version for rel, version in remote_entries if _is_metadata_object(rel)
    }

    if os.path.lexists(local_dir):
        _verify_owned_path(local_dir, expect_dir=True)
        if _manifest_matches_local(
            local_dir, marker_path, normalized, expected_objects
        ):
            return local_dir

    os.makedirs(base_cache, mode=0o700, exist_ok=True)
    with get_lock(f"object-store-meta:{normalized}", base_cache):
        if os.path.lexists(local_dir):
            _verify_owned_path(local_dir, expect_dir=True)
            if _manifest_matches_local(
                local_dir, marker_path, normalized, expected_objects
            ):
                return local_dir
        else:
            os.makedirs(local_dir, mode=0o700, exist_ok=True)
            os.chmod(local_dir, 0o700)

        prev_manifest = _read_metadata_manifest(marker_path)
        prev_objects: dict[str, str] = (
            dict(prev_manifest["objects"])
            if prev_manifest is not None and prev_manifest["uri"] == normalized
            else {}
        )

        # Validate all remote object keys before writing or removing any files.
        resolved_targets = {
            rel: _resolve_safe_target(local_dir, rel, normalized)
            for rel in expected_objects
        }

        to_download = [
            rel
            for rel, ver in expected_objects.items()
            if prev_objects.get(rel) != ver or not os.path.isfile(resolved_targets[rel])
        ]
        to_remove = [rel for rel in prev_objects if rel not in expected_objects]

        logger.info(
            f"Syncing {len(to_download)} metadata file(s) "
            f"(removing {len(to_remove)} stale) from {normalized} to {local_dir}"
        )
        for rel in to_remove:
            stale_target = _resolve_safe_target(local_dir, rel, normalized)
            if os.path.isfile(stale_target):
                os.unlink(stale_target)

        for rel in to_download:
            target = resolved_targets[rel]
            os.makedirs(os.path.dirname(target), mode=0o700, exist_ok=True)
            download_fn(join_object_store_uri(normalized, rel), target)
            if os.path.isfile(target):
                size_bytes = os.path.getsize(target)
                if size_bytes > _METADATA_WARN_BYTES:
                    logger.warning(
                        f"Synced metadata object {rel!r} from {normalized} is "
                        f"{size_bytes / (1024 * 1024):.1f} MiB (> "
                        f"{_METADATA_WARN_BYTES // (1024 * 1024)} MiB)."
                    )

        _write_text_atomic(
            os.path.join(local_dir, _SOURCE_URI_FILENAME), normalized + "\n"
        )
        manifest_payload = json.dumps(
            {
                "version": _METADATA_MANIFEST_VERSION,
                "uri": normalized,
                "objects": expected_objects,
            },
            sort_keys=True,
        )
        _write_text_atomic(marker_path, manifest_payload + "\n")
    return local_dir


def resolve_checkpoint_metadata_dir(
    model_path: str,
    *,
    sync: bool,
    download_dir: str | None = None,
) -> str | None:
    """Resolve a local metadata directory for a local path or object-store URI.

    Args:
        model_path: Local directory, object-store URI (``gs://`` / ``s3://``),
            or Hugging Face Hub repository id.
        sync: When ``True``, sync remote metadata via
            :func:`sync_object_store_metadata` for object-store URIs. When
            ``False``, return an already-synced cache directory only if it
            exists, passes ownership checks, and carries a valid manifest for
            ``model_path`` (making no network calls), or ``None`` otherwise.
        download_dir: Optional base cache directory (``ServerArgs.download_dir``).

    Returns:
        A local directory path for local directories and object-store URIs
        (when ``sync=True`` or already cached), or ``None`` for non-local
        non-object-store identifiers (such as Hugging Face Hub repo ids) or
        uncached object-store URIs when ``sync=False``.
    """
    if not model_path:
        return None
    if os.path.isdir(model_path):
        return model_path
    if not is_object_store_uri(model_path):
        return None
    if sync:
        return sync_object_store_metadata(model_path, cache_dir=download_dir)
    normalized, _, local_dir = _metadata_cache_paths(model_path, download_dir)
    if not os.path.lexists(local_dir):
        return None
    _verify_owned_path(local_dir, expect_dir=True)
    manifest = _read_metadata_manifest(
        os.path.join(local_dir, _METADATA_MARKER_FILENAME)
    )
    if manifest is None or manifest["uri"] != normalized:
        return None
    return local_dir


def list_object_store_safetensors(uri: str) -> list[str]:
    """List top-level ``*.safetensors`` object URIs under ``uri``."""
    normalized = uri.rstrip("/")
    entries = _list_object_store_objects(normalized)
    return [
        join_object_store_uri(normalized, rel)
        for rel, _ in entries
        if "/" not in rel and rel.endswith(".safetensors")
    ]


def _decode_safetensors_bytes(
    raw: bytes,
    dtype_str: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Decode raw little-endian safetensors bytes into a CPU ``torch.Tensor``."""
    if dtype_str not in _SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype_str!r}")
    torch_dtype, elem_bytes = _SAFETENSORS_DTYPE_MAP[dtype_str]
    expected_shape = shape
    if elem_bytes is None:
        # Sub-byte packed FP4 tensors store 2 values per packed byte along the last dim.
        if not shape or shape[-1] % 2 != 0:
            raise ValueError(
                f"Sub-byte safetensors dtype {dtype_str!r} requires even last dim, got {shape}"
            )
        expected_shape = (*shape[:-1], shape[-1] // 2)
        expected_bytes = math.prod(expected_shape)
    else:
        expected_bytes = math.prod(shape) * elem_bytes
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Safetensors byte length mismatch for {dtype_str} {shape}: "
            f"expected {expected_bytes} bytes, got {len(raw)}"
        )
    if expected_bytes == 0:
        return torch.empty(expected_shape, dtype=torch_dtype, device="cpu")
    buf = bytearray(raw)
    return (
        torch.frombuffer(buf, dtype=torch.uint8).view(torch_dtype).view(expected_shape)
    )


class RemoteSafetensorsSlice:
    """Lazy row-slice reader implementing safetensors' ``get_slice`` protocol over range reads.

    Sequential forward row scans (such as loading a large embedding table in
    fixed row chunks) automatically coalesce into bounded parallel range reads
    to saturate object-store bandwidth without materializing the full tensor.
    """

    _PREFETCH_WINDOW_BYTES = 256 * 1024 * 1024
    _PREFETCH_PART_BYTES = 32 * 1024 * 1024
    _PREFETCH_WORKERS = 8

    def __init__(
        self,
        *,
        range_reader: Callable[[int, int], bytes],
        dtype_str: str,
        shape: tuple[int, ...],
        tensor_data_start: int,
        tensor_data_end: int,
        prefetch_window_bytes: int | None = None,
    ) -> None:
        self._range_reader = range_reader
        self._dtype_str = dtype_str
        self._shape = shape
        self._tensor_data_start = tensor_data_start
        self._tensor_data_end = tensor_data_end
        self._prefetch_window_bytes: int = (
            self._PREFETCH_WINDOW_BYTES
            if prefetch_window_bytes is None
            else prefetch_window_bytes
        )
        self._buf: bytes = b""
        self._buf_start: int = 0
        self._buf_end: int = 0
        self._last_read_end: int = -1
        self._sequential_hits: int = 0
        if dtype_str not in _SAFETENSORS_DTYPE_MAP:
            raise ValueError(f"Unsupported safetensors dtype: {dtype_str!r}")
        _, elem_bytes = _SAFETENSORS_DTYPE_MAP[dtype_str]
        if elem_bytes is None:
            if len(shape) < 2:
                raise ValueError(
                    f"Row slicing is not supported for sub-byte safetensors dtype "
                    f"{dtype_str!r} with rank < 2 (shape={shape})"
                )
            if shape[-1] % 2 != 0:
                raise ValueError(
                    f"Sub-byte safetensors dtype {dtype_str!r} requires even last dim, got {shape}"
                )
            self._row_bytes = math.prod(shape[1:-1]) * (shape[-1] // 2)
        else:
            self._row_bytes = math.prod(shape[1:]) * elem_bytes if shape else 0

    def _read_parallel(self, start: int, end: int) -> bytes:
        total = end - start
        if total <= self._PREFETCH_PART_BYTES:
            return self._range_reader(start, end)
        parts: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            nxt = min(pos + self._PREFETCH_PART_BYTES, end)
            parts.append((pos, nxt))
            pos = nxt
        executor = _get_range_read_executor()
        chunks = list(executor.map(lambda se: self._range_reader(se[0], se[1]), parts))
        return b"".join(chunks)

    def _read_bytes(self, start: int, end: int) -> bytes:
        if start == end:
            return b""
        if self._buf_start <= start and end <= self._buf_end:
            rel_s = start - self._buf_start
            rel_e = end - self._buf_start
            self._last_read_end = end
            return self._buf[rel_s:rel_e]

        if start == self._last_read_end:
            self._sequential_hits += 1
        else:
            self._sequential_hits = 0

        if self._prefetch_window_bytes > (end - start) and self._sequential_hits >= 1:
            fetch_end = min(
                self._tensor_data_end,
                start + self._prefetch_window_bytes,
            )
            self._buf = self._read_parallel(start, fetch_end)
            self._buf_start = start
            self._buf_end = fetch_end
            self._last_read_end = end
            return self._buf[: end - start]

        self._buf = b""
        self._buf_start = 0
        self._buf_end = 0
        self._last_read_end = end
        return self._read_parallel(start, end)

    def get_shape(self) -> list[int]:
        return list(self._shape)

    def __getitem__(self, key: Any) -> torch.Tensor:
        if not self._shape:
            raw = self._read_bytes(self._tensor_data_start, self._tensor_data_end)
            return _decode_safetensors_bytes(raw, self._dtype_str, self._shape)
        if isinstance(key, tuple):
            row_sel = key[0]
            rest = key[1:]
            for s in rest:
                if not (isinstance(s, slice) and s == slice(None)):
                    raise NotImplementedError(
                        f"RemoteSafetensorsSlice supports full trailing slices only, got {key!r}"
                    )
        else:
            row_sel = key

        if isinstance(row_sel, int):
            idx = row_sel if row_sel >= 0 else self._shape[0] + row_sel
            if not 0 <= idx < self._shape[0]:
                raise IndexError(f"Row index {row_sel} out of range for {self._shape}")
            start_byte = self._tensor_data_start + idx * self._row_bytes
            end_byte = start_byte + self._row_bytes
            raw = self._read_bytes(start_byte, end_byte)
            return _decode_safetensors_bytes(raw, self._dtype_str, self._shape[1:])

        if not isinstance(row_sel, slice) or row_sel.step not in (None, 1):
            raise NotImplementedError(
                f"RemoteSafetensorsSlice supports contiguous row slices only, got {key!r}"
            )
        start_row, stop_row, _ = row_sel.indices(self._shape[0])
        num_rows = max(0, stop_row - start_row)
        start_byte = self._tensor_data_start + start_row * self._row_bytes
        end_byte = start_byte + num_rows * self._row_bytes
        raw = self._read_bytes(start_byte, end_byte) if num_rows > 0 else b""
        return _decode_safetensors_bytes(
            raw, self._dtype_str, (num_rows, *self._shape[1:])
        )


class RemoteSafetensorsFile:
    """Safetensors reader backed by byte-range reads (object store or custom reader)."""

    def __init__(
        self,
        uri: str,
        *,
        range_reader: Callable[[int, int], bytes] | None = None,
    ) -> None:
        self.uri = uri
        self._range_reader: Callable[[int, int], bytes] = (
            range_reader
            if range_reader is not None
            else (lambda s, e: read_object_store_range(uri, s, e))
        )
        self._header: dict[str, Any] = {}
        self._data_base: int = 0
        self._load_header()

    def _load_header(self) -> None:
        len_bytes = self._range_reader(0, 8)
        if len(len_bytes) != 8:
            raise ValueError(f"{self.uri}: truncated safetensors header length")
        (header_len,) = struct.unpack("<Q", len_bytes)
        if header_len > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"{self.uri}: safetensors header claims {header_len} bytes"
            )
        header_bytes = self._range_reader(8, 8 + header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{self.uri}: truncated safetensors header")
        parsed = json.loads(header_bytes.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"{self.uri}: safetensors header is not a JSON object")
        self._header = parsed
        self._data_base = 8 + header_len

    def keys(self) -> list[str]:
        return [k for k in self._header if k != "__metadata__"]

    def _entry(self, name: str) -> tuple[str, tuple[int, ...], int, int]:
        if name not in self._header or name == "__metadata__":
            raise KeyError(f"Tensor {name!r} not found in {self.uri}")
        meta = self._header[name]
        dtype_str = str(meta["dtype"])
        shape = tuple(int(dim) for dim in meta["shape"])
        rel_start, rel_end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        return dtype_str, shape, self._data_base + rel_start, self._data_base + rel_end

    def get_tensor(self, name: str) -> torch.Tensor:
        dtype_str, shape, start, end = self._entry(name)
        raw = self._range_reader(start, end)
        return _decode_safetensors_bytes(raw, dtype_str, shape)

    def get_slice(self, name: str) -> RemoteSafetensorsSlice:
        dtype_str, shape, start, end = self._entry(name)
        return RemoteSafetensorsSlice(
            range_reader=self._range_reader,
            dtype_str=dtype_str,
            shape=shape,
            tensor_data_start=start,
            tensor_data_end=end,
        )
