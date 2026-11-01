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

"""The L3 namespace's identity: everything that shapes the stored bytes.

Objects written under one identity are never read under another: the
checkpoints, the cache layout and quantization, and the producer facts that
change the bytes go into the deployment id, and the weight versions into the
namespace. The engine host names and moves the objects (``ts-l3``) from the
id this module computes at startup.
"""

from __future__ import annotations

import ast
import fnmatch
import functools
import glob
import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from typing import Any

_HF_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}")
_HF_HUB_REPO_DIR_RE = re.compile(r"(?:models|datasets|spaces)--.+")
_EXT_DEF_FILE_BLOCK_RE = re.compile(
    r"^(?:['\"]ext_def_file['\"]|ext_def_file)\s*:\s*"
    r"(?:['\"]([^'\"]+)['\"]|(\S+))\s*(?:#.*)?$",
    re.MULTILINE,
)
_EXT_DEF_FILE_FLOW_RE = re.compile(
    r"(?:['\"]ext_def_file['\"]|ext_def_file)\s*:\s*"
    r"(?:['\"]([^'\"]+)['\"]|([^\s,#}]+))"
)
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
    "sharded_state": (("model-rank-*-part-*.safetensors",),),
    "dummy": (),
    "extensible": (("*.safetensors",), ("*.bin",), ("*.pt",)),
}
# Keep in lockstep with ShardedStateLoader.DEFAULT_PATTERN.
_SHARDED_STATE_DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
_SHARDED_STATE_FIELD_RE = re.compile(r"\{(?:rank|part)(?::[^}]*)?\}")


def cache_layout_signature(
    layout: Any,
    *,
    cache_dtype: str,
    checkpoint_granularities: tuple[tuple[str, int], ...] = (),
) -> str:
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
    identity = {"cache_dtype": str(cache_dtype), "groups": groups}
    if checkpoint_granularities:
        identity["checkpoint_granularities"] = sorted(checkpoint_granularities)
    payload = json.dumps(
        identity,
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
    model_loader_extra_config: dict,
    ext_yaml: str,
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
    JSON/weights. ``extensible`` also hashes ``--ext-yaml`` and the
    ``ext_def_file`` that ``ExtensibleLM`` imports — resolved with
    ``os.path.abspath`` relative to the process working directory, the
    same way the loader does — plus that module's transitive local
    helpers under the directory inserted into ``sys.path``, so a custom
    input processor cannot share a namespace with the same checkpoint.
    ``sharded_state`` fingerprints the files
    ``model_loader_extra_config["pattern"]`` selects
    (``ShardedStateLoader.DEFAULT_PATTERN`` when unset), not leftover
    default-named shards. ``npcache`` fingerprints ``np/weight_names.json``
    and the listed NumPy files when that cache exists, because the loader
    then skips the ``*.bin`` bytes. Hugging Face hub ids still prefer the
    loaded config commit, then a cached snapshot directory, then a
    pinned ``--revision``. The returned id always includes the
    normalized load format so two deployments that share a directory
    (or commit) but select different ``.safetensors`` / ``.bin`` /
    ``.pt`` sets cannot restore each other's KV.
    """

    if not isinstance(model_loader_extra_config, dict):
        raise TypeError("model_loader_extra_config must be a dict")
    extra_json = json.dumps(
        model_loader_extra_config, sort_keys=True, separators=(",", ":")
    )
    fmt = _normalize_load_format(load_format)
    ext_digest = _extensible_fingerprint(ext_yaml, load_format=fmt)
    if os.path.isdir(model_path):
        snapshot = _snapshot_commit_hash(model_path)
        if snapshot is not None:
            return _checkpoint_id_with_load_format(
                snapshot,
                load_format=fmt,
                extra_config=model_loader_extra_config,
                ext_digest=ext_digest,
            )
        return _checkpoint_id_with_load_format(
            "local-" + _local_checkpoint_fingerprint(model_path, fmt, extra_json),
            load_format=fmt,
            extra_config=model_loader_extra_config,
            ext_digest=ext_digest,
        )
    commit = getattr(hf_config, "_commit_hash", None)
    if isinstance(commit, str) and _HF_COMMIT_HASH_RE.fullmatch(commit):
        return _checkpoint_id_with_load_format(
            commit,
            load_format=fmt,
            extra_config=model_loader_extra_config,
            ext_digest=ext_digest,
        )
    model_dir = _resolved_model_dir(model_path, revision=revision)
    if model_dir is not None:
        snapshot = _snapshot_commit_hash(model_dir)
        if snapshot is not None:
            return _checkpoint_id_with_load_format(
                snapshot,
                load_format=fmt,
                extra_config=model_loader_extra_config,
                ext_digest=ext_digest,
            )
        return _checkpoint_id_with_load_format(
            "local-" + _local_checkpoint_fingerprint(model_dir, fmt, extra_json),
            load_format=fmt,
            extra_config=model_loader_extra_config,
            ext_digest=ext_digest,
        )
    if isinstance(revision, str) and _HF_COMMIT_HASH_RE.fullmatch(revision):
        return _checkpoint_id_with_load_format(
            revision,
            load_format=fmt,
            extra_config=model_loader_extra_config,
            ext_digest=ext_digest,
        )
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


def _ext_def_path_from_match(match: re.Match[str]) -> str | None:
    path = None
    for group in match.groups():
        if group:
            path = group
            break
    if not path or path in ("|", ">", "null", "~", "{}", "[]"):
        return None
    return path


def _blank_nested_flow_maps(text: str) -> str:
    """Keep only the outermost ``{...}`` mapping so nested keys cannot match."""

    chars: list[str] = []
    depth = 0
    quote = ""
    escape = False
    for char in text:
        if quote:
            chars.append(" " if depth > 1 else char)
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
            chars.append(" " if depth > 1 else char)
            continue
        if char == "{":
            depth += 1
            chars.append(char if depth == 1 else " ")
            continue
        if char == "}":
            chars.append(char if depth == 1 else " ")
            if depth > 0:
                depth -= 1
            continue
        chars.append(" " if depth > 1 else char)
    return "".join(chars)


def _yaml_mapping_body(yaml_text: str) -> str:
    body = yaml_text.lstrip("\ufeff").lstrip()
    if body.startswith("---"):
        body = body[3:].lstrip()
    return body


def _ext_def_file_from_yaml_text(yaml_text: str) -> str | None:
    """Return the top-level ``ext_def_file`` path from an extensible yaml.

    The L3 namespace must be the same whether or not PyYAML is installed
    (Mooncake L3 CI does not ship it). ``ExtensibleModelLoader`` reads
    this as a top-level mapping key via ``yaml.safe_load``, including
    quoted keys, spaces around ``:``, and a document-level flow mapping.
    Nested ``ext_def_file`` keys and indented block entries are ignored.
    The yaml bytes are hashed in full, so nested processor config still
    rotates the id.
    """

    body = _yaml_mapping_body(yaml_text)
    if body.startswith("{"):
        match = _EXT_DEF_FILE_FLOW_RE.search(_blank_nested_flow_maps(body))
        if match is None:
            return None
        return _ext_def_path_from_match(match)
    match = _EXT_DEF_FILE_BLOCK_RE.search(yaml_text)
    if match is None:
        return None
    return _ext_def_path_from_match(match)


def _contained_in_dir(path: str, root: str) -> bool:
    path_abs = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    if path_abs == root_abs:
        return True
    try:
        return os.path.commonpath([path_abs, root_abs]) == root_abs
    except ValueError:
        return False


def _extension_module_files(module_name: str, *, ext_def_dir: str) -> tuple[str, ...]:
    """Return local files ExtensibleLM could load for ``module_name``.

    Prefixes are included because ``import pkg.sub`` loads ``pkg`` then
    ``pkg.sub``. Only paths that stay under the inserted ``sys.path``
    directory are returned.
    """

    found: list[str] = []
    parts = module_name.split(".")
    for index in range(len(parts)):
        prefix = parts[: index + 1]
        base = os.path.join(ext_def_dir, *prefix)
        for candidate in (f"{base}.py", os.path.join(base, "__init__.py")):
            if not os.path.isfile(candidate):
                continue
            if not _contained_in_dir(candidate, ext_def_dir):
                continue
            found.append(candidate)
    return tuple(found)


def _ast_local_extension_imports(
    source: str, *, current_file: str, ext_def_dir: str
) -> tuple[str, ...]:
    """Return local files imported by ``source`` under ``ext_def_dir``."""

    try:
        tree = ast.parse(source, filename=current_file)
    except SyntaxError:
        return ()
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.extend(
                    _extension_module_files(alias.name, ext_def_dir=ext_def_dir)
                )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        names = tuple(alias.name for alias in node.names)
        if node.level:
            # ``from . import helper`` is level 1: the current package is
            # ``dirname(file)``. Ascend ``level - 1`` parents, matching
            # ``__package__.rsplit(".", level - 1)``, not ``level``.
            package_dir = os.path.dirname(os.path.abspath(current_file))
            for _ in range(node.level - 1):
                package_dir = os.path.dirname(package_dir)
            if not _contained_in_dir(package_dir, ext_def_dir):
                continue
            module_parts = () if node.module is None else tuple(node.module.split("."))
            search_root = (
                os.path.join(package_dir, *module_parts)
                if module_parts
                else package_dir
            )
            if module_parts:
                for index in range(len(module_parts)):
                    prefix = module_parts[: index + 1]
                    base = os.path.join(package_dir, *prefix)
                    for candidate in (
                        f"{base}.py",
                        os.path.join(base, "__init__.py"),
                    ):
                        if os.path.isfile(candidate) and _contained_in_dir(
                            candidate, ext_def_dir
                        ):
                            found.append(candidate)
            for name in names:
                if name == "*":
                    continue
                base = os.path.join(search_root, name)
                for candidate in (f"{base}.py", os.path.join(base, "__init__.py")):
                    if os.path.isfile(candidate) and _contained_in_dir(
                        candidate, ext_def_dir
                    ):
                        found.append(candidate)
            continue
        if node.module is None:
            continue
        found.extend(_extension_module_files(node.module, ext_def_dir=ext_def_dir))
        for name in names:
            if name == "*":
                continue
            found.extend(
                _extension_module_files(
                    f"{node.module}.{name}", ext_def_dir=ext_def_dir
                )
            )
    return tuple(found)


def _extension_imported_code_files(entrypoint: str) -> tuple[tuple[str, str], ...]:
    """Return ``(relative posix path, path)`` for the imported extension.

    ``ExtensibleLM`` inserts ``dirname(abspath(ext_def_file))`` into
    ``sys.path`` and imports the file's stem. Sibling modules and
    packages loaded from that directory can change embeddings and KV, so
    the L3 identity follows those local imports, including package-relative
    ``from . import helper`` (resolved in the current package, not its
    parent). Stdlib and site
    packages are omitted because they do not live under that directory.
    Directory and file symlinks are followed the same way Python imports
    them; lexical containment keeps ``../`` relative imports from
    escaping the extension directory.
    """

    entry = os.path.abspath(entrypoint)
    ext_def_dir = os.path.dirname(entry)
    pending = [entry]
    seen_real: set[str] = set()
    found: list[tuple[str, str]] = []
    while pending:
        path = pending.pop()
        real = os.path.realpath(path)
        if real in seen_real:
            continue
        if not os.path.isfile(path):
            continue
        if not _contained_in_dir(path, ext_def_dir):
            continue
        seen_real.add(real)
        rel = os.path.relpath(path, ext_def_dir)
        rel_posix = "/".join(rel.split(os.sep))
        found.append((rel_posix, path))
        try:
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
        except (OSError, UnicodeDecodeError):
            continue
        pending.extend(
            _ast_local_extension_imports(
                source, current_file=path, ext_def_dir=ext_def_dir
            )
        )
    found.sort(key=lambda item: item[0])
    return tuple(found)


def _extensible_fingerprint(ext_yaml: str, *, load_format: str) -> str:
    """Hash ``--ext-yaml`` and the extension module it loads, if any.

    ``ExtensibleLM`` reads both. A Hugging Face snapshot commit does not
    cover those files, so they must enter the checkpoint id before the
    snapshot shortcut returns. Relative ``ext_def_file`` values are
    resolved with ``os.path.abspath`` against the process working
    directory, matching the loader, not the YAML directory. The digest
    includes that file and the local helpers it transitively imports
    from the inserted ``sys.path`` directory. Non-extensible loaders
    must pass an empty path. Parsing does not import PyYAML: two hosts
    that share the yaml and extension code must produce the same digest
    even when only one has the yaml package. Top-level ``ext_def_file``
    is recognized with the same quoted-key, spaced-colon, and flow-mapping
    forms ``yaml.safe_load`` accepts.
    """

    if load_format != "extensible":
        return ""
    if not isinstance(ext_yaml, str) or not ext_yaml.strip():
        raise ValueError("extensible L3 ids require --ext-yaml")
    yaml_path = os.path.abspath(ext_yaml)
    hasher = hashlib.sha256()
    hasher.update(b"ext_yaml")
    with open(yaml_path, "rb") as handle:
        yaml_bytes = handle.read()
    hasher.update(yaml_bytes)
    try:
        yaml_text = yaml_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return hasher.hexdigest()
    ext_def = _ext_def_file_from_yaml_text(yaml_text)
    if ext_def is None:
        return hasher.hexdigest()
    def_path = os.path.abspath(ext_def)
    hasher.update(b"ext_def_file")
    hasher.update(os.path.basename(def_path).encode())
    if not os.path.isfile(def_path):
        raise FileNotFoundError(def_path)
    for rel, path in _extension_imported_code_files(def_path):
        hasher.update(rel.encode())
        _update_file_digest(hasher, path)
    return hasher.hexdigest()


def _checkpoint_id_with_load_format(
    identity: str, *, load_format: str, extra_config: dict, ext_digest: str
) -> str:
    base = f"{identity}:{load_format}"
    if extra_config:
        payload = json.dumps(extra_config, sort_keys=True, separators=(",", ":"))
        base = f"{base}:{hashlib.sha256(payload.encode()).hexdigest()}"
    if ext_digest:
        return f"{base}:ext-{ext_digest}"
    return base


def _selected_weight_names(names: Sequence[str], *, load_format: str) -> frozenset[str]:
    """Return the weight files ``--load-format`` would load from ``names``.

    Pattern groups match ``DefaultModelLoader._prepare_weights``: the first
    group that matches any file wins, so ``auto`` hashes ``*.safetensors``
    when those exist and does not mix in leftover ``*.bin`` / ``*.pt``.
    ``sharded_state`` hashes the files the configured shard pattern
    selects (``model-rank-*-part-*.safetensors`` when
    ``model_loader_extra_config`` omits ``pattern``). Each rank
    fingerprints the files it can read; replica gather combines those
    digests. Unknown loaders raise rather than hashing metadata alone.
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
    changing KV computation. Directory symlinks are followed the same way
    Python imports them: two checkpoints that link ``model_helpers`` at
    different package trees must not share a fingerprint. Real-path
    identity skips cycles and the filesystem root. ``__pycache__`` and
    hidden directories are skipped; bytecode and VCS metadata are not
    part of the loaded model.
    """

    found: list[tuple[str, str]] = []
    seen_real_dirs: set[str] = set()
    filesystem_root = os.path.abspath(os.sep)
    for dirpath, dirnames, filenames in os.walk(
        model_dir, topdown=True, onerror=None, followlinks=True
    ):
        real_dir = os.path.realpath(dirpath)
        if real_dir == filesystem_root or real_dir in seen_real_dirs:
            dirnames[:] = []
            continue
        seen_real_dirs.add(real_dir)
        keep: list[str] = []
        for name in sorted(
            entry
            for entry in dirnames
            if entry != "__pycache__" and not entry.startswith(".")
        ):
            child_real = os.path.realpath(os.path.join(dirpath, name))
            if child_real == filesystem_root or child_real in seen_real_dirs:
                continue
            keep.append(name)
        dirnames[:] = keep
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
def _local_checkpoint_fingerprint(
    model_dir: str, load_format: str, extra_config_json: str
) -> str:
    """Hash config/index/quant-config bytes and the selected weight files.

    Cached by directory path, load format, and loader extra-config so a
    process that resolves the same local checkpoint more than once (target
    plus draft, or a repeated prefix rebuild) does not re-read every shard.
    ``hf_quant_config.json`` is hashed with ``config.json``: ModelOpt
    mixed-precision maps, group sizes, and KV quantization live there,
    not in the weight tensors. ``consolidated.safetensors.index.json`` is
    hashed so two Mistral dumps with the same ``consolidated*.safetensors``
    candidates but different shard maps cannot share a namespace.
    Local ``*.py`` is hashed, including files imported from package
    subdirectories and from directory symlinks Python would follow on
    import, so two trees with identical JSON/weights but
    different ``--trust-remote-code`` configuration modules cannot share
    a namespace. ``--load-format`` selects so a directory that contains
    more than one checkpoint encoding cannot share a namespace across
    loaders. ``sharded_state`` uses ``model_loader_extra_config["pattern"]``
    when set. ``npcache`` hashes the NumPy cache when
    ``np/weight_names.json`` exists, because that is what the loader
    reads.
    """
    hasher = hashlib.sha256()
    try:
        names = tuple(sorted(os.listdir(model_dir)))
    except OSError:
        return hasher.hexdigest()
    extra_config = json.loads(extra_config_json)
    if not isinstance(extra_config, dict):
        raise TypeError("model_loader_extra_config must be a dict")
    selected_weights, extra_weight_files = _selected_weight_targets(
        model_dir, names=names, load_format=load_format, extra_config=extra_config
    )
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
    for rel, path in extra_weight_files:
        hasher.update(rel.encode())
        _update_file_digest(hasher, path)
    return hasher.hexdigest()


def _sharded_state_pattern(extra_config: dict) -> str:
    pattern = extra_config.get("pattern")
    if pattern is None:
        return _SHARDED_STATE_DEFAULT_PATTERN
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("sharded_state pattern must be a non-empty str")
    return pattern


def _sharded_state_match_glob(pattern: str) -> str:
    return _SHARDED_STATE_FIELD_RE.sub("*", pattern)


def _sharded_state_weight_files(
    model_dir: str, *, pattern: str
) -> tuple[tuple[str, str], ...]:
    match_glob = _sharded_state_match_glob(pattern)
    matched = glob.glob(os.path.join(model_dir, match_glob))
    found: list[tuple[str, str]] = []
    for path in sorted(matched):
        if not os.path.isfile(path):
            continue
        rel = os.path.relpath(path, model_dir).replace(os.sep, "/")
        found.append((rel, path))
    return tuple(found)


def _npcache_weight_files(model_dir: str) -> tuple[tuple[str, str], ...] | None:
    names_file = os.path.join(model_dir, "np", "weight_names.json")
    if not os.path.isfile(names_file):
        return None
    with open(names_file) as handle:
        weight_names = json.load(handle)
    if not isinstance(weight_names, list):
        raise ValueError("np/weight_names.json must be a JSON list")
    found: list[tuple[str, str]] = [("np/weight_names.json", names_file)]
    for name in weight_names:
        rel = "np/" + str(name)
        path = os.path.join(model_dir, "np", str(name))
        if os.path.isfile(path):
            found.append((rel, path))
    return tuple(found)


def _selected_weight_targets(
    model_dir: str,
    *,
    names: Sequence[str],
    load_format: str,
    extra_config: dict,
) -> tuple[frozenset[str], tuple[tuple[str, str], ...]]:
    """Return top-level weight names and extra (relative, path) weight files.

    Top-level names are hashed in directory order with metadata so default
    loaders keep a stable fingerprint. Nested npcache / sharded-state files
    are hashed after that pass.
    """

    if load_format == "npcache":
        np_files = _npcache_weight_files(model_dir)
        if np_files is not None:
            return frozenset(), np_files
        return _selected_weight_names(names, load_format=load_format), ()
    if load_format == "sharded_state":
        pattern = _sharded_state_pattern(extra_config)
        shard_files = _sharded_state_weight_files(model_dir, pattern=pattern)
        top_level: list[str] = []
        nested: list[tuple[str, str]] = []
        for rel, path in shard_files:
            if "/" in rel:
                nested.append((rel, path))
            else:
                top_level.append(rel)
        return frozenset(top_level), tuple(nested)
    return _selected_weight_names(names, load_format=load_format), ()


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


# Bump when a runtime change alters produced KV for the same checkpoint,
# packed Host layout, and listed options (built-in model code, RoPE,
# cache-producing kernels). Unrelated rolling upgrades must keep this
# value so they still share L3. This is not a git SHA or package version:
# those would split the store on every non-cache commit while staying
# 0.1.0 across cache-affecting edits.
# Isolates this Host-storage contract from earlier producer implementations.
L3_RUNTIME_COMPAT = "2"


def storage_deployment_id(
    model_name: str,
    *,
    revision: str,
    model_overrides: dict,
    cache_signature: str,
    pipeline_rank: int,
    attn_tp_size: int,
    group_shard_counts: Sequence[int],
    draft_model: str,
    draft_revision: str,
    cache_quantization: str,
    runtime_compat: str,
    attention_backend: str,
    draft_attention_backend: str,
    skip_softmax_threshold: float,
    eagle3_layers_to_capture: Sequence[int],
    kv_producer: dict[str, Any],
) -> str:
    """Snapshot static cache compatibility facts into an immutable identity.

    All inputs are explicit: loaded checkpoints and overrides, packed Host
    layout and quantization, pipeline stage, attention-TP width, resolved
    target/draft attention backends, numerical/partition producer options,
    skip-softmax threshold, EAGLE3 capture layers and runtime compatibility.
    The packed-cache signature already includes the target compute dtype;
    producer options include draft dtype when a draft pool is present.

    The returned digest retains no mutable inputs. Weight generations are
    added separately, in the engine host's namespace (``ts-l3``'s
    ``key_prefix``). Empty draft facts mean no
    draft pool. ``attn_tp_size`` distinguishes head assignments that can
    otherwise have identical local layouts under replicated KV heads.
    ``group_shard_counts`` isolates the per-group replica numbering used by
    page-cyclic objects. An all-replicated layout keeps its existing identity.
    """
    if not isinstance(model_overrides, dict):
        raise TypeError("model_overrides must be a dict")
    if not isinstance(kv_producer, dict):
        raise TypeError("kv_producer must be a dict")
    if not kv_producer:
        raise ValueError("kv_producer must be explicit and nonempty")
    if isinstance(eagle3_layers_to_capture, (str, bytes)):
        raise TypeError("eagle3_layers_to_capture must be a sequence of ints")
    if isinstance(group_shard_counts, (str, bytes)):
        raise TypeError("group_shard_counts must be a sequence of ints")
    shards = [int(count) for count in group_shard_counts]
    if not shards or any(count < 1 for count in shards):
        raise ValueError("group_shard_counts must contain positive shard counts")
    payload = json.dumps(
        {
            "model": str(model_name),
            "revision": str(revision),
            "model_overrides": model_overrides,
            "cache_signature": str(cache_signature),
            "pipeline_rank": int(pipeline_rank),
            "attn_tp_size": int(attn_tp_size),
            "cp_size": 1,
            "draft_model": str(draft_model),
            "draft_revision": str(draft_revision),
            "cache_quantization": str(cache_quantization),
            "runtime_compat": str(runtime_compat),
            "kv_producer": kv_producer,
            **(
                {"group_shard_counts": shards}
                if any(count > 1 for count in shards)
                else {}
            ),
            "attention_backend": str(attention_backend),
            "draft_attention_backend": str(draft_attention_backend),
            "skip_softmax_threshold": float(skip_softmax_threshold),
            "eagle3_layers_to_capture": [
                int(layer) for layer in eagle3_layers_to_capture
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
