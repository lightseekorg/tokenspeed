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

"""Build a layer-truncated HY V4 checkpoint so two engines can be compared.

A full 78-layer HY V4 is ~1.5 TB in BF16, far too heavy to iterate on. This
module materialises a small checkpoint that is *bit-identical* to the released
weights on the layers it keeps, so TokenSpeed and vLLM load
exactly the same tensors and any per-layer divergence is a kernel/wiring bug
rather than a weight difference.

What gets copied: the embedding, the final norm, the LM head, the iHC head, the
first ``num_layers`` decoder layers, and the MTP block. Everything else is
dropped, and ``num_hidden_layers`` plus every per-layer list in the config is
rewritten to match, so an engine cannot tell the checkpoint was trimmed.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

HY_V4_REPO_ID = "tencent/Hy4-preview"
# Kept regardless of layer count: without these the model cannot produce logits.
_GLOBAL_PREFIXES = (
    "model.embed_tokens.",
    "model.norm.",
    "model.hc_head.",
    "lm_head.",
)
_MTP_PREFIX = "model.mtp_layers."
# Small files an engine may read; the tokenizer is deliberately not included
# because callers drive the engines with token ids (``--skip-tokenizer-init``).
_SIDE_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class Truncation:
    """How much of the released checkpoint to keep.

    Args:
        num_layers: Decoder layers to keep, taken from the front. Layer 0 is
            dense, layer 1 is the first MoE layer and layer 2 the first shared
            indexer, so 3 is the smallest size that covers every layer kind.
        num_experts: Routed experts to keep, taken from the front of the packed
            expert tensors. None keeps the released 256.
    """

    num_layers: int = 3
    num_experts: int | None = 8

    def describe(self) -> str:
        experts = "all" if self.num_experts is None else str(self.num_experts)
        return f"{self.num_layers}L/{experts}E"


def _config_dir(repo_id: str) -> Path:
    """Locate a local config directory for ``repo_id`` (works for both hub ids
    and already-downloaded paths)."""
    if os.path.isdir(repo_id):
        return Path(repo_id)
    return Path(hf_hub_download(repo_id, "config.json")).parent


def load_released_config(repo_id: str = HY_V4_REPO_ID) -> dict:
    """Read the released ``config.json`` as a dict, without transforming it."""
    return json.loads((_config_dir(repo_id) / "config.json").read_text())


def truncated_config(
    config: dict,
    truncation: Truncation,
) -> dict:
    """Return ``config`` rewritten for a truncated checkpoint.

    Every list whose length matches the released layer count is a per-layer
    vector, so it is sliced alongside ``num_hidden_layers``. Deriving that from
    the data rather than a hardcoded field list keeps the rewrite correct if the
    checkpoint grows another per-layer vector.
    """
    original_layers = config["num_hidden_layers"]
    if not 0 < truncation.num_layers <= original_layers:
        raise ValueError(
            f"num_layers must be in 1..{original_layers}, got {truncation.num_layers}"
        )
    updated = dict(config)
    updated["num_hidden_layers"] = truncation.num_layers
    for key, value in config.items():
        if isinstance(value, list) and len(value) == original_layers:
            updated[key] = value[: truncation.num_layers]
    if truncation.num_experts is not None:
        if not 0 < truncation.num_experts <= config["n_routed_experts"]:
            raise ValueError(
                f"num_experts must be in 1..{config['n_routed_experts']}, "
                f"got {truncation.num_experts}"
            )
        updated["n_routed_experts"] = truncation.num_experts
    return updated


def truncation_hf_overrides(truncation: Truncation) -> str:
    """The engine-side equivalent of :func:`truncated_config`.

    Returns a ``--hf-overrides`` JSON string so an engine can load the full
    checkpoint metadata while materializing only the truncated layers. Useful
    for a quick sanity run on ``load_format=dummy``.
    """
    original = load_released_config()
    truncated = truncated_config(original, truncation)
    changed = {
        key: value for key, value in truncated.items() if original.get(key) != value
    }
    return json.dumps(changed)


def _keep(key: str, truncation: Truncation) -> bool:
    if key.startswith(_MTP_PREFIX):
        # The MTP block ships as ``model.mtp_layers.<i>``; HY V4 has exactly one.
        return True
    if any(key.startswith(prefix) for prefix in _GLOBAL_PREFIXES):
        return True
    if not key.startswith("model.layers."):
        return False
    return int(key.split(".", 3)[2]) < truncation.num_layers


def _shrink_experts(key: str, tensor, num_experts: int | None, released_experts: int):
    """Slice the leading experts off a packed or per-expert routed tensor.

    HY V4 packs every routed expert into one ``experts.gate_up_proj`` /
    ``experts.down_proj`` tensor whose leading dimension indexes the expert, so
    keeping the first ``num_experts`` is a slice on dim 0. The router's own
    ``gate.weight`` (and its bias) are indexed by expert for the same reason.
    """
    if num_experts is None or num_experts >= released_experts:
        return tensor
    if ".mlp.experts.gate_up_proj" in key or ".mlp.experts.down_proj" in key:
        return tensor[:num_experts]
    if ".mlp.gate.weight" in key or ".mlp.gate.e_score_correction_bias" in key:
        return tensor[:num_experts]
    return tensor


def _needed_keys(weight_map: dict, truncation: Truncation) -> list[str]:
    keys = [key for key in weight_map if _keep(key, truncation)]
    if not keys:
        raise RuntimeError("no checkpoint tensors selected; check the layer pattern")
    return sorted(keys)


def build_truncated_checkpoint(
    output_dir: str | os.PathLike[str],
    truncation: Truncation = Truncation(),
    repo_id: str = HY_V4_REPO_ID,
    *,
    overwrite: bool = False,
    log: callable | None = None,
) -> Path:
    """Write a truncated HY V4 checkpoint to ``output_dir`` and return the path.

    Args:
        output_dir: Destination directory; created if missing.
        truncation: Which layers and experts to keep.
        repo_id: Source checkpoint, a hub id or a local snapshot directory.
        overwrite: Replace an existing output directory. Without it, only a
            complete export matching this request is reused; a nonempty
            destination that cannot be reused is rejected.
        log: Optional ``callable(str)`` used to report progress.

    Returns:
        The output directory, ready to pass as a model path to either engine.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    report = log or (lambda message: None)
    target = Path(output_dir)
    source = _config_dir(repo_id)
    source_path, target_path = source.resolve(), target.resolve()
    if target_path == source_path or target_path in source_path.parents:
        raise ValueError("truncated output must not contain the source checkpoint")
    if target.is_symlink():
        raise ValueError("truncated output must not be a symlink")
    if target.exists() and not target.is_dir():
        raise FileExistsError("truncated output must be a directory")
    released_config = load_released_config(repo_id)
    rewritten_config = truncated_config(released_config, truncation)
    marker = target / ".hy_v4_truncated.json"
    if not overwrite and target.exists():
        if marker.is_file():
            existing = json.loads(marker.read_text())
            complete = all(
                (target / name).is_file()
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "model.safetensors",
                )
            )
            if complete and existing == {
                "repo_id": repo_id,
                "truncation": truncation.describe(),
            }:
                report(f"reusing truncated checkpoint at {target}")
                return target
        # A mismatched or incomplete marker is not permission to overwrite.
        if any(target.iterdir()):
            raise FileExistsError(
                f"{target} is not a reusable export; choose a new output directory "
                "or explicitly request overwrite=True (--rebuild)"
            )
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        index_path = Path(hf_hub_download(repo_id, "model.safetensors.index.json"))
    weight_map = json.loads(index_path.read_text())["weight_map"]
    released_experts = released_config["n_routed_experts"]

    keys = _needed_keys(weight_map, truncation)
    shards = sorted({weight_map[key] for key in keys})
    report(
        f"truncating {repo_id} -> {target} "
        f"({truncation.describe()}, {len(keys)} tensors from {len(shards)} shards)"
    )

    tensors: dict[str, object] = {}
    for shard_index, shard in enumerate(shards, start=1):
        shard_path = source / shard
        if not shard_path.exists():
            shard_path = Path(hf_hub_download(repo_id, shard))
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key not in keys:
                    continue
                tensor = handle.get_tensor(key)
                tensors[key] = _shrink_experts(
                    key, tensor, truncation.num_experts, released_experts
                ).contiguous()
        report(f"  [{shard_index}/{len(shards)}] {shard}")

    missing = sorted(set(keys) - set(tensors))
    if missing:
        raise RuntimeError(
            f"{len(missing)} tensors were not found in the shards, e.g. {missing[:5]}"
        )

    # Validate and read all source tensors before replacing a previous export.
    if target.exists() and overwrite:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    # One shard keeps the comparison setup simple; these checkpoints are small.
    shard_name = "model.safetensors"
    save_file(dict(tensors), str(target / shard_name))
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() for t in tensors.values())},
                "weight_map": {key: shard_name for key in sorted(tensors)},
            },
            indent=2,
        )
    )
    (target / "config.json").write_text(json.dumps(rewritten_config, indent=2))
    for side_file in _SIDE_FILES:
        candidate = source / side_file
        if candidate.exists():
            shutil.copy2(candidate, target / side_file)
    marker.write_text(
        json.dumps({"repo_id": repo_id, "truncation": truncation.describe()}, indent=2)
    )
    report(f"wrote {len(tensors)} tensors to {target / shard_name}")
    return target


def iter_layer_keys(keys: Iterable[str]) -> list[str]:
    """Sort dump keys by pass then layer so a report reads in execution order."""

    def sort_key(key: str) -> tuple:
        pass_id, _, rest = key.partition("_")
        layer = 0
        if rest.startswith("layer"):
            layer = int(rest[len("layer") :].split("_", 1)[0])
        order = {"in": 0, "out": 1}
        stage = "in" if rest.endswith("_in") else "out"
        return (int(pass_id.lstrip("p")), layer, order[stage], rest)

    return sorted(keys, key=sort_key)
