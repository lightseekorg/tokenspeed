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

"""Load independent WeightStore publications through TokenSpeed's native loader.

This module is owned by TokenSpeed and imported lazily inside its scheduler
through the native model-update plugin hook. The native weight_update_session
owns finalization; the native control plane owns distributed commit. This
adapter never changes physical parameter layout.
"""

from __future__ import annotations

import base64
import json
import os
import zlib
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from weight_store.adapters import TargetAdapter
from weight_store.engine_adapters._hf_tensor_catalog import HFTensorCatalog
from weight_store.engine_adapters._mtflash import (
    MTFLASH_HF_TYPES,
    canonical_transfer_dtype,
    natural_hf_name_key,
)
from weight_store.publication import PublishedSnapshot
from weight_store.tensor_region import TensorRegion


def decompress_snapshot(envelope: dict[str, Any]) -> dict[str, Any]:
    """Restore the full directory and reject a different enclosed version."""
    if set(envelope) != {"version", "zlib_base64"}:
        raise ValueError("TokenSpeed requires a compressed publication directory")
    content = zlib.decompress(base64.b64decode(envelope["zlib_base64"], validate=True))
    snapshot = json.loads(content)
    if snapshot["version"] != envelope["version"]:
        raise ValueError("TokenSpeed compressed publication version mismatch")
    return snapshot


class EngineType(Enum):
    """The engine member required by the existing TokenSpeed SDK hook."""

    TOKENSPEED = "tokenspeed"


@dataclass
class FluentLlmModelUpdateInitConfig:
    """Translate the frozen hook's configuration shape to independent WeightStore.

    This name is prescribed by TokenSpeed's existing plugin protocol. The
    adapter does not import or construct the legacy updater SDK.
    """

    weight_store_config: dict[str, Any]
    local_host: str
    hf_type: str
    hf_safetensors_path: str

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> FluentLlmModelUpdateInitConfig:
        """Preserve the receiver configuration for the hook's store argument."""
        return cls(
            config,
            config["local_host"],
            config["hf_type"],
            config["hf_safetensors_path"],
        )


@dataclass
class FluentLlmEngineConfig:
    """Checkpoint fields passed by the existing native SDK constructor."""

    hf_type: str
    hf_safetensors_path: str


@dataclass
class ModelUpdaterConfig:
    """Reader identity supplied by TokenSpeed for each scheduler process."""

    role: str
    reader_rank: int
    engine_type: EngineType
    engine_config: FluentLlmEngineConfig


@dataclass
class MooncakeWeightStore:
    """Carry configuration through the frozen SDK hook without the old SDK."""

    config: dict[str, Any]
    local_host: str


class TokenSpeedTargetAdapter(TargetAdapter):
    """Declare logical HF tensors; leave TP/EP slicing to the live native model."""

    def __init__(self, *, hf_type: str, hf_safetensors_path: str | Path) -> None:
        if hf_type not in MTFLASH_HF_TYPES:
            raise ValueError(f"TokenSpeed WeightStore does not support {hf_type!r}")
        self.catalog = HFTensorCatalog.from_safetensors_checkpoint(hf_safetensors_path)

    def _regions(self, models: list[torch.nn.Module]) -> tuple[TensorRegion, ...]:
        """Use the target inventory independently of the publication directory."""
        if not isinstance(models, list) or len(models) != 1:
            raise ValueError(
                "TokenSpeed WeightStore requires one target with draft retain"
            )
        model = models[0]
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TokenSpeed target must be a torch module")
        if model.quant_config is not None:
            raise ValueError("Quantized TokenSpeed updates are not supported")
        names = sorted(
            (
                name
                for name in model.checkpoint_layout.iter_source_names()
                if model.checkpoint_weight_name_filter(name)
                and not name.endswith(".rotary_emb.inv_freq")
            ),
            key=natural_hf_name_key,
        )
        if not names:
            raise ValueError("TokenSpeed target requested no checkpoint weights")
        regions = []
        for name in names:
            shape = self.catalog.shape(name)
            dtype = canonical_transfer_dtype(name, self.catalog.dtype(name))
            if not model.checkpoint_layout.spec(name).accepts(shape, dtype):
                raise ValueError(f"TokenSpeed checkpoint schema mismatch: {name}")
            # Source transposition belongs to PublishedSnapshot, not the target.
            regions.append(TensorRegion.full(name, shape, dtype, is_transposed=None))
        return tuple(regions)

    def required_regions(
        self, model: list[torch.nn.Module]
    ) -> tuple[TensorRegion, ...]:
        """Return every required logical tensor, including redundant expert slots."""
        return self._regions(model)

    def load_weights(
        self,
        model: list[torch.nn.Module],
        inputs: Iterable[tuple[TensorRegion, torch.Tensor]],
    ) -> tuple[torch.device, ...]:
        """Consume the complete ordered stream inside the native update session."""
        regions = self._regions(model)
        root = model[0]
        if not root._weight_update_active:
            raise RuntimeError(
                "TokenSpeed must open weight_update_session before loading"
            )
        iterator = iter(inputs)
        consumed = 0

        def weights() -> Iterator[tuple[str, torch.Tensor]]:
            nonlocal consumed
            for expected in regions:
                try:
                    actual, tensor = next(iterator)
                except StopIteration as error:
                    raise ValueError("Incomplete TokenSpeed weight stream") from error
                if (
                    actual != expected
                    or tuple(tensor.shape) != expected.global_shape
                    or tensor.dtype != expected.dtype
                ):
                    raise ValueError(
                        f"Invalid TokenSpeed tensor: {expected.tensor_name}"
                    )
                consumed += 1
                yield expected.tensor_name, tensor

        loaded = root.load_weights(weights())
        if consumed != len(regions) or not loaded:
            raise RuntimeError(
                "TokenSpeed loader did not write the complete weight stream"
            )
        if next(iterator, None) is not None:
            raise ValueError("Unexpected trailing TokenSpeed tensor")
        return tuple(dict.fromkeys(parameter.device for parameter in root.parameters()))


class TokenSpeedWeightStoreUpdater:
    """Load a validated publication on the Store device and its current stream."""

    def __init__(
        self,
        store: Any,
        adapter: TokenSpeedTargetAdapter,
        *,
        instance_id: str,
        device: torch.device,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.instance_id = instance_id
        self.device = device

    def update_weights(
        self, models: list[torch.nn.Module], *, version: int, snapshot: dict[str, Any]
    ) -> Any:
        """Restore compressed metadata and load the complete requested publication."""
        publication = PublishedSnapshot.from_dict(decompress_snapshot(snapshot))
        if type(version) is not int or version < 0 or publication.version != version:
            raise ValueError(
                "TokenSpeed snapshot version differs from the requested version"
            )
        return self.store.update_weights(
            models,
            self.adapter,
            snapshot=publication,
            instance_id=self.instance_id,
            stream=(
                torch.cuda.current_stream(self.device)
                if self.device.type == "cuda"
                else None
            ),
        )


def _make_weight_store_updater(
    config: dict[str, Any], *, reader_rank: int
) -> TokenSpeedWeightStoreUpdater:
    """Attach a native scheduler to the configured Hub weights Store."""
    from data_dist_hub import DataDistHub, HubConfig
    from weight_store.engine_adapters.runtime_config import WeightStoreModelUpdateConfig
    from weight_store.mooncake_store import MooncakeKVStore
    from weight_store.streaming_types import IOConfig
    from weight_store.weight_store import WeightStore

    settings = WeightStoreModelUpdateConfig.from_dict(config)
    if not settings.instance_id or not 0 <= reader_rank < settings.target_world_size:
        raise ValueError(
            "TokenSpeed requires a valid instance identity and reader rank"
        )
    store_config = deepcopy(settings.store)
    store_config.setdefault("client", {}).update(global_segment_size=0)
    hub = DataDistHub.get()
    hub.init(
        HubConfig.from_dict(
            {
                "stores": [store_config],
                "local_ip": settings.local_host,
                "worker_name": f"tokenspeed-{settings.instance_id}-{reader_rank}-{os.getpid()}",
            }
        )
    )
    device = torch.device("cuda", torch.cuda.current_device())
    store = WeightStore(
        MooncakeKVStore(hub.get_store(store_config["name"])),
        rank=reader_rank,
        io_config=IOConfig(device, settings.buffer_capacity_bytes),
        transfer_chunk_bytes=settings.transfer_chunk_bytes,
    )
    adapter = TokenSpeedTargetAdapter(
        hf_type=settings.hf_type,
        hf_safetensors_path=settings.hf_safetensors_path,
    )
    return TokenSpeedWeightStoreUpdater(
        store, adapter, instance_id=settings.instance_id, device=device
    )


def make_model_updater(
    config: ModelUpdaterConfig, *, weight_store: MooncakeWeightStore
) -> TokenSpeedWeightStoreUpdater:
    """Implement TokenSpeed's existing SDK factory with independent WeightStore."""
    return _make_weight_store_updater(
        weight_store.config, reader_rank=config.reader_rank
    )
