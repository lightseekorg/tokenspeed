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

"""Check the snapshot SDK boundary without starting a device or remote store."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("weight_store")

import base64
import json
import zlib

from safetensors.torch import save_file
from weight_store.adapters import SimpleExportBinding
from weight_store.kv_store import KVStore
from weight_store.metadata import (
    PublisherMetadata,
    StoredObject,
    build_weight_region_key,
)
from weight_store.planner import build_write_objects
from weight_store.publication import PublishedSnapshot
from weight_store.streaming_types import IOConfig
from weight_store.tensor_region import TensorRegion
from weight_store.weight_store import WeightStore


def compress_snapshot(snapshot):
    """Build the metadata envelope sent by a control-plane client."""
    content = json.dumps(snapshot, separators=(",", ":")).encode("utf-8")
    return {
        "version": snapshot["version"],
        "zlib_base64": base64.b64encode(zlib.compress(content)).decode("ascii"),
    }


from tokenspeed.runtime.execution.weight_store import (
    EngineType,
    FluentLlmEngineConfig,
    ModelUpdaterConfig,
    MooncakeWeightStore,
    TokenSpeedTargetAdapter,
    TokenSpeedWeightStoreUpdater,
    decompress_snapshot,
    make_model_updater,
)


@pytest.fixture
def publication():
    region = TensorRegion.full("weight", (2,), torch.float32, is_transposed=False)
    key = build_weight_region_key(3, region, publisher_rank=0)
    return PublishedSnapshot(
        3, (PublisherMetadata(3, 0, (StoredObject(key, region),)),)
    )


@pytest.mark.parametrize("device_name", ["cpu", "cuda:2"])
def test_snapshot_receiver_uses_its_device_current_stream(
    monkeypatch, publication, device_name
):
    device = torch.device(device_name)
    stream = object()
    current_stream = Mock(return_value=stream)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    store = SimpleNamespace(update_weights=Mock(return_value="loaded"))
    adapter, models = object(), [object()]
    updater = TokenSpeedWeightStoreUpdater(
        store, adapter, instance_id="decode-0", device=device
    )

    assert (
        updater.update_weights(
            models, version=3, snapshot=compress_snapshot(publication.to_dict())
        )
        == "loaded"
    )
    store.update_weights.assert_called_once_with(
        models,
        adapter,
        snapshot=publication,
        instance_id="decode-0",
        stream=stream if device.type == "cuda" else None,
    )
    if device.type == "cuda":
        current_stream.assert_called_once_with(device)
    else:
        current_stream.assert_not_called()


def test_snapshot_receiver_rejects_another_version_before_reading(publication):
    store = SimpleNamespace(update_weights=Mock())
    updater = TokenSpeedWeightStoreUpdater(
        store, object(), instance_id="prefill-0", device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="snapshot version differs"):
        updater.update_weights(
            [], version=4, snapshot=compress_snapshot(publication.to_dict())
        )
    store.update_weights.assert_not_called()


def test_snapshot_receiver_propagates_store_failure(publication):
    store = SimpleNamespace(
        update_weights=Mock(side_effect=RuntimeError("read failed"))
    )
    updater = TokenSpeedWeightStoreUpdater(
        store, object(), instance_id="prefill-0", device=torch.device("cpu")
    )
    with pytest.raises(RuntimeError, match="read failed"):
        updater.update_weights(
            [], version=3, snapshot=compress_snapshot(publication.to_dict())
        )


class MemoryWeightStore(KVStore):
    def __init__(self):
        self.objects = {}
        self.registered = set()
        self.reads = []

    def _check_buffer(self, buffer):
        assert any(
            base.address <= buffer.address
            and buffer.address + buffer.size <= base.address + base.size
            for base in self.registered
        ), "I/O must use registered memory"

    def seed_write_objects(self, requests):
        # BufferedWriter 仅支持 CUDA；CPU 夹具只物化其约定的紧凑 payload，不模拟 DMA。
        for request in requests:
            assert request.source.device.type == "cpu"
            payload = request.source.contiguous()
            self.objects[request.key] = ctypes.string_at(
                payload.data_ptr(), payload.nbytes
            )

    def batch_put(self, requests):
        for request in requests:
            self._check_buffer(request.source)
            self.objects[request.key] = ctypes.string_at(
                request.source.address, request.source.size
            )

    def batch_get(self, requests):
        for request in requests:
            self._check_buffer(request.destination)
            value = self.objects[request.key]
            stop = request.offset + request.destination.size
            assert stop <= len(value), "read exceeds object"
            ctypes.memmove(
                request.destination.address,
                value[request.offset : stop],
                request.destination.size,
            )
            self.reads.append((request.key, request.offset, request.destination.size))

    def batch_delete(self, keys):
        for key in keys:
            self.objects.pop(key, None)

    def register_buffer(self, buffer):
        assert buffer not in self.registered
        self.registered.add(buffer)

    def unregister_buffer(self, buffer):
        self.registered.remove(buffer)


MATRIX = "model.layers.0.self_attn.0.o_proj.weight"
EXPERT = "model.layers.0.mlp.experts.3.up_proj.weight"


class NativeTarget(torch.nn.Module):
    """Replace unavailable GPU-native TP/EP loading with visible CPU writes."""

    def __init__(self, tensors):
        super().__init__()
        self.quant_config = None
        self._weight_update_active = True
        self.shapes = {name: value.shape for name, value in tensors.items()}
        self.checkpoint_layout = SimpleNamespace(
            iter_source_names=self.iter_source_names, spec=self.spec
        )
        self.received = {}
        self.local = torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.bfloat16))

    def iter_source_names(self):
        return self.shapes.keys()

    def checkpoint_weight_name_filter(self, name):
        return name in self.shapes

    def spec(self, name):
        return SimpleNamespace(accepts=lambda shape, dtype: shape == self.shapes[name])

    def load_weights(self, weights):
        for name, value in weights:
            self.received[name] = value.clone()
            if name == MATRIX:
                self.local.data.copy_(value[2:])
        return set(self.received)


def publish_directory(kv, tensors, version, *, omit=None):
    publishers = []
    for rank in range(2):
        bindings = []
        for name, value in tensors.items():
            if name == omit:
                continue
            local = value.chunk(2, dim=0)[rank]
            region = TensorRegion(
                name,
                tuple(value.shape),
                (rank * local.size(0), 0),
                tuple(local.shape),
                value.dtype,
                is_transposed=name == MATRIX,
            )
            bindings.append(
                SimpleExportBinding(region, (rank,), lambda local=local: local)
            )
        metadata, requests = build_write_objects(
            bindings, version=version, publisher_rank=rank, max_object_bytes=16
        )
        kv.seed_write_objects(requests)
        publishers.append(metadata)
    return PublishedSnapshot(version, tuple(publishers)).to_dict()


@pytest.fixture
def target(tmp_path):
    tensors = {
        MATRIX: torch.arange(16, dtype=torch.float32).reshape(4, 4),
        EXPERT: torch.arange(8, dtype=torch.float32).reshape(2, 4),
    }
    save_file(
        {name: value.bfloat16() for name, value in tensors.items()},
        str(tmp_path / "model.safetensors"),
    )
    model = NativeTarget(tensors)
    adapter = TokenSpeedTargetAdapter(
        hf_type="meituan_sc_moe_mla_oe_dsa", hf_safetensors_path=tmp_path
    )
    return tensors, model, adapter


def test_tokenspeed_loads_sharded_transposed_snapshots_and_updated_content(target):
    tensors, model, adapter = target
    kv = MemoryWeightStore()
    store = WeightStore(
        kv, rank=0, io_config=IOConfig("cpu", 16), transfer_chunk_bytes=8
    )
    updater = TokenSpeedWeightStoreUpdater(
        store, adapter, instance_id="decode-0", device=torch.device("cpu")
    )
    try:
        for version in range(3):
            values = {name: value + version for name, value in tensors.items()}
            snapshot = publish_directory(kv, values, version)
            result = updater.update_weights(
                [model], version=version, snapshot=compress_snapshot(snapshot)
            )
            assert result.version == version
            assert set(model.received) == set(tensors)
            for name, value in values.items():
                assert torch.equal(model.received[name], value.bfloat16())
            assert torch.equal(model.local, values[MATRIX][2:].bfloat16())
        assert all(
            region.is_transposed is None for region in adapter.required_regions([model])
        )
    finally:
        store.close()
    assert not kv.registered


def test_tokenspeed_missing_published_tensor_cannot_shrink_target_inventory(target):
    tensors, model, adapter = target
    kv = MemoryWeightStore()
    store = WeightStore(kv, rank=0, io_config=IOConfig("cpu", 16))
    updater = TokenSpeedWeightStoreUpdater(
        store, adapter, instance_id="prefill-0", device=torch.device("cpu")
    )
    try:
        with pytest.raises(
            (ValueError, AssertionError), match="cover|missing|source|tensor|region"
        ):
            updater.update_weights(
                [model],
                version=0,
                snapshot=compress_snapshot(
                    publish_directory(kv, tensors, 0, omit=EXPERT)
                ),
            )
        assert not model.received
    finally:
        store.close()


def test_tokenspeed_rejects_snapshot_version_mismatch_before_reading(target):
    tensors, model, adapter = target
    kv = MemoryWeightStore()
    store = WeightStore(kv, rank=0, io_config=IOConfig("cpu", 16))
    updater = TokenSpeedWeightStoreUpdater(
        store, adapter, instance_id="prefill-0", device=torch.device("cpu")
    )
    try:
        with pytest.raises(ValueError, match="version"):
            updater.update_weights(
                [model],
                version=1,
                snapshot=compress_snapshot(publish_directory(kv, tensors, 0)),
            )
        assert not kv.reads and not model.received
    finally:
        store.close()


def test_tokenspeed_requires_native_update_session(target):
    _, model, adapter = target
    model._weight_update_active = False
    with pytest.raises(RuntimeError, match="weight_update_session"):
        adapter.load_weights([model], [])


@pytest.mark.parametrize("count", [0, 2])
def test_tokenspeed_requires_exactly_one_target(target, count):
    _, model, adapter = target
    with pytest.raises(ValueError, match="one target with draft retain"):
        adapter.required_regions([model] * count)


def test_tokenspeed_rejects_a_quantized_target(target):
    _, model, adapter = target
    model.quant_config = object()
    with pytest.raises(ValueError, match="Quantized TokenSpeed"):
        adapter.required_regions([model])


def test_tokenspeed_rejects_tampered_compressed_version_before_reading(target):
    tensors, model, adapter = target
    kv = MemoryWeightStore()
    store = WeightStore(kv, rank=0, io_config=IOConfig("cpu", 16))
    updater = TokenSpeedWeightStoreUpdater(
        store, adapter, instance_id="prefill-0", device=torch.device("cpu")
    )
    envelope = compress_snapshot(publish_directory(kv, tensors, 0))
    envelope["version"] = 1
    try:
        with pytest.raises(ValueError, match="version"):
            updater.update_weights([model], version=1, snapshot=envelope)
        assert not kv.reads and not model.received
    finally:
        store.close()


def test_native_sdk_factory_uses_the_configured_store_identity_and_io_device(
    target, tmp_path, monkeypatch
):
    from data_dist_hub import DataDistHub

    class Hub:
        def __init__(self):
            self.config = None

        def init(self, config):
            self.config = config

        def get_store(self, name):
            assert name == "weights"
            return object()

    hub = Hub()
    monkeypatch.setattr(DataDistHub, "get", lambda: hub)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    settings = {
        "backend": "weight_store",
        "hf_type": "meituan_sc_moe_mla_oe_dsa",
        "hf_safetensors_path": str(tmp_path),
        "local_host": "127.0.0.1",
        "instance_id": "prefill-0",
        "target_world_size": 8,
        "buffer_capacity_bytes": 16,
        "transfer_chunk_bytes": 8,
        "store": {
            "name": "weights",
            "client": {"protocol": "rdma", "global_segment_size": 1024},
        },
    }
    config = ModelUpdaterConfig(
        "target",
        3,
        EngineType.TOKENSPEED,
        FluentLlmEngineConfig(settings["hf_type"], str(tmp_path)),
    )
    updater = make_model_updater(
        config, weight_store=MooncakeWeightStore(settings, "127.0.0.1")
    )
    try:
        assert isinstance(updater, TokenSpeedWeightStoreUpdater)
        assert updater.instance_id == "prefill-0"
        assert updater.device == torch.device("cuda:2")
        assert hub.config.stores[0].client["global_segment_size"] == 0
        assert settings["store"]["client"]["global_segment_size"] == 1024
        assert updater.adapter.required_regions([target[1]]) == target[
            2
        ].required_regions([target[1]])
    finally:
        updater.store.close()


def test_snapshot_envelope_preserves_integral_metadata():
    snapshot = {"version": 3, "publishers": [{"offset": (1 << 63) + 99}]}
    assert decompress_snapshot(compress_snapshot(snapshot)) == snapshot


def test_snapshot_envelope_rejects_uncompressed_metadata():
    with pytest.raises(ValueError, match="compressed publication directory"):
        decompress_snapshot({"version": 3, "publishers": []})
