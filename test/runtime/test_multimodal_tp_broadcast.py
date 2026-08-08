import pytest
import torch

from tokenspeed.runtime.multimodal.encoder_feature_transport import (
    EncoderFeatureTransport,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle


def _shm_item(nbytes: int, dtype: torch.dtype = torch.bfloat16):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=ShmTensorHandle(
            shm_name="unused",
            shape=(nbytes // dtype.itemsize,),
            dtype=dtype,
        ),
    )


def _transport(tp_size: int) -> EncoderFeatureTransport:
    transport = EncoderFeatureTransport()
    transport._vision_tp_group = tuple(range(tp_size))
    transport._vision_tp_process_group = object()
    transport._vision_tp_src_rank = 0
    return transport


def test_tp_broadcast_selection():
    tp2 = _transport(2)
    assert not tp2._should_use_tp_broadcast([_shm_item(128 * 1024 * 1024 - 2)])
    assert tp2._should_use_tp_broadcast([_shm_item(128 * 1024 * 1024)])

    tp4 = _transport(4)
    assert tp4._should_use_tp_broadcast([_shm_item(8 * 1024 * 1024) for _ in range(8)])
    assert not tp4._should_use_tp_broadcast(
        [
            _shm_item(32 * 1024 * 1024, torch.bfloat16),
            _shm_item(32 * 1024 * 1024, torch.float16),
        ]
    )
    assert not tp4._should_use_tp_broadcast(
        [
            _shm_item(64 * 1024 * 1024),
            MultimodalDataItem(modality=Modality.IMAGE, feature=torch.empty(1)),
        ]
    )


def test_tp_broadcast_stays_on_model_stream(monkeypatch):
    transport = _transport(2)
    items = [_shm_item(128 * 1024 * 1024)]
    moved = []

    monkeypatch.setattr(
        transport,
        "_execute_tp_broadcasts",
        lambda batches, device, *, mode: moved.append((batches, device, mode)),
    )
    monkeypatch.setattr(
        transport,
        "_h2d_stream_on",
        lambda _device: pytest.fail("TP broadcast must not use the H2D stream"),
    )

    device = torch.device("cuda")
    transport.move_to_device(items, device)

    assert len(moved) == 1
    batches, moved_device, mode = moved[0]
    assert moved_device == device
    assert mode == "local"
    assert [entry.item for entry in batches[0].entries] == items
