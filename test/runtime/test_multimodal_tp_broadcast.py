import torch

from tokenspeed.runtime.multimodal.embedder import VisionEmbedder
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


def _embedder(tp_size: int) -> VisionEmbedder:
    embedder = VisionEmbedder()
    embedder._vision_tp_group = tuple(range(tp_size))
    embedder._vision_tp_process_group = object()
    embedder._vision_tp_src_rank = 0
    return embedder


def test_tp2_broadcasts_at_128_mib():
    embedder = _embedder(2)

    assert not embedder._should_move_shm_via_tp_broadcast(
        [_shm_item(128 * 1024 * 1024 - 2)]
    )
    assert embedder._should_move_shm_via_tp_broadcast([_shm_item(128 * 1024 * 1024)])


def test_tp4_broadcasts_aggregate_64_mib_payload():
    embedder = _embedder(4)
    items = [_shm_item(8 * 1024 * 1024) for _ in range(8)]

    assert embedder._should_move_shm_via_tp_broadcast(items)


def test_tp_broadcast_requires_uniform_shm_dtype():
    embedder = _embedder(4)
    items = [
        _shm_item(32 * 1024 * 1024, torch.bfloat16),
        _shm_item(32 * 1024 * 1024, torch.float16),
    ]

    assert not embedder._should_move_shm_via_tp_broadcast(items)


def test_tp_broadcast_rejects_inline_tensors():
    embedder = _embedder(4)
    items = [
        _shm_item(64 * 1024 * 1024),
        MultimodalDataItem(modality=Modality.IMAGE, feature=torch.empty(1)),
    ]

    assert not embedder._should_move_shm_via_tp_broadcast(items)
