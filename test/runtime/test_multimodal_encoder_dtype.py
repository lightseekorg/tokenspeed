from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.model_runner import infer_multimodal_encoder_dtype


class _VisionTower(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1, dtype=dtype))

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype


def test_infers_direct_visual_dtype():
    model = SimpleNamespace(visual=_VisionTower(torch.bfloat16))

    assert infer_multimodal_encoder_dtype(model) == "bfloat16"


def test_infers_nested_vision_tower_dtype():
    model = SimpleNamespace(
        model=SimpleNamespace(vision_tower=_VisionTower(torch.float16))
    )

    assert infer_multimodal_encoder_dtype(model) == "float16"


def test_returns_none_without_vision_tower():
    assert infer_multimodal_encoder_dtype(SimpleNamespace()) is None
