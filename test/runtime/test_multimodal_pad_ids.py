import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest
import torch

from tokenspeed.runtime.multimodal.embedder import pad_input_tokens
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    is_mm_pad_value,
    is_mm_pad_value_for,
    resolve_mm_pad_substitute_ids,
    substitute_mm_pad_,
)


def _item(modality: Modality, content_hash: int) -> MultimodalDataItem:
    item = MultimodalDataItem(modality=modality, hash=content_hash)
    item.set_pad_value()
    return item


def test_content_pad_ids_preserve_modality_inside_int32_range():
    items = [_item(modality, -1) for modality in Modality]
    values = torch.tensor([item.pad_value for item in items], dtype=torch.int64)

    assert len(set(values.tolist())) == len(Modality)
    assert bool(is_mm_pad_value(values).all())
    assert int(values.max()) < 2**31
    # The execution buffer uses int32, including the top of the audio range.
    assert torch.tensor(values.tolist(), dtype=torch.int32).tolist() == values.tolist()
    for index, modality in enumerate(Modality):
        expected = [False] * len(Modality)
        expected[index] = True
        assert is_mm_pad_value_for(values, modality).tolist() == expected


def test_substitution_restores_each_modality_token_in_place():
    image = _item(Modality.IMAGE, 1)
    audio = _item(Modality.AUDIO, 2)
    input_ids = torch.tensor(
        [7, image.pad_value, audio.pad_value, 8], dtype=torch.int32
    )

    output = substitute_mm_pad_(
        input_ids,
        {Modality.IMAGE: 200005, Modality.AUDIO: 200023},
    )

    assert output is input_ids
    assert input_ids.tolist() == [7, 200005, 200023, 8]


def test_resolve_mtp_tokens_supports_specific_and_shared_model_configs():
    inkling = SimpleNamespace(
        image_placeholder_token_id=200005,
        audio_placeholder_token_id=200023,
    )
    assert resolve_mm_pad_substitute_ids(inkling) == {
        Modality.IMAGE: 200005,
        Modality.AUDIO: 200023,
    }

    shared = SimpleNamespace(media_placeholder_token_id=163605)
    assert resolve_mm_pad_substitute_ids(shared) == {
        modality: 163605 for modality in Modality
    }

    explicit_zero = SimpleNamespace(image_token_id=0, media_placeholder_token_id=9)
    assert resolve_mm_pad_substitute_ids(explicit_zero)[Modality.IMAGE] == 0

    qwen_omni = SimpleNamespace(
        thinker_config=SimpleNamespace(
            image_token_id=151_655,
            video_token_id=151_656,
            audio_token_id=151_676,
        )
    )
    assert resolve_mm_pad_substitute_ids(qwen_omni) == {
        Modality.IMAGE: 151_655,
        Modality.VIDEO: 151_656,
        Modality.AUDIO: 151_676,
    }

    glm53_flash = SimpleNamespace(
        model_type="glm53_flash",
        image_token_id=154_854,
        video_token_id=154_855,
    )
    assert resolve_mm_pad_substitute_ids(glm53_flash) == {
        Modality.IMAGE: 154_854,
        Modality.VIDEO: 154_854,
    }

    nested_explicit_beats_outer_transport_placeholder = SimpleNamespace(
        image_placeholder_token_id=9,
        thinker_config=SimpleNamespace(image_token_id=10),
    )
    assert (
        resolve_mm_pad_substitute_ids(
            nested_explicit_beats_outer_transport_placeholder
        )[Modality.IMAGE]
        == 10
    )


@pytest.mark.parametrize("modality", [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO])
@pytest.mark.parametrize("token_id", [None, 0])
def test_padding_uses_placeholder_ids(modality, token_id):
    item = MultimodalDataItem(modality=modality, pad_value=123, offsets=[(1, 3)])
    inputs = MultimodalInputs(
        mm_items=[item], im_token_id=token_id, video_token_id=token_id
    )
    tokens = [9, 0, 7, 0, 8]
    expected = [9, 123, 7, 123, 8]
    if token_id is None or modality == Modality.AUDIO:
        expected[2] = 123
    assert pad_input_tokens(tokens, inputs) == expected
    assert tokens == [9, 0, 7, 0, 8]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
