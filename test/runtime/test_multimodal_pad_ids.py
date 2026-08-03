from types import SimpleNamespace

import torch

from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    is_mm_pad_value,
    is_mm_pad_value_for,
    maybe_substitute_mm_pad,
    resolve_mm_pad_substitute_ids,
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


def test_mtp_substitution_restores_each_modality_token():
    image = _item(Modality.IMAGE, 1)
    audio = _item(Modality.AUDIO, 2)
    input_ids = torch.tensor(
        [7, image.pad_value, audio.pad_value, 8], dtype=torch.int32
    )

    output = maybe_substitute_mm_pad(
        input_ids,
        {Modality.IMAGE: 200005, Modality.AUDIO: 200023},
    )

    assert output.tolist() == [7, 200005, 200023, 8]
    assert input_ids.tolist() == [7, image.pad_value, audio.pad_value, 8]


def test_scalar_mtp_substitution_remains_backward_compatible():
    image = _item(Modality.IMAGE, 3)
    audio = _item(Modality.AUDIO, 4)
    input_ids = torch.tensor([image.pad_value, audio.pad_value], dtype=torch.int32)

    assert maybe_substitute_mm_pad(input_ids, 42).tolist() == [42, 42]


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
