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

"""CPU integration through native precision loading and generic image scatter."""

from collections import defaultdict
from test.runtime.models.test_inkling2_model import (
    _checkpoint_weights,
    _mapping,
    _tiny_config,
    _tiny_quant_config,
)
from test.runtime.models.test_inkling2_vision import _config as _vision_config
from test.runtime.test_inkling2_quantization import _group

import pytest
import torch

from tokenspeed.runtime.configs.inkling2 import Inkling2MMConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.quantization.inkling2 import Inkling2QuantConfig
from tokenspeed.runtime.models.inkling2 import Inkling2ForConditionalGeneration
from tokenspeed.runtime.models.inkling2_vision import (
    Inkling2VisionEncoder,
    varlen_encoder_attention,
)
from tokenspeed.runtime.multimodal.embedder import warmup_multimodal_encoders
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
    MultimodalInputs,
)


def _config() -> Inkling2MMConfig:
    config = _tiny_config()
    config.vision_config = _vision_config()
    return config


def _quant(config: Inkling2MMConfig, vision_kind: str) -> Inkling2QuantConfig:
    by_kind = defaultdict(list)
    for name, precision in _tiny_quant_config().tensor_precisions.items():
        if name != "model.llm.unembed.weight":
            by_kind[precision.compute_dtype].append(name)
    groups = [_group(kind, names) for kind, names in by_kind.items()]
    head = _group("bfloat16", ["model.llm.unembed.weight"])
    head["fwd_output_dtype"] = "float32"
    groups.append(head)
    with torch.device("meta"):
        visual = Inkling2VisionEncoder(
            config.vision_config,
            config.text_config.hidden_size,
            mm_attention_backend=None,
        )
    groups.append(
        _group(
            vision_kind,
            ["model.visual." + name for name, _ in visual.named_parameters()],
        )
    )
    return Inkling2QuantConfig({"model": {"parameter_groups": groups}})


def _model(active: bool) -> Inkling2ForConditionalGeneration:
    config = _config()
    model = Inkling2ForConditionalGeneration(
        config=config,
        mapping=_mapping(),
        quant_config=_quant(config, "bfloat16"),
        is_multimodal_active=active,
        mm_attention_backend=None,
    )
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_()
        model.model.embed_norm.weight.fill_(3)
        if model.visual is not None:
            for name, parameter in model.visual.named_parameters():
                if name.endswith("gamma"):
                    parameter.fill_(0.1)
                elif "ln_" in name and name.endswith("weight"):
                    parameter.fill_(1)
                else:
                    parameter.normal_(mean=0, std=0.1)
    return model


def _item(height: int, width: int, start: int) -> MultimodalDataItem:
    count = height * width
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=height * 100 + width,
        offsets=[(start, start + count - 1)],
        feature=torch.randn(count, 2, 4, 4, 3, dtype=torch.bfloat16),
        model_specific_data={
            "vision_grid_thw": torch.tensor([[1, height, width]], dtype=torch.int64)
        },
    )


def _context(mode: ForwardMode, tokens: int) -> ForwardContext:
    extends = int(mode == ForwardMode.EXTEND)
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=extends,
        input_num_tokens=tokens,
        forward_mode=mode,
        output_layout=ForwardOutputLayout(extends, extends, 1 - extends, 1),
    )


def _mm(
    items: list[MultimodalDataItem], prefix: int, count: int
) -> MultimodalForwardContext:
    return MultimodalForwardContext(
        mm_inputs=[MultimodalInputs(mm_items=items)],
        extend_prefix_lens=[prefix],
        extend_seq_lens=[count],
        max_encoder_tokens=32,
    )


def _weights(model: Inkling2ForConditionalGeneration) -> list[tuple[str, torch.Tensor]]:
    return [
        ("model." + name if name.startswith("visual.") else name, value)
        for name, value in _checkpoint_weights(model)
    ]


def test_native_vision_features_preserve_multiple_image_order() -> None:
    model = _model(True)
    items = [_item(1, 1, 1), _item(1, 2, 3)]
    expected = model.visual(
        torch.cat([item.feature for item in items]),
        torch.cat([item.vision_grid_thw for item in items]),
    )
    actual = model.get_image_feature(items)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (3, 32)
    assert actual.dtype == torch.bfloat16


def test_one_item_can_concatenate_separate_image_grids() -> None:
    model = _model(True)
    first, second = _item(1, 1, 1), _item(1, 2, 3)
    expected = model.get_image_feature([first, second])
    combined = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(1, 1), (3, 4)],
        feature=torch.cat([first.feature, second.feature]),
        model_specific_data={
            "vision_grid_thw": torch.cat(
                [first.vision_grid_thw, second.vision_grid_thw]
            )
        },
    )
    torch.testing.assert_close(
        model.get_image_feature([combined]), expected, rtol=0, atol=0
    )


def test_vision_uses_existing_item_dp_and_rejects_weight_tp() -> None:
    config = _config()
    quant = _quant(config, "bfloat16")
    for vision_tp, vision_dp in ((1, 2), (2, 1)):
        mapping = Mapping(
            rank=0,
            world_size=2,
            attn_tp_size=2,
            attn_dp_size=1,
            dense_tp_size=2,
            dense_dp_size=1,
            moe_tp_size=1,
            moe_ep_size=2,
            moe_dp_size=1,
            vision_tp_size=vision_tp,
            vision_dp_size=vision_dp,
            nprocs_per_node=2,
            nnodes=1,
        )
        if vision_tp == 2:
            with pytest.raises(NotImplementedError, match="mm-encoder-tp-mode data"):
                Inkling2ForConditionalGeneration(
                    config=config,
                    mapping=mapping,
                    quant_config=quant,
                    is_multimodal_active=True,
                    mm_attention_backend=None,
                )
        else:
            with torch.device("meta"):
                model = Inkling2ForConditionalGeneration(
                    config=config,
                    mapping=mapping,
                    quant_config=quant,
                    is_multimodal_active=True,
                    mm_attention_backend=None,
                )
            assert model.vision_embedder.has_encoder_dp


def test_image_scatter_norms_text_once_and_preserves_header_position() -> None:
    model = _model(True)
    item = _item(1, 2, 2)
    expected_image = model.get_image_feature([item])
    # Admission rewrites only image spans; the header stays a text token.
    ids = torch.tensor([1, 5, 10000, 10000, 3])
    context = _context(ForwardMode.EXTEND, 5)
    actual = model.prepare_model_kwargs(
        context, ids, {"multimodal_context": _mm([item], 0, 5)}
    )["input_embeds"]
    expected_text = model.get_decoder_input_embedding()(ids[[0, 1, 4]])
    torch.testing.assert_close(actual[[0, 1, 4]], expected_text, rtol=0, atol=0)
    torch.testing.assert_close(actual[2:4], expected_image, rtol=0, atol=0)
    assert not torch.equal(expected_image, model.model.embed_norm(expected_image))


def test_chunked_prefill_reuses_then_releases_generic_item_encoding() -> None:
    model = _model(True)
    item = _item(1, 2, 1)
    expected = model.get_image_feature([item])
    first = model.multimodal_input_embeds(
        torch.tensor([1, 5]), _context(ForwardMode.EXTEND, 2), _mm([item], 0, 2)
    )
    torch.testing.assert_close(first[1:], expected[:1], rtol=0, atol=0)
    assert item.feature is None and item.encoded is not None
    second = model.multimodal_input_embeds(
        torch.tensor([5, 3]), _context(ForwardMode.EXTEND, 2), _mm([item], 2, 2)
    )
    torch.testing.assert_close(second[:1], expected[1:], rtol=0, atol=0)
    assert item.feature is None and item.encoded is None
    assert (
        model.multimodal_input_embeds(
            torch.tensor([4]), _context(ForwardMode.EXTEND, 1), _mm([item], 4, 1)
        )
        is None
    )


def test_preencoded_image_rows_use_the_same_scatter_path() -> None:
    model = _model(True)
    item = _item(1, 1, 1)
    item.feature = None
    item.encoded = torch.full((1, 32), 7, dtype=torch.bfloat16)
    actual = model.multimodal_input_embeds(
        torch.tensor([1, 5]), _context(ForwardMode.EXTEND, 2), _mm([item], 0, 2)
    )
    torch.testing.assert_close(
        actual[1], torch.full((32,), 7, dtype=torch.bfloat16), rtol=0, atol=0
    )


def test_image_inputs_reject_client_embeddings_but_text_inputs_preserve_them() -> None:
    model = _model(True)
    ids = torch.tensor([1, 5])
    supplied = torch.randn(2, 32)
    assert (
        model.prepare_model_kwargs(None, ids, {"input_embeds": supplied})[
            "input_embeds"
        ]
        is supplied
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        model.prepare_model_kwargs(
            _context(ForwardMode.EXTEND, 2),
            ids,
            {
                "input_embeds": supplied,
                "multimodal_context": _mm([_item(1, 1, 1)], 0, 2),
            },
        )


def test_decode_does_not_encode_images_again() -> None:
    model = _model(True)
    assert (
        model.multimodal_input_embeds(
            torch.tensor([1]),
            _context(ForwardMode.DECODE, 1),
            _mm([_item(1, 1, 1)], 0, 1),
        )
        is None
    )


@pytest.mark.parametrize("native_patch_size", [2, 20])
def test_registered_warmup_uses_native_geometry(native_patch_size: int) -> None:
    config = _config()
    config.vision_config.native_patch_size = native_patch_size
    model = Inkling2ForConditionalGeneration(
        config=config,
        mapping=_mapping(),
        quant_config=_quant(config, "bfloat16"),
        is_multimodal_active=True,
        mm_attention_backend="triton_attn",
    )
    with torch.no_grad():
        for parameter in model.visual.parameters():
            parameter.fill_(0.1)
    spec = model.get_multimodal_encoder_specs()[Modality.IMAGE]
    items = spec.make_warmup_items()
    assert items[0].feature.shape == (
        1,
        2,
        native_patch_size * 2,
        native_patch_size * 2,
        3,
    )
    assert items[0].vision_grid_thw.tolist() == [[1, 1, 1]]
    output = spec.fn(items)
    assert output.shape == (1, 32) and torch.isfinite(output).all()
    warmup_multimodal_encoders(model, device="cpu")


def test_complete_loader_includes_native_vision_tensors() -> None:
    model = _model(True)
    weights = _weights(model)
    loaded = model.load_weights(reversed(weights))
    assert loaded == set(dict(model.named_parameters()))
    for name, parameter in model.visual.named_parameters():
        torch.testing.assert_close(
            parameter, torch.ones_like(parameter), rtol=0, atol=0
        )


def test_inactive_target_skips_vision_weights_and_rejects_image_execution() -> None:
    weights = _weights(_model(True))
    inactive = _model(False)
    loaded = inactive.load_weights(weights)
    assert loaded == set(dict(inactive.named_parameters()))
    assert inactive.get_multimodal_encoder_specs() == {}
    with pytest.raises(ValueError, match="active vision"):
        inactive.multimodal_input_embeds(
            torch.tensor([1]),
            _context(ForwardMode.EXTEND, 1),
            _mm([_item(1, 1, 0)], 0, 1),
        )


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "unknown", "dtype", "shape"]
)
def test_native_vision_loader_rejects_invalid_weights(failure: str) -> None:
    model = _model(True)
    weights = _weights(model)
    index = next(
        i
        for i, (name, _) in enumerate(weights)
        if name == "model.visual.proj_in.weight"
    )
    if failure == "missing":
        weights.pop(index)
    elif failure == "duplicate":
        weights.append(weights[index])
    elif failure == "unknown":
        weights.append(("model.visual.proj_mid.weight", weights[index][1]))
    elif failure == "dtype":
        weights[index] = (weights[index][0], weights[index][1].float())
    else:
        weights[index] = (weights[index][0], weights[index][1][:1])
    with pytest.raises(
        ValueError,
        match={
            "missing": "Missing",
            "duplicate": "Repeated",
            "unknown": "No checkpoint precision declared",
            "dtype": "expected.*got",
            "shape": "shape mismatch",
        }[failure],
    ):
        model.load_weights(weights)


@pytest.mark.parametrize(
    "failure",
    [
        "audio",
        "video",
        "grid_name",
        "grid_dtype",
        "grid_device",
        "grid_shape",
        "temporal",
        "zero_grid",
        "offset_count",
        "offset_length",
        "offset_negative",
        "dtype",
        "patch_count",
        "patch_shape",
        "encoded_shape",
    ],
)
def test_image_items_reject_invalid_contracts(failure: str) -> None:
    model = _model(True)
    item = _item(1, 2, 1)
    grid = item.vision_grid_thw
    if failure in ("audio", "video"):
        item.modality = Modality.AUDIO if failure == "audio" else Modality.VIDEO
    elif failure == "grid_name":
        item.model_specific_data = {"image_grid_thw": grid}
    elif failure == "grid_dtype":
        item.model_specific_data["vision_grid_thw"] = grid.float()
    elif failure == "grid_device":
        item.model_specific_data["vision_grid_thw"] = grid.to("meta")
    elif failure == "grid_shape":
        item.model_specific_data["vision_grid_thw"] = grid.flatten()
    elif failure == "temporal":
        grid[0, 0] = 2
    elif failure == "zero_grid":
        grid[0, 1] = 0
    elif failure == "offset_count":
        item.offsets = []
    elif failure == "offset_length":
        item.offsets = [(1, 3)]
    elif failure == "offset_negative":
        item.offsets = [(-1, 0)]
    elif failure == "dtype":
        item.feature = item.feature.float()
    elif failure == "patch_count":
        item.feature = item.feature[:1]
    elif failure == "patch_shape":
        item.feature = item.feature[:, :1]
    else:
        item.encoded = torch.empty(1, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        model.multimodal_input_embeds(
            torch.tensor([1, 10000, 10000]),
            _context(ForwardMode.EXTEND, 3),
            _mm([item], 0, 3),
        )


def test_item_validation_runs_before_generic_deduplication() -> None:
    model = _model(True)
    first, alias = _item(1, 1, 1), _item(1, 1, 2)
    alias.model_specific_data["vision_grid_thw"][0, 0] = 2
    with pytest.raises(ValueError, match="temporal size 1"):
        model.multimodal_input_embeds(
            torch.tensor([1, 10000, 10000]),
            _context(ForwardMode.EXTEND, 3),
            _mm([first, alias], 0, 3),
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("video", "only image"),
        ("grid_dtype", "CPU integer"),
        ("grid_device", "CPU integer"),
        ("grid_shape", "CPU integer"),
        ("temporal", "temporal size 1"),
        ("zero_grid", "positive spatial"),
        ("missing", "materialized patches"),
        ("dtype", "Image patches"),
        ("patch_count", "Image patches"),
        ("patch_shape", "Image patches"),
    ],
)
def test_encoder_entry_rejects_invalid_items(failure: str, message: str) -> None:
    # The EPD encode worker calls the registered encoder directly, without a prefill.
    model = _model(True)
    item = _item(1, 2, 1)
    grid = item.vision_grid_thw
    if failure == "video":
        item.modality = Modality.VIDEO
    elif failure == "grid_dtype":
        item.model_specific_data["vision_grid_thw"] = grid.float()
    elif failure == "grid_device":
        item.model_specific_data["vision_grid_thw"] = grid.to("meta")
    elif failure == "grid_shape":
        item.model_specific_data["vision_grid_thw"] = grid.flatten()
    elif failure == "temporal":
        grid[0, 0] = 2
    elif failure == "zero_grid":
        grid[0, 1] = 0
    elif failure == "missing":
        item.feature = None
    elif failure == "dtype":
        item.feature = item.feature.float()
    elif failure == "patch_count":
        item.feature = item.feature[:1]
    else:
        item.feature = item.feature[:, :1]
    encode = model.get_multimodal_encoder_specs()[Modality.IMAGE].fn
    with pytest.raises(ValueError, match=message):
        encode([item])


def test_image_ranges_cannot_overlap_across_items() -> None:
    model = _model(True)
    with pytest.raises(ValueError, match="overlap across items"):
        model.multimodal_input_embeds(
            torch.tensor([1, 10000, 10000]),
            _context(ForwardMode.EXTEND, 3),
            _mm([_item(1, 2, 1), _item(1, 1, 2)], 0, 3),
        )


def test_vision_precision_must_be_declared_as_bfloat16() -> None:
    config = _config()
    for quant in (_tiny_quant_config(), _quant(config, "float32")):
        with pytest.raises(ValueError, match="precision declared|expected.*got"):
            Inkling2ForConditionalGeneration(
                config=config,
                mapping=_mapping(),
                quant_config=quant,
                is_multimodal_active=True,
                mm_attention_backend=None,
            )


def test_vision_rejects_cudnn_metadata_contract() -> None:
    config = _config()
    with pytest.raises(NotImplementedError, match="cuDNN packed offsets"):
        Inkling2ForConditionalGeneration(
            config=config,
            mapping=_mapping(),
            quant_config=_quant(config, "bfloat16"),
            is_multimodal_active=True,
            mm_attention_backend="flashinfer_cudnn",
        )


def test_model_preserves_registered_vision_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.layers.attention import mm_encoder_attention

    def reference_backend(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        bsz: int,
        seq_len: int,
        softmax_scale: float,
        max_seqlen: int,
    ) -> torch.Tensor:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        assert bsz == len(lengths) and seq_len == max_seqlen == max(lengths)
        return varlen_encoder_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            lengths=lengths,
            softmax_scale=softmax_scale,
            backend=None,
        )

    monkeypatch.setitem(
        mm_encoder_attention._BACKENDS, "test_model_segments", reference_backend
    )
    reference = _model(True)
    config = _config()
    model = Inkling2ForConditionalGeneration(
        config=config,
        mapping=_mapping(),
        quant_config=_quant(config, "bfloat16"),
        is_multimodal_active=True,
        mm_attention_backend="test_model_segments",
    )
    model.load_state_dict(reference.state_dict(), strict=True)
    items = [_item(1, 1, 1), _item(2, 3, 3)]
    with torch.inference_mode():
        torch.testing.assert_close(
            model.get_image_feature(items),
            reference.get_image_feature(items),
            rtol=0,
            atol=0,
        )
