import pytest
import torch

from tokenspeed.runtime.multimodal.embedder import (
    EncodePlan,
    EncoderSpec,
    ScatterRange,
    VisionEmbedder,
    _coalesced_scatter_ranges,
    _merged_visual_ranges,
    _text_token_spans,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
    MultimodalInputs,
)


def test_visual_ranges_are_coalesced_for_embedding_assembly():
    item = MultimodalDataItem(modality=Modality.VIDEO)
    other = MultimodalDataItem(modality=Modality.VIDEO)
    scatter_ranges = [
        ScatterRange(1, 2, item, 0, 1),
        ScatterRange(3, 5, item, 2, 4),
        ScatterRange(7, 8, other, 0, 1),
    ]

    coalesced = _coalesced_scatter_ranges(scatter_ranges)
    visual_ranges = _merged_visual_ranges(coalesced, total_rows=10)

    assert coalesced == [
        ScatterRange(1, 5, item, 0, 4),
        ScatterRange(7, 8, other, 0, 1),
    ]
    assert visual_ranges == [(1, 6), (7, 9)]
    assert _text_token_spans(10, visual_ranges) == [(0, 1), (6, 7), (9, 10)]


def test_embedding_assembly_clamps_placeholder_ids_before_visual_overwrite():
    embedding = torch.nn.Embedding(10, 3)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(30, dtype=torch.float32).reshape(10, 3))
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        encoded=torch.tensor(
            [[100.0, 101.0, 102.0], [110.0, 111.0, 112.0]],
            dtype=torch.float32,
        ),
    )
    plan = EncodePlan(
        scatter_ranges=[ScatterRange(2, 3, item, 0, 1)],
    )
    input_ids = torch.tensor([1, 2, 999, 999, 3], dtype=torch.long)

    input_embeds, kwargs = VisionEmbedder()._assemble(
        input_ids=input_ids,
        text_embedding=embedding,
        plan=plan,
        encoders={Modality.IMAGE: EncoderSpec(lambda _: torch.empty(0))},
        multimodal_model=None,
    )

    expected = embedding(input_ids.clamp(max=embedding.num_embeddings - 1))
    expected[2:4] = item.encoded
    assert kwargs == {}
    assert torch.equal(input_embeds, expected)


@pytest.mark.parametrize("modality", [Modality.IMAGE, Modality.VIDEO])
def test_deepstack_combined_contract_for_chunked_prefill(modality):
    embedding = torch.nn.Embedding(10, 3, dtype=torch.bfloat16)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(30, dtype=torch.bfloat16).reshape(10, 3))

    main = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    deepstack = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    combined = torch.cat([main, deepstack], dim=1)
    input_ids = torch.tensor([999, 1, 2, 999, 999], dtype=torch.long)

    class Model:
        deepstack_visual_indexes = [1, 2]

        @staticmethod
        def separate_deepstack_embeds(output):
            return output[:, :3], output[:, 3:]

    def run():
        item = MultimodalDataItem(
            modality=modality,
            offsets=[(2, 3), (6, 7)],
            feature=torch.empty(0),
        )
        context = MultimodalForwardContext(
            mm_inputs=[MultimodalInputs(mm_items=[item])],
            extend_prefix_lens=[3],
            extend_seq_lens=[5],
        )
        return VisionEmbedder().apply(
            input_ids=input_ids,
            text_embedding=embedding,
            ctx=context,
            encoders={modality: EncoderSpec(lambda _items: combined, deepstack=True)},
            multimodal_model=Model(),
        )

    input_embeds, kwargs = run()

    expected = embedding(input_ids.clamp(max=embedding.num_embeddings - 1))
    expected[0] = main[1]
    expected[3:] = main[2:]
    assert torch.equal(input_embeds, expected)
    assert torch.equal(kwargs["input_deepstack_embeds"][0], deepstack[1])
    assert torch.equal(kwargs["input_deepstack_embeds"][3:], deepstack[2:])


def test_dense_deepstack_reuses_full_visual_tensor():
    item = MultimodalDataItem(
        modality=Modality.VIDEO,
        encoded=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        encoded_deepstack=torch.arange(6, 12, dtype=torch.float32).reshape(2, 3),
    )
    plan = EncodePlan(scatter_ranges=[ScatterRange(0, 1, item, 0, 1)])

    class Model:
        deepstack_visual_indexes = [8]

    _, kwargs = VisionEmbedder()._assemble(
        input_ids=torch.tensor([999, 999], dtype=torch.long),
        text_embedding=torch.nn.Embedding(10, 3),
        plan=plan,
        encoders={
            Modality.VIDEO: EncoderSpec(lambda _: torch.empty(0), deepstack=True)
        },
        multimodal_model=Model(),
    )

    deepstack = kwargs["input_deepstack_embeds"]
    assert torch.equal(deepstack, item.encoded_deepstack)
    assert (
        deepstack.untyped_storage().data_ptr()
        == item.encoded_deepstack.untyped_storage().data_ptr()
    )


def test_encoder_output_must_match_planned_tokens():
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(0, 0)])
    plan = EncodePlan()
    plan.misses_by_modality[Modality.IMAGE] = [item]
    output = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    with pytest.raises(RuntimeError, match="does not match planned tokens"):
        VisionEmbedder()._encode(
            plan=plan,
            encoders={Modality.IMAGE: EncoderSpec(lambda _: output)},
            multimodal_model=None,
            device=torch.device("cpu"),
        )
