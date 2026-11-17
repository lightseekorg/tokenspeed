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

"""CPU model loading and FP32 output tests through real runtime classes."""

from test.runtime.models.test_inkling2_config import _checkpoint
from test.runtime.test_inkling2_quantization import _group

import pytest
import torch
import torch.nn.functional as F

from tokenspeed.runtime.configs.inkling2 import Inkling2MMConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.quantization.inkling2 import Inkling2QuantConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.models.inkling2 import (
    Inkling2ForConditionalGeneration,
    Inkling2LMHeadMethod,
    Inkling2LogitsProcessor,
)
from tokenspeed.runtime.models.inkling2_attention import FP32AccumulationLinearMethod


def _tiny_config() -> Inkling2MMConfig:
    values = _checkpoint()
    values["text_config"].update(
        hidden_size=32,
        num_hidden_layers=2,
        vocab_size=64,
        unpadded_vocab_size=60,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=2,
        rel_extent=8,
        local_layer_ids=[0],
        swa_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        sliding_window_size=8,
        n_layers_per_attn_res_block=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        intermediate_size=32,
    )
    values["vision_config"] = None
    values["mtp_config"] = None
    values["eos_token_id"] = 2
    return Inkling2MMConfig(**values)


def _mapping() -> Mapping:
    return Mapping(
        rank=0,
        world_size=1,
        attn_tp_size=1,
        attn_dp_size=1,
        dense_tp_size=1,
        dense_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        moe_dp_size=1,
        nprocs_per_node=1,
        nnodes=1,
    )


def _model() -> Inkling2ForConditionalGeneration:
    return Inkling2ForConditionalGeneration(
        config=_tiny_config(),
        mapping=_mapping(),
        quant_config=_tiny_quant_config(),
        is_multimodal_active=False,
        mm_attention_backend=None,
    )


def _tiny_quant_config() -> Inkling2QuantConfig:
    bf16 = [
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
    ]
    fp32 = ["model.llm.attn_res.attn_queries.0", "model.llm.attn_res.output_query"]
    groups = []
    for layer in range(2):
        prefix = f"model.llm.layers.{layer}"
        fp32.extend(
            [f"model.llm.attn_res.mlp_queries.{layer}", f"{prefix}.mlp.gate.bias"]
        )
        bf16.extend(
            f"{prefix}.attn.{projection}.weight"
            for projection in (
                "wq_du",
                "wk_dv",
                "wv_dv",
                "wr_du",
                "gate_attn",
                "attn_sink_proj",
                "wo_ud",
            )
        )
        bf16.extend(
            f"{prefix}.attn.{leaf}"
            for leaf in ("q_l2_clip.scale", "rel_l2_clip.scale", "rel_logits_proj.proj")
        )
        bf16.extend(
            f"{prefix}.{leaf}.weight"
            for leaf in ("mlp.gate", "attn_sconv", "mlp_sconv", "mlp_norm")
        )
        norms = (
            ("attn_norm",)
            if layer == 0
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        )
        bf16.extend(f"{prefix}.{norm}.weight" for norm in norms)
        for stream in ("k_sconv", "v_sconv"):
            leaves = ("weight",) if layer == 0 else ("static_w", "weight_proj")
            bf16.extend(f"{prefix}.attn.{stream}.{leaf}" for leaf in leaves)
        groups.append(
            _group(
                "nvfp4" if layer == 0 else "mxfp8",
                [f"{prefix}.mlp.experts.{leaf}_weight" for leaf in ("w13", "w2")],
            )
        )
    groups.extend([_group("bfloat16", bf16), _group("float32", fp32)])
    output = _group("bfloat16", ["model.llm.unembed.weight"])
    output["fwd_output_dtype"] = "float32"
    groups.append(output)
    return Inkling2QuantConfig({"model": {"parameter_groups": groups}})


def _head(config: Inkling2MMConfig) -> ParallelLMHead:
    head = ParallelLMHead(
        config.text_config.padded_vocab_size,
        config.text_config.hidden_size,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=None,
        prefix="model.llm.unembed",
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    )
    head.quant_method = Inkling2LMHeadMethod()
    return head


def test_attention_projections_preserve_fp32_results_until_bf16_store() -> None:
    model = _model()
    for layer in model.model.layers:
        for projection in (layer.attn.qkvr, layer.attn.wo_ud):
            assert isinstance(projection.quant_method, FP32AccumulationLinearMethod)
            assert projection.weight.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_full_layer_projects_adjacent_separate_sources_like_one_buffer() -> None:
    attn = _model().model.layers[1].attn.cuda()
    assert not attn.is_local
    torch.manual_seed(37)
    attn.qkvr.weight.data.normal_()
    rows = torch.randn((3, 8, 32), dtype=torch.bfloat16, device="cuda")
    # Three allocations at adjacent addresses, as separately normalized sources can be.
    separate = tuple(torch.from_dlpack(source) for source in rows.unbind(0))
    expected = attn._project_full(rows.unbind(0))
    for want, got in zip(expected, attn._project_full(separate), strict=True):
        for a, b in zip(want, got, strict=True):
            assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("buffer_rows", [1, 4])
def test_full_layer_projects_one_token_column_slices_like_separate_sources(
    buffer_rows: int,
) -> None:
    attn = _model().model.layers[1].attn.cuda()
    torch.manual_seed(41)
    attn.qkvr.weight.data.normal_()
    hidden = attn.qkvr.weight.shape[1]
    wide = torch.randn((buffer_rows, 3 * hidden), dtype=torch.bfloat16, device="cuda")
    # Adjacent one-token views count as contiguous although their row stride is 3 * hidden.
    sources = tuple(wide[:1, i * hidden : (i + 1) * hidden] for i in range(3))
    expected = attn._project_full(tuple(source.clone() for source in sources))
    for want, got in zip(expected, attn._project_full(sources), strict=True):
        for a, b in zip(want, got, strict=True):
            assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_full_layer_projects_its_conv_taps_from_the_k_and_v_sources() -> None:
    """The projection returns each source row's tap projection after the
    checkpoint's segments: K taps from the K source, V taps from the V one."""
    attn = _model().model.layers[1].attn.cuda()
    torch.manual_seed(43)
    for weight in (
        attn.qkvr.weight,
        attn.k_sconv.weight_proj,
        attn.v_sconv.weight_proj,
    ):
        weight.data.normal_()
    attn.derive_projected_taps()
    sources = tuple(
        torch.randn((8, 32), dtype=torch.bfloat16, device="cuda") for _ in range(3)
    )
    projections = attn._project_full(sources)
    for part, conv, slot in ((1, attn.k_sconv, 6), (2, attn.v_sconv, 7)):
        want = sources[part].float() @ conv.weight_proj.float().T
        torch.testing.assert_close(
            projections[part][slot].float(), want, atol=0.02, rtol=2**-7
        )


def test_logits_width_matches_sampling_vocabulary_not_stored_head_rows() -> None:
    values = _checkpoint()
    values["text_config"].update(
        hidden_size=32, vocab_size=1056, unpadded_vocab_size=1024
    )
    values["eos_token_id"] = 2
    config = Inkling2MMConfig(**values)
    head = _head(config)
    head.weight.data.fill_(100.0)
    head.weight.data[:1024].fill_(0.25)
    processor = Inkling2LogitsProcessor(config.text_config, _mapping())
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    hidden = torch.ones(2, 32, dtype=torch.bfloat16)
    logits = processor._get_logits(hidden, head, metadata, require_full_vocab=True)
    # Leaked stored padding would push the logits across a 1024-column sampling tile.
    assert logits.shape == (2, config.text_config.vocab_size) == (2, 1024)
    assert logits.dtype == torch.float32 and logits.is_contiguous()
    expected = 30 * torch.tanh(
        F.linear(hidden.float(), head.weight[:1024].float()) / 30
    )
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_fp32_head_result_precedes_softcap_and_valid_vocab_crop() -> None:
    torch.manual_seed(29)
    config = _tiny_config()
    head = _head(config)
    head.weight.data.normal_()
    hidden = torch.randn(3, 32, dtype=torch.bfloat16)
    processor = Inkling2LogitsProcessor(config.text_config, _mapping())
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    logits = processor._get_logits(hidden, head, metadata, require_full_vocab=True)
    projection = F.linear(hidden.float(), head.weight.float())
    expected = 30 * torch.tanh(projection[..., :60] / 30)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-6)
    assert not torch.equal(projection, projection.to(torch.bfloat16).float())
    assert logits.shape == (3, 60)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_gpu_softcap_matches_the_fp32_statement_byte_for_byte() -> None:
    torch.manual_seed(31)
    config = _tiny_config()
    head = _head(config).cuda()
    # Large logits reach the saturated tails of the cap.
    head.weight.data.normal_(std=8.0)
    hidden = torch.randn(5, 32, dtype=torch.bfloat16, device="cuda")
    processor = Inkling2LogitsProcessor(config.text_config, _mapping())
    metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
    logits = processor._get_logits(hidden, head, metadata, require_full_vocab=True)
    # The cap reads the head result's vocabulary columns in place (a strided
    # slice: the stored head has padding rows).
    processor.logit_softcap = None
    uncapped = processor._get_logits(hidden, head, metadata, require_full_vocab=True)
    assert logits.is_contiguous() and logits.shape[1] < head.weight.shape[0]
    assert torch.equal(logits, uncapped.div_(30.0).tanh_().mul_(30.0))


def test_logits_processor_uses_common_sampled_row_selection() -> None:
    config = _tiny_config()
    head = _head(config)
    head.weight.data.fill_(0.25)
    hidden = torch.arange(96, dtype=torch.float32).reshape(3, 32).to(torch.bfloat16)
    processor = Inkling2LogitsProcessor(config.text_config, _mapping())
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.EXTEND,
        query_shard=None,
        gather_ids=torch.tensor([2]),
    )
    output = processor(torch.tensor([1, 2, 3]), hidden, head, metadata)
    expected = 30 * torch.tanh(
        F.linear(hidden[-1:].float(), head.weight[:60].float()) / 30
    )
    torch.testing.assert_close(output.next_token_logits, expected, atol=1e-6, rtol=1e-6)


def test_head_keeps_fp32_accumulation_with_large_cancelling_terms() -> None:
    config = _tiny_config()
    head = _head(config)
    head.weight.data.zero_()
    head.weight.data[:, :3] = torch.tensor([128, 0.5, -128], dtype=torch.bfloat16)
    hidden = torch.ones(1, 32, dtype=torch.bfloat16)
    actual = head.quant_method.apply(head, hidden, None)
    torch.testing.assert_close(actual, torch.full((1, 64), 0.5), atol=0, rtol=0)


def _checkpoint_weights(
    model: Inkling2ForConditionalGeneration,
) -> list[tuple[str, torch.Tensor]]:
    weights = []
    projections = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")
    for name, param in model.named_parameters():
        name = name.replace("model.", "model.llm.", 1)
        name = name.replace("model.llm.embed_tokens.weight", "model.llm.embed.weight")
        name = name.replace("lm_head.weight", "model.llm.unembed.weight")
        if name.endswith(".attn.qkvr.weight"):
            sizes = (32, 16, 16, 8, 32, 4)
            for index, (projection, size) in enumerate(
                zip(projections, sizes, strict=True)
            ):
                checkpoint_name = name.replace(".qkvr.", f".{projection}.")
                weights.append(
                    (
                        checkpoint_name,
                        torch.full(
                            (size, 32),
                            index + 1,
                            dtype=param.dtype,
                        ),
                    )
                )
            continue
        for leaf in ("w13", "w2"):
            for suffix in ("scale", "amax", "outer_scale_denominator"):
                name = name.replace(
                    f".experts.{leaf}_{suffix}", f".experts.{leaf}_weight_{suffix}"
                )
        weights.append((name, torch.ones_like(param)))
    return weights


def test_complete_target_load_preserves_parameter_precision() -> None:
    model = _model()
    weights = _checkpoint_weights(model)
    loaded = model.load_weights(weights)
    assert len(loaded) == len(list(model.named_parameters()))
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.model.attn_res.output_query.dtype == torch.float32
    assert model.model.layers[0].mlp.gate.bias.dtype == torch.float32
    for name, param in model.named_parameters():
        if name.endswith(".attn.qkvr.weight"):
            offset = 0
            for index, size in enumerate((32, 16, 16, 8, 32, 4)):
                torch.testing.assert_close(
                    param[offset : offset + size],
                    torch.full_like(param[offset : offset + size], index + 1),
                    atol=0,
                    rtol=0,
                )
                offset += size
            attn = model.get_submodule(name.removesuffix(".qkvr.weight"))
            if not attn.is_local:
                # A full layer's tap rows are its loaded K and V tap projections.
                taps = torch.cat((attn.k_sconv.weight_proj, attn.v_sconv.weight_proj))
                assert torch.equal(param[offset : offset + len(taps)], taps)
                offset += len(taps)
            assert torch.count_nonzero(param[offset:]) == 0
        else:
            torch.testing.assert_close(
                param.float(), torch.ones_like(param.float()), atol=0, rtol=0
            )


def test_target_loader_rejects_missing_weights() -> None:
    model = _model()
    with pytest.raises(ValueError, match="Missing checkpoint tensors"):
        model.load_weights(_checkpoint_weights(model)[:-1])


def test_target_loader_rejects_repeated_weights() -> None:
    model = _model()
    weights = _checkpoint_weights(model)
    with pytest.raises(ValueError, match="Repeated"):
        model.load_weights([weights[0], weights[0]])


def test_target_loader_requires_every_projection_shard() -> None:
    model = _model()
    weights = [
        (name, value)
        for name, value in _checkpoint_weights(model)
        if name != "model.llm.layers.0.attn.wr_du.weight"
    ]
    with pytest.raises(ValueError, match=r"qkvr.weight shards \[3\]"):
        model.load_weights(weights)


def test_target_loader_rejects_weight_dtype_conversion() -> None:
    model = _model()
    name, value = _checkpoint_weights(model)[0]
    wrong_dtype = torch.float32 if value.dtype == torch.bfloat16 else torch.bfloat16
    with pytest.raises(
        ValueError, match="expected.*got|would change checkpoint precision"
    ):
        model.load_weights([(name, value.to(wrong_dtype))])


def test_target_requires_vision_config_for_image_execution() -> None:
    with pytest.raises(ValueError, match="vision_config"):
        Inkling2ForConditionalGeneration(
            config=_tiny_config(),
            mapping=_mapping(),
            quant_config=_tiny_quant_config(),
            is_multimodal_active=True,
            mm_attention_backend=None,
        )


def test_target_requires_original_precision_manifest() -> None:
    with pytest.raises(ValueError, match="checkpoint precision metadata"):
        Inkling2ForConditionalGeneration(
            config=_tiny_config(),
            mapping=_mapping(),
            quant_config=None,
            is_multimodal_active=False,
            mm_attention_backend=None,
        )
