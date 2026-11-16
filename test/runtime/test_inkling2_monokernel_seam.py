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

"""The Inkling2 decode monokernel seam: which forwards launch, the rows'
positions, and which layers it packs. CPU only."""

from __future__ import annotations

import os
import sys
from test.runtime.models.test_inkling2_config import _checkpoint
from test.runtime.test_inkling2_quantization import _group

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.inkling2 import Inkling2MMConfig  # noqa: E402
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.layers.quantization.inkling2 import (  # noqa: E402
    Inkling2QuantConfig,
)
from tokenspeed.runtime.models.inkling2 import (  # noqa: E402
    Inkling2ForConditionalGeneration,
)
from tokenspeed.runtime.models.inkling2_monokernel import (  # noqa: E402
    BATCHES,
    CHAIN,
    ROWS,
    Inkling2Monokernel,
    launch_requests,
    verify_positions,
)


def _ctx(mode: ForwardMode | None, bs: int, tokens: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=tokens,
        forward_mode=mode,
        output_layout=ForwardOutputLayout(0, 0, bs, CHAIN),
    )


@pytest.mark.parametrize("bs", BATCHES)
def test_a_verify_batch_of_the_graph_sizes_launches(bs) -> None:
    """A verify batch of each served size (``BATCHES``), a block of 8 tokens each,
    launches as ``8 bs`` rows."""
    assert launch_requests(_ctx(ForwardMode.DECODE, bs, CHAIN * bs), CHAIN) == bs
    assert CHAIN * bs in ROWS


def test_other_forwards_keep_the_regular_path() -> None:
    decode = ForwardMode.DECODE
    # A batch past the largest launch (16 requests).
    assert launch_requests(_ctx(decode, 16, CHAIN * 16), CHAIN) is None
    # A batch size the decode graphs do not capture.
    assert launch_requests(_ctx(decode, 3, CHAIN * 3), CHAIN) is None
    # Decode without speculation, or a verify of another width.
    assert launch_requests(_ctx(decode, 1, 1), 1) is None
    assert launch_requests(_ctx(decode, 1, 4), 4) is None
    # Rows that are not one block a request.
    assert launch_requests(_ctx(decode, 2, CHAIN), CHAIN) is None
    assert launch_requests(_ctx(ForwardMode.EXTEND, 1, CHAIN), CHAIN) is None
    assert launch_requests(_ctx(None, 1, CHAIN), CHAIN) is None


def test_row_positions_follow_each_request_length() -> None:
    """Row ``j`` of request ``i`` sits at ``seq_lens[i] - 8 + j``. A padded
    request's length is the verify floor 8: its rows sit at 0 .. 7."""
    offsets = torch.arange(-CHAIN, 0, dtype=torch.int64)
    out = torch.full((3 * CHAIN + 3,), -1, dtype=torch.int64)
    seq_lens = torch.tensor([1500, 9, CHAIN], dtype=torch.int32)
    got = verify_positions(seq_lens, offsets, out)
    want = [1492 + j for j in range(8)] + [1 + j for j in range(8)] + list(range(8))
    assert got.tolist() == want
    # The launch reads the first rows of the seam's buffer.
    assert got.data_ptr() == out.data_ptr()
    assert out[3 * CHAIN :].eq(-1).all()


def _tiny_config(layers: int = 2, full: tuple[int, ...] = ()) -> Inkling2MMConfig:
    """Local layers in attention-residual blocks of two: layer 1 is a mid
    local layer with NVFP4 experts, layer 2 (with three layers) the first
    of the second block. The layers in ``full`` are full attention layers."""
    values = _checkpoint()
    values["text_config"].update(
        hidden_size=32,
        num_hidden_layers=layers,
        vocab_size=64,
        unpadded_vocab_size=60,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=2,
        rel_extent=8,
        local_layer_ids=[i for i in range(layers) if i not in full],
        swa_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        sliding_window_size=8,
        n_layers_per_attn_res_block=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        intermediate_size=32,
    )
    values["vision_config"] = None
    values["mtp_config"] = None
    values["eos_token_id"] = 2
    return Inkling2MMConfig(**values)


def _tiny_quant_config(
    layers: int = 2, full: tuple[int, ...] = ()
) -> Inkling2QuantConfig:
    bf16 = [
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
    ]
    fp32 = [f"model.llm.attn_res.attn_queries.{i}" for i in range(layers - 1)]
    fp32.append("model.llm.attn_res.output_query")
    groups = []
    for layer in range(layers):
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
        norms = (
            ("q_input_norm", "k_input_norm", "v_input_norm")
            if layer in full
            else ("attn_norm",)
        )
        bf16.extend(
            f"{prefix}.{leaf}.weight"
            for leaf in ("mlp.gate", "attn_sconv", "mlp_sconv", "mlp_norm", *norms)
        )
        leaves = ("static_w", "weight_proj") if layer in full else ("weight",)
        bf16.extend(
            f"{prefix}.attn.{stream}.{leaf}"
            for stream in ("k_sconv", "v_sconv")
            for leaf in leaves
        )
        groups.append(
            _group(
                "nvfp4",
                [f"{prefix}.mlp.experts.{leaf}_weight" for leaf in ("w13", "w2")],
            )
        )
    groups.extend([_group("bfloat16", bf16), _group("float32", fp32)])
    output = _group("bfloat16", ["model.llm.unembed.weight"])
    output["fwd_output_dtype"] = "float32"
    groups.append(output)
    return Inkling2QuantConfig({"model": {"parameter_groups": groups}})


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


def test_the_seam_packs_the_mid_local_layers_or_names_the_unserved_one() -> None:
    """The local NVFP4 layers (layer 0 and the mid one) are the ones the seam
    packs; a layer that no kernel serves (here: not the kernel's shapes, or
    no kernel on this platform) fails startup with its index, and no layer
    launches."""
    mapping = _mapping()
    model = Inkling2ForConditionalGeneration(
        config=_tiny_config(),
        mapping=mapping,
        quant_config=_tiny_quant_config(),
        is_multimodal_active=False,
        mm_attention_backend=None,
    ).model
    seam = Inkling2Monokernel(model, mapping)
    assert seam.eligible == [0, 1]
    with pytest.raises(RuntimeError, match="layer 0"):
        seam.pack()
    assert seam.layers == {}
    assert all(layer.monokernel is None for layer in model.layers)


def test_the_seam_takes_the_boundary_local_layers_and_layer_0() -> None:
    """The first local layer of every attention-residual block, layer 0 (it
    starts from the embedding) included, is eligible with the mid ones."""
    mapping = _mapping()
    model = Inkling2ForConditionalGeneration(
        config=_tiny_config(3),
        mapping=mapping,
        quant_config=_tiny_quant_config(3),
        is_multimodal_active=False,
        mm_attention_backend=None,
    ).model
    seam = Inkling2Monokernel(model, mapping)
    assert seam.eligible == [0, 1, 2]


def test_the_seam_takes_the_full_attention_layers() -> None:
    """A full attention layer (the last of its attention-residual block) is
    eligible with the local ones."""
    mapping = _mapping()
    model = Inkling2ForConditionalGeneration(
        config=_tiny_config(3, full=(1,)),
        mapping=mapping,
        quant_config=_tiny_quant_config(3, full=(1,)),
        is_multimodal_active=False,
        mm_attention_backend=None,
    ).model
    assert not model.layers[1].is_local
    seam = Inkling2Monokernel(model, mapping)
    assert seam.eligible == [0, 1, 2]
