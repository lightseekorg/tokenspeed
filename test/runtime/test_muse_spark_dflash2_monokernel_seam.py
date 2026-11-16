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

"""The DFlash2 draft monokernel seam: which draft forwards take the kernels,
and which drafts the kernels serve. CPU only (the drafts are built on the meta
device)."""

from __future__ import annotations

import os
import sys

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.layers.attention.backends.paged import mha  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig  # noqa: E402
from tokenspeed.runtime.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
)
from tokenspeed.runtime.models.dflash2 import (  # noqa: E402
    DFlash2DraftModel,
    DFlash2MonokernelDecoderLayer,
)
from tokenspeed.runtime.models.muse_spark_dflash2_monokernel import (  # noqa: E402
    BLOCK_ROWS,
    CONTEXT_OTHER,
    HEAD_NOT_SHARDED,
    HEAD_OTHER_SHAPES,
    MAX_REQUESTS,
    MLA_DRAFT,
    NOT_BF16_KV,
    NOT_TP8,
    OTHER_ATTENTION,
    OTHER_SHAPES,
    config_reasons,
    model_reasons,
    request_seq_lens,
    takes_head,
    takes_monokernels,
)
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402


def _ctx(mode: ForwardMode, bs: int, rows: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=rows,
        forward_mode=mode,
        output_layout=ForwardOutputLayout(0, 0, bs, BLOCK_ROWS),
    )


def test_the_blocks_of_up_to_max_requests_take_the_kernels() -> None:
    """A launch computes the blocks of 8 rows of up to MAX_REQUESTS requests;
    more requests, other row counts and other modes keep the regular path.
    The head and the context write serve the same requests: 7 draft steps
    and 8 verify rows each."""
    assert takes_monokernels(BLOCK_ROWS, _ctx(ForwardMode.DECODE, 1, BLOCK_ROWS))
    # Two requests' blocks only while MAX_REQUESTS allows them.
    two = MAX_REQUESTS >= 2
    assert takes_monokernels(2 * BLOCK_ROWS, _ctx(ForwardMode.DECODE, 2, 16)) == two
    assert not takes_monokernels(3 * BLOCK_ROWS, _ctx(ForwardMode.DECODE, 3, 24))
    assert not takes_monokernels(BLOCK_ROWS, _ctx(ForwardMode.DECODE, 2, 8))
    assert not takes_monokernels(BLOCK_ROWS, _ctx(ForwardMode.EXTEND, 1, 8))
    assert not takes_monokernels(BLOCK_ROWS, _ctx(ForwardMode.IDLE, 1, 8))
    assert [takes_head(r) for r in (7, 14, 21, 8)] == [True, two, False, False]


def _config(**overrides) -> PretrainedConfig:
    """Muse Spark Open's DFlash2 draft. ``decode_monokernel`` is what
    load_model_config declares on it: True under --decode-monokernel."""
    layers = overrides.pop("num_hidden_layers", 5)
    config = {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "is_causal": False,
        "layer_types": ["sliding_attention"] * layers,
        "max_position_embeddings": 4096,
        "num_attention_heads": 64,
        "num_hidden_layers": layers,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 500000, "rope_type": "default"},
        "rope_theta": 500000,
        "sliding_window": 2048,
        "vocab_size": 1024,
        "decode_monokernel": False,
        "dflash_config": {
            "attention_mode": "gqa",
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "final_logit_softcapping": 20,
            "mask_token_id": 2,
            "output_multiplier": 0.1767766952966369,
            "selector_rank": 16,
            "selector_top_k": 16,
            "target_layer_ids": [1, 16, 30, 44, 59],
        },
    }
    config.update(overrides)
    return PretrainedConfig(**config)


def _mapping(tp: int) -> Mapping:
    return Mapping(
        rank=0,
        world_size=tp,
        attn_tp_size=tp,
        attn_dp_size=1,
        dense_tp_size=tp,
        dense_dp_size=1,
        moe_tp_size=tp,
        moe_ep_size=1,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=tp,
        linear_attn_tp_size=tp,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=tp,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def _draft(config: PretrainedConfig, tp: int = 8) -> DFlash2DraftModel:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return DFlash2DraftModel(config, _mapping(tp))
    finally:
        torch.set_default_dtype(previous)


def test_muse_spark_opens_draft_at_tp8_has_monokernels() -> None:
    assert model_reasons(_draft(_config())) == []


def test_a_tp4_draft_has_no_layer_monokernels() -> None:
    assert model_reasons(_draft(_config(), tp=4)) == [NOT_TP8]


def test_decode_monokernel_refuses_a_draft_it_cannot_pack() -> None:
    """--decode-monokernel never serves a draft piece without its kernel:
    loading a TP4 draft fails and names its layers and context write and
    why, instead of drafting with the regular code."""
    draft = _draft(_config(decode_monokernel=True), tp=4)
    with pytest.raises(ValueError, match="--decode-monokernel") as error:
        draft.load_weights(iter(()))
    message = str(error.value)
    assert f"layers: {NOT_TP8}" in message
    assert f"context write: {CONTEXT_OTHER}" in message
    assert draft.monokernel_state is None


def _mla_checkpoint(draft: DFlash2DraftModel):
    """``draft``'s own parameters under the names of an MLA DFlash2
    checkpoint, which keeps the projections the draft fuses apart."""
    q_lora_rank = int(draft.config.q_lora_rank)
    for name, param in draft.named_parameters():
        if ".gate_up_proj." in name:
            gate, up = param.chunk(2)
            yield name.replace("gate_up_proj", "gate_proj"), gate
            yield name.replace("gate_up_proj", "up_proj"), up
        elif ".fused_qkv_a_proj_with_mqa." in name:
            fused = "fused_qkv_a_proj_with_mqa"
            yield name.replace(fused, "q_a_proj"), param[:q_lora_rank]
            yield name.replace(fused, "kv_a_proj_with_mqa"), param[q_lora_rank:]
        else:
            yield name, param


def test_an_mla_draft_under_decode_monokernel_fails_at_packing() -> None:
    """An MLA draft is not the one the kernels bake in: under the switch it
    still builds the monokernel layer class, and loading its weights packs
    it, which refuses it with the reason, so startup fails instead of serving
    it on the regular path. One rank: the weights load unsharded."""
    mla = _config(
        decode_monokernel=True,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
    )
    mla.dflash_config = {**mla.dflash_config, "attention_mode": "mla"}
    draft = _draft(mla, tp=1)
    assert config_reasons(mla) == [MLA_DRAFT]
    assert draft.decoder_layer_cls is DFlash2MonokernelDecoderLayer
    with pytest.raises(ValueError, match=f"layers: {MLA_DRAFT}"):
        draft.load_weights(_mla_checkpoint(draft))


def test_without_decode_monokernel_the_same_draft_loads() -> None:
    draft = _draft(_config(), tp=4)
    draft.load_weights(iter(()))
    assert draft.monokernel_state is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna4,
    reason="the draft monokernels are gfx950 kernels",
)
def test_decode_monokernel_packs_a_bf16_tp8_draft() -> None:
    """Muse Spark Open's BF16 draft at TP8 loads with every layer and the
    context write packed."""
    draft = _packed_draft()
    state = draft.monokernel_state
    assert state is not None
    assert all(layer.fused is not None for layer in draft.layers)
    assert state.context is not None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna4,
    reason="the draft monokernels are gfx950 kernels",
)
@pytest.mark.parametrize("requests", [1, 2])
def test_the_draft_reserves_every_iris_heap_buffer_it_allocates(
    monkeypatch, requests
) -> None:
    """What the draft's monokernels allocate on the Iris heap once it exists
    (the layers', the context write's and the head's mailbox scratch) is
    exactly what the draft reserved before, since the heap is created with
    only the reservations on top of its planned size: at one and two
    requests a launch."""
    from tokenspeed_kernel.ops.communication import _iris

    from tokenspeed.runtime.models import muse_spark_dflash2_monokernel as seam

    monkeypatch.setattr(seam, "MAX_REQUESTS", requests)
    monkeypatch.setattr(seam, "LAYER_ROWS", requests * seam.BLOCK_ROWS)
    monkeypatch.setattr(seam, "HEAD_ROWS", requests * seam.STEPS)
    monkeypatch.setattr(seam, "CONTEXT_WRITE_ROWS", requests * seam.CONTEXT_ROWS)
    monkeypatch.setattr(_iris, "_reserved", {})
    draft = _packed_draft()
    reserved = _iris._reserved["muse_spark_dflash2_monokernel"]

    # A heap that records each allocation at the Torch allocator's alignment.
    allocated: list[int] = []

    class Heap:
        def zeros(self, shape, dtype):
            allocated.append(_iris.allocation_bytes(tuple(shape), dtype))
            return torch.zeros(shape, dtype=dtype)

    class Group:
        def __init__(self, ctx, group=None, rank=0, device=None):
            self.bases = torch.zeros(8, dtype=torch.int64)

        def finish_prepare(self) -> None:
            pass

    monkeypatch.setattr(_iris, "existing_context", Heap)
    monkeypatch.setattr(_iris, "IrisGroup", Group)
    # The layers', the context write's and the head's scratch, in one group.
    draft.monokernel_state.prepare(None, None, 0)
    assert len(allocated) == 3
    assert sum(allocated) == reserved


def _packed_draft() -> DFlash2DraftModel:
    """Muse Spark Open's BF16 TP8 draft (rank 0) on the GPU, loaded with
    --decode-monokernel (set by the caller): its layers and context write
    packed."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            draft = DFlash2DraftModel(_config(decode_monokernel=True), _mapping(8))
    finally:
        torch.set_default_dtype(previous)
    draft.load_weights(iter(()))
    return draft


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna4,
    reason="the draft monokernels are gfx950 kernels",
)
def test_decode_monokernel_refuses_a_head_it_cannot_pack() -> None:
    """A draft head over an LM head that is not an 8-way vocabulary shard
    (here the whole 1024-token head on one rank, and the test config's
    1024-token selector) fails startup in the draft's configure_target, before
    the KV-cache budget is profiled, and names the head's reasons."""
    draft = _packed_draft()
    lm_head = ParallelLMHead(1024, 8192, params_dtype=torch.bfloat16).to("cuda")

    class Target(torch.nn.Module):
        """A target that installs the draft's captures and owns the head."""

        def __init__(self) -> None:
            super().__init__()
            self.lm_head = lm_head
            self.captured: list[int] = []

        def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
            self.captured = layer_ids

    with pytest.raises(
        ValueError, match="no monokernel for the DFlash2 draft's head"
    ) as error:
        draft.configure_target(Target(), None)
    assert f"head: {HEAD_NOT_SHARDED}, {HEAD_OTHER_SHAPES}" in str(error.value)
    assert draft.monokernel_state.packed_head is None
    assert draft.monokernel_state.head is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna4,
    reason="the draft monokernels are gfx950 kernels",
)
def test_a_drafter_wired_before_the_head_is_packed_fails_as_an_ordering_error() -> None:
    """The drafter enables the head launch only after configure_target packed
    it and prepare_communication_runtime allocated its scratch; wired before
    either, it raises an internal ordering error, not an unserved head."""
    draft = _packed_draft()
    with pytest.raises(RuntimeError, match="configure_target packed its head"):
        draft.monokernel_state.enable_head(top_k_exact=True, scale=1.0, softcap=20.0)
    assert draft.monokernel_state.head is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna4,
    reason="the draft monokernels are gfx950 kernels",
)
def test_the_draft_launches_in_serving_order_never_repeat_a_tag(monkeypatch) -> None:
    """Each verify writes the draft's context, runs the draft block's layer
    launches, then the head. The context write and the head launch under the
    draft's step counter, which the block moves: every context write gets a
    new tag, and a second context write before the block (it would repeat
    the tag) or a head without a block before it fails before launching."""
    from tokenspeed.runtime.models import muse_spark_dflash2_monokernel as seam

    draft = _packed_draft()
    state = draft.monokernel_state
    # The launches need the TP8 group; these record each one's tag.
    context_tags: list[int] = []
    head_tags: list[int] = []

    def context_launch(fused, *, compile_only, step, **operands) -> None:
        if not compile_only:
            context_tags.append(int(step))

    def head_launch(hidden, first_token):
        head_tags.append(int(state.step[0]))
        return hidden, first_token, first_token

    monkeypatch.setattr(seam, "muse_spark_draft_context_decode", context_launch)
    monkeypatch.setattr(seam.dist, "barrier", lambda group=None: None)
    state.head = head_launch

    class Pool:
        """The draft's KV pool: one cache page for each layer."""

        def get_kv_buffer(self, layer_id):
            kv = torch.zeros(64, 128, dtype=torch.bfloat16, device="cuda")
            return kv, kv

    rows = seam.CONTEXT_ROWS
    # write_context reads the target's five tapped layers, 8192 columns each.
    taps = torch.zeros(rows, 5 * 8192, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(rows, device="cuda")
    hidden = torch.zeros(
        seam.STEPS, int(draft.config.hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    first = torch.zeros(1, dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="after the draft layers' launches"):
        state.run_head(hidden, first)
    for verify in range(3):
        state.write_context(taps, positions, positions, Pool())
        with pytest.raises(RuntimeError, match="would repeat its launch's tag"):
            state.write_context(taps, positions, positions, Pool())
        # The block forward: its layers launch, and the step moves after them.
        state.start_step()
        state.launched = True
        state.finish_step()
        state.run_head(hidden, first)
        with pytest.raises(RuntimeError, match="after the draft layers' launches"):
            state.run_head(hidden, first)
    assert context_tags == [0, 1, 2]
    assert head_tags == [1, 2, 3]


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"num_hidden_layers": 4}, OTHER_SHAPES),
        ({"intermediate_size": 14336}, OTHER_SHAPES),
        # A 4096-token window, and a full-attention layer.
        ({"sliding_window": 4096}, OTHER_ATTENTION),
        (
            {"layer_types": ["sliding_attention"] * 4 + ["full_attention"]},
            OTHER_ATTENTION,
        ),
        ({"rms_norm_eps": 1e-6}, OTHER_ATTENTION),
    ],
)
def test_other_drafts_have_no_layer_monokernels(overrides, reason) -> None:
    assert reason in model_reasons(_draft(_config(**overrides)))


@pytest.mark.parametrize(
    "drafter_attention_backend, reasons", [(None, []), ("trtllm", [NOT_BF16_KV])]
)
def test_an_fp8_server_keeps_the_drafts_cache_bf16_unless_trtllm_attends(
    monkeypatch, drafter_attention_backend, reasons
) -> None:
    """The kernels read and write BF16 keys and values. Under an FP8
    --kv-cache-dtype the DFLASH draft's own cache stays BF16, except when it
    attends with trtllm."""
    monkeypatch.setitem(global_server_args_dict, "kv_cache_dtype", "fp8_e4m3")
    monkeypatch.setitem(global_server_args_dict, "speculative_algorithm", "DFLASH")
    monkeypatch.setitem(
        global_server_args_dict, "drafter_attention_backend", drafter_attention_backend
    )
    assert model_reasons(_draft(_config())) == reasons


def test_each_request_of_a_block_forward_reads_its_own_length() -> None:
    """Under block decode the leaf keeps one cache-seqlens entry per block
    position (``block_decode_expansion`` a request), so the launches' per
    request lengths are every expansion-th entry: with two requests decoding
    together, the second reads its own length, not the first request's
    broadcast copy. One request's length stays a view (no copy on C1)."""
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=8,
        num_kv_heads=1,
        head_dim=128,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=8192,
        max_bs=4,
        max_resident_requests=4,
        speculative_num_draft_tokens=BLOCK_ROWS,
        is_draft=True,
        draft_block_decode=True,
        components=(spec,),
    )
    leaf = mha.MHAAttnBackend(config, spec, kernel_page_size=64)
    leaf.init_cuda_graph_state(4)
    assert leaf.block_decode_expansion == BLOCK_ROWS
    # The drafter's in-graph broadcast of each request's block-end length.
    leaf.fill_block_decode_seq_lens(2, torch.tensor([100, 3000], dtype=torch.int32))
    assert request_seq_lens(leaf, 2).tolist() == [100, 3000]
    one = request_seq_lens(leaf, 1)
    assert one.tolist() == [100]
    assert one.data_ptr() == leaf.decode_seq_lens_buffer.data_ptr()
