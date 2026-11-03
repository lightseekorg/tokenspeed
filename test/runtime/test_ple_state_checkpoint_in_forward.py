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

"""A Qwen4-Exp PLE layer writes the in-forward State checkpoint on CUDA.

The PLE analogue of ``test_gdn_state_checkpoint_in_forward.py``: an extend
row whose chunk spans a checkpoint stores, into its third block, the PLE
state a prefix ending at the checkpoint leaves -- the trailing token context
and the short conv's input window. That block must equal, bit for bit, what a
forward over the same tokens ending exactly at the checkpoint writes into its
live block.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.configs.qwen4_exp_config import (  # noqa: E402
    Qwen4ExpTextConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (  # noqa: E402
    Qwen4ExpPLEBackend,
)
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (  # noqa: E402
    QWEN4_EXP_PLE_CACHE_GROUP,
    qwen4_exp_ple_context_field,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.layers.qwen4_exp_ple import Qwen4ExpPLELayer  # noqa: E402

PAGES = 16
# Blocks of the PLE group's pool (0 is the null block).
SEED_BLOCK = 5  # row 1's history, written by a first forward
LIVE_BLOCK = 6  # row 1's live block in the spanning forward
CHECKPOINT_BLOCK = 7  # row 1's in-forward checkpoint block
REFERENCE_BLOCK = 8  # the live block of the forward ending at the checkpoint
HISTORY = 24  # row 1's tokens before the spanning forward
CHUNK = 40  # row 1's tokens in the spanning forward
OFFSET = 32  # the checkpoint, OFFSET tokens into row 1's chunk
OTHER = 20  # row 0: a fresh request in the same batch, without a checkpoint


def _config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=257,
        make_ngram_vocab_size_divisible_by=8,
        ple_offload_embedding=False,
        hc_count=2,
        num_experts=None,
        eos_token_id=7,
    )


def _backend(fields: dict[str, torch.Tensor]) -> Qwen4ExpPLEBackend:
    """The real PLE backend over a stand-in arena holding ``fields``."""
    backend = Qwen4ExpPLEBackend(
        SimpleNamespace(
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            is_draft=False,
            speculative_num_draft_tokens=1,
        ),
        SimpleNamespace(
            num_attention_heads=1, num_kv_heads=1, attn_tp_size=1, head_dim=8
        ),
    )
    backend.set_cache_pool(
        SimpleNamespace(
            field_layer_range=range(2),
            _field_layer_id=lambda layer: layer,
            layerwise_load_tracker=None,
            arena=SimpleNamespace(
                plan=SimpleNamespace(
                    fields=tuple(
                        SimpleNamespace(
                            group_id=QWEN4_EXP_PLE_CACHE_GROUP, field_id=field_id
                        )
                        for field_id in fields
                    )
                ),
                field=fields.__getitem__,
                runtime_contract=SimpleNamespace(
                    group_specs=(
                        SimpleNamespace(
                            group_id=QWEN4_EXP_PLE_CACHE_GROUP, attn_kind="state"
                        ),
                    )
                ),
            ),
        )
    )
    backend.init_cuda_graph_state(4)
    return backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_spanning_forward_stores_what_a_forward_ending_there_leaves():
    torch.manual_seed(0)
    config = _config()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            layer = Qwen4ExpPLELayer(
                config,
                Mapping(rank=0, world_size=1),
                layer_id=0,
                ple_layer_index=0,
                quant_config=None,
                prefix="model.layers.0.ple",
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    with torch.no_grad():
        for parameter in layer.parameters():
            if parameter.is_floating_point():
                parameter.copy_(torch.randn_like(parameter, dtype=torch.float32))
        layer.ple_embedding.refresh_ngram_reciprocals()
    hc_hidden, state_len = config.short_conv_state_shape
    assert state_len > 0 and config.ngram_context_len > 0

    context_field = qwen4_exp_ple_context_field(0)
    conv_field = qwen4_exp_ple_conv_field(0)
    fields = {
        context_field: torch.zeros(
            PAGES, config.ngram_context_len, dtype=torch.int64, device="cuda"
        ),
        conv_field: torch.zeros(
            PAGES, hc_hidden, state_len, dtype=torch.bfloat16, device="cuda"
        ),
    }
    backend = _backend(fields)
    layer._ple_backend = lambda ctx: backend

    history_ids = torch.randint(0, 128, (HISTORY,), device="cuda")
    history_hidden = torch.randn(HISTORY, hc_hidden, device="cuda").bfloat16()
    other_ids = torch.randint(0, 128, (OTHER,), device="cuda")
    other_hidden = torch.randn(OTHER, hc_hidden, device="cuda").bfloat16()
    chunk_ids = torch.randint(0, 128, (CHUNK,), device="cuda")
    chunk_hidden = torch.randn(CHUNK, hc_hidden, device="cuda").bfloat16()

    def forward(rows, input_ids, hidden_states, checkpoint_lens=None):
        """One extend forward over ``rows``.

        Each row is ``(tokens before, new tokens, block row)`` for one request.
        """
        before = torch.tensor([row[0] for row in rows], dtype=torch.int32)
        lengths = torch.tensor([row[1] for row in rows], dtype=torch.int32)
        bs = len(rows)
        backend.init_forward_metadata(
            bs,
            bs,
            torch.arange(bs, dtype=torch.int32, device="cuda"),
            (before + lengths).cuda(),
            ForwardMode.EXTEND,
            block_tables={
                QWEN4_EXP_PLE_CACHE_GROUP: torch.tensor(
                    [row[2] for row in rows], dtype=torch.int32, device="cuda"
                )
            },
            extend_seq_lens=lengths.cuda(),
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens=before.cuda(),
            extend_prefix_lens_cpu=before,
            extend_replay_lens_cpu=torch.zeros_like(before),
            extend_prompt_lens_cpu=before + lengths,
            extend_with_prefix=bool(before.any()),
            query_shard=None,
            extend_state_checkpoint_lens_cpu=checkpoint_lens,
        )
        ctx = SimpleNamespace(
            bs=bs,
            forward_mode=ForwardMode.EXTEND,
            token_to_kv_pool=backend.cache_pool,
            global_num_tokens=None,
        )
        return layer(hidden_states, input_ids, ctx)

    # Row 1's history: its first chunk leaves the seed block.
    forward([(0, HISTORY, [1, SEED_BLOCK])], history_ids, history_hidden)
    # The spanning forward: a fresh row 0 beside row 1, whose chunk starts at
    # packed row OTHER and writes the checkpoint OFFSET tokens in.
    forward(
        [
            (0, OTHER, [2, 3, -1]),
            (HISTORY, CHUNK, [SEED_BLOCK, LIVE_BLOCK, CHECKPOINT_BLOCK]),
        ],
        torch.cat((other_ids, chunk_ids)),
        torch.cat((other_hidden, chunk_hidden)),
        checkpoint_lens=torch.tensor([0, OFFSET], dtype=torch.int32),
    )
    assert backend.forward_metadata.state_checkpoints is not None
    # A forward over the same tokens ending at the checkpoint.
    forward(
        [(HISTORY, OFFSET, [SEED_BLOCK, REFERENCE_BLOCK])],
        chunk_ids[:OFFSET],
        chunk_hidden[:OFFSET],
    )
    assert backend.forward_metadata.state_checkpoints is None
    torch.cuda.synchronize()

    context = fields[context_field]
    conv = fields[conv_field]
    assert torch.equal(context[CHECKPOINT_BLOCK], context[REFERENCE_BLOCK])
    assert torch.equal(conv[CHECKPOINT_BLOCK], conv[REFERENCE_BLOCK])
    assert context[CHECKPOINT_BLOCK].tolist() == (
        chunk_ids[OFFSET - config.ngram_context_len : OFFSET].tolist()
    )
    assert conv[CHECKPOINT_BLOCK].abs().sum() > 0, "the window holds real inputs"
    assert not torch.equal(
        conv[LIVE_BLOCK], conv[CHECKPOINT_BLOCK]
    ), "the live block holds the chunk end"
    # Row 0 wrote only its live block.
    assert conv[3].abs().sum() > 0
