"""Tests for draft-to-target wiring.

Model-to-model wiring (shared embed/head, EAGLE3 capture ids) happens in
``factory._wire_draft_to_target_model`` right after both models load, so
shared weights are released before the KV-cache budget is profiled.
Drafter-instance wiring (capture hooks on the target) happens in
``BaseDrafter.wire_target`` implementations.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import tokenspeed.runtime.execution.factory as factory  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.drafter import get_drafter_impl  # noqa: E402
from tokenspeed.runtime.execution.drafter.base import BaseDrafter  # noqa: E402
from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (  # noqa: E402
    DeepseekV4DSpark,
)
from tokenspeed.runtime.execution.drafter.dflash import DFlash  # noqa: E402
from tokenspeed.runtime.execution.drafter.dspark import DSpark  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import Eagle  # noqa: E402
from tokenspeed.runtime.execution.drafter.mtp import Mtp  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402


def _server_args(algo: str, capture_ids=None) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_algorithm=algo,
        eagle3_layers_to_capture=capture_ids,
    )


def test_get_drafter_impl_routing():
    from tokenspeed.runtime.models.deepseek_v4_dspark import (
        DeepseekV4ForCausalLMDSpark,
    )
    from tokenspeed.runtime.models.inkling_nextn import (
        InklingForConditionalGenerationNextN,
    )

    assert get_drafter_impl("EAGLE3", mock.MagicMock()) is Eagle
    assert get_drafter_impl("MTP", mock.MagicMock()) is Eagle
    assert (
        get_drafter_impl(
            "MTP", mock.MagicMock(spec=InklingForConditionalGenerationNextN)
        )
        is Mtp
    )
    assert get_drafter_impl("DFLASH", mock.MagicMock()) is DFlash
    assert get_drafter_impl("DSPARK", mock.MagicMock()) is DSpark
    assert (
        get_drafter_impl("DSPARK", mock.MagicMock(spec=DeepseekV4ForCausalLMDSpark))
        is DeepseekV4DSpark
    )


def test_shares_target_embed_head_flags():
    # Eagle-like drafters and the checkpoint-local V4 DSpark reuse the
    # target's embed/LM head; DFlash and generic DSpark ship their own.
    assert Eagle.shares_target_embed_head
    assert Mtp.shares_target_embed_head
    assert DeepseekV4DSpark.shares_target_embed_head
    assert not DFlash.shares_target_embed_head
    assert not DSpark.shares_target_embed_head
    assert not BaseDrafter.shares_target_embed_head


def test_pd_layerwise_finalization_capability_matches_supported_drafters():
    assert Eagle.supports_pd_layerwise_finalization
    assert DFlash.supports_pd_layerwise_finalization
    assert DSpark.supports_pd_layerwise_finalization
    assert not Mtp.supports_pd_layerwise_finalization
    assert not DeepseekV4DSpark.supports_pd_layerwise_finalization
    assert not BaseDrafter.supports_pd_layerwise_finalization


def test_wire_eagle3_shares_embed_head_and_installs_capture_ids():
    target, draft = mock.MagicMock(), mock.MagicMock()
    target.model.get_embed_and_head.return_value = ("EMBED", "HEAD")
    draft.model_config.hf_config = {
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}
    }

    with mock.patch.object(factory, "get_drafter_impl", return_value=Eagle):
        factory._wire_draft_to_target_model(_server_args("EAGLE3"), target, draft)

    draft.model.set_embed_and_head.assert_called_once_with("EMBED", "HEAD")
    target.model.set_eagle3_layers_to_capture.assert_called_once_with([1, 2, 3])


def test_wire_eagle3_explicit_capture_ids_override_checkpoint():
    target, draft = mock.MagicMock(), mock.MagicMock()
    target.model.get_embed_and_head.return_value = ("E", "H")
    draft.model_config.hf_config = {
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}
    }

    with mock.patch.object(factory, "get_drafter_impl", return_value=Eagle):
        factory._wire_draft_to_target_model(
            _server_args("EAGLE3", capture_ids=[7, 8]), target, draft
        )

    target.model.set_eagle3_layers_to_capture.assert_called_once_with([7, 8])


def test_wire_dflash_keeps_own_embed_head():
    target, draft = mock.MagicMock(), mock.MagicMock()

    with mock.patch.object(factory, "get_drafter_impl", return_value=DFlash):
        factory._wire_draft_to_target_model(_server_args("DFLASH"), target, draft)

    target.model.get_embed_and_head.assert_not_called()
    draft.model.set_embed_and_head.assert_not_called()


def test_base_wire_target_is_a_noop():
    drafter = mock.MagicMock(spec=BaseDrafter)
    target_model = mock.MagicMock()
    BaseDrafter.wire_target(drafter, target_model)
    target_model.assert_not_called()


def test_dflash_wire_target_requires_capture_support():
    drafter = mock.MagicMock(spec=DFlash)
    target_model = mock.MagicMock(
        spec=["get_input_embeddings", "lm_head", "logits_processor"]
    )
    with pytest.raises(ValueError, match="set_dflash_layers_to_capture"):
        DFlash.wire_target(drafter, target_model)


def test_dflash_wire_target_installs_capture_layers():
    drafter = mock.MagicMock(spec=DFlash)
    drafter.target_layer_ids = [4, 5]
    target_model = mock.MagicMock(
        spec=[
            "get_input_embeddings",
            "lm_head",
            "logits_processor",
            "set_dflash_layers_to_capture",
        ]
    )

    DFlash.wire_target(drafter, target_model)

    target_model.set_dflash_layers_to_capture.assert_called_once_with([4, 5])
    assert drafter.embed_tokens is target_model.get_input_embeddings.return_value
    assert drafter.lm_head is target_model.lm_head


def test_dspark_wire_target_installs_capture_layers():
    drafter = mock.MagicMock(spec=DeepseekV4DSpark)
    drafter.target_layer_ids = [10, 20]
    target_model = mock.MagicMock(
        spec=["lm_head", "logits_processor", "set_dspark_layers_to_capture"]
    )

    DeepseekV4DSpark.wire_target(drafter, target_model)

    target_model.set_dspark_layers_to_capture.assert_called_once_with([10, 20])
    assert drafter.lm_head is target_model.lm_head


# --------------------------------------------------------------------------
# prepare_target_forward: what a drafter attaches to the target's context
# --------------------------------------------------------------------------


def _target_ctx(num_extends: int, num_tokens: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=SimpleNamespace(
            decode_window_locations=lambda: torch.arange(100, 100 + 2 * num_tokens)
        ),
        token_to_kv_pool=None,
        bs=2,
        num_extends=num_extends,
        input_num_tokens=num_tokens,
        forward_mode=ForwardMode.DECODE,
    )


def _incremental_dflash(enabled: bool) -> DFlash:
    """A DFlash with only the incremental-projection state set up."""
    drafter = DFlash.__new__(DFlash)
    drafter._incremental_proj_enabled = enabled
    drafter._fused_kv_enabled = True
    drafter._kv_aux_stream = object()
    drafter._incremental_kv_write_done = True  # stale from a previous round
    drafter._incr_num_tokens = 0
    drafter._incr_acc_buf = torch.ones(16, 3)
    drafter.input_buffers = SimpleNamespace(positions_buf=torch.arange(16))
    return drafter


def test_base_prepare_target_forward_attaches_nothing():
    ctx = _target_ctx(num_extends=0, num_tokens=4)
    BaseDrafter.prepare_target_forward(mock.MagicMock(spec=BaseDrafter), ctx)
    assert ctx.target_capture_sink is None


def test_dflash_arms_the_incremental_projection_on_the_forward_context():
    drafter = _incremental_dflash(enabled=True)
    ctx = _target_ctx(num_extends=0, num_tokens=4)

    drafter.prepare_target_forward(ctx)

    assert ctx.target_capture_sink is drafter
    assert drafter._incremental_kv_write_done is False
    assert drafter._incr_num_tokens == 4
    assert torch.equal(drafter._incr_positions, torch.arange(4))
    assert torch.equal(drafter._incr_cache_locs, torch.arange(100, 104))
    # The accumulator rows of this forward are cleared, nothing else.
    assert torch.equal(drafter._incr_acc_buf[:4], torch.zeros(4, 3))
    assert torch.equal(drafter._incr_acc_buf[4:], torch.ones(12, 3))


@pytest.mark.parametrize(
    "enabled, num_extends, graph_warmup",
    [
        (False, 0, False),  # projection disabled
        (True, 1, False),  # a mixed round: extend rows in the write vector
        (True, 0, True),  # graph warmup runs auxiliary branches serially
    ],
)
def test_dflash_leaves_the_context_alone_when_it_will_not_overlap(
    enabled, num_extends, graph_warmup
):
    import tokenspeed.runtime.execution.drafter.dflash as dflash_module

    drafter = _incremental_dflash(enabled=enabled)
    ctx = _target_ctx(num_extends=num_extends, num_tokens=4)

    with mock.patch.object(
        dflash_module, "get_is_cuda_graph_phase", return_value=graph_warmup
    ):
        drafter.prepare_target_forward(ctx)

    assert ctx.target_capture_sink is None
    # A stale "already written" from an earlier round never survives into
    # a round that did not arm.
    assert drafter._incremental_kv_write_done is False
    assert torch.equal(drafter._incr_acc_buf, torch.ones(16, 3))


def test_dflash_rejects_a_capture_of_the_wrong_width():
    drafter = _incremental_dflash(enabled=True)
    drafter.prepare_target_forward(_target_ctx(num_extends=0, num_tokens=4))
    with pytest.raises(RuntimeError, match="armed for 4"):
        drafter.on_target_capture(0, torch.zeros(3, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="aux-stream projection")
def test_dflash_sink_folds_the_taps_and_writes_the_kv_once():
    """fc over the concatenated taps == the sum of per-tap GEMMs, accumulated
    on the aux stream as each tap lands; the last tap writes the KV."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, hidden, n_taps = 4, 8, 2
    drafter = _incremental_dflash(enabled=True)
    drafter.input_buffers = SimpleNamespace(
        positions_buf=torch.arange(16, device=device)
    )
    drafter._incr_acc_buf = torch.ones(16, hidden, device=device)
    drafter._incr_slot_bufs = [
        torch.empty(16, hidden, device=device) for _ in range(n_taps)
    ]
    drafter._incr_capture_events = [torch.cuda.Event() for _ in range(n_taps)]
    drafter._incr_sub_weights_t = [
        torch.randn(hidden, hidden, device=device) for _ in range(n_taps)
    ]
    drafter._incr_n_captures = n_taps
    drafter._incr_hidden_norm = lambda acc: acc * 2
    drafter._kv_aux_stream = torch.cuda.Stream()
    drafter._kv_join_event = torch.cuda.Event()
    writes = []
    drafter._write_native_cache_fused = (
        lambda ctx_hidden, positions, locs: writes.append(
            (ctx_hidden.clone(), positions.clone(), locs.clone())
        )
    )
    ctx = _target_ctx(num_extends=0, num_tokens=num_tokens)
    ctx.attn_backend = SimpleNamespace(
        decode_window_locations=lambda: torch.arange(100, 132, device=device)
    )
    taps = [torch.randn(num_tokens, hidden, device=device) for _ in range(n_taps)]

    drafter.prepare_target_forward(ctx)
    assert ctx.target_capture_sink is drafter
    ctx.target_capture_sink.on_target_capture(0, taps[0])
    assert writes == []
    ctx.target_capture_sink.on_target_capture(1, taps[1])
    torch.cuda.synchronize()

    expected = (
        taps[0] @ drafter._incr_sub_weights_t[0]
        + taps[1] @ drafter._incr_sub_weights_t[1]
    )
    torch.testing.assert_close(drafter._incr_acc_buf[:num_tokens], expected)
    assert drafter._incremental_kv_write_done is True
    ((ctx_hidden, positions, locs),) = writes
    torch.testing.assert_close(ctx_hidden, expected * 2)
    assert torch.equal(positions, torch.arange(num_tokens, device=device))
    assert torch.equal(locs, torch.arange(100, 100 + num_tokens, device=device))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
