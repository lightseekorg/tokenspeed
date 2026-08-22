"""Semantic coverage for the DSpark drafter's config and wiring contract.

These are the cheap guards that catch a mis-specified launch before eight GPUs
are committed to loading a 1T-parameter target: block geometry, Markov-head
resolution, algorithm dispatch, the draft-worker architecture rewrite, and the
target-side capture the draft is fed from.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.execution import model_executor as model_executor_module
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.dflash import (
    DFlash,
    _resolve_block_geometry,
    _resolve_draft_query_width,
)
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.layers.attention.configs.base import (
    resolve_speculative_num_tokens,
)
from tokenspeed.runtime.layers.attention.configs.mla import (
    resolve_mla_kv_cache_dtype,
)
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.base.transformer_model import BaseTransformerModel
from tokenspeed.runtime.models.dspark import _get_markov_params
from tokenspeed.runtime.utils.hf_transformers_utils import get_config

# --------------------------------------------------------------------------
# Block geometry: verify width vs draft block size
# --------------------------------------------------------------------------


def test_geometry_splits_verify_width_from_draft_count() -> None:
    """spec_num_tokens is the verify width; drafts are one fewer."""
    verify_width, draft_block_size = _resolve_block_geometry(
        SimpleNamespace(), spec_num_tokens=8
    )
    assert (verify_width, draft_block_size) == (8, 7)


def test_dspark_queries_one_row_per_draft_token() -> None:
    assert _resolve_draft_query_width(verify_width=8, sample_from_anchor=True) == 7


def test_dflash_keeps_the_anchor_plus_draft_query_layout() -> None:
    assert _resolve_draft_query_width(verify_width=8, sample_from_anchor=False) == 8


@pytest.mark.parametrize(
    ("algorithm", "is_draft", "expected"),
    (("DSPARK", True, 7), ("DSPARK", False, 8), ("DFLASH", True, 8)),
)
def test_attention_query_width_matches_algorithm(
    algorithm: str, is_draft: bool, expected: int
) -> None:
    args = SimpleNamespace(
        speculative_num_draft_tokens=8,
        speculative_algorithm=algorithm,
    )
    assert resolve_speculative_num_tokens(args, is_draft) == expected


def test_k3_dspark_draft_cache_stays_bf16_when_target_cache_is_fp8() -> None:
    args = SimpleNamespace(kv_cache_dtype="fp8_e4m3", speculative_algorithm="DSPARK")
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="k3_dspark"))
    assert resolve_mla_kv_cache_dtype(args, config, is_draft=True) == torch.bfloat16
    assert (
        resolve_mla_kv_cache_dtype(args, config, is_draft=False) == torch.float8_e4m3fn
    )


def test_other_mla_drafts_keep_the_requested_cache_dtype() -> None:
    args = SimpleNamespace(kv_cache_dtype="fp8_e4m3", speculative_algorithm="DSPARK")
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="other"))
    assert (
        resolve_mla_kv_cache_dtype(args, config, is_draft=True) == torch.float8_e4m3fn
    )


def test_geometry_accepts_torchspec_draft_count_convention() -> None:
    """DSpark/TorchSpec checkpoints store the draft count (7 for K3)."""
    cfg = SimpleNamespace(block_size=7)
    assert _resolve_block_geometry(cfg, spec_num_tokens=8) == (8, 7)


def test_geometry_accepts_legacy_dflash_verify_width_convention() -> None:
    """Older DFlash checkpoints store the verify width instead."""
    cfg = SimpleNamespace(block_size=8)
    assert _resolve_block_geometry(cfg, spec_num_tokens=8) == (8, 7)


def test_geometry_reads_nested_dflash_config() -> None:
    cfg = SimpleNamespace(dflash_config={"block_size": 7})
    assert _resolve_block_geometry(cfg, spec_num_tokens=8) == (8, 7)


def test_geometry_rejects_true_mismatch_with_actionable_message() -> None:
    """A block_size matching neither convention is a launch error, not a warning."""
    cfg = SimpleNamespace(block_size=5)
    with pytest.raises(ValueError) as excinfo:
        _resolve_block_geometry(cfg, spec_num_tokens=8)
    message = str(excinfo.value)
    assert "block_size=5" in message
    # The remedy names the flag and the value that would work.
    assert "--speculative-num-draft-tokens 6" in message


def test_geometry_rejects_degenerate_verify_width() -> None:
    """A verify window with no room for a draft is not block decoding."""
    with pytest.raises(ValueError, match=r">= 2"):
        _resolve_block_geometry(SimpleNamespace(), spec_num_tokens=1)


# --------------------------------------------------------------------------
# Markov head resolution
# --------------------------------------------------------------------------


def test_markov_params_read_top_level_fields() -> None:
    """Inferact/TorchSpec checkpoints declare markov fields at the top level."""
    cfg = SimpleNamespace(markov_rank=256, markov_head_type="vanilla")
    assert _get_markov_params(cfg) == (256, "vanilla")


def test_markov_params_prefer_nested_dspark_config() -> None:
    cfg = SimpleNamespace(
        markov_rank=256,
        dspark_config={"markov_rank": 128, "markov_head_type": "VANILLA"},
    )
    assert _get_markov_params(cfg) == (128, "vanilla")


def test_markov_params_default_to_disabled() -> None:
    assert _get_markov_params(SimpleNamespace()) == (0, "vanilla")


# --------------------------------------------------------------------------
# Algorithm dispatch
# --------------------------------------------------------------------------


def test_dspark_dispatches_to_dspark_drafter() -> None:
    assert get_drafter_impl("DSPARK", SimpleNamespace()) is DSpark


def test_step_acceptance_log_separates_committed_and_draft_tokens() -> None:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.config = SimpleNamespace(global_rank=0, spec_num_steps=7)
    executor.num_generated_tokens = 0
    executor.num_decode_steps = 0
    result = SimpleNamespace(output_lengths=torch.tensor([1, 3, 8]))

    with (
        mock.patch.object(model_executor_module, "LOG_SPEC_ACCEPT_LENGTHS", True),
        mock.patch.object(model_executor_module.logger, "info") as log,
    ):
        executor.accumulate_decode_stats(result, bs=3)

    assert executor.num_generated_tokens == 12
    assert executor.num_decode_steps == 3
    log.assert_called_once_with(
        "Spec verify step. accept_lengths=%s, accepted_draft_tokens=%s",
        [1, 3, 8],
        [0, 2, 7],
    )


def test_step_token_log_aligns_drafts_with_predecessor_target_logits() -> None:
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.config = SimpleNamespace(
        global_rank=0, spec_num_steps=3, spec_num_tokens=4
    )
    executor.num_generated_tokens = 0
    executor.num_decode_steps = 0
    result = SimpleNamespace(
        output_lengths=torch.tensor([3]),
        output_tokens=torch.tensor([11, 12, 99, 100]),
        spec_candidate_tokens=torch.tensor([10, 11, 12, 13]),
    )

    with (
        mock.patch.object(model_executor_module, "LOG_SPEC_ACCEPT_LENGTHS", True),
        mock.patch.object(model_executor_module.logger, "info") as log,
    ):
        executor.accumulate_decode_stats(result, bs=1)

    assert log.call_args_list[1] == mock.call(
        "Spec token compare. anchor=%s, draft=%s, target=%s, match=%s",
        [10],
        [[11, 12, 13]],
        [[11, 12, 99]],
        [[True, True, False]],
    )


def test_dflash_dispatch_is_unchanged_by_dspark() -> None:
    assert get_drafter_impl("DFLASH", SimpleNamespace()) is DFlash


# --------------------------------------------------------------------------
# Draft-worker architecture rewrite
# --------------------------------------------------------------------------


def _write_config(tmp_path, **fields) -> str:
    payload = {
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        **fields,
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))
    return str(tmp_path)


def test_qwen3_dspark_arch_is_rewritten_to_the_entry_class(tmp_path) -> None:
    path = _write_config(tmp_path, architectures=["Qwen3DSparkModel"])
    config = get_config(path, trust_remote_code=False, is_draft_worker=True)
    assert config.architectures[0] == "DSparkDraftModel"


def test_dspark_archs_are_never_suffixed_with_nextn(tmp_path) -> None:
    """The NextN rewrite must not fire for a DSpark draft checkpoint."""
    for arch in ("DSparkDraftModel", "K3DSparkModel"):
        path = _write_config(tmp_path, architectures=[arch])
        config = get_config(path, trust_remote_code=False, is_draft_worker=True)
        assert config.architectures[0] == arch


def test_k3_dspark_config_defaults_to_the_entry_architecture(tmp_path) -> None:
    path = _write_config(tmp_path, model_type="k3_dspark")

    config = get_config(path, trust_remote_code=False, is_draft_worker=True)

    assert config.architectures == ["K3DSparkModel"]


def test_non_spec_archs_still_get_the_nextn_rewrite(tmp_path) -> None:
    path = _write_config(tmp_path, architectures=["Qwen3ForCausalLM"])
    config = get_config(path, trust_remote_code=False, is_draft_worker=True)
    assert config.architectures[0] == "Qwen3ForCausalLMNextN"


# --------------------------------------------------------------------------
# Sliding-window geometry of a windowed DSpark draft
# --------------------------------------------------------------------------


def _write_swa_draft_config(tmp_path, **overrides) -> str:
    """A DSpark draft whose layers are all sliding, as MiniMax-M3's ships."""
    return _write_config(
        tmp_path,
        architectures=["Qwen3DSparkModel"],
        layer_types=["sliding_attention", "sliding_attention"],
        dflash_config={
            "mask_token_id": 127,
            "target_layer_ids": [1, 12],
            "use_swa": True,
            "swa_window_size": 1024,
            "markov_rank": 16,
        },
        **overrides,
    )


@pytest.mark.parametrize("declare_top_level", (True, False))
def test_dspark_draft_keeps_the_window_transformers_would_null(
    tmp_path, declare_top_level: bool
) -> None:
    """``Qwen3Config`` drops ``sliding_window`` unless ``use_sliding_window``.

    DSpark checkpoints never write that flag, so without the restore every
    consumer -- draft layer construction, the draft attention config, the cache
    recipe -- sees None on a sliding draft. The window is read from the raw
    config when present, else from ``dflash_config.swa_window_size``.
    """
    extra = {"sliding_window": 1024} if declare_top_level else {}
    path = _write_swa_draft_config(tmp_path, **extra)

    config = get_config(path, trust_remote_code=False, is_draft_worker=True)

    assert config.sliding_window == 1024


# --------------------------------------------------------------------------
# Draft dtype
# --------------------------------------------------------------------------


def _draft_dtype(server_dtype: str, target_dtype: torch.dtype) -> object:
    from tokenspeed.runtime.engine import event_loop as event_loop_module

    loop = event_loop_module.EventLoop.__new__(event_loop_module.EventLoop)
    loop.server_args = SimpleNamespace(
        dtype=server_dtype,
        quantization=None,
        speculative_draft_model_quantization=None,
        trust_remote_code=True,
        revision=None,
        max_model_len=262144,
        hf_overrides="{}",
    )
    loop.model_config = SimpleNamespace(dtype=target_dtype)

    with mock.patch.object(event_loop_module, "ModelConfig") as model_config:
        loop._load_model_config("draft", is_draft_worker=True)

    return model_config.call_args.kwargs["dtype"]


def test_draft_inherits_the_targets_dtype_instead_of_its_own() -> None:
    """A DSpark draft stored as fp32 master weights must not land on fp16.

    It is fed the target's hidden states and borrows its LM head, so "auto"
    has to resolve against the target; the standalone fp32 -> fp16 rule would
    leave the first GEMM mixing bf16 and fp16 with no kernel.
    """
    assert _draft_dtype("auto", torch.bfloat16) is torch.bfloat16


def test_an_explicit_dtype_still_wins_for_the_draft() -> None:
    assert _draft_dtype("bfloat16", torch.bfloat16) == "bfloat16"


# --------------------------------------------------------------------------
# Draft cache grouping
# --------------------------------------------------------------------------


def _draft_attn_config(algorithm: str, layer_types: tuple[str, ...]):
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    server_args = SimpleNamespace(
        speculative_algorithm=algorithm,
        speculative_num_steps=7,
        speculative_num_draft_tokens=8,
        kv_cache_dtype="fp8_e4m3",
        kv_cache_quant_method="none",
        device="cuda",
        attention_backend="trtllm",
        drafter_attention_backend="trtllm",
        spec_context_pad=0,
        prefix_granularity=128,
        max_num_seqs=16,
        data_parallel_size=None,
        max_cudagraph_capture_size=80,
        chunked_prefill_size=8192,
        disaggregation_mode="null",
        attn_tp_size=4,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_size=4, dp_size=1)),
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(layer_types=layer_types, sliding_window=1024),
        num_attention_layers=len(layer_types),
        context_len=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        dtype=torch.bfloat16,
    )
    return MHAConfig.generate(server_args, model_config, is_draft=True)


def test_block_draft_shares_the_targets_retention() -> None:
    """A DSpark draft's window is a mask, not a cache-retention policy.

    Its KV rows are written at the target's cache locations, so they live and
    die with the target's pages; a sliding cache group of its own would both
    evict rows the target still owns and collide with the target's planes.
    """
    config = _draft_attn_config("DSPARK", ("sliding_attention",) * 6)

    assert config.layer_types == ()
    assert config.sliding_window_tokens is None


def test_a_non_block_draft_keeps_its_own_labels() -> None:
    config = _draft_attn_config("EAGLE3", ("sliding_attention",) * 6)

    assert config.layer_types == ("sliding_attention",) * 6
    assert config.sliding_window_tokens == 1024


# --------------------------------------------------------------------------
# Target-side capture on the base stack
# --------------------------------------------------------------------------


class _CaptureModel:
    """Stand-in for the transformer stack: the capture state, no modules."""

    def __init__(self, num_layers: int) -> None:
        self.layers = [object()] * num_layers
        self.layers_to_capture = []
        self._dflash_capture_idx_map = {}
        self._dflash_incremental_callback = None
        self._dflash_slot_bufs = None
        self._dflash_incr_active = False

    notify = BaseTransformerModel._notify_dflash_capture


class _CaptureCausalLM:
    """Stand-in exposing only the setter under test."""

    def __init__(self, num_layers: int) -> None:
        self.model = _CaptureModel(num_layers)
        self.capture_aux_hidden_states = False

    set_dflash_layers_to_capture = BaseCausalLM.set_dflash_layers_to_capture


def test_taps_shift_by_one_and_sort_for_positional_concat() -> None:
    """MiniMax-M3's DSpark taps, shuffled: they name completed-layer outputs,
    and the draft concatenates the captures in ascending layer order."""
    causal_lm = _CaptureCausalLM(num_layers=60)

    causal_lm.set_dflash_layers_to_capture([57, 1, 35, 12, 46, 23])

    assert causal_lm.model.layers_to_capture == [2, 13, 24, 36, 47, 58]
    assert causal_lm.model._dflash_capture_idx_map == {
        2: 0,
        13: 1,
        24: 2,
        36: 3,
        47: 4,
        58: 5,
    }
    assert causal_lm.capture_aux_hidden_states is True


def test_each_capture_reaches_the_drafter_in_concat_order() -> None:
    slot_bufs = [torch.zeros(4, 3) for _ in range(2)]
    seen: list[tuple[int, int]] = []
    causal_lm = _CaptureCausalLM(num_layers=8)
    causal_lm.set_dflash_layers_to_capture(
        [1, 5],
        incremental_callback=lambda idx, num_tokens: seen.append((idx, num_tokens)),
        slot_bufs=slot_bufs,
    )
    model = causal_lm.model
    model._dflash_incr_active = True

    aux_hidden_states: list[torch.Tensor] = []
    aux_hidden_states.append(torch.ones(2, 3))
    model.notify(2, aux_hidden_states)
    aux_hidden_states.append(torch.full((2, 3), 2.0))
    model.notify(6, aux_hidden_states)

    assert seen == [(0, 2), (1, 2)]
    assert torch.equal(slot_bufs[0][:2], torch.ones(2, 3))
    assert torch.equal(slot_bufs[1][:2], torch.full((2, 3), 2.0))


# --------------------------------------------------------------------------
# CLI contract
# --------------------------------------------------------------------------


def test_cli_accepts_dspark_algorithm() -> None:
    import argparse

    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--speculative-algorithm", "DSPARK", "--speculative-num-draft-tokens", "8"]
    )
    assert args.speculative_algorithm == "DSPARK"
    assert args.speculative_num_draft_tokens == 8
