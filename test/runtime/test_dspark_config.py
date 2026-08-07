"""Semantic coverage for the DSpark drafter's config and CLI contract.

These are the cheap guards that catch a mis-specified launch before eight GPUs
are committed to loading a 1T-parameter target: block geometry, Markov-head
resolution, algorithm dispatch, and the draft-worker architecture rewrite.
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
