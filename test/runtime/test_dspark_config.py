"""Semantic coverage for the DSpark drafter's config and CLI contract.

These are the cheap guards that catch a mis-specified launch before eight GPUs
are committed to loading a 1T-parameter target: block geometry, Markov-head
resolution, algorithm dispatch, and the draft-worker architecture rewrite.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tokenspeed.runtime.execution.drafter.dflash import (
    DFlash,
    _resolve_block_geometry,
)
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.execution.model_executor import _get_drafter_impl
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


@pytest.mark.parametrize("verify_width", [2, 3, 4, 6, 8])
def test_k3_checkpoint_without_block_metadata_allows_width_sweep(
    verify_width: int,
) -> None:
    assert _resolve_block_geometry(SimpleNamespace(), spec_num_tokens=verify_width) == (
        verify_width,
        verify_width - 1,
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
    assert _get_drafter_impl("DSPARK", SimpleNamespace()) is DSpark


def test_dflash_dispatch_is_unchanged_by_dspark() -> None:
    assert _get_drafter_impl("DFLASH", SimpleNamespace()) is DFlash


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
