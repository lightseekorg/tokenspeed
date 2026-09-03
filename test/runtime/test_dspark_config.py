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

from tokenspeed.runtime.configs.model_config import _apply_block_spec_widths
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.dflash import (
    DFlash,
    _resolve_block_geometry,
    _resolve_draft_query_width,
)
from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.layers.attention.configs.base import (
    SoftmaxAttnConfig,
    resolve_speculative_num_tokens,
)
from tokenspeed.runtime.layers.attention.configs.mla import (
    resolve_mla_kv_cache_dtype,
)
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.base.transformer_model import BaseTransformerModel
from tokenspeed.runtime.models.dspark import _get_markov_params
from tokenspeed.runtime.utils.hf_transformers_utils import get_config
from tokenspeed.runtime.utils.spec_block_geometry import (
    BLOCK_SPEC_RULES,
    read_checkpoint_block_size,
    resolve_block_widths,
    validate_block_widths,
)

# --------------------------------------------------------------------------
# Draft query width and cache dtype
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# The one rule: checkpoint block size -> launch widths
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("algorithm", "block_size", "expected"),
    (
        ("DSPARK", 5, (5, 6)),
        ("DSPARK", 7, (7, 8)),
        ("DFLASH", 8, (7, 8)),
        ("DFLASH", 16, (15, 16)),
    ),
)
def test_block_widths_follow_the_family_convention(
    algorithm: str, block_size: int, expected: tuple[int, int]
) -> None:
    """DSpark stores the draft count; DFlash stores the verify width."""
    assert resolve_block_widths(algorithm, block_size) == expected


def test_block_widths_reject_a_block_that_drafts_nothing() -> None:
    with pytest.raises(ValueError, match=r"drafts nothing"):
        resolve_block_widths("DFLASH", 1)


@pytest.mark.parametrize(
    ("cfg", "expected"),
    (
        (SimpleNamespace(dspark_block_size=5), 5),
        (SimpleNamespace(block_size=5), 5),
        (SimpleNamespace(dflash_config={"block_size": 5}), 5),
        (SimpleNamespace(dspark_config={"block_size": 5}), 5),
        # A nested block size wins over a top-level one.
        (SimpleNamespace(dflash_config={"block_size": 5}, block_size=9), 5),
        # A checkpoint that declares none.
        (SimpleNamespace(), None),
    ),
)
def test_checkpoint_block_size_is_read_from_every_spelling(cfg, expected) -> None:
    assert read_checkpoint_block_size(cfg) == expected


def test_validate_holds_each_family_to_its_own_convention() -> None:
    """The same block_size means different widths per family, and only one."""
    validate_block_widths("DSPARK", 7, num_steps=7, num_draft_tokens=8)
    validate_block_widths("DFLASH", 8, num_steps=7, num_draft_tokens=8)

    with pytest.raises(ValueError) as excinfo:
        validate_block_widths("DSPARK", 8, num_steps=7, num_draft_tokens=8)
    message = str(excinfo.value)
    # The remedy names both flags, their working values, and the rule.
    assert "block_size=8" in message
    assert "--speculative-num-steps 8" in message
    assert "--speculative-num-draft-tokens 9" in message
    assert BLOCK_SPEC_RULES in message


# --------------------------------------------------------------------------
# The drafter resolves its geometry through that rule
# --------------------------------------------------------------------------


def test_geometry_applies_the_family_convention() -> None:
    """spec_num_tokens is the verify width; drafts are one fewer."""
    assert _resolve_block_geometry(SimpleNamespace(), spec_num_tokens=8) == (8, 7)
    assert _resolve_block_geometry(
        SimpleNamespace(block_size=7), spec_num_tokens=8, spec_algorithm="DSPARK"
    ) == (8, 7)
    assert _resolve_block_geometry(SimpleNamespace(block_size=8), 8) == (8, 7)
    assert _resolve_block_geometry(
        SimpleNamespace(dflash_config={"block_size": 8}), 8
    ) == (8, 7)

    with pytest.raises(ValueError, match=r"--speculative-num-steps 6"):
        _resolve_block_geometry(SimpleNamespace(block_size=7), spec_num_tokens=8)


def test_geometry_rejects_degenerate_verify_width() -> None:
    """A verify window with no room for a draft is not block decoding."""
    with pytest.raises(ValueError, match=r">= 2"):
        _resolve_block_geometry(SimpleNamespace(), spec_num_tokens=1)


def test_drafters_declare_their_checkpoint_convention() -> None:
    """DFlash2 launches as DFLASH, so it inherits the DFlash convention."""
    assert DFlash.spec_algorithm == "DFLASH"
    assert DFlash2.spec_algorithm == "DFLASH"
    assert DSpark.spec_algorithm == "DSPARK"


# --------------------------------------------------------------------------
# ModelConfig applies that rule to the launch flags
# --------------------------------------------------------------------------


def _spec_args(algorithm: str, *, steps: int, draft_tokens: int, explicit: bool):
    return SimpleNamespace(
        speculative_algorithm=algorithm,
        speculative_num_steps=steps,
        speculative_num_draft_tokens=draft_tokens,
        _speculative_widths_explicit=explicit,
    )


def test_model_config_sets_defaulted_widths_from_the_checkpoint() -> None:
    args = _spec_args("DSPARK", steps=3, draft_tokens=4, explicit=False)
    cfg = SimpleNamespace(block_size=7)

    assert _apply_block_spec_widths(args, cfg, cfg) == 7
    assert (args.speculative_num_steps, args.speculative_num_draft_tokens) == (7, 8)


def test_model_config_checks_explicit_widths_against_the_checkpoint() -> None:
    """A DFlash block_size of 8 is 7 steps; 8 steps is the DSpark reading."""
    cfg = SimpleNamespace(dflash_config={"block_size": 8})

    args = _spec_args("DFLASH", steps=7, draft_tokens=8, explicit=True)
    assert _apply_block_spec_widths(args, cfg, cfg) == 8
    assert (args.speculative_num_steps, args.speculative_num_draft_tokens) == (7, 8)

    with pytest.raises(ValueError, match=r"--speculative-num-steps 7"):
        _apply_block_spec_widths(
            _spec_args("DFLASH", steps=8, draft_tokens=9, explicit=True), cfg, cfg
        )


@pytest.mark.parametrize(
    ("algorithm", "cfg"),
    (
        # Nothing to check against.
        ("DSPARK", SimpleNamespace()),
        # Not a block drafter.
        ("EAGLE3", SimpleNamespace(block_size=8)),
    ),
)
def test_model_config_leaves_widths_alone(algorithm: str, cfg) -> None:
    args = _spec_args(algorithm, steps=7, draft_tokens=8, explicit=True)

    assert _apply_block_spec_widths(args, cfg, cfg) is None
    assert (args.speculative_num_steps, args.speculative_num_draft_tokens) == (7, 8)


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
    spec = _draft_attn_config("DSPARK", ("sliding_attention",) * 6).component(
        SoftmaxAttnConfig
    )

    assert spec.layer_types == ()
    assert spec.sliding_window_tokens is None


def test_a_non_block_draft_keeps_its_own_labels() -> None:
    spec = _draft_attn_config("EAGLE3", ("sliding_attention",) * 6).component(
        SoftmaxAttnConfig
    )

    assert spec.layer_types == ("sliding_attention",) * 6
    assert spec.sliding_window_tokens == 1024


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
