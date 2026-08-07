"""Shared Inkling test config: the released hub checkpoints, truncated in-memory.

Tests load the real config/tokenizer from the released snapshots (weightless
``snapshot_download`` via ``get_config``) and shrink only explicit size knobs;
kernel-relevant geometry stays the released values. ``NUM_TEST_LAYERS = 6`` is
the minimal prefix covering every layer kind: dense MLP (0-1), SWA MoE (2-4),
full-attention MoE (5).
"""

from __future__ import annotations

import json

INKLING_BF16 = "thinkingmachines/Inkling"
INKLING_NVFP4 = "thinkingmachines/Inkling-NVFP4"
NUM_TEST_LAYERS = 6


def has_blackwell() -> bool:
    """True on an NVIDIA Blackwell GPU (the Inkling serving target); the
    in-process counterpart of the CI-level ``disabled_on_runners`` gates,
    which only apply under ``run_ci_suite``."""
    import torch

    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


def truncate_text_config(text, num_layers: int = NUM_TEST_LAYERS):
    """Truncate a loaded Inkling text config to its first ``num_layers`` layers.

    Args:
        text: The Inkling text config to modify in place.
        num_layers: Number of leading decoder layers to keep.

    Returns:
        The same config object, truncated.
    """
    text.num_hidden_layers = num_layers
    text.local_layer_ids = [i for i in text.local_layer_ids if i < num_layers]
    return text


NUM_TEST_EXPERTS = 8


def shrink_routed_experts(text, num_experts: int = NUM_TEST_EXPERTS):
    """Shrink the routed expert count in place — a pure size knob for tests
    that materialize weights; returns the same config object."""
    assert num_experts >= text.num_experts_per_tok
    text.n_routed_experts = num_experts
    text.num_experts = num_experts
    return text


def load_inkling_config(
    model_id: str = INKLING_NVFP4,
    num_layers: int | None = NUM_TEST_LAYERS,
    num_experts: int | None = None,
):
    """Load a released Inkling hub config, optionally truncated/shrunk.

    Args:
        model_id: Hub id of a released Inkling checkpoint.
        num_layers: Truncate to this many leading layers; None keeps all.
        num_experts: Shrink the routed expert count (for tests that
            materialize weights); None keeps the released count.

    Returns:
        The ``InklingMMConfig`` for the snapshot.
    """
    from tokenspeed.runtime.configs.utils import get_config

    cfg = get_config(model_id, revision=None)
    if num_layers is not None:
        truncate_text_config(cfg.get_text_config(), num_layers)
    if num_experts is not None:
        shrink_routed_experts(cfg.get_text_config(), num_experts)
    return cfg


def truncation_hf_overrides(
    num_layers: int = NUM_TEST_LAYERS, num_experts: int | None = NUM_TEST_EXPERTS
) -> str:
    """The engine-side equivalent of :func:`load_inkling_config`'s surgery.

    Returns:
        A ``--hf-overrides`` JSON string applying the same layer truncation
        (and expert shrink) — ``get_config`` flat-updates the text config
        with these keys.
    """
    cfg = load_inkling_config(num_layers=None)
    text = cfg.get_text_config()
    overrides = {
        "num_hidden_layers": num_layers,
        "local_layer_ids": [i for i in text.local_layer_ids if i < num_layers],
    }
    if num_experts is not None:
        overrides["n_routed_experts"] = num_experts
        overrides["num_experts"] = num_experts
    return json.dumps(overrides)
