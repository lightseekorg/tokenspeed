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

"""Block geometry shared by the DSpark and DFlash speculative families.

A block drafter's width is fixed by the checkpoint it was trained at. DSpark
stores the drafted-token count; DFlash/DFlash2 store the verify width, which is
the drafts plus the anchor row.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

__all__ = [
    "BLOCK_SPEC_ALGORITHMS",
    "BLOCK_SPEC_RULES",
    "count_dspark_stages",
    "read_checkpoint_block_size",
    "resolve_block_sample_from_anchor",
    "resolve_block_widths",
    "resolve_draft_query_width",
    "validate_block_widths",
]

#: Speculative algorithms whose widths come from a checkpoint block size.
BLOCK_SPEC_ALGORITHMS = ("DFLASH", "DSPARK")

BLOCK_SPEC_RULES = (
    "DSpark checkpoints store block_size (dspark_block_size) == "
    "--speculative-num-steps; DFlash/DFlash2 checkpoints store block_size == "
    "--speculative-num-steps + 1. --speculative-num-draft-tokens is "
    "--speculative-num-steps + 1 for both."
)

_BLOCK_SIZE_KEYS = ("dspark_block_size", "block_size")
_NESTED_CONFIG_KEYS = ("dflash_config", "dspark_config")
#: Rows the checkpoint's block_size counts beyond the drafted tokens.
_STEP_OFFSET = {"DSPARK": 0, "DFLASH": 1}
_SAMPLE_FROM_ANCHOR = {"DSPARK": True, "DFLASH": False}


def resolve_block_sample_from_anchor(algorithm: str, *configs: Any) -> bool:
    """Resolve the draft's query-to-proposal layout from checkpoint metadata.

    Args:
        algorithm: Block drafting algorithm, used for older checkpoints.
        *configs: Draft configs, most specific first.

    Returns:
        Whether the anchor query predicts the first proposed token.
    """
    for config in configs:
        value = getattr(config, "sample_from_anchor", None)
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError("sample_from_anchor must be a boolean")
            return value
    return _SAMPLE_FROM_ANCHOR[algorithm]


def resolve_draft_query_width(verify_width: int, sample_from_anchor: bool) -> int:
    """Return query rows per request for a block draft.

    Args:
        verify_width: Anchor plus proposed tokens, at least two.
        sample_from_anchor: Whether row zero predicts the first proposal.

    Returns:
        Query rows, including the anchor when it has no proposal output.
    """
    if verify_width < 2:
        raise ValueError("Block drafting requires a verify width of at least two")
    return verify_width - 1 if sample_from_anchor else verify_width


def count_dspark_stages(
    model_path: str,
    revision: str | None = None,
) -> int | None:
    """Count contiguous ``mtp.<stage>`` namespaces in a safetensors index."""

    index_filename = "model.safetensors.index.json"
    if os.path.isdir(model_path):
        index_path = os.path.join(model_path, index_filename)
    else:
        try:
            index_path = hf_hub_download(
                repo_id=model_path,
                filename=index_filename,
                revision=revision,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed below
            logger.debug(
                f"Unable to resolve DSpark safetensors index for {model_path!s}: "
                f"{exc!s}",
            )
            return None
    if not os.path.isfile(index_path):
        return None
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map", {})
    stages = {
        int(match.group(1))
        for name in weight_map
        if (match := re.match(r"^mtp\.(\d+)\.(.+)$", name))
    }
    if not stages:
        return None
    expected = set(range(max(stages) + 1))
    if stages != expected:
        raise ValueError(
            "DSpark checkpoint stages must be contiguous from zero; "
            f"found {sorted(stages)}."
        )
    return len(stages)


def read_checkpoint_block_size(*configs: Any) -> int | None:
    """Read the trained block size off draft checkpoint configs.

    Args:
        *configs: Checkpoint configs to search, most specific first. Both the
            nested ``dflash_config`` / ``dspark_config`` dicts and the
            top-level ``dspark_block_size`` / ``block_size`` attributes are
            accepted, in that order.

    Returns:
        The checkpoint's block size, or None when no config declares one.
    """
    for config in configs:
        if config is None:
            continue
        for nested_key in _NESTED_CONFIG_KEYS:
            nested = getattr(config, nested_key, None)
            if not isinstance(nested, dict):
                continue
            for key in _BLOCK_SIZE_KEYS:
                value = nested.get(key)
                if value is not None:
                    return int(value)
        for key in _BLOCK_SIZE_KEYS:
            value = getattr(config, key, None)
            if value is not None:
                return int(value)
    return None


def resolve_block_widths(algorithm: str, block_size: int) -> tuple[int, int]:
    """Derive the launch widths a checkpoint block size implies.

    Args:
        algorithm: ``"DSPARK"`` or ``"DFLASH"`` (DFlash2 launches as DFLASH).
        block_size: The checkpoint's block size.

    Returns:
        ``(speculative_num_steps, speculative_num_draft_tokens)``.

    Raises:
        KeyError: The algorithm is not a block drafter.
        ValueError: The block size leaves no room for a drafted token.
    """
    num_steps = int(block_size) - _STEP_OFFSET[algorithm]
    if num_steps < 1:
        raise ValueError(
            f"{algorithm} checkpoint block_size={int(block_size)} implies "
            f"--speculative-num-steps {num_steps}, which drafts nothing. "
            f"{BLOCK_SPEC_RULES}"
        )
    return num_steps, num_steps + 1


def validate_block_widths(
    algorithm: str,
    block_size: int,
    num_steps: int,
    num_draft_tokens: int,
    *,
    allow_shorter: bool = False,
) -> None:
    """Reject launch widths that disagree with the checkpoint block size.

    Args:
        algorithm: ``"DSPARK"`` or ``"DFLASH"``.
        block_size: The checkpoint's block size.
        num_steps: The requested ``speculative_num_steps``.
        num_draft_tokens: The requested ``speculative_num_draft_tokens``.
        allow_shorter: Also accept fewer steps than the checkpoint drafts, for
            drafters that size their block from the launch width.

    Raises:
        ValueError: The requested widths are not the ones the checkpoint was
            trained at, or with ``allow_shorter`` draft more than it does.
    """
    expected_steps, expected_draft_tokens = resolve_block_widths(algorithm, block_size)
    if (
        int(num_steps) == expected_steps
        and int(num_draft_tokens) == expected_draft_tokens
    ):
        return
    if (
        allow_shorter
        and 1 <= int(num_steps) < expected_steps
        and int(num_draft_tokens) == int(num_steps) + 1
    ):
        return
    if allow_shorter:
        raise ValueError(
            f"{algorithm} checkpoint block_size={int(block_size)} drafts at most "
            f"--speculative-num-steps {expected_steps}, with "
            "--speculative-num-draft-tokens one more than the steps; got "
            f"--speculative-num-steps {int(num_steps)} and "
            f"--speculative-num-draft-tokens {int(num_draft_tokens)}. "
            f"{BLOCK_SPEC_RULES}"
        )
    raise ValueError(
        f"{algorithm} checkpoint block_size={int(block_size)} requires "
        f"--speculative-num-steps {expected_steps} and "
        f"--speculative-num-draft-tokens {expected_draft_tokens}; got "
        f"--speculative-num-steps {int(num_steps)} and "
        f"--speculative-num-draft-tokens {int(num_draft_tokens)}. "
        f"{BLOCK_SPEC_RULES}"
    )
