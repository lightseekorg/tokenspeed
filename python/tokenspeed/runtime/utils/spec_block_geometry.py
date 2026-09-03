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

from typing import Any

__all__ = [
    "BLOCK_SPEC_ALGORITHMS",
    "BLOCK_SPEC_RULES",
    "read_checkpoint_block_size",
    "resolve_block_widths",
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
) -> None:
    """Reject launch widths that disagree with the checkpoint block size.

    Args:
        algorithm: ``"DSPARK"`` or ``"DFLASH"``.
        block_size: The checkpoint's block size.
        num_steps: The requested ``speculative_num_steps``.
        num_draft_tokens: The requested ``speculative_num_draft_tokens``.

    Raises:
        ValueError: The requested widths are not the ones the checkpoint was
            trained at.
    """
    expected_steps, expected_draft_tokens = resolve_block_widths(algorithm, block_size)
    if (
        int(num_steps) == expected_steps
        and int(num_draft_tokens) == expected_draft_tokens
    ):
        return
    raise ValueError(
        f"{algorithm} checkpoint block_size={int(block_size)} requires "
        f"--speculative-num-steps {expected_steps} and "
        f"--speculative-num-draft-tokens {expected_draft_tokens}; got "
        f"--speculative-num-steps {int(num_steps)} and "
        f"--speculative-num-draft-tokens {int(num_draft_tokens)}. "
        f"{BLOCK_SPEC_RULES}"
    )
