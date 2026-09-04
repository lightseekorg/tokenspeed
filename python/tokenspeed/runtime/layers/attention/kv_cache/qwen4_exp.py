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

"""Qwen4-Exp cache group ids, field ids and rows-per-page constants.

This module is deliberately weight-free. Cache recipes, attention backends,
and model layers share this contract without importing model code or QSA
kernels as a side effect.
"""

QWEN4_EXP_PLE_CACHE_GROUP = "qwen4_exp_ple"

QWEN4_EXP_QSA_CACHE_GROUP = "qwen4_exp_qsa"
QWEN4_EXP_QSA_RECENT_CACHE_GROUP = "qwen4_exp_qsa_recent"
QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE = 64
QWEN4_EXP_QSA_RECENT_ROWS_PER_PAGE = 64


def qwen4_exp_ple_context_field(layer_id: int) -> str:
    """Return the shared PLE context field owned by its first consumer."""

    return f"layer.{layer_id}.qwen4_exp.ple.context"


def qwen4_exp_ple_conv_field(layer_id: int) -> str:
    """Return the cache field id for one PLE short-convolution state."""

    return f"layer.{layer_id}.qwen4_exp.ple.conv"


def qsa_raw_key_field(layer_id: int) -> str:
    """Return the cache field id for one QSA recent raw-key window."""

    return f"layer.{layer_id}.qsa.raw_key"


def qsa_compressed_field(layer_id: int) -> str:
    """Return the cache field id for one QSA compressed-key history."""

    return f"layer.{layer_id}.qsa.compressed_key"


def qsa_rope_position_field(layer_id: int) -> str:
    """Return the cache field id for one QSA recent RoPE position."""

    return f"layer.{layer_id}.qsa.rope_position"


__all__ = [
    "QWEN4_EXP_PLE_CACHE_GROUP",
    "QWEN4_EXP_QSA_CACHE_GROUP",
    "QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE",
    "QWEN4_EXP_QSA_RECENT_CACHE_GROUP",
    "QWEN4_EXP_QSA_RECENT_ROWS_PER_PAGE",
    "qsa_compressed_field",
    "qsa_raw_key_field",
    "qsa_rope_position_field",
    "qwen4_exp_ple_context_field",
    "qwen4_exp_ple_conv_field",
]
