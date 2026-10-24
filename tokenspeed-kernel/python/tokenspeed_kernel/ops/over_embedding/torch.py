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

"""Portable PyTorch projection and word merge for LongCat OE."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "over_embedding",
    "project_add_word",
    name="torch_longcat_oe_project_add_word",
    solution="torch",
    signatures=format_signatures(
        ("word", "activation", "projection"), "dense", {torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
)
def torch_longcat_oe_project_add_word(
    *,
    word_partial: torch.Tensor,
    activation: torch.Tensor,
    projection: torch.Tensor,
    scale: int,
) -> None:
    """Merge scaled word and local OE projection into word storage."""
    inverse_scale = 1.0 / scale
    word_partial.addmm_(
        activation,
        projection,
        beta=inverse_scale,
        alpha=inverse_scale,
    )
