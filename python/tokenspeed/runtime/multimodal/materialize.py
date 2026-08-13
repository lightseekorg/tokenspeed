# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Materialize gateway-precomputed multimodal inputs for the scheduler.

An upstream preprocessor ships ``MultimodalInputs`` whose items carry
preprocessed features, content hashes, and placeholder ``offsets``, with the
prompt's ``input_ids`` containing expanded placeholder tokens at those
offsets. Before the scheduler can admit such a request, three derived pieces
must exist, in this order:

1. per-item ``pad_value``s (hash-derived ids in the >1M interval space),
2. M-RoPE positions, computed on the UN-padded ids (``get_rope_index`` must
   still see the placeholder tokens to locate the image regions) — skipped
   when the payload already carries them,
3. ``pad_input_tokens``: placeholder runs rewritten to each item's
   ``pad_value`` so distinct images prefix-compare unequal in the text-only
   prefix cache, keeping the original ids as ``input_ids_unpadded`` for
   detokenization.

This is one function so every admission path derives them identically: the
in-process frontend (``InputProcessor.tokenize_one_request``) and the direct
msgpack ZMQ ingest, which bypasses the frontend entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenspeed.runtime.multimodal.embedder import pad_input_tokens
from tokenspeed.runtime.multimodal.mrope import compute_mrope_positions

if TYPE_CHECKING:
    from tokenspeed.runtime.multimodal.inputs import MultimodalInputs


def materialize_precomputed_inputs(
    hf_config,
    input_ids: list[int],
    multimodal_inputs: "MultimodalInputs",
) -> tuple[list[int], list[int]]:
    """Derive pad values, M-RoPE positions, and padded ids in place.

    Mutates ``multimodal_inputs`` (pad values, mrope fields) and returns
    ``(padded_input_ids, input_ids_unpadded)``.
    """
    multimodal_inputs.ensure_pad_values()

    # MRoPE-aware models (Qwen-VL family, ...) require 3-axis position ids
    # derived from grid metadata plus the placeholder positions. A payload may
    # precompute them upstream (any mrope field set means "upstream owns it");
    # left entirely None they would silently degrade to 1-D linear positions.
    if (
        multimodal_inputs.mrope_positions is None
        and multimodal_inputs.mrope_position_delta is None
        and multimodal_inputs.mrope_position_delta_scalar is None
    ):
        mrope_positions, mrope_position_delta = compute_mrope_positions(
            hf_config,
            input_ids,
            multimodal_inputs.mm_items,
        )
        multimodal_inputs.mrope_positions = mrope_positions
        multimodal_inputs.mrope_position_delta = mrope_position_delta
        if mrope_position_delta is not None:
            multimodal_inputs.mrope_position_delta_scalar = int(
                mrope_position_delta.flatten()[0].item()
            )

    padded = pad_input_tokens(input_ids, multimodal_inputs)
    return padded, input_ids
