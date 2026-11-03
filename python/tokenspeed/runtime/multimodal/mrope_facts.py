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

"""The checkpoint facts a frontend computes multimodal items' MRoPE positions
from (``mrope.py`` computes them in the engine): read from the config alone,
so the wire module that reports them imports no model code.
"""

from __future__ import annotations


def _config_section(config, name: str):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _optional_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _rope_is_mrope(hf_config) -> bool | None:
    """Whether the language model's rotary positions are MRoPE: a rope config
    (``rope_scaling`` or ``rope_parameters``, on the text config or the top
    level) with an ``mrope_section`` or an ``mrope`` type. None when the
    checkpoint has no rope config at all."""
    sections = [hf_config]
    text = _config_section(hf_config, "text_config")
    if text is not None:
        sections.insert(0, text)
    found = False
    for section in sections:
        for key in ("rope_scaling", "rope_parameters"):
            rope = _config_section(section, key)
            if rope is None:
                continue
            found = True
            if _config_section(rope, "mrope_section") is not None:
                return True
            kind = _config_section(rope, "rope_type") or _config_section(rope, "type")
            if isinstance(kind, str) and "mrope" in kind:
                return True
    return False if found else None


def mrope_facts(hf_config) -> dict:
    """The checkpoint facts a frontend computes a multimodal request's MRoPE
    positions from, since it sends the items without them: whether the
    rotary positions are MRoPE, and the vision tower's ``spatial_merge_size``
    and the image and video placeholder ids the position rule reads.

    The engine reports them in its ready response (``zmq_wire``); SMG's
    gateway runs the rule for the Qwen3.5 lineage and refuses an item whose
    positions it cannot compute.
    """
    vision = _config_section(hf_config, "vision_config")
    return {
        "mrope_positions": _rope_is_mrope(hf_config),
        "spatial_merge_size": _optional_int(
            _config_section(vision, "spatial_merge_size")
            if vision is not None
            else None
        ),
        "image_token_id": _optional_int(_config_section(hf_config, "image_token_id")),
        "video_token_id": _optional_int(_config_section(hf_config, "video_token_id")),
    }
