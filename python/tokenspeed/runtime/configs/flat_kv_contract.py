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

"""Import-light constants shared by flat-KV planning and execution.

This module deliberately depends only on Python builtins so configuration
planners share one owner/layer vocabulary without importing model, torch, or
compiled-extension dependencies into standalone tests.
"""

from __future__ import annotations

# Model-side owners in the shared target/draft plan.
CACHE_OWNER_TARGET = 1 << 0
CACHE_OWNER_DRAFT = 1 << 1

# Paged-cache label vocabulary, not the checkpoint's serialized layer enum.
LINEAR_ATTENTION = "linear_attention"
STATE_LAYER_TYPES = frozenset({LINEAR_ATTENTION})

__all__ = [
    "CACHE_OWNER_DRAFT",
    "CACHE_OWNER_TARGET",
    "LINEAR_ATTENTION",
    "STATE_LAYER_TYPES",
]
