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

# ruff: noqa: E402,F401
# Import all backend modules to trigger register_backend() calls.
import logging
import os

from tokenspeed_kernel.platform import current_platform

platform = current_platform()
logger = logging.getLogger(__name__)


# DIAGNOSTIC (DeepSeek V4 AMD port): the V4 backend still calls FlashMLA
# directly, so registering it here does not make V4 work on AMD -- it moves the
# failure from an "unknown attention backend" ValueError at startup into the
# first forward, where the actual unsupported call site can be observed. Revert
# this, or make it conditional on real kernel availability, before merging.
_TOKENSPEED_FORCE_V4_BACKEND = (
    os.environ.get("TOKENSPEED_FORCE_DEEPSEEK_V4_BACKEND", "0") == "1"
)

if platform.is_nvidia or _TOKENSPEED_FORCE_V4_BACKEND:
    from tokenspeed.runtime.layers.attention.backends import deepseek_v4  # noqa: F401

if platform.is_nvidia:
    from tokenspeed.runtime.layers.attention.backends import flashmla  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends import trtllm  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends import trtllm_mla  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends import (  # noqa: F401
        tokenspeed_mla,
    )

from tokenspeed.runtime.layers.attention.backends import dsa  # noqa: F401
from tokenspeed.runtime.layers.attention.backends import mha  # noqa: F401
from tokenspeed.runtime.layers.attention.backends import mla  # noqa: F401
from tokenspeed.runtime.layers.attention.backends import msa  # noqa: F401
