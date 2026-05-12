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
import importlib
import logging

from tokenspeed_kernel.platform import current_platform

platform = current_platform()
logger = logging.getLogger(__name__)


def _is_dependency_import_error(exc: ImportError, dependency: str) -> bool:
    return getattr(exc, "name", None) == dependency or dependency in str(exc)


def _try_import_optional_backend(name: str, dependency: str) -> None:
    try:
        importlib.import_module(f"tokenspeed.runtime.layers.attention.backends.{name}")
    except ImportError as exc:
        if not _is_dependency_import_error(exc, dependency):
            raise
        logger.warning(
            "Skipping optional attention backend %s because package %s is missing",
            name,
            dependency,
        )


if platform.is_nvidia:
    from tokenspeed.runtime.layers.attention.backends import deepseek_v4  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends import flashmla  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends import (
        tokenspeed_mla,
    )  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends import trtllm  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends import trtllm_mla  # noqa: F401

from tokenspeed.runtime.layers.attention.backends import mha  # noqa: F401
