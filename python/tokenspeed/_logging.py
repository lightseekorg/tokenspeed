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

import logging
import os
import warnings
from importlib import import_module

_original_showwarning = warnings.showwarning

_CUTLASS_POINTER_WARNING = "Use explicit `struct.scalar.ptr` for pointer instead."
_CUTLASS_NAMED_BARRIER_WARNING = (
    "NamedBarrier wait also arrives on the barrier. "
    "Routing call to NamedBarrier.arrive_and_wait()."
)


def _is_noisy_cutlass_dsl_warning(message, category) -> bool:
    message_text = str(message)
    return (
        issubclass(category, DeprecationWarning)
        and message_text == _CUTLASS_POINTER_WARNING
    ) or (
        issubclass(category, UserWarning)
        and message_text == _CUTLASS_NAMED_BARRIER_WARNING
    )


def _showwarning(message, category, filename, lineno, file=None, line=None):
    if _is_noisy_cutlass_dsl_warning(message, category):
        return
    _original_showwarning(message, category, filename, lineno, file=file, line=line)


def _suppress_cutlass_dsl_warnings():
    if warnings.showwarning is not _showwarning:
        warnings.showwarning = _showwarning

    warnings.filterwarnings(
        "ignore",
        message=r"Use explicit `struct\.scalar\.ptr` for pointer instead\.",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"NamedBarrier wait also arrives on the barrier\. "
            r"Routing call to NamedBarrier\.arrive_and_wait\(\)\."
        ),
        category=UserWarning,
    )


def _suppress_flash_attn_jit_cache_debug_log():
    logger_name = "flash_attn.cute.cache_utils"
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        import_module(logger_name)
    except ImportError:
        return
    finally:
        logging.disable(previous_disable_level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers:
        handler.setLevel(logging.WARNING)


def suppress_noisy_third_party_logs():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _suppress_cutlass_dsl_warnings()

    for logger_name in (
        "transformers",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "httpx",
        "httpcore",
        "flash_attn.cute.cache_utils",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    _suppress_flash_attn_jit_cache_debug_log()

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
