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

import atexit
import logging
import os
import re
import threading
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


def _suppress_misc_runtime_warnings():
    # tvm-ffi: every tilelang import re-warns about subclass fields that
    # shadow ancestor fields. The Python path (PyErr_WarnEx from
    # tvm_ffi/registry.py) is muted here; the sibling C++ std::cerr path
    # from libtvm_ffi.so::object.cc is handled by the FD 2 line filter.
    warnings.filterwarnings(
        "ignore",
        message=r"Field .* duplicates an ancestor field.*",
        category=UserWarning,
    )
    # torch: HF tokenizer / dataset paths convert non-writable numpy
    # arrays. Self-rate-limited but each TP rank still emits it once.
    warnings.filterwarnings(
        "ignore",
        message=r"The given NumPy array is not writable.*",
        category=UserWarning,
    )
    # torch.distributed.barrier: device_id is set later in our init
    # path, but barrier() runs first and nags every rank.
    warnings.filterwarnings(
        "ignore",
        message=r"barrier\(\): using the device under current context\..*",
        category=UserWarning,
    )
    # transformers: some chat templates only ship a slow tokenizer; the
    # nag offers no actionable advice.
    warnings.filterwarnings(
        "ignore",
        message=r"Using a slow tokenizer\..*",
        category=UserWarning,
    )


_TVM_FFI_CERR_PATTERN = re.compile(
    r'\[WARNING\] Field ".*?" in type ".*?" duplicates an ancestor field'
)

_stderr_filter_installed = False


def _stderr_line_should_drop(line: bytes) -> bool:
    try:
        text = line.decode("utf-8", errors="replace")
    except Exception:
        return False
    return _TVM_FFI_CERR_PATTERN.search(text) is not None


def _stderr_filter_reader(read_fd: int, forward_fd: int) -> None:
    buf = b""
    try:
        with os.fdopen(read_fd, "rb", buffering=0) as pipe:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    idx = buf.find(b"\n")
                    if idx == -1:
                        break
                    line, buf = buf[: idx + 1], buf[idx + 1 :]
                    if _stderr_line_should_drop(line):
                        continue
                    try:
                        os.write(forward_fd, line)
                    except OSError:
                        pass
    except Exception:
        pass
    finally:
        if buf and not _stderr_line_should_drop(buf):
            try:
                os.write(forward_fd, buf)
            except OSError:
                pass


def _install_tvm_ffi_cerr_filter() -> None:
    # tvm-ffi's libtvm_ffi.so::object.cc emits "[WARNING] Field ... duplicates
    # an ancestor field ..." directly to std::cerr (FD 2), bypassing Python's
    # warnings system. Splice a line-filter onto FD 2 so these lines are
    # dropped before reaching the inherited stderr.
    global _stderr_filter_installed
    if _stderr_filter_installed:
        return

    try:
        saved_stderr_fd = os.dup(2)
    except OSError:
        return

    try:
        read_fd, write_fd = os.pipe()
    except OSError:
        os.close(saved_stderr_fd)
        return

    try:
        os.dup2(write_fd, 2)
    except OSError:
        os.close(saved_stderr_fd)
        os.close(read_fd)
        os.close(write_fd)
        return
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass

    _stderr_filter_installed = True

    threading.Thread(
        target=_stderr_filter_reader,
        args=(read_fd, saved_stderr_fd),
        name="tokenspeed-stderr-filter",
        daemon=True,
    ).start()

    def _restore() -> None:
        try:
            os.dup2(saved_stderr_fd, 2)
        except OSError:
            pass

    atexit.register(_restore)


def suppress_noisy_third_party_logs():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TLLM_LOG_LEVEL", "WARNING")
    _suppress_cutlass_dsl_warnings()
    _suppress_misc_runtime_warnings()
    _install_tvm_ffi_cerr_filter()

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
