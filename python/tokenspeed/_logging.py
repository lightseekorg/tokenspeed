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
import logging.handlers
import os
import queue
import threading
import warnings
from importlib import import_module
from typing import TextIO

ASYNC_LOG_QUEUE_CAPACITY = 8192
ASYNC_LOG_SHUTDOWN_TIMEOUT = 5.0


class AsyncStreamHandler(logging.handlers.QueueHandler):
    """A bounded, lossy log queue whose worker owns all stream I/O.

    Args:
        stream: The output stream; it remains owned by the caller.
        capacity: Maximum number of queued records, excluding the active write.
        shutdown_timeout: Seconds to allow queued records to drain on close.
    """

    def __init__(self, stream: TextIO, *, capacity: int, shutdown_timeout: float):
        if capacity <= 0 or shutdown_timeout <= 0:
            raise ValueError("Log capacity and shutdown timeout must be positive")
        super().__init__(queue.Queue(maxsize=capacity))
        self._stream = stream
        self._shutdown_timeout = shutdown_timeout
        self._stopping = threading.Event()
        self.dropped_records = 0
        self._thread = threading.Thread(
            target=self._write_records, name="tokenspeed-log-writer", daemon=True
        )
        self._thread.start()

    def enqueue(self, record: logging.LogRecord) -> None:
        # Handler.handle serializes producers, including this counter. Never
        # use QueueHandler.handleError for a full queue: it writes to stderr
        # on the producer and can recreate the stall we are avoiding.
        if self._stopping.is_set():
            self.dropped_records += 1
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped_records += 1

    def _write_records(self) -> None:
        while not (self._stopping.is_set() and self.queue.empty()):
            try:
                record = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                # QueueHandler.prepare has already formatted and detached the
                # record. Do not use another Handler here: its I/O lock would
                # make logging.shutdown wait indefinitely on a stalled sink.
                self._stream.write(record.getMessage() + "\n")
                self._stream.flush()
            except Exception:
                self.handleError(record)
            finally:
                self.queue.task_done()

    def close(self) -> None:
        self.acquire()
        try:
            if self._stopping.is_set():
                return
            # No producer can enqueue after this point. An event, rather than
            # an in-band sentinel, also works when the bounded queue is full.
            self._stopping.set()
        finally:
            self.release()
        self._thread.join(timeout=self._shutdown_timeout)
        super().close()


_async_log_handler: AsyncStreamHandler | None = None


def stop_async_logging() -> None:
    """Drain the default async logger, allowing at most five seconds.

    The Rust host calls this explicitly: returning from its main function
    does not finalize the embedded interpreter or run Python atexit hooks.
    """
    global _async_log_handler
    handler, _async_log_handler = _async_log_handler, None
    if handler is not None:
        handler.close()


def configure_root_logger(
    *,
    level: int,
    log_format: str,
    datefmt: str,
    stream: TextIO,
    enable_async_logging: bool,
) -> None:
    """Configure the default root output and explicitly select its I/O mode."""
    global _async_log_handler
    stop_async_logging()
    if enable_async_logging:
        handler = AsyncStreamHandler(
            stream,
            capacity=ASYNC_LOG_QUEUE_CAPACITY,
            shutdown_timeout=ASYNC_LOG_SHUTDOWN_TIMEOUT,
        )
        _async_log_handler = handler
    else:
        handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
    logging.basicConfig(level=level, handlers=[handler], force=True)


atexit.register(stop_async_logging)

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
    except Exception:
        return
    finally:
        logging.disable(previous_disable_level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers:
        handler.setLevel(logging.WARNING)


def suppress_noisy_third_party_logs():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TLLM_LOG_LEVEL", "WARNING")
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
