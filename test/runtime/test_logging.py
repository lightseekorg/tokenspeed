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

import contextlib
import io
import logging
import sys
import threading
import time
from importlib import import_module

import pytest

from tokenspeed._logging import stop_async_logging, suppress_noisy_third_party_logs
from tokenspeed.runtime.utils.common import (
    CustomFormatter,
    configure_logger,
    get_colorful_logger,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


class TestColorfulLogger:
    def test_repeated_construction_emits_each_record_once(self):
        name = "tokenspeed.test_logging.repeated_construction"
        first = get_colorful_logger(name)
        second = get_colorful_logger(name)
        assert (first) is (second)
        colorful = [
            h for h in first.handlers if isinstance(h.formatter, CustomFormatter)
        ]
        assert (len(colorful)) == (1)

        stderr = io.StringIO()
        for handler in colorful:
            handler.setStream(stderr)
        first.info("memory summary")
        assert (stderr.getvalue().count("memory summary")) == (1)


class _BlockingStream(io.StringIO):
    """A stderr replacement whose writes block until the test releases them."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, text):
        self.entered.set()
        self.release.wait(10)
        return super().write(text)


class TestAsyncRootLogger:
    def test_configure_logger_never_blocks_the_logging_thread(self):
        # A stalled stderr target must not stall the thread that logs: the
        # serving loop's records (the profiler lines among them) go through a
        # queue and a listener thread owns the write.
        args = ServerArgs(model="x", world_size=2, ep_size=2, enable_async_logging=True)
        stream = _BlockingStream()
        try:
            with contextlib.redirect_stderr(stream):
                configure_logger(args, prefix=" TEST")
                logger = logging.getLogger("tokenspeed.test_logging.async_root")
                logger.info("first record")
                # The listener took the record and is now stuck in write().
                assert stream.entered.wait(5.0)
                started = time.monotonic()
                logger.info("second record")
                elapsed = time.monotonic() - started
                # The second log call only enqueues; it does not wait for the
                # blocked write.
                assert elapsed < 1.0
                stream.release.set()
                deadline = time.monotonic() + 5.0
                while "second record" not in stream.getvalue():
                    assert time.monotonic() < deadline, "listener did not drain"
                    time.sleep(0.01)
                assert "first record" in stream.getvalue()
        finally:
            stream.release.set()
            stop_async_logging()
            logging.getLogger().handlers.clear()


class TestThirdPartyLogging:
    def test_flash_attn_jit_cache_debug_log_is_suppressed(self):
        try:
            cache_utils = import_module("flash_attn.cute.cache_utils")
        except ImportError:
            pytest.skip("flash_attn.cute.cache_utils is unavailable")

        logging.basicConfig(level=logging.DEBUG, force=True)
        suppress_noisy_third_party_logs()

        logger = logging.getLogger("flash_attn.cute.cache_utils")
        assert (logger.getEffectiveLevel()) >= (logging.WARNING)
        for handler in logger.handlers:
            assert (handler.level) >= (logging.WARNING)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cache_utils.get_jit_cache()

        assert ("Persistent cache disabled") not in (stderr.getvalue())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
