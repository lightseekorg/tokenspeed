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
import unittest
from importlib import import_module

from tokenspeed._logging import suppress_noisy_third_party_logs


class TestThirdPartyLogging(unittest.TestCase):
    def test_flash_attn_jit_cache_debug_log_is_suppressed(self):
        try:
            cache_utils = import_module("flash_attn.cute.cache_utils")
        except ImportError:
            self.skipTest("flash_attn.cute.cache_utils is unavailable")

        logging.basicConfig(level=logging.DEBUG, force=True)
        suppress_noisy_third_party_logs()

        logger = logging.getLogger("flash_attn.cute.cache_utils")
        self.assertGreaterEqual(logger.getEffectiveLevel(), logging.WARNING)
        for handler in logger.handlers:
            self.assertGreaterEqual(handler.level, logging.WARNING)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            cache_utils.get_jit_cache()

        self.assertNotIn("Persistent cache disabled", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
