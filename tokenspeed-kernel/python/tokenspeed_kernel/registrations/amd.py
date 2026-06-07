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

from __future__ import annotations

import importlib
import logging

from tokenspeed_kernel.registrations import apply_vendor_registrations

logger = logging.getLogger(__name__)

_MODULES = (
    "tokenspeed_kernel_amd.attention.gluon",
    "tokenspeed_kernel_amd.moe.gluon",
)


def load() -> None:
    try:
        registration = importlib.import_module("tokenspeed_kernel_amd.registration")
    except ImportError as exc:
        logger.debug("Skipping AMD registrations: %s", exc)
        return

    registrations = registration.REGISTRATIONS
    registrations.clear()
    for module_name in _MODULES:
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)
        except ImportError as exc:
            logger.debug(
                "Skipping built-in registration module %s: %s", module_name, exc
            )
    apply_vendor_registrations(registrations)
