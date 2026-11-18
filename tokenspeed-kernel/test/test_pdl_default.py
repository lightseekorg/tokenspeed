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

"""A missing or None ``enable_pdl`` takes the process PDL default."""

import sys

import pytest
from tokenspeed_kernel.platform import pdl_enabled, resolve_enable_pdl, with_pdl_default


def _launch(x, enable_pdl=None, *, tag="t"):
    return x, enable_pdl, tag


launch = with_pdl_default(_launch)


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_enable_pdl_passes_through(explicit: bool) -> None:
    assert launch(1, explicit, tag="a") == (1, explicit, "a")
    assert launch(1, enable_pdl=explicit) == (1, explicit, "t")
    assert resolve_enable_pdl(explicit) is explicit


def test_missing_or_none_enable_pdl_takes_the_process_default() -> None:
    default = pdl_enabled()
    assert launch(2) == (2, default, "t")
    assert launch(2, None) == (2, default, "t")
    assert launch(2, enable_pdl=None, tag="b") == (2, default, "b")
    assert resolve_enable_pdl(None) is default


def test_wrapper_keeps_the_wrapped_name() -> None:
    assert launch.__name__ == "_launch" and launch.__wrapped__ is _launch


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
