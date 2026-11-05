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

import sys

from tokenspeed_kernel._optional import has_optional_package

_MISSING = "tokenspeed_no_such_package"


def test_present_package():
    assert has_optional_package("torch")


def test_missing_package():
    assert not has_optional_package(_MISSING)


def test_one_miss_fails_the_probe():
    assert not has_optional_package("torch", _MISSING)


def test_dotted_probe_short_circuits_at_a_missing_parent():
    # Probing the submodule of an absent parent would raise; the parent probe
    # must answer first.
    assert not has_optional_package(_MISSING, f"{_MISSING}.sub")


def test_blocked_name_counts_as_missing(monkeypatch):
    # test_optional_imports blocks packages this way in a fresh interpreter.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert not has_optional_package("torch")
