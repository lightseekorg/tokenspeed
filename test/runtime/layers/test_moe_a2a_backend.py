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
"""CPU-only unit tests for the MORI all-to-all backend enum + predicates.

No GPU and no ``mori`` package required — pure Python logic added by the MORI EP
backend (``All2AllBackend.MORI`` / ``is_mori`` / ``is_all_to_all``).
"""

from __future__ import annotations

from tokenspeed.runtime.layers.moe.utils import All2AllBackend


def test_mori_enum_value_and_parse() -> None:
    assert All2AllBackend.MORI.value == "mori"
    assert All2AllBackend("mori") is All2AllBackend.MORI
    # _missing_ maps None -> NONE (unchanged by this backend)
    assert All2AllBackend(None) is All2AllBackend.NONE


def test_is_mori_predicate() -> None:
    assert All2AllBackend.MORI.is_mori()
    assert not All2AllBackend.NONE.is_mori()
    assert not All2AllBackend.DEEPEP.is_mori()


def test_is_all_to_all_predicate() -> None:
    # Real dispatch/combine backends return the COMPLETE per-token result and must
    # take the model's all-to-all forward path.
    assert All2AllBackend.MORI.is_all_to_all()
    assert All2AllBackend.DEEPEP.is_all_to_all()
    # The masked-replicate fallback is NOT all-to-all.
    assert not All2AllBackend.NONE.is_all_to_all()
    assert not All2AllBackend.FLASHINFER_NVLINK_ONE_SIDED.is_all_to_all()


def test_backend_predicates_are_mutually_consistent() -> None:
    all_to_all = {All2AllBackend.DEEPEP, All2AllBackend.MORI}
    for b in All2AllBackend:
        assert b.is_all_to_all() == (b in all_to_all)
        assert b.is_mori() == (b is All2AllBackend.MORI)
        # is_none and is_all_to_all are mutually exclusive
        assert not (b.is_none() and b.is_all_to_all())
