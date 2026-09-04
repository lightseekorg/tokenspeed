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

"""Logical context_len vs physical KV extent (spec-decode overshoot pad).

Spec verify on the overlap scheduler commits past the logical context_len for
a finished request that lingers exactly one extra step. The physical extent
(context_len + ServerArgs.spec_context_pad) absorbs that overshoot so no
per-request table needs a runtime clamp; these tests pin the pad math and the
tripwire that guards the one-lingering-step assumption.
"""

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.utils.server_args import _SPEC_OVERSHOOT_SPANS, ServerArgs

spec_context_pad = ServerArgs.spec_context_pad.fget


def _args(spec_algo, spec_num_tokens=None):
    return SimpleNamespace(
        speculative_algorithm=spec_algo,
        speculative_num_draft_tokens=spec_num_tokens,
    )


class TestSpecContextPad:
    def test_zero_without_spec(self):
        assert spec_context_pad(_args(None)) == 0

    @pytest.mark.parametrize("algo", ["DFLASH", "DSPARK", "EAGLE3", "MTP"])
    def test_three_spans_for_every_spec_algo(self, algo):
        assert spec_context_pad(_args(algo, 4)) == _SPEC_OVERSHOOT_SPANS * 4

    def test_spans_cover_the_overshoot_bound(self):
        # The overlap loop lets a finished request run ONE extra step. The
        # worst case needs three spec_num_tokens spans past context_len:
        # the finishing step's accept, the lingering step's accept, and the
        # lingering step's next draft block. If the constant ever shrinks
        # below 3 the physical extent no longer covers the overshoot.
        assert _SPEC_OVERSHOOT_SPANS >= 3


class TestDraftTableWidth:
    @pytest.mark.parametrize(
        "context_len,spec,block_granularity",
        [(4096, 4, 64), (4096, 4, 1), (131072, 8, 64), (7, 2, 4)],
    )
    def test_physical_extent_fits_the_table(self, context_len, spec, block_granularity):
        # Mirrors the model_executor sizing: table width derives from the
        # physical extent alone, no per-algorithm slack.
        physical = context_len + _SPEC_OVERSHOOT_SPANS * spec
        width = (physical + block_granularity - 1) // block_granularity
        assert width * block_granularity >= physical
        # No regression vs the old formula's non-slack part
        # (context_len + spec_num_tokens).
        assert width * block_granularity >= context_len + spec


class TestOutputProcessorTripwire:
    def _processor(self, physical_context_len):
        from tokenspeed.runtime.engine.generation_output_processor import (
            OutputProcesser,
        )

        return OutputProcesser(
            send_to_tokenizer=lambda *_: None,
            spec_algorithm="DFLASH",
            spec_num_tokens=4,
            physical_context_len=physical_context_len,
            metrics=SimpleNamespace(),
        )

    @staticmethod
    def _state(prompt_len, prior_output):
        return SimpleNamespace(
            prompt_input_ids=list(range(prompt_len)),
            output_ids=list(range(prior_output)),
        )

    def test_at_the_physical_limit_passes(self):
        p = self._processor(physical_context_len=100)
        p._check_physical_extent("r0", self._state(50, 46), output_length=4)

    def test_past_the_physical_limit_raises(self):
        p = self._processor(physical_context_len=100)
        with pytest.raises(RuntimeError, match="physical_context_len=100"):
            p._check_physical_extent("r0", self._state(50, 47), output_length=4)

    def test_disabled_when_none(self):
        p = self._processor(physical_context_len=None)
        p._check_physical_extent("r0", self._state(50, 470), output_length=4)
