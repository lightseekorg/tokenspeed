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
per-request table needs a runtime clamp; these tests pin the pad math, the
tripwire that guards the one-lingering-step assumption, and the startup warning
for a RoPE table the pad runs past.
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


class TestRopeTableWarning:
    @pytest.fixture
    def warning(self, monkeypatch, caplog):
        """The startup warning for a model at a 4096-token context and
        ``physical`` positions, or an empty string."""
        import tokenspeed.runtime.execution.model_executor as module

        # The colorful logger does not propagate; let caplog see its records.
        monkeypatch.setattr(module.logger, "propagate", True)
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(max_position_embeddings=4096)
        )

        def warn(model, physical):
            caplog.clear()
            with caplog.at_level("WARNING"):
                module.warn_past_rope_table(model, model_config, 4096, physical)
            return caplog.text

        return warn

    def test_tables_are_assumed_to_end_at_the_derived_context(self, warning):
        model = object()
        assert "Lower --max-model-len by at least 48" in warning(model, 4096 + 48)
        assert warning(model, 4096) == ""

    def test_a_model_that_sizes_its_table_is_checked_against_it(self, warning):
        assert warning(SimpleNamespace(rope_positions=4096 + 64), 4096 + 48) == ""
        short = SimpleNamespace(rope_positions=4096)
        assert "Lower --max-model-len by at least 48" in warning(short, 4096 + 48)
        assert warning(SimpleNamespace(rope_positions=None), 4096 + 48) == ""
