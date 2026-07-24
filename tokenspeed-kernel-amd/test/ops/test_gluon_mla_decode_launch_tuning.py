# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Launch-tuning tests for the GFX950 Gluon MLA decode kernel."""

from __future__ import annotations

import pytest
from tokenspeed_kernel.ops.attention import gluon as _gluon_registrations  # noqa: F401
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import spec_matches_traits

mla_decode = pytest.importorskip(
    "tokenspeed_kernel_amd.ops.attention.gluon.mla_decode_gfx950",
    reason="tokenspeed-kernel-amd is required for MLA launch tuning tests",
)


def test_small_batch_launch_defaults_match_measured_winners() -> None:
    assert mla_decode._DEFAULT_SMALL_BATCH_LAUNCH == {
        1: ("bh16-multiblock", 256),
        2: ("bh64-small", 128),
        4: ("bh64-small", 256),
    }


@pytest.mark.parametrize(
    "batch,block_h,target_workgroups,expected_splits",
    [
        pytest.param(1, 16, 256, 64, id="bh16-b1-w256"),
        pytest.param(2, 64, 64, 32, id="bh64-b2-w64"),
        pytest.param(2, 64, 128, 64, id="bh64-b2-w128"),
        pytest.param(2, 64, 256, 128, id="bh64-b2-w256"),
        pytest.param(2, 64, 512, 256, id="bh64-b2-w512"),
        pytest.param(4, 64, 64, 16, id="bh64-b4-w64"),
        pytest.param(4, 64, 128, 32, id="bh64-b4-w128"),
        pytest.param(4, 64, 256, 64, id="bh64-b4-w256"),
        pytest.param(4, 64, 512, 128, id="bh64-b4-w512"),
        pytest.param(2, 16, 64, 8, id="bh16-b2-w64"),
        pytest.param(2, 16, 128, 16, id="bh16-b2-w128"),
        pytest.param(2, 16, 256, 32, id="bh16-b2-w256"),
        pytest.param(2, 16, 512, 64, id="bh16-b2-w512"),
        pytest.param(4, 16, 64, 4, id="bh16-b4-w64"),
        pytest.param(4, 16, 128, 8, id="bh16-b4-w128"),
        pytest.param(4, 16, 256, 16, id="bh16-b4-w256"),
        pytest.param(4, 16, 512, 32, id="bh16-b4-w512"),
    ],
)
def test_small_batch_split_selection_hits_target_workgroups(
    batch: int,
    block_h: int,
    target_workgroups: int,
    expected_splits: int,
) -> None:
    assert (
        mla_decode._select_num_kv_splits_small_batch(
            batch=batch,
            nhead=64,
            block_h=block_h,
            max_seqlen_k=80_000,
            block_n=64,
            target_workgroups=target_workgroups,
        )
        == expected_splits
    )


def test_small_batch_split_selection_caps_at_available_kv_blocks() -> None:
    assert (
        mla_decode._select_num_kv_splits_small_batch(
            batch=2,
            nhead=64,
            block_h=64,
            max_seqlen_k=65,
            block_n=64,
            target_workgroups=512,
        )
        == 2
    )


@pytest.mark.parametrize(
    "batch,expected",
    [
        pytest.param(1, True, id="b1"),
        pytest.param(2, True, id="b2"),
        pytest.param(3, False, id="b3"),
        pytest.param(4, True, id="b4"),
        pytest.param(64, False, id="b64"),
    ],
)
def test_small_batch_registration_matches_only_supported_batches(
    batch: int,
    expected: bool,
) -> None:
    spec = KernelRegistry.get().get_by_name(
        "gluon_mla_decode_bf16xbf16_gfx950_h64_small_batch"
    )
    if spec is None:
        pytest.skip("gfx950 Gluon MLA registrations are unavailable")

    traits = {
        "batch_size": batch,
        "batch_size_div_64": batch % 64 == 0,
        "q_len": 1,
        "num_q_heads": 64,
        "page_size": 64,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "support_logit_cap": False,
        "return_lse": False,
    }
    assert spec_matches_traits(spec, traits) is expected
