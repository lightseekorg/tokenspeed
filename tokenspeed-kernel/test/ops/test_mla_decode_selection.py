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
import sys

import pytest
import torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel

pytestmark = pytest.mark.usefixtures("fresh_registry")

_MLA_FEATURES = frozenset({"mla", "paged"})


@pytest.fixture(autouse=True)
def _reset_platform_after_test():
    yield
    Platform.reset()


def _reload_mla_registrations(platform):
    Platform.override(platform)
    for module_name in (
        "tokenspeed_kernel.ops.attention.tokenspeed_mla",
        "tokenspeed_kernel.ops.attention",
    ):
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)


def test_hopper_does_not_select_blackwell_mla_decode(h100_platform):
    _reload_mla_registrations(h100_platform)

    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            "mla_decode_with_kvcache",
            torch.bfloat16,
            features=_MLA_FEATURES,
            platform=h100_platform,
            traits={"query_len": 1},
        )


def test_blackwell_selects_tokenspeed_mla_decode(b200_platform):
    _reload_mla_registrations(b200_platform)

    selected = select_kernel(
        "attention",
        "mla_decode_with_kvcache",
        torch.float8_e4m3fn,
        features=_MLA_FEATURES,
        platform=b200_platform,
        traits={"query_len": 1},
    )

    assert selected.name == "tokenspeed_mla_decode_with_kvcache"


def test_ampere_has_no_mla_decode_kernel(a100_platform):
    _reload_mla_registrations(a100_platform)

    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            "mla_decode_with_kvcache",
            torch.bfloat16,
            features=_MLA_FEATURES,
            platform=a100_platform,
            traits={"query_len": 1},
        )


def test_public_mla_decode_api_passes_mla_shaped_inputs(h100_platform):
    _reload_mla_registrations(h100_platform)

    from tokenspeed_kernel.ops.attention import mla_decode_with_kvcache

    captured = {}
    expected = torch.empty((2, 1, 4, 512), dtype=torch.bfloat16)

    def _impl(**kwargs):
        captured.update(kwargs)
        return expected

    KernelRegistry.get().register(
        KernelSpec(
            name="test_mla_decode",
            family="attention",
            mode="mla_decode_with_kvcache",
            solution="test",
            features=_MLA_FEATURES,
            dtypes=frozenset({torch.bfloat16}),
        ),
        _impl,
    )

    query = torch.empty((2, 1, 4, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((8, 64, 576), dtype=torch.bfloat16)
    block_tables = torch.zeros((2, 8), dtype=torch.int32)
    seq_lens = torch.tensor([64, 128], dtype=torch.int32)

    result = mla_decode_with_kvcache(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=0.07216882,
        override="test_mla_decode",
    )

    assert result is expected
    assert captured["query"] is query
    assert captured["kv_cache"] is kv_cache
    assert captured["block_tables"] is block_tables
    assert captured["seq_lens"] is seq_lens
    assert captured["kv_lora_rank"] == 512
    assert captured["qk_rope_head_dim"] == 64
    assert captured["softmax_scale"] == 0.07216882
