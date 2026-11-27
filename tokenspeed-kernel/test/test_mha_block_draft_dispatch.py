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

"""Draft dispatch uses real registry specs; only GPU execution is substituted."""

from dataclasses import replace

import pytest
import tokenspeed_kernel.ops.attention.mha.triton as triton_mha
import torch
from tokenspeed_kernel import platform as kernel_platform
from tokenspeed_kernel.ops.attention import mha
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    Platform,
    PlatformInfo,
)
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@pytest.fixture
def dispatch_platform(monkeypatch):
    registry = KernelRegistry.get()
    snapshot = registry.snapshot()
    monkeypatch.setattr(mha, "_SCHEDULE_KERNELS", {})
    monkeypatch.setattr(mha, "_SCHEDULE_PREFERS", dict(mha._SCHEDULE_PREFERS))
    original = Platform.get()
    # Do not leave a synthetic platform's launch policy in the process cache.
    monkeypatch.setattr(kernel_platform, "_pdl_enabled", False)
    platform = PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(9, 0),
        device_name="dispatch test",
        device_count=1,
        total_memory=0,
        memory_bandwidth=0,
        sm_count=1,
        max_threads_per_sm=1,
        max_shared_memory_per_sm=0,
    )

    def switch(vendor):
        Platform.override(replace(platform, vendor=vendor))
        registry.clear_cache()

    switch("nvidia")
    try:
        yield switch
    finally:
        registry.restore(snapshot)
        Platform.override(original)


@pytest.fixture
def inputs():
    return {
        "q": torch.ones(8, 4, 128, dtype=torch.bfloat16),
        "k_cache": torch.zeros(2, 128, 1, 128, dtype=torch.bfloat16),
        "v_cache": torch.zeros(2, 128, 1, 128, dtype=torch.bfloat16),
        "page_table": torch.ones(8, 1, dtype=torch.int32),
        "cache_seqlens": torch.full((8,), 16, dtype=torch.int32),
        "max_seqlen_k": 128,
        "max_seqlen_q": 1,
    }


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_legacy_gpu_leaf_keeps_expanded_metadata(
    dispatch_platform, inputs, monkeypatch, vendor
):
    dispatch_platform(vendor)
    # Retain the built-in registration AND its strict public leaf signature.
    # Only replace the unavailable accelerator computation below that wrapper.
    monkeypatch.setattr(
        triton_mha, "_triton_mha_decode_with_kvcache_impl", lambda **kwargs: kwargs["q"]
    )
    output = mha.mha_decode_with_kvcache(
        **inputs, block_draft=True, batch_invariant=False, solution="triton"
    )
    assert output is inputs["q"]


def _register_plugin_leaves():
    common = {
        "solution": "dispatch_test_plugin",
        "capability": CapabilityRequirement(vendors=frozenset({"ascend"})),
        "signatures": format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}
        ),
    }

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="test_plugin_draft",
        features=frozenset({"block_draft"}),
        priority=Priority.PLUGIN,
        traits={"block_draft": frozenset({True})},
        **common,
    )
    def draft(q, **kwargs):
        return q * 2

    @register_kernel(
        "attention",
        "mha_decode_with_kvcache",
        name="test_plugin_ordinary",
        priority=Priority.PLUGIN + 1,
        traits={"block_draft": frozenset({False})},
        **common,
    )
    def ordinary(q, **kwargs):
        return -q


def test_plugin_draft_trait_isolation(dispatch_platform, inputs):
    dispatch_platform("ascend")
    _register_plugin_leaves()
    output = mha.mha_decode_with_kvcache(
        **inputs,
        block_draft=True,
        batch_invariant=False,
        solution="dispatch_test_plugin",
    )
    torch.testing.assert_close(output, inputs["q"] * 2)
    # The higher-priority ordinary leaf must remain isolated by the trait.
    ordinary = mha.mha_decode_with_kvcache(
        **inputs,
        block_draft=False,
        batch_invariant=False,
        solution="dispatch_test_plugin",
    )
    torch.testing.assert_close(ordinary, -inputs["q"])


def test_legacy_positional_arguments_keep_their_meaning(
    dispatch_platform, inputs, monkeypatch
):
    def device_decode(**kwargs):
        output = kwargs["q"] * kwargs["softmax_scale"] + kwargs["logit_cap"]
        return (output, kwargs["sinks"]) if kwargs["return_lse"] else output

    monkeypatch.setattr(
        triton_mha, "_triton_mha_decode_with_kvcache_impl", device_decode
    )
    sinks = torch.arange(4, dtype=torch.bfloat16)
    output, lse = mha.mha_decode_with_kvcache(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["page_table"],
        inputs["cache_seqlens"],
        128,
        1,
        -1,
        0,
        2.0,
        sinks,
        True,
        0.25,  # window, sink, cap, sinks, LSE, scale
        None,
        None,
        None,
        None,
        None,  # scales and schedule
        None,
        "triton",  # override and solution retain their positional slots
        block_draft=False,
        batch_invariant=False,
    )
    torch.testing.assert_close(output, inputs["q"] * 0.25 + 2.0)
    assert lse is sinks


@pytest.mark.parametrize("draft_first", [False, True])
def test_schedule_and_decode_share_draft_selection(
    dispatch_platform, inputs, draft_first
):
    dispatch_platform("ascend")
    common = {
        "solution": "dispatch_test_schedule",
        "capability": CapabilityRequirement(vendors=frozenset({"ascend"})),
        "signatures": format_signatures(
            ("q", "k_cache", "v_cache"), "dense", {torch.bfloat16}
        ),
        "priority": Priority.PLUGIN,
    }
    for block_draft in (False, True):
        marker = 2 if block_draft else 1
        name = f"test_scheduled_decode_{marker}"

        def decode(q, schedule, marker=marker, **kwargs):
            return q * schedule[0] + marker

        def schedule(cache_seqlens, marker=marker, **kwargs):
            return torch.full_like(cache_seqlens, marker)

        traits = {"block_draft": frozenset({block_draft})}
        register_kernel(
            "attention",
            "mha_decode_with_kvcache",
            name=name,
            features=frozenset({"split_schedule"}),
            traits=traits,
            **common,
        )(decode)
        register_kernel(
            "attention",
            "mha_decode_schedule",
            name=f"test_schedule_{marker}",
            traits=traits,
            **common,
        )(schedule)
        mha._SCHEDULE_PREFERS[name] = lambda **kwargs: True

    for block_draft in (draft_first, not draft_first):
        options = dict(
            q=inputs["q"],
            k_cache=inputs["k_cache"],
            v_cache=inputs["v_cache"],
            max_seqlen_q=inputs["max_seqlen_q"],
            block_draft=block_draft,
            solution="dispatch_test_schedule",
        )
        assert mha.mha_decode_takes_schedule(**options)
        schedule = mha.mha_decode_schedule(
            **options, cache_seqlens=inputs["cache_seqlens"]
        )
        output = mha.mha_decode_with_kvcache(
            **inputs,
            block_draft=block_draft,
            batch_invariant=False,
            solution="dispatch_test_schedule",
            schedule=schedule,
        )
        marker = 2 if block_draft else 1
        torch.testing.assert_close(output, inputs["q"] * marker + marker)
