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

"""Qwen4 cache consumers are assembled per view and budgeted independently."""

from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.layers.attention.registry as registry
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
)
from tokenspeed.runtime.layers.attention.registry import (
    _compose_qwen4_exp_backend,
    _prepare_verify_workspace,
)


@pytest.mark.parametrize("has_ple", [False, True])
@pytest.mark.parametrize("has_qsa", [False, True])
@pytest.mark.parametrize("is_draft", [False, True])
def test_composition_uses_local_fields_without_requiring_linear_layers(
    has_ple, has_qsa, is_draft
):
    spec = SimpleNamespace(
        num_attention_heads=8, num_kv_heads=8, attn_tp_size=1, head_dim=16
    )
    config = SimpleNamespace(
        component=lambda component_type: spec,
        device="cpu",
        dtype=torch.bfloat16,
        is_draft=is_draft,
        speculative_num_draft_tokens=4,
        context_len=512,
        max_bs=4,
    )
    groups = [QWEN4_EXP_PLE_CACHE_GROUP] if has_ple else []
    if has_qsa:
        groups += [QWEN4_EXP_QSA_CACHE_GROUP, QWEN4_EXP_QSA_RECENT_CACHE_GROUP]
    fields = [
        SimpleNamespace(field_id=f"layer.3.{group}", group_id=group) for group in groups
    ]
    # A shared arena also publishes other views' fields. They must not create
    # consumers in this view merely because a family exists in the contract.
    fields += [
        SimpleNamespace(field_id=f"layer.9.{group}", group_id=group)
        for group in (
            QWEN4_EXP_PLE_CACHE_GROUP,
            QWEN4_EXP_QSA_CACHE_GROUP,
            QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
        )
    ]
    pool = SimpleNamespace(
        field_layer_range=range(3, 4),
        layer_num=1,
        arena=SimpleNamespace(plan=SimpleNamespace(fields=fields)),
    )
    full = SimpleNamespace(device="cpu", spec_num_tokens=4)
    backend = _compose_qwen4_exp_backend(config, pool, full, None, [3, 7])
    assert backend.linear_attn_backend is None
    assert (backend.ple_backend is not None) == has_ple
    assert (backend.indexer_backend is not None) == has_qsa
    # NextN layers are local even if the HF config still names target layers.
    assert backend._backend_for_layer(0) is full
    assert len(backend.child_backends()) == 1 + has_ple + has_qsa


@pytest.mark.parametrize("width", [1, 4])
def test_verify_workspace_counts_each_consumer_once_and_checks_zero_budget(width):
    calls = []

    def consumer(name, nbytes):
        def preallocate(max_bs, draft_token_num):
            calls.append((name, max_bs, draft_token_num))
            return nbytes

        return SimpleNamespace(preallocate_verify_workspace=preallocate)

    root = Qwen4ExpBackend(
        SimpleNamespace(device="cpu"),
        consumer("gdn", 3),
        [0],
        consumer("ple", 5),
        consumer("qsa", 7),
    )
    kwargs = dict(
        server_args=SimpleNamespace(speculative_num_draft_tokens=width),
        config=SimpleNamespace(max_bs=2, speculative_num_draft_tokens=width),
        backend=root,
        draft_backend=None,
        uses_paged_state_verify=True,
        is_inkling=False,
    )
    _prepare_verify_workspace(**kwargs, expected_bytes=15 if width > 1 else 0)
    assert calls == (
        [("gdn", 2, width), ("ple", 2, width), ("qsa", 2, width)] if width > 1 else []
    )
    with pytest.raises(RuntimeError, match="does not match allocated tensors"):
        _prepare_verify_workspace(**kwargs, expected_bytes=0 if width > 1 else 1)


def test_hybrid_factory_skips_gdn_when_only_other_ranks_own_state(monkeypatch):
    full = SimpleNamespace(device="cpu", spec_num_tokens=1)
    components = {
        SoftmaxAttnConfig: SimpleNamespace(),
        LinearAttnConfig: SimpleNamespace(layer_ids=(3,)),
    }
    config = SimpleNamespace(component=components.get)
    pool = SimpleNamespace(
        state_group_by_layer={},
        field_layer_range=range(0, 1),
        layer_num=1,
        arena=SimpleNamespace(plan=SimpleNamespace(fields=[])),
    )
    monkeypatch.setattr(registry, "is_qwen4_exp", lambda hf_config: True)
    monkeypatch.setattr(
        registry, "_create_attn_backend_with_name", lambda name, arch, config: full
    )
    backend = registry._create_hybrid_linear_attn_backend(
        SimpleNamespace(speculative_algorithm=None),
        SimpleNamespace(
            hf_config=SimpleNamespace(full_attention_layer_ids=[0]),
            attention_arch="mha",
        ),
        config,
        pool=pool,
        full_attn_backend_name=None,
        is_kda=False,
    )
    assert backend.linear_attn_backend is None
    assert backend.child_backends() == (full,)
    assert backend._backend_for_layer(0) is full
