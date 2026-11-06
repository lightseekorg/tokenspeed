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

"""NVIDIA CuTe ReplaySSM dispatch, state ownership and accepted-path replay."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import ShapeCapture
from tokenspeed_kernel.registry import KernelRegistry

_TEST_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TEST_DIR))
sys.path.insert(0, str(_TEST_DIR.parent))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=45,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="CuTe GDN requires NVIDIA SM90+",
)

if not torch.cuda.is_available() or not current_platform().is_hopper_plus:
    pytest.skip("CuTe GDN requires NVIDIA SM90+", allow_module_level=True)
pytest.importorskip("cutlass")

from test.runtime.cache_pool_test_utils import one_group
from test.runtime.test_gdn_state_paging import _mamba_config_pair
from test.runtime.test_qwen35_gdn_replay import _tree_verify_inputs

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state.gdn import GdnAttnBackend
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
    HybridMHATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    cache_dtype_name,
    pack,
)

_BATCH = 2
_HEADS = 2


@dataclass
class VerifyCase:
    backend: GdnAttnBackend
    pool: HybridMHATokenToKVPool
    inputs: dict[str, torch.Tensor]
    tokens: int
    head_dim: int
    value_head_dim: int
    vheads: int
    parents: list[list[int]] | None

    @property
    def state(self):
        return self.pool.get_component(0, "recurrent_state")

    def refresh(self, *, actual_bs, read_pages, for_graph_replay):
        self.backend.refresh_decode_metadata(
            _BATCH,
            actual_bs,
            torch.arange(_BATCH, device="cuda", dtype=torch.int32),
            torch.full((_BATCH,), self.tokens + 4, device="cuda", dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=for_graph_replay,
            block_tables={
                "linear_attention": torch.tensor(
                    [[read_pages[0], 5], [read_pages[1], 6]],
                    device="cuda",
                    dtype=torch.int32,
                )
            },
        )

    def forward(self):
        return self.backend.forward_decode(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=self.pool,
            bs=_BATCH,
            mixed_qkv=self.inputs["mixed_qkv"].clone(),
            conv_weights=self.inputs["conv_weights"],
            bias=None,
            activation="silu",
            key_dim=_HEADS * self.head_dim,
            value_dim=self.vheads * self.value_head_dim,
            attention_tp_size=1,
            head_k_dim=self.head_dim,
            head_v_dim=self.value_head_dim,
            a=self.inputs["a"],
            b=self.inputs["b"],
            A_log=self.inputs["A_log"],
            dt_bias=self.inputs["dt_bias"],
            layer_id=0,
            seq_len=_BATCH * self.tokens,
        )


def make_case(*, tokens, head_dim, value_head_dim, vheads, state_dtype, replay, tree):
    width = 2 * _HEADS * head_dim + vheads * value_head_dim
    group = one_group(
        "linear_attention",
        CacheFieldSpec("layer.0.conv", "conv", (width, 3), "bfloat16"),
        CacheFieldSpec(
            "layer.0.ssm",
            "ssm",
            (vheads, value_head_dim, head_dim),
            cache_dtype_name(state_dtype),
        ),
        attn_kind="state",
    )
    plan = pack((group,), prefix_granularity=64).bind(8)
    arena = CacheArena(plan, "cuda", cache_group_specs=(group[0],))
    pool = HybridMHATokenToKVPool(
        arena=arena,
        dtype=torch.bfloat16,
        head_num=_HEADS,
        head_dim=head_dim,
        layer_num=1,
        rank=0,
        layer_types=("linear_attention",),
    )
    generator = torch.Generator(device="cuda").manual_seed(8273)

    def rand(shape, dtype):
        return torch.randn(shape, device="cuda", generator=generator).to(dtype)

    conv = pool.get_component(0, "conv_state")
    state = pool.get_component(0, "recurrent_state")
    conv.copy_(rand(conv.shape, conv.dtype) * 0.02)
    state.copy_(rand(state.shape, state.dtype) * 0.02)
    conv[0].zero_()
    state[0].zero_()
    inputs = dict(
        mixed_qkv=rand((_BATCH * tokens, width), torch.bfloat16),
        a=rand((_BATCH * tokens, vheads), torch.bfloat16),
        b=rand((_BATCH * tokens, vheads), torch.bfloat16),
        conv_weights=rand((width, 4), torch.bfloat16) * 0.1,
        A_log=rand((vheads,), torch.float32) * 0.1,
        dt_bias=rand((vheads,), torch.float32) * 0.1,
    )
    config, spec = _mamba_config_pair(
        torch,
        heads=_HEADS,
        head_dim=head_dim,
        spec_tokens=tokens,
        device="cuda",
        replay_ssm=replay,
        draft_tree=tree,
    )
    linear = replace(
        config.component(LinearAttnConfig),
        num_v_heads=vheads,
        head_v_dim=value_head_dim,
    )
    backend = GdnAttnBackend(replace(config, components=(spec, linear)), spec)
    backend.set_kv_pool(pool)
    backend.init_cuda_graph_state(_BATCH)
    backend.preallocate_verify_workspace(_BATCH, tokens)
    parents = None
    if tree:
        parents = [
            [-1] + [(i - 1) // 2 for i in range(1, tokens)],
            list(range(-1, tokens - 1)),
        ]
        backend.bind_tree_verify(_tree_verify_inputs(parents))
    return VerifyCase(
        backend, pool, inputs, tokens, head_dim, value_head_dim, vheads, parents
    )


def recorded_forward(case, path):
    capture = ShapeCapture.get()
    previous = capture.enabled
    capture.enabled = True
    try:
        output = case.forward()
    finally:
        capture.enabled = previous
    capture.dump(path)
    records = [r for r in json.loads(path.read_text()) if r["mode"] == "gdn_decode_mtp"]
    return output, records[-1]["kernel_name"]


def check_commit(case, before, *, actual_bs, read_pages):
    paths = []
    for request in range(actual_bs):
        if case.parents is None:
            path = list(range(case.tokens - request))
        else:
            path = []
            node = case.tokens - 1
            while node >= 0:
                path.append(node)
                node = case.parents[request][node]
            path.reverse()
        paths.append(path)
    payload = case.backend._gdn_replay.payload[0].view(_BATCH, case.tokens, -1)
    keys, values, a, b = payload.split(
        (
            _HEADS * case.head_dim,
            case.vheads * case.value_head_dim,
            case.vheads,
            case.vheads,
        ),
        dim=-1,
    )
    keys = keys.double().view(_BATCH, case.tokens, _HEADS, case.head_dim)
    keys = keys / (keys.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    keys = keys.repeat_interleave(case.vheads // _HEADS, 2)
    values = values.double().view(_BATCH, case.tokens, case.vheads, case.value_head_dim)
    decay = (
        -case.inputs["A_log"].double().exp()
        * torch.nn.functional.softplus(a.double() + case.inputs["dt_bias"].double())
    ).exp()
    beta = b.double().sigmoid()
    expected = []
    for request, path in enumerate(paths):
        state = before[read_pages[request]].double()
        for node in path:
            state = state * decay[request, node, :, None, None]
            delta = (
                values[request, node]
                - (state * keys[request, node, :, None, :]).sum(-1)
            ) * beta[request, node, :, None]
            state = state + delta[..., None] * keys[request, node, :, None, :]
        expected.append(state)
    # Capture includes padded requests; commit receives only live requests.
    accepted = torch.tensor([len(p) for p in paths], device="cuda", dtype=torch.int32)
    accepted_path = None
    if case.parents is not None:
        accepted_path = torch.full(
            (actual_bs, case.tokens), -1, device="cuda", dtype=torch.int32
        )
        for request, path in enumerate(paths):
            accepted_path[request, : len(path)] = torch.tensor(
                path, device="cuda", dtype=torch.int32
            )
    case.backend.commit_verified_state(accepted, accepted_path=accepted_path)
    torch.cuda.synchronize()
    for request, state in enumerate(expected):
        torch.testing.assert_close(
            case.state[5 + request].double(), state, atol=1e-4, rtol=2**-7
        )
    untouched = [
        i for i in range(case.state.shape[0]) if i not in range(5, 5 + actual_bs)
    ]
    assert torch.equal(case.state[untouched], before[untouched])


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "tokens,tree,vheads,value_head_dim",
    [
        (4, False, 4, 128),
        (8, False, 4, 128),
        (16, False, 4, 128),
        (16, False, 24, 128),
        (8, True, 4, 128),
        (16, True, 4, 128),
        (4, False, 4, 64),
        (4, False, 4, 256),
    ],
)
def test_replay_verifies_and_commits_the_accepted_path(
    tmp_path, state_dtype, tokens, tree, vheads, value_head_dim
):
    case = make_case(
        tokens=tokens,
        head_dim=128,
        value_head_dim=value_head_dim,
        vheads=vheads,
        state_dtype=state_dtype,
        replay=True,
        tree=tree,
    )
    reference = make_case(
        tokens=tokens,
        head_dim=128,
        value_head_dim=value_head_dim,
        vheads=vheads,
        state_dtype=torch.float32,
        replay=False,
        tree=tree,
    )
    reference.state.copy_(case.state)
    for item in (case, reference):
        item.refresh(actual_bs=_BATCH, read_pages=(1, 2), for_graph_replay=False)
    before = case.state.clone()
    output, selected = recorded_forward(case, tmp_path / "verify.json")
    expected = (
        "cute_dsl_gdn_decode_mtp" if value_head_dim == 128 else "triton_gdn_decode_mtp"
    )
    assert selected == expected
    assert case.backend._verify_packed_qkv_views
    torch.testing.assert_close(
        output.double(), reference.forward().double(), atol=1e-4, rtol=2**-7
    )
    assert torch.equal(case.state, before)
    check_commit(case, before, actual_bs=_BATCH, read_pages=(1, 2))


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "tokens,head_dim,value_head_dim,replay,tree",
    [
        (4, 64, 64, True, False),
        (4, 128, 64, True, False),
        (4, 128, 256, True, False),
        (17, 128, 128, True, False),
        (17, 128, 128, True, True),
        (4, 128, 128, True, True),
        (4, 128, 128, False, False),
        (4, 128, 128, False, True),
    ],
)
def test_verify_falls_back_for_unsupported_or_state_writing_calls(
    tmp_path, state_dtype, tokens, head_dim, value_head_dim, replay, tree
):
    case = make_case(
        tokens=tokens,
        head_dim=head_dim,
        value_head_dim=value_head_dim,
        vheads=4,
        state_dtype=state_dtype,
        replay=replay,
        tree=tree,
    )
    case.refresh(actual_bs=1, read_pages=(1, 2), for_graph_replay=False)
    before = case.state.clone()
    output, selected = recorded_forward(case, tmp_path / "verify.json")
    expected = "triton_gdn_decode_mtp"
    if (
        not tree
        and head_dim == 128
        and value_head_dim == 128
        and (state_dtype != torch.bfloat16 or replay)
        and KernelRegistry.get().get_by_name("flashinfer_gdn_decode_mtp") is not None
    ):
        expected = "flashinfer_gdn_decode_mtp"
    if (
        current_platform().is_blackwell
        and replay
        and not tree
        and head_dim == value_head_dim == 64
        and state_dtype == torch.bfloat16
    ):
        expected = "triton_gdn_decode_mtp_chain"
    assert selected == expected
    assert case.backend._verify_packed_qkv_views == (
        selected in {"triton_gdn_decode_mtp", "triton_gdn_decode_mtp_chain"}
    )
    assert case.backend._verify_reads_committed_recurrent_state == (
        not replay and selected == "triton_gdn_decode_mtp"
    )
    assert output.isfinite().all()
    if replay:
        assert torch.equal(case.state, before)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tree", [False, True])
def test_replay_graph_refreshes_inputs_and_padding(tmp_path, state_dtype, tree):
    case = make_case(
        tokens=8,
        head_dim=128,
        value_head_dim=128,
        vheads=4,
        state_dtype=state_dtype,
        replay=True,
        tree=tree,
    )
    reference = make_case(
        tokens=8,
        head_dim=128,
        value_head_dim=128,
        vheads=4,
        state_dtype=torch.float32,
        replay=False,
        tree=tree,
    )
    reference.state.copy_(case.state)
    case.refresh(actual_bs=_BATCH, read_pages=(1, 2), for_graph_replay=False)
    case.forward()
    capture = ShapeCapture.get()
    previous = capture.enabled
    capture.enabled = True
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output = case.forward()
    finally:
        capture.enabled = previous
    capture.dump(tmp_path / "graph.json")
    assert (
        json.loads((tmp_path / "graph.json").read_text())[-1]["kernel_name"]
        == "cute_dsl_gdn_decode_mtp"
    )
    for item in (case, reference):
        item.inputs["mixed_qkv"].neg_()
        item.inputs["a"].add_(0.25)
        item.inputs["b"].sub_(0.5)
        item.refresh(actual_bs=1, read_pages=(2, 1), for_graph_replay=True)
    before = case.state.clone()
    graph.replay()
    expected = reference.forward()
    torch.testing.assert_close(
        output[:, :8].double(), expected[:, :8].double(), atol=1e-4, rtol=2**-7
    )
    assert torch.equal(case.state, before)
    check_commit(case, before, actual_bs=1, read_pages=(2, 1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
