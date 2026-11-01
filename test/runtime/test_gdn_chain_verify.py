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

"""GDN's chain-form verify commits what ``gdn_decode_mtp`` commits and compiles nothing."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from test.runtime.test_gdn_state_paging import _ContractPool

from tokenspeed_kernel.ops.attention.gdn import (
    GdnChainVerifyForm,
    gdn_chain_verify_form,
    gdn_replay_commit_supported,
)
from tokenspeed_kernel.ops.attention.gdn._triton.chain_verify import (
    gdn_chain_verify_kernel,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state.gdn import GdnAttnBackend
from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

pytestmark = [
    pytest.mark.skipif(
        not current_platform().is_blackwell,
        reason="the chain form is measured and selected on Blackwell",
    ),
    pytest.mark.skipif(
        not gdn_replay_commit_supported(torch.bfloat16),
        reason="GDN ReplaySSM kernel unavailable",
    ),
]

DEV = "cuda"
LAYERS = (0, 1)
CONV_WIDTH = 4
COMMITTED = 9
# One TP4 rank of Qwen3.8-27B.
K_HEADS, V_HEADS, HEAD_DIM = 4, 12, 128
KEY_DIM, VALUE_DIM = K_HEADS * HEAD_DIM, V_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM


def _state(batch: int, seed: int) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer bf16 (conv, recurrent) pools: two pages per request after the null page."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    state = {}
    for layer in LAYERS:
        conv = 0.1 * torch.randn(
            1 + 2 * batch, CONV_DIM, CONV_WIDTH - 1, device=DEV, generator=gen
        )
        ssm = 0.05 * torch.randn(
            1 + 2 * batch, V_HEADS, HEAD_DIM, HEAD_DIM, device=DEV, generator=gen
        )
        conv, ssm = conv.bfloat16(), ssm.bfloat16()
        conv[0].zero_()
        ssm[0].zero_()
        state[layer] = (conv, ssm)
    return state


def _backend(state, *, width: int, max_bs: int, cls=GdnAttnBackend):
    """A ``cls`` backend bound to a copy of ``state``, its verify workspace preallocated."""
    pool = _ContractPool(
        {
            layer: ("linear_attention", conv.clone(), ssm.clone())
            for layer, (conv, ssm) in state.items()
        }
    )
    spec = MHAConfig(
        num_attention_heads=K_HEADS,
        num_kv_heads=K_HEADS,
        head_dim=HEAD_DIM,
        attn_tp_size=1,
    )
    linear = LinearAttnConfig(
        num_k_heads=K_HEADS,
        num_v_heads=V_HEADS,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        conv_kernel_size=CONV_WIDTH,
        layer_ids=LAYERS,
        tp_size=1,
        replay_ssm=True,
        draft_tree=False,
    )
    config = AttnConfig(
        device=DEV,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=4096,
        max_bs=max_bs,
        speculative_num_draft_tokens=width,
        components=(spec, linear),
    )
    backend = cls(config, spec)
    backend.set_kv_pool(pool)
    backend.init_cuda_graph_state(max_bs)
    backend.preallocate_verify_workspace(max_bs, width)
    return backend, pool


def _round_inputs(batch: int, width: int, seed: int):
    """Per layer, row-strided views of one ``[q | k | v | z | b | a]`` projection, and weights."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return {
        layer: dict(
            projection=torch.randn(
                batch * width,
                CONV_DIM + VALUE_DIM + 2 * V_HEADS,
                device=DEV,
                generator=gen,
            ).bfloat16(),
            conv_weights=(
                0.2 * torch.randn(CONV_DIM, CONV_WIDTH, device=DEV, generator=gen)
            ).bfloat16(),
            A_log=0.1 * torch.randn(V_HEADS, device=DEV, generator=gen),
            dt_bias=0.1 * torch.randn(V_HEADS, device=DEV, generator=gen),
        )
        for layer in LAYERS
    }


def _forward(backend, pool, batch: int, width: int, inputs) -> list[torch.Tensor]:
    outputs = []
    for layer in LAYERS:
        x = inputs[layer]
        projection = x["projection"]
        gates = projection[:, CONV_DIM + VALUE_DIM :]
        out = backend.forward_decode(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=pool,
            bs=batch,
            mixed_qkv=projection[:, :CONV_DIM],
            conv_weights=x["conv_weights"],
            bias=None,
            activation="silu",
            key_dim=KEY_DIM,
            value_dim=VALUE_DIM,
            attention_tp_size=1,
            head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM,
            a=gates[:, V_HEADS:],
            b=gates[:, :V_HEADS],
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            layer_id=layer,
            seq_len=batch * width,
        )
        outputs.append(out.reshape(batch, width, V_HEADS, HEAD_DIM))
    return outputs


def _refresh(backend, batch: int, width: int, *, graph_replay: bool) -> None:
    tables = 1 + torch.arange(2 * batch, dtype=torch.int32, device=DEV).view(batch, 2)
    backend.refresh_decode_metadata(
        batch,
        batch,
        torch.arange(batch, dtype=torch.int32, device=DEV),
        torch.full((batch,), COMMITTED + width, dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.DECODE,
        block_tables={"linear_attention": tables},
        for_graph_replay=graph_replay,
    )


def _capture(backend, pool, batch: int, width: int, inputs):
    """A CUDA graph of both layers' verify at ``batch`` and its outputs."""
    backend.init_forward_metadata_capture_cuda_graph(
        batch,
        torch.arange(batch, dtype=torch.int32, device=DEV),
        torch.full((batch,), COMMITTED + width, dtype=torch.int32, device=DEV),
        ForwardMode.DECODE,
        num_tokens=batch * width,
    )
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _forward(backend, pool, batch, width, inputs)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = _forward(backend, pool, batch, width, inputs)
    return graph, outputs


def _cuda_kernels(run) -> set[str]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        run()
        torch.cuda.synchronize()
    return {
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    }


def _committed(pool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            pool.get_component(layer, "conv_state"),
            pool.get_component(layer, "recurrent_state"),
        )
        for layer in LAYERS
    ]


@pytest.mark.parametrize("width", [3, 4, 16])
def test_chain_form_commits_like_the_mtp_verify(width):
    """Graph-replayed chain-form verifies commit bitwise what ``gdn_decode_mtp`` commits."""
    batch = 64
    assert (
        gdn_chain_verify_form(width, batch * V_HEADS, torch.bfloat16, chain_form=True)
        is GdnChainVerifyForm.CHAIN
    )
    state = _state(batch, seed=width)
    gen = torch.Generator().manual_seed(width)
    accepted = torch.randint(1, width + 1, (batch,), generator=gen, dtype=torch.int32)
    accepted = accepted.to(DEV)

    reference, reference_pool = _backend(
        state, width=width, max_bs=batch, cls=MambaAttnBackend
    )
    _refresh(reference, batch, width, graph_replay=False)
    reference_out = _forward(
        reference,
        reference_pool,
        batch,
        width,
        _round_inputs(batch, width, seed=width),
    )
    reference.commit_verified_state(accepted, accepted_path=None)

    chain, chain_pool = _backend(state, width=width, max_bs=batch)
    inputs = _round_inputs(batch, width, seed=width)
    graph, chain_out = _capture(chain, chain_pool, batch, width, inputs)
    # Restore the round's projection, which the capture forwards may have convolved.
    for layer, x in _round_inputs(batch, width, seed=width).items():
        inputs[layer]["projection"].copy_(x["projection"])
    _refresh(chain, batch, width, graph_replay=True)
    replayed = _cuda_kernels(graph.replay)
    chain.commit_verified_state(accepted, accepted_path=None)
    torch.cuda.synchronize()

    assert any("gdn_chain_verify_kernel" in name for name in replayed), replayed
    assert not any("gdn_decode_update" in name for name in replayed), replayed
    # The conv update fills the replay payload: no q/k/v split.
    assert not any("qkv_split" in name for name in replayed), replayed
    for (reference_conv, reference_ssm), (chain_conv, chain_ssm) in zip(
        _committed(reference_pool), _committed(chain_pool)
    ):
        assert torch.equal(reference_conv, chain_conv)
        assert torch.equal(reference_ssm, chain_ssm)
    for reference_layer, chain_layer in zip(reference_out, chain_out):
        # The chain form rounds q and the normalized keys to bf16 for its products.
        torch.testing.assert_close(
            chain_layer.float(), reference_layer.float(), atol=1e-4, rtol=2**-7
        )


def test_warmed_chain_verify_compiles_nothing_while_serving():
    """After the workspace warm, eager chain-form verifies of every tier compile nothing."""
    width, capacity = 4, 40
    backend, pool = _backend(_state(capacity, seed=21), width=width, max_bs=capacity)
    with patch.object(
        gdn_chain_verify_kernel,
        "_do_compile",
        wraps=gdn_chain_verify_kernel._do_compile,
    ) as compiles:
        for batch in (1, 3, 8, 9, 31, 32, capacity):
            inputs = _round_inputs(batch, width, seed=batch)
            _refresh(backend, batch, width, graph_replay=False)
            kernels = _cuda_kernels(
                lambda: _forward(backend, pool, batch, width, inputs)
            )
            assert any("gdn_chain_verify_kernel" in name for name in kernels), kernels
    assert compiles.call_count == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
