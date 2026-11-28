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

"""SM100 DeepGEMM BF16 fused router GEMM + sqrt-softplus top-k tests."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.moe import deep_gemm as deep_gemm_moe
from tokenspeed_kernel.ops.moe import moe_topk, set_mega_gate_workspace_region
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.thirdparty.cuda import mega_gate as mega_gate_mod
from tokenspeed_kernel.thirdparty.cuda.mega_gate import (
    _mega_gate_candidates,
    _mega_gate_heuristic_key,
    bf16_mega_gate,
    is_bf16_mega_gate_available,
    mega_gate_warmup_token_counts,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_nvidia
    or current_platform().arch_version.major != 10
    or not is_bf16_mega_gate_available(),
    reason="requires SM100-family GPU and deep_gemm.bf16_mega_gate",
)


def _reference_sqrt_softplus_topk(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
    *,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None = None,
    image_bias: torch.Tensor | None = None,
    image_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = hidden_states.float() @ gate_weight.float().T
    scores = F.softplus(logits).sqrt()
    selection_scores = scores
    if correction_bias is not None:
        bias = correction_bias.unsqueeze(0).expand(hidden_states.shape[0], -1)
        if image_bias is not None and image_token_mask is not None:
            bias = torch.where(
                image_token_mask.unsqueeze(-1),
                image_bias.unsqueeze(0).expand_as(bias),
                bias,
            )
        selection_scores = scores + bias
    elif image_bias is not None and image_token_mask is not None:
        bias = torch.where(
            image_token_mask.unsqueeze(-1),
            image_bias.unsqueeze(0).expand(hidden_states.shape[0], -1),
            torch.zeros(
                hidden_states.shape[0],
                gate_weight.shape[0],
                device=hidden_states.device,
                dtype=scores.dtype,
            ),
        )
        selection_scores = scores + bias
    topk_ids = torch.topk(selection_scores, k=top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, topk_ids)
    weights = (weights / weights.sum(dim=-1, keepdim=True)) * routed_scaling_factor
    return weights, topk_ids


def _canonicalize_routes(weights: torch.Tensor, ids: torch.Tensor):
    order = torch.argsort(ids, dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


@pytest.mark.parametrize("tokens", [0, 1, 17, 128, 300])
@pytest.mark.parametrize("experts", [64, 256, 384])
@pytest.mark.parametrize("top_k", [1, 8])
@pytest.mark.parametrize("with_bias", [False, True])
def test_bf16_mega_gate_matches_fp32_reference(
    tokens: int,
    experts: int,
    top_k: int,
    with_bias: bool,
) -> None:
    torch.manual_seed(tokens + experts + top_k + int(with_bias))
    hidden = 1024
    # Construct structured rows with separated expert scores to verify exact ID
    # selection alongside FP32 weight accuracy across SM100 TensorCore GEMMs.
    gate_weight = (torch.randn(experts, hidden, device="cuda") * 0.1).bfloat16()
    gate_weight[:, :experts] += (
        torch.eye(experts, device="cuda", dtype=torch.bfloat16) * 1.5
    )
    hidden_states = (torch.randn(tokens, hidden, device="cuda") * 0.05).bfloat16()
    for row in range(tokens):
        primary = (row * 13) % experts
        hidden_states[row, primary] = 2.0
        if top_k > 1:
            for offset in range(1, top_k):
                hidden_states[row, (primary + offset * 7) % experts] = (
                    1.8 - offset * 0.1
                )
    bias = torch.randn(experts, device="cuda") * 0.02 if with_bias else None

    got_weights, got_ids = moe_topk(
        None,
        top_k,
        "sqrt_softplus",
        "topk",
        True,
        2.5,
        correction_bias=bias,
        hidden_states=hidden_states,
        gate_weight=gate_weight,
        topk_weights_dtype=torch.float32,
        topk_indices_dtype=torch.int64,
        override="deep_gemm_bf16_mega_gate",
    )
    ref_weights, ref_ids = _reference_sqrt_softplus_topk(
        hidden_states,
        gate_weight,
        top_k,
        routed_scaling_factor=2.5,
        correction_bias=bias,
    )
    got_weights, got_ids = _canonicalize_routes(got_weights, got_ids.to(torch.int64))
    ref_weights, ref_ids = _canonicalize_routes(ref_weights, ref_ids)
    torch.testing.assert_close(got_ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(got_weights, ref_weights, atol=2e-3, rtol=2e-3)


def test_bf16_mega_gate_image_bias_and_out_buffers() -> None:
    torch.manual_seed(7)
    tokens, hidden, experts, top_k = 12, 512, 32, 4
    gate_weight = torch.zeros(experts, hidden, device="cuda", dtype=torch.bfloat16)
    gate_weight[:, :experts] = torch.eye(experts, device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.full(
        (tokens, hidden), 0.1, device="cuda", dtype=torch.bfloat16
    )
    text_bias = torch.arange(experts, device="cuda", dtype=torch.float32) * 0.25
    image_bias = (
        torch.arange(experts - 1, -1, -1, device="cuda", dtype=torch.float32) * 0.25
    )
    image_mask = torch.tensor(
        [i % 2 == 1 for i in range(tokens)], device="cuda", dtype=torch.bool
    )
    out_weights = torch.empty(tokens, top_k, device="cuda", dtype=torch.float32)
    out_ids = torch.empty(tokens, top_k, device="cuda", dtype=torch.int64)

    got_weights, got_ids = moe_topk(
        None,
        top_k,
        "sqrt_softplus",
        "topk",
        True,
        1.25,
        correction_bias=text_bias,
        hidden_states=hidden_states,
        gate_weight=gate_weight,
        image_bias=image_bias,
        image_token_mask=image_mask,
        out=(out_weights, out_ids),
        topk_weights_dtype=torch.float32,
        topk_indices_dtype=torch.int64,
        override="deep_gemm_bf16_mega_gate",
    )
    assert got_weights.data_ptr() == out_weights.data_ptr()
    assert got_ids.data_ptr() == out_ids.data_ptr()

    ref_weights, ref_ids = _reference_sqrt_softplus_topk(
        hidden_states,
        gate_weight,
        top_k,
        routed_scaling_factor=1.25,
        correction_bias=text_bias,
        image_bias=image_bias,
        image_token_mask=image_mask,
    )
    got_weights, got_ids = _canonicalize_routes(got_weights, got_ids)
    ref_weights, ref_ids = _canonicalize_routes(ref_weights, ref_ids)
    torch.testing.assert_close(got_ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(got_weights, ref_weights, atol=2e-3, rtol=2e-3)


def test_bf16_mega_gate_static_replica_placement() -> None:
    tokens, hidden, experts, top_k = 10, 512, 16, 2
    gate_weight = torch.zeros(experts, hidden, device="cuda", dtype=torch.bfloat16)
    gate_weight[:, :experts] = torch.eye(experts, device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.zeros(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    # Every token prefers logical experts 0 and 1.
    hidden_states[:, 0] = 4.0
    hidden_states[:, 1] = 3.0
    to_physical = torch.full((experts, 4), -1, device="cuda", dtype=torch.int32)
    logical_count = torch.ones(experts, device="cuda", dtype=torch.int32)
    for expert in range(experts):
        to_physical[expert, 0] = expert
    # Replicate logical experts 0 and 1 across physical slots >= experts.
    to_physical[0, :3] = torch.tensor([0, 16, 17], device="cuda", dtype=torch.int32)
    logical_count[0] = 3
    to_physical[1, :2] = torch.tensor([1, 18], device="cuda", dtype=torch.int32)
    logical_count[1] = 2

    for ep_rank in (0, 2):
        weights, ids = moe_topk(
            None,
            top_k,
            "sqrt_softplus",
            "topk",
            True,
            1.0,
            hidden_states=hidden_states,
            gate_weight=gate_weight,
            logical_to_physical_map=to_physical,
            logical_replica_counts=logical_count,
            ep_rank=ep_rank,
            topk_weights_dtype=torch.float32,
            topk_indices_dtype=torch.int64,
            override="deep_gemm_bf16_mega_gate",
        )
        expected = torch.empty(tokens, top_k, device="cuda", dtype=torch.int64)
        for token_idx in range(tokens):
            expected[token_idx, 0] = to_physical[0, (ep_rank + token_idx * 23333) % 3]
            expected[token_idx, 1] = to_physical[1, (ep_rank + token_idx * 23333) % 2]
        assert weights.shape == (tokens, top_k)
        torch.testing.assert_close(ids, expected, atol=0, rtol=0)


def test_bf16_mega_gate_validation_and_registry_guards() -> None:
    hidden = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(NoKernelFoundError):
        moe_topk(
            None,
            4,
            "sqrt_softplus",
            "topk",
            False,
            1.0,
            hidden_states=hidden,
            gate_weight=weight,
            solution="deep_gemm",
        )
    with pytest.raises(ValueError, match="normalized"):
        bf16_mega_gate(
            hidden,
            weight,
            4,
            renormalize=False,
            routed_scaling_factor=1.0,
            ep_rank=0,
        )
    with pytest.raises(ValueError, match="INT64"):
        bf16_mega_gate(
            hidden,
            weight,
            4,
            renormalize=True,
            routed_scaling_factor=1.0,
            ep_rank=0,
            out=(
                torch.empty(4, 4, device="cuda", dtype=torch.float32),
                torch.empty(4, 4, device="cuda", dtype=torch.int32),
            ),
        )
    with pytest.raises(NoKernelFoundError):
        moe_topk(
            None,
            4,
            "sqrt_softplus",
            "topk",
            True,
            1.0,
            hidden_states=hidden[:, :128].contiguous(),
            gate_weight=weight[:, :128].contiguous(),
            solution="deep_gemm",
        )
    to_physical = torch.arange(16, device="cuda", dtype=torch.int32).unsqueeze(-1)
    logical_count = torch.ones(16, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="ep_rank is required"):
        moe_topk(
            None,
            4,
            "sqrt_softplus",
            "topk",
            True,
            1.0,
            hidden_states=hidden,
            gate_weight=weight,
            logical_to_physical_map=to_physical,
            logical_replica_counts=logical_count,
        )
    with pytest.raises(ValueError, match="ep_rank is required"):
        bf16_mega_gate(
            hidden,
            weight,
            4,
            renormalize=True,
            routed_scaling_factor=1.0,
            ep_rank=None,
            to_physical_map=to_physical,
            logical_count=logical_count,
        )
    # Selection resolves to deep_gemm_bf16_mega_gate by default on SM100.
    w, idx = moe_topk(
        None,
        4,
        "sqrt_softplus",
        "topk",
        True,
        1.0,
        hidden_states=hidden,
        gate_weight=weight,
    )
    assert w.shape == (4, 4) and idx.shape == (4, 4)


def test_bf16_mega_gate_heuristic_warmup_covers_all_configs() -> None:
    for hidden, experts, num_sms, deterministic in (
        (5120, 384, 148, False),
        (5120, 128, 148, False),
        (5120, 384, 132, False),
        (1024, 64, 148, False),
        (512, 64, 148, True),
    ):
        max_tokens = 8192
        warm = mega_gate_warmup_token_counts(
            max_tokens,
            hidden=hidden,
            num_experts=experts,
            num_sms=num_sms,
            deterministic=deterministic,
        )
        cands = _mega_gate_candidates(hidden, experts, num_sms, deterministic)
        warmed_configs = {
            _mega_gate_heuristic_key(t, cands, num_sms, deterministic) for t in warm
        }
        all_configs = {
            _mega_gate_heuristic_key(t, cands, num_sms, deterministic)
            for t in range(1, max_tokens + 1)
        }
        assert len(warm) == len(warmed_configs)
        assert warmed_configs == all_configs


def test_bf16_mega_gate_unpadded_mask_and_exact_out() -> None:
    torch.manual_seed(29)
    hidden, experts, top_k = 512, 64, 4
    gate_weight = (torch.randn(experts, hidden, device="cuda") * 0.1).bfloat16()
    bias = torch.randn(experts, device="cuda") * 0.05

    for tokens in (1, 17, 48, 96, 300):
        hidden_states = (torch.randn(tokens, hidden, device="cuda") * 0.1).bfloat16()
        mask = torch.tensor([i % 3 != 0 for i in range(tokens)], device="cuda")
        out_w = torch.empty(tokens, top_k, device="cuda", dtype=torch.float32)
        out_idx = torch.empty(tokens, top_k, device="cuda", dtype=torch.int64)
        got_w, got_idx = bf16_mega_gate(
            hidden_states,
            gate_weight,
            top_k,
            renormalize=True,
            routed_scaling_factor=1.5,
            ep_rank=0,
            correction_bias=bias,
            mask=mask,
            out=(out_w, out_idx),
        )
        assert got_w.data_ptr() == out_w.data_ptr()
        assert got_idx.data_ptr() == out_idx.data_ptr()
        assert (got_idx[~mask] == -1).all()
        assert (got_w[~mask] == 0).all()

        if mask.any():
            ref_w, ref_idx = _reference_sqrt_softplus_topk(
                hidden_states[mask],
                gate_weight,
                top_k,
                routed_scaling_factor=1.5,
                correction_bias=bias,
            )
            got_cw, got_ci = _canonicalize_routes(got_w[mask], got_idx[mask])
            ref_cw, ref_ci = _canonicalize_routes(ref_w, ref_idx)
            torch.testing.assert_close(got_ci, ref_ci, atol=0, rtol=0)
            torch.testing.assert_close(got_cw, ref_cw, atol=2e-3, rtol=2e-3)


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_bf16_mega_gate_cuda_graph_replay_and_workspace_region() -> None:
    torch.manual_seed(11)
    tokens_a, tokens_b, hidden, experts, top_k = 16, 8, 512, 64, 8
    weight = (torch.randn(experts, hidden, device="cuda") * 0.1).bfloat16()
    hidden_a = (torch.randn(tokens_a, hidden, device="cuda") * 0.1).bfloat16()
    hidden_b = (torch.randn(tokens_b, hidden, device="cuda") * 0.1).bfloat16()
    bias = torch.randn(experts, device="cuda") * 0.05
    out_w_a = torch.empty(tokens_a, top_k, device="cuda", dtype=torch.float32)
    out_idx_a = torch.empty(tokens_a, top_k, device="cuda", dtype=torch.int64)
    out_w_b = torch.empty(tokens_b, top_k, device="cuda", dtype=torch.float32)
    out_idx_b = torch.empty(tokens_b, top_k, device="cuda", dtype=torch.int64)

    entered: list[bool] = []

    @contextmanager
    def _track_workspace():
        entered.append(True)
        yield

    set_mega_gate_workspace_region(_track_workspace)
    cold_stream = torch.cuda.Stream()
    stream = torch.cuda.Stream()
    try:
        # Graph capture on an uninitialized stream must fail closed so DeepGEMM
        # never allocates its per-stream barrier inside the graph pool.
        cold_key = (
            torch.cuda.current_device(),
            int(cold_stream.cuda_stream),
        )
        cold_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold_graph, stream=cold_stream):
            with pytest.raises(RuntimeError, match="before graph capture"):
                moe_topk(
                    None,
                    top_k,
                    "sqrt_softplus",
                    "topk",
                    True,
                    1.0,
                    correction_bias=bias,
                    hidden_states=hidden_a,
                    gate_weight=weight,
                    out=(out_w_a, out_idx_a),
                    topk_weights_dtype=torch.float32,
                    topk_indices_dtype=torch.int64,
                    override="deep_gemm_bf16_mega_gate",
                )
        assert cold_key not in mega_gate_mod._INITIALIZED_STREAMS
        assert len(entered) == 0

        # First uncaptured call on a fresh stream invokes the workspace region;
        # later calls on the same stream do not re-enter it.
        with torch.cuda.stream(stream):
            moe_topk(
                None,
                top_k,
                "sqrt_softplus",
                "topk",
                True,
                1.0,
                correction_bias=bias,
                hidden_states=hidden_a,
                gate_weight=weight,
                out=(out_w_a, out_idx_a),
                topk_weights_dtype=torch.float32,
                topk_indices_dtype=torch.int64,
                override="deep_gemm_bf16_mega_gate",
            )
            moe_topk(
                None,
                top_k,
                "sqrt_softplus",
                "topk",
                True,
                1.0,
                correction_bias=bias,
                hidden_states=hidden_b,
                gate_weight=weight,
                out=(out_w_b, out_idx_b),
                topk_weights_dtype=torch.float32,
                topk_indices_dtype=torch.int64,
                override="deep_gemm_bf16_mega_gate",
            )
        assert len(entered) == 1

        graph_a = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_a, stream=stream):
            moe_topk(
                None,
                top_k,
                "sqrt_softplus",
                "topk",
                True,
                1.0,
                correction_bias=bias,
                hidden_states=hidden_a,
                gate_weight=weight,
                out=(out_w_a, out_idx_a),
                topk_weights_dtype=torch.float32,
                topk_indices_dtype=torch.int64,
                override="deep_gemm_bf16_mega_gate",
            )
        graph_b = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_b, stream=stream):
            moe_topk(
                None,
                top_k,
                "sqrt_softplus",
                "topk",
                True,
                1.0,
                correction_bias=bias,
                hidden_states=hidden_b,
                gate_weight=weight,
                out=(out_w_b, out_idx_b),
                topk_weights_dtype=torch.float32,
                topk_indices_dtype=torch.int64,
                override="deep_gemm_bf16_mega_gate",
            )
        assert len(entered) == 1

        # Replay graph_a three times with changing inputs, then interleave A/B replays.
        for replay_idx in range(3):
            hidden_a.normal_(mean=0.0, std=0.1 * (replay_idx + 1))
            stream.synchronize()
            graph_a.replay()
            stream.synchronize()
            eager_w, eager_idx = bf16_mega_gate(
                hidden_a,
                weight,
                top_k,
                renormalize=True,
                routed_scaling_factor=1.0,
                ep_rank=None,
                correction_bias=bias,
            )
            torch.testing.assert_close(out_idx_a, eager_idx, atol=0, rtol=0)
            torch.testing.assert_close(out_w_a, eager_w, atol=0, rtol=0)

        for step in range(4):
            if step % 2 == 0:
                hidden_a.normal_()
                stream.synchronize()
                graph_a.replay()
                stream.synchronize()
                eager_w, eager_idx = bf16_mega_gate(
                    hidden_a,
                    weight,
                    top_k,
                    renormalize=True,
                    routed_scaling_factor=1.0,
                    ep_rank=None,
                    correction_bias=bias,
                )
                torch.testing.assert_close(out_idx_a, eager_idx, atol=0, rtol=0)
                torch.testing.assert_close(out_w_a, eager_w, atol=0, rtol=0)
            else:
                hidden_b.normal_()
                stream.synchronize()
                graph_b.replay()
                stream.synchronize()
                eager_w, eager_idx = bf16_mega_gate(
                    hidden_b,
                    weight,
                    top_k,
                    renormalize=True,
                    routed_scaling_factor=1.0,
                    ep_rank=None,
                    correction_bias=bias,
                )
                torch.testing.assert_close(out_idx_b, eager_idx, atol=0, rtol=0)
                torch.testing.assert_close(out_w_b, eager_w, atol=0, rtol=0)
        assert deep_gemm_moe._HAS_MEGA_GATE
    finally:
        set_mega_gate_workspace_region(nullcontext)


_MEGA_GATE_WARMUP_SCRIPT = """
import sys
from pathlib import Path
import torch
from tokenspeed_kernel.ops.moe import mega_gate_warmup_token_counts, moe_topk

cache = Path(sys.argv[1])
hidden, experts, top_k = 5120, 256, 8
weight = (torch.randn(experts, hidden, device="cuda") * 0.05).bfloat16()
bias = torch.randn(experts, device="cuda")
image_bias = torch.randn(experts, device="cuda")
to_physical = torch.arange(experts, device="cuda", dtype=torch.int32).unsqueeze(-1)
logical_count = torch.ones(experts, device="cuda", dtype=torch.int32)

def _run(tokens: int) -> None:
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    image_mask = torch.zeros(tokens, device="cuda", dtype=torch.bool)
    moe_topk(
        None,
        top_k,
        "sqrt_softplus",
        "topk",
        True,
        2.5,
        correction_bias=bias,
        hidden_states=x,
        gate_weight=weight,
        override="deep_gemm_bf16_mega_gate",
    )
    moe_topk(
        None,
        top_k,
        "sqrt_softplus",
        "topk",
        True,
        2.5,
        correction_bias=bias,
        hidden_states=x,
        gate_weight=weight,
        image_bias=image_bias,
        image_token_mask=image_mask,
        logical_to_physical_map=to_physical,
        logical_replica_counts=logical_count,
        ep_rank=1,
        override="deep_gemm_bf16_mega_gate",
    )

num_sms = torch.cuda.get_device_properties(0).multi_processor_count
for warm_tokens in mega_gate_warmup_token_counts(
    2048,
    hidden=hidden,
    num_experts=experts,
    num_sms=num_sms,
):
    _run(warm_tokens)
torch.cuda.synchronize()
compiled = {str(p) for p in cache.rglob("*")}

for serve_tokens in (2, 3, 8, 17, 48, 96, 192, 288, 300, 384, 432, 480, 528, 576, 720, 777, 1024, 1500):
    _run(serve_tokens)
torch.cuda.synchronize()
new = sorted({str(p) for p in cache.rglob("*")} - compiled)
assert not new, f"unwarmed mega_gate compiled {new}"
"""


def test_warmup_mega_gate_leaves_nothing_to_compile(tmp_path: Path) -> None:
    env = {**os.environ, "DG_JIT_CACHE_DIR": str(tmp_path / "jit")}
    python_path = [str(Path(__file__).resolve().parents[4] / "python")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    result = subprocess.run(
        [sys.executable, "-c", _MEGA_GATE_WARMUP_SCRIPT, str(tmp_path / "jit")],
        env=env,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
