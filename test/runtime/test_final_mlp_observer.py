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

"""Observe and replay real Qwen3 suffix modules without changing inference.

CPU cases replace only GPU kernel entry points with the repository's numerical
reference. GPU cases use the actual selected kernels. Attention/cache execution
is outside these suffix tests; full-model integration uses the same public API.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.layernorm import torch_rmsnorm

from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.inspection.final_mlp import FinalMLPObserver
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.mlp import GatedMLPFusion
from tokenspeed.runtime.models.qwen3 import Qwen3ForCausalLM
from tokenspeed.runtime.utils.env import global_server_args_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")


def _cpu_norm(x, weight, eps, residual=None, out=None):
    result = torch_rmsnorm(x, weight, eps, residual)
    if residual is not None:
        # Exercise the aliasing of in-place fused norms, too: the observer
        # must preserve the pre-MLP residual before final norm rewrites it.
        normalized, total = result
        x.copy_(normalized)
        residual.copy_(total)
        return x, residual
    if out is not None:
        out.copy_(result)
        return out
    return result


def _model(monkeypatch, *, device, dtype, mode):
    monkeypatch.setitem(global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", mode)
    monkeypatch.setitem(global_server_args_dict, "logprob_order", "torch")
    monkeypatch.setitem(global_server_args_dict, "moe_combine_order", "rank")
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", False)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    model = Qwen3ForCausalLM(config, Mapping(rank=0, world_size=1)).to(
        device=device, dtype=dtype
    )
    torch.manual_seed(91)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0, std=0.15)
    model.eval()
    if device == "cpu":
        # These dispatchers require a GPU even for CPU tensors on a GPU host.
        # Keep the real MLP, projection, norm and logits processor classes.
        def mm(x, weight, *, bias=None, **kwargs):
            return F.linear(x, weight, bias)

        monkeypatch.setattr("tokenspeed.runtime.layers.dense.unquant.kernel_mm", mm)
        monkeypatch.setattr("tokenspeed.runtime.layers.logits_processor.kernel_mm", mm)
        model.model.norm.kernel = _cpu_norm
        model.model.layers[-1].post_attention_layernorm.kernel = _cpu_norm
    return model


def _observer(model, *, rows=3, logits_rows=1, **choices):
    return FinalMLPObserver(
        model,
        num_rows=rows,
        num_logits_rows=logits_rows,
        execution=choices.get("execution", "eager"),
        speculative=choices.get("speculative", False),
        monokernel=choices.get("monokernel", False),
    )


def _inputs(model, rows=3):
    weight = model.model.layers[-1].mlp.down_proj.weight
    torch.manual_seed(143)
    return tuple(
        torch.randn(rows, weight.shape[0], device=weight.device, dtype=weight.dtype)
        for _ in range(2)
    )


@torch.no_grad()
def _run_suffix(model, hidden, residual, *, gather):
    layer = model.model.layers[-1]
    normalized, residual = layer.post_attention_layernorm(
        hidden.clone(), residual.clone()
    )
    expected_residual = residual.clone()
    down = layer.mlp(normalized)
    if layer.comm_manager.layer_boundary_norm == "fused":
        normalized, _ = model.model.norm(down, residual)
    else:
        normalized = model.model.norm(down + residual)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        query_shard=None,
        gather_ids=gather,
    )
    output = model.logits_processor(
        torch.zeros(hidden.shape[0], dtype=torch.int64, device=hidden.device),
        normalized,
        model.lm_head,
        metadata,
    )
    return output.next_token_logits, expected_residual


@pytest.mark.parametrize("mode", ["fused", "unfused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_observation_and_noop_replay_preserve_logits(monkeypatch, mode, dtype):
    model = _model(monkeypatch, device="cpu", dtype=dtype, mode=mode)
    hidden, residual = _inputs(model)
    gather = torch.tensor([2])
    expected, expected_residual = _run_suffix(model, hidden, residual, gather=gather)
    with _observer(model) as observer:
        with observer.record():
            actual, _ = _run_suffix(model, hidden, residual, gather=gather)
        observation = observer.snapshot()
        replay = observer.replay(observation, ablate_channels=())
        assert torch.equal(actual, expected)
        assert torch.equal(observation.selected_logits, expected)
        assert torch.equal(observation.residual_before_mlp, expected_residual)
        assert torch.equal(replay.suffix_logits, expected)
        assert torch.equal(replay.recomputed_logits, expected)
        assert torch.equal(replay.intervened_logits, expected)
        assert torch.equal(replay.recomputed_mlp_output, observation.mlp_output)
        gate, up = observation.gate_up.float().chunk(2, dim=-1)
        assert torch.equal(observation.activation, (F.silu(gate) * up).to(dtype))


@pytest.mark.parametrize("mode", ["fused", "unfused"])
def test_channel_ablation_recomputes_down_and_preserves_state(monkeypatch, mode):
    model = _model(monkeypatch, device="cpu", dtype=torch.bfloat16, mode=mode)
    hidden, residual = _inputs(model)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    with _observer(model) as observer:
        with observer.record():
            _run_suffix(model, hidden, residual, gather=torch.tensor([1]))
        observation = observer.snapshot()
        original_activation = observation.activation.clone()
        original_residual = observation.residual_before_mlp.clone()
        original_down = observation.mlp_output.clone()
        replay = observer.replay(observation, ablate_channels=(0, 3, 15))
        intervened = observation.activation.clone()
        intervened[:, [0, 3, 15]] = 0
        expected = F.linear(intervened, model.model.layers[-1].mlp.down_proj.weight)
        assert torch.equal(replay.intervened_mlp_output, expected)
        full = observer.replay(
            observation, ablate_channels=tuple(range(intervened.shape[1]))
        )
        assert torch.count_nonzero(full.intervened_mlp_output) == 0
        residual_hidden = torch_rmsnorm(
            original_residual,
            model.model.norm.weight,
            model.model.norm.variance_epsilon,
        )
        residual_logits = F.linear(residual_hidden, model.lm_head.weight)[
            [1], : model.config.vocab_size
        ]
        assert torch.equal(full.intervened_logits, residual_logits)
        assert torch.equal(observation.activation, original_activation)
        assert torch.equal(observation.residual_before_mlp, original_residual)
        assert torch.equal(observation.mlp_output, original_down)
    assert all(
        torch.equal(value, weights[name]) for name, value in model.state_dict().items()
    )


def test_snapshots_survive_reuse_and_failed_records(monkeypatch):
    model = _model(monkeypatch, device="cpu", dtype=torch.float32, mode="fused")
    hidden, residual = _inputs(model)
    with _observer(model) as observer:
        with observer.record():
            _run_suffix(model, hidden, residual, gather=torch.tensor([0]))
        first = observer.snapshot()
        saved = first.activation.clone()
        with observer.record():
            _run_suffix(model, hidden * 2, residual, gather=torch.tensor([2]))
        second = observer.snapshot()
        assert first.logit_rows.tolist() == [0]
        assert second.logit_rows.tolist() == [2]
        assert torch.equal(first.activation, saved)
        assert not torch.equal(second.activation, saved)
        second.activation.zero_()
        assert torch.count_nonzero(observer.snapshot().activation) > 0
        with pytest.raises(RuntimeError, match="bypassed"):
            with observer.record():
                pass
        with pytest.raises(RuntimeError, match="No completed"):
            observer.snapshot()
        assert torch.equal(first.activation, saved)


def test_effective_identity_rows_match_logits_processor(monkeypatch):
    model = _model(monkeypatch, device="cpu", dtype=torch.float32, mode="unfused")
    hidden, residual = _inputs(model)
    with _observer(model, logits_rows=3) as observer:
        # The existing processor treats a full-length gather as already
        # selected and does not reorder it. Record its actual semantics.
        with observer.record():
            output, _ = _run_suffix(
                model, hidden, residual, gather=torch.tensor([2, 0, 1])
            )
        observation = observer.snapshot()
        assert observation.logit_rows.tolist() == [0, 1, 2]
        assert torch.equal(
            observer.replay(observation, ablate_channels=()).suffix_logits, output
        )


@pytest.mark.parametrize(
    "choices", [{"execution": "captured"}, {"speculative": True}, {"monokernel": True}]
)
def test_unsupported_execution_is_rejected(monkeypatch, choices):
    model = _model(monkeypatch, device="cpu", dtype=torch.float32, mode="fused")
    with pytest.raises(ValueError, match="requires eager"):
        _observer(model, **choices)


def test_unsupported_model_layout_and_fusion_are_rejected(monkeypatch):
    model = _model(monkeypatch, device="cpu", dtype=torch.float32, mode="fused")
    model.mapping = Mapping(rank=0, world_size=2)
    with pytest.raises(ValueError, match="one device"):
        _observer(model)
    model.mapping = Mapping(rank=0, world_size=1)
    model.model.layers[-1].mlp.fusion = GatedMLPFusion.MXFP4_GEMM_SWIGLU
    with pytest.raises(ValueError, match="unfused"):
        _observer(model)


@pytest.mark.parametrize("channels", [(-1,), (128,), (1, 1), (True,), (1.5,)])
def test_invalid_channel_intervention_is_rejected(monkeypatch, channels):
    model = _model(monkeypatch, device="cpu", dtype=torch.float32, mode="fused")
    hidden, residual = _inputs(model)
    with _observer(model) as observer:
        with observer.record():
            _run_suffix(model, hidden, residual, gather=torch.tensor([2]))
        with pytest.raises(ValueError, match="unique in-range integers"):
            observer.replay(observer.snapshot(), ablate_channels=channels)


def test_bad_shape_and_multiple_forwards_invalidate_record(monkeypatch):
    model = _model(monkeypatch, device="cpu", dtype=torch.float32, mode="fused")
    hidden, residual = _inputs(model)
    with _observer(model) as observer:
        with pytest.raises(ValueError, match="prepared shape"):
            with observer.record():
                _run_suffix(model, hidden[:2], residual[:2], gather=torch.tensor([1]))
        with pytest.raises(RuntimeError, match="No completed"):
            observer.snapshot()
        with pytest.raises(RuntimeError, match="repeated"):
            with observer.record():
                _run_suffix(model, hidden, residual, gather=torch.tensor([1]))
                _run_suffix(model, hidden, residual, gather=torch.tensor([1]))
    observer.close()
    with pytest.raises(RuntimeError, match="attached"):
        with observer.record():
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", ["fused", "unfused"])
def test_gpu_observation_and_same_shape_suffix_replay(monkeypatch, mode):
    model = _model(monkeypatch, device="cuda", dtype=torch.bfloat16, mode=mode)
    hidden, residual = _inputs(model)
    gather = torch.tensor([2], device="cuda")
    for _ in range(3):
        _run_suffix(model, hidden, residual, gather=gather)
    expected, _ = _run_suffix(model, hidden, residual, gather=gather)
    with _observer(model) as observer:
        with observer.record():
            actual, _ = _run_suffix(model, hidden, residual, gather=gather)
        observation = observer.snapshot()
        replay = observer.replay(observation, ablate_channels=())
        assert torch.equal(actual, expected)
        assert torch.equal(observation.selected_logits, expected)
        assert torch.equal(replay.suffix_logits, expected)
        assert torch.equal(replay.recomputed_logits, expected)
        assert torch.equal(replay.intervened_logits, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
