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

"""Public fused-all-reduce input contracts and backend dispatch."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

api = importlib.import_module("tokenspeed_kernel.ops.communication.allreduce_fusion")
backend_module = importlib.import_module(
    "tokenspeed_kernel.thirdparty.flashinfer.allreduce_fusion"
)


@pytest.fixture
def flashinfer_modules(monkeypatch):
    prefix = "flashinfer.comm.mnnvl_cutedsl"
    ht = SimpleNamespace(
        HTProtocol=Mock(),
        HT_FINALIZE_GB300_H3584_K16=object(),
        HT_ALL_REDUCE_GB300_H3584=object(),
    )
    workspace = Mock()
    modules = {
        "flashinfer.comm": SimpleNamespace(allreduce_fusion=Mock()),
        "flashinfer.comm.allreduce": SimpleNamespace(AllReduceFusionPattern=Mock()),
        "flashinfer.comm.mnnvl": SimpleNamespace(is_multicast_supported=Mock()),
        "flashinfer.comm.mnnvl_cutedsl_ar": SimpleNamespace(
            MNNVLCuteDSLAllReduceFusionWorkspace=workspace
        ),
        f"{prefix}.config": SimpleNamespace(
            KernelTarget=lambda protocol, preset: (protocol, preset),
            MNNVLCuteDSLConfig=SimpleNamespace,
            MRangeDispatch=SimpleNamespace,
            ProtocolKind=SimpleNamespace(LL="ll", BT="bt"),
            StaticProfile=SimpleNamespace,
        ),
        f"{prefix}.kernel_bt.protocol": SimpleNamespace(
            BTAllReduceTuning=SimpleNamespace,
            BTCollectiveTuning=SimpleNamespace,
            BTFinalizeTuning=SimpleNamespace,
        ),
        f"{prefix}.kernel_ll.protocol": SimpleNamespace(
            LLAllReduceTuning=SimpleNamespace,
            LLCollectiveTuning=SimpleNamespace,
            LLFinalizeTuning=SimpleNamespace,
        ),
        f"{prefix}.kernel_ht": SimpleNamespace(protocol=ht),
        f"{prefix}.kernel_ht.protocol": ht,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return ht, workspace


@pytest.mark.parametrize("tp_size", [4, 8, 16])
@pytest.mark.parametrize("capacity", [1024, 16384])
def test_backend_uses_upstream_ht_with_unchanged_capacity(
    monkeypatch, flashinfer_modules, tp_size, capacity
):
    ht, workspace = flashinfer_modules
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "empty", Mock())
    monkeypatch.setattr(torch, "zeros", Mock())
    group = SimpleNamespace(size=lambda: tp_size, rank=lambda: tp_size - 1)
    backend = backend_module.MNNVLAllReduceFusionBackend(
        group, 3584, 16, capacity, 1e-5
    )
    assert workspace.call_args.kwargs["max_token_num"] == 1024
    profile = workspace.call_args.kwargs["config"].profiles[0]
    assert profile.finalize_routes.upper_bounds == (32, 1024)
    assert profile.all_reduce_routes.upper_bounds == (32, 1024)
    assert backend._ht_finalize_tuning is ht.HT_FINALIZE_GB300_H3584_K16
    assert backend._ht_allreduce_tuning is ht.HT_ALL_REDUCE_GB300_H3584
    if capacity <= 1024:
        ht.HTProtocol.assert_not_called()
        assert backend._ht is None
    else:
        ht.HTProtocol.assert_called_once_with(
            hidden_size=3584,
            top_k=16,
            tp_size=tp_size,
            rank=tp_size - 1,
            capacity_m=capacity,
            rms_epsilon=1e-5,
            routed_scaling_factor=1.0,
            weight_bias=0.0,
            include_shared_expert=False,
            add_residual=False,
            write_residual_output=False,
            finalize_tunings=(ht.HT_FINALIZE_GB300_H3584_K16,),
            all_reduce_tunings=(ht.HT_ALL_REDUCE_GB300_H3584,),
            group=group,
        )
        assert backend._ht is ht.HTProtocol.return_value


def test_support_probe_rejects_flashinfer_without_h3584_presets(
    monkeypatch, flashinfer_modules
):
    ht, _ = flashinfer_modules
    del ht.HT_FINALIZE_GB300_H3584_K16
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    assert (
        backend_module.allreduce_fusion_support_error(
            object(), 3584, 16, 8192, torch.bfloat16
        )
        == "FlashInfer H3584 HT presets are unavailable"
    )


def inputs():
    backend = SimpleNamespace(device=torch.device("cpu"))
    kernel = SimpleNamespace(
        impl=Mock(return_value=torch.ones(2, 8, dtype=torch.bfloat16))
    )
    workspace = api.AllReduceFusionWorkspace(backend, kernel, 8, 2, 2048)
    x = torch.ones(4, 8, dtype=torch.bfloat16)
    gamma = torch.ones(8, dtype=torch.bfloat16)
    weights = torch.ones(4, dtype=torch.bfloat16)
    indices = torch.arange(4, dtype=torch.int32)
    return workspace, x, gamma, weights, indices


@pytest.mark.parametrize("finalize", [False, True])
def test_input_patterns_preserve_route_metadata(finalize):
    workspace, x, gamma, weights, indices = inputs()
    if not finalize:
        x, weights, indices = x[:2], None, None
    pattern = (
        api.AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM
        if finalize
        else api.AllReduceFusionPattern.ALLREDUCE_RMSNORM
    )
    api.allreduce_fusion(
        x,
        workspace,
        pattern=pattern,
        rms_gamma=gamma,
        num_tokens=2,
        expert_weights=weights,
        expanded_idx_to_permuted_idx=indices,
    )
    args = workspace.kernel.impl.call_args.args
    assert args[2] is pattern
    if finalize:
        torch.testing.assert_close(args[5], weights.view(2, 2), rtol=0, atol=0)
        torch.testing.assert_close(args[6], indices.view(2, 2), rtol=0, atol=0)
    else:
        assert args[5:] == (None, None)


@pytest.mark.parametrize("finalize", [False, True])
def test_route_metadata_must_match_the_input_pattern(finalize):
    workspace, x, gamma, weights, indices = inputs()
    if finalize:
        weights, indices = None, None
    else:
        x = x[:2]
    with pytest.raises(ValueError):
        api.allreduce_fusion(
            x,
            workspace,
            pattern=(
                api.AllReduceFusionPattern.MOE_FINALIZE_ALLREDUCE_RMSNORM
                if finalize
                else api.AllReduceFusionPattern.ALLREDUCE_RMSNORM
            ),
            rms_gamma=gamma,
            num_tokens=2,
            expert_weights=weights,
            expanded_idx_to_permuted_idx=indices,
        )
    workspace.kernel.impl.assert_not_called()


@pytest.mark.parametrize("m", [1024, 1025])
@pytest.mark.parametrize("finalize", [False, True])
def test_flashinfer_bt_to_ht_dispatch_boundary(m, finalize):
    backend = object.__new__(backend_module.MNNVLAllReduceFusionBackend)
    backend.output = torch.empty(2048, 8)
    backend.rms_eps = 1e-5
    backend._workspace = object()
    backend._patterns = SimpleNamespace(
        kARResidualRMSNorm=1, kMoEFinalizeARResidualRMSNorm=7
    )
    result = torch.ones(m, 8)
    backend._allreduce_fusion = Mock(return_value=result)
    backend._ht_finalize_tuning = "finalize"
    backend._ht_allreduce_tuning = "allreduce"
    final = Mock(return_value=(result, None))
    reduced = Mock(return_value=(result, None))
    backend._ht = SimpleNamespace(
        state=object(),
        finalize_kernels={"finalize": final},
        all_reduce_kernels={"allreduce": reduced},
    )
    x = torch.ones(m, 8)
    gamma = torch.ones(8)
    weights = torch.ones(m, 2) if finalize else None
    indices = torch.zeros(m, 2, dtype=torch.int32) if finalize else None
    assert backend.run(x, gamma, m, finalize, weights, indices) is result
    if m <= 1024:
        kwargs = backend._allreduce_fusion.call_args.kwargs
        assert kwargs["pattern"] == (7 if finalize else 1)
        assert kwargs["shared_expert_output"] is None
        assert kwargs["residual_in"] is None
        assert kwargs["workspace"] is backend._workspace
        assert kwargs["launch_with_pdl"] is True
        final.assert_not_called()
        reduced.assert_not_called()
    else:
        backend._allreduce_fusion.assert_not_called()
        selected = final if finalize else reduced
        selected.assert_called_once()
        assert selected.call_args.args[0] is x
