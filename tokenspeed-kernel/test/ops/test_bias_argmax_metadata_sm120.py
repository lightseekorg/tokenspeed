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

"""GPU checks for current descriptor handoff and selection override compatibility."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch
from tokenspeed_kernel.ops.sampling import bias_argmax as ops
from tokenspeed_kernel.ops.sampling.triton import bias_argmax as kernels
from tokenspeed_kernel.selection import SelectedKernel


def test_native_pair_warm_launch_and_dynamic_hooks(monkeypatch):
    monkeypatch.setattr(kernels, "_LAUNCHERS", {})
    original = kernels.NativeLaunchPair
    calls = []

    class ObservedPair:
        def __init__(self, *args):
            self.pair = original(*args)

        def run(self, *args):
            calls.append(args[-1])
            return self.pair.run(*args)

    monkeypatch.setattr(kernels, "NativeLaunchPair", ObservedPair)
    logits, bias, out, workspace = inputs(30)
    assert ops.try_bias_argmax(logits, bias, out, workspace, 11)
    assert not calls
    assert any(value[2] is not None for value in kernels._LAUNCHERS.values())
    assert ops.try_bias_argmax(logits, bias, out, workspace, 11)
    assert len(calls) == 1
    torch.testing.assert_close(
        out, (logits + bias).max(-1).indices + 11, rtol=0, atol=0
    )
    hooks = []
    monkeypatch.setattr(
        kernels.triton.knobs.runtime,
        "launch_enter_hook",
        lambda metadata: hooks.append(metadata),
    )
    assert ops.try_bias_argmax(logits, bias, out, workspace, 11)
    assert len(calls) == 1 and len(hooks) == 2
    torch.testing.assert_close(
        out, (logits + bias).max(-1).indices + 11, rtol=0, atol=0
    )


def test_native_pair_rejects_unreviewed_scratch_launches_and_ptx_abi():
    def kernel(pointer_count):
        return SimpleNamespace(
            run=SimpleNamespace(
                global_scratch_size=0,
                profile_scratch_size=0,
                gsan_enabled=False,
                launch_cooperative_grid=False,
                launch_pdl=False,
            ),
            metadata=SimpleNamespace(num_ctas=1, num_warps=4),
            asm={
                "ptx": ".visible .entry fake("
                + ",\n".join(
                    ".param .u64 .ptr .global a" + str(i) for i in range(pointer_count)
                )
                + ")"
            },
        )

    partial, final = kernel(6), kernel(5)
    for flag in (
        "global_scratch_size",
        "profile_scratch_size",
        "gsan_enabled",
        "launch_cooperative_grid",
        "launch_pdl",
    ):
        with mock.patch.object(partial.run, flag, 1):
            assert kernels._make_native_pair(partial, final) is None
    assert kernels._make_native_pair(kernel(5), final) is None
    assert kernels._make_native_pair(partial, kernel(4)) is None


def inputs(seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        (3, 8197), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    bias = torch.randn_like(logits)
    out = torch.empty((3, 5), dtype=torch.int64, device="cuda")[:, 2]
    workspace = ops.create_bias_argmax_workspace(3, 8197, "cuda")
    return logits, bias, out, workspace


def test_live_hook_chains_add_remove_and_restore_native_path(monkeypatch):
    monkeypatch.setattr(kernels, "_LAUNCHERS", {})
    enter = kernels.triton.knobs.HookChain()
    leave = kernels.triton.knobs.HookChain(reversed=True)
    monkeypatch.setattr(kernels.triton.knobs.runtime, "launch_enter_hook", enter)
    monkeypatch.setattr(kernels.triton.knobs.runtime, "launch_exit_hook", leave)
    original = kernels.NativeLaunchPair
    native_calls = []

    class ObservedPair:
        def __init__(self, *args):
            self.pair = original(*args)

        def run(self, *args):
            native_calls.append(1)
            return self.pair.run(*args)

    monkeypatch.setattr(kernels, "NativeLaunchPair", ObservedPair)
    logits, bias, out, workspace = inputs(31)

    def invoke():
        assert ops.try_bias_argmax(logits, bias, out, workspace, 13)
        torch.testing.assert_close(
            out, (logits + bias).max(-1).indices + 13, rtol=0, atol=0
        )

    invoke()
    invoke()
    assert len(native_calls) == 1
    observed = []
    callback = lambda metadata: observed.append(metadata)
    for chain in (enter, leave):
        previous = len(native_calls)
        chain.add(callback)
        invoke()
        assert len(native_calls) == previous and len(observed) == 2
        chain.remove(callback)
        invoke()
        assert len(native_calls) == previous + 1
        observed.clear()


def test_current_metadata_handoff_and_fresh_storage(monkeypatch):
    original = kernels.bias_argmax_sm120._from_checked_metadata
    seen = []

    def record(logits, bias, out, workspace, offset, metadata):
        codes = {dtype: code for code, dtype in ops._SCALAR_DTYPES.items()}
        assert metadata[:6] == (
            3,
            8197,
            logits.device.index,
            codes[logits.dtype],
            codes[bias.dtype],
            codes[out.dtype],
        )
        assert metadata[6:10] == (
            logits.stride(),
            bias.stride(),
            out.stride(0),
            workspace.values.stride(0),
        )
        assert metadata[10] == tuple(
            t.data_ptr() % 16
            for t in (logits, bias, out, workspace.values, workspace.indices)
        )
        seen.append((logits.data_ptr(), bias.data_ptr(), out.data_ptr()))
        return original(logits, bias, out, workspace, offset, metadata)

    monkeypatch.setattr(kernels.bias_argmax_sm120, "_from_checked_metadata", record)
    keep_alive = []
    for seed in (13, 14):
        tensors = inputs(seed)
        keep_alive.append(tensors)
        logits, bias, out, workspace = tensors
        assert ops.try_bias_argmax(logits, bias, out, workspace, 29)
        expected = (logits + bias.to(logits.dtype)).max(-1).indices + 29
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert len(seen) == 2 and all(a != b for a, b in zip(seen[0], seen[1]))


def test_selected_override_keeps_original_five_argument_protocol():
    logits, bias, out, workspace = inputs(15)
    calls = []

    def alternate(a, b, output, scratch, offset):
        calls.append(
            (a is logits, b is bias, output is out, scratch is workspace, offset)
        )
        output.copy_((a + b.to(a.dtype)).max(-1).indices + offset)

    with mock.patch.object(
        ops, "select_kernel", return_value=SelectedKernel("alternate", alternate)
    ):
        assert ops.try_bias_argmax(logits, bias, out, workspace, 31)
    assert calls == [(True, True, True, True, 31)]
    torch.testing.assert_close(
        out, (logits + bias).max(-1).indices + 31, rtol=0, atol=0
    )


def test_workspace_live_resize_is_rejected_before_prepared_launch(monkeypatch):
    logits, bias, out, workspace = inputs(16)
    workspace.values.resize_(1, 1)
    out.fill_(-7)
    called = mock.Mock()
    monkeypatch.setattr(kernels.bias_argmax_sm120, "_from_checked_metadata", called)
    assert not ops.try_bias_argmax(logits, bias, out, workspace, 0)
    called.assert_not_called()
    assert torch.equal(out, torch.full_like(out, -7))


def test_workspace_architecture_is_bound_once_but_live_inputs_change():
    logits, bias, out, workspace = inputs(17)
    assert workspace._device == logits.device
    assert workspace._capability == (12, 0)
    assert ops.try_bias_argmax(logits, bias, out, workspace, 0)
    with mock.patch.object(
        torch.cuda,
        "get_device_capability",
        side_effect=AssertionError("unexpected per-call architecture query"),
    ):
        for value in (0.0, 3.0):
            logits.fill_(value)
            bias.zero_()
            bias[:, 37] = 10
            assert ops.try_bias_argmax(logits, bias, out, workspace, 9)
            torch.testing.assert_close(out, torch.full_like(out, 46), rtol=0, atol=0)


def test_workspace_replacement_rebinds_hardware_and_rejects_unsupported():
    logits, bias, out, workspace = inputs(18)
    out.fill_(-19)
    cpu = replace(workspace, values=workspace.values.cpu())
    assert cpu._device.type == "cpu" and cpu._capability is None
    assert not ops.try_bias_argmax(logits, bias, out, cpu, 0)
    with mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        unsupported = replace(workspace)
    assert unsupported._capability == (9, 0)
    assert not ops.try_bias_argmax(logits, bias, out, unsupported, 0)
    assert torch.equal(out, torch.full_like(out, -19))
    restored = replace(cpu, values=workspace.values)
    assert restored._capability == (12, 0)
    assert ops.try_bias_argmax(logits, bias, out, restored, 0)
    torch.testing.assert_close(out, (logits + bias).max(-1).indices, rtol=0, atol=0)


def test_bound_device_does_not_bypass_current_device_or_tensor_validation():
    logits, bias, out, workspace = inputs(19)
    out.fill_(-23)
    with mock.patch.object(
        torch.cuda, "current_device", return_value=logits.device.index + 1
    ):
        assert not ops.try_bias_argmax(logits, bias, out, workspace, 0)
    invalid = ops.BiasArgmaxWorkspace(values=None, indices=workspace.indices)
    assert invalid._device is None and invalid._capability is None
    assert not ops.try_bias_argmax(logits, bias, out, invalid, 0)
    assert torch.equal(out, torch.full_like(out, -23))


def test_native_metadata_preserves_exact_python_integer_bounds():
    logits, bias, out, workspace = inputs(20)
    out.fill_(-17)
    for offset in (True, 3.0, -(2**100), 2**100):
        assert not ops.try_bias_argmax(logits, bias, out, workspace, offset)
        assert torch.equal(out, torch.full_like(out, -17))
    offset = 2**63 - 1 - (logits.shape[1] - 1)
    assert ops.try_bias_argmax(logits, bias, out, workspace, offset)
    torch.testing.assert_close(
        out, (logits + bias).max(-1).indices + offset, rtol=0, atol=0
    )


def test_native_validator_rejects_non_tensor_operands_without_raising():
    logits, bias, out, workspace = inputs(21)
    for slot in range(5):
        args = [logits, bias, out, workspace.values, workspace.indices]
        args[slot] = None
        assert (
            ops.validate_metadata(*args, logits.device.index, logits.device.index, 0)
            is None
        )


def test_native_tuple_reuse_never_reuses_eligibility_or_input_pointers():
    first = inputs(22)
    second = inputs(23)

    def metadata(tensors, offset):
        x, b, o, w = tensors
        return ops.validate_metadata(
            x, b, o, w.values, w.indices, x.device.index, x.device.index, offset
        )

    a = metadata(first, 0)
    b = metadata(second, 0)
    assert a is b
    assert all(not isinstance(value, torch.Tensor) for value in a)
    for tensors, value, index in ((first, 1.0, 37), (second, 2.0, 42)):
        x, bias, out, workspace = tensors
        x.fill_(value)
        bias.zero_()
        bias[:, index] = 10
        assert ops.try_bias_argmax(x, bias, out, workspace, 0)
        torch.testing.assert_close(out, torch.full_like(out, index), rtol=0, atol=0)
    assert metadata(second, 2**100) is None
    second[3].values.resize_(1, 1)
    assert metadata(second, 0) is None
