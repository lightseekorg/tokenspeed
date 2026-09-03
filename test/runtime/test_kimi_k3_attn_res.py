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

"""Kimi-K3 AttnRes mixing + router numerics tests (cheap; kernel parity on GPU).

Covers the ``attn_res_fwd`` op the model routes AttnRes mixing through: the
torch fallback must match the reference ``modeling_kimi.py::_apply_attn_res``
math, the model wiring must slice candidates correctly, and (when a Blackwell
kernel build is present) the CUDA kernel must match the torch fallback.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import tokenspeed_kernel.ops.attn_res as attn_res_ops  # noqa: E402
from tokenspeed_kernel.ops.attn_res import attn_res_fwd  # noqa: E402
from tokenspeed_kernel.ops.attn_res.torch import torch_attn_res_fwd  # noqa: E402

from tokenspeed.runtime.layers.layernorm import RMSNorm  # noqa: E402
from tokenspeed.runtime.models import kimi_k3  # noqa: E402

_HIDDEN = 64
_BLOCKS = 4
_EPS = 1e-5


def _make_inputs(num_tokens: int, seed: int = 0):
    torch.manual_seed(seed)
    prefix_sum = torch.randn(num_tokens, _HIDDEN, dtype=torch.bfloat16)
    # Block-major [num_blocks, T, hidden], matching the model's scratch layout.
    block_residual = torch.randn(_BLOCKS, num_tokens, _HIDDEN, dtype=torch.bfloat16)
    norm = RMSNorm(_HIDDEN, eps=_EPS)
    norm.weight.data.uniform_(0.5, 1.5)
    proj = torch.nn.Linear(_HIDDEN, 1, bias=False)
    return prefix_sum, block_residual, proj, norm


def _reference_apply_attn_res(prefix_sum, block_residual, proj, norm):
    """Verbatim math from the checkpoint's modeling_kimi.py::_apply_attn_res,
    over block-major candidates [N, T, H] = blocks then the current stream."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(0)), dim=0)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)  # [N, T]
    probs = scores.softmax(0)
    return (probs.unsqueeze(-1) * v_float).sum(0).to(v.dtype)


class AttnResTests(unittest.TestCase):
    def test_batched_iris_reduce_consumes_attnres_combine(self):
        import tokenspeed_kernel.ops.communication.triton as triton_comm

        import tokenspeed.runtime.models.kimi_k3_comm as kimi_k3_comm

        group = object()
        state = SimpleNamespace(
            attn_ar_fusion_ok=False,
            mapping=SimpleNamespace(
                nprocs_per_node=8,
                attn=SimpleNamespace(tp_rank=0, tp_group=tuple(range(8))),
            ),
        )
        comm = kimi_k3_comm.K3AttnComm(state)
        partial = torch.randn(4, _HIDDEN, dtype=torch.bfloat16)
        prefix = torch.randn_like(partial)
        scratch = (object(), object(), object())
        mlp_wp = torch.randn(_HIDDEN, dtype=torch.bfloat16)
        out_weight = torch.randn(_HIDDEN, dtype=torch.bfloat16)
        combine = (scratch, object(), object(), out_weight, _EPS)
        expected_h = torch.randn_like(partial)
        expected_residual = torch.randn_like(partial)

        with (
            mock.patch.object(kimi_k3_comm, "_get_process_group", return_value=group),
            mock.patch.object(
                triton_comm,
                "allreduce_residual_attnres_combine_supported",
                return_value=True,
            ) as supported,
            mock.patch.object(
                triton_comm,
                "allreduce_residual_attnres_combine",
                return_value=(expected_h, expected_residual),
            ) as fused,
        ):
            residual, hidden = comm.attn_reduce(
                partial,
                prefix,
                combine,
                mlp_wp=mlp_wp,
            )

        self.assertIs(residual, expected_residual)
        self.assertIs(hidden, expected_h)
        supported.assert_called_once()
        self.assertIs(supported.call_args.args[0], partial)
        self.assertIs(supported.call_args.args[1], prefix)
        self.assertIs(supported.call_args.args[2], mlp_wp)
        fused.assert_called_once()

    def test_unsupported_iris_reduce_defers_attnres_combine(self):
        import tokenspeed_kernel.ops.communication.triton as triton_comm

        import tokenspeed.runtime.models.kimi_k3_comm as kimi_k3_comm

        group = object()
        state = SimpleNamespace(
            attn_ar_fusion_ok=False,
            mapping=SimpleNamespace(
                nprocs_per_node=8,
                attn=SimpleNamespace(tp_rank=0, tp_group=tuple(range(8))),
            ),
        )
        comm = kimi_k3_comm.K3AttnComm(state)
        partial = torch.randn(17, _HIDDEN, dtype=torch.bfloat16)
        prefix = torch.randn_like(partial)
        reduced = torch.randn_like(partial)
        combine = (
            (object(), object(), object()),
            object(),
            object(),
            torch.randn(_HIDDEN, dtype=torch.bfloat16),
            _EPS,
        )

        with (
            mock.patch.object(kimi_k3_comm, "_get_process_group", return_value=group),
            mock.patch.object(
                triton_comm,
                "allreduce_residual_attnres_combine_supported",
                return_value=False,
            ),
            mock.patch.object(
                kimi_k3_comm,
                "all_reduce",
                return_value=reduced,
            ) as fallback_reduce,
        ):
            residual, hidden = comm.attn_reduce(
                partial,
                prefix,
                combine,
                mlp_wp=torch.randn(_HIDDEN, dtype=torch.bfloat16),
            )

        torch.testing.assert_close(residual, prefix + reduced)
        self.assertIsNone(hidden)
        fallback_reduce.assert_called_once_with(partial, state.mapping.attn.tp_group)

    def test_torch_fallback_matches_reference(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(17)
        got = torch_attn_res_fwd(
            layer_residual=prefix_sum,
            block_residual=block_residual,
            res_weight=proj.weight.reshape(-1).to(torch.bfloat16),
            rms_weight=norm.weight.to(torch.bfloat16),
            eps=_EPS,
        )
        ref = _reference_apply_attn_res(prefix_sum, block_residual, proj, norm)
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_model_wiring_slices_valid_blocks(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(9, seed=2)
        got = kimi_k3._apply_attn_res(prefix_sum, block_residual, proj, norm, 2)
        ref = _reference_apply_attn_res(prefix_sum, block_residual[:2], proj, norm)
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_model_wiring_keeps_full_storage_only_for_snapshot_writes(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(3)
        with mock.patch.object(
            kimi_k3, "attn_res_fwd", return_value=prefix_sum
        ) as forward:
            kimi_k3._apply_attn_res(prefix_sum, block_residual, proj, norm, 2)
            read_blocks = forward.call_args.args[1]
            self.assertEqual(read_blocks.shape[0], 2)
            self.assertEqual(read_blocks.data_ptr(), block_residual.data_ptr())

            kimi_k3._apply_attn_res(
                prefix_sum,
                block_residual,
                proj,
                norm,
                2,
                block_write_idx=2,
            )
            self.assertIs(forward.call_args.args[1], block_residual)

    def test_zero_valid_blocks_is_identity(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(5, seed=3)
        got = kimi_k3._apply_attn_res(prefix_sum, block_residual, proj, norm, 0)
        torch.testing.assert_close(got, prefix_sum)

    def test_delta_update_and_block_write_match_reference(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(7, seed=7)
        delta = torch.randn_like(prefix_sum)
        original_prefix = prefix_sum.clone()
        original_blocks = block_residual.clone()

        got = attn_res_fwd(
            layer_residual=prefix_sum,
            block_residual=block_residual,
            res_weight=proj.weight.reshape(-1).to(torch.bfloat16),
            rms_weight=norm.weight.to(torch.bfloat16),
            eps=_EPS,
            delta=delta,
            num_valid_blocks=2,
            block_write_idx=2,
        )

        updated_prefix = (original_prefix + delta).to(torch.bfloat16)
        ref = _reference_apply_attn_res(updated_prefix, original_blocks[:2], proj, norm)
        torch.testing.assert_close(prefix_sum, updated_prefix, rtol=0, atol=0)
        torch.testing.assert_close(block_residual[2], updated_prefix, rtol=0, atol=0)
        torch.testing.assert_close(block_residual[:2], original_blocks[:2])
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_partial_snapshot_storage_is_a_dispatch_trait(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(3)
        selected = SimpleNamespace(impl=torch_attn_res_fwd)
        with mock.patch.object(
            attn_res_ops, "select_kernel", return_value=selected
        ) as select:
            self.assertFalse(
                attn_res_ops.attn_res_fwd_available(
                    prefix_sum,
                    block_residual,
                    proj.weight.reshape(-1).to(torch.bfloat16),
                    norm.weight.to(torch.bfloat16),
                    num_valid_blocks=2,
                )
            )

        self.assertTrue(select.call_args.kwargs["traits"]["partial_block_storage"])

    def test_specialized_availability_rejects_cpu_inputs(self):
        hidden = 7168
        prefix_sum = torch.zeros(1, hidden, dtype=torch.bfloat16)
        block_residual = torch.zeros(1, 1, hidden, dtype=torch.bfloat16)
        weight = torch.ones(hidden, dtype=torch.bfloat16)

        self.assertFalse(
            attn_res_ops.attn_res_fwd_available(
                prefix_sum,
                block_residual,
                weight,
                weight,
                out_norm_weight=weight,
            )
        )
        actual = attn_res_fwd(
            prefix_sum,
            block_residual,
            weight,
            weight,
            out_norm_weight=weight,
        )
        expected = torch_attn_res_fwd(
            layer_residual=prefix_sum,
            block_residual=block_residual,
            res_weight=weight,
            rms_weight=weight,
            eps=1e-6,
            out_norm_weight=weight,
        )
        torch.testing.assert_close(actual, expected)

    def test_fused_model_graph_matches_vllm_operation_order(self):
        for is_block_write_layer in (False, True):
            with self.subTest(is_block_write_layer=is_block_write_layer):
                events = []
                hidden_states = torch.ones(2, 4, dtype=torch.bfloat16)
                original_prefix = hidden_states.clone()
                block_residual = torch.zeros(3, 2, 4, dtype=torch.bfloat16)
                reduced = torch.full_like(hidden_states, 3)

                def apply_attn_res(
                    prefix,
                    blocks,
                    proj,
                    norm,
                    num_valid_blocks,
                    out_norm=None,
                    *,
                    delta=None,
                    block_write_idx=-1,
                ):
                    del proj, norm, out_norm
                    events.append(
                        ("attn_res", prefix, delta, num_valid_blocks, block_write_idx)
                    )
                    if delta is not None:
                        prefix.add_(delta)
                    if block_write_idx >= 0:
                        blocks[block_write_idx].copy_(prefix)
                    return torch.full_like(prefix, len(events))

                class SelfAttention:
                    def __call__(self, **kwargs):
                        events.append(("attention", kwargs["hidden_states"]))
                        return torch.full_like(hidden_states, 2)

                def reduce_attention(value, group):
                    del value, group
                    events.append(("all_reduce",))
                    return reduced

                def mlp(value):
                    events.append(("mlp", value))
                    return torch.full_like(value, 5)

                layer = SimpleNamespace(
                    is_block_write_layer=is_block_write_layer,
                    block_write_idx=1,
                    prev_valid_blocks=1,
                    self_attention_res_proj=object(),
                    self_attention_res_norm=object(),
                    input_layernorm=object(),
                    self_attn=SelfAttention(),
                    comm_manager=object(),
                    mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=object())),
                    mlp_res_proj=object(),
                    mlp_res_norm=object(),
                    post_attention_layernorm=object(),
                    is_moe_layer=False,
                    mlp=mlp,
                    _prepare_next_fallback_attnres_partial=mock.Mock(),
                )

                with (
                    mock.patch.object(kimi_k3, "_apply_attn_res", apply_attn_res),
                    mock.patch.object(kimi_k3, "all_reduce", reduce_attention),
                ):
                    result, actual_blocks = (
                        kimi_k3.KimiLinearDecoderLayer._forward_fused_attnres_graph(
                            layer,
                            positions=torch.empty(0),
                            hidden_states=hidden_states,
                            ctx=object(),
                            out_cache_loc=torch.empty(0),
                            block_residual=block_residual,
                        )
                    )

                self.assertEqual(
                    [event[0] for event in events],
                    ["attn_res", "attention", "all_reduce", "attn_res", "mlp"],
                )
                pre, post = events[0], events[3]
                self.assertIs(pre[1], hidden_states)
                self.assertIsNone(pre[2])
                self.assertEqual(pre[3], 1)
                self.assertEqual(pre[4], 1 if is_block_write_layer else -1)
                self.assertEqual(post[3], 2 if is_block_write_layer else 1)
                self.assertEqual(post[4], -1)
                self.assertIs(actual_blocks, block_residual)
                if is_block_write_layer:
                    self.assertIs(post[1], reduced)
                    self.assertIsNone(post[2])
                    torch.testing.assert_close(block_residual[1], original_prefix)
                    torch.testing.assert_close(result, reduced + 5)
                else:
                    self.assertIs(post[1], hidden_states)
                    self.assertIs(post[2], reduced)
                    torch.testing.assert_close(result, original_prefix + reduced + 5)

    def test_fused_model_graph_preserves_single_token_collective_path(self):
        weight = torch.empty(_HIDDEN, dtype=torch.bfloat16)
        norm = SimpleNamespace(weight=weight, variance_epsilon=_EPS)
        layer = SimpleNamespace(
            is_block_write_layer=False,
            block_write_idx=1,
            prev_valid_blocks=1,
            self_attention_res_proj=SimpleNamespace(weight=weight.reshape(1, -1)),
            self_attention_res_norm=norm,
            input_layernorm=norm,
            mlp_res_proj=SimpleNamespace(weight=weight.reshape(1, -1)),
            mlp_res_norm=norm,
            post_attention_layernorm=norm,
        )
        block_residual = torch.empty(2, 2, _HIDDEN, dtype=torch.bfloat16)

        with mock.patch.object(
            kimi_k3, "attn_res_fwd_available", return_value=True
        ) as available:
            self.assertFalse(
                kimi_k3.KimiLinearDecoderLayer._fused_attnres_graph_available(
                    layer,
                    torch.empty(1, _HIDDEN, dtype=torch.bfloat16),
                    block_residual[:, :1],
                )
            )
            available.assert_not_called()

            self.assertTrue(
                kimi_k3.KimiLinearDecoderLayer._fused_attnres_graph_available(
                    layer,
                    torch.empty(2, _HIDDEN, dtype=torch.bfloat16),
                    block_residual,
                )
            )
            self.assertEqual(available.call_count, 2)

    def test_fused_to_fallback_populates_next_split_partial(self):
        hidden_states = SimpleNamespace(shape=(4, _HIDDEN), is_cuda=True)
        block_residual = torch.zeros(3, 4, _HIDDEN, dtype=torch.bfloat16)
        attn_weight = torch.empty(_HIDDEN, dtype=torch.bfloat16)
        mlp_weight = torch.empty(_HIDDEN, dtype=torch.bfloat16)
        attn_scratch = (object(), object(), object())
        mlp_scratch = (object(), object(), object())

        def sliced_scratch(_, slot, num_tokens):
            self.assertEqual(num_tokens, 4)
            return attn_scratch if slot == 1 else mlp_scratch

        for next_fused in (False, True):
            for hoist_mlp in (False, True):
                with self.subTest(next_fused=next_fused, hoist_mlp=hoist_mlp):
                    next_layer = SimpleNamespace(
                        _attn_wp=attn_weight,
                        _mlp_wp=mlp_weight,
                        _mlp_slot=7,
                        self_attention_res_norm=SimpleNamespace(variance_epsilon=_EPS),
                        mlp_res_norm=SimpleNamespace(variance_epsilon=_EPS),
                        _fused_attnres_graph_available=mock.Mock(
                            return_value=next_fused
                        ),
                    )
                    layer = SimpleNamespace(
                        _next_attn_mix=(next_layer, 2),
                        _hoist_next_mlp=hoist_mlp,
                    )
                    with (
                        mock.patch.object(
                            kimi_k3, "_sliced_scratch", side_effect=sliced_scratch
                        ) as get_scratch,
                        mock.patch.object(kimi_k3, "attnres_partial") as partial,
                        mock.patch.object(
                            kimi_k3, "attnres_partial_dual"
                        ) as partial_dual,
                    ):
                        kimi_k3.KimiLinearDecoderLayer._prepare_next_fallback_attnres_partial(
                            layer, hidden_states, block_residual
                        )

                    next_layer._fused_attnres_graph_available.assert_called_once_with(
                        hidden_states, block_residual
                    )
                    if next_fused:
                        get_scratch.assert_not_called()
                        partial.assert_not_called()
                        partial_dual.assert_not_called()
                    elif hoist_mlp:
                        self.assertEqual(
                            get_scratch.call_args_list,
                            [
                                mock.call(hidden_states, 1, 4),
                                mock.call(hidden_states, 7, 4),
                            ],
                        )
                        partial.assert_not_called()
                        partial_dual.assert_called_once()
                        args = partial_dual.call_args.args
                        torch.testing.assert_close(args[0], block_residual[:2])
                        self.assertIs(args[1], mlp_weight)
                        self.assertIs(args[2], attn_weight)
                        self.assertEqual(args[3], _EPS)
                        self.assertIs(args[4], mlp_scratch)
                        self.assertIs(args[5], attn_scratch)
                    else:
                        get_scratch.assert_called_once_with(hidden_states, 1, 4)
                        partial.assert_called_once()
                        partial_dual.assert_not_called()
                        args = partial.call_args.args
                        torch.testing.assert_close(args[0], block_residual[:2])
                        self.assertIs(args[1], attn_weight)
                        self.assertEqual(args[2], _EPS)
                        self.assertIs(args[3], attn_scratch)

    def test_cuda_kernel_matches_torch_fallback(self):
        # Only runs where the Blackwell attn_res build is present (e.g. B300 CI).
        try:
            from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL
        except ImportError:
            _HAS_CUDA_KERNEL = False
        if not (_HAS_CUDA_KERNEL and torch.cuda.is_available()):
            self.skipTest("Blackwell attn_res kernel not available")
        from tokenspeed_kernel.ops.attn_res import attn_res_fwd

        torch.manual_seed(0)
        T, H, K = 128, 7168, 8  # kernel-eligible shape (H in supported set)
        dev = "cuda"
        prefix = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
        blocks = torch.randn(K, T, H, dtype=torch.bfloat16, device=dev)
        res_w = torch.randn(H, dtype=torch.bfloat16, device=dev)
        rms_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
        got = attn_res_fwd(prefix, blocks, res_w, rms_w, _EPS)  # cuda path
        ref = torch_attn_res_fwd(
            layer_residual=prefix,
            block_residual=blocks,
            res_weight=res_w,
            rms_weight=rms_w,
            eps=_EPS,
        )
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_cuda_kernel_single_token_dispatches_match_torch(self):
        """T=1 specialized dispatches.

        N in {1, 2, 4, 8, 12} at H=7168 routes to the single-CTA / split-K
        single-token kernels when no fused out-norm is requested, and to the
        online kernel (which fuses the norm) otherwise. N counts the layer
        residual, so K = N - 1 blocks. Cover both, plus N=13 (> N_MAX) which
        must stay on the torch fallback.
        """
        try:
            from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL
        except ImportError:
            _HAS_CUDA_KERNEL = False
        if not (_HAS_CUDA_KERNEL and torch.cuda.is_available()):
            self.skipTest("Blackwell attn_res kernel not available")
        from tokenspeed_kernel.ops.attn_res import attn_res_fwd

        torch.manual_seed(0)
        T, H = 1, 7168
        dev = "cuda"
        for K in (1, 3, 7, 11, 12):
            prefix = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
            blocks = torch.randn(K, T, H, dtype=torch.bfloat16, device=dev)
            res_w = torch.randn(H, dtype=torch.bfloat16, device=dev)
            rms_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
            got = attn_res_fwd(prefix, blocks, res_w, rms_w, _EPS)
            ref = torch_attn_res_fwd(
                layer_residual=prefix,
                block_residual=blocks,
                res_weight=res_w,
                rms_weight=rms_w,
                eps=_EPS,
            )
            torch.testing.assert_close(
                got, ref, atol=2e-2, rtol=2e-2, msg=f"N={K} no-norm"
            )
            out_norm_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
            got_n = attn_res_fwd(
                prefix, blocks, res_w, rms_w, _EPS, out_norm_weight=out_norm_w
            )
            ref_n = _manual_rmsnorm(ref, out_norm_w, _EPS)
            torch.testing.assert_close(
                got_n, ref_n, atol=2e-2, rtol=2e-2, msg=f"N={K} out-norm"
            )


def _manual_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    xf = x.float()
    rs = (xf.square().mean(-1, keepdim=True) + eps).rsqrt()
    return (xf * rs * weight.float()).to(x.dtype)


class AttnResOutNormTests(unittest.TestCase):
    def test_output_eps_is_ignored_without_output_norm(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(11, seed=4)
        kwargs = {
            "layer_residual": prefix_sum,
            "block_residual": block_residual,
            "res_weight": proj.weight.reshape(-1).to(torch.bfloat16),
            "rms_weight": norm.weight.to(torch.bfloat16),
            "eps": _EPS,
        }
        expected = attn_res_fwd(**kwargs)
        actual = attn_res_fwd(**kwargs, out_norm_eps=2 * _EPS)
        torch.testing.assert_close(actual, expected)

    def test_public_fallback_supports_distinct_output_eps(self):
        prefix_sum, block_residual, proj, norm = _make_inputs(11, seed=5)
        out_norm = RMSNorm(_HIDDEN, eps=_EPS)
        out_norm.weight.data.uniform_(0.5, 1.5)
        output_eps = 2 * _EPS
        fused = attn_res_fwd(
            layer_residual=prefix_sum,
            block_residual=block_residual,
            res_weight=proj.weight.reshape(-1).to(torch.bfloat16),
            rms_weight=norm.weight.to(torch.bfloat16),
            eps=_EPS,
            out_norm_weight=out_norm.weight.to(torch.bfloat16),
            out_norm_eps=output_eps,
        )
        mixed = attn_res_fwd(
            layer_residual=prefix_sum,
            block_residual=block_residual,
            res_weight=proj.weight.reshape(-1).to(torch.bfloat16),
            rms_weight=norm.weight.to(torch.bfloat16),
            eps=_EPS,
        )
        ref = _manual_rmsnorm(mixed, out_norm.weight, output_eps)
        torch.testing.assert_close(fused, ref, atol=2e-2, rtol=2e-2)

    def test_model_helper_out_norm_wiring(self):
        if not torch.cuda.is_available():
            self.skipTest("RMSNorm.forward requires CUDA")
        prefix_sum, block_residual, proj, norm = _make_inputs(6, seed=6)
        prefix_sum = prefix_sum.cuda()
        block_residual = block_residual.cuda()
        proj = proj.to(torch.bfloat16).cuda()
        norm = norm.cuda()
        out_norm = RMSNorm(_HIDDEN, eps=_EPS).to(torch.bfloat16).cuda()
        out_norm.weight.data.uniform_(0.5, 1.5)
        got = kimi_k3._apply_attn_res(
            prefix_sum, block_residual, proj, norm, 2, out_norm=out_norm
        )
        mixed = kimi_k3._apply_attn_res(prefix_sum, block_residual, proj, norm, 2)
        torch.testing.assert_close(
            got, _manual_rmsnorm(mixed, out_norm.weight, _EPS), atol=2e-2, rtol=2e-2
        )
        # Zero valid blocks: the helper must still apply the out-norm.
        got0 = kimi_k3._apply_attn_res(
            prefix_sum, block_residual, proj, norm, 0, out_norm=out_norm
        )
        torch.testing.assert_close(
            got0,
            _manual_rmsnorm(prefix_sum, out_norm.weight, _EPS),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_cuda_kernel_out_norm_matches_torch(self):
        try:
            from tokenspeed_kernel.ops.attn_res.cuda import _HAS_CUDA_KERNEL
        except ImportError:
            _HAS_CUDA_KERNEL = False
        if not (_HAS_CUDA_KERNEL and torch.cuda.is_available()):
            self.skipTest("Blackwell attn_res kernel not available")
        from tokenspeed_kernel.ops.attn_res import attn_res_fwd

        torch.manual_seed(1)
        T, H, K = 64, 7168, 8
        dev = "cuda"
        prefix = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
        blocks = torch.randn(K, T, H, dtype=torch.bfloat16, device=dev)
        res_w = torch.randn(H, dtype=torch.bfloat16, device=dev)
        rms_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
        out_w = torch.rand(H, dtype=torch.bfloat16, device=dev) + 0.5
        got = attn_res_fwd(prefix, blocks, res_w, rms_w, _EPS, out_norm_weight=out_w)
        ref = torch_attn_res_fwd(
            layer_residual=prefix,
            block_residual=blocks,
            res_weight=res_w,
            rms_weight=rms_w,
            eps=_EPS,
            out_norm_weight=out_w,
        )
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


class RouterGateTests(unittest.TestCase):
    def test_router_dispatch_matches_fp32_gemm(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        gate = kimi_k3.KimiLinearMoEGate(hidden_size=7168, num_experts=896).cuda()
        torch.manual_seed(0)
        gate.weight.data.copy_(torch.randn(896, 7168, dtype=torch.bfloat16).cuda())
        x = torch.randn(3, 7168, dtype=torch.bfloat16, device="cuda")
        got = gate(x)
        ref = torch.nn.functional.linear(x.float(), gate.weight)
        self.assertEqual(got.dtype, torch.float32)
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
        # Routing consumes top-k ids: they must be identical.
        self.assertTrue(
            torch.equal(
                got.topk(16, dim=-1).indices.sort(-1).values,
                ref.topk(16, dim=-1).indices.sort(-1).values,
            )
        )

    def test_router_gemm_runs_in_fp32(self):
        gate = kimi_k3.KimiLinearMoEGate(hidden_size=_HIDDEN, num_experts=8)
        torch.manual_seed(0)
        # fp32 at rest: the checkpoint's bf16 weight is cast once at load time
        # (default_weight_loader copy_), mirrored here.
        loaded = torch.randn(8, _HIDDEN, dtype=torch.bfloat16)
        gate.weight.data.copy_(loaded)
        self.assertEqual(gate.weight.dtype, torch.float32)
        x = torch.randn(6, _HIDDEN, dtype=torch.bfloat16)
        logits = gate(x)
        # fp32 GEMM: logits must match the fully up-cast bf16 reference.
        self.assertEqual(logits.dtype, torch.float32)
        ref = torch.nn.functional.linear(x.float(), loaded.float())
        torch.testing.assert_close(logits, ref)

    def test_bf16_router_weight_uses_fp32_fallback(self):
        gate = kimi_k3.KimiLinearMoEGate(
            hidden_size=_HIDDEN,
            num_experts=8,
        ).to(dtype=torch.bfloat16)
        torch.manual_seed(1)
        gate.weight.data.copy_(torch.randn_like(gate.weight))
        x = torch.randn(2, _HIDDEN, dtype=torch.bfloat16)

        logits = gate(x)

        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(
            logits,
            torch.nn.functional.linear(x.float(), gate.weight.float()),
        )


class KimiKDAMergedProjTests(unittest.TestCase):
    def test_loader_layout_and_forward_parity(self):
        torch.manual_seed(0)
        hidden, head_dim, num_heads, tp = 16, 4, 8, 2
        proj = num_heads * head_dim
        ws = {
            n: torch.randn(proj, hidden, dtype=torch.bfloat16)
            for n in ("q", "k", "v", "g")
        }
        ws["f_a"] = torch.randn(head_dim, hidden, dtype=torch.bfloat16)
        ws["b"] = torch.randn(num_heads, hidden, dtype=torch.bfloat16)
        for rank in range(tp):
            m = kimi_k3.KimiKDAMergedProj(
                hidden_size=hidden,
                proj=proj,
                num_heads=num_heads,
                head_dim=head_dim,
                tp_rank=rank,
                tp_size=tp,
            )
            for sid, w in ws.items():
                m.weight.weight_loader(m.weight, w, sid)
            x = torch.randn(3, hidden, dtype=torch.bfloat16)
            mixed_qkv, gate, f_a_out, beta = m(x)
            pl = proj // tp
            hl = num_heads // tp

            def ref(w, rows, rk=rank):
                return x @ w[rk * rows : (rk + 1) * rows].t()

            torch.testing.assert_close(
                mixed_qkv,
                torch.cat(
                    [ref(ws["q"], pl), ref(ws["k"], pl), ref(ws["v"], pl)], dim=-1
                ),
            )
            self.assertFalse(mixed_qkv.is_contiguous())
            self.assertEqual(mixed_qkv.stride(), (m.weight.shape[0], 1))
            self.assertEqual(
                mixed_qkv.untyped_storage().data_ptr(),
                gate.untyped_storage().data_ptr(),
            )
            torch.testing.assert_close(gate, ref(ws["g"], pl))
            # f_a is replicated: full output on every rank.
            torch.testing.assert_close(f_a_out, x @ ws["f_a"].t())
            torch.testing.assert_close(beta, ref(ws["b"], hl))
            # Rows are padded to the tactic-friendly multiple.
            self.assertEqual(m.weight.shape[0] % m._ROW_ALIGN, 0)

    def test_decode_single_row_slice_is_zero_copy(self):
        m = kimi_k3.KimiKDAMergedProj(
            hidden_size=8, proj=8, num_heads=2, head_dim=4, tp_rank=0, tp_size=1
        )
        torch.nn.init.normal_(m.weight)
        mixed, gate, _, _ = m(torch.randn(1, 8, dtype=torch.bfloat16))
        # [1, 3p] slice of a [1, total] row is already contiguous: no copy.
        self.assertTrue(mixed.is_contiguous())
        self.assertEqual(
            mixed.untyped_storage().data_ptr(),
            gate.untyped_storage().data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
