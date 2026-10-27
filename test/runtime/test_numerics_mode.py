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

"""The --numerics envelope folds into the individual determinism switches."""

import sys

import pytest

from tokenspeed.runtime.utils.server_args import ServerArgs


class TestNumericsMode:
    def test_auto_keeps_performance_defaults(self):
        args = ServerArgs(model="x")
        assert (args.numerics) == ("auto")
        assert not (args.force_deterministic_rsag)
        assert not (args.disable_autotune)
        assert not (args.disable_tf32)

    def test_rl_bitwise_tightens_every_switch(self):
        args = ServerArgs(model="x", numerics="rl-bitwise")
        # The envelope routes reductions itself (batch_invariant_collectives);
        # the NCCL-only knob stays the user's.
        assert not (args.force_deterministic_rsag)
        assert ServerArgs(
            model="x", numerics="rl-bitwise", force_deterministic_rsag=True
        ).force_deterministic_rsag
        assert args.disable_autotune
        assert args.disable_tf32
        assert args.disable_pdl
        assert not (args.enable_allreduce_fusion)
        assert (args.comm_fusion_max_num_tokens) == (-1)
        assert args.moe_backend == "auto"
        assert args.batch_invariant_collectives

    def test_rl_bitwise_leaves_provider_validation_to_expert_planning(self):
        assert (
            ServerArgs(
                model="x", numerics="rl-bitwise", moe_backend="triton"
            ).moe_backend
            == "triton"
        )
        assert (
            ServerArgs(
                model="x", numerics="rl-bitwise", draft_moe_backend="triton"
            ).draft_moe_backend
            == "triton"
        )
        # Auto asks the registry for a provider with the required capabilities.
        args = ServerArgs(model="x", numerics="rl-bitwise", draft_moe_backend="auto")
        assert args.draft_moe_backend == "auto"
        with pytest.raises((ValueError), match=("--sampling-backend triton")):
            ServerArgs(model="x", numerics="rl-bitwise", sampling_backend="triton")
        args = ServerArgs(
            model="x", numerics="rl-bitwise", sampling_backend="flashinfer_full"
        )
        assert (args.sampling_backend) == ("flashinfer_full")

    def test_auto_keeps_the_moe_backend_auto(self):
        args = ServerArgs(model="x")
        assert (args.moe_backend) == ("auto")

    def test_rl_bitwise_overrides_the_fusion_auto_enable(self):
        # resolve_communication auto-enables allreduce fusion on capable
        # topologies; the envelope must win regardless.
        args = ServerArgs(model="x", numerics="rl-bitwise", world_size=1)
        assert not (args.enable_allreduce_fusion)

    def test_batch_invariant_collectives_vetoes_the_fused_all_reduce_norm(self):
        args = ServerArgs(model="x", enable_allreduce_fusion=True)
        assert args.enable_allreduce_fusion and args.comm_fusion_max_num_tokens > 0
        # On its own, under auto, it vetoes both the flag and the token-count gate.
        args = ServerArgs(
            model="x", batch_invariant_collectives=True, enable_allreduce_fusion=True
        )
        assert not (args.enable_allreduce_fusion)
        assert (args.comm_fusion_max_num_tokens) == (-1)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises((ValueError), match=("rl-bitwise")):
            ServerArgs(model="x", numerics="bitwise")

    def test_rl_bitwise_folds_the_per_request_sampling_stream(self):
        assert (ServerArgs(model="x").sampling_stream) == ("batch")
        args = ServerArgs(model="x", numerics="rl-bitwise")
        assert (args.sampling_stream) == ("per-request")
        args = ServerArgs(model="x", sampling_stream="per-request")
        assert (args.sampling_stream) == ("per-request")
        with pytest.raises((ValueError), match=("--sampling-stream")):
            ServerArgs(model="x", sampling_stream="philox")

    def test_rl_bitwise_computes_the_yarn_ramp_on_cpu(self):
        assert (ServerArgs(model="x").yarn_ramp_mask_device) == ("cuda")
        assert (ServerArgs(model="x", numerics="rl-bitwise").yarn_ramp_mask_device) == (
            "cpu"
        )
        with pytest.raises((ValueError), match=("--yarn-ramp-mask-device")):
            ServerArgs(model="x", yarn_ramp_mask_device="npu")

    def test_rl_bitwise_applies_the_mla_lora_scale_at_runtime(self):
        assert (ServerArgs(model="x").mla_lora_scale) == ("folded")
        assert (ServerArgs(model="x", numerics="rl-bitwise").mla_lora_scale) == (
            "runtime"
        )
        with pytest.raises((ValueError), match=("--mla-lora-scale")):
            ServerArgs(model="x", mla_lora_scale="both")

    def test_rl_bitwise_unfuses_the_layer_boundary_norm(self):
        assert (ServerArgs(model="x").layer_boundary_norm) == ("fused")
        assert (ServerArgs(model="x", numerics="rl-bitwise").layer_boundary_norm) == (
            "unfused"
        )
        # On its own, under auto, it vetoes both the flag and the token-count gate.
        args = ServerArgs(
            model="x", layer_boundary_norm="unfused", enable_allreduce_fusion=True
        )
        assert not (args.enable_allreduce_fusion)
        assert (args.comm_fusion_max_num_tokens) == (-1)
        with pytest.raises((ValueError), match=("--layer-boundary-norm")):
            ServerArgs(model="x", layer_boundary_norm="half")

    def test_rl_bitwise_routes_with_the_torch_router_topk(self):
        assert (ServerArgs(model="x").router_topk) == ("fused")
        assert (ServerArgs(model="x", numerics="rl-bitwise").router_topk) == ("torch")
        with pytest.raises((ValueError), match=("--router-topk")):
            ServerArgs(model="x", router_topk="cuda")

    def test_rl_bitwise_reports_megatron_order_logprobs(self):
        assert (ServerArgs(model="x").logprob_order) == ("torch")
        assert (ServerArgs(model="x", numerics="rl-bitwise").logprob_order) == (
            "megatron"
        )
        with pytest.raises((ValueError), match=("--logprob-order")):
            ServerArgs(model="x", logprob_order="apex")

    def test_rl_bitwise_reduces_dsa_slots_sorted(self):
        assert (ServerArgs(model="x").dsa_slot_order) == ("selection")
        assert (ServerArgs(model="x", numerics="rl-bitwise").dsa_slot_order) == (
            "sorted"
        )
        with pytest.raises((ValueError), match=("--dsa-slot-order")):
            ServerArgs(model="x", dsa_slot_order="shuffled")

    def test_rl_bitwise_combines_moe_slots_in_the_leaf(self):
        assert (ServerArgs(model="x").moe_combine_order) == ("rank")
        assert (ServerArgs(model="x", numerics="rl-bitwise").moe_combine_order) == (
            "slot"
        )
        # The leaf returns complete rows, so a fused all-reduce+norm at the
        # next layer boundary would sum them again: vetoed under auto too.
        args = ServerArgs(
            model="x", moe_combine_order="slot", enable_allreduce_fusion=True
        )
        assert not (args.enable_allreduce_fusion)
        with pytest.raises((ValueError), match=("--moe-combine-order")):
            ServerArgs(model="x", moe_combine_order="tree")

    def test_slot_combine_is_validated_against_the_launch_at_startup(self):
        # The slot fold runs over the EP group inside the leaf: a K-split down
        # projection would need a second fold, and DeepEP owns the exchange.
        ServerArgs(model="x", moe_combine_order="slot", world_size=2, ep_size=2)
        # MoE TP defaults to the stage world over EP, so world_size=2 alone
        # is MoE TP 2.
        with pytest.raises((ValueError), match=("needs MoE TP 1")):
            ServerArgs(model="x", moe_combine_order="slot", world_size=2)
        with pytest.raises((ValueError), match=("--all2all-backend deepep")):
            ServerArgs(
                model="x",
                moe_combine_order="slot",
                world_size=2,
                ep_size=2,
                all2all_backend="deepep",
            )
        # The envelope inherits both refusals: rl-bitwise needs MoE TP 1.
        with pytest.raises((ValueError), match=("needs MoE TP 1")):
            ServerArgs(model="x", numerics="rl-bitwise", world_size=2)
        ServerArgs(model="x", numerics="rl-bitwise", world_size=2, ep_size=2)

    def test_rl_bitwise_is_the_one_bitwise_envelope(self):
        from tokenspeed.runtime.configs.numerics import NUMERICS_ENVELOPES

        assert (NUMERICS_ENVELOPES) == (("auto", "rl-bitwise"))
        with pytest.raises((ValueError), match=("rl-bitwise")):
            ServerArgs(model="x", numerics="trainer-aligned")

    def test_bitwise_envelopes_cover_every_pinning_envelope(self):
        from tokenspeed.runtime.configs.numerics import (
            BITWISE_ENVELOPES,
            NUMERICS_ENVELOPES,
        )

        assert (BITWISE_ENVELOPES) == (set(NUMERICS_ENVELOPES) - {"auto"})

    def test_ordered_fold_matches_the_sum_and_only_depends_on_rank_order(self):
        import torch

        from tokenspeed.runtime.distributed.comm_backend.auto import ordered_fold_sum

        torch.manual_seed(7)
        parts = torch.randn(8, 5, 64, dtype=torch.float32)
        out = torch.empty(5, 64, dtype=torch.bfloat16)
        ordered_fold_sum(parts, out)
        expected = parts[0].clone()
        for rank in range(1, 8):
            expected = expected + parts[rank]
        assert torch.equal(out, expected.to(torch.bfloat16))
        # The same row folds to the same bits inside a larger payload: the
        # batch-invariance claim a ring all-reduce cannot make.
        wide = torch.cat((torch.randn(8, 300, 64), parts.narrow(1, 2, 1)), dim=1)
        wide_out = torch.empty(301, 64, dtype=torch.bfloat16)
        ordered_fold_sum(wide, wide_out)
        assert torch.equal(wide_out[300], out[2])

    def test_token_reduce_scatter_folds_each_slice_in_rank_order(self):
        from unittest import mock

        import torch

        from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend

        group = (0, 1, 2)
        counts = [2, 1, 3]
        width = max(counts)
        torch.manual_seed(3)
        inputs = [torch.randn(sum(counts), 4) for _ in group]
        offsets = [0, 2, 3]
        padded = []
        for full in inputs:
            rows = torch.zeros(len(group) * width, 4)
            for i, (offset, count) in enumerate(zip(offsets, counts)):
                rows[i * width : i * width + count] = full[offset : offset + count]
            padded.append(rows)

        for rank in group:

            def exchange(out, inp, grp, rank=rank):
                out.copy_(
                    torch.cat([p[rank * width : (rank + 1) * width] for p in padded])
                )

            backend = AutoBackend.__new__(AutoBackend)
            backend._nccl = mock.Mock(all_to_all_single=exchange)
            with mock.patch("torch.distributed.get_rank", return_value=rank):
                got = backend._ordered_fold_token_reduce_scatter(
                    inputs[rank], group, counts
                )
            expected = inputs[0][offsets[rank] : offsets[rank] + counts[rank]].clone()
            for other in group[1:]:
                expected = (
                    expected
                    + inputs[other][offsets[rank] : offsets[rank] + counts[rank]]
                )
            assert torch.equal(got, expected)


class TestModelVerificationGate:
    def _profile(self, envelopes):
        from tokenspeed.runtime.configs.model_config import configure_mla_attention
        from tokenspeed.runtime.configs.model_profile import ModelProfile

        return ModelProfile(
            configure_attention=configure_mla_attention,
            cache_family="fixture_family",
            linear_attention=None,
            default_attention_backend=None,
            default_prefix_granularity=None,
            request_token_history=False,
            is_multimodal=False,
            tokenizer_kwargs={},
            attention_instances_per_layer=1,
            numerics_envelopes=frozenset(envelopes),
        )

    def test_auto_serves_every_model(self):
        from tokenspeed.runtime.configs.numerics import require_verified_numerics

        require_verified_numerics(
            "auto",
            model_profile=None,
            architecture="X",
            quantization="fp8",
            vocab_size=1000,
        )

    def test_rl_bitwise_requires_a_verified_unquantized_model(self):
        from tokenspeed.runtime.configs.numerics import (
            MEGATRON_VOCAB_BLOCK,
            require_verified_numerics,
        )

        with pytest.raises((ValueError), match=("has not been verified")):
            require_verified_numerics(
                "rl-bitwise",
                model_profile=None,
                architecture="X",
                quantization=None,
                vocab_size=MEGATRON_VOCAB_BLOCK,
            )
        with pytest.raises((ValueError), match=("has not been verified")):
            require_verified_numerics(
                "rl-bitwise",
                model_profile=self._profile({"auto"}),
                architecture="X",
                quantization=None,
                vocab_size=MEGATRON_VOCAB_BLOCK,
            )
        verified = self._profile({"auto", "rl-bitwise"})
        with pytest.raises((ValueError), match=("fp8-quantized")):
            require_verified_numerics(
                "rl-bitwise",
                model_profile=verified,
                architecture="X",
                quantization="fp8",
                vocab_size=MEGATRON_VOCAB_BLOCK,
            )
        require_verified_numerics(
            "rl-bitwise",
            model_profile=verified,
            architecture="X",
            quantization=None,
            vocab_size=4 * MEGATRON_VOCAB_BLOCK,
        )

    def test_rl_bitwise_needs_a_whole_number_of_megatron_vocab_blocks(self):
        # The envelope folds --logprob-order megatron, whose sum(exp) runs
        # over fixed vocab blocks; a model whose vocabulary cannot be cut into
        # them is refused at startup with that reason.
        from tokenspeed.runtime.configs.numerics import (
            MEGATRON_VOCAB_BLOCK,
            require_verified_numerics,
        )

        with pytest.raises((ValueError), match=("vocab_size 32000, not a multiple")):
            require_verified_numerics(
                "rl-bitwise",
                model_profile=self._profile({"auto", "rl-bitwise"}),
                architecture="X",
                quantization=None,
                vocab_size=32000,
            )
        assert (MEGATRON_VOCAB_BLOCK) == (32768)

    def test_profile_envelopes_are_validated(self):
        with pytest.raises((ValueError), match=("must include 'auto'")):
            self._profile({"rl-bitwise"})
        with pytest.raises((ValueError), match=("must include 'auto'")):
            self._profile({"auto", "bitwise"})
        with pytest.raises((ValueError), match=("must include 'auto'")):
            self._profile({"auto", "trainer-aligned"})
        assert isinstance((self._profile(["auto"]).numerics_envelopes), (frozenset))


class TestCanonicalGreedyTies:
    def test_sampled_greedy_rows_take_the_lowest_tied_id(self):
        import torch

        from tokenspeed.runtime.sampling.backends.flashinfer import (
            canonical_greedy_tokens,
        )

        logits = torch.zeros(3, 8)
        logits[:, 2] = logits[:, 5] = 4.0
        sampled = torch.tensor([5, 5, 7], dtype=torch.int32)
        top_ks = torch.tensor([1, 20, 1])
        out = canonical_greedy_tokens(logits, top_ks, sampled)
        # Greedy rows resolve the tie to id 2; the sampled row is untouched.
        assert (out.tolist()) == ([2, 5, 2])
        assert (out.dtype) == (torch.int32)

    def test_verified_greedy_rows_follow_the_exact_match_chain(self):
        import torch

        from tokenspeed.runtime.sampling.backends.flashinfer import (
            canonical_greedy_verify,
        )

        bs, n, vocab = 2, 3, 8
        logits = torch.zeros(bs * n, vocab)
        # Every position ties ids 2 and 5; the canonical argmax is 2.
        logits[:, 2] = logits[:, 5] = 1.0
        candidates = torch.tensor([[9, 2, 2], [9, 5, 2]], dtype=torch.int32)
        # Pretend the stochastic kernel accepted differently on both rows.
        predict = torch.full((bs * n,), 5, dtype=torch.int32)
        accept_index = torch.tensor([[0, -1, -1], [3, 4, 5]], dtype=torch.int32)
        accept_length = torch.tensor([0, 2], dtype=torch.int32)
        top_ks = torch.tensor([1, 1, 1, 50, 50, 50])
        canonical_greedy_verify(
            logits=logits,
            top_ks=top_ks,
            candidates=candidates,
            predict=predict,
            accept_index=accept_index,
            accept_length=accept_length,
        )
        # Row 0 (greedy): drafts 2, 2 both match argmax 2 -> two accepted.
        assert (accept_length.tolist()) == ([2, 2])
        assert (accept_index[0].tolist()) == ([0, 1, 2])
        assert (predict[:3].tolist()) == ([2, 2, 2])
        # Row 1 (sampling) keeps the stochastic kernel's outputs.
        assert (accept_index[1].tolist()) == ([3, 4, 5])
        assert (predict[3:].tolist()) == ([5, 5, 5])

    def test_verify_overlay_leaves_every_predict_slot_a_token_id(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("the CUDA chain kernel writes a partial predict row")
        from tokenspeed.runtime.sampling.backends.flashinfer import (
            canonical_greedy_verify,
        )

        bs, n, vocab = 4, 4, 64
        logits = torch.randn(bs * n, vocab, device="cuda")
        candidates = torch.randint(0, vocab, (bs, n), dtype=torch.int32, device="cuda")
        # Sentinels where the stochastic kernel left slots it never reads.
        predict = torch.full((bs * n,), 10**6, dtype=torch.int32, device="cuda")
        accept_index = torch.full((bs, n), -1, dtype=torch.int32, device="cuda")
        accept_length = torch.zeros(bs, dtype=torch.int32, device="cuda")
        canonical_greedy_verify(
            logits=logits,
            top_ks=torch.ones(bs * n, dtype=torch.int32, device="cuda"),
            candidates=candidates,
            predict=predict,
            accept_index=accept_index,
            accept_length=accept_length,
        )
        # Every slot is gathered for logprobs, accepted or not.
        assert bool((predict < vocab).all())
        assert torch.equal(predict.view(bs, n), logits.argmax(-1).view(bs, n).int())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
