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

"""Beam TP4 runtime integration for mixed-batch triton attention over cache groups."""

import math
import os
import sys
from datetime import datetime, timezone

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=50, suite="runtime-1gpu")

# The breakable-graph refresh test loads the kernel compile monitor from
# tokenspeed-kernel/test, relative to the repo's test/ directory.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Q_HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128


@pytest.fixture(scope="module", autouse=True)
def verified_amd_device():
    platform = current_platform()
    if not platform.is_amd or platform.arch_version != ArchVersion(9, 5):
        pytest.skip("Beam portable attention qualification requires gfx950")
    properties = torch.cuda.get_device_properties(0)
    expected_bus = os.environ.get("TOKENSPEED_TEST_EXPECTED_PCI_BUS")
    if expected_bus is not None:
        assert torch.cuda.device_count() == 1
        assert properties.pci_bus_id == int(expected_bus, 0)
        assert properties.pci_domain_id == 0 and properties.pci_device_id == 0
        expected_uuid = os.environ["ROCR_VISIBLE_DEVICES"].removeprefix("GPU-")
        assert bytes(properties.uuid.bytes).decode("ascii") == expected_uuid
    print(
        f"Beam attention GPU test: pid={os.getpid()} "
        f"start={datetime.now(timezone.utc).isoformat()} "
        f"pci={properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0 uuid={properties.uuid}"
    )
    torch.manual_seed(42)
    yield
    torch.cuda.synchronize()


def _reference(q, k, v, *, prefix, window_left, round_unnormalized_probabilities):
    # Float64 on the host is independent of the GPU kernel's reduction order.
    q, k, v = (tensor.cpu().double() for tensor in (q, k, v))
    k = k.repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
    v = v.repeat_interleave(Q_HEADS // KV_HEADS, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(HEAD_DIM)
    positions = prefix + torch.arange(q.shape[0])
    distance = positions[:, None] - torch.arange(k.shape[0])[None, :]
    mask = distance >= 0
    if window_left >= 0:
        mask &= distance <= window_left
    scores = scores.masked_fill(~mask[None], -torch.inf)
    if round_unnormalized_probabilities:
        # One key tile: the value product reads BF16 exp(score - max), while
        # the normalizer sums the original probabilities in higher precision.
        weights = (scores - scores.amax(dim=-1, keepdim=True)).exp()
        probabilities = weights.bfloat16().double() / weights.sum(dim=-1, keepdim=True)
    else:
        probabilities = scores.softmax(dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v).float()


def _assert_output(actual, expected):
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=1e-2, atol=2e-3)


def _assert_prefill_output(output, q, k, v, *, window_left):
    length = q.shape[0]
    mathematical = _reference(
        q,
        k,
        v,
        prefix=0,
        window_left=window_left,
        round_unnormalized_probabilities=False,
    )
    errors = (output.cpu().float() - mathematical).abs()
    print(
        f"prefill length={length} window_left={window_left}: "
        f"FP64 math max_abs={errors.max().item():.8f} "
        f"rms={errors.square().mean().sqrt().item():.8f}"
    )
    # Every causal query in the first 64 rows attends at most one AMD key
    # tile. Check its BF16 probability boundary directly; later queries keep
    # the independent mathematical oracle and the original tight tolerance.
    first = min(length, 64)
    expected_first = _reference(
        q[:first],
        k[:first],
        v[:first],
        prefix=0,
        window_left=window_left,
        round_unnormalized_probabilities=True,
    )
    _assert_output(output[:first], expected_first)
    if first < length:
        _assert_output(output[first:], mathematical[first:])


def _runtime_mixed_round(width, *, larger, all_fresh, first_prefix):
    if larger:
        requests, queries, prefixes = [9, 7, 2], [65], [512]
        histories = [63, 513]
    else:
        requests, queries = [7, 2, 9], [3, 65 if all_fresh else 17]
        prefixes = [first_prefix, 0 if all_fresh else 63]
        histories = [513]
    lengths = [q + p for q, p in zip(queries, prefixes)]
    lengths += [history + width for history in histories]
    return requests, queries, prefixes, lengths


class _MixedCache:
    """Storage only: routing and registered GPU attention stay real."""

    def __init__(self, buffers):
        self.buffers = buffers

    def get_key_buffer(self, layer_id):
        return self.buffers[layer_id][0]

    def get_value_buffer(self, layer_id):
        return self.buffers[layer_id][1]


class _RuntimeMixedAttention:
    def __init__(self, width, *, device):
        from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
            CacheGroupGeometry,
        )
        from tokenspeed.runtime.layers.attention.backends.paged.mha import (
            MHAAttnBackend,
        )
        from tokenspeed.runtime.layers.attention.backends.paged.router import (
            CacheGroupRouter,
        )
        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
        from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
        from tokenspeed.runtime.layers.paged_attention import PagedAttention

        self.width, self.device, self.bucket = width, device, 128
        self.groups = {"full": 16, "sliding": 8}
        self.leaves, self.layers, self.buffers = {}, {}, {}
        self.logical, self.tables = {}, {}
        self.round = None
        self.q_cpu = None
        self.q = torch.empty(
            self.bucket, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        for index, (gid, page_size) in enumerate(self.groups.items()):
            spec = MHAConfig(
                backend_name="triton",
                num_attention_heads=Q_HEADS,
                num_kv_heads=KV_HEADS,
                head_dim=HEAD_DIM,
                attn_tp_size=1,
            )
            config = AttnConfig(
                device=device,
                dtype=torch.bfloat16,
                kv_cache_dtype=torch.bfloat16,
                kv_cache_quant_method="none",
                prefix_granularity=16,
                context_len=1024,
                max_bs=3,
                speculative_num_draft_tokens=width,
                is_draft=False,
                draft_block_decode=False,
                components=(spec,),
            )
            self.leaves[gid] = MHAAttnBackend(config, spec, kernel_page_size=page_size)
            layer = PagedAttention(
                Q_HEADS,
                HEAD_DIM,
                HEAD_DIM**-0.5,
                KV_HEADS,
                index,
                sliding_window_size=-1 if gid == "full" else 511,
                rotary_emb=None,
                qk_norm=None,
            )
            layer.bind_cache_group(gid)
            self.layers[gid] = layer
            self.buffers[index] = tuple(
                torch.zeros(
                    193 * 16, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
                )
                for _ in range(2)
            )
        self.cache = _MixedCache(self.buffers)
        assert self.q.shape == (128, 12, 128)
        assert all(pair[0].shape[1:] == (2, 128) for pair in self.buffers.values())
        self.router = CacheGroupRouter(
            None,
            is_draft=False,
            spec_num_tokens=width,
            device=device,
            consumed_group_ids=None,
        )
        self.router.bind(
            CacheGroupGeometry(
                granularities={gid: 16 for gid in self.groups},
                attn_kinds={"full": "full_history", "sliding": "sliding"},
                full_history_group_id="full",
                row_geometry={gid: (16, 1) for gid in self.groups},
                windows={"full": None, "sliding": 512},
            ),
            self.leaves,
        )
        self.router.init_cuda_graph_state(3, max_extend_tokens=self.bucket)

    def prepare(self, description, *, seed):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        self.round = description
        requests, queries, prefixes, lengths = description
        generator = torch.Generator(device="cpu").manual_seed(seed)
        all_queries = queries + [self.width] * (3 - len(queries))
        self.q_cpu = torch.randn(
            self.bucket,
            Q_HEADS,
            HEAD_DIM,
            generator=generator,
        ).bfloat16()
        self.q_cpu[sum(all_queries) :] = float("nan")
        self.q.copy_(self.q_cpu)
        for index, gid in enumerate(self.groups):
            # Independent group/round permutations in allocation-block units.
            table = (
                torch.arange(1, 193, dtype=torch.int32)
                .flip(0)
                .roll((seed + index * 37) % 192)
                .reshape(3, 64)
            )
            self.tables[gid] = table
            slots = (table.long()[:, :, None] * 16 + torch.arange(16)).reshape(-1)
            logical = tuple(
                torch.randn(3, 1024, KV_HEADS, HEAD_DIM, generator=generator).bfloat16()
                for _ in range(2)
            )
            self.logical[gid] = logical
            for storage, values in zip(self.buffers[index], logical):
                storage.index_copy_(
                    0, slots.to(self.device), values.flatten(0, 1).to(self.device)
                )
        query_cpu = torch.tensor(queries, dtype=torch.int32)
        prefix_cpu = torch.tensor(prefixes, dtype=torch.int32)
        self.router.init_forward_metadata(
            3,
            len(queries),
            torch.tensor(requests, dtype=torch.int32, device=self.device),
            torch.tensor(lengths, dtype=torch.int32, device=self.device),
            ForwardMode.MIXED,
            block_tables={
                gid: table.to(self.device) for gid, table in self.tables.items()
            },
            extend_seq_lens=query_cpu.to(self.device),
            extend_seq_lens_cpu=query_cpu,
            extend_prefix_lens=prefix_cpu.to(self.device),
            extend_prefix_lens_cpu=prefix_cpu,
            extend_replay_lens_cpu=torch.zeros_like(prefix_cpu),
            extend_prompt_lens_cpu=query_cpu + prefix_cpu,
            extend_with_prefix=any(prefixes),
            query_shard=None,
            block_tables_cpu={},
        )
        return sum(all_queries)

    def forward(self):
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        outputs = {}
        # The clone consumes each shape-keyed handoff before the next break
        # overwrites it, matching the lifetime rule of the model's o-projection.
        q = self.q + 0
        for gid, layer in self.layers.items():
            output = self.router.forward(
                q,
                q[:, :KV_HEADS],
                q[:, :KV_HEADS],
                layer,
                self.cache,
                ForwardMode.MIXED,
                3,
                save_kv_cache=False,
            )
            outputs[gid] = output.clone()
        return outputs

    def assert_outputs(self, outputs, *, selected_groups):
        _, queries, prefixes, lengths = self.round
        all_queries = queries + [self.width] * (3 - len(queries))
        real_rows = sum(all_queries)
        for gid in selected_groups:
            window = self.layers[gid].sliding_window_size
            key, value = self.logical[gid]
            cursor = 0
            for request, count in enumerate(all_queries):
                q = self.q_cpu[cursor : cursor + count]
                actual = outputs[gid][cursor : cursor + count].view(
                    count, Q_HEADS, HEAD_DIM
                )
                prefix = lengths[request] - count
                if prefix == 0:
                    _assert_prefill_output(
                        actual,
                        q,
                        key[request, :count],
                        value[request, :count],
                        window_left=window,
                    )
                else:
                    _assert_output(
                        actual,
                        _reference(
                            q,
                            key[request, : lengths[request]],
                            value[request, : lengths[request]],
                            prefix=prefix,
                            window_left=window,
                            round_unnormalized_probabilities=False,
                        ),
                    )
                cursor += count
            assert outputs[gid].shape[0] >= real_rows
            assert torch.isfinite(outputs[gid]).all()
            if outputs[gid].shape[0] > real_rows:
                assert torch.count_nonzero(outputs[gid][real_rows:]) == 0


@pytest.mark.parametrize("width", [1, 16])
@pytest.mark.parametrize("window_left", [-1, 511], ids=["full", "sliding512"])
@pytest.mark.parametrize("history", ["fresh", "resumed", "all_fresh"])
def test_beam_triton_runtime_mixed_matches_reference(width, window_left, history):
    harness = _RuntimeMixedAttention(width, device="cuda")
    description = _runtime_mixed_round(
        width,
        larger=False,
        all_fresh=history == "all_fresh",
        first_prefix=512 if history == "resumed" else 0,
    )
    real_rows = harness.prepare(description, seed=23)
    before = {
        index: tuple(t.clone() for t in pair) for index, pair in harness.buffers.items()
    }
    before_query = harness.q.clone()
    outputs = harness.forward()
    assert all(out.shape[0] == real_rows for out in outputs.values())
    harness.assert_outputs(
        outputs, selected_groups=["full" if window_left < 0 else "sliding"]
    )
    for index, pair in harness.buffers.items():
        assert all(torch.equal(t, original) for t, original in zip(pair, before[index]))
    torch.testing.assert_close(harness.q, before_query, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("width", [1, 16])
def test_beam_triton_runtime_mixed_cache_groups_refresh(width):
    harness = _RuntimeMixedAttention(width, device="cuda")
    for larger, seed in ((False, 31), (True, 43), (False, 59)):
        description = _runtime_mixed_round(
            width, larger=larger, all_fresh=False, first_prefix=0
        )
        harness.prepare(description, seed=seed)
        harness.assert_outputs(harness.forward(), selected_groups=harness.groups)


@pytest.mark.parametrize("width", [1, 16])
def test_beam_triton_runtime_mixed_breakable_graph_refresh(width):
    import importlib.util

    from tokenspeed_kernel.ops.attention.mha._triton.prefill import _fwd_kernel

    from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
    from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
        _unpack_group_kernel,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.write_locations import (
        _decode_locs_kernel,
        _extend_locs_kernel,
    )

    # This is the existing compile monitor, not a second validation layer.
    kernel_test_dir = os.path.join(_TEST_DIR, "..", "tokenspeed-kernel", "test")
    helper_spec = importlib.util.spec_from_file_location(
        "beam_mixed_kernel_test_utils", os.path.join(kernel_test_dir, "utils.py")
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)

    harness = _RuntimeMixedAttention(width, device="cuda")
    rounds = {
        larger: _runtime_mixed_round(
            width, larger=larger, all_fresh=False, first_prefix=0
        )
        for larger in (False, True)
    }
    for larger in (False, True):
        harness.prepare(rounds[larger], seed=71 + int(larger))
        for _ in range(2):
            harness.forward()
    # The larger real-token count sizes the stable handoff once. Metadata
    # remains eager; A/B/A refreshes the SAME capture and runtime objects.
    harness.prepare(rounds[True], seed=72)
    torch.cuda.synchronize()
    capture = BreakableCapture()
    with capture:
        captured = harness.forward()
    segments = capture.num_segments
    pointers = {gid: out.data_ptr() for gid, out in captured.items()}
    input_pointer = harness.q.data_ptr()
    assert segments == 5
    segment_identities = tuple(id(segment) for segment in capture.segments)
    graph_identities = tuple(
        id(capture.segments[index].__self__) for index in (0, 2, 4)
    )
    assert all(
        isinstance(capture.segments[index].__self__, torch.cuda.CUDAGraph)
        for index in (0, 2, 4)
    )
    handoff_identities = {
        key: (id(tensor), tensor.data_ptr()) for key, tensor in capture._handoff.items()
    }
    assert len(handoff_identities) == 1  # Both groups share one shape-keyed handoff.
    with helper.assert_no_triton_compile(
        _fwd_kernel, _unpack_group_kernel, _decode_locs_kernel, _extend_locs_kernel
    ):
        for larger, seed in ((False, 71), (True, 72), (False, 73)):
            real_rows = harness.prepare(rounds[larger], seed=seed)
            eager = harness.forward()
            capture.replay(valid_rows=real_rows)
            torch.cuda.synchronize()
            assert capture.num_segments == segments
            assert (
                tuple(id(segment) for segment in capture.segments) == segment_identities
            )
            assert (
                tuple(id(capture.segments[index].__self__) for index in (0, 2, 4))
                == graph_identities
            )
            assert {
                key: (id(tensor), tensor.data_ptr())
                for key, tensor in capture._handoff.items()
            } == handoff_identities
            assert harness.q.data_ptr() == input_pointer
            assert {gid: out.data_ptr() for gid, out in captured.items()} == pointers
            harness.assert_outputs(captured, selected_groups=harness.groups)
            for gid in harness.groups:
                torch.testing.assert_close(
                    captured[gid][:real_rows], eager[gid], rtol=0, atol=0
                )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
