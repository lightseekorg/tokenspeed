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

"""Small Engram correctness checks; never allocate checkpoint-sized tables.

Optional local reference parity: set DEEPSEEK_V41_REFERENCE_DIR to a snapshot
containing config.json, tokenizer.json and inference/{engram,model}.py.
Four-device collective check: torchrun --standalone --nproc_per_node=4 -m pytest
    test/runtime/test_deepseek_v41_engram.py -k distributed_four_way
Use CUDA_VISIBLE_DEVICES to select only free GPUs.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config, Mxfp8Config
from tokenspeed.runtime.models import deepseek_v41_engram as engram
from tokenspeed.runtime.models.deepseek_v41_engram import (
    DeepseekV41Engram,
    EngramHashState,
    EngramLayout,
    RowShardedEngramEmbedding,
    build_compressed_token_map,
    build_engram_previous_tokens,
    compute_hash_multipliers,
)


class _Tokenizer:
    texts = [
        "<bos>",
        "<eos>",
        "<pad>",
        " The",
        "the",
        "THE",
        "café",
        "CAFE",
        "\t",
        " ",
        "",
        "\ufffd",
        "\ufffd",
    ]

    def __init__(self):
        self.backend_tokenizer = self

    def __len__(self):
        return len(self.texts)

    def decode(self, ids, skip_special_tokens):
        assert not skip_special_tokens
        return self.texts[ids[0]]

    def id_to_token(self, token_id):
        return f"raw_byte_{token_id}"


def _config():
    return SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[72, 204],
        engram_max_ngram_size=4,
        engram_vocab_size=5,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=9,
        hc_mult=4,
        hidden_size=32,
        rms_norm_eps=1e-20,
    )


def _mapping(rank, tp_size, world_size):
    return Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        attn_dp_size=world_size // tp_size,
        dense_tp_size=world_size,
        dense_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=world_size,
        moe_dp_size=1,
        vision_tp_size=tp_size,
        vision_dp_size=1,
        linear_attn_tp_size=tp_size,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=world_size,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def _weights(rows, width):
    codes = ((torch.arange(rows * width).reshape(rows, width) % 17 - 8) / 4).to(
        torch.float8_e4m3fn
    )
    scales = (
        torch.arange(rows * (width // 32)).reshape(rows, width // 32) % 5 + 125
    ).to(torch.uint8)
    return codes, scales


def _dequant(codes, scales):
    return (
        (
            codes.float().unflatten(-1, (-1, 32))
            * scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        )
        .flatten(-2)
        .to(torch.bfloat16)
    )


def _reference_hashes(state, histories, requests, positions):
    token_map = state.token_map.tolist()
    multipliers = state.multipliers.tolist()
    result = []
    for request, position in zip(requests, positions, strict=True):
        tokens, blocked = [], False
        for shift in range(4):
            source = histories[request][position - shift] if position >= shift else -1
            blocked |= source == -1
            tokens.append(state.pad_id if blocked else token_map[source])
        layers = []
        for primes, mult in zip(state.layout.primes, multipliers, strict=True):
            rolling, offset, ids = tokens[0] * mult[0], 0, []
            for shift, heads in enumerate(primes, start=1):
                rolling ^= tokens[shift] * mult[shift]
                for prime in heads:
                    ids.append(rolling % prime + offset)
                    offset += prime
            layers.append(ids)
        result.append(layers)
    return torch.tensor(result, dtype=torch.int64)


def test_token_compression_and_layout():
    assert build_compressed_token_map(_Tokenizer()) == (
        [0, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8],
        9,
    )
    layout = EngramLayout.from_config(_config())
    assert layout.primes == (
        ((5, 7), (11, 13), (17, 19)),
        ((23, 29), (31, 37), (41, 43)),
    )
    config = _config()
    config.engram_vocab_size = 16_000_000
    config.engram_n_heads = 8
    config.engram_head_dim = 256
    config.engram_num_embeddings = [384006168, 384016682]
    real_layout = EngramLayout.from_config(config)
    assert real_layout.num_embeddings == (384006168, 384016682)
    multipliers = compute_hash_multipliers((1, 14), 4, 99092)
    assert torch.all(multipliers % 2 == 1)
    assert torch.all(multipliers * 99091 > 0)
    config.engram_num_embeddings[1] -= 1
    with pytest.raises(ValueError, match="prime buckets"):
        EngramLayout.from_config(config)
    config = _config()
    config.engram_compressed_vocab_size = 10
    with pytest.raises(ValueError, match="compressed vocabulary"):
        EngramHashState(config, _Tokenizer(), "cpu")


def test_history_chunk_prefix_reorder_and_speculative_rollback():
    state = EngramHashState(_config(), _Tokenizer(), "cpu")
    # -1 is a physical-position barrier, not a token to remove from the history.
    histories = [[3, 6, 7, -1, -1, 4, 5, 1], [1, 11, 12, 3, 4, 6]]
    original = [list(history) for history in histories]
    requests = [0] * 8 + [1] * 6
    positions = list(range(8)) + list(range(6))
    ids = torch.tensor(
        [histories[r][p] for r, p in zip(requests, positions, strict=True)]
    )
    mask = ids != -1
    previous = build_engram_previous_tokens(histories, requests, positions)
    full = state(ids, previous, mask)
    torch.testing.assert_close(
        full, _reference_hashes(state, histories, requests, positions), rtol=0, atol=0
    )
    for order in ([7, 10, 5, 6], [4, 5, 6, 7], [11, 12, 13], [3]):
        reqs, pos = [requests[i] for i in order], [positions[i] for i in order]
        actual = state(
            ids[order], build_engram_previous_tokens(histories, reqs, pos), mask[order]
        )
        torch.testing.assert_close(actual, full[order], rtol=0, atol=0)
    proposed = [histories[0][:6] + [6, 7, 8]]
    _ = state(
        torch.tensor(proposed[0][6:]),
        build_engram_previous_tokens(proposed, [0, 0, 0], [6, 7, 8]),
        torch.ones(3, dtype=torch.bool),
    )
    # Reject two proposed tokens and replay the accepted history: nothing to reset.
    torch.testing.assert_close(
        state(ids[:8], previous[:8], mask[:8]), full[:8], rtol=0, atol=0
    )
    assert histories == original
    assert set(dict(state.named_buffers())) == {
        "token_map",
        "primes",
        "offsets",
        "multipliers",
    }
    assert not state.state_dict()
    assert build_engram_previous_tokens([], [], []).shape == (0, 3)
    with pytest.raises(ValueError, match="missing tokens"):
        build_engram_previous_tokens([[3]], [0], [2])
    with pytest.raises(ValueError, match="equal lengths"):
        build_engram_previous_tokens(histories, [0], [])
    # Mask the out-of-vocab current image placeholder before the token map gather.
    torch.testing.assert_close(
        state(torch.tensor([999999]), torch.tensor([[3, 4, 5]]), torch.tensor([False])),
        state(torch.tensor([-1]), torch.tensor([[3, 4, 5]]), torch.tensor([False])),
    )


def test_bounded_shard_loading_and_four_way_lookup(tmp_path, monkeypatch):
    rows, width = 19, 64
    codes, scales = _weights(rows, width)
    path = tmp_path / "table.safetensors"
    save_file(
        {"weight": codes, "scale": scales.view(torch.float8_e8m0fnu)},
        str(path),
        metadata=None,
    )
    calls = []

    def reduce_local(tensor, group, backend, op):
        assert group == (4, 5, 6, 7)  # not WORLD or the other attention DP replica
        assert backend is None and op == torch.distributed.ReduceOp.SUM
        return tensor

    monkeypatch.setattr(engram, "all_reduce", reduce_local)
    indices = torch.tensor([[0, 4, 5, 9, 10, 14, 15, 18, -1, 19]])
    outputs = []
    for rank in range(4, 8):
        embed = RowShardedEngramEmbedding(
            rows, width, _mapping(rank, 4, 8), "cpu", False, "gpu", 1
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in ("weight", "scale"):
                source = handle.get_slice(name)

                class TrackedSlice:
                    def get_shape(self):
                        return source.get_shape()

                    def __getitem__(self, key):
                        start, end = key[0].start, key[0].stop
                        assert embed.row_start <= start < end <= embed.row_end
                        assert end - start <= 2
                        calls.append((name, start, end))
                        return source[key]

                embed.load_sharded(name, TrackedSlice(), 2)
        assert embed.weight.dtype == torch.float8_e4m3fn
        assert embed.scale.dtype == torch.uint8
        assert embed.weight.numel() + embed.scale.numel() == 5 * (64 + 2)
        assert embed.weight.engram_row_sharded
        outputs.append(embed(indices))
        valid_rows = embed.row_end - embed.row_start
        torch.testing.assert_close(
            embed.weight[:valid_rows].view(torch.uint8),
            codes[embed.row_start : embed.row_end].view(torch.uint8),
        )
        if rank == 7:
            assert torch.all(embed.weight[-1].view(torch.uint8) == 0)
            assert torch.all(embed.scale[-1] == 127)
    expected = _dequant(codes, scales)[indices.clamp(0, rows - 1)]
    expected[indices < 0] = 0
    expected[indices >= rows] = 0
    torch.testing.assert_close(torch.stack(outputs).sum(0), expected, rtol=0, atol=0)
    assert len(calls) == 22


def test_scale_bytes_and_loader_validation():
    embed = RowShardedEngramEmbedding(8, 256, _mapping(0, 1, 1), "cpu", False, "gpu", 1)
    codes = torch.ones(8, 256).to(torch.float8_e4m3fn)
    # Includes E8M0 subnormal, extreme finite and NaN encodings.
    scales = torch.arange(256, dtype=torch.uint8).reshape(8, 32)[:, :8].contiguous()
    scales[0] = torch.tensor([0, 1, 125, 126, 127, 128, 254, 255], dtype=torch.uint8)
    embed.weight_loader(embed.weight, codes)
    embed.weight_loader(embed.scale, scales.view(torch.float8_e8m0fnu))
    torch.testing.assert_close(
        embed(torch.arange(8)), _dequant(codes, scales), rtol=0, atol=0, equal_nan=True
    )
    with pytest.raises(TypeError, match="must stay FP8"):
        embed.weight_loader(embed.weight, codes.float())
    with pytest.raises(TypeError, match="exponent bytes"):
        embed.weight_loader(embed.scale, scales.float())
    with pytest.raises(ValueError, match="row width"):
        embed.weight_loader(embed.weight, codes[:, :32])


def _model(device, quant_config):
    return DeepseekV41Engram(
        _config(),
        1,
        _mapping(0, 1, 1),
        quant_config,
        "model.layers.1.engram",
        device,
        False,
        "gpu",
    )


def test_gate_matches_reference_and_mask_is_identity():
    torch.manual_seed(41)
    model = _model("cpu", None)
    codes, scales = _weights(72, 32)
    model.embed.weight_loader(model.embed.weight, codes)
    model.embed.weight_loader(model.embed.scale, scales)
    model.wkv.weight.data.copy_(torch.randn(160, 192) * 0.03)
    model.q_weight.data.copy_(torch.randn(4, 32))
    model.k_weight.data.copy_(torch.randn(4, 32))
    ids = torch.randint(72, (2, 3, 6))
    hidden = torch.randn(2, 3, 4, 32, dtype=torch.bfloat16)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    kv = F.linear(_dequant(codes, scales)[ids].flatten(-2), model.wkv.weight)
    key, value = kv[..., :128].float().reshape(2, 3, 4, 32), kv[..., 128:].float()
    h = hidden.float()
    qnorm = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-20)
    knorm = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + 1e-20)
    dot = (qnorm * model.q_weight.float() * knorm * model.k_weight.float()).sum(
        -1
    ) / 32**0.5
    assert torch.any(dot < 0) and torch.any(dot > 0)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    gate[~mask] = 0
    expected = (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(hidden.dtype)
    torch.testing.assert_close(model(hidden, ids, mask), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        model(hidden, ids, mask)[~mask], hidden[~mask], rtol=0, atol=0
    )
    # Signed sqrt clamps zero to +sqrt(1e-6), rather than using sign(0) == 0.
    model.q_weight.data.zero_()
    expected = (h + torch.sigmoid(torch.tensor(0.001)) * value.unsqueeze(-2)).to(
        hidden.dtype
    )
    torch.testing.assert_close(
        model(hidden, ids, torch.ones_like(mask)), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("config_class", [Fp8Config, Mxfp8Config])
def test_quantized_projection_loader_aliases_and_real_table_metadata(config_class):
    quant = config_class.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
        }
    )
    model = _model("cpu", quant)
    assert model.wkv.weight.dtype == torch.float8_e4m3fn
    scale = torch.arange(30, dtype=torch.uint8).reshape(5, 6) % 4 + 125
    model.wkv.weight_scale_inv.weight_loader(
        model.wkv.weight_scale_inv, scale.view(torch.float8_e8m0fnu)
    )
    assert model.wkv.weight_scale_inv.dtype == torch.uint8
    assert model.wkv.quant_config.weight_block_size == [1, 32]
    assert quant.weight_block_size == [32, 32]
    expected = scale.repeat_interleave(32, dim=0)
    torch.testing.assert_close(model.wkv.weight_scale_inv, expected, rtol=0, atol=0)
    aliases = model.checkpoint_weight_aliases()
    assert aliases["layers.1.engram.embed.scale"] == "model.layers.1.engram.embed.scale"
    assert (
        aliases["layers.1.engram.wkv.scale"]
        == "model.layers.1.engram.wkv.weight_scale_inv"
    )
    assert {
        name.removeprefix("model.layers.1.engram.") for name in aliases.values()
    } == set(dict(model.named_parameters()))
    for rank in range(4):
        large = RowShardedEngramEmbedding(
            384016682, 256, _mapping(rank, 4, 4), "meta", False, "gpu", 1
        )
        assert large.weight.shape == (96004171, 256)
        assert large.scale.shape == (96004171, 8)
        assert large.weight.element_size() == large.scale.element_size() == 1
        assert large.row_end <= 384016682


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_lookup_hash_and_graph_replay():
    state = EngramHashState(_config(), _Tokenizer(), "cuda:0")
    embed = RowShardedEngramEmbedding(
        72, 32, _mapping(0, 1, 1), "cuda:0", False, "gpu", 1
    )
    codes, scales = _weights(72, 32)
    embed.weight_loader(embed.weight, codes)
    embed.weight_loader(embed.scale, scales)
    ids = torch.tensor([3, 4, 5], device="cuda:0")
    previous = torch.tensor([[-1, -1, -1], [3, -1, -1], [4, 3, -1]], device="cuda:0")
    mask = torch.ones(3, dtype=torch.bool, device="cuda:0")
    for _ in range(3):
        expected = embed(state(ids, previous, mask)[:, 0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = embed(state(ids, previous, mask)[:, 0])
    previous.copy_(torch.tensor([[6, 7, 8], [6, -1, 9], [11, 12, 1]], device="cuda:0"))
    mask[1] = False
    graph.replay()
    expected = embed(state(ids, previous, mask)[:, 0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_quantized_projection_forward():
    torch.manual_seed(41)
    quant = Mxfp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
        }
    )
    model = _model("cuda:0", quant)
    codes, scales = _weights(72, 32)
    model.embed.weight_loader(model.embed.weight, codes)
    model.embed.weight_loader(model.embed.scale, scales)
    # Power-of-two scales and exactly representable activations isolate W8A8 dispatch.
    weight, _ = _weights(160, 192)
    model.wkv.weight.weight_loader(
        model.wkv.weight, weight, shard_id=None, begin_size=None
    )
    scale = torch.full((5, 6), 121, dtype=torch.uint8)
    model.wkv.weight_scale_inv.weight_loader(model.wkv.weight_scale_inv, scale)
    model.wkv.quant_method.process_weights_after_loading(model.wkv)
    ids = torch.arange(24, device="cuda:0").reshape(4, 6)
    hidden = torch.randn(4, 4, 32, dtype=torch.bfloat16, device="cuda:0")
    mask = torch.ones(4, dtype=torch.bool, device="cuda:0")
    actual = model(hidden, ids, mask)
    reference = _model("cuda:0", None)
    reference.embed.weight_loader(reference.embed.weight, codes)
    reference.embed.weight_loader(reference.embed.scale, scales)
    reference.wkv.weight.data.copy_((weight.float() / 64).to(torch.bfloat16))
    expected = reference(hidden, ids, mask)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert model.wkv.weight.dtype == torch.float8_e4m3fn
    for _ in range(2):
        model(hidden, ids, mask)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model(hidden, ids, mask)
    hidden.mul_(0.5)
    mask[1] = False
    graph.replay()
    torch.testing.assert_close(captured, model(hidden, ids, mask), rtol=0, atol=0)
    empty = model(hidden[:0], ids[:0], mask[:0])
    assert empty.shape == (0, 4, 32)


def test_distributed_four_way_lookup():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("launch with torchrun --nproc_per_node=4")
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    rank = int(os.environ["RANK"])
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    mapping = _mapping(rank, 4, 4)
    process_group_manager.init_distributed(
        mapping,
        distributed_init_method="env://",
        backend="nccl",
        timeout=60,
        device_id=device,
    )
    process_group_manager.init_process_group(mapping.attn.tp_group, backend="nccl")
    try:
        codes, scales = _weights(19, 64)
        table = RowShardedEngramEmbedding(19, 64, mapping, device, False, "gpu", 1)
        table.weight_loader(table.weight, codes)
        table.weight_loader(table.scale, scales)
        ids = torch.tensor([[0, 4, 5, 9, 10, 14, 15, 18]], device=device)
        torch.testing.assert_close(
            table(ids), _dequant(codes, scales)[ids.cpu()].to(device), rtol=0, atol=0
        )
    finally:
        torch.distributed.destroy_process_group()


def test_local_snapshot_reference_parity():
    directory = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
    if directory is None:
        pytest.skip("set DEEPSEEK_V41_REFERENCE_DIR for local upstream parity")
    from transformers import PreTrainedTokenizerFast

    snapshot = Path(directory)
    config = SimpleNamespace(
        **json.loads((snapshot / "config.json").read_text())["text_config"]
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(snapshot / "tokenizer.json"))
    spec = importlib.util.spec_from_file_location(
        "_v41_reference_engram", snapshot / "inference/engram.py"
    )
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    state = EngramHashState(config, tokenizer, "cpu")
    quant = Fp8Config.from_config(
        json.loads((snapshot / "config.json").read_text())["quantization_config"]
    )
    full = DeepseekV41Engram(
        config,
        1,
        _mapping(0, 4, 4),
        quant,
        "model.layers.1.engram",
        "meta",
        False,
        "gpu",
    )
    assert full.wkv.weight.shape == (25600, 6144)
    assert full.wkv.weight_scale_inv.shape == (25600, 192)
    assert full.q_weight.shape == full.k_weight.shape == (4, 5120)
    assert full.embed.weight.shape == (96001542, 256)
    token_map, vocab_size = reference.build_compressed_token_map(tokenizer)
    assert vocab_size == 99092
    assert state.token_map.tolist() == token_map
    torch.testing.assert_close(
        state.multipliers,
        reference.compute_hash_multipliers((1, 14), 4, vocab_size),
        rtol=0,
        atol=0,
    )
    args = SimpleNamespace(
        **vars(config),
        engram_pad_id=config.engram_pad_token_id,
        max_batch_size=2,
        max_seq_len=16,
    )
    layout = reference.EngramLayout.from_args(args)
    assert state.layout.primes == layout.primes
    ref_hash = reference.NgramHashState(args, layout, tokenizer)
    ids = torch.tensor(
        [[0, 3, 700, 12456, 129264, 129264, 500, 1], [0, 2, 3, 4, 5, 6, 7, 8]]
    )
    mask = ids != 129264
    histories = ids.masked_fill(~mask, -1).tolist()
    previous = build_engram_previous_tokens(
        histories, [0] * 8 + [1] * 8, list(range(8)) * 2
    ).reshape(2, 8, 3)
    expected = ref_hash(ids, start_pos=0, token_mask=mask)
    torch.testing.assert_close(state(ids, previous, mask), expected, rtol=0, atol=0)
    for begin, end in ((0, 3), (3, 5), (5, 8)):
        torch.testing.assert_close(
            state(ids[:, begin:end], previous[:, begin:end], mask[:, begin:end]),
            ref_hash(ids[:, begin:end], start_pos=begin, token_mask=mask[:, begin:end]),
            rtol=0,
            atol=0,
        )
    # Execute the actual upstream gate method without importing its GPU kernels.
    tree = ast.parse((snapshot / "inference/model.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Engram"
    )
    forward = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[forward], type_ignores=[]),
            "reference_engram_forward",
            "exec",
        ),
        namespace,
    )
    model = _model("cpu", None)
    codes, scales = _weights(72, 32)
    model.embed.weight_loader(model.embed.weight, codes)
    model.embed.weight_loader(model.embed.scale, scales)
    torch.manual_seed(41)
    model.wkv.weight.data.copy_(torch.randn(160, 192) * 0.02)
    hidden = torch.randn(2, 3, 4, 32, dtype=torch.bfloat16)
    hash_ids = torch.randint(72, (2, 3, 6))
    token_mask = torch.tensor([[True, False, True], [False, True, True]])
    ref_model = SimpleNamespace(
        embed=model.embed,
        wkv=lambda x: F.linear(x, model.wkv.weight),
        q_weight=model.q_weight,
        k_weight=model.k_weight,
        hc_mult=4,
        dim=32,
        eps=1e-20,
        clamp_value=1e-6,
    )
    torch.testing.assert_close(
        model(hidden, hash_ids, token_mask),
        namespace["forward"](ref_model, hidden, hash_ids, token_mask),
        rtol=0,
        atol=0,
    )


def test_host_table_matches_gpu_shard_and_skips_allreduce(tmp_path, monkeypatch):
    rows, width = 19, 64
    codes, scales = _weights(rows, width)
    path = tmp_path / "table.safetensors"
    save_file(
        {"weight": codes, "scale": scales.view(torch.float8_e8m0fnu)},
        str(path),
        metadata=None,
    )
    reduces = []

    def reduce_local(tensor, group, backend, op):
        reduces.append(group)
        return tensor

    monkeypatch.setattr(engram, "all_reduce", reduce_local)
    indices = torch.tensor([[0, 4, 5, 9, 10, 14, 15, 18, -1, 19]])
    sharded = []
    for rank in range(4):
        embed = RowShardedEngramEmbedding(
            rows, width, _mapping(rank, 4, 4), "cpu", False, "gpu", 1
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            embed.load_sharded("weight", handle.get_slice("weight"), 4)
            embed.load_sharded("scale", handle.get_slice("scale"), 4)
        sharded.append(embed(indices))
    gpu_path = torch.stack(sharded).sum(0)
    assert reduces
    host = RowShardedEngramEmbedding(
        rows, width, _mapping(0, 4, 4), "cpu", True, "shared", 1
    )
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        host.load_sharded("weight", handle.get_slice("weight"), 4)
        host.load_sharded("scale", handle.get_slice("scale"), 4)
    reduces.clear()
    host_out = host(indices)
    assert reduces == []
    assert host.weight.shape == (rows, width)
    torch.testing.assert_close(host_out, gpu_path, rtol=0, atol=0)


def test_host_table_meta_keeps_full_rows():
    table = RowShardedEngramEmbedding(
        384016682, 256, _mapping(0, 4, 4), "meta", True, "shared", 14
    )
    assert table.weight.shape == (384016682, 256)
    assert table.scale.shape == (384016682, 8)
    assert table.row_start == 0 and table.row_end == 384016682


def test_engram_host_table_server_arg():
    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model", "x", "--engram-host-table"])
    assert args.engram_host_table is True
    assert args.engram_host_table_layout == "auto"
    args = parser.parse_args(["--model", "x", "--no-engram-host-table"])
    assert args.engram_host_table is False
    args = parser.parse_args(
        ["--model", "x", "--engram-host-table-dir", "/scratch/engram"]
    )
    assert args.engram_host_table_dir == "/scratch/engram"
    args = parser.parse_args(
        ["--model", "x", "--engram-host-table", "--engram-host-table-layout", "sharded"]
    )
    assert args.engram_host_table_layout == "sharded"


def test_resolve_engram_host_layout_auto_and_explicit():
    from tokenspeed.runtime.models.deepseek_v41_engram import resolve_engram_host_layout
    from tokenspeed.runtime.utils.env import global_server_args_dict

    previous = global_server_args_dict["engram_host_table_layout"]
    try:
        global_server_args_dict["engram_host_table_layout"] = "auto"
        assert resolve_engram_host_layout(False, 4) == "gpu"
        assert resolve_engram_host_layout(True, 1) == "shared"
        assert resolve_engram_host_layout(True, 4) == "sharded"
        global_server_args_dict["engram_host_table_layout"] = "shared"
        assert resolve_engram_host_layout(True, 4) == "shared"
        global_server_args_dict["engram_host_table_layout"] = "sharded"
        assert resolve_engram_host_layout(True, 1) == "sharded"
    finally:
        global_server_args_dict["engram_host_table_layout"] = previous


def test_host_sharded_table_matches_gpu_and_allreduces(tmp_path, monkeypatch):
    rows, width = 19, 64
    codes, scales = _weights(rows, width)
    path = tmp_path / "table.safetensors"
    save_file(
        {"weight": codes, "scale": scales.view(torch.float8_e8m0fnu)},
        str(path),
        metadata=None,
    )
    reduces = []

    def reduce_local(tensor, group, backend, op):
        reduces.append(group)
        return tensor

    monkeypatch.setattr(engram, "all_reduce", reduce_local)
    indices = torch.tensor([[0, 4, 5, 9, 10, 14, 15, 18, -1, 19]])
    gpu_parts = []
    host_parts = []
    for rank in range(4):
        mapping = _mapping(rank, 4, 4)
        gpu = RowShardedEngramEmbedding(rows, width, mapping, "cpu", False, "gpu", 1)
        host = RowShardedEngramEmbedding(
            rows, width, mapping, "cpu", True, "sharded", 1
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            gpu.load_sharded("weight", handle.get_slice("weight"), 4)
            gpu.load_sharded("scale", handle.get_slice("scale"), 4)
            host.load_sharded("weight", handle.get_slice("weight"), 4)
            host.load_sharded("scale", handle.get_slice("scale"), 4)
        assert host.weight.shape == gpu.weight.shape
        assert host.row_start == gpu.row_start
        assert host.weight.engram_row_sharded
        gpu_parts.append(gpu(indices))
        host_parts.append(host(indices))
    assert reduces
    torch.testing.assert_close(
        torch.stack(host_parts).sum(0),
        torch.stack(gpu_parts).sum(0),
        rtol=0,
        atol=0,
    )


def test_host_sharded_meta_keeps_rank_rows():
    table = RowShardedEngramEmbedding(
        384016682, 256, _mapping(1, 4, 4), "meta", True, "sharded", 14
    )
    assert table.weight.shape == (96004171, 256)
    assert table.row_start == 96004171
    assert table.row_end == 192008342


def test_advise_hugepages_is_linux_only(monkeypatch):
    from tokenspeed.runtime.models.deepseek_v41_engram import _advise_hugepages

    called = []

    def fake_madvise(addr, size, advice):
        called.append(int(advice))
        return 0

    class FakeLibc:
        madvise = staticmethod(fake_madvise)

    monkeypatch.setattr(engram.sys, "platform", "darwin")
    monkeypatch.setattr(engram.ctypes.util, "find_library", lambda name: "c")
    monkeypatch.setattr(engram.ctypes, "CDLL", lambda name: FakeLibc())
    buffer = engram.np.empty(4096, dtype=engram.np.uint8)
    _advise_hugepages(buffer, 4096)
    assert called == []
    monkeypatch.setattr(engram.sys, "platform", "linux")
    _advise_hugepages(buffer, 4096)
    assert called == [14]


def test_host_table_dir_skips_small_shm(monkeypatch, tmp_path):
    from tokenspeed.runtime.models.deepseek_v41_engram import _host_table_dir
    from tokenspeed.runtime.utils.env import global_server_args_dict

    global_server_args_dict["engram_host_table_dir"] = None

    class Usage:
        def __init__(self, free):
            self.free = free

    def fake_usage(path):
        if path == "/dev/shm":
            return Usage(32 * 1024**3)
        return Usage(8 * 1024**4)

    monkeypatch.setattr(engram.shutil, "disk_usage", fake_usage)
    monkeypatch.setattr(engram.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(engram.os, "access", lambda p, mode: True)
    chosen = _host_table_dir(90 * 1024**3)
    assert chosen != "/dev/shm"
    global_server_args_dict["engram_host_table_dir"] = str(tmp_path)
    assert _host_table_dir(90 * 1024**3) == str(tmp_path)
    global_server_args_dict["engram_host_table_dir"] = None


def test_host_table_job_id_is_pid_scoped(tmp_path, monkeypatch):
    from tokenspeed.runtime.models.deepseek_v41_engram import (
        _engram_host_table_path,
        _host_table_job_id,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    global_server_args_dict["engram_host_table_dir"] = str(tmp_path)
    engram._HOST_TABLE_JOB_ID = None
    path = _engram_host_table_path(14, "weight", 1024)
    assert str(os.getpid()) in path
    assert path.startswith(str(tmp_path))
    assert _host_table_job_id() == str(os.getpid())
    global_server_args_dict["engram_host_table_dir"] = None
    engram._HOST_TABLE_JOB_ID = None


def test_checkpoint_filter_skips_engram_embed_only():
    from tokenspeed.runtime.models.deepseek_v41_engram import (
        is_engram_embed_checkpoint_name,
    )

    assert is_engram_embed_checkpoint_name("layers.1.engram.embed.weight")
    assert is_engram_embed_checkpoint_name("model.layers.1.engram.embed.scale")
    assert not is_engram_embed_checkpoint_name("layers.1.engram.wkv.weight")
    assert not is_engram_embed_checkpoint_name("layers.0.attn.wq_a.weight")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_host_table_gather_matches_device_shard():
    codes, scales = _weights(72, 32)
    device_embed = RowShardedEngramEmbedding(
        72, 32, _mapping(0, 1, 1), "cuda:0", False, "gpu", 1
    )
    device_embed.weight_loader(device_embed.weight, codes)
    device_embed.weight_loader(device_embed.scale, scales)
    host_embed = RowShardedEngramEmbedding(
        72, 32, _mapping(0, 1, 1), "cpu", True, "shared", 1
    )
    host_embed.weight_loader(host_embed.weight, codes)
    host_embed.weight_loader(host_embed.scale, scales)
    ids = torch.tensor([3, 4, 5, 0, 71], device="cuda:0")
    torch.testing.assert_close(host_embed(ids), device_embed(ids), rtol=0, atol=0)
    host_embed._register_host_for_gpu = lambda: (_ for _ in ()).throw(
        AssertionError("host gather must not register")
    )
    host_embed(ids)
