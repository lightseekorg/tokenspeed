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

"""Two-rank CPU regression for V4.1 DP/EP collective ordering.

Run with ``torchrun --standalone --nproc_per_node=2 -m pytest`` and this file,
or use ordinary pytest to launch the registered two-rank regression.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from test.runtime.test_deepseek_v41_model import (
    DeepseekV41Model,
    _config,
    _ctx,
    _DenseFFN,
    _initialize,
)
from types import SimpleNamespace

import psutil
import pytest
import torch
import torch.distributed as dist
from ci_system.ci_register import register_cuda_ci
from torch import nn

from tokenspeed.runtime.distributed import comm_ops
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    V41DecoderView,
)
from tokenspeed.runtime.layers.moe.utils import MoeBackend
from tokenspeed.runtime.models import deepseek_v41 as v41
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(est_time=60, suite="runtime-1gpu")


class _GlooCommTransport:
    """CPU transport for the real CommManager row collectives."""

    def __init__(self):
        self.gathered: list[torch.Tensor] = []

    def token_all_gather(self, tensor, group, scattered_num_tokens):
        assert tuple(group) == (0, 1)
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.bfloat16
        assert len(scattered_num_tokens) == 2
        max_rows = max(scattered_num_tokens)
        padded = tensor.new_zeros((max_rows, tensor.shape[1]))
        padded[: tensor.shape[0]].copy_(tensor)
        received = [torch.empty_like(padded) for _ in scattered_num_tokens]
        dist.all_gather(received, padded, group=dist.group.WORLD)
        result = torch.cat(
            [part[:rows] for part, rows in zip(received, scattered_num_tokens)],
            dim=0,
        )
        self.gathered.append(result.detach().clone())
        return result

    def token_reduce_scatter(self, tensor, group, scattered_num_tokens):
        assert tuple(group) == (0, 1)
        assert tensor.device.type == "cpu"
        assert tensor.shape[0] == sum(scattered_num_tokens)
        reduced = tensor.contiguous().clone()
        dist.all_reduce(reduced, group=dist.group.WORLD)
        rank = dist.get_rank()
        offset = sum(scattered_num_tokens[:rank])
        return reduced[offset : offset + scattered_num_tokens[rank]].contiguous()


class _CpuAttention(nn.Module):
    """Replace only the unavailable attention kernel while retaining layer flow."""

    def __init__(self, mapping, layer_index, decoder_start):
        super().__init__()
        self.mapping = mapping
        self.layer_index = layer_index
        self.decoder_start = decoder_start
        self.wq_a_wkv = SimpleNamespace(quant_method=None)

    def forward(self, positions, hidden_states, ctx, *, reduce_output, **kwargs):
        assert reduce_output
        assert hidden_states.device.type == "cpu"
        if self.layer_index == self.decoder_start:
            keep = ctx.attn_backend.decoder_view().keep_rows
            if keep is not None:
                return hidden_states.index_select(0, keep)
        return hidden_states


def _cpu_model(rank, world_size, moe_ep_size, monkeypatch, transport):
    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=moe_ep_size,
    )
    config = _config()
    config.num_hidden_layers = 3
    config.compress_ratios = [1, 1, 1]
    config.kv_source_layer_ids = [1]
    config.index_source_layer_ids = [1]
    config.candidate_source_layer_id = 1
    config.engram_layer_ids = []
    config.engram_num_embeddings = []
    monkeypatch.setattr(v41, "DeepseekV41MoE", _DenseFFN)
    monkeypatch.setattr(v41, "get_moe_backend", lambda: MoeBackend.TRITON)
    monkeypatch.setattr(comm_ops, "get_global_backend", lambda: transport)
    monkeypatch.setitem(global_server_args_dict, "disable_prefill_graph", True)
    torch.manual_seed(17)
    model = DeepseekV41Model(config, mapping, None, "model", False, "cpu", None)
    _initialize(model)
    for index, layer in enumerate(model.layers):
        layer.attn = _CpuAttention(mapping, index, model.ced_decoder_start)
    with torch.no_grad():
        model.embed_tokens.weight.zero_()
        for token in range(config.vocab_size):
            model.embed_tokens.weight[token, 0] = 1
            model.embed_tokens.weight[token, 1] = (token + 1) / 16
    return model


class _CpuAttentionBackend:
    def __init__(self, token_ids, keep_rows):
        self.meta = SimpleNamespace(
            positions=torch.arange(token_ids.numel(), dtype=torch.int64),
            request_indices=torch.zeros(token_ids.numel(), dtype=torch.int64),
        )
        tail_positions = self.meta.positions.index_select(0, keep_rows)
        tail_requests = self.meta.request_indices.index_select(0, keep_rows)
        self.view = V41DecoderView(
            metadata=SimpleNamespace(
                positions=tail_positions, request_indices=tail_requests
            ),
            prefill=None,
            spans=(),
            keep_rows=keep_rows,
            logits_rows=None,
        )

    def begin_indexer_forward(self):
        pass

    def end_indexer_forward(self):
        pass

    def begin_indexer_stage(self):
        pass

    def prepare_decode_index_geometry(self, forward_mode, *, index_heads):
        # Decode builds index geometry ahead of the layers; this backend
        # keeps none.
        pass

    def prepare_reindex_index_geometry(self, forward_mode, *, index_heads):
        pass

    def query_metadata(self, mode):
        return self.meta

    def decoder_view(self):
        return self.view


def _stop_owned_process_tree(process, root, known_children):
    """Stop a Popen process and the descendants it still owns, with bounded waits."""
    tree_error = None
    descendants = dict(known_children)
    if root is not None:
        try:
            descendants.update(
                (child.pid, child) for child in root.children(recursive=True)
            )
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            tree_error = error
        except OSError as error:
            tree_error = error

    descendants = list(descendants.values())
    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if root is not None:
        try:
            root.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(descendants, timeout=2)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=2)
    running = []
    for child in alive:
        try:
            if child.status() != psutil.STATUS_ZOMBIE:
                running.append(child.pid)
        except psutil.NoSuchProcess:
            pass
    if running:
        raise TimeoutError(f"could not stop owned child processes: {running}")

    if process.poll() is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if root is not None:
                try:
                    root.kill()
                except psutil.NoSuchProcess:
                    pass
            process.wait(timeout=2)
    if tree_error is not None:
        raise TimeoutError("could not enumerate owned child processes") from tree_error


def _run_gloo_workers(command, timeout):
    """Run a subprocess tree with bounded completion and timeout cleanup."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        root = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        root = None
    known_children = {}
    deadline = time.monotonic() + timeout
    try:
        try:
            while True:
                if root is not None:
                    try:
                        known_children.update(
                            (child.pid, child)
                            for child in root.children(recursive=True)
                        )
                    except psutil.NoSuchProcess:
                        pass
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
        except subprocess.TimeoutExpired as timeout_error:
            _stop_owned_process_tree(process, root, known_children)
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired as cleanup_error:
                stdout = cleanup_error.output or timeout_error.output or ""
                stderr = cleanup_error.stderr or timeout_error.stderr or ""
                raise TimeoutError(
                    f"child pipes remained open after process cleanup\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                ) from timeout_error
            raise TimeoutError(
                f"command timed out after {timeout} seconds\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            ) from timeout_error
        return process.returncode, stdout, stderr
    finally:
        try:
            _stop_owned_process_tree(process, root, known_children)
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


def _case_rows(name, rank):
    if name == "prefill_idle":
        return (2, 2) if rank == 0 else (0, 0)
    if name == "idle_prefill":
        return (0, 0) if rank == 0 else (2, 2)
    if name == "open_prefill":
        return (1, 0) if rank == 0 else (2, 2)
    if name == "prefill_open":
        return (2, 2) if rank == 0 else (1, 0)
    if name == "decode_prefill":
        return (1, 1) if rank == 0 else (2, 2)
    if name == "prefill_decode":
        return (2, 2) if rank == 0 else (1, 1)
    if name == "both_open":
        return (1, 0) if rank == 0 else (2, 0)
    if name == "both_open_swapped":
        return (2, 0) if rank == 0 else (1, 0)
    raise AssertionError(name)


def _run_round(model, reference_model, rank, name, round_index, transport):
    local_rows, local_tail_rows = _case_rows(name, rank)
    peer_rows, peer_tail_rows = _case_rows(name, 1 - rank)
    if name in ("prefill_idle", "idle_prefill"):
        mode = ForwardMode.EXTEND if local_rows else ForwardMode.IDLE
    elif name in ("decode_prefill", "prefill_decode"):
        mode = (
            ForwardMode.DECODE
            if (name == "decode_prefill") == (rank == 0)
            else ForwardMode.EXTEND
        )
    else:
        mode = ForwardMode.EXTEND

    token_base = 10 + rank * 10 + round_index
    ids = torch.arange(token_base, token_base + local_rows, dtype=torch.int64)
    keep_rows = torch.arange(local_tail_rows, dtype=torch.int64)
    backend = _CpuAttentionBackend(ids, keep_rows)
    ctx = _ctx(backend, local_rows, mode)
    ctx.global_num_tokens = [
        _case_rows(name, 0)[0],
        _case_rows(name, 1)[0],
    ]
    ctx.all_decode_or_idle = False

    gathered_before = len(transport.gathered)
    output, captured = model(
        ids,
        backend.meta.positions,
        ctx,
        None,
        None,
        engram_hash_ids=None,
        engram_token_mask=None,
        image_mask=None,
    )
    assert captured is None
    assert output.shape == (local_tail_rows, model.config.hidden_size)
    assert torch.isfinite(output).all()
    if local_tail_rows:
        assert output.abs().sum() > 0
    assert len(transport.gathered) - gathered_before == model.config.num_hidden_layers

    # The first gather is before CED; later gathers use counts returned by the
    # two real _exchange_decoder_counts calls (candidate layer and decoder).
    round_gathers = transport.gathered[gathered_before:]
    expected_rows = [local_rows + peer_rows, local_tail_rows + peer_tail_rows]
    expected_rows.extend([local_tail_rows + peer_tail_rows] * (len(round_gathers) - 2))
    assert [gather.shape[0] for gather in round_gathers] == expected_rows

    first_layer_gather = round_gathers[0]
    if local_tail_rows == 0:
        assert output.numel() == 0

    # Compare the distributed result with a single-rank execution over the
    # same rank-ordered input. The reference FFN doubles its weight to match
    # the two identical FFN contributions summed by Gloo reduce-scatter.
    ids_by_rank = [
        torch.arange(
            10 + source_rank * 10 + round_index,
            10 + source_rank * 10 + round_index + _case_rows(name, source_rank)[0],
            dtype=torch.int64,
        )
        for source_rank in range(2)
    ]
    all_ids = torch.cat(ids_by_rank)
    global_keep = torch.cat(
        [
            sum(_case_rows(name, prior_rank)[0] for prior_rank in range(source_rank))
            + torch.arange(_case_rows(name, source_rank)[1], dtype=torch.int64)
            for source_rank in range(2)
        ]
    )
    reference_backend = _CpuAttentionBackend(all_ids, global_keep)
    reference_ctx = _ctx(reference_backend, all_ids.numel(), ForwardMode.EXTEND)
    reference_layer_inputs = []
    reference_hook = reference_model.layers[0].ffn.register_forward_pre_hook(
        lambda _module, args: reference_layer_inputs.append(args[0].detach().clone())
    )
    expected_output, expected_captured = reference_model(
        all_ids,
        reference_backend.meta.positions,
        reference_ctx,
        None,
        None,
        engram_hash_ids=None,
        engram_token_mask=None,
        image_mask=None,
    )
    reference_hook.remove()
    assert expected_captured is None
    assert len(reference_layer_inputs) == 1
    torch.testing.assert_close(
        first_layer_gather, reference_layer_inputs[0], atol=1e-2, rtol=1e-2
    )

    counts = [_case_rows(name, source_rank)[1] for source_rank in range(2)]
    max_rows = max(counts)
    if max_rows:
        padded = output.new_zeros((max_rows, output.shape[-1]))
        padded[: output.shape[0]].copy_(output)
        received = [torch.empty_like(padded) for _ in counts]
        dist.all_gather(received, padded, group=dist.group.WORLD)
        actual_output = torch.cat(
            [part[:rows] for part, rows in zip(received, counts)], dim=0
        )
    else:
        actual_output = output.new_empty((0, output.shape[-1]))
    torch.testing.assert_close(actual_output, expected_output, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo is required"
)
@pytest.mark.skipif(
    "RANK" not in os.environ or os.environ.get("WORLD_SIZE") != "2",
    reason="run this regression with torchrun and two ranks",
)
def test_dp_ep_collectives_keep_real_two_rank_data_flow(monkeypatch):
    if not dist.is_initialized():
        pg_manager.init_distributed(
            Mapping(
                rank=int(os.environ["RANK"]),
                world_size=2,
                attn_tp_size=1,
                dense_tp_size=1,
                moe_tp_size=1,
                moe_ep_size=2,
            ),
            backend="gloo",
            timeout=90,
        )
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    transport = _GlooCommTransport()
    model = _cpu_model(rank, 2, 2, monkeypatch, transport)
    reference_model = _cpu_model(0, 1, 1, monkeypatch, transport)
    with torch.no_grad():
        for layer in reference_model.layers:
            layer.ffn.weight.mul_(2)

    # Repeat with the active role on each rank so count and payload ownership
    # cannot accidentally be tied to rank zero.
    for round_index, name in enumerate(
        (
            "prefill_idle",
            "open_prefill",
            "decode_prefill",
            "both_open",
            "both_open_swapped",
            "prefill_decode",
            "prefill_open",
            "idle_prefill",
        )
    ):
        _run_round(model, reference_model, rank, name, round_index, transport)

    dist.barrier()


@pytest.mark.skipif(
    "RANK" in os.environ,
    reason="the distributed worker is already running under torchrun",
)
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
def test_dp_ep_collectives_ci_runs_two_rank_gloo():
    """Run the real distributed regression from ordinary single-process pytest."""
    test_file = str(Path(__file__).resolve())
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "pytest",
        "-q",
        f"{test_file}::test_dp_ep_collectives_keep_real_two_rank_data_flow",
    ]
    try:
        returncode, stdout, stderr = _run_gloo_workers(command, timeout=180)
    except TimeoutError as error:
        pytest.fail(str(error))

    assert returncode == 0, (
        f"two-rank Gloo regression exited with {returncode}\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )


@pytest.mark.parametrize("launcher_exits_early", [False, True])
def test_gloo_worker_timeout_kills_detached_descendant(tmp_path, launcher_exits_early):
    marker = tmp_path / "worker.json"
    worker_code = (
        "import json, os, psutil, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"worker = psutil.Process(); open({str(marker)!r}, 'w').write("
        "json.dumps([worker.pid, worker.create_time()]))\n"
        "print('detached worker ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    launcher_code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {worker_code!r}], "
        "start_new_session=True)\n"
        f"time.sleep({0.5 if launcher_exits_early else 60})\n"
    )

    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="command timed out"):
            _run_gloo_workers([sys.executable, "-c", launcher_code], timeout=5)
        assert time.monotonic() - started < 20

        worker_pid, worker_create_time = json.loads(marker.read_text())
        try:
            worker = psutil.Process(worker_pid)
        except psutil.NoSuchProcess:
            return
        if worker.create_time() == worker_create_time:
            assert worker.status() == psutil.STATUS_ZOMBIE or not worker.is_running()
    finally:
        if marker.exists():
            worker_pid, worker_create_time = json.loads(marker.read_text())
            try:
                worker = psutil.Process(worker_pid)
            except psutil.NoSuchProcess:
                pass
            else:
                if (
                    worker.create_time() == worker_create_time
                    and worker.status() != psutil.STATUS_ZOMBIE
                ):
                    try:
                        worker.kill()
                        worker.wait(timeout=2)
                    except psutil.NoSuchProcess:
                        pass
