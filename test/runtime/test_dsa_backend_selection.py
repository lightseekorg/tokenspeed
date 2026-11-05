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

"""One DSA launch name selects the device implementation and its constraints."""

import logging
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=1, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import AttentionArch  # noqa: E402
from tokenspeed.runtime.layers.attention import backends  # noqa: E402
from tokenspeed.runtime.layers.attention.configs import base as configs  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.registry import (  # noqa: E402
    _get_backend_cls,
    validate_attention_backend_name,
)


def _config(monkeypatch, device, backend_name, **overrides):
    # The NPU device spelling is registered only when torch_npu is installed.
    # Exercise the device-dependent config policy on CPU/CUDA CI as well.
    monkeypatch.setattr(
        configs,
        "torch",
        SimpleNamespace(
            device=lambda name: SimpleNamespace(type=name),
            bfloat16=torch.bfloat16,
            int8=torch.int8,
        ),
    )
    packed_width = overrides.pop("packed_width", 576)
    spec = DSAConfig(
        backend_name=backend_name,
        num_attention_heads=64,
        num_kv_heads=1,
        attn_tp_size=8,
        head_dim=192,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=1.0,
        kv_cache_dim=packed_width,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=16,
        index_k_format="bf16",
    )
    fields = dict(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=4096,
        max_bs=4,
        kvp_size=8,
        kvp_rank=0,
        kvp_group=tuple(range(8)),
        components=(spec,),
    )
    fields.update(overrides)
    return configs.AttnConfig(**fields), spec


@pytest.mark.parametrize("name", [None, "dsa"])
def test_dsa_name_resolves_to_the_device_implementation(name):
    expected = (
        backends.ascend_dsa.AscendDSABackend
        if current_platform().is_npu
        else backends.dsa.DSABackend
    )
    validate_attention_backend_name(name, flag="--attention-backend")
    assert _get_backend_cls(name, AttentionArch.DSA) is expected


@pytest.mark.parametrize("name", [None, "dsa"])
@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_dsa_dcp_config_accepts_the_common_name(monkeypatch, device, name):
    config, _ = _config(monkeypatch, device, name)
    assert config.kvp_size == 8


def test_dsa_dcp_still_rejects_cpu(monkeypatch):
    with pytest.raises(ValueError, match="requires CUDA or Ascend NPU"):
        _config(monkeypatch, "cpu", "dsa")


def test_gpu_dsa_dcp_still_rejects_block_drafts(monkeypatch):
    with pytest.raises(ValueError, match="non-causal block drafts"):
        _config(monkeypatch, "cuda", "dsa", is_draft=True, draft_block_decode=True)


def test_ascend_draft_constraint_is_owned_by_the_backend(monkeypatch):
    config, spec = _config(monkeypatch, "npu", "dsa", is_draft=True)

    def initialize_base(self, config, spec, *, kernel_page_size):
        self.is_draft = config.is_draft

    monkeypatch.setattr(backends.dsa.DSABackend, "__init__", initialize_base)
    backend = _get_backend_cls("dsa", AttentionArch.DSA)
    # CUDA CI verifies the NPU guard directly, before native ops initialize.
    if not current_platform().is_npu:
        backend = backends.ascend_dsa.AscendDSABackend
    with pytest.raises(ValueError, match="draft attention is disabled"):
        backend(config, spec, kernel_page_size=128)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_ascend_bf16_kv_keeps_native_row_layout(monkeypatch):
    config, spec = _config(monkeypatch, "npu", "dsa")
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        _mla_layer_fields,
    )

    fields = _mla_layer_fields(config, 128, 0, 0)
    assert fields[0].shape == (128, 1, 576)
    assert fields[0].dtype == "bfloat16"
    assert fields[0].payload_bytes == 128 * 576 * 2
    assert spec.cache_cell_size(config) == (576 + 128) * 2


@pytest.mark.parametrize("device", ["cuda", "npu"])
@pytest.mark.parametrize("width", [576, 672])
def test_int8_cache_is_rejected_before_storage_planning(monkeypatch, device, width):
    with pytest.raises(ValueError, match="INT8 KV cache is not supported"):
        _config(
            monkeypatch, device, "dsa", kv_cache_dtype=torch.int8, packed_width=width
        )


def test_int8_cache_is_rejected_by_server_arguments():
    import argparse

    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--model", "x", "--kv-cache-dtype", "int8"])
    assert error.value.code == 2


# A K3.1 DSA leaf as the sparse-leaf log sees it: kvp 1, BF16 KV, 8-token DSpark verify.
_K31_SPARSE_LEAF = dict(
    device="cpu",
    kernel_page_size=64,
    kvp_group=(0,),
    kvp_rank=0,
    qcp_group=(0,),
    data_type=torch.bfloat16,
    q_data_type=torch.bfloat16,
    qk_nope_head_dim=128,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    index_topk=2048,
    index_n_heads=32,
    index_head_dim=128,
    index_k_format="fp8_scaled",
    slot_order="selection",
    spec_num_tokens=8,
    kernel_solution=None,
    prefill_kernel_plan=None,
    decode_kernel_plan=None,
    kpool_runtime=None,
    kvp_prefill_query_layout=False,
    _dcp_records_core=False,
    _dcp_padded_query=False,
    _dcp_decode_records_core=False,
)


def _sparse_leaf_backend(monkeypatch, **fields):
    """A DSA leaf holding only what the sparse-leaf log reads (K3.1 geometry),
    with fake selection and registry and a fresh log-once set. Selection
    finds no verify-width decode core."""
    dsa = backends.dsa
    backend = object.__new__(dsa.DSABackend)
    values = dict(_K31_SPARSE_LEAF)
    values.update(fields)
    for name, value in values.items():
        setattr(backend, name, value)
    calls = []

    def select(family, mode, signature, *, traits, solution):
        assert family == "attention"
        calls.append((mode, signature, traits, solution))
        if mode == "dsa_decode" and traits["q_len"] > 1:
            raise NoKernelFoundError("no verify-width core")
        return SimpleNamespace(name=f"fake_{mode}")

    priorities = {
        "fake_dsa_prefill": 12,
        "fake_dsa_decode": 4,
        "fake_dsa_index_candidates": 9,
        "planned_prefill": 8,
        "planned_decode": 8,
    }
    registry = SimpleNamespace(
        get_by_name=lambda name: (
            SimpleNamespace(priority=priorities[name]) if name in priorities else None
        )
    )
    monkeypatch.setattr(dsa, "select_kernel", select)
    monkeypatch.setattr(dsa, "KernelRegistry", SimpleNamespace(get=lambda: registry))
    monkeypatch.setattr(dsa, "current_platform", lambda: SimpleNamespace(is_npu=False))
    monkeypatch.setattr(dsa, "_LOGGED_SPARSE_LEAVES", set())
    return backend, calls


def _sparse_leaf_records(caplog, level):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == backends.dsa.__name__ and record.levelno == level
    ]


def test_dsa_logs_sparse_leaves_once(monkeypatch, caplog):
    backend, calls = _sparse_leaf_backend(monkeypatch)
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        # The router's post-bind hook probes once the leaf is final.
        backend.configure_runtime(
            block_granularity=128, virtual_block_count=1, shard_count=1
        )
    (line,) = _sparse_leaf_records(caplog, logging.INFO)
    assert (
        "dsa_prefill(q_len=1, return_lse=False)=fake_dsa_prefill [SPECIALIZED]" in line
    )
    assert "dsa_decode(q_len=1, return_lse=False)=fake_dsa_decode [PORTABLE]" in line
    assert "dsa_decode(q_len=8, return_lse=False)=none" in line
    assert "dsa_index_candidates" not in line
    (warning,) = _sparse_leaf_records(caplog, logging.WARNING)
    assert "dsa_decode(q_len=8" in warning and "no verify-width core" in warning
    # One probe per facade call the forward makes, with its selection inputs.
    assert [(mode, traits["q_len"]) for mode, _, traits, _ in calls] == [
        ("dsa_prefill", 1),
        ("dsa_decode", 1),
        ("dsa_decode", 8),
    ]
    bf16 = format_signature(q=dense_tensor_format(torch.bfloat16))
    for _, signature, traits, solution in calls:
        assert signature == bf16 and solution is None
        assert traits["return_lse"] is False
        assert traits["topk"] == 2048 and traits["page_size"] == 64
        assert traits["has_kv_cache"] and not traits["has_sparse_kv_cache"]
        assert traits["slot_order"] == "selection"

    # A rebind reconfigures the leaf; its configuration is logged already.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend.configure_runtime(
            block_granularity=128, virtual_block_count=1, shard_count=1
        )
    assert not _sparse_leaf_records(caplog, logging.INFO)
    assert not _sparse_leaf_records(caplog, logging.WARNING)
    assert len(calls) == 3


@pytest.mark.parametrize("index_k_format", ["fp8_scaled", "bf16"])
def test_dsa_sparse_leaf_log_follows_fp8_kv_and_dcp(
    monkeypatch, caplog, index_k_format
):
    backend, calls = _sparse_leaf_backend(
        monkeypatch,
        data_type=torch.float8_e4m3fn,
        kvp_group=(0, 1, 2, 3),
        spec_num_tokens=1,
        index_k_format=index_k_format,
    )
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend._log_sparse_leaves()
    (line,) = _sparse_leaf_records(caplog, logging.INFO)
    assert "kvp=4" in line
    # Cores take the forward's FP8 query and return the LSE that the DCP combine merges.
    cores = [call for call in calls if call[0] != "dsa_index_candidates"]
    assert [(mode, traits["q_len"]) for mode, _, traits, _ in cores] == [
        ("dsa_prefill", 1),
        ("dsa_decode", 1),
    ]
    fp8 = format_signature(q=dense_tensor_format(torch.float8_e4m3fn))
    for _, signature, traits, _ in cores:
        assert signature == fp8 and traits["return_lse"] is True
    assert "dsa_decode(q_len=1, return_lse=True)=fake_dsa_decode [PORTABLE]" in line
    candidates = [call for call in calls if call[0] == "dsa_index_candidates"]
    if index_k_format == "fp8_scaled":
        ((_, signature, traits, solution),) = candidates
        assert signature == format_signature(
            q=dense_tensor_format(torch.bfloat16),
            weights=dense_tensor_format(torch.bfloat16),
        )
        assert traits == {
            "index_heads": 32,
            "head_dim": 128,
            "page_size": 64,
            "index_k_layout": "packed",
        }
        assert solution is None
        assert "dsa_index_candidates=fake_dsa_index_candidates [PERFORMANT+1]" in line
        assert not _sparse_leaf_records(caplog, logging.WARNING)
    else:
        # The facade refuses a bf16 plane before it selects.
        assert not candidates
        assert "dsa_index_candidates=unresolved" in line
        (warning,) = _sparse_leaf_records(caplog, logging.WARNING)
        assert "index_k_format is 'bf16'" in warning


def _plan(name, validations, error=None):
    """A batch-invariant plan of kernel ``name`` recording its validations."""

    def validate(*args):
        validations.append(args)
        if error is not None:
            raise error

    return SimpleNamespace(kernel=SimpleNamespace(name=name), validate=validate)


def test_dsa_sparse_leaf_log_names_planned_kernels(monkeypatch, caplog):
    validations = []
    backend, calls = _sparse_leaf_backend(
        monkeypatch,
        prefill_kernel_plan=_plan("planned_prefill", validations),
        decode_kernel_plan=_plan("planned_decode", validations),
    )
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend._log_sparse_leaves()
    (line,) = _sparse_leaf_records(caplog, logging.INFO)
    assert (
        "dsa_prefill(q_len=1, return_lse=False)=planned_prefill "
        "[PERFORMANT, planned]" in line
    )
    for q_len in (1, 8):
        assert (
            f"dsa_decode(q_len={q_len}, return_lse=False)=planned_decode "
            "[PERFORMANT, planned]" in line
        )
    # Facades launch a plan's kernel without reselection, after validating each launch.
    assert not calls
    bf16 = format_signature(q=dense_tensor_format(torch.bfloat16))
    assert [(args[1], args[3]["q_len"]) for args in validations] == [
        ("dsa_prefill", 1),
        ("dsa_decode", 1),
        ("dsa_decode", 8),
    ]
    for family, _, signature, traits, features, solution, override in validations:
        assert family == "attention" and signature == bf16
        assert traits["return_lse"] is False and traits["topk"] == 2048
        assert features is None and solution is None and override is None


def test_dsa_sparse_leaf_log_reports_plans_the_forward_rejects(monkeypatch, caplog):
    # Planned for a model-dtype query, but FP8 KV passes FP8: the forward would raise.
    rejected = ValueError("Input contract changed for planned kernel planned")
    backend, calls = _sparse_leaf_backend(
        monkeypatch,
        data_type=torch.float8_e4m3fn,
        prefill_kernel_plan=_plan("planned_prefill", [], rejected),
        decode_kernel_plan=_plan("planned_decode", []),
    )
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend._log_sparse_leaves()
    (line,) = _sparse_leaf_records(caplog, logging.INFO)
    assert "dsa_prefill(q_len=1, return_lse=False)=unresolved" in line
    assert (
        "dsa_decode(q_len=1, return_lse=False)=planned_decode "
        "[PERFORMANT, planned]" in line
    )
    (warning,) = _sparse_leaf_records(caplog, logging.WARNING)
    assert "dsa_prefill" in warning and "ValueError" in warning
    assert "Input contract changed" in warning
    assert not calls


def test_dsa_sparse_leaf_probes_widen_for_a_kpool_tail(monkeypatch, caplog):
    # GLM-5.3-Flash: cores take index_topk + pool_size - 1 slots, the registered 2051.
    backend, calls = _sparse_leaf_backend(
        monkeypatch,
        kpool_runtime=SimpleNamespace(pool_size=4),
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
    )
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend._log_sparse_leaves()
    assert [(mode, traits["topk"]) for mode, _, traits, _ in calls] == [
        ("dsa_prefill", 2051),
        ("dsa_decode", 2051),
        ("dsa_decode", 2051),
    ]


def test_dsa_sparse_leaf_log_prints_once_per_kvp_group(monkeypatch, caplog):
    backend, calls = _sparse_leaf_backend(
        monkeypatch, kvp_group=(0, 1, 2, 3), kvp_rank=2
    )
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend._log_sparse_leaves()
    # The group's other ranks hold the same configuration; only rank 0 logs.
    assert not _sparse_leaf_records(caplog, logging.INFO)
    assert not _sparse_leaf_records(caplog, logging.WARNING)
    assert not calls
    backend.kvp_rank = 0
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend._log_sparse_leaves()
    (line,) = _sparse_leaf_records(caplog, logging.INFO)
    assert "kvp=4" in line


def test_dsa_sparse_leaf_log_never_fails_startup(monkeypatch, caplog):
    backend, calls = _sparse_leaf_backend(monkeypatch)
    del backend.slot_order
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend._log_sparse_leaves()
    assert not _sparse_leaf_records(caplog, logging.INFO)
    (warning,) = _sparse_leaf_records(caplog, logging.WARNING)
    assert "AttributeError" in warning
    assert not calls

    # The Ascend leaf launches its own operators, so NPU probes nothing.
    backend.slot_order = "selection"
    monkeypatch.setattr(
        backends.dsa, "current_platform", lambda: SimpleNamespace(is_npu=True)
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        backend._log_sparse_leaves()
    assert not _sparse_leaf_records(caplog, logging.INFO)
    assert not _sparse_leaf_records(caplog, logging.WARNING)
    assert not calls


class _SelectionProbe(Exception):
    """Raised by the patched facade selection once it recorded its inputs."""


@pytest.mark.parametrize("kvp", [1, 4])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_dsa_sparse_leaf_probes_match_the_facade_selections(monkeypatch, kv_dtype, kvp):
    # Record the forward's facade selections; the probes must ask for the same leaves.
    from tokenspeed_kernel.ops.attention import dsa as dsa_ops

    from tokenspeed.runtime.layers.attention.dcp.indexer import select_dsa_topk

    dsa = backends.dsa
    backend, _ = _sparse_leaf_backend(
        monkeypatch, data_type=kv_dtype, kvp_group=tuple(range(kvp))
    )
    for name, value in dict(
        is_draft=False,
        max_context_len=4096,
        num_local_heads=2,
        num_attention_heads=2 * kvp,
        kvp_block_granularity=128,
        kvp_virtual_block_count=17,
        query_shard_metadata=None,
        step_counter=None,
        _dense_backend=SimpleNamespace(
            forward_decode_metadata=SimpleNamespace(
                num_extends=0,
                seq_lens_k=torch.tensor([600], dtype=torch.int32),
                max_seq_len_k=600,
                _dsa_kv_seq_lens=None,
            )
        ),
    ).items():
        setattr(backend, name, value)
    recorded = []

    def select(family, mode, signature, *, traits, solution, **kwargs):
        assert family == "attention"
        recorded.append((mode, signature, traits, solution))
        raise _SelectionProbe

    def gather(q, group):
        return q.new_empty((q.shape[0], q.shape[1] * len(group), q.shape[2]))

    monkeypatch.setattr(dsa_ops, "select_kernel", select)
    monkeypatch.setattr(dsa, "gather_query_heads", gather)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        head_dim=576,
        v_head_dim=512,
        scaling=0.1,
        logit_cap=0.0,
    )
    pool = SimpleNamespace(
        quant_method=None,
        get_key_buffer=lambda layer_id: torch.empty(17 * 128, 576, dtype=kv_dtype),
    )

    def selection(rows):
        slots = torch.full((rows, 2048), -1, dtype=torch.int32)
        slots[:, :4] = torch.tensor([128, 256, 384, 512])
        return slots

    with pytest.raises(_SelectionProbe):
        backend.forward_sparse_prefill(
            q=torch.zeros(2, 2, 576, dtype=torch.bfloat16),
            layer=layer,
            token_to_kv_pool=pool,
            kv_seq_lens=torch.tensor([300, 301], dtype=torch.int32),
            topk_slots=selection(2),
            topk_lens=torch.tensor([4, 3], dtype=torch.int32),
            max_seq_len=301,
        )
    for q_len in (1, backend.spec_num_tokens):
        with pytest.raises(_SelectionProbe):
            backend.forward_sparse_decode(
                q=torch.zeros(q_len, 2, 576, dtype=torch.bfloat16),
                layer=layer,
                token_to_kv_pool=pool,
                bs=1,
                topk_indices=selection(q_len),
                topk_lens=None,
            )
    if kvp > 1:
        # The decode and prefill indexers' call, with model-dtype query and weights.
        rows = backend.spec_num_tokens
        with pytest.raises(_SelectionProbe):
            select_dsa_topk(
                torch.zeros(rows, 32, 128, dtype=torch.bfloat16),
                torch.zeros(rows, 32, dtype=torch.bfloat16),
                torch.zeros(17 * 128, 132, dtype=torch.uint8),
                torch.tensor([[2, 3, 4, 5]], dtype=torch.int32),
                torch.zeros(rows, dtype=torch.int32),
                torch.full((rows,), 200, dtype=torch.int32),
                placement=backend.cache_placement(layer),
                page_size=64,
                topk=2048,
                softmax_scale=0.1,
                initial_tokens=0,
                local_tokens=0,
                max_logits_bytes=None,
            )
    probes = [
        (mode, signature, traits, solution)
        for _, mode, signature, traits, solution, _ in backend._sparse_leaf_probes()
    ]
    assert recorded == probes


def test_dsa_sparse_leaf_log_sees_a_router_factory_solution_pin(monkeypatch, caplog):
    # A leaf factory's solution pin precedes configure_runtime, so the probes honor it.
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )

    built = []

    def leaf_factory(group_id, block_granularity):
        leaf, calls = _sparse_leaf_backend(monkeypatch)
        leaf.set_cache_pool = lambda pool: None
        leaf.kernel_solution = "triton"
        built.append(calls)
        return leaf

    spec = SimpleNamespace(
        group_id="dsa",
        attn_kind="full_history",
        block_granularity=64,
        rows_per_page=64,
        entry_stride_tokens=1,
        sliding_window_tokens=None,
        shard_count=1,
    )
    pool = SimpleNamespace(
        paged_group_ids=("dsa",),
        arena=SimpleNamespace(
            offload=None,
            cache_group_specs=(spec,),
            runtime_contract=SimpleNamespace(virtual_block_counts={"dsa": 1}),
        ),
    )
    router = CacheGroupRouter(
        leaf_factory,
        is_draft=False,
        spec_num_tokens=8,
        device="cpu",
        consumed_group_ids=None,
    )
    router.set_cache_pool(pool)
    (calls,) = built
    assert not calls
    with caplog.at_level(logging.INFO, logger=backends.dsa.__name__):
        router.configure_runtime()
    assert [(mode, solution) for mode, _, _, solution in calls] == [
        ("dsa_prefill", "triton"),
        ("dsa_decode", "triton"),
        ("dsa_decode", "triton"),
    ]
    (line,) = _sparse_leaf_records(caplog, logging.INFO)
    assert "solution=triton" in line


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not current_platform().is_hopper_plus,
    reason="the K3.1 DSA leaves register on NVIDIA Hopper+ GPUs",
)
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_dsa_sparse_leaf_log_names_the_real_selection(monkeypatch, caplog, kv_dtype):
    # K3.1 at KVP 8 on the real registry: each probe resolves to select_kernel's pick.
    from tokenspeed_kernel.selection import select_kernel

    dsa = backends.dsa
    backend = object.__new__(dsa.DSABackend)
    values = dict(_K31_SPARSE_LEAF, data_type=kv_dtype, kvp_group=tuple(range(8)))
    for name, value in values.items():
        setattr(backend, name, value)
    monkeypatch.setattr(dsa, "_LOGGED_SPARSE_LEAVES", set())
    with caplog.at_level(logging.INFO, logger=dsa.__name__):
        backend._log_sparse_leaves()
    (line,) = _sparse_leaf_records(caplog, logging.INFO)
    assert not _sparse_leaf_records(caplog, logging.WARNING)
    assert "=none" not in line and "=unresolved" not in line
    probes = backend._sparse_leaf_probes()
    assert [mode for _, mode, *_ in probes] == [
        "dsa_prefill",
        "dsa_decode",
        "dsa_decode",
        "dsa_index_candidates",
    ]
    for label, mode, signature, traits, solution, _ in probes:
        kernel = select_kernel(
            "attention", mode, signature, traits=traits, solution=solution
        )
        assert f"{label}={kernel.name} [" in line


@pytest.mark.parametrize(
    "case",
    [
        "triton",
        "trtllm_core",
        "other_core",
        "no_records_leaf",
        "planned",
        "no_kvp",
        "bf16_cache",
    ],
)
def test_dcp_records_core_follows_the_ordinary_prefill_selection(monkeypatch, case):
    """The qlayout lever's core stands in for the Triton dsa_prefill only, and
    its padded query for the trtllm-gen one only.

    configure_runtime, once the solution pin has landed, asks the forward's
    own LSE prefill selection (an FP8 query on the FP8 cache) and the records
    op for the same inputs. A planned leaf keeps its core and a leaf without
    KVP runs no head-sharded prefill, so neither asks. Nor does a BF16 cache,
    although its selection here is the Triton core: its query keeps the
    ordinary gather, which takes the multicast kernels where they reach.
    """
    dsa = backends.dsa
    backend, _ = _sparse_leaf_backend(
        monkeypatch,
        data_type=torch.bfloat16 if case == "bf16_cache" else torch.float8_e4m3fn,
        kvp_group=(0,) if case == "no_kvp" else (0, 1, 2, 3),
        kernel_solution="triton",
        prefill_kernel_plan=_plan("planned_prefill", []) if case == "planned" else None,
    )
    calls = []

    def select(family, mode, signature, *, traits, solution):
        assert family == "attention"
        calls.append((mode, signature, traits, solution))
        if mode == "dsa_prefill" and traits["return_lse"]:
            core = {
                "other_core": "fake_dsa_prefill",
                "trtllm_core": "flashinfer_trtllm_dsa_prefill",
            }.get(case, "triton_dsa_prefill")
            return SimpleNamespace(name=core)
        if mode == "dsa_prefill_dcp_records" and case == "no_records_leaf":
            raise NoKernelFoundError("no records leaf")
        return SimpleNamespace(name=f"fake_{mode}")

    monkeypatch.setattr(dsa, "select_kernel", select)
    backend.configure_runtime(
        block_granularity=128,
        virtual_block_count=17,
        shard_count=len(backend.kvp_group),
    )
    assert backend._dcp_records_core is (case == "triton")
    assert backend._dcp_padded_query is (case == "trtllm_core")
    records = [call for call in calls if call[0] == "dsa_prefill_dcp_records"]
    if case in ("planned", "no_kvp", "bf16_cache"):
        assert not records
        return
    # The probe precedes the startup log and asks the records op what it asked prefill.
    prefill, records_call = calls[:2]
    assert records == [records_call]
    assert prefill[0] == "dsa_prefill" and prefill[1:] == records_call[1:]
    _, signature, traits, solution = prefill
    assert signature == format_signature(q=dense_tensor_format(torch.float8_e4m3fn))
    assert traits["return_lse"] is True and traits["topk"] == 2048
    assert solution == "triton"


@pytest.mark.parametrize(
    "case",
    [
        "gfx950",
        "verify_other_core",
        "other_core",
        "no_records_leaf",
        "planned",
        "no_kvp",
        "bf16_cache",
    ],
)
def test_dcp_decode_records_core_follows_every_decode_selection(monkeypatch, case):
    """The decode records core stands in for gfx950's dense decode core only,
    and only where both decode selections pick it: one row per request and
    the verify rows.

    configure_runtime asks each LSE decode selection (an FP8 query on the FP8
    cache) and the records op for the same inputs. A planned decode keeps its
    core, a leaf without KVP runs no head-sharded decode and a BF16 cache's
    query keeps the ordinary gather, so none of them asks.
    """
    dsa = backends.dsa
    backend, _ = _sparse_leaf_backend(
        monkeypatch,
        data_type=torch.bfloat16 if case == "bf16_cache" else torch.float8_e4m3fn,
        kvp_group=(0,) if case == "no_kvp" else (0, 1, 2, 3),
        decode_kernel_plan=_plan("planned_decode", []) if case == "planned" else None,
    )
    calls = []

    def select(family, mode, signature, *, traits, solution):
        assert family == "attention"
        calls.append((mode, signature, traits, solution))
        if mode == "dsa_decode":
            other = case == "other_core" or (
                case == "verify_other_core" and traits["q_len"] > 1
            )
            return SimpleNamespace(
                name="triton_dsa_decode" if other else "gluon_dsa_decode_gfx950"
            )
        if mode == "dsa_decode_dcp_records" and case == "no_records_leaf":
            raise NoKernelFoundError("no records leaf")
        return SimpleNamespace(name=f"fake_{mode}")

    monkeypatch.setattr(dsa, "select_kernel", select)
    backend.configure_runtime(
        block_granularity=128,
        virtual_block_count=17,
        shard_count=len(backend.kvp_group),
    )
    assert backend._dcp_decode_records_core is (case == "gfx950")
    records = [call for call in calls if call[0] == "dsa_decode_dcp_records"]
    if case in ("planned", "no_kvp", "bf16_cache"):
        assert not records
        return
    if case != "gfx950":
        return
    # The probe runs before the startup log, and asks the records op what it
    # asked each decode selection.
    decodes = [call for call in calls if call[0].startswith("dsa_decode")][:4]
    assert [mode for mode, *_ in decodes] == [
        "dsa_decode",
        "dsa_decode_dcp_records",
    ] * 2
    assert records[:2] == decodes[1::2]
    for core, records_call in zip(decodes[::2], decodes[1::2]):
        assert core[1:] == records_call[1:]
    assert [traits["q_len"] for _, _, traits, _ in decodes[::2]] == [1, 8]
    _, signature, traits, solution = decodes[0]
    assert signature == format_signature(q=dense_tensor_format(torch.float8_e4m3fn))
    assert traits["return_lse"] is True and traits["topk"] == 2048
    assert solution is None


@pytest.mark.parametrize("kvp", [1, 8])
def test_trtllm_sparse_buffers_are_reserved_before_any_forward(monkeypatch, kvp):
    """configure_runtime reserves the trtllm-gen cores' persistent buffers for
    each sparse call that selects one, at the heads that call carries (the
    KVP group's 96 gathered heads, as every head, or one rank's 12) and with
    its LSE; the call that selects another core, and a CPU leaf, reserve
    nothing."""
    dsa = backends.dsa
    backend, _ = _sparse_leaf_backend(
        monkeypatch,
        device="cuda",
        data_type=torch.float8_e4m3fn,
        kvp_group=tuple(range(kvp)),
        num_local_heads=12,
        num_attention_heads=96,
    )

    def select(family, mode, signature, *, traits, solution):
        if mode == "dsa_decode" and traits["q_len"] > 1:
            return SimpleNamespace(name="triton_dsa_decode")
        return SimpleNamespace(name=f"flashinfer_trtllm_{mode}")

    reserved = []
    monkeypatch.setattr(dsa, "select_kernel", select)
    monkeypatch.setattr(
        dsa,
        "reserve_flashinfer_trtllm_dsa_buffers",
        lambda device, *, num_heads, return_lse: reserved.append(
            (device, num_heads, return_lse)
        ),
    )
    backend.configure_runtime(
        block_granularity=128, virtual_block_count=17, shard_count=kvp
    )
    heads = 96 if kvp > 1 else 12
    # dsa_prefill and the one-row dsa_decode; the verify-width decode runs Triton.
    assert reserved == [("cuda", heads, kvp > 1)] * 2

    reserved.clear()
    backend.device = "cpu"
    backend.configure_runtime(
        block_granularity=128, virtual_block_count=17, shard_count=kvp
    )
    assert not reserved


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not current_platform().is_hopper_plus,
    reason="the K3.1 DSA leaves register on NVIDIA Hopper+ GPUs",
)
@pytest.mark.parametrize("solution", [None, "triton", "flashmla"])
def test_dcp_records_core_on_the_real_k31_selection(monkeypatch, solution):
    # On Blackwell trtllm-gen outranks the Triton core; FlashMLA (BF16 only) gets none.
    dsa = backends.dsa
    backend = object.__new__(dsa.DSABackend)
    values = dict(
        _K31_SPARSE_LEAF,
        data_type=torch.float8_e4m3fn,
        kvp_group=tuple(range(8)),
        kernel_solution=solution,
    )
    for name, value in values.items():
        setattr(backend, name, value)
    monkeypatch.setattr(dsa, "_LOGGED_SPARSE_LEAVES", set())
    backend.configure_runtime(
        block_granularity=128, virtual_block_count=17, shard_count=8
    )
    trtllm_outranks = current_platform().is_blackwell
    assert backend._dcp_records_core is (
        solution == "triton" or (solution is None and not trtllm_outranks)
    )
    # Where trtllm-gen runs the 96 gathered heads, as 128, qlayout pads them.
    assert backend._dcp_padded_query is (solution is None and trtllm_outranks)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_amd
    or current_platform().arch_version != ArchVersion(9, 5),
    reason="the gfx950 DSA decode core registers on AMD gfx950 GPUs",
)
@pytest.mark.parametrize("solution", [None, "gluon", "triton"])
def test_dcp_decode_records_core_on_the_real_k31_selection(monkeypatch, solution):
    # K3.1's FP8 cache at KVP8 with its 4 verify rows (the 3.1 drafter's three
    # tokens) against the real registry: gfx950's dense core decodes both
    # widths, so the records core stands in unless another solution is pinned.
    dsa = backends.dsa
    backend = object.__new__(dsa.DSABackend)
    values = dict(
        _K31_SPARSE_LEAF,
        data_type=torch.float8_e4m3fn,
        kvp_group=tuple(range(8)),
        kernel_solution=solution,
        spec_num_tokens=4,
    )
    for name, value in values.items():
        setattr(backend, name, value)
    monkeypatch.setattr(dsa, "_LOGGED_SPARSE_LEAVES", set())
    backend.configure_runtime(
        block_granularity=128, virtual_block_count=17, shard_count=8
    )
    assert backend._dcp_decode_records_core is (solution != "triton")
