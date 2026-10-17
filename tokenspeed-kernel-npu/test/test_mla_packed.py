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

"""Packed reader dispatch contracts and changed-input NPUGraph regression."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenspeed_kernel_npu.ops import mla_packed as adapter


@pytest.fixture(autouse=True)
def reset_loader():
    loader = adapter._packed_op
    prefill_loader = adapter._packed_prefill_op
    loader.cache_clear()
    prefill_loader.cache_clear()
    yield
    loader.cache_clear()
    prefill_loader.cache_clear()


def geometry(batch, heads, page_size):
    device = SimpleNamespace(type="npu")

    def tensor(shape, dtype):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=device,
            is_contiguous=lambda: True,
        )

    return (
        tensor((batch, 1, heads, 576), torch.bfloat16),
        tensor((64, page_size, 1, 576), torch.bfloat16),
        tensor((batch, 16), torch.int32),
    )


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("batch,page_size", [(1, 64), (32, 64), (2, 128)])
@pytest.mark.parametrize("legacy_flag", [None, "0", "1"])
def test_supported_geometry_ignores_legacy_flag(
    monkeypatch, heads, batch, page_size, legacy_flag
):
    op = Mock()
    monkeypatch.setattr(adapter, "_packed_op", lambda: op)
    if legacy_flag is None:
        monkeypatch.delenv("TOKENSPEED_NPU_PACKED_FIA", raising=False)
    else:
        monkeypatch.setenv("TOKENSPEED_NPU_PACKED_FIA", legacy_flag)
    assert (
        adapter.packed_mla_decode_op(*geometry(batch, heads, page_size), 1048576) is op
    )


@pytest.mark.parametrize("case", ["dtype", "page", "stride", "table", "bound", "heads"])
def test_native_fallback(monkeypatch, case):
    q, kv, table = geometry(2, 32, 64)
    bound = 81920
    loader = Mock()
    monkeypatch.setattr(adapter, "_packed_op", loader)
    if case == "dtype":
        q.dtype = torch.float16
    elif case == "page":
        kv.shape = (64, 16, 1, 576)
    elif case == "stride":
        kv.is_contiguous = lambda: False
    elif case == "table":
        table.dtype = torch.int64
    elif case == "bound":
        bound = 1048577
    else:
        q.shape = (2, 1, 65, 576)
    assert adapter.packed_mla_decode_op(q, kv, table, bound) is None
    loader.assert_not_called()


def test_missing_package(monkeypatch):
    loader = Mock(side_effect=ModuleNotFoundError("missing", name="flash_ops"))
    monkeypatch.setattr(adapter.importlib, "import_module", loader)
    assert adapter._packed_op() is None


def test_broken_dependency_not_hidden(monkeypatch):
    loader = Mock(side_effect=ModuleNotFoundError("missing", name="dependency"))
    monkeypatch.setattr(adapter.importlib, "import_module", loader)
    with pytest.raises(ModuleNotFoundError):
        adapter._packed_op()


@pytest.mark.parametrize(
    "capability", ["operator", "out", "host_lengths", "handler_module", "handler_api"]
)
def test_missing_capability_falls_back(monkeypatch, capability):
    op = Mock(__name__="npu_mla_fia_packed.out")
    op._schema = SimpleNamespace(
        arguments=[None] * 4
        + [
            SimpleNamespace(
                type="Tensor" if capability == "host_lengths" else "List[int]"
            )
        ]
    )
    packet = None if capability == "operator" else SimpleNamespace()
    if packet is not None and capability != "out":
        packet.out = op
    monkeypatch.setattr(
        adapter.torch,
        "ops",
        SimpleNamespace(custom=SimpleNamespace(npu_mla_fia_packed=packet)),
    )

    def load(name):
        if "npugraph_handler" in name and capability == "handler_module":
            raise ModuleNotFoundError("missing", name=name)
        return SimpleNamespace()

    monkeypatch.setattr(adapter.importlib, "import_module", load)
    assert adapter._packed_op() is None


def test_registered_handler_updates_only_lengths(monkeypatch):
    class Base:
        @classmethod
        def record_wrap_kwarg(cls, key, value, tensor_names):
            return (key, value, tensor_names)

    registry = {}
    handlers = SimpleNamespace(
        NpuGraphOpHandler=Base,
        register_npu_graph_handler=lambda names: lambda cls: registry.update(
            {name: cls for name in names}
        ),
    )
    op = Mock(__name__="npu_mla_fia_packed.out")
    op._schema = SimpleNamespace(
        arguments=[None] * 4 + [SimpleNamespace(type="List[int]")]
    )
    monkeypatch.setattr(
        adapter.torch,
        "ops",
        SimpleNamespace(
            custom=SimpleNamespace(npu_mla_fia_packed=SimpleNamespace(out=op))
        ),
    )
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        lambda name: handlers if "npugraph_handler" in name else object(),
    )
    assert adapter._packed_op() is op
    record = SimpleNamespace(args=[object(), object(), object(), object(), [1, 1]])
    before = record.args[:4]
    handler = registry[op.__name__]
    handler.update_args(record, {"actual_seq_lengths_kv": [65, 127]})
    assert record.args[:4] == before
    assert record.args[4] == [65, 127]
    assert "lse" in handler.record_wrap_kwarg("lse", object(), [])[2]


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires NPU"
)
@pytest.mark.parametrize(
    "batch,heads,page_size", [(1, 32, 64), (2, 4, 64), (2, 8, 128), (32, 32, 64)]
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_packed_graph_live_updates(monkeypatch, batch, heads, page_size, return_lse):
    import torch_npu
    from tokenspeed_kernel_npu.ops.mla import mla_decode_with_kvcache

    assert adapter._packed_op() is not None, "packed op and graph handler are required"
    torch.manual_seed(1729)
    device = "npu"
    q = torch.randn(batch, 1, heads, 576, dtype=torch.bfloat16, device=device) * 0.1
    # Nonzero storage offset, as when a layer is a view into the arena.
    backing = (
        torch.randn(batch * 3 + 3, page_size, 1, 576, dtype=q.dtype, device=device)
        * 0.1
    )
    cache = backing[3:]
    table = torch.arange(batch * 3, dtype=torch.int32, device=device).view(batch, 3)
    lengths = [page_size - 1] * batch

    def invoke(live):
        return mla_decode_with_kvcache(
            q,
            cache,
            table,
            live,
            3 * page_size,
            128,
            512,
            64,
            192**-0.5,
            0.0,
            return_lse,
            None,
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke(lengths)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        captured = invoke(lengths)
    records = graph.graph_dispatch_mode.graph_dispatch_records
    assert len(records) == 1
    assert "npu_mla_fia_packed" in records[0].op_cache_entry.__name__
    address = (captured[0] if return_lse else captured).data_ptr()
    for length in (page_size + 1, 2 * page_size + 3, 1):
        lengths = [max(0, length - i % 2) for i in range(batch)]
        q.add_(0.01)
        cache.mul_(0.98)
        table.copy_(table.roll(1, dims=1))
        graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
        graph.replay()
        actual = captured if return_lse else (captured,)
        torch.npu.synchronize()
        saved = tuple(t.clone() for t in actual)
        with monkeypatch.context() as native:
            native.setattr(adapter, "_packed_op", lambda: None)
            expected = invoke(lengths)
        expected = expected if return_lse else (expected,)
        for a, b in zip(saved, expected):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.01)
        assert actual[0].data_ptr() == address


@pytest.fixture
def internal_format():
    import torch_npu

    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    yield
    torch.npu.config.allow_internal_format = previous == b"enable"


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires NPU"
)
@pytest.mark.parametrize("batch,hidden,heads", [(32, 3072, 32), (1, 4096, 64)])
@torch.inference_mode()
def test_fused_prolog_to_packed_graph(
    monkeypatch, internal_format, batch, hidden, heads
):
    import torch_npu
    from tokenspeed_kernel_npu.ops.mla import mla_decode_with_kvcache
    from tokenspeed_kernel_npu.ops.mla_prolog import mla_prolog, mla_prolog_available

    assert adapter._packed_op() is not None and mla_prolog_available()
    torch.manual_seed(1730)

    def random(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="npu") * 0.01

    x = random((batch, hidden))
    wdq = torch_npu.npu_format_cast(random((hidden, 1536)), 29)
    wuq = torch_npu.npu_format_cast(random((1536, heads * 192)), 29)
    wuk = random((heads, 128, 512))
    wdkv = torch_npu.npu_format_cast(random((hidden, 576)), 29)
    gcq = torch.ones(1536, dtype=x.dtype, device=x.device)
    gkv = torch.ones(512, dtype=x.dtype, device=x.device)
    cache = torch.zeros(batch, 64, 1, 576, dtype=x.dtype, device=x.device)
    table = torch.arange(batch, dtype=torch.int32, device=x.device).view(batch, 1)
    slots = torch.arange(batch, dtype=torch.int32, device=x.device) * 64

    def invoke(lengths):
        projected = mla_prolog(
            x,
            wdq,
            wuq,
            wuk,
            wdkv,
            gcq,
            gkv,
            cache,
            slots,
            rmsnorm_epsilon_cq=1e-5,
            rmsnorm_epsilon_ckv=1e-5,
        )
        assert projected is not None, "the fused writer must not fall back"
        query = torch.cat(projected, dim=-1).view(batch, 1, heads, 576)
        return mla_decode_with_kvcache(
            query,
            cache,
            table,
            lengths,
            64,
            128,
            512,
            64,
            192**-0.5,
            0.0,
            False,
            None,
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke([1] * batch)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        captured = invoke([1] * batch)
    records = graph.graph_dispatch_mode.graph_dispatch_records
    assert sum("npu_mla_fia_packed" in r.op_cache_entry.__name__ for r in records) == 1
    for step in range(1, 4):
        x.add_(0.001)
        slots.add_(1)
        lengths = [step + 1] * batch
        graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
        graph.replay()
        torch.npu.synchronize()
        saved = captured.clone()
        with monkeypatch.context() as native:
            native.setattr(adapter, "_packed_op", lambda: None)
            reference = invoke(lengths)
        torch.testing.assert_close(saved, reference, rtol=0.01, atol=0.001)


@pytest.mark.parametrize("queries,page_size", [(1, 64), (7, 128), (4096, 64)])
def test_prefill_capability_geometry(monkeypatch, queries, page_size):
    q, cache, table = geometry(2, 32, page_size)
    q.shape, q.ndim = (queries * 2, 32, 576), 3
    op = Mock()
    monkeypatch.setattr(adapter, "_packed_prefill_op", lambda: op)
    assert adapter.packed_mla_prefill_op(q, cache, table, queries, 65536) is op


@pytest.mark.parametrize(
    "case", ["explicit", "page", "stride", "query_bound", "kv_bound", "dtype"]
)
def test_prefill_unsupported_geometry_falls_back(monkeypatch, case):
    q, cache, table = geometry(2, 32, 64)
    q.shape, q.ndim = (8192, 32, 576), 3
    queries, context = 4096, 65536
    if case == "explicit":
        q.shape = (8192, 32, 192)
    elif case == "page":
        cache.shape = (64, 16, 1, 576)
    elif case == "stride":
        cache.is_contiguous = lambda: False
    elif case == "query_bound":
        queries = 4097
    elif case == "kv_bound":
        context = 1048577
    else:
        q.dtype = torch.float16
    loader = Mock()
    monkeypatch.setattr(adapter, "_packed_prefill_op", loader)
    assert adapter.packed_mla_prefill_op(q, cache, table, queries, context) is None
    loader.assert_not_called()


def test_prefill_missing_package(monkeypatch):
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        Mock(side_effect=ModuleNotFoundError("missing", name="flash_ops")),
    )
    assert adapter._packed_prefill_op() is None


def test_prefill_broken_dependency_propagates(monkeypatch):
    monkeypatch.setattr(
        adapter.importlib,
        "import_module",
        Mock(side_effect=ModuleNotFoundError("missing", name="dependency")),
    )
    with pytest.raises(ModuleNotFoundError):
        adapter._packed_prefill_op()


@pytest.mark.parametrize("capability", ["operator", "out", "lengths", "supported"])
def test_prefill_package_capability(monkeypatch, capability):
    op = Mock()
    op._schema = SimpleNamespace(
        arguments=[SimpleNamespace(type="Tensor") for _ in range(12)]
    )
    for i in (4, 5):
        op._schema.arguments[i].type = (
            "Tensor" if capability == "lengths" else "List[int]"
        )
    packet = None if capability == "operator" else SimpleNamespace()
    if packet is not None and capability != "out":
        packet.out = op
    monkeypatch.setattr(
        adapter.torch,
        "ops",
        SimpleNamespace(custom=SimpleNamespace(npu_mla_fia_packed_prefill=packet)),
    )
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: object())
    assert adapter._packed_prefill_op() is (op if capability == "supported" else None)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires NPU"
)
@torch.inference_mode()
def test_packed_graph_service_geometry(monkeypatch):
    from tokenspeed_kernel_npu.ops.mla import mla_decode_with_kvcache

    assert adapter._packed_op() is not None
    torch.manual_seed(1730)
    batch, heads, page, maximum = 32, 32, 64, 81920
    pages_per_request = maximum // page
    q = torch.randn(batch, 1, heads, 576, dtype=torch.bfloat16, device="npu") * 0.1
    backing = torch.empty(
        batch * pages_per_request + 4, page, 1, 576, dtype=q.dtype, device=q.device
    )
    backing.normal_(0, 0.1)
    cache = backing[2:-2]
    table = torch.randperm(
        batch * pages_per_request, dtype=torch.int32, device=q.device
    ).view(batch, pages_per_request)
    lengths = [65536 - b for b in range(batch)]

    def invoke(live):
        return mla_decode_with_kvcache(
            q, cache, table, live, maximum, 128, 512, 64, 192**-0.5, 0.0, True, None
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke(lengths)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        output, lse = invoke(lengths)
    records = graph.graph_dispatch_mode.graph_dispatch_records
    assert len(records) == 1
    assert "npu_mla_fia_packed" in records[0].op_cache_entry.__name__
    addresses = output.data_ptr(), lse.data_ptr()
    guards = backing[[0, 1, -2, -1]].clone()
    for base in (65537, 65473, 65599):
        lengths = [base - b for b in range(batch)]
        q.add_(0.01)
        cache[:32].add_(0.01)
        table.copy_(table.roll(1, 1))
        graph.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}])
        graph.replay()
        torch.npu.synchronize()
        saved = output.clone(), lse.clone()
        with monkeypatch.context() as native:
            native.setattr(adapter, "_packed_op", lambda: None)
            expected = invoke(lengths)
        for a, b in zip(saved, expected):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.0015)
        assert (output.data_ptr(), lse.data_ptr()) == addresses
        assert torch.equal(backing[[0, 1, -2, -1]], guards)
