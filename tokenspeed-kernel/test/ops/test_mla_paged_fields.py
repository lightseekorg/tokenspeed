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

"""Paged-field kernel regressions; standalone isolation is explicitly test-only."""

from pathlib import Path

import pytest
import torch


def _field(storage, pages, page_size, width, page_stride, row_stride, offset):
    return storage.view(torch.bfloat16).as_strided(
        (pages, page_size, 1, width),
        (page_stride // 2, row_stride, width, 1),
        offset // 2,
    )


@pytest.mark.parametrize("n", [4, 513])
@pytest.mark.parametrize(
    "page_size,width,page_stride,row_padding",
    [
        (64, 576, 1068800, 0),
        (32, 1088, 1068800, 0),
        (32, 1088, 213760, 0),
        (32, 1088, 213760, 16),
    ],
)
@pytest.mark.parametrize("layout", ["paged", "flat2", "flat3"])
def test_mla_scatter_gather_strided(
    n, page_size, width, page_stride, row_padding, layout
):
    from tokenspeed_kernel.ops.kvcache.triton import (
        get_mla_kv_buffer_triton,
        set_mla_kv_buffer_triton,
    )

    torch.manual_seed(7)
    pages, offset = 20, 256
    row_stride = width + row_padding
    if layout != "paged":
        page_stride = page_size * row_stride * 2
    storage = torch.full(
        (pages * page_stride + offset,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    paged = _field(storage, pages, page_size, width, page_stride, row_stride, offset)
    paged.fill_(float("nan"))
    cache = paged if layout == "paged" else paged.view(-1, 1, width)
    if layout == "flat2":
        cache = cache.squeeze(1)
    expected_storage = storage.clone()
    expected = _field(
        expected_storage, pages, page_size, width, page_stride, row_stride, offset
    )
    loc = torch.randperm((pages - 1) * page_size, device="cuda")[:n] + page_size
    # Padding writes may use slot 0, but must never reach a live page.
    loc[0] = 0
    latent = torch.randn((n, 1, width - 64), dtype=torch.bfloat16, device="cuda")
    rope = torch.randn((n, 1, 64), dtype=torch.bfloat16, device="cuda")
    out_latent, out_rope = torch.empty_like(latent), torch.empty_like(rope)

    def run():
        set_mla_kv_buffer_triton(
            cache, loc, latent, rope, enable_pdl=False, sanitize=False
        )
        get_mla_kv_buffer_triton(cache, loc, out_latent, out_rope, enable_pdl=False)

    def verify():
        expected[loc // page_size, loc % page_size] = torch.cat((latent, rope), -1)
        torch.testing.assert_close(storage, expected_storage, atol=0, rtol=0)
        torch.testing.assert_close(out_latent, latent, atol=0, rtol=0)
        torch.testing.assert_close(out_rope, rope, atol=0, rtol=0)
        assert torch.isnan(paged[0, 1:]).all()

    run()
    verify()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    latent.mul_(0.5)
    rope.neg_()
    loc.copy_(loc.flip(0))
    graph.replay()
    verify()


@pytest.mark.parametrize("layout", ["flat", "paged", "planar"])
@pytest.mark.parametrize("masked", [False, True])
def test_index_scatter_page_stride(layout, masked):
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter

    pages, page_size, offset = 8, 64, 256
    page_bytes = page_size * 132
    stride = page_bytes if layout == "flat" else 1068800
    storage = torch.full(
        (pages * stride + offset,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    planar = storage.as_strided((pages, page_bytes), (stride, 1), offset)
    if layout == "flat":
        cache = planar.view(-1, 132)
    elif layout == "paged":
        cache = planar.view(pages, page_size, 132)
    else:
        cache = planar
    expected_storage = storage.clone()
    expected = expected_storage.as_strided(planar.shape, planar.stride(), offset)
    loc = torch.tensor([0, 64, 95, 127, 128, 511], device="cuda", dtype=torch.int64)
    write_mask = (
        torch.tensor([False, True, False, True, True, False], device="cuda")
        if masked
        else None
    )
    keys = (
        torch.arange(loc.numel() * 128, device="cuda")
        .remainder(17)
        .reshape(-1, 128)
        .to(torch.float8_e4m3fn)
    )
    scales = (
        torch.arange(1, loc.numel() + 1, device="cuda", dtype=torch.float32)[:, None]
        / 4
    )

    def run():
        index_k_block_split_scatter(
            cache,
            keys,
            scales,
            loc,
            page_size=64,
            head_dim=128,
            group_size=128,
            write_mask=write_mask,
        )

    def verify():
        for i, slot in enumerate(loc.tolist()):
            if write_mask is not None and not write_mask[i].item():
                continue
            page, row = divmod(slot, page_size)
            expected[page, row * 128 : (row + 1) * 128] = keys[i].view(torch.uint8)
            expected[page, 8192 + row * 4 : 8192 + (row + 1) * 4] = scales[i].view(
                torch.uint8
            )
        torch.testing.assert_close(storage, expected_storage, atol=0, rtol=0)

    run()
    verify()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    scales.mul_(2)
    loc.copy_(loc.flip(0))
    graph.replay()
    verify()


@pytest.mark.parametrize("layout", ["flat2", "flat3", "nhd", "hnd"])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_dsa_reads_original_storage(layout, mode, monkeypatch):
    from tokenspeed_kernel.ops.attention.dsa import triton as dsa
    from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton

    torch.manual_seed(11)
    pages, page_size, width, offset = 12, 64, 576, 256
    stride = 1068800 if layout in ("nhd", "hnd") else page_size * (width + 16) * 2
    storage = torch.full(
        (pages * stride + offset,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    paged = _field(storage, pages, page_size, width, stride, width + 16, offset)
    paged.fill_(float("nan"))
    cache = paged
    if layout == "hnd":
        cache = paged.transpose(1, 2)
    elif layout.startswith("flat"):
        cache = paged.view(-1, 1, width)
        if layout == "flat2":
            cache = cache.squeeze(1)
    loc = torch.randperm((pages - 1) * page_size, device="cuda")[:513] + page_size
    rows = torch.randn((513, 1, width), dtype=torch.bfloat16, device="cuda")
    set_mla_kv_buffer_triton(
        cache, loc, rows[..., :512], rows[..., 512:], enable_pdl=False, sanitize=False
    )
    slots = loc[:64].to(torch.int32).repeat(3, 1)
    slots[1, 7] = -1
    slots[1, 33:] = 0  # Poisoned null slots outside the live length must not load.
    slots[2] = 0
    lens = torch.tensor([64, 33, 0], dtype=torch.int32, device="cuda")
    q = torch.randn((3, 4, width), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((3, 4, 512), dtype=torch.bfloat16, device="cuda")
    normalized = dsa._flatten_dense_kv_cache(cache)
    assert normalized.untyped_storage().data_ptr() == storage.data_ptr()

    # Guard the regression directly: cache materialization is forbidden even
    # when the resulting attention would be numerically correct.
    for name in ("reshape", "contiguous", "clone"):
        original = getattr(torch.Tensor, name)

        def checked(tensor, *args, _original=original, **kwargs):
            assert tensor.untyped_storage().data_ptr() != storage.data_ptr()
            return _original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, checked)
    attention = dsa.triton_dsa_decode if mode == "decode" else dsa.triton_dsa_prefill

    def run():
        return attention(
            q, cache, None, slots, lens, 513, 128, 512, 64, 192**-0.5, 64, out=out
        )

    def verify():
        for b, length in enumerate(lens.tolist()):
            chosen = slots[b, :length].long()
            chosen = chosen[chosen >= 0]
            selected = paged[chosen // 64, chosen % 64, 0].float()
            expected = (q[b].float() @ selected.T * 192**-0.5).softmax(-1) @ selected[
                :, :512
            ]
            torch.testing.assert_close(out[b].float(), expected, atol=1e-3, rtol=5e-3)

    assert run() is out
    verify()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.mul_(0.5)
    slots[0].copy_(loc[-64:].to(torch.int32))
    lens[1] = 0
    graph.replay()
    verify()
    lens.zero_()
    graph.replay()
    verify()


@pytest.mark.parametrize("n", [4, 513])
def test_flat_mla_fp8_roundtrip(n):
    from tokenspeed_kernel.ops.kvcache.triton import (
        get_mla_kv_buffer_triton,
        set_mla_kv_buffer_triton,
    )

    torch.manual_seed(31)
    cache = torch.zeros((1024, 1, 576), device="cuda", dtype=torch.float8_e4m3fn)
    rows = torch.randn((n, 1, 576), device="cuda", dtype=torch.bfloat16).to(cache.dtype)
    loc = torch.randperm(1023, device="cuda")[:n] + 1
    expected = rows.to(torch.bfloat16)
    set_mla_kv_buffer_triton(
        cache, loc, rows[..., :512], rows[..., 512:], enable_pdl=False, sanitize=False
    )
    out = torch.empty_like(expected)
    get_mla_kv_buffer_triton(
        cache, loc, out[..., :512], out[..., 512:], enable_pdl=False
    )
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    assert (cache[0].float() == 0).all()


def test_index_scatter_topk_dsa_chain():
    from tokenspeed_kernel.ops.attention.dsa._triton.topk import dsa_decode_topk_fp8
    from tokenspeed_kernel.ops.attention.dsa.triton import triton_dsa_decode
    from tokenspeed_kernel.ops.kvcache.triton import (
        index_k_block_split_scatter,
        set_mla_kv_buffer_triton,
    )

    torch.manual_seed(23)
    n, pages, stride, offset = 2053, 35, 1068800, 256
    storage = torch.full(
        (pages * stride + offset,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    latent = _field(storage, pages, 64, 576, stride, 576, offset)
    latent.fill_(float("nan"))
    index = storage.as_strided((pages, 64 * 132), (stride, 1), offset + 64 * 576 * 2)
    index.fill_(255)  # FP8 and FP32 poison for null and unwritten rows.
    physical = torch.randperm(33, device="cuda", dtype=torch.int32) + 1
    positions = torch.arange(n, device="cuda")
    slots = physical[positions // 64].long() * 64 + positions % 64
    rows = torch.randn((n, 1, 576), device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((n, 128), device="cuda")
    scales = keys.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 448
    fp8 = (keys / scales).to(torch.float8_e4m3fn)
    dequant = fp8.float() * scales
    query = torch.randn((2, 4, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((2, 4), device="cuda")
    lens = torch.tensor([0, n], device="cuda", dtype=torch.int32)
    table = physical.repeat(2, 1)
    table[0].zero_()
    q = torch.randn((2, 4, 576), device="cuda", dtype=torch.bfloat16)

    def run():
        set_mla_kv_buffer_triton(
            latent,
            slots,
            rows[..., :512],
            rows[..., 512:],
            enable_pdl=False,
            sanitize=False,
        )
        index_k_block_split_scatter(
            index,
            fp8,
            scales,
            slots,
            page_size=64,
            head_dim=128,
            group_size=128,
            write_mask=None,
        )
        chosen, counts = dsa_decode_topk_fp8(
            query,
            index,
            weights,
            lens,
            table,
            page_size=64,
            topk=2048,
            softmax_scale=128**-0.5,
            q_len_per_req=1,
            topk_layout="global_slots",
            block_table_base_offsets=None,
            out=None,
            lens_out=None,
        )
        out = triton_dsa_decode(
            q, latent, None, chosen, counts, n, 128, 512, 64, 192**-0.5, 64
        )
        return chosen, counts, out

    def verify(chosen, counts, out):
        assert counts.tolist() == [0, 2048]
        assert (chosen[0] == -1).all()
        assert (out[0] == 0).all()
        scores = (query[1].float() @ dequant.T).relu()
        scores = (scores * weights[1, :, None]).sum(0) * 128**-0.5
        expected_slots = slots[scores.topk(2048).indices]
        torch.testing.assert_close(
            chosen[1].long().sort().values, expected_slots.sort().values, atol=0, rtol=0
        )
        selected = latent[chosen[1].long() // 64, chosen[1].long() % 64, 0].float()
        expected = (q[1].float() @ selected.T * 192**-0.5).softmax(-1) @ selected[
            :, :512
        ]
        torch.testing.assert_close(out[1].float(), expected, atol=1e-3, rtol=5e-3)
        assert torch.isnan(latent[0]).all()
        assert (index[0] == 255).all()

    verify(*run())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chosen, counts, out = run()
    query.neg_()
    rows.mul_(0.5)
    graph.replay()
    verify(chosen, counts, out)


@pytest.mark.parametrize("stride", [1068800, 213760])
def test_swa_scatter_decode(stride):
    from tokenspeed_kernel.ops.attention.mla.triton import (
        triton_mla_decode_with_kvcache,
    )
    from tokenspeed_kernel.ops.kvcache.triton import set_mla_kv_buffer_triton

    torch.manual_seed(29)
    lengths = (0, 1, 31, 32, 33, 512, 513, 514, 2049)
    pages = sum((n + 31) // 32 for n in lengths) + 1
    storage = torch.full(
        (pages * stride + 256,), 0xA5, device="cuda", dtype=torch.uint8
    )
    cache = _field(storage, pages, 32, 1088, stride, 1088, 256)
    cache.fill_(float("nan"))
    table = torch.zeros((len(lengths), 65), device="cuda", dtype=torch.int32)
    physical = torch.randperm(pages - 1, device="cuda", dtype=torch.int32) + 1
    rows = torch.randn((sum(lengths), 1, 1088), device="cuda", dtype=torch.bfloat16)
    locations = []
    cursor = 0
    for b, n in enumerate(lengths):
        count = (n + 31) // 32
        table[b, :count] = physical[cursor : cursor + count]
        pos = torch.arange(n, device="cuda")
        locations.append(table[b, pos // 32].long() * 32 + pos % 32)
        table[b, : max(0, n - 513) // 32] = 0
        cursor += count
    loc = torch.cat(locations)
    seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    q = torch.randn((len(lengths), 1, 16, 1088), device="cuda", dtype=torch.bfloat16)

    def run():
        set_mla_kv_buffer_triton(
            cache,
            loc,
            rows[..., :1024],
            rows[..., 1024:],
            enable_pdl=False,
            sanitize=False,
        )
        return triton_mla_decode_with_kvcache(
            q,
            cache,
            table,
            seq,
            max(lengths),
            192,
            1024,
            64,
            1 / 16,
            window_left=512,
            noncausal_block_size=1,
        )

    def verify(out):
        start = 0
        for b, n in enumerate(lengths):
            visible = rows[start + max(0, n - 513) : start + n, 0].float()
            expected = (q[b, 0].float() @ visible.T / 16).softmax(-1) @ visible[
                :, :1024
            ]
            torch.testing.assert_close(
                out[b, 0].float(), expected, atol=5e-4, rtol=5e-3
            )
            start += n
        assert torch.isnan(cache[0]).all()

    verify(run())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    q.mul_(0.5)
    rows.neg_()
    graph.replay()
    verify(out)


if __name__ == "__main__":
    import importlib.util
    import sys

    args = sys.argv[1:]
    if "--isolate-package-imports" in args:
        args.remove("--isolate-package-imports")
        root = Path(__file__).resolve().parents[2] / "python"
        # Use the real MLA API/dispatch and leaf kernels. Skip only unrelated
        # native initialization; this does not validate normal runtime imports.
        for name in (
            "tokenspeed_kernel",
            "tokenspeed_kernel.ops.attention.dsa",
            "tokenspeed_kernel.ops.attention.mla.cuda",
        ):
            spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
            module = importlib.util.module_from_spec(spec)
            module.__path__ = [str(root / name.replace(".", "/"))]
            sys.modules[name] = module
    raise SystemExit(
        pytest.main([__file__, f"--confcutdir={Path(__file__).parent}", *args])
    )
