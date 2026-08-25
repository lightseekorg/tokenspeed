"""Parity tests: a recorded prep tape must equal the eager op chain."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata import PrepTape, Reg

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def test_tape_matches_eager_chain():
    torch.manual_seed(0)
    dev = "cuda"
    MAXB, W = 32, 24
    idxbuf = torch.full((MAXB,), -1, dtype=torch.int32, device=dev)
    table = torch.randint(0, 1000, (4096,), dtype=torch.int32, device=dev)
    srcrows = torch.randint(0, 99, (MAXB * 16,), dtype=torch.int32, device=dev)
    twod = torch.zeros(MAXB, W, dtype=torch.int32, device=dev)
    qsl = torch.zeros(MAXB + 1, dtype=torch.int32, device=dev)
    cached = torch.arange(MAXB + 1, dtype=torch.int32, device=dev)
    gsrc = torch.randint(0, 4096, (MAXB,), dtype=torch.int32, device=dev)

    tape = PrepTape(dev)
    tape.fill(idxbuf, n=Reg.BS, value=-1)
    tape.barrier()
    tape.gather(idxbuf, table, gsrc, n=Reg.REAL_BS, oob_value=-7)
    tape.copy(qsl, cached, n=Reg.TOKENS)
    tape.copy2d(twod, srcrows, rows=Reg.BS, src_cols=16, pad_value=-1)
    tape.filltail(qsl, live=Reg.TOKENS, total=MAXB + 1, value=555)
    tape.finalize()
    assert len(tape) == 5

    for bs, real in [(8, 8), (8, 5), (32, 32), (1, 1)]:
        ref_idx = table[gsrc[:real].long()]
        tape.run({Reg.BS: bs, Reg.REAL_BS: real, Reg.TOKENS: bs + 1})
        torch.cuda.synchronize()
        assert torch.equal(idxbuf[:real], ref_idx), (bs, real)
        assert (idxbuf[real:bs] == -1).all()
        assert torch.equal(qsl[: bs + 1], cached[: bs + 1])
        assert (qsl[bs + 1 :] == 555).all()
        assert torch.equal(twod[:bs, :16], srcrows[: bs * 16].view(bs, 16))
        assert (twod[:bs, 16:] == -1).all()


def test_gather_negative_index_pads():
    dev = "cuda"
    dst = torch.zeros(4, dtype=torch.int32, device=dev)
    src = torch.arange(10, dtype=torch.int32, device=dev)
    idx = torch.tensor([3, -1, 7, -1], dtype=torch.int32, device=dev)
    tape = PrepTape(dev)
    tape.gather(dst, src, idx, n=4, oob_value=-7)
    tape.finalize()
    tape.run({})
    torch.cuda.synchronize()
    assert dst.tolist() == [3, -7, 7, -7]


def test_finalized_tape_rejects_new_ops():
    dev = "cuda"
    buf = torch.zeros(8, dtype=torch.int32, device=dev)
    tape = PrepTape(dev)
    tape.fill(buf, n=8, value=1)
    tape.finalize()
    with pytest.raises(RuntimeError):
        tape.fill(buf, n=8, value=2)
    with pytest.raises(RuntimeError):
        tape.barrier()


def test_dependent_source_write_precedes_gather_read():
    """Exercise a cross-program RAW hazard, not only two writes to dst."""
    dev = "cuda"
    n = 1 << 18
    src = torch.zeros(n, dtype=torch.int32, device=dev)
    idx = torch.tensor([n - 1], dtype=torch.int32, device=dev)
    dst = torch.zeros(1, dtype=torch.int32, device=dev)

    tape = PrepTape(dev)
    tape.fill(src, n=n, value=123)
    tape.barrier()
    tape.gather(dst, src, idx, n=1)
    tape.finalize()
    assert isinstance(tape._descs, tuple) and len(tape._descs) == 2

    tape.run({})
    torch.cuda.synchronize()
    assert dst.item() == 123


def test_state_pages_matches_reference():
    torch.manual_seed(1)
    dev = "cuda"
    BS, SLOTS, P = 8, 6, 64
    rows = torch.randint(1, 500, (BS, SLOTS), dtype=torch.int32, device=dev)
    seq_lens = torch.tensor(
        [1, 63, 64, 65, 128, 129, 200, 384], dtype=torch.int32, device=dev
    )
    state_in = torch.zeros(BS + 4, dtype=torch.int32, device=dev)
    state_out = torch.zeros(BS + 4, dtype=torch.int32, device=dev)

    tape = PrepTape(dev)
    tape.state_pages(
        state_in,
        state_out,
        rows_ptr=Reg.PTR0,
        seq_lens_ptr=Reg.PTR1,
        bs=Reg.BS,
        max_slots=SLOTS,
        page_size=P,
    )
    tape.filltail(state_in, live=Reg.BS, total=BS + 4, value=-1)
    tape.filltail(state_out, live=Reg.BS, total=BS + 4, value=-1)
    tape.finalize()
    assert isinstance(tape._descs, torch.Tensor)
    tape.run({Reg.BS: BS, Reg.PTR0: rows, Reg.PTR1: seq_lens})
    torch.cuda.synchronize()

    after = seq_lens.long()
    before = after - 1
    in_slots = torch.div(before - 1, P, rounding_mode="floor").clamp(min=0)
    out_slots = torch.div(after - 1, P, rounding_mode="floor").clamp(
        min=0, max=SLOTS - 1
    )
    ref_in = rows.gather(1, in_slots.unsqueeze(1)).squeeze(1)
    ref_in = torch.where(before > 0, ref_in, torch.zeros_like(ref_in))
    ref_out = rows.gather(1, out_slots.unsqueeze(1)).squeeze(1)
    assert torch.equal(state_in[:BS], ref_in.int())
    assert torch.equal(state_out[:BS], ref_out.int())
    assert (state_in[BS:] == -1).all()
    assert (state_out[BS:] == -1).all()


def test_filltail_writes_int64_buffers():
    """Graph input buffers that index a request pool are int64; a tail written
    as int32 would leave every other 4 bytes of the pair stale."""
    dst = torch.arange(16, dtype=torch.int64, device="cuda")
    before = dst.clone()
    tape = PrepTape("cuda")
    tape.filltail(dst, Reg.BS, 16, -7)
    tape.finalize()
    tape.run({Reg.BS: 5})
    assert torch.equal(dst[:5], before[:5])
    assert dst[5:].tolist() == [-7] * 11
    assert dst.dtype == torch.int64


def test_filltail_int32_and_int64_share_one_launch():
    """Mixed widths must coexist in a stage: that is what the input buffers are."""
    a = torch.zeros(12, dtype=torch.int32, device="cuda")
    b = torch.zeros(12, dtype=torch.int64, device="cuda")
    tape = PrepTape("cuda")
    tape.filltail(a, Reg.BS, 12, 3)
    tape.filltail(b, Reg.BS, 12, 1 << 40)
    tape.finalize()
    tape.run({Reg.BS: 4})
    assert a.tolist() == [0] * 4 + [3] * 8
    assert b.tolist() == [0] * 4 + [1 << 40] * 8


def test_filltail_rejects_a_width_it_cannot_address():
    tape = PrepTape("cuda")
    with pytest.raises(TypeError):
        tape.filltail(torch.zeros(4, dtype=torch.int16, device="cuda"), Reg.BS, 4, 0)
