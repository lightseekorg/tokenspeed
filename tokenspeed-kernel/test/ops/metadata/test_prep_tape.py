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
    tape.gather(idxbuf, table, gsrc, n=Reg.REAL_BS, oob_value=-7)
    tape.copy(qsl, cached, n=Reg.TOKENS)
    tape.copy2d(twod, srcrows, rows=Reg.BS, src_cols=16, pad_value=-1)
    tape.filltail(qsl, live=Reg.TOKENS, total=MAXB + 1, value=555)
    tape.finalize()
    assert len(tape) == 5

    for _ in range(10):
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


def test_recorded_order_waw():
    n = 65536
    buf = torch.zeros(n, dtype=torch.int32, device="cuda")
    tape = PrepTape("cuda")
    tape.fill(buf, n=n, value=1)
    tape.fill(buf[-64:], n=64, value=2)
    tape.finalize()

    for _ in range(50):
        tape.run({})
        torch.cuda.synchronize()
        assert (buf[:-64] == 1).all()
        assert (buf[-64:] == 2).all()


def test_recorded_order_raw():
    n = 65536
    src = torch.zeros(n, dtype=torch.int32, device="cuda")
    dst = torch.zeros(64, dtype=torch.int32, device="cuda")
    tape = PrepTape("cuda")
    tape.fill(src, n=n, value=7)
    tape.copy(dst, src[-64:], n=64)
    tape.finalize()

    for _ in range(50):
        src.zero_()
        tape.run({})
        torch.cuda.synchronize()
        assert (dst == 7).all()


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


@pytest.mark.parametrize("bs", [8, 200])
def test_state_pages_matches_reference(bs):
    torch.manual_seed(1)
    dev, slots, page_size = "cuda", 6, 64
    rows = torch.randint(1, 500, (bs, slots), dtype=torch.int32, device=dev)
    boundary_seq_lens = torch.tensor(
        [1, 63, 64, 65, 128, 129, 200, 384], dtype=torch.int32, device=dev
    )
    seq_lens = boundary_seq_lens.repeat((bs + 7) // 8)[:bs]
    state_in = torch.zeros(bs, dtype=torch.int32, device=dev)
    state_out = torch.zeros(bs, dtype=torch.int32, device=dev)

    tape = PrepTape(dev)
    tape.state_pages(
        state_in,
        state_out,
        rows_ptr=Reg.PTR0,
        seq_lens_ptr=Reg.PTR1,
        bs=Reg.BS,
        max_slots=slots,
        page_size=page_size,
    )
    tape.finalize()
    tape.run({Reg.BS: bs, Reg.PTR0: rows, Reg.PTR1: seq_lens})
    torch.cuda.synchronize()

    after = seq_lens.long()
    before = after - 1
    in_slots = torch.div(before - 1, page_size, rounding_mode="floor").clamp(min=0)
    out_slots = torch.div(after - 1, page_size, rounding_mode="floor").clamp(
        min=0, max=slots - 1
    )
    ref_in = rows.gather(1, in_slots.unsqueeze(1)).squeeze(1)
    ref_in = torch.where(before > 0, ref_in, torch.zeros_like(ref_in))
    ref_out = rows.gather(1, out_slots.unsqueeze(1)).squeeze(1)
    assert torch.equal(state_in, ref_in.int())
    assert torch.equal(state_out, ref_out.int())
