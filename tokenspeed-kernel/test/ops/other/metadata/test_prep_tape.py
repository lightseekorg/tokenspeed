"""Parity tests: a recorded prep tape must equal the eager op chain."""

import pytest
import torch
from tokenspeed_kernel.ops.other.metadata.prep_tape import PrepTape, Reg

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


def test_state_pages_matches_reference():
    torch.manual_seed(1)
    dev = "cuda"
    BS, SLOTS, P = 8, 6, 64
    rows = torch.randint(1, 500, (BS, SLOTS), dtype=torch.int32, device=dev)
    seq_lens = torch.tensor(
        [1, 63, 64, 65, 128, 129, 200, 384], dtype=torch.int32, device=dev
    )
    state_in = torch.zeros(BS, dtype=torch.int32, device=dev)
    state_out = torch.zeros(BS, dtype=torch.int32, device=dev)

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
    tape.finalize()
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
    assert torch.equal(state_in, ref_in.int())
    assert torch.equal(state_out, ref_out.int())
