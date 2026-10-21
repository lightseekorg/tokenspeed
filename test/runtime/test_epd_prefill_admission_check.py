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

"""CPU-only tests for the data plane's side of EPD prefill admission: the
pipeline admission's agreement across stages and the invariant check that
every handshaked image a forward scatters was received. The control plane's
side (staging, the replica agreement, submitting or failing) is
``ts-control``'s; these drive the admission against fakes, so no Mooncake,
ZMQ, or CUDA context is created.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.epd.prefill_admission import (  # noqa: E402
    assert_embeddings_received,
)


def _handshaked_item(offsets=((0, 3),)):
    return SimpleNamespace(
        encode_handshake=object(), encoded=None, offsets=list(offsets)
    )


def _plain_item():
    return SimpleNamespace(encode_handshake=None, encoded=None, offsets=[(0, 3)])


def _context(*mm_items, prefix: int = 0, seq: int = 8):
    return SimpleNamespace(
        has_extend_inputs=lambda: True,
        mm_inputs=[SimpleNamespace(mm_items=list(mm_items))],
        extend_prefix_lens=[prefix],
        extend_seq_lens=[seq],
    )


class _Admission:
    receives_embeddings = True

    def __init__(self) -> None:
        self.staged: list[str] = []
        self.drain_result: tuple[list, list] = ([], [])
        self.pending = False

    def stage(self, request_id, mm_items) -> None:
        self.staged.append(request_id)

    def drain(self):
        return self.drain_result

    def has_pending(self) -> bool:
        return self.pending


# --- assert_embeddings_received --------------------------------------------------


def test_assert_embeddings_received_raises_on_unreceived_handshaked_item() -> None:
    with pytest.raises(RuntimeError, match="un-received"):
        assert_embeddings_received(_Admission(), _context(_handshaked_item()))


def test_assert_embeddings_received_accepts_filled_items_and_non_epd() -> None:
    filled = SimpleNamespace(
        encode_handshake=object(), encoded=object(), offsets=[(0, 3)]
    )
    assert_embeddings_received(_Admission(), _context(filled, _plain_item()))
    # Non-EPD node: no-op even with an un-received item.
    assert_embeddings_received(None, _context(_handshaked_item()))


def test_assert_embeddings_received_ignores_items_an_earlier_chunk_prefilled() -> None:
    """A chunked prompt: the image's tokens [0, 3] were all prefilled by the first
    chunk, which dropped the encoding; the second chunk [8, 16) scatters nothing
    of it. An item still overlapping the chunk must hold its embedding."""
    assert_embeddings_received(
        _Admission(), _context(_handshaked_item(offsets=[(0, 3)]), prefix=8, seq=8)
    )
    with pytest.raises(RuntimeError, match="un-received"):
        assert_embeddings_received(
            _Admission(),
            _context(_handshaked_item(offsets=[(0, 3), (10, 12)]), prefix=8, seq=8),
        )


def test_pipeline_admission_waits_for_pp0_without_receiving_on_followers(monkeypatch):
    from tokenspeed.runtime.epd import prefill_admission

    local = _Admission()
    first = prefill_admission.PipelineEpdPrefillAdmission(
        local_admission=local, pp_ranks=(0, 2, 4, 6), pp_cpu_group="pp"
    )
    follower = prefill_admission.PipelineEpdPrefillAdmission(
        local_admission=None, pp_ranks=(0, 2, 4, 6), pp_cpu_group="pp"
    )
    wire = []

    def broadcast(payload, *, src, group):
        assert src == 0 and group == "pp"
        if payload[0] is not None:
            wire.append(payload[0])
        else:
            payload[0] = wire.pop(0)

    monkeypatch.setattr(prefill_admission.dist, "broadcast_object_list", broadcast)
    for admission in (first, follower):
        admission.stage("ok", [_handshaked_item()])
        admission.stage("failed", [_handshaked_item()])
    assert first.receives_embeddings and not follower.receives_embeddings
    assert local.staged == ["ok", "failed"]
    assert first.drain() == follower.drain() == ([], [])
    assert first.has_pending() and follower.has_pending()
    local.drain_result = (["ok"], ["failed"])
    assert first.drain() == follower.drain() == (["ok"], ["failed"])
    assert not first.has_pending() and not follower.has_pending()
    assert first.drain() == follower.drain() == ([], [])
    assert not wire


def test_pipeline_follower_neither_allocates_pool_nor_reads_encoder_facts(monkeypatch):
    from tokenspeed.runtime.epd import prefill_admission, recv_pool

    mapping = SimpleNamespace(is_first_pp_rank=False, pp_group=(0, 2, 4, 6))
    server_args = SimpleNamespace(disaggregation_mode="prefill", mapping=mapping)

    def forbidden(*args, **kwargs):
        raise AssertionError("non-PP0 attempted to build an encoder receiver")

    monkeypatch.setattr(prefill_admission, "build_prefill_embedding_manager", forbidden)
    admission = prefill_admission.make_epd_prefill_admission(
        server_args,
        2,
        model_config=SimpleNamespace(is_multimodal_active=True),
        encoder_model_facts=forbidden,
        mapping=mapping,
        attn_tp_rank=0,
        attn_tp_size=2,
        attn_tp_cpu_group=None,
        pg_manager=SimpleNamespace(get_process_group=lambda *args: "pp"),
    )
    assert not admission.receives_embeddings
    assert recv_pool.recv_pool_bytes(server_args, True) == 0
    assert_embeddings_received(
        admission,
        SimpleNamespace(
            has_extend_inputs=lambda: True,
            mm_inputs=[SimpleNamespace(mm_items=[_handshaked_item()])],
        ),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
