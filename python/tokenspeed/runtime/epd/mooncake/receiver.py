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

"""Prefill-side per-request receiver for the Mooncake embedding transfer.

Split out of :mod:`tokenspeed.runtime.epd.mooncake.embedding_transfer`; the
prefill-side manager it binds to lives in
:mod:`tokenspeed.runtime.epd.mooncake.prefill`.

Wire-frame dataclasses live in :mod:`tokenspeed.runtime.epd.entities`: they
are pure data plus codecs and import no torch, so the protocol contract stays
importable and unit-testable on CPU.
"""

from __future__ import annotations

import requests

from tokenspeed.runtime.epd.entities import (
    REGISTER_ROOM_SENTINEL,
    EmbeddingArgsRegisterInfo,
    EmbeddingTransferInfo,
)
from tokenspeed.runtime.epd.mooncake.prefill import MooncakeEmbeddingManagerPrefill
from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.network import get_local_ip_by_remote

logger = get_colorful_logger(__name__)


def _route_get(bootstrap_addr: str, engine_rank: int, target_dp_group: int):
    """GET the bootstrap server's /route endpoint; ``None`` on any failure.

    Must never raise: callers in the receiver ``__init__`` treat ``None`` as a
    per-room failure, but an uncaught exception would escape the prefill
    scheduler thread and take the whole engine down via SIGUSR1.
    """
    url = (
        f"http://{bootstrap_addr}/route?"
        f"engine_rank={engine_rank}&target_dp_group={target_dp_group}"
    )
    try:
        resp = requests.get(url, timeout=5)
    except Exception as e:  # noqa: BLE001 -- any transport failure -> per-room fail
        logger.error("EPD bootstrap /route fetch failed (%s): %s", url, e)
        return None
    if resp.status_code == 200:
        return resp.json()
    return None


class MooncakeEmbeddingReceiver:
    """Prefill-side per-request receiver: discovers the encode endpoint via the
    bootstrap server, registers its receive buffer, and on ``pre_alloc`` tells
    the encode side where/how big to write this request's embedding. 1->N
    broadcast (prefill_tp a multiple of encode_tp): contiguous blocks of prefill
    ranks pair one encode rank; encode_tp=1 -> all prefill ranks pair encode
    rank 0 and receive the same TP-gathered embedding.
    """

    def __init__(
        self,
        mgr: MooncakeEmbeddingManagerPrefill,
        bootstrap_addr: str,
        bootstrap_room: int,
    ):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = mgr.get_session_id()
        mgr.update_status(bootstrap_room, TransferPoll.Bootstrapping)

        pinfo = mgr.prefill_parallel_info.get(bootstrap_addr)
        if pinfo is None or pinfo.get("prefill_tp_size") is None:
            # SINGLE-attempt fail-fast: this runs in the prefill scheduler thread,
            # so a retry-loop would stall the whole TP group on a slow/unreachable
            # encode bootstrap. /route can answer 200 with un-populated
            # parallel-info while the encode worker is still registering (startup
            # race); fail this room fast and let the client retry once the encode
            # has registered. A partial dict is never cached (it would crash later
            # requests on int(None)). _route_get bounds its own GET at timeout=5,
            # so this attempt cannot block beyond that.
            cand = _route_get(bootstrap_addr, -1, -1)
            if cand is None or cand.get("prefill_tp_size") is None:
                mgr.record_failure(
                    bootstrap_room, "no (complete) parallel info from bootstrap"
                )
                mgr.update_status(bootstrap_room, TransferPoll.Failed)
                return
            pinfo = cand
            mgr.prefill_parallel_info[bootstrap_addr] = pinfo

        encode_tp = int(pinfo["prefill_tp_size"])
        encode_dp = int(pinfo["prefill_dp_size"])
        local_tp = mgr.world_size // mgr.dp_size
        # prefill_tp must be a whole multiple of encode_tp. The vision tower
        # output is TP-gathered (identical on every encode rank) and every prefill
        # rank needs the full embedding, so encode_tp=1 -> prefill_tp=N is a 1->N
        # broadcast. Contiguous blocks of `fanout` prefill ranks share one encode
        # rank.
        assert local_tp % encode_tp == 0, (
            f"EPD requires prefill_tp to be a multiple of encode_tp "
            f"(encode_tp={encode_tp}, prefill_tp={local_tp})"
        )
        fanout = local_tp // encode_tp  # prefill ranks served by one encode rank
        # Drives the ENCODE-side gate: how many prefill ranks register with this
        # request's encode rank before it may send + conclude (1->N broadcast).
        self.required_dst_info_num = fanout
        # This prefill rank still pulls from exactly ONE encode rank, so it expects
        # a single completion sync. The N-way fan-out is purely the encode side's
        # concern; bumping this to N would hang every prefill rank.
        mgr.required_response_num[bootstrap_room] = 1

        # Which encode rank this prefill rank pulls from: contiguous grouping.
        # encode_tp=1 -> fanout=local_tp -> my_encode_rank == 0 for all prefill
        # ranks. Use `// fanout`, NOT `% encode_tp`: both are 0 for encode_tp=1,
        # but only `// fanout` groups ranks correctly for encode_tp>1.
        my_tp_rank = mgr.embedding_args.engine_rank % local_tp
        my_encode_rank = my_tp_rank // fanout
        target_dp_group = bootstrap_room % encode_dp
        key = f"{bootstrap_addr}_{target_dp_group}_{my_encode_rank}"
        if key not in mgr.connection_pool:
            info = _route_get(bootstrap_addr, my_encode_rank, target_dp_group)
            if info is None:
                mgr.record_failure(bootstrap_room, "no encode rank info from bootstrap")
                mgr.update_status(bootstrap_room, TransferPoll.Failed)
                return
            self.encode_infos = [info]
            mgr.connection_pool[key] = self.encode_infos
            self._register_args()
        else:
            self.encode_infos = mgr.connection_pool[key]
        mgr.update_status(bootstrap_room, TransferPoll.Bootstrapped)

    def _register_args(self):
        ea = self.mgr.embedding_args
        reg = EmbeddingArgsRegisterInfo(
            room=REGISTER_ROOM_SENTINEL,
            endpoint=get_local_ip_by_remote(),
            dst_port=self.mgr.rank_port,
            mooncake_session_id=self.session_id,
            dst_embedding_ptr=ea.embedding_data_ptr,
            dst_deepstack_ptr=ea.deepstack_data_ptr,
        )
        for info in self.encode_infos:
            sock = self.mgr._connect(f"tcp://{info['rank_ip']}:{info['rank_port']}")
            sock.send_multipart(reg.to_zmq())

    def pre_alloc(
        self,
        *,
        dst_embedding_ptr: int,
        n_tokens: int,
        hidden: int,
        dtype: str,
        dst_deepstack_ptr: int = 0,
        has_deepstack: bool = False,
        row_start: int = 0,
        span: int = 0,
    ) -> None:
        """Tell the encode side where/how big to write. In shard mode the caller
        passes this rank's row sub-range: ``row_start`` within the image and
        ``n_tokens`` = the SHARD's row count, both dst pointers already offset to
        the shard's first row. ``span`` is the image's FULL row count (pass it in
        identity mode too: it is the encode side's token-count tripwire). A
        ``n_tokens == 0`` frame is still sent: it doubles as this receiver's
        registration heartbeat (the encode-side fanout gate counts frames, not
        bytes)."""
        for info in self.encode_infos:
            ti = EmbeddingTransferInfo(
                room=self.bootstrap_room,
                endpoint=get_local_ip_by_remote(),
                dst_port=self.mgr.rank_port,
                mooncake_session_id=self.session_id,
                dst_embedding_ptr=dst_embedding_ptr,
                dst_deepstack_ptr=dst_deepstack_ptr,
                n_tokens=n_tokens,
                hidden=hidden,
                dtype=dtype,
                has_deepstack=has_deepstack,
                required_dst_info_num=self.required_dst_info_num,
                row_start=row_start,
                span=span,
            )
            sock = self.mgr._connect(f"tcp://{info['rank_ip']}:{info['rank_port']}")
            sock.send_multipart(ti.to_zmq())

    def poll(self) -> int:
        return self.mgr.check_status(self.bootstrap_room)

    def clear(self) -> None:
        """Drop this room's bookkeeping from the (singleton) prefill manager on a
        terminal receive, else every request leaks its status/tracker entries.
        Mirrors the KV receiver; called post-terminal so it can't race check_status."""
        room = self.bootstrap_room
        self.mgr.request_status.pop(room, None)
        self.mgr.required_response_num.pop(room, None)
        self.mgr.response_tracker.pop(room, None)
        with self.mgr.failure_lock:
            self.mgr.failure_records.pop(room, None)
