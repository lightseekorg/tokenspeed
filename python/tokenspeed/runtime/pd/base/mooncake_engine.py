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

import logging
import os

import numpy as np

from tokenspeed.runtime.utils.network import is_valid_ipv6_address, join_host_port

logger = logging.getLogger(__name__)


def enable_mooncake_ipv6_for_host(hostname: str) -> None:
    """Make Mooncake's handshake use IPv6 when ``hostname`` is an IPv6 address.

    Mooncake opens its P2P handshake sockets as IPv4 unless ``MC_USE_IPV6`` is
    set. Mooncake reads ``MC_USE_IPV6`` when its first engine starts, so set it
    before then; an explicit setting is kept.
    """
    if is_valid_ipv6_address(hostname):
        os.environ.setdefault("MC_USE_IPV6", "1")


class MooncakeTransferEngine:
    def __init__(self, hostname: str, gpu_id: int, ib_device: str | None = None):
        enable_mooncake_ipv6_for_host(hostname)
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run TokenSpeed with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = ib_device
        # Prefer Mooncake's page API when available; older builds expose only
        # the ordinary batched WRITE API. This is a capability, not a setting.
        self.page_transfer_mode = (
            "native"
            if callable(getattr(self.engine, "batch_transfer_sync_write_pages", None))
            else "batch"
        )
        required_api = (
            "batch_transfer_sync_write_pages"
            if self.page_transfer_mode == "native"
            else "batch_transfer_sync_write"
        )
        if not callable(getattr(self.engine, required_api, None)):
            raise RuntimeError(
                f"Mooncake's TransferEngine lacks {required_api} required by "
                f"page_transfer_mode={self.page_transfer_mode!r}."
            )

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        # Bracket IPv6 hosts the way Mooncake writes its own segment names.
        self.session_id = join_host_port(self.hostname, self.engine.get_rpc_port())
        # The peer identifies this rank by session_id in its transfer logs
        # (Prefill's "session=..."), so name it once per process here.
        logger.info(
            f"Mooncake transfer engine ready: session_id={self.session_id} "
            f"gpu_id={self.gpu_id} ib_device={self.ib_device} "
            f"page_transfer_mode={self.page_transfer_mode}"
        )

    def register(self, ptr, length):
        """Register ``ptr`` with Mooncake.

        Returns:
            0 on success, nonzero on failure. KV-arena registration may ignore
            the result; pack scratch must check it before swapping buffers.
        """
        try:
            ret_value = self.engine.register_memory(ptr, length)
        except Exception:
            # Mark register as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug(f"Mooncake memory registration {ptr!s} failed.")
        return ret_value

    def deregister(self, ptr):
        """Deregister ``ptr`` from Mooncake.

        Returns:
            0 on success, nonzero on failure.
        """
        try:
            ret_value = self.engine.unregister_memory(ptr)
        except Exception:
            # Mark deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug(f"Mooncake memory deregistration {ptr!s} failed.")
        return ret_value

    def initialize(
        self,
        hostname: str,
        device_name: str | None,
    ) -> None:
        """Initialize the mooncake instance."""
        ret_value = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "rdma",
            device_name if device_name is not None else "",
        )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        try:
            # the first time: based on session_id (which contains remote_ip) to construct a queue pair, and cache the queue pair
            # later: based on the cached queue pair to send data
            ret = self.engine.transfer_sync_write(
                session_id, buffer, peer_buffer_address, length
            )
        except Exception:
            # Mark transfer request as failed
            ret = -1

        if ret < 0:
            # Do not raise an exception here, since some transfer requests fail should be accepted and the execution thread should not be stopped.
            logger.debug(
                f"Failed to transfer data from {buffer!s} to {session_id!s} - "
                f"{peer_buffer_address!s}.",
            )

        return ret

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: list[int],
        peer_buffer_addresses: list[int],
        lengths: list[int],
    ) -> int:
        """Synchronously transfer data to the specified addresses in batches."""
        try:
            ret = self.engine.batch_transfer_sync_write(
                session_id, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            ret = -1
            # Inform user to upgrade mooncake-transfer-engine >= 0.3.4.post2
            if not hasattr(self.engine, "batch_transfer_sync_write"):
                raise RuntimeError(
                    "Mooncake's batch transfer requires mooncake-transfer-engine >= 0.3.4.post2. "
                    "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'"
                )

        if ret < 0:
            logger.debug(
                f"Failed to batch transfer data. Buffers: {buffers!s}, Session: "
                f"{session_id!s}, Peer addresses: {peer_buffer_addresses!s}",
            )
        return ret

    def batch_transfer_sync_pages(
        self,
        session_id: str,
        src_pages: np.ndarray,
        dst_pages: np.ndarray,
        fields: np.ndarray,
        *,
        max_batch_size: int,
    ) -> int:
        """WRITE the pages x fields grid described by ``fields`` (see
        ``PageFieldCopies``) in batches of at most ``max_batch_size``."""
        try:
            if self.page_transfer_mode == "native":
                ret = self.engine.batch_transfer_sync_write_pages(
                    session_id, src_pages, dst_pages, fields, max_batch_size
                )
            else:
                ret = self._batch_transfer_pages(
                    session_id, src_pages, dst_pages, fields, max_batch_size
                )
        except Exception:
            logger.exception("Mooncake page-gathered batch transfer raised")
            ret = -1
        if ret != 0:
            logger.debug(
                f"Page transfer failed: session={session_id!s} "
                f"mode={self.page_transfer_mode} ret={ret}"
            )
        return ret

    def _batch_transfer_pages(
        self, session_id, src_pages, dst_pages, fields, max_batch_size
    ) -> int:
        """Expand metadata only; device payloads remain direct WRITE sources.

        Validate the entire grid before submission, without materializing it.
        A failed batch may still have writes in flight: stop, never retry it
        here, and let the receiver apply the transfer-failure lifetime policy.
        """
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        for pages in (src_pages, dst_pages):
            if (
                not isinstance(pages, np.ndarray)
                or pages.ndim != 1
                or pages.dtype != np.int64
                or not pages.flags.c_contiguous
            ):
                raise TypeError("page IDs must be contiguous 1-D int64 arrays")
            if np.any(pages < 0):
                raise ValueError("page IDs must be nonnegative")
        if src_pages.shape != dst_pages.shape:
            raise ValueError("source and destination page counts differ")
        if (
            not isinstance(fields, np.ndarray)
            or fields.ndim != 2
            or fields.shape[1] != 5
            or fields.dtype != np.int64
            or not fields.flags.c_contiguous
        ):
            raise TypeError("fields must be a contiguous int64 [F, 5] array")
        src_max = int(src_pages.max()) if src_pages.size else 0
        dst_max = int(dst_pages.max()) if dst_pages.size else 0
        max_address = int(np.iinfo(np.uintp).max)
        for row in fields:
            src_base, src_stride, dst_base, dst_stride, length = map(int, row)
            if min(src_base, dst_base, length) <= 0 or min(src_stride, dst_stride) < 0:
                raise ValueError(
                    "bases/length must be positive and strides nonnegative"
                )
            if (
                src_base + src_max * src_stride + length - 1 > max_address
                or dst_base + dst_max * dst_stride + length - 1 > max_address
            ):
                raise OverflowError("page transfer address exceeds uintptr_t")

        sources: list[int] = []
        destinations: list[int] = []
        lengths: list[int] = []
        for row in fields:
            src_base, src_stride, dst_base, dst_stride, length = map(int, row)
            for i in range(src_pages.size):
                sources.append(src_base + int(src_pages[i]) * src_stride)
                destinations.append(dst_base + int(dst_pages[i]) * dst_stride)
                lengths.append(length)
                if len(sources) == max_batch_size:
                    ret = self.engine.batch_transfer_sync_write(
                        session_id, sources, destinations, lengths
                    )
                    if ret != 0:
                        return ret
                    sources.clear()
                    destinations.clear()
                    lengths.clear()
        if sources:
            return self.engine.batch_transfer_sync_write(
                session_id, sources, destinations, lengths
            )
        return 0

    def transfer_submit_write(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """ASynchronously transfer data to the specified address."""

        batch_id = self.engine.transfer_submit_write(
            session_id, buffer, peer_buffer_address, length
        )
        return batch_id

    def transfer_check_status(self, batch_id: int) -> int:
        status = self.engine.transfer_check_status(batch_id)
        return status

    def get_session_id(self):
        return self.session_id
