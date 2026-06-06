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

"""Weight-transfer lifecycle manager for RL online weight sync.

``WeightTransferManager`` is the control-plane state machine that the HTTP
handlers in ``runtime/entrypoints/vllm_compat_http.py`` call into. It
implements the weight-update lifecycle that RL trainers drive over HTTP
(init / start / update / finish / pause / resume / is_paused).

It does NOT move tensors itself: heavy payloads travel out-of-band over NCCL
broadcast / CUDA-IPC. The manager parses the backend-specific ``init_info`` /
``update_info`` metadata, enforces lifecycle ordering, and delegates the actual
transfer to ``AsyncLLM``'s existing scheduler-control methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.engine.io_struct import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from tokenspeed.runtime.engine.weight_transfer.config import (
    SUPPORTED_BACKENDS,
    WeightTransferConfig,
)
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

logger = get_colorful_logger(__name__)

# Pause modes accepted by ``/pause``.
PAUSE_MODES = ("abort", "wait", "keep")


class WeightTransferStateError(RuntimeError):
    """Raised when lifecycle methods are called out of order.

    The HTTP layer maps this to ``409 Conflict``. ``ValueError`` (bad input) is
    mapped to ``400 Bad Request`` instead.
    """


class WeightTransferManager:
    """Lifecycle state machine for RL weight transfer."""

    def __init__(
        self,
        async_llm: "AsyncLLM",
        config: WeightTransferConfig | None = None,
    ) -> None:
        self._async_llm = async_llm
        self._config = config or WeightTransferConfig()
        if self._config.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported weight transfer backend: {self._config.backend!r}"
            )
        # Lifecycle state.
        self._engine_inited = False
        self._update_active = False

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def backend(self) -> str:
        return self._config.backend

    def is_paused(self) -> bool:
        """Whether generation admission is currently paused."""
        return self._async_llm.weight_transfer_admission_paused()

    def get_world_size(self, include_dp: bool = True) -> int:
        """Return the inference world size used to size the NCCL group.

        Args:
            include_dp: If True (default), return the total world size across
                data-parallel replicas (TP*CP*DP). If False, return a single
                replica's world size (TP*CP).
        """
        mapping = self._async_llm.server_args.mapping
        if include_dp:
            # tokenspeed's world_size already counts every GPU (TP*CP*DP).
            return mapping.world_size
        # A single DP replica (TP*CP).
        return mapping.world_size // mapping.attn.dp_size

    # ------------------------------------------------------------------ #
    # Weight-update lifecycle
    # ------------------------------------------------------------------ #

    async def init_engine(self, init_info: dict[str, Any]) -> None:
        """Initialize the weight transfer engine (once per training run).

        NCCL: set up the process group with the trainer. IPC: no-op.
        """
        if not isinstance(init_info, dict):
            raise ValueError("init_info must be a JSON object")

        if self.backend == "nccl":
            obj = self._parse_nccl_init(init_info)
            success, message = await self._async_llm.init_weights_update_group(obj)
            if not success:
                raise RuntimeError(f"Failed to init weight transfer engine: {message}")
        else:  # ipc
            self._parse_ipc_init(init_info)  # validate only; IPC needs no group
        self._engine_inited = True
        logger.info("Weight transfer engine initialized (backend=%s)", self.backend)

    async def start_update(self) -> None:
        """Begin a weight update. Trainers call this once per RL step."""
        if not self._engine_inited:
            raise WeightTransferStateError(
                "Weight transfer engine not initialized; call "
                "/init_weight_transfer_engine first."
            )
        if self._update_active:
            raise WeightTransferStateError(
                "A weight update is already in progress; call "
                "/finish_weight_update before starting another."
            )
        self._update_active = True
        logger.info("Weight update started")

    async def update(self, update_info: dict[str, Any]) -> None:
        """Receive one chunk of weights (metadata in ``update_info``)."""
        if not self._update_active:
            raise WeightTransferStateError(
                "No active weight update; call /start_weight_update first."
            )
        if not isinstance(update_info, dict):
            raise ValueError("update_info must be a JSON object")

        if self.backend == "nccl":
            obj = self._parse_nccl_update(update_info)
            success, message = await self._async_llm.update_weights_from_distributed(
                obj
            )
            if not success:
                raise RuntimeError(f"Failed to update weights: {message}")
        else:  # ipc
            self._parse_ipc_update(update_info)
            # The CUDA-IPC receive path (rebuild_cuda_tensor + load_weights on the
            # colocated worker) is not yet wired through the scheduler. The
            # metadata is fully parsed/validated above.
            #
            # SECURITY: when wiring the worker-side receive, prefer the structured
            # `ipc_handles` form (JSON-safe handle fields) and avoid
            # ``pickle.loads`` on ``ipc_handles_pickled`` — the control plane is
            # reachable by anything that can reach rl_control_port, so unpickling
            # caller-supplied bytes there is an RCE vector.
            raise NotImplementedError(
                "IPC weight receive is not yet implemented on the worker side; "
                "use backend='nccl' for now."
            )

    async def finish_update(self) -> None:
        """Finalize the current weight update."""
        if not self._update_active:
            raise WeightTransferStateError(
                "No active weight update to finish; call /start_weight_update first."
            )
        self._update_active = False
        # A worker-side layerwise-reload finalize lives on the (deferred) worker
        # path; the checkpoint-format flag will be threaded through start_update
        # when that lands. Cache invalidation is handled by the update step's
        # flush_cache.
        logger.info("Weight update finished")

    # ------------------------------------------------------------------ #
    # Pause / resume
    # ------------------------------------------------------------------ #

    async def pause(self, mode: str = "abort", clear_cache: bool = True) -> None:
        """Pause generation so weights can be updated safely.

        Args:
            mode: ``"abort"`` aborts in-flight requests and blocks new ones;
                ``"wait"`` drains in-flight requests then blocks new ones;
                ``"keep"`` blocks new requests but preserves in-flight state.
            clear_cache: Flush KV/prefix cache after draining. Ignored for
                ``mode="keep"``.
        """
        if mode not in PAUSE_MODES:
            raise ValueError(
                f"Invalid pause mode: {mode!r}. Must be one of {PAUSE_MODES}."
            )

        # Block new admission first so nothing slips in while we drain/abort.
        self._async_llm.weight_transfer_block_admission()

        if mode == "abort":
            self._async_llm.weight_transfer_abort_inflight()
        elif mode == "wait":
            await self._async_llm.weight_transfer_drain_inflight()
        # mode == "keep": leave in-flight requests untouched.

        if clear_cache and mode != "keep":
            try:
                await self._async_llm.flush_cache()
            except Exception as e:  # noqa: BLE001 - flush is best-effort on pause
                logger.warning("flush_cache during pause failed: %s", e)

        logger.info("Generation paused (mode=%s, clear_cache=%s)", mode, clear_cache)

    async def resume(self) -> None:
        """Resume generation after a pause."""
        self._async_llm.weight_transfer_allow_admission()
        logger.info("Generation resumed")

    # ------------------------------------------------------------------ #
    # Backend-specific parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_keys(
        info: dict[str, Any],
        required: tuple[str, ...],
        allowed: tuple[str, ...],
        what: str,
    ) -> None:
        missing = [k for k in required if k not in info]
        if missing:
            raise ValueError(f"{what} missing required key(s): {missing}")
        unknown = sorted(set(info) - set(allowed))
        if unknown:
            raise ValueError(
                f"{what} has unknown key(s): {unknown}. Allowed: {sorted(allowed)}."
            )

    @staticmethod
    def _check_dense(update_info: dict[str, Any], what: str) -> None:
        """Accept-and-ignore ``update_kind="dense"`` (the default); reject sparse.

        ``update_kind`` / ``num_updates_list`` are part of the weight-update wire
        format; only dense updates are implemented here.
        """
        update_kind = update_info.get("update_kind", "dense")
        if update_kind != "dense":
            raise ValueError(
                f"{what}: unsupported update_kind={update_kind!r}; only 'dense' "
                "weight updates are supported (sparse updates are not implemented)."
            )

    def _parse_nccl_init(
        self, init_info: dict[str, Any]
    ) -> InitWeightsUpdateGroupReqInput:
        required = ("master_address", "master_port", "rank_offset", "world_size")
        self._require_keys(
            init_info, required, required + ("group_name",), "NCCL init_info"
        )
        return InitWeightsUpdateGroupReqInput(
            master_address=str(init_info["master_address"]),
            master_port=int(init_info["master_port"]),
            rank_offset=int(init_info["rank_offset"]),
            world_size=int(init_info["world_size"]),
            group_name=str(init_info.get("group_name", "weight_update_group")),
            backend="nccl",
        )

    def _parse_nccl_update(
        self, update_info: dict[str, Any]
    ) -> UpdateWeightsFromDistributedReqInput:
        required = ("names", "dtype_names", "shapes")
        allowed = required + (
            "packed",
            "packed_buffer_size_bytes",
            "packed_num_buffers",
            "group_name",
            "flush_cache",
            "update_kind",
            "num_updates_list",
        )
        self._require_keys(update_info, required, allowed, "NCCL update_info")
        self._check_dense(update_info, "NCCL update_info")
        names = list(update_info["names"])
        dtype_names = list(update_info["dtype_names"])
        shapes = [list(s) for s in update_info["shapes"]]
        if not (len(names) == len(dtype_names) == len(shapes)):
            raise ValueError(
                "names, dtype_names, shapes must have equal length: "
                f"{len(names)}, {len(dtype_names)}, {len(shapes)}"
            )
        return UpdateWeightsFromDistributedReqInput(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=bool(update_info.get("packed", False)),
            packed_buffer_size_bytes=int(
                update_info.get(
                    "packed_buffer_size_bytes", DEFAULT_PACKED_BUFFER_SIZE_BYTES
                )
            ),
            packed_num_buffers=int(
                update_info.get("packed_num_buffers", DEFAULT_PACKED_NUM_BUFFERS)
            ),
            group_name=str(update_info.get("group_name", "weight_update_group")),
            flush_cache=bool(update_info.get("flush_cache", True)),
        )

    def _parse_ipc_init(self, init_info: dict[str, Any]) -> None:
        # IPC needs no init payload.
        self._require_keys(init_info, (), (), "IPC init_info")

    def _parse_ipc_update(self, update_info: dict[str, Any]) -> dict[str, Any]:
        required = ("names", "dtype_names", "shapes")
        allowed = required + (
            "ipc_handles",
            "ipc_handles_pickled",
            "tensor_sizes",
            "packed",
            "update_kind",
            "num_updates_list",
        )
        self._require_keys(update_info, required, allowed, "IPC update_info")
        self._check_dense(update_info, "IPC update_info")
        names = list(update_info["names"])
        dtype_names = list(update_info["dtype_names"])
        shapes = [list(s) for s in update_info["shapes"]]
        if not (len(names) == len(dtype_names) == len(shapes)):
            raise ValueError(
                "names, dtype_names, shapes must have equal length: "
                f"{len(names)}, {len(dtype_names)}, {len(shapes)}"
            )
        has_handles = update_info.get("ipc_handles") is not None
        has_pickled = update_info.get("ipc_handles_pickled") is not None
        if has_handles and has_pickled:
            raise ValueError(
                "Cannot specify both `ipc_handles` and `ipc_handles_pickled`"
            )
        if not has_handles and not has_pickled:
            raise ValueError(
                "Either `ipc_handles` or `ipc_handles_pickled` must be provided"
            )
        packed = bool(update_info.get("packed", False))
        if packed and update_info.get("tensor_sizes") is None:
            raise ValueError("`tensor_sizes` is required when packed=True")
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": packed,
        }
