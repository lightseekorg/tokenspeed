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

"""Vendored FlashInfer/Cake Kimi-K3 fused front projections (prefill, TP1).

One persistent launch computes, for ``M`` BF16 rows of width 7168, the FP32
router logits over 896 experts, the BF16 latent projection (3584 wide) and the
BF16 SiTU-activated shared-expert input (6144 wide, full shared experts). The
generated programs under ``csrc/kimi3_latent_front/`` are unchanged copies from
FlashInfer v0.7.3's ``flashinfer.experimental.kimi_k3_latent_moe`` (provenance in
``THIRDPARTYNOTICES``), and the host plan rules below are those of the same
version. Release wheels of FlashInfer do not ship the experimental kernel
sources, so they live here and are compiled through FlashInfer's JIT toolchain
like ``kimi3_router``.

Scope: the prefill front route of one rank holding the full shared experts
(``shared_gate_up_weight`` of ``2 * 6144`` rows) for 513 rows or more on
148- and 152-SM SM100/SM103 parts. Rows of at most 512 would take programs
that are not vendored (the decode front and the 224-wide tile); the
sharded-shared-expert (TP8) programs are not vendored either. There is no
upper row bound: the row count is a runtime argument, the plan rules scale
with the tile count, and the programs index rows in 64-bit.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
SHARED_INTERMEDIATE = 6144
#: Compute capabilities the programs are compiled for.
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
#: Physical SM counts the plan rules were validated on: 148 (B200, B300) and 152 (GB200, GB300).
QUALIFIED_SM_COUNTS = (148, 152)
#: Smallest row count the vendored programs serve; see the module docstring.
MIN_TOKENS = 513

# Tile geometry of the front GEMM programs.
BLOCK_M = 128
FRONT_BLOCK_N = 256
CTA_GROUP = 2  # one two-CTA cluster per output tile pair
BLOCK_K = 64
FRONT_NUM_K_ITERS = HIDDEN // BLOCK_K  # 112 K iterations per 256 x 256 pair tile
FRONT_SPLIT = 2  # trailing-wave tiles cut into two aligned K halves
FRONT_SK_EPI_ITERS = (
    18  # exposed stream-K epilogue of the last contributor, in K iterations
)
FRONT_SK_MIN_GAIN = (
    0.04  # split only when the modelled saving is >= 4 % of the whole-tile cost
)
#: Weight boxes re-read by at most this many pair rows stream with the ``evict_first`` L2 policy
#: (a separate program); 152-SM parts take it up to eight pair rows (M <= 2048).
FRONT_EVICT_FIRST_MAX_PAIR_ROWS = 2
FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_SM_COUNT = {152: 8}

#: Keyword order of the programs' ``run`` entry; ``grid`` expands to ``grid_x/y/z``.
FRONT_KWARGS = (
    "A",
    "WG",
    "WD",
    "WS",
    "logits",
    "latent",
    "shared_act",
    "ws",
    "counters",
    "M",
    "m_tiles",
    "num_items",
    "full_items",
    "sk_ipc",
    "sk_max_seg",
    "sk_total",
    "grid",
)
#: Argument position of the partial workspace in ``FRONT_KWARGS``.
WS_POSITION = FRONT_KWARGS.index("ws")
#: Launch shape shared by both programs.
BLOCK_THREADS = 320
DYNAMIC_SMEM_BYTES = 230528
#: Generated program per logical kernel key: ``front:i6144`` streams the weights with the default
#: L2 policy, ``front:i6144e1`` with ``evict_first``. Both take the same arguments.
PROGRAMS = {
    "front:i6144": "cake_kimi_k3_latent_moe_b36cca5dfbea2e45cbac",
    "front:i6144e1": "cake_kimi_k3_latent_moe_14ba3299b6dbce540fa9",
}
CSRC_DIR = Path(__file__).parent / "csrc" / "kimi3_latent_front"

# Control words are handed out in 128-byte slots of 32 four-byte words.
CTRL_SLOT_WORDS = 32
# Four-byte words of one slab chunk (256 KiB): one zero fill for hundreds of runners.
CTRL_CHUNK_WORDS = 1 << 16


# ---------------------------------------------------------------------------
# Host plan
# ---------------------------------------------------------------------------


def m_tiles_for(M: int) -> int:
    """Row tiles of ``M`` rows, rounded up to whole cluster pairs."""
    tiles = (M + BLOCK_M - 1) // BLOCK_M
    return tiles + (tiles % CTA_GROUP)


def front_n_tiles() -> int:
    """Column tile pairs per row pair: router, latent, then the shared gate/up halves."""
    shared_half = FRONT_BLOCK_N // CTA_GROUP
    return (
        (NUM_EXPERTS + FRONT_BLOCK_N - 1) // FRONT_BLOCK_N
        + LATENT // FRONT_BLOCK_N
        + SHARED_INTERMEDIATE // shared_half
    )


def front_split_plan(cluster_tiles: int, sm_count: int) -> dict[str, int]:
    """Persistent work plan: the full waves run whole pair tiles; the trailing partial wave is cut
    into two aligned K halves per tile when the halves fit one wave and the modelled saving
    reaches ``FRONT_SK_MIN_GAIN`` of the whole-tile cost."""
    resident = max(1, sm_count // CTA_GROUP)
    full_waves, rem = divmod(cluster_tiles, resident)
    whole = {
        "num_items": cluster_tiles,
        "full_items": cluster_tiles,
        "sk_ipc": FRONT_NUM_K_ITERS,
        "sk_max_seg": 1,
        "sk_total": 0,
        "sk_tiles": 0,
    }
    if rem == 0 or rem * FRONT_SPLIT > resident:
        return whole
    whole_cost = (full_waves + 1) * FRONT_NUM_K_ITERS
    saving = FRONT_NUM_K_ITERS - FRONT_NUM_K_ITERS // FRONT_SPLIT - FRONT_SK_EPI_ITERS
    if saving < FRONT_SK_MIN_GAIN * whole_cost:
        return whole
    return {
        "num_items": cluster_tiles - rem + rem * FRONT_SPLIT,
        "full_items": cluster_tiles - rem,
        "sk_ipc": FRONT_NUM_K_ITERS // FRONT_SPLIT,
        "sk_max_seg": FRONT_SPLIT,
        "sk_total": rem * FRONT_NUM_K_ITERS,
        "sk_tiles": rem,
    }


def front_evict_first(m_tiles: int, sm_count: int) -> bool:
    """Whether the ``evict_first`` weight-policy program serves ``m_tiles`` row tiles."""
    limit = FRONT_EVICT_FIRST_MAX_PAIR_ROWS_BY_SM_COUNT.get(
        int(sm_count), FRONT_EVICT_FIRST_MAX_PAIR_ROWS
    )
    return (m_tiles // CTA_GROUP) <= limit


@functools.lru_cache(maxsize=None)
def _front_plan(M: int, sm_count: int) -> dict[str, Any]:
    m_tiles = m_tiles_for(M)
    cluster_tiles = (m_tiles // CTA_GROUP) * front_n_tiles()
    sk = front_split_plan(cluster_tiles, sm_count)
    return dict(
        M=M,
        m_tiles=m_tiles,
        cluster_tiles=cluster_tiles,
        evict_first=front_evict_first(m_tiles, sm_count),
        grid=sk["num_items"] * CTA_GROUP,
        **sk,
    )


def front_plan(M: int, sm_count: int) -> dict[str, Any]:
    """Host plan of the front launch for ``M`` rows on a device of ``sm_count`` SMs.

    Args:
        M: Row count, at least ``MIN_TOKENS``.
        sm_count: SM count of the device, one of ``QUALIFIED_SM_COUNTS``.

    Returns:
        A copy of the plan: the kernel arguments (``m_tiles``, ``num_items``, ``full_items``,
        ``sk_ipc``, ``sk_max_seg``, ``sk_total``, ``grid``), the split-tile count ``sk_tiles``
        and the weight policy ``evict_first``.
    """
    if int(M) < MIN_TOKENS:
        raise ValueError(
            f"the vendored Kimi-K3 front programs serve at least {MIN_TOKENS} rows, got {M}"
        )
    if int(sm_count) not in QUALIFIED_SM_COUNTS:
        raise ValueError(
            f"the Kimi-K3 front plan rules are validated for {QUALIFIED_SM_COUNTS} SMs, got {sm_count}"
        )
    return dict(_front_plan(int(M), int(sm_count)))


def front_kernel_key(plan: dict[str, Any]) -> str:
    """The logical kernel key of a plan."""
    return "front:i6144e1" if plan["evict_first"] else "front:i6144"


def partial_ws_numel(plan: dict[str, Any]) -> int:
    """FP32 elements of the stream-K partial workspace one launch uses: one ``128 x 256`` slot per
    contributor CTA of a split tile (0 for a whole plan, which writes none)."""
    sk_tiles = int(plan["sk_tiles"])
    return sk_tiles * int(plan["sk_max_seg"]) * CTA_GROUP * BLOCK_M * FRONT_BLOCK_N


def arrival_counter_words(plan: dict[str, Any]) -> int:
    """Int32 stream-K arrival counters of a plan: one per split tile and CTA rank, at least two."""
    return max(2, int(plan["sk_tiles"]) * CTA_GROUP)


# ---------------------------------------------------------------------------
# Device facts and program loading
# ---------------------------------------------------------------------------


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _device_facts(index: int) -> tuple[tuple[int, int], int]:
    properties = torch.cuda.get_device_properties(index)
    return (int(properties.major), int(properties.minor)), int(
        properties.multi_processor_count
    )


def available(device: torch.device | None = None) -> bool:
    """Whether the vendored programs run on ``device`` (the current device when omitted)."""
    if not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device() if device is None else _device_index(device)
    capability, count = _device_facts(index)
    return capability in SUPPORTED_COMPUTE_CAPABILITIES and count in QUALIFIED_SM_COUNTS


@functools.cache
def _module(key: str):
    """The JIT-built program serving ``key``, compiled once per process for SM100a and SM103a."""
    from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

    name = PROGRAMS[key]
    sources = [CSRC_DIR / f"{name}_kernel.cu", CSRC_DIR / f"{name}_binding.cu"]
    return gen_jit_spec(
        f"tokenspeed_kimi3_latent_front_{name[-20:]}",
        sources,
        extra_cuda_cflags=[*sm100a_nvcc_flags, *sm103a_nvcc_flags, "--use_fast_math"],
        # The bindings encode TMA descriptors through the driver API.
        extra_ldflags=["-lcuda"],
        extra_include_paths=[CSRC_DIR],
        use_fast_math=False,
    ).build_and_load()


# ---------------------------------------------------------------------------
# Device scratch: control words and the stream-K partial workspace
# ---------------------------------------------------------------------------
#
# The programs use two kinds of mutable device scratch:
#
# * Control words: the stream-K arrival counters, which must read zero before a launch and which
#   every launch returns to zero. Each prepared runner takes its own slot from the device's slab;
#   a slot is never reused, so a CUDA Graph that captured a runner replays into valid private
#   words after the runner is dropped. One-shot eager calls share one slot per (device, stream,
#   word count): launches on one stream are stream-ordered.
# * Partials: the FP32 workspace of split plans, written before it is read inside one launch. An
#   eager launch uses the workspace of its (device, stream), grown to the largest split plan
#   launched there; a launch under CUDA Graph capture takes a block of the graph's memory pool.


class _CtrlSlab:
    """Zero-initialised control words of one device, handed out in 128-byte slots, never reused."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._lock = threading.Lock()
        self._chunk: torch.Tensor | None = None
        self._used = 0
        self._pinned: list[torch.Tensor] = []

    def take(self, words: int) -> torch.Tensor:
        """``words`` (a multiple of ``CTRL_SLOT_WORDS``) zero int32 words owned by the caller for good."""
        with self._lock:
            chunk = self._chunk
            if chunk is not None and self._used + words <= chunk.numel():
                slot = chunk[self._used : self._used + words]
                self._used += words
                return slot
            device = torch.device("cuda", self.index)
            if torch.cuda.is_current_stream_capturing():
                # No free slot inside a capture: zero fresh words as one node of that graph.
                slot = torch.zeros(words, dtype=torch.int32, device=device)
                self._pinned.append(slot)
                return slot
            chunk = torch.zeros(
                max(CTRL_CHUNK_WORDS, words), dtype=torch.int32, device=device
            )
            # Make the zero fill visible to every stream before any launch binds a slot.
            torch.cuda.synchronize(self.index)
            self._pinned.append(chunk)
            self._chunk, self._used = chunk, words
            return chunk[:words]


_LOCK = threading.Lock()
_CTRL_SLABS: dict[int, _CtrlSlab] = {}
_CTRL_LANES: dict[tuple[int, int, int], torch.Tensor] = {}
_READ_NEVER: dict[int, torch.Tensor] = {}
_WS_LANES: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
_FFI_DEVICES: dict[int, Any] = {}


def _raw_stream(index: int) -> int:
    return torch.cuda.current_stream(index).cuda_stream


def _ctrl_slab(index: int) -> _CtrlSlab:
    slab = _CTRL_SLABS.get(index)
    if slab is None:
        with _LOCK:
            slab = _CTRL_SLABS.setdefault(index, _CtrlSlab(index))
    return slab


def _control_words(index: int, words: int, *, lane: bool) -> torch.Tensor:
    """Zero int32 control words on ``cuda:index``: a fresh slot (``lane=False``), or the slot eager
    one-shot calls share per (device, current stream, word count) (``lane=True``, outside capture).
    """
    slot_words = -(-int(words) // CTRL_SLOT_WORDS) * CTRL_SLOT_WORDS
    if lane and not torch.cuda.is_current_stream_capturing():
        key = (index, _raw_stream(index), slot_words)
        slot = _CTRL_LANES.get(key)
        if slot is None:
            slab = _ctrl_slab(index)
            with _LOCK:
                slot = _CTRL_LANES.get(key)
                if slot is None:
                    slot = _CTRL_LANES[key] = slab.take(slot_words)
        return slot[:words]
    return _ctrl_slab(index).take(slot_words)[:words]


def _read_never(index: int) -> torch.Tensor:
    """The FP32 placeholder bound as ``ws`` by whole plans, which never touch it."""
    entry = _READ_NEVER.get(index)
    if entry is None:
        with _LOCK:
            entry = _READ_NEVER.get(index)
            if entry is None:
                entry = _READ_NEVER[index] = torch.empty(
                    1024, dtype=torch.float32, device=torch.device("cuda", index)
                )
    return entry


def _grow_ws_lane(index: int, stream: int, numel: int) -> tuple[int, torch.Tensor]:
    with _LOCK:
        lane = _WS_LANES.get((index, stream))
        if lane is None or lane[0] < numel:
            workspace = torch.empty(
                numel, dtype=torch.float32, device=torch.device("cuda", index)
            )
            lane = _WS_LANES[(index, stream)] = (numel, workspace)
    return lane


def _ffi_device(index: int) -> Any:
    import tvm_ffi

    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    return device


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KimiK3LatentFrontRunner:
    """The prepared launch of one front call.

    ``launch()`` submits the bound program on the current torch stream of the runner's device
    with no host synchronisation; the kernel reads every operand on device at launch, so the
    runner (or a CUDA Graph capturing it) replays for new values written into the same buffers.
    Prepare a new runner when a shape or a tensor binding changes. The runner owns its control
    words, which are never freed.
    """

    num_tokens: int
    sm_count: int
    key: str
    plan: dict[str, Any] = field(repr=False)
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    ws_numel: int = field(repr=False)
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = field(repr=False)
    device_index: int = field(repr=False)

    def _partial_ws(self, stream: int) -> torch.Tensor:
        """The partial workspace of a split-plan launch on ``stream``: the stream's lane when eager,
        a block of the graph's memory pool under capture."""
        if torch.cuda.is_current_stream_capturing():
            return torch.empty(
                self.ws_numel,
                dtype=torch.float32,
                device=torch.device("cuda", self.device_index),
            )
        lane = _WS_LANES.get((self.device_index, stream))
        if lane is None or lane[0] < self.ws_numel:
            lane = _grow_ws_lane(self.device_index, stream, self.ws_numel)
        return lane[1]

    def launch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import tvm_ffi

        index = self.device_index
        stream = _raw_stream(index)
        with torch.cuda.device(index), tvm_ffi.use_raw_stream(
            _ffi_device(index), stream
        ):
            if self.ws_numel:
                arguments = (
                    *self.arguments[:WS_POSITION],
                    self._partial_ws(stream),
                    *self.arguments[WS_POSITION + 1 :],
                )
            else:
                arguments = self.arguments
            self.entry(*arguments)
        return self.outputs

    __call__ = launch


def _check(
    t: torch.Tensor, shape: tuple[int, ...], name: str, dtype: torch.dtype
) -> None:
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}, "
            f"got {tuple(t.shape)} {t.dtype}"
        )


def _prepare(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
    *,
    lane: bool,
) -> KimiK3LatentFrontRunner:
    if x.dim() != 2:
        raise ValueError(f"x must be a [rows, {HIDDEN}] tensor, got {tuple(x.shape)}")
    M = int(x.shape[0])
    bf16 = torch.bfloat16
    _check(x, (M, HIDDEN), "x", bf16)
    _check(router_weight, (NUM_EXPERTS, HIDDEN), "router_weight", bf16)
    _check(routed_weight, (LATENT, HIDDEN), "routed_weight", bf16)
    _check(
        shared_gate_up_weight,
        (2 * SHARED_INTERMEDIATE, HIDDEN),
        "shared_gate_up_weight",
        bf16,
    )
    _check(logits, (M, NUM_EXPERTS), "logits", torch.float32)
    _check(latent, (M, LATENT), "latent", bf16)
    _check(shared_act, (M, SHARED_INTERMEDIATE), "shared_act", bf16)
    tensors = (
        x,
        router_weight,
        routed_weight,
        shared_gate_up_weight,
        logits,
        latent,
        shared_act,
    )
    devices = {t.device for t in tensors}
    if len(devices) != 1 or x.device.type != "cuda":
        raise ValueError(
            f"every operand must live on one CUDA device, got {sorted(map(str, devices))}"
        )
    index = _device_index(x.device)
    capability, sm_count = _device_facts(index)
    if capability not in SUPPORTED_COMPUTE_CAPABILITIES:
        raise ValueError(
            f"the Kimi-K3 front programs require compute capability 10.0 or 10.3, got {capability}"
        )
    plan = front_plan(M, sm_count)
    key = front_kernel_key(plan)
    with torch.cuda.device(index):
        counters = _control_words(index, arrival_counter_words(plan), lane=lane)
        kwargs = dict(
            A=x,
            WG=router_weight,
            WD=routed_weight,
            WS=shared_gate_up_weight,
            logits=logits,
            latent=latent,
            shared_act=shared_act,
            ws=_read_never(
                index
            ),  # a split plan's partial workspace replaces it at launch
            counters=counters,
            M=M,
            m_tiles=int(plan["m_tiles"]),
            num_items=int(plan["num_items"]),
            full_items=int(plan["full_items"]),
            sk_ipc=int(plan["sk_ipc"]),
            sk_max_seg=int(plan["sk_max_seg"]),
            sk_total=int(plan["sk_total"]),
            grid=(int(plan["grid"]), 1, 1),
        )
        assert tuple(kwargs) == FRONT_KWARGS
        module = _module(key)
    arguments = tuple(kwargs[name] for name in FRONT_KWARGS[:-1]) + kwargs["grid"]
    return KimiK3LatentFrontRunner(
        num_tokens=M,
        sm_count=sm_count,
        key=key,
        plan=plan,
        entry=module.run,
        arguments=arguments,
        ws_numel=partial_ws_numel(plan),
        outputs=(logits, latent, shared_act),
        device_index=index,
    )


def prepare_front(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> KimiK3LatentFrontRunner:
    """Validate the operands, plan the launch and bind it into a reusable runner.

    Args:
        x: Contiguous BF16 input ``[rows, 7168]``, at least 513 rows.
        router_weight: Contiguous BF16 router weight ``[896, 7168]``.
        routed_weight: Contiguous BF16 latent projection ``[3584, 7168]``.
        shared_gate_up_weight: Contiguous BF16 shared gate rows followed by up rows
            ``[12288, 7168]``.
        logits: Caller-owned FP32 output ``[rows, 896]``.
        latent: Caller-owned BF16 output ``[rows, 3584]``.
        shared_act: Caller-owned BF16 output ``[rows, 6144]``.

    Returns:
        A runner whose ``launch()`` fills the three outputs. The program is built and loaded
        and the runner's control words are taken here: prepare before CUDA Graph capture.
    """
    return _prepare(
        x,
        router_weight,
        routed_weight,
        shared_gate_up_weight,
        logits,
        latent,
        shared_act,
        lane=False,
    )


def front(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete front operator into the caller-owned outputs; arguments as :func:`prepare_front`.

    Eager calls share one set of control words per device, stream and plan size and the
    stream's partial workspace; a call under CUDA Graph capture takes fresh words, which are
    never freed, and a workspace block of the graph's memory pool.

    Returns:
        ``(logits, latent, shared_act)``.
    """
    return _prepare(
        x,
        router_weight,
        routed_weight,
        shared_gate_up_weight,
        logits,
        latent,
        shared_act,
        lane=True,
    )()
