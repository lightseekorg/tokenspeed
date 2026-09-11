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

"""DeepSeek V4.1 Engram, with caller-owned history and FP8-resident tables.

Integration contract:
* Construct one EngramHashState from the text config and the actual tokenizer.
  Pass raw previous-three IDs, newest first, alongside every current token.
  -1 denotes a sequence boundary or a nonparticipating (e.g. image) token.
* Input preparation owns these small windows, including pending overlap tokens
  and the current speculative branch. Refresh pointer-stable input buffers for
  eager and graph forwards alike; never put request tensors in ForwardContext.
* Hidden states, hashes and masks must be replicated within mapping.attn.tp_group.
  GPU-resident tables are row-sharded across attention TP and all-reduced.
  Host tables stay in anonymous or shared host memory and gather through UVA.
  ``shared`` keeps one full copy per node and skips the lookup all-reduce.
  ``sharded`` keeps a host shard per attention-TP rank and retains the all-reduce.
  ``auto`` selects sharded when attention TP > 1. Output matches the GPU path.
* Intercept embed.weight/scale BEFORE a generic GPU weight iterator. Open the
  safetensors file on CPU, pass get_slice(name) to embed.load_sharded(), and mark
  the corresponding runtime parameter name loaded. GPU and host-sharded tables
  copy local rows; shared host tables copy every row on the writer rank.
  Do not cast the model wholesale to BF16: table codes/scales must stay bytes.
* wkv uses the existing quantized ReplicatedLinear and its normal post-load
  processing. Its 32x32 scales are repeated across output rows as 1x32 MXFP8
  scales, losslessly; no weight values are expanded or requantized.
  checkpoint_weight_aliases() maps raw checkpoint names, including
  wkv.scale, without renaming the embedding's per-row embed.scale.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sympy import isprime
from tokenizers import Regex, normalizers
from tokenspeed_kernel.ops.embedding import host_gather
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config, Mxfp8Config
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.utils.env import global_server_args_dict

DEAD_TOKEN_ID = -1
_ENGRAM_EMBED_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")


def is_engram_embed_checkpoint_name(name: str) -> bool:
    """True for the two huge FP8 table tensors that must not be load_file'd."""
    return name.endswith(_ENGRAM_EMBED_SUFFIXES)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Return raw-ID to normalized-ID lookup and its multiplier-defining size.

    ``tokenizer`` must expose the fast ``backend_tokenizer``; decoding deliberately
    includes special tokens and does not use HF whitespace cleanup.
    """
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_id: dict[str, int] = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in text
            else normalizer.normalize_str(text) or text
        )
        lookup.append(key_to_id.setdefault(key, len(key_to_id)))
    return lookup, len(key_to_id)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """Return CPU int64 [layers, lookbacks] odd multipliers, using reference RNG."""
    if not layer_ids or max_ngram_size < 2 or tokenizer_vocab_size < 1:
        raise ValueError(
            "Engram requires layers, at least two lookbacks and a vocabulary"
        )
    bound = max(1, (np.iinfo(np.int64).max // tokenizer_vocab_size) // 2)
    return torch.from_numpy(
        np.stack(
            [
                np.random.default_rng(10007 * layer_id).integers(
                    low=0, high=bound, size=(max_ngram_size,), dtype=np.int64
                )
                * 2
                + 1
                for layer_id in layer_ids
            ]
        )
    )


@dataclass(frozen=True)
class EngramLayout:
    """Prime bucket geometry in checkpoint layer/ngram/head order."""

    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    primes: tuple[tuple[tuple[int, ...], ...], ...]

    @classmethod
    def from_config(cls, config) -> EngramLayout:
        """Build and validate geometry from a V4.1 text config, not its wrapper."""
        layer_ids = tuple(config.engram_layer_ids)
        rows = tuple(config.engram_num_embeddings)
        if (
            not layer_ids
            or len(set(layer_ids)) != len(layer_ids)
            or len(rows) != len(layer_ids)
            or config.engram_max_ngram_size != 4
            or config.engram_n_heads < 1
            or config.engram_vocab_size < 2
            or config.engram_head_dim < 32
            or config.engram_head_dim % 32
        ):
            raise ValueError(
                "Invalid V4.1 Engram geometry (requires max ngram=4 and 32-wide blocks)"
            )
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(config.engram_max_ngram_size - 1):
                current, sizes = config.engram_vocab_size - 1, []
                for _ in range(config.engram_n_heads):
                    current += 1
                    while current in seen or not isprime(current):
                        current += 1
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        expected_rows = tuple(sum(sum(heads) for heads in layer) for layer in primes)
        if rows != expected_rows:
            raise ValueError(
                f"Engram table rows {rows} do not match prime buckets {expected_rows}"
            )
        return cls(
            layer_ids=layer_ids,
            num_embeddings=rows,
            max_ngram_size=config.engram_max_ngram_size,
            n_heads=config.engram_n_heads,
            head_dim=config.engram_head_dim,
            primes=tuple(primes),
        )


def build_engram_previous_tokens(
    token_histories: Sequence[Sequence[int]],
    request_indices: Sequence[int],
    positions: Sequence[int],
) -> torch.Tensor:
    """Return CPU int64 [tokens, 3] raw lookbacks, newest first.

    Histories are existing physical-position request histories (prompt + accepted
    output), with nonparticipating positions marked -1. For verification, supply
    the proposed branch as well. Request indices and positions describe each
    packed forward token, so reordering, prefix hits and chunking need no state.
    No history is mutated or copied in full. The forward thread transfers this
    small result; callers must overlay pending GPU-owned overlap/draft IDs there.
    """
    if len(request_indices) != len(positions):
        raise ValueError("Engram request indices and positions must have equal lengths")
    windows = []
    for request_index, position in zip(request_indices, positions, strict=True):
        if not 0 <= request_index < len(token_histories):
            raise ValueError(f"Invalid Engram history request index {request_index}")
        history = token_histories[request_index]
        if not 0 <= position <= len(history):
            raise ValueError(
                f"Engram history is missing tokens before position {position}"
            )
        window = [
            history[position - shift] if position >= shift else DEAD_TOKEN_ID
            for shift in (1, 2, 3)
        ]
        if any(token_id < DEAD_TOKEN_ID for token_id in window):
            raise ValueError("Mark nonparticipating history tokens with -1")
        windows.append(window)
    return torch.tensor(windows, dtype=torch.int64, device="cpu").reshape(-1, 3)


class EngramHashState(nn.Module):
    """Immutable tokenizer/hash constants only; no per-request token cache."""

    def __init__(self, config, tokenizer, device: torch.device | str):
        super().__init__()
        self.layout = EngramLayout.from_config(config)
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"Engram compressed vocabulary: got {vocab_size}, expected {config.engram_compressed_vocab_size}"
            )
        if not 0 <= config.engram_pad_token_id < len(token_map):
            raise ValueError("Engram pad token is outside the tokenizer vocabulary")
        self.pad_id = token_map[config.engram_pad_token_id]
        primes = torch.tensor(self.layout.primes, dtype=torch.int64, device=device)
        flat = primes.flatten(1)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", flat.cumsum(-1) - flat, persistent=False)
        self.register_buffer(
            "token_map",
            torch.tensor(token_map, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(self.layout.layer_ids, 4, vocab_size).to(
                device=device
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        previous_token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hash raw IDs [...] and lookbacks [..., 3] to [..., layers, hash columns].

        ``token_mask`` is bool [...] (False for padding/images). Past barriers and
        sequence starts are -1 in ``previous_token_ids``. Real IDs must be in the
        tokenizer vocabulary; callers must not pass clamped image placeholder IDs
        without their mask. All tensors live on the constants' device.
        """
        if (
            previous_token_ids.shape != (*input_ids.shape, 3)
            or token_mask.shape != input_ids.shape
        ):
            raise ValueError(
                "Engram IDs, previous-three window and mask shapes disagree"
            )
        if (
            token_mask.dtype != torch.bool
            or input_ids.dtype not in (torch.int32, torch.int64)
            or previous_token_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise TypeError("Engram expects int32/int64 IDs and a bool token mask")
        raw = torch.cat((input_ids.unsqueeze(-1), previous_token_ids), dim=-1).long()
        dead = raw == DEAD_TOKEN_ID
        dead[..., 0] |= ~token_mask
        # Replace BEFORE indexing: nonparticipating current IDs can be out of vocab.
        mapped = self.token_map[raw.masked_fill(dead, 0)]
        blocked = dead.long().cumsum(-1) > 0
        tokens = mapped.masked_fill(blocked, self.pad_id)
        products = tokens.unsqueeze(-2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for shift in range(1, 4):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, shift - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


def _host_table_dir(nbytes: int) -> str:
    """Pick a writable directory with room for one Engram host-table file.

    ``/dev/shm`` is preferred when it is large enough. Docker often caps it
    at 32–64 GiB, which cannot hold a V4.1 table, so fall back to scratch.
    """
    explicit = global_server_args_dict.get("engram_host_table_dir")
    if explicit:
        return explicit
    margin = 1 << 30
    candidates = []
    if os.path.isdir("/dev/shm"):
        candidates.append("/dev/shm")
    for path in ("/scratch", "/tmp", tempfile.gettempdir()):
        if path not in candidates:
            candidates.append(path)
    for directory in candidates:
        if not os.path.isdir(directory) or not os.access(directory, os.W_OK):
            continue
        if shutil.disk_usage(directory).free >= nbytes + margin:
            return directory
    raise OSError(
        f"No filesystem with {nbytes + margin} free bytes for an Engram host table; "
        "pass --engram-host-table-dir"
    )


_HOST_TABLE_JOB_ID: str | None = None
_CREATED_HOST_FILES: list[str] = []


def _host_table_job_id() -> str:
    """Return a per-job id shared by every rank, unique across concurrent jobs."""
    global _HOST_TABLE_JOB_ID
    if _HOST_TABLE_JOB_ID is not None:
        return _HOST_TABLE_JOB_ID
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        payload = [str(os.getpid())]
        torch.distributed.broadcast_object_list(payload, src=0)
        _HOST_TABLE_JOB_ID = payload[0]
    else:
        _HOST_TABLE_JOB_ID = str(os.getpid())
    return _HOST_TABLE_JOB_ID


def _engram_host_table_path(layer_id: int, field: str, nbytes: int) -> str:
    return os.path.join(
        _host_table_dir(nbytes),
        f"ts-engram-{_host_table_job_id()}-L{layer_id}-{field}",
    )


def _is_host_table_writer(mapping: Mapping) -> bool:
    local = os.environ.get("LOCAL_RANK")
    if local is not None:
        return int(local) == 0
    return mapping.rank % max(mapping.nprocs_per_node, 1) == 0


def resolve_engram_host_layout(host_table: bool, tp_size: int) -> str:
    """Return ``gpu``, ``shared``, or ``sharded`` from the CLI opt-in and TP."""
    if not host_table:
        return "gpu"
    if tp_size < 1:
        raise ValueError("Engram host layout requires a positive attention TP size")
    requested = global_server_args_dict["engram_host_table_layout"]
    if requested == "auto":
        return "sharded" if tp_size > 1 else "shared"
    if requested in ("shared", "sharded"):
        return requested
    raise ValueError(
        "engram_host_table_layout must be auto, shared, or sharded; "
        f"got {requested!r}"
    )


def _cleanup_host_files() -> None:
    while _CREATED_HOST_FILES:
        path = _CREATED_HOST_FILES.pop()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


atexit.register(_cleanup_host_files)


def _advise_hugepages(buffer, nbytes: int) -> None:
    """Best-effort Linux THP hint. Shared file maps often ignore it."""
    if sys.platform != "linux" or nbytes <= 0:
        return
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return
    libc = ctypes.CDLL(libc_name)
    madvise = getattr(libc, "madvise", None)
    if madvise is None:
        return
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    # Linux MADV_HUGEPAGE=14. Failure is ignored: THP may be disabled.
    madvise(ctypes.c_void_p(int(buffer.ctypes.data)), ctypes.c_size_t(nbytes), 14)


def _create_host_file(path: str, nbytes: int) -> None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, nbytes)
    finally:
        os.close(fd)
    _CREATED_HOST_FILES.append(path)


def _unlink_host_file(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    try:
        _CREATED_HOST_FILES.remove(path)
    except ValueError:
        pass


def _host_table_nbytes(shape: tuple[int, int], dtype: torch.dtype) -> int:
    return int(shape[0] * shape[1] * torch.empty((), dtype=dtype).element_size())


def _tensor_from_host_bytes(
    buffer, shape: tuple[int, int], dtype: torch.dtype
) -> torch.Tensor:
    tensor = torch.frombuffer(buffer, dtype=torch.uint8)
    if dtype != torch.uint8:
        tensor = tensor.view(dtype)
    return tensor.view(shape)


def _allocate_host_bytes(
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device | str,
    shared_file: bool,
    path: str,
) -> tuple[torch.Tensor, object | None]:
    """Return a host uint8/fp8 table and a keep-alive for the backing store.

    Shared multi-rank tables map a job-scoped file. Rank-local host shards use
    an anonymous buffer so Linux transparent huge pages can apply.
    """
    device = torch.device(device)
    if device.type == "meta":
        return torch.empty(shape, dtype=dtype, device="meta"), None
    nbytes = _host_table_nbytes(shape, dtype)
    if shared_file:
        mmap = np.memmap(path, dtype=np.uint8, mode="r+", shape=(nbytes,))
        _advise_hugepages(mmap, nbytes)
        return _tensor_from_host_bytes(mmap, shape, dtype), mmap
    buffer = np.empty(nbytes, dtype=np.uint8)
    _advise_hugepages(buffer, nbytes)
    return _tensor_from_host_bytes(buffer, shape, dtype), buffer


def _dequant_fp8_e8m0(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = codes.view(torch.uint8).view(torch.float8_e4m3fn).float()
    scale = scales.view(torch.uint8).view(torch.float8_e8m0fnu).float()
    return (
        (values.unflatten(-1, (-1, 32)) * scale.unsqueeze(-1))
        .flatten(-2)
        .to(torch.bfloat16)
    )


class RowShardedEngramEmbedding(nn.Module):
    """FP8 codes and raw E8M0 scale bytes, GPU-sharded or host-resident."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mapping: Mapping,
        device: torch.device | str,
        host_table: bool,
        host_layout: str,
        layer_id: int,
    ):
        super().__init__()
        if (
            num_embeddings < mapping.attn.tp_size
            or embedding_dim < 32
            or embedding_dim % 32
        ):
            raise ValueError(
                "Engram embedding requires rows per TP rank and 32-wide scale blocks"
            )
        if host_table:
            if host_layout not in ("shared", "sharded"):
                raise ValueError(
                    "Host Engram tables require host_layout shared or sharded"
                )
        elif host_layout != "gpu":
            raise ValueError("GPU Engram tables require host_layout='gpu'")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_group = mapping.attn.tp_group
        self.host_table = host_table
        self.host_layout = host_layout
        self.layer_id = layer_id
        self._mmap_keep: list[object] = []
        self._host_gpu_registered = False
        row_sharded = (not host_table) or host_layout == "sharded"
        if row_sharded:
            self.part_num_embeddings = (
                num_embeddings + mapping.attn.tp_size - 1
            ) // mapping.attn.tp_size
            self.row_start = mapping.attn.tp_rank * self.part_num_embeddings
            self.row_end = min(
                self.row_start + self.part_num_embeddings, num_embeddings
            )
        else:
            self.part_num_embeddings = num_embeddings
            self.row_start = 0
            self.row_end = num_embeddings
        shared_file = (
            host_layout == "shared"
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        self._shared_writer = (not shared_file) or _is_host_table_writer(mapping)
        if host_table:
            weight_path = ""
            scale_path = ""
            weight_shape = (self.part_num_embeddings, embedding_dim)
            scale_shape = (self.part_num_embeddings, embedding_dim // 32)
            if shared_file:
                weight_bytes = _host_table_nbytes(weight_shape, torch.float8_e4m3fn)
                scale_bytes = _host_table_nbytes(scale_shape, torch.uint8)
                weight_path = _engram_host_table_path(layer_id, "weight", weight_bytes)
                scale_path = _engram_host_table_path(layer_id, "scale", scale_bytes)
                if self._shared_writer:
                    _create_host_file(weight_path, weight_bytes)
                    _create_host_file(scale_path, scale_bytes)
                torch.distributed.barrier()
            weight, weight_mmap = _allocate_host_bytes(
                weight_shape,
                torch.float8_e4m3fn,
                device,
                shared_file,
                weight_path,
            )
            scale, scale_mmap = _allocate_host_bytes(
                scale_shape,
                torch.uint8,
                device,
                shared_file,
                scale_path,
            )
            for keep in (weight_mmap, scale_mmap):
                if keep is not None:
                    self._mmap_keep.append(keep)
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.scale = nn.Parameter(scale, requires_grad=False)
            if shared_file:
                torch.distributed.barrier()
                if self._shared_writer:
                    _unlink_host_file(weight_path)
                    _unlink_host_file(scale_path)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    embedding_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    embedding_dim // 32,
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )
        for param in (self.weight, self.scale):
            param.weight_loader = self.weight_loader
            param.engram_row_sharded = row_sharded
            param.engram_host_table = host_table
            param.engram_row_start = self.row_start
            param.engram_row_end = self.row_end
        local_rows = self.row_end - self.row_start
        if torch.device(device).type != "meta":
            self.weight.data[local_rows:].zero_()
            self.scale.data[local_rows:].fill_(127)
            self._register_host_for_gpu()

    def _register_host_for_gpu(self) -> None:
        """cudaHostRegister once. Gather kernels must not register."""
        if self._host_gpu_registered or not self.host_table:
            return
        if (
            torch.device(self.weight.device).type != "cpu"
            or not torch.cuda.is_available()
        ):
            self._host_gpu_registered = True
            return
        platform = current_platform()
        weight_bytes = self.weight.view(torch.uint8)
        if not weight_bytes.is_pinned():
            platform.register_host_tensor_for_gpu_access(weight_bytes)
        if not self.scale.is_pinned():
            platform.register_host_tensor_for_gpu_access(self.scale)
        self._host_gpu_registered = True

    def _apply(self, fn, recurse=True):
        if self.host_table:
            return self
        return super()._apply(fn, recurse)

    def _copy_rows(
        self, param: nn.Parameter, rows: torch.Tensor, local_start: int
    ) -> None:
        if param is self.scale:
            if rows.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise TypeError(
                    "Engram scales must be E8M0 or raw uint8 exponent bytes"
                )
            rows = rows.view(torch.uint8)
        elif rows.dtype != torch.float8_e4m3fn:
            raise TypeError("Engram table weights must stay FP8 E4M3")
        param.data[local_start : local_start + rows.shape[0]].copy_(rows)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load CPU full-table views or already-local tensors without dequantization.

        The streaming loader should prefer load_sharded(): this callback cannot
        undo a full-table allocation already made by a generic weight iterator.
        """
        if param is not self.weight and param is not self.scale:
            raise ValueError("Parameter does not belong to this Engram table")
        if loaded_weight.ndim != 2 or loaded_weight.shape[1] != param.shape[1]:
            raise ValueError("Engram table weight has the wrong row width")
        rows = loaded_weight.shape[0]
        local_rows = self.row_end - self.row_start
        if rows == self.num_embeddings:
            if loaded_weight.device.type != "cpu" and len(self.tp_group) > 1:
                raise ValueError(
                    "Do not materialize a full Engram table on GPU; use load_sharded"
                )
            loaded_weight = loaded_weight[self.row_start : self.row_end]
        elif rows in (local_rows, self.part_num_embeddings):
            loaded_weight = loaded_weight[:local_rows]
        else:
            raise ValueError(
                "Engram weight is neither a full table nor this rank's row shard"
            )
        self._copy_rows(param, loaded_weight, 0)

    def load_sharded(self, parameter_name: str, tensor_slice, chunk_rows: int) -> None:
        """Copy table rows from a CPU safetensors get_slice() in bounded chunks.

        GPU and host-sharded tables copy only this rank's rows. Shared host
        tables copy every row on the writer rank; followers map the file.
        Args: parameter_name is 'weight' or 'scale'; tensor_slice is the global
        checkpoint tensor's lazy CPU slice; chunk_rows bounds transient host
        storage. No full tensor, remote shard, or BF16 table is allocated.
        Returns: None; the caller records the corresponding parameter as loaded.
        """
        if parameter_name not in ("weight", "scale") or chunk_rows < 1:
            raise ValueError("Expected weight/scale and a positive row chunk size")
        if self.host_table and not self._shared_writer:
            return
        param = getattr(self, parameter_name)
        if tuple(tensor_slice.get_shape()) != (self.num_embeddings, param.shape[1]):
            raise ValueError("Checkpoint Engram shape does not match the global table")
        for start in range(self.row_start, self.row_end, chunk_rows):
            end = min(start + chunk_rows, self.row_end)
            rows = tensor_slice[start:end, :]
            if rows.device.type != "cpu":
                raise ValueError("Open Engram safetensors on CPU before slicing")
            self._copy_rows(param, rows, start - self.row_start)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Gather [..., hash columns] IDs into BF16 [..., hash columns, head dim].

        All TP peers must pass identical indices. Shared host tables gather the
        full copy and skip the all-reduce. GPU and host-sharded tables expand
        only local rows and reduce. Dequantization uses the same FP8/E8M0
        torch casts in both paths.
        """
        local = (indices >= self.row_start) & (indices < self.row_end)
        local_ids = (indices - self.row_start).masked_fill(~local, 0)
        if self.host_table:
            codes = host_gather.uint8_row_gather(
                self.weight.view(torch.uint8), local_ids, None
            )
            scales = host_gather.uint8_row_gather(self.scale, local_ids, None)
            values = _dequant_fp8_e8m0(codes, scales)
        else:
            local_ids = local_ids.long()
            # Byte gathers work on CPU and CUDA, including dtypes without index kernels.
            values = _dequant_fp8_e8m0(
                self.weight.view(torch.uint8)[local_ids], self.scale[local_ids]
            )
        values = values.masked_fill(~local.unsqueeze(-1), 0)
        if len(self.tp_group) > 1 and self.host_layout != "shared":
            values = all_reduce(
                values,
                group=self.tp_group,
                backend=None,
                op=torch.distributed.ReduceOp.SUM,
            )
        return values


def _load_projection_scale(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    if loaded_weight.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError("Engram wkv.scale must contain E8M0 exponents")
    if loaded_weight.shape != ((param.shape[0] + 31) // 32, param.shape[1]):
        raise ValueError("Engram wkv.scale must use checkpoint 32x32 blocks")
    codes = loaded_weight.view(torch.uint8)
    # Existing MXFP8 linear kernels accept per-row scales. Repeating exponents
    # exactly preserves the checkpoint's 32x32 values without requantization.
    values = codes.repeat_interleave(32, dim=0)[: param.shape[0]]
    default_weight_loader(param, values)


class DeepseekV41Engram(nn.Module):
    """Reference normalized-dot, signed-sqrt gate over four hyperconnection copies."""

    def __init__(
        self,
        config,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        device: torch.device | str,
        host_table: bool,
        host_layout: str,
    ):
        """Construct from text config, attention mapping and the model's quant config.

        prefix is the runtime module path (e.g. model.layers.1.engram). The
        checkpoint's wkv is [hidden_size * (hc_mult + 1), n_hash_cols * head_dim],
        replicated and quantized through the existing linear implementation.
        """
        super().__init__()
        layout = EngramLayout.from_config(config)
        if config.hc_mult != 4:
            raise ValueError("V4.1 Engram requires four hyperconnection copies")
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.prefix = prefix
        self.n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        if quant_config is not None:
            if (
                not isinstance(quant_config, Fp8Config)
                or not quant_config.is_checkpoint_fp8_serialized
                or quant_config.weight_block_size != [32, 32]
                or quant_config.scale_fmt != "ue8m0"
            ):
                raise ValueError(
                    "Engram wkv requires checkpoint FP8 with 32x32 E8M0 scales"
                )
            quant_config = Mxfp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                ignored_layers=quant_config.ignored_layers,
                weight_block_size=[1, 32],
                scale_fmt="ue8m0",
            )
        with torch.device(device):
            self.wkv = ReplicatedLinear(
                input_size=self.n_hash_cols * layout.head_dim,
                output_size=self.dim * (self.hc_mult + 1),
                bias=False,
                skip_bias_add=False,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.wkv",
            )
            self.q_weight = nn.Parameter(
                torch.ones(self.hc_mult, self.dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.k_weight = nn.Parameter(
                torch.ones(self.hc_mult, self.dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
        if quant_config is not None and self.wkv.weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "Do not exclude Engram wkv from checkpoint FP8 quantization"
            )
        self.embed = RowShardedEngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            mapping,
            device,
            host_table,
            host_layout,
            layer_id,
        )
        self.q_weight.weight_loader = default_weight_loader
        self.k_weight.weight_loader = default_weight_loader
        if hasattr(self.wkv, "weight_scale_inv"):
            self.wkv.weight_scale_inv._weight_loader = _load_projection_scale

    def checkpoint_weight_aliases(self) -> dict[str, str]:
        """Return raw layers.<id>.engram checkpoint names to runtime param paths.

        Use these before the generic '.scale' rename. Embedding params expose
        engram_row_sharded/row_start/row_end attributes for streaming dispatch.
        """
        names = {
            "embed.weight": "embed.weight",
            "embed.scale": "embed.scale",
            "wkv.weight": "wkv.weight",
            "q_weight": "q_weight",
            "k_weight": "k_weight",
        }
        if hasattr(self.wkv, "weight_scale_inv"):
            names["wkv.scale"] = "wkv.weight_scale_inv"
        return {
            f"layers.{self.layer_id}.engram.{source}": f"{self.prefix}.{target}"
            for source, target in names.items()
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inject Engram into [..., 4, hidden_size], returning the same shape/dtype.

        hash_ids contains this layer's [..., n_hash_cols] IDs; token_mask is bool
        [...]. False positions pass through unchanged, including image/pad rows.
        Norms and gates accumulate in FP32, independently for each HC copy.
        """
        shape = hidden_states.shape[:-2]
        if (
            hidden_states.shape[-2:] != (self.hc_mult, self.dim)
            or hash_ids.shape != (*shape, self.n_hash_cols)
            or token_mask.shape != shape
        ):
            raise ValueError("Engram hidden states, hashes and mask shapes disagree")
        if token_mask.dtype != torch.bool:
            raise TypeError("Engram token mask must be bool")
        if hidden_states.numel() == 0:
            return hidden_states
        embeddings = self.embed(hash_ids).flatten(-2)
        kv, _ = self.wkv(embeddings, block_scale=None, output_dtype=None)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        h = hidden_states.float()
        weight = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(
            hidden_states.dtype
        )
