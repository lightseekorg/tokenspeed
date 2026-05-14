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

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.sampling.backends.base import (
    DEFAULT_RANDOM_SEED,
    SamplingBackend,
    SamplingBackendConfig,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs


_BACKEND_REGISTRY: dict[str, type[SamplingBackend]] = {}


def _get_default_backend_name() -> str:
    if current_platform().is_nvidia:
        return "flashinfer"
    return "greedy"


def _resolve_backend_name(server_args: ServerArgs) -> str:
    return server_args.sampling_backend or _get_default_backend_name()


def register_backend(name: str, cls: type[SamplingBackend]) -> None:
    _BACKEND_REGISTRY[name] = cls


def create_sampling_backend(
    server_args: ServerArgs,
    *,
    max_bs: int,
    max_draft_tokens_per_req: int,
    device: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_req_pool_size: int = 0,
    vocab_size: int = 0,
    tp_group: tuple[int, ...] | None = None,
) -> SamplingBackend:
    # Trigger concrete-backend registration on first use.
    from tokenspeed.runtime.sampling.backends import flashinfer as _fi  # noqa: F401
    from tokenspeed.runtime.sampling.backends import (  # noqa: F401
        flashinfer_full as _ff,
    )
    from tokenspeed.runtime.sampling.backends import greedy as _g  # noqa: F401

    name = _resolve_backend_name(server_args)
    if name not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown sampling backend: {name!r}. "
            f"Available: {list(_BACKEND_REGISTRY)}"
        )
    cls = _BACKEND_REGISTRY[name]

    return cls(
        SamplingBackendConfig.from_server_args(
            server_args,
            max_bs=max_bs,
            max_draft_tokens_per_req=max_draft_tokens_per_req,
            device=device,
            random_seed=random_seed,
            max_req_pool_size=max_req_pool_size,
            vocab_size=vocab_size,
            tp_group=tp_group,
        )
    )
