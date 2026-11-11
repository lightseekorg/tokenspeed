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

"""Zero-overhead NVTX range wrapper.

Off by default. Enable via `--enable-nvtx` or `TOKENSPEED_NVTX=1`.
`nvtx_range(...)` is dual-purpose — context manager or decorator:

    @nvtx_range("forward", color="blue")
    def forward(self, ...): ...

    with nvtx_range("sampling", color="yellow"):
        backend.sample(...)

The `_enabled` check is re-read at each call, so a decorator applied at
import time still arms once the CLI flag flips the flag at runtime.
"""

from __future__ import annotations

import functools

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.utils.env import envs

NVTX_DOMAIN: str = "tokenspeed"
_enabled: bool = envs.TOKENSPEED_NVTX.get()


@functools.cache
def _nvtx_backend():
    if not current_platform().is_nvidia:
        return None, frozenset()
    try:
        import nvtx
        from nvtx.colors import _NVTX_COLORS
    except ModuleNotFoundError as error:
        if error.name != "nvtx":
            raise
        raise RuntimeError(
            "NVTX ranges require the tokenspeed-kernel[nvtx] extra"
        ) from error
    return nvtx, frozenset(color for color in _NVTX_COLORS if color is not None)


def set_nvtx_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


class _Range:
    __slots__ = ("name", "color", "category")

    def __init__(
        self,
        name: str,
        color: str,
        category: str | None,
    ) -> None:
        self.name = name
        self.color = color
        self.category = category

    def __enter__(self):
        if _enabled:
            nvtx, valid_colors = _nvtx_backend()
            if nvtx is not None:
                nvtx.push_range(
                    message=self.name,
                    color=self.color if self.color in valid_colors else "blue",
                    domain=NVTX_DOMAIN,
                    category=self.category,
                )
        return self

    def __exit__(self, *exc):
        if _enabled:
            nvtx, _ = _nvtx_backend()
            if nvtx is not None:
                nvtx.pop_range(domain=NVTX_DOMAIN)
        return False

    def __call__(self, func):
        name, color, category = self.name, self.color, self.category

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return func(*args, **kwargs)
            nvtx, valid_colors = _nvtx_backend()
            if nvtx is None:
                return func(*args, **kwargs)
            nvtx.push_range(
                message=name,
                color=color if color in valid_colors else "blue",
                domain=NVTX_DOMAIN,
                category=category,
            )
            try:
                return func(*args, **kwargs)
            finally:
                nvtx.pop_range(domain=NVTX_DOMAIN)

        return wrapper


def nvtx_range(
    name: str,
    *,
    color: str = "blue",
    category: str | None = None,
) -> _Range:
    return _Range(name, color, category)
