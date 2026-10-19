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

"""Retry helper for Hugging Face Hub rate limits.

Keep this module free of tokenspeed imports and heavy dependencies so every
entry point (CLI, scheduler, config/tokenizer loading, tests) can use it
without pulling in the runtime stack.
"""

import logging
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from huggingface_hub.utils import HfHubHTTPError

if TYPE_CHECKING:
    # Only annotated, never touched at runtime: the response type differs
    # between huggingface_hub releases (httpx on 1.x, requests before), and
    # httpx is not a declared tokenspeed dependency. The body duck-types
    # .status_code and .headers.get, which both response classes provide.
    import httpx

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Anonymous Hugging Face API access is bucketed per egress IP (a few hundred
# requests per five-minute window). Data/expert-parallel startup fans out one
# model loader per rank, so a single server start can drain that bucket and
# the hub answers 429. Wait out the window instead of failing engine startup;
# the engine startup timeout still bounds the total wait across calls.
HF_RATE_LIMIT_RETRY_BUDGET_S = 900.0
HF_RATE_LIMIT_BASE_DELAY_S = 5.0
HF_RATE_LIMIT_MAX_DELAY_S = 300.0


def call_with_hf_rate_limit_retry(description: str, call: Callable[[], _T]) -> _T:
    """Run ``call``, retrying Hugging Face Hub 429 rate limits.

    The hub answers an exhausted rate-limit window with 429 and usually a
    ``Retry-After`` hint; without the hint, back off exponentially from
    ``HF_RATE_LIMIT_BASE_DELAY_S`` up to ``HF_RATE_LIMIT_MAX_DELAY_S``. Once
    waiting again would exceed ``HF_RATE_LIMIT_RETRY_BUDGET_S``, re-raise the
    last error so a hard quota failure still surfaces instead of hanging.
    Any non-429 error propagates immediately.

    Args:
        description (str): What ``call`` is doing, embedded in the retry
            log warning.
        call (Callable[[], _T]): The zero-argument callable to run.

    Returns:
        _T: The result of ``call``.

    Raises:
        HfHubHTTPError: Immediately for any non-429 hub error, and for a 429
            once waiting again would exceed ``HF_RATE_LIMIT_RETRY_BUDGET_S``.
    """
    waited_s = 0.0
    attempt = 0
    while True:
        try:
            return call()
        except HfHubHTTPError as exc:
            response = exc.response
            if response is None or response.status_code != 429:
                raise
            retry_after_s = _retry_after_s(response)
            delay_s = (
                retry_after_s
                if retry_after_s is not None
                else HF_RATE_LIMIT_BASE_DELAY_S * (2**attempt)
            )
            delay_s = min(delay_s, HF_RATE_LIMIT_MAX_DELAY_S) * random.uniform(
                1.0, 1.25
            )
            if waited_s + delay_s > HF_RATE_LIMIT_RETRY_BUDGET_S:
                raise
            waited_s += delay_s
            attempt += 1
            logger.warning(
                f"Hugging Face rate limit (429) while {description}; "
                f"retrying in {delay_s:.1f}s "
                f"({HF_RATE_LIMIT_RETRY_BUDGET_S - waited_s:.0f}s of retry budget left)",
            )
            time.sleep(delay_s)


def _retry_after_s(response: "httpx.Response") -> float | None:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        seconds = float(retry_after)
    except ValueError:
        # The hub sends seconds; ignore the HTTP-date form and back off instead.
        return None
    return seconds if seconds > 0 else None
