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

"""Gated-boundary test cases, mailbox arming, and the sentinel-wait child."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from tokenspeed_kernel.ops.residual import residual_gate_coefficients

_WAITER = """
import torch
from tokenspeed_kernel.ops.moe._cute_dsl.latent_down import HALF_POLLED_SENTINEL

device = torch.device("cuda", 0)
{setup}
words = mailbox.view(torch.int32)
def arm():
    words.fill_(-1)
    mailbox[:1].copy_(row)
# Compile and prove the arrived case on a full row, so the window below is the wait alone.
arm()
consume()
torch.cuda.synchronize(device)
arm()
words[0, {word}] = {empty}
torch.cuda.synchronize(device)
open({marker!r}, "w").close()
consume()
torch.cuda.synchronize(device)
print("COMPLETED", flush=True)
"""


def boundary_case(boundary: str, rows: int, device: str) -> tuple:
    """Triton boundary arguments up to the pre-norm offset for a named boundary."""
    torch.manual_seed(17)
    width = 8192
    stream = torch.randn(rows, width, device=device) * 3.0
    branch = (torch.randn(rows, width, device=device) * 0.5).to(torch.bfloat16)
    alpha, beta = residual_gate_coefficients(
        torch.randn(width, device=device) * 3.0, 0.3, 1e-3
    )
    weights = [
        torch.randn(width, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(2)
    ]
    post_weight, pre_weight = {
        "attention": (None, weights[1]),
        "ffn": (weights[0], None),
        "ffn_linked": (weights[0], weights[1]),
    }[boundary]
    return (
        stream,
        branch,
        post_weight,
        1e-8,
        1.0,
        alpha,
        beta,
        pre_weight,
        1e-5,
        1.0,
    )


def armed_mailbox(rows: torch.Tensor, capacity: int) -> torch.Tensor:
    """A mailbox armed with the half-polled sentinel holding ``rows`` first."""
    mailbox = torch.empty(
        capacity, rows.shape[1], dtype=torch.bfloat16, device=rows.device
    )
    mailbox.view(torch.int32).fill_(-1)
    mailbox[: rows.shape[0]].copy_(rows)
    return mailbox


def assert_child_keeps_waiting(setup: str, word: int, empty: int) -> None:
    """Assert a child consuming a word with one sentinel half left never finishes."""
    # ``setup`` defines ``mailbox`` (two rows), ``row`` and ``consume()``; streams serialize in-process.
    with tempfile.TemporaryDirectory() as scratch:
        marker = Path(scratch) / "reached"
        script = _WAITER.format(setup=setup, word=word, empty=empty, marker=str(marker))
        child = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # The marker separates startup and compile from the wait under test.
        for _ in range(900):
            if marker.exists() or child.poll() is not None:
                break
            time.sleep(0.1)
        assert marker.exists() or child.poll() is not None, "the child never started"
        try:
            out, err = child.communicate(timeout=6)
        except subprocess.TimeoutExpired:
            return  # still spinning on the empty half, which is the contract
        finally:
            child.kill()
        assert "COMPLETED" not in out, "the consumer took a word that had not arrived"
        assert child.returncode == 0, f"the child failed otherwise: {err[-2000:]}"
