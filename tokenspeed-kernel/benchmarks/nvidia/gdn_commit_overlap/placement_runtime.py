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

"""Benchmark-only placement controls for the real serving GDN commit.

A one-byte shared-memory control is changed only while HTTP serving is idle.
All ranks prepare the same choices before graph capture. Only GDN's existing
post-replay launch changes stream; target and captured draft nodes keep their
original execution contexts, algorithms, pointers and scheduling dependencies.
"""

import json
import mmap
from functools import wraps
from pathlib import Path

import torch
from graph_split import install as install_graph_split
from placement import PLACEMENTS, CommitPlacement


class PlacementControl:
    def __init__(self, *, control_path: Path):
        self.file = control_path.open("r+b")
        self.control = mmap.mmap(self.file.fileno(), 1)
        self.names = tuple(PLACEMENTS)
        self.placements = {
            name: CommitPlacement(spec=spec) for name, spec in PLACEMENTS.items()
        }
        self.previous_name: str | None = None
        print(
            f"GDN_COMMIT_PLACEMENTS_READY {json.dumps([p.metadata() for p in self.placements.values()])}",
            flush=True,
        )

    def launch(self, payload, parameters, **arguments):
        index = self.control[0]
        if index >= len(self.names):
            raise RuntimeError(f"Invalid benchmark placement index {index}")
        name = self.names[index]
        placement = self.placements[name]
        if name != self.previous_name:
            print(
                f"GDN_COMMIT_PLACEMENT_SWITCH {json.dumps(placement.metadata())}",
                flush=True,
            )
            self.previous_name = name
        with torch.cuda.nvtx.range(f"gdn_commit_placement={name}"):
            return placement.launch(payload, parameters, **arguments)

    def close(self):
        for placement in self.placements.values():
            placement.close()
        self.control.close()
        self.file.close()


_CONTROL: PlacementControl | None = None
_INSTALLED = False


def install(*, control_path: Path, scheduling: str):
    """Install the real-graph adapter and explicit per-launch placement control."""
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("Placement adapter was installed twice")
    install_graph_split(scheduling=scheduling)
    from tokenspeed.runtime.execution.drafter.eagle import Eagle
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        MambaAttnBackend,
    )

    original_init = Eagle.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        global _CONTROL
        original_init(self, *args, **kwargs)
        if _CONTROL is not None:
            raise RuntimeError("This benchmark expects a single drafter per rank")
        _CONTROL = PlacementControl(control_path=control_path)

    def commit(self, payload, parameters, **arguments):
        if _CONTROL is None:
            raise RuntimeError(
                "Placement control must be initialized before GDN commit"
            )
        return _CONTROL.launch(payload, parameters, **arguments)

    Eagle.__init__ = init
    MambaAttnBackend._replay_commit = commit
    _INSTALLED = True
