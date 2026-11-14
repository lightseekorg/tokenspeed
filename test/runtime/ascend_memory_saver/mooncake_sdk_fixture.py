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

"""Test provider for the Model Updater ABI, with file or CPU Mooncake staging.

The fixture's checkpoint paths and object keys are local to this test.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import batched
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load


class EngineType(Enum):
    FLUENT_LLM = "fluent_llm"


@dataclass
class FluentLlmModelUpdateInitConfig:
    weight_store_config: dict
    local_host: str
    hf_type: str
    hf_safetensors_path: str

    @classmethod
    def from_dict(cls, fields):
        return cls(**fields)


@dataclass
class FluentLlmEngineConfig:
    hf_type: str
    hf_safetensors_path: str


@dataclass
class ModelUpdaterConfig:
    role: str
    reader_rank: int
    engine_type: EngineType
    engine_config: FluentLlmEngineConfig


class MooncakeWeightStore:
    def __init__(self, config, *, local_host):
        self.directory = Path(config["directory"])
        self.expect_snapshot = config["expect_snapshot"]
        self.client = None
        if "master" in config:
            self.client = open_store(local_host, config["master"], 0)

    def checkpoint(self, version, role, snapshot):
        if self.client is not None:
            for key in snapshot["fixture_objects"][role]:
                payload = self.client.get(key)
                if not payload:
                    raise RuntimeError(f"Fixture weight object is missing: {key}")
                yield from load(payload).items()
            return
        path = self.directory / f"version-{version}" / role / "model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                yield key, checkpoint.get_tensor(key)


def open_store(host, master, segment_size):
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    result = store.setup(
        host, "P2PHANDSHAKE", segment_size, 512 << 20, "tcp", "", master
    )
    if result != 0:
        raise RuntimeError(f"Fixture Mooncake store setup failed: {result}")
    return store


class FixtureUpdater:
    def __init__(self, config, store):
        self.config = config
        self.store = store

    def update_weights(self, models, *, version, snapshot=None):
        if self.config.reader_rank != 0 or self.config.role != "target":
            raise ValueError("This fixture needs one target reader")
        if (snapshot is not None) != self.store.expect_snapshot:
            raise ValueError("Fixture received the wrong snapshot mode")
        if snapshot is not None and snapshot["version"] != version:
            raise ValueError("Fixture snapshot version did not match the request")
        for role, model in zip(("target", "draft")[: len(models)], models, strict=True):
            for chunk in batched(self.store.checkpoint(version, role, snapshot), 7):
                model.load_weights(iter(chunk))
        return f"fixture checkpoint version {version}"


def make_model_updater(config, *, weight_store):
    return FixtureUpdater(config, weight_store)
