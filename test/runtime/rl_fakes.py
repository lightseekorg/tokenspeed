"""Shared fake ``AsyncLLM`` double for the RL control-route tests."""

from __future__ import annotations

from types import SimpleNamespace


class FakeLLM:
    def __init__(self) -> None:
        self.server_args = SimpleNamespace(
            weight_version="default",
            model="model-x",
            kvstore_storage_backend=None,
        )
        self.updates = []
        self.scheduler_calls = []
        self.admission_calls = []
        self.memory_calls = []
        self.succeed = True

    async def init_weights_update_group(self, obj):
        return True, "initialized"

    async def update_weights_from_distributed(self, obj):
        self.updates.append(obj)
        return self.succeed, "distributed"

    async def update_weights_from_tensor(self, obj):
        self.updates.append(obj)
        return self.succeed, "tensor"

    async def update_weights_from_disk(self, obj):
        self.updates.append(obj)
        return self.succeed, "disk", None

    def block_generation_admission(self):
        self.admission_calls.append("block")

    def allow_generation_admission(self):
        self.admission_calls.append("allow")

    async def pause_scheduler(self, *, mode="abort"):
        self.scheduler_calls.append(("pause", mode))
        return True

    async def resume_scheduler(self):
        self.scheduler_calls.append(("resume", None))
        return True

    async def get_load(self):
        return [
            SimpleNamespace(
                dp_rank=0,
                num_reqs=2,
                num_waiting_reqs=1,
                num_pages=3,
            )
        ]

    async def release_memory_occupation(self, obj):
        self.memory_calls.append(("release", obj.tags))
        return SimpleNamespace(success=True, message="released")

    async def resume_memory_occupation(self, obj):
        self.memory_calls.append(("resume", obj.tags))
        return SimpleNamespace(success=True, message="resumed")

    def abort_request(self, rid):
        self.scheduler_calls.append(("abort", rid))

    async def flush_cache(self):
        return None
