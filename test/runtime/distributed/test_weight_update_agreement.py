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

"""One reader failure must abort every rank before collective post-load hooks."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, path):
    from transformers import PretrainedConfig

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.base import BaseCausalLM
    from tokenspeed.runtime.models.base.weight_update import weight_update_session

    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )

    class CollectiveModel(BaseCausalLM):
        def __init__(self):
            super().__init__(
                config=PretrainedConfig(),
                mapping=Mapping(rank=rank, world_size=2),
                encoder_only=True,
            )
            self.a = torch.nn.Parameter(torch.zeros(2))
            self.b = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("total", torch.zeros(()))
            self.derived = False

        def required_weight_update_names(self):
            return {"a", "b"}

        def load_weights(self, weights):
            names = set()
            with torch.no_grad():
                for name, value in weights:
                    self.get_parameter(name).copy_(value)
                    names.add(name)
            self.post_load_weights()
            return names

        def post_load_weights(self):
            with torch.no_grad():
                self.total.copy_((self.a + self.b).sum())
                dist.all_reduce(self.total)
            self.derived = True

    try:
        for failure in ("stream", "missing", "draft_missing"):
            models = [CollectiveModel(), CollectiveModel()]

            def values(model_index):
                yield "a", torch.ones(2)
                if rank == 0 and model_index == (
                    1 if failure == "draft_missing" else 0
                ):
                    if failure == "stream":
                        raise RuntimeError("injected reader interruption")
                    return
                yield "b", torch.full((2,), 2.0)

            try:
                with weight_update_session(
                    models, require_complete=True, cpu_group=dist.group.WORLD
                ):
                    for i, model in enumerate(models):
                        model.load_weights(values(i))
            except (RuntimeError, ValueError) as error:
                if failure == "stream":
                    assert "injected reader interruption" in str(error)
                else:
                    assert "missing" in str(error).lower()
            else:
                raise AssertionError("A rank accepted incomplete weights")
            assert all(not model.derived for model in models)
            assert all(not model._weight_update_active for model in models)
            # A direct session test can retry; the serving validity state still
            # requires restart after a failed live update.
            with weight_update_session(
                models, require_complete=True, cpu_group=dist.group.WORLD
            ):
                for model in models:
                    model.load_weights(
                        [("a", torch.ones(2)), ("b", torch.full((2,), 2.0))]
                    )
            assert all(model.derived and model.total.item() == 12 for model in models)
            assert all(not model._weight_update_active for model in models)
    finally:
        dist.destroy_process_group()


def test_weight_update_agrees_before_collective_post_load(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "gloo_init"),), nprocs=2, join=True)
