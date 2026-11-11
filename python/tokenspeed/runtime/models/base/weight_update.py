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

"""Host-side bracket for live weight updates: the model update session.

An RL trainer replaces a serving model's parameters in place through many
partial ``load_weights`` calls -- one per NCCL broadcast on the distributed
path, whatever chunking the Model Updater SDK streams on the Mooncake path.
``weight_update_session`` opens ``begin_weight_update`` on every model that
implements the protocol (``BaseCausalLM``) before the first call and closes
``end_weight_update`` after the last, so each model derives its post-load
state exactly once over the whole update instead of once per chunk.

Complete updates require every model to provide its canonical checkpoint-name
contract. The session validates all models before deriving any post-load state.
Models outside ``BaseCausalLM`` (speculative draft shells, multimodal wrappers)
take no hooks in partial mode and are rejected when completeness is required.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch.distributed as dist
from torch import nn

from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM


@contextmanager
def weight_update_session(
    models: Sequence[nn.Module],
    *,
    require_complete: bool,
    cpu_group: dist.ProcessGroup | None,
) -> Iterator[None]:
    """Bracket a live weight update of ``models``.

    ``end_weight_update`` runs only for the models whose session was opened,
    and only when the update body did not raise: a failed update leaves the
    model half-written, and deriving state from it would hide the failure
    behind a later, unrelated error. The session flag is still cleared on
    every opened model so the next update can start -- also when one model's
    ``end_weight_update`` raises, in which case the models after it are
    aborted rather than derived.

    Args:
        models: The modules the update streams into, target first.
        require_complete: Require every canonical checkpoint source exactly
            once before any model derives post-load state. Discarded weights
            need this contract; ordinary live updates may remain partial.
        cpu_group: Replica CPU group for agreement before post-load hooks.
            None explicitly selects a local session without rank agreement.
    """
    opened: list[BaseCausalLM] = []
    local_error: BaseException | None = None
    try:
        if require_complete and not models:
            raise ValueError("A complete weight update requires at least one model")
        for model in models:
            if not isinstance(model, BaseCausalLM):
                if require_complete:
                    raise TypeError(
                        f"{type(model).__name__} cannot validate "
                        "a complete weight update"
                    )
                continue
            if model._weight_update_active:
                raise RuntimeError(
                    f"{type(model).__name__}: a weight-update session is already active"
                )
            try:
                model.begin_weight_update()
            except BaseException:
                # An override may open the base session before its own setup
                # fails. Do not leave that half-open session behind.
                model.abort_weight_update()
                raise
            opened.append(model)
            if require_complete:
                names = model.required_weight_update_names()
                if not isinstance(names, set) or any(
                    not isinstance(name, str) or not name for name in names
                ):
                    raise TypeError(
                        f"{type(model).__name__}: required checkpoint names must "
                        "be a set of nonempty strings"
                    )
                model._weight_update_required_names = set(names)
                model._weight_update_source_names = set()
        yield
        # Target and draft can share storage. No model may derive from a
        # partially received version merely because it is first in this list.
        for model in opened:
            model._validate_weight_update_complete()
    except BaseException as exc:
        local_error = exc
    try:
        if cpu_group is not None:
            # Post-load hooks can use collectives. Every rank must finish
            # receiving and validating before any rank enters those hooks.
            errors: list[str | None] = [None] * dist.get_world_size(cpu_group)
            dist.all_gather_object(
                errors,
                (
                    None
                    if local_error is None
                    else f"{type(local_error).__name__}: {local_error}"
                ),
                group=cpu_group,
            )
            if local_error is None and any(error is not None for error in errors):
                failures = "; ".join(
                    f"group rank {rank}: {error}"
                    for rank, error in enumerate(errors)
                    if error is not None
                )
                raise RuntimeError(
                    f"Weight update aborted before post-load: {failures}"
                )
        if local_error is not None:
            raise local_error
    except BaseException:
        for model in opened:
            model.abort_weight_update()
        raise
    for index, model in enumerate(opened):
        try:
            model.end_weight_update()
        except BaseException:
            # The raising override may not have reached the base cleanup.
            # Close it as well as the models not yet derived.
            for remaining in opened[index:]:
                remaining.abort_weight_update()
            raise
