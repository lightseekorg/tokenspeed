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

Models outside ``BaseCausalLM`` (speculative draft shells, multimodal
wrappers) take no session hooks: their ``load_weights`` stays self-contained
per call, as it was before sessions existed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from torch import nn

from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM


@contextmanager
def weight_update_session(models: Sequence[nn.Module]) -> Iterator[None]:
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
    """
    opened: list[BaseCausalLM] = []
    try:
        for model in models:
            if isinstance(model, BaseCausalLM):
                model.begin_weight_update()
                opened.append(model)
        yield
    except BaseException:
        for model in opened:
            model.abort_weight_update()
        raise
    for index, model in enumerate(opened):
        try:
            model.end_weight_update()
        except BaseException:
            # ``end_weight_update`` closes its own model's session; the models
            # not yet ended would otherwise stay open and reject the next
            # update.
            for remaining in opened[index + 1 :]:
                remaining.abort_weight_update()
            raise
