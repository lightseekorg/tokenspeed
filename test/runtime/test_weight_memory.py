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

"""CPU contracts for weight validity; allocator actions simulate lost pages."""

import os
import sys

import pytest
import torch
from torch import nn

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.weight_memory import (  # noqa: E402
    WeightMemoryState,
    backup_weights,
)


class TinyModel(nn.Module):
    supports_weight_discard = True

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4.0))
        self.register_buffer("constant", torch.tensor([2, 3]))

    def required_weight_update_names(self):
        return {"weight"}

    def weight_discard_preserved_tensors(self):
        return (self.constant,)


def discard(model, state):
    state.mark_kv_released()

    def lose_pages():
        with torch.no_grad():
            model.weight.fill_(float("nan"))
            model.constant.zero_()

    state.release(lose_pages)


def test_remap_restores_constants_but_only_complete_update_restores_usability():
    model = TinyModel()
    address = model.weight.data_ptr()
    state = WeightMemoryState(policy="discard", models=[model])
    discard(model, state)
    with pytest.raises(RuntimeError, match="unavailable"):
        state.require_usable()
    state.resume(lambda: None)
    assert model.constant.tolist() == [2, 3]
    assert torch.isnan(model.weight).all()
    assert model.weight.data_ptr() == address
    with pytest.raises(RuntimeError, match="unavailable"):
        state.before_kv_resume()
    assert state.begin_update() is True
    with torch.no_grad():
        model.weight.copy_(torch.arange(4.0) + 10)
    state.finish_update(success=True)
    with pytest.raises(RuntimeError, match="unavailable"):
        state.before_kv_resume()
    state.commit_update(True)
    state.before_kv_resume()
    state.mark_kv_resumed()
    state.require_usable()
    assert model.weight.tolist() == [10, 11, 12, 13]
    assert model.weight.data_ptr() == address


def test_constant_inventory_deduplicates_aliases_without_backing_up_weights():
    model = TinyModel()
    state = WeightMemoryState(policy="discard", models=[model, model])
    assert len(state.preserved) == 1
    assert state.preserved[0].value.numel() == 2
    model.constant.zero_()
    assert state.preserved[0].value.tolist() == [2, 3]


def test_constants_cannot_alias_even_a_slice_of_parameter_storage():
    model = TinyModel()
    model.constant = model.weight.detach()[1:2]
    with pytest.raises(ValueError, match="alias parameter storage"):
        WeightMemoryState(policy="discard", models=[model])


@pytest.mark.parametrize("models", [[], [nn.Linear(2, 2)]])
def test_discard_requires_audited_models(models):
    with pytest.raises(ValueError):
        WeightMemoryState(policy="discard", models=models)


def test_discard_requires_kv_release_and_resident_update_storage():
    model = TinyModel()
    state = WeightMemoryState(policy="discard", models=[model])
    with pytest.raises(RuntimeError, match="Release KV"):
        state.release(lambda: pytest.fail("must not release"))
    with pytest.raises(RuntimeError, match="Release KV"):
        state.begin_update()
    discard(model, state)
    with pytest.raises(RuntimeError, match="Resume weight storage"):
        state.begin_update()
    assert not state.failed


@pytest.mark.parametrize("phase", ["release", "resume", "update"])
def test_transition_failure_is_sticky_and_blocks_all_recovery(phase):
    model = TinyModel()
    state = WeightMemoryState(policy="discard", models=[model])
    state.mark_kv_released()

    def fail():
        raise RuntimeError("allocator failed")

    if phase == "release":
        with pytest.raises(RuntimeError, match="allocator failed"):
            state.release(fail)
    else:
        state.release(lambda: None)
        if phase == "resume":
            with pytest.raises(RuntimeError, match="allocator failed"):
                state.resume(fail)
        else:
            state.resume(lambda: None)
            state.begin_update()
            state.finish_update(success=False)
    assert state.failed
    for action in (
        state.require_usable,
        state.before_kv_resume,
        state.begin_update,
        lambda: state.resume(lambda: None),
        lambda: state.release(lambda: None),
    ):
        with pytest.raises(RuntimeError, match="restart"):
            action()


def test_reallocated_parameter_is_rejected_after_an_otherwise_successful_update():
    model = TinyModel()
    state = WeightMemoryState(policy="discard", models=[model])
    discard(model, state)
    state.resume(lambda: None)
    state.begin_update()
    model.weight = nn.Parameter(torch.ones(4))
    with pytest.raises(RuntimeError, match="captured parameter storage"):
        state.finish_update(success=True)
    assert state.failed and not state.valid


def test_reallocated_constant_is_rejected_on_remap():
    model = TinyModel()
    state = WeightMemoryState(policy="discard", models=[model])
    discard(model, state)
    # Resize the actual inventoried tensor, forcing a storage replacement.
    model.constant.resize_(1000)
    with pytest.raises(RuntimeError, match="captured storage"):
        state.resume(lambda: None)
    assert state.failed


def test_cpu_backup_preserves_legacy_kv_before_weight_resume_order():
    model = TinyModel()
    state = WeightMemoryState(policy="cpu_backup", models=[model])
    assert state.preserved == []
    saved = model.weight.detach().clone()
    state.release(lambda: None)
    state.before_kv_resume()
    state.mark_kv_resumed()
    state.resume(lambda: None)
    state.require_usable()
    torch.testing.assert_close(model.weight, saved)
    assert state.begin_update() is False
    state.finish_update(success=True)
    state.commit_update(True)


@pytest.mark.parametrize(
    "policy,draft_policy,is_draft,expected",
    [
        (None, None, False, True),
        ("cpu_backup", None, False, True),
        ("cpu_backup", "retain", True, True),
        ("discard", "retain", False, False),
        ("discard", "retain", True, True),
        ("discard", "refresh", False, False),
        ("discard", "refresh", True, False),
    ],
)
def test_backup_policy_distinguishes_frozen_draft(
    policy, draft_policy, is_draft, expected
):
    assert (
        backup_weights(policy=policy, is_draft=is_draft, draft_policy=draft_policy)
        is expected
    )


def test_discard_does_not_guess_an_absent_draft_policy():
    with pytest.raises(ValueError, match="explicit draft"):
        backup_weights(policy="discard", is_draft=False, draft_policy=None)


def test_replica_failure_invalidates_a_locally_successful_update():
    model = TinyModel()
    state = WeightMemoryState(policy="discard", models=[model])
    discard(model, state)
    state.resume(lambda: None)
    state.begin_update()
    state.finish_update(success=True)
    state.commit_update(False)
    assert state.failed and not state.valid and not state.updating
    with pytest.raises(RuntimeError, match="restart"):
        state.before_kv_resume()


@pytest.mark.parametrize("policy", ["cpu_backup", "discard"])
def test_prerequisite_check_refuses_without_starting_a_transition(policy):
    model = TinyModel()
    state = WeightMemoryState(policy=policy, models=[model])
    state.mark_kv_released()
    state.release(lambda: None)
    with pytest.raises(RuntimeError, match="Resume weight storage"):
        state.check_update()
    assert not state.failed and not state.updating
    state.resume(lambda: None)
    state.check_update()
    assert not state.updating
    state.begin_update()
    state.finish_update(success=True)
    state.commit_update(True)
    state.before_kv_resume()
    state.require_usable()


def test_replica_failure_poisons_a_rank_without_a_local_session():
    # The control plane commits only a load every rank started, so a failed
    # result reaching a rank without a session still means a mixed replica.
    state = WeightMemoryState(policy="discard", models=[TinyModel()])
    state.commit_update(False)
    with pytest.raises(RuntimeError, match="restart"):
        state.require_usable()


def test_local_completeness_is_required_even_for_a_successful_global_decision():
    state = WeightMemoryState(policy="discard", models=[TinyModel()])
    with pytest.raises(RuntimeError, match="complete local update"):
        state.commit_update(True)
    assert state.failed


def test_update_cannot_replace_a_preserved_constant_object():
    model = TinyModel()
    state = WeightMemoryState(policy="discard", models=[model])
    discard(model, state)
    state.resume(lambda: None)
    state.begin_update()
    model.constant = torch.tensor([2, 3])
    with pytest.raises(RuntimeError, match="replaced preserved model storage"):
        state.finish_update(success=True)
    assert state.failed
