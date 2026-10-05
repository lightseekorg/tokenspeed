"""Tests for draft-to-target wiring.

Model-to-model wiring (shared embed/head, EAGLE3 capture ids) happens in
``factory.configure_draft_target`` right after both models load, so
shared weights are released before the KV-cache budget is profiled.
Drafter-instance wiring binds execution resources only. Capture is configured
by the loaded model during setup, before cache allocation.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import tokenspeed.runtime.execution.factory as factory  # noqa: E402
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    UpdateWeightsFromDistributedReqInput,
)
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.device import DeviceHandle  # noqa: E402
from tokenspeed.runtime.execution.drafter import get_drafter_impl  # noqa: E402
from tokenspeed.runtime.execution.drafter.base import BaseDrafter  # noqa: E402
from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (  # noqa: E402
    DeepseekV4DSpark,
)
from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import (  # noqa: E402
    DeepseekV41DSpark,
)
from tokenspeed.runtime.execution.drafter.dflash import DFlash  # noqa: E402
from tokenspeed.runtime.execution.drafter.dspark import DSpark  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import Eagle  # noqa: E402
from tokenspeed.runtime.execution.drafter.mtp import Mtp  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.models.target_capture import (  # noqa: E402
    TargetCaptureConfigurator,
)


def _server_args(algo: str, capture_ids=None) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_algorithm=algo,
        eagle3_layers_to_capture=capture_ids,
    )


def _draft_runner(model) -> SimpleNamespace:
    """A draft model runner whose model reads no request-token history."""
    return SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(requires_request_token_history=False),
    )


def _mapping(*, has_pp: bool = False, is_last_pp_rank: bool = True) -> SimpleNamespace:
    """The pipeline facts resource binding reads off the target runner."""
    return SimpleNamespace(has_pp=has_pp, is_last_pp_rank=is_last_pp_rank)


def _target_runner(model, mapping=None) -> SimpleNamespace:
    return SimpleNamespace(
        model=model, mapping=_mapping() if mapping is None else mapping
    )


def _head_layout(**overrides) -> SimpleNamespace:
    """The LM-head layout the factory compares between a target and the draft
    sharing its head (``_check_shared_head_layout``): the logits processor's
    group, width and attention-DP flag. Default: one replicated head."""
    layout = dict(tp_group=None, tp_size=1, dp_lm_head_tp=False)
    layout.update(overrides)
    return SimpleNamespace(**layout)


def test_get_drafter_impl_routing():
    from tokenspeed.runtime.models.deepseek_v4_dspark import (
        DeepseekV4ForCausalLMDSpark,
    )
    from tokenspeed.runtime.models.deepseek_v41_dspark import (
        DeepseekV41ForCausalLMDSpark,
    )
    from tokenspeed.runtime.models.inkling_nextn import (
        InklingForConditionalGenerationNextN,
    )

    assert get_drafter_impl("EAGLE3", mock.MagicMock()) is Eagle
    assert get_drafter_impl("MTP", mock.MagicMock()) is Eagle
    assert (
        get_drafter_impl(
            "MTP", mock.MagicMock(spec=InklingForConditionalGenerationNextN)
        )
        is Mtp
    )
    assert get_drafter_impl("DFLASH", mock.MagicMock()) is DFlash
    assert get_drafter_impl("DSPARK", mock.MagicMock()) is DSpark
    assert (
        get_drafter_impl("DSPARK", mock.MagicMock(spec=DeepseekV4ForCausalLMDSpark))
        is DeepseekV4DSpark
    )
    assert (
        get_drafter_impl("DSPARK", mock.MagicMock(spec=DeepseekV41ForCausalLMDSpark))
        is DeepseekV41DSpark
    )


def test_shares_target_embed_head_flags():
    # Eagle-like drafters and the checkpoint-local V4 DSpark reuse the
    # target's embed/LM head; DFlash and generic DSpark ship their own.
    assert Eagle.shares_target_embed_head
    assert Mtp.shares_target_embed_head
    assert DeepseekV4DSpark.shares_target_embed_head
    assert not DFlash.shares_target_embed_head
    assert not DSpark.shares_target_embed_head
    assert not BaseDrafter.shares_target_embed_head


def test_pd_layerwise_finalization_capability_matches_supported_drafters():
    assert Eagle.supports_pd_layerwise_finalization
    assert DFlash.supports_pd_layerwise_finalization
    assert DSpark.supports_pd_layerwise_finalization
    # Mtp.run enqueues every depth's forward (and its plane's KV write) on
    # the caller's stream before returning, so the draft-final step is exact.
    assert Mtp.supports_pd_layerwise_finalization
    assert not DeepseekV4DSpark.supports_pd_layerwise_finalization
    assert not BaseDrafter.supports_pd_layerwise_finalization


def test_wire_eagle3_shares_embed_head_and_installs_capture_ids():
    target, draft = mock.MagicMock(), mock.MagicMock()
    target.mapping = _mapping()
    target.model.get_embed_and_head.return_value = ("EMBED", "HEAD")
    target.model.logits_processor = draft.model.logits_processor = _head_layout()
    draft.model_config.hf_config = {
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}
    }
    draft.model_config.requires_request_token_history = False

    with mock.patch.object(factory, "get_drafter_impl", return_value=Eagle):
        factory.configure_draft_target(_server_args("EAGLE3"), target, draft)

    draft.model.set_embed_and_head.assert_called_once_with("EMBED", "HEAD")
    target.model.set_eagle3_layers_to_capture.assert_called_once_with([1, 2, 3])


class _ModuleSharingDraft:
    """A draft taking the target's complete lm_head module (Qwen3.5 NextN).

    Like the in-tree pipeline-capable drafts it keeps its checkpoint embedding
    when the target shares none and reports what it drafts with.
    """

    def __init__(self):
        self.shared = None
        self.legacy = None
        self.embedding = "CHECKPOINT_EMBED"
        self.logits_processor = _head_layout()

    def set_embed_and_head_module(self, embed, lm_head):
        self.shared = (embed, lm_head)
        if embed is not None:
            self.embedding = embed

    def set_embed_and_head(self, embed, head):
        self.legacy = (embed, head)

    def get_embed_and_head(self):
        return self.embedding, "DRAFT_HEAD"


class _WeightSharingDraft:
    """A draft aliasing the target's weights (the K3 / V4 NextN contract)."""

    def __init__(self):
        self.shared = None
        self.embedding = "CHECKPOINT_EMBED"
        self.logits_processor = _head_layout()

    def set_embed_and_head(self, embed, head):
        self.shared = (embed, head)
        if embed is not None:
            self.embedding = embed

    def get_embed_and_head(self):
        return self.embedding, "DRAFT_HEAD"


class _EmbedDroppingDraft:
    """A generic draft that aliases whatever it is handed, None included."""

    def __init__(self):
        self.embedding = "CHECKPOINT_EMBED"
        self.logits_processor = _head_layout()

    def set_embed_and_head(self, embed, head):
        self.embedding = embed

    def get_embed_and_head(self):
        return self.embedding, "DRAFT_HEAD"


def test_wire_mtp_shares_complete_lm_head_for_opted_in_draft():
    lm_head = object()
    target = _target_runner(
        SimpleNamespace(
            lm_head=lm_head,
            get_embed_and_head=lambda: ("EMBED", "HEAD_WEIGHT"),
            logits_processor=_head_layout(),
        )
    )
    draft_model = _ModuleSharingDraft()
    draft = _draft_runner(draft_model)

    with mock.patch.object(factory, "get_drafter_impl", return_value=Mtp):
        factory.configure_draft_target(_server_args("MTP"), target, draft)

    assert draft_model.shared == ("EMBED", lm_head)
    assert draft_model.legacy is None


def test_wire_mtp_module_sharing_requires_target_lm_head():
    target = _target_runner(
        SimpleNamespace(
            get_embed_and_head=lambda: ("EMBED", "HEAD_WEIGHT"),
            logits_processor=_head_layout(),
        )
    )
    draft = _draft_runner(_ModuleSharingDraft())

    with (
        mock.patch.object(factory, "get_drafter_impl", return_value=Mtp),
        pytest.raises(ValueError, match="complete lm_head module"),
    ):
        factory.configure_draft_target(_server_args("MTP"), target, draft)


def test_wire_off_pipeline_requires_both_target_weights():
    draft = _draft_runner(_WeightSharingDraft())
    for embed, head, message in (
        (None, "HEAD", "embedding weight"),
        ("EMBED", None, "lm_head weight"),
    ):
        target = _target_runner(
            SimpleNamespace(
                get_embed_and_head=lambda e=embed, h=head: (e, h),
                logits_processor=_head_layout(),
            )
        )
        with (
            mock.patch.object(factory, "get_drafter_impl", return_value=Eagle),
            pytest.raises(ValueError, match=message),
        ):
            factory.configure_draft_target(_server_args("MTP"), target, draft)


def test_wire_rejects_a_draft_head_in_another_layout():
    # The target vocab-shards its head over the LM-head TP group under
    # attention DP (--lm-head-tp-size); a draft that built a replicated head
    # would take the shard as the whole vocab, so the factory refuses the
    # share before anything is bound.
    target = _target_runner(
        SimpleNamespace(
            get_embed_and_head=mock.Mock(side_effect=AssertionError("bound")),
            logits_processor=_head_layout(
                tp_group=(0, 1), tp_size=2, dp_lm_head_tp=True
            ),
        )
    )
    with (
        mock.patch.object(factory, "get_drafter_impl", return_value=Eagle),
        pytest.raises(ValueError, match="--lm-head-tp-size"),
    ):
        factory.configure_draft_target(
            _server_args("MTP"), target, _draft_runner(_WeightSharingDraft())
        )


@pytest.mark.parametrize("drafter_cls", [Eagle, Mtp])
def test_wire_skips_embed_head_sharing_before_the_last_pipeline_stage(drafter_cls):
    # Stages before the last run no drafter (a draft built there only
    # produces context), and the target holds neither side there (embedding
    # on the first stage, head on the last). Nothing is read or bound.
    target_model = SimpleNamespace(
        get_embed_and_head=mock.Mock(side_effect=AssertionError("read a shell"))
    )
    target = _target_runner(target_model, _mapping(has_pp=True, is_last_pp_rank=False))
    for draft_model in (_WeightSharingDraft(), _ModuleSharingDraft()):
        with mock.patch.object(factory, "get_drafter_impl", return_value=drafter_cls):
            factory.configure_draft_target(
                _server_args("MTP"), target, _draft_runner(draft_model)
            )
        assert draft_model.shared is None
    target_model.get_embed_and_head.assert_not_called()


def _last_stage_target(lm_head):
    """The last stage reports the target head but no embedding."""
    return _target_runner(
        SimpleNamespace(
            lm_head=lm_head,
            get_embed_and_head=lambda: (None, "HEAD_WEIGHT"),
            logits_processor=_head_layout(),
        ),
        _mapping(has_pp=True, is_last_pp_rank=True),
    )


def test_wire_last_pipeline_stage_shares_the_head_and_keeps_the_draft_embedding():
    # The last stage owns the target head but not its embedding: the draft is
    # handed embed=None, binds the head, and must still draft with the
    # embedding its checkpoint ships.
    lm_head = object()
    weight_draft, module_draft = _WeightSharingDraft(), _ModuleSharingDraft()
    with mock.patch.object(factory, "get_drafter_impl", return_value=Eagle):
        factory.configure_draft_target(
            _server_args("MTP"),
            _last_stage_target(lm_head),
            _draft_runner(weight_draft),
        )
    with mock.patch.object(factory, "get_drafter_impl", return_value=Mtp):
        factory.configure_draft_target(
            _server_args("MTP"),
            _last_stage_target(lm_head),
            _draft_runner(module_draft),
        )
    assert weight_draft.shared == (None, "HEAD_WEIGHT")
    assert weight_draft.get_embed_and_head()[0] == "CHECKPOINT_EMBED"
    assert module_draft.shared == (None, lm_head)
    assert module_draft.get_embed_and_head()[0] == "CHECKPOINT_EMBED"
    assert module_draft.legacy is None


def test_wire_last_pipeline_stage_rejects_a_draft_that_dropped_its_embedding():
    # A draft that aliases None into its embedding would fail at its first
    # forward; the factory names it at construction instead.
    with (
        mock.patch.object(factory, "get_drafter_impl", return_value=Eagle),
        pytest.raises(ValueError, match="_EmbedDroppingDraft has no embedding"),
    ):
        factory.configure_draft_target(
            _server_args("MTP"),
            _last_stage_target(object()),
            _draft_runner(_EmbedDroppingDraft()),
        )


def test_wire_last_pipeline_stage_rejects_a_draft_that_reports_no_embedding():
    # Without get_embed_and_head nothing proves the draft kept one.
    with (
        mock.patch.object(factory, "get_drafter_impl", return_value=Eagle),
        pytest.raises(ValueError, match="report it from get_embed_and_head"),
    ):
        factory.configure_draft_target(
            _server_args("MTP"),
            _last_stage_target(object()),
            _draft_runner(
                SimpleNamespace(
                    set_embed_and_head=lambda e, h: None,
                    logits_processor=_head_layout(),
                )
            ),
        )


STAGE_GROUP = (4, 5, 6, 7)


class _RecordingRunner:
    """Stands in for ModelRunner: records what the factory asked to build."""

    built: list[tuple[str, tuple[int, ...] | None, bool]] = []

    def __init__(
        self,
        *,
        model_config,
        gpu_id,
        server_args,
        global_rank,
        checkpoint_load_group,
        is_draft_worker=False,
    ):
        del gpu_id, server_args, global_rank
        self.model = model_config
        _RecordingRunner.built.append(
            (model_config, checkpoint_load_group, is_draft_worker)
        )


@pytest.mark.parametrize(
    "spec_algo,has_pp,is_last_pp_rank,builds_draft",
    [
        ("MTP", False, True, True),
        ("MTP", True, True, True),
        # An MTP draft reads only the last stage's hidden states: the stages
        # before it build nothing -- no weights, no memory, no wiring.
        ("MTP", True, False, False),
        # A DSPARK draft produces its context on every stage.
        ("DSPARK", True, False, True),
        ("DSPARK", True, True, True),
    ],
)
def test_create_model_runner_builds_the_draft_where_it_is_used(
    spec_algo, has_pp, is_last_pp_rank, builds_draft
):
    _RecordingRunner.built = []
    configured = []
    mapping = SimpleNamespace(
        has_pp=has_pp,
        is_last_pp_rank=is_last_pp_rank,
        attn=SimpleNamespace(world_group=STAGE_GROUP),
    )
    server_args = SimpleNamespace(speculative_algorithm=spec_algo, mapping=mapping)
    with (
        mock.patch.object(factory, "ModelRunner", _RecordingRunner),
        mock.patch.object(
            factory, "configure_draft_target", lambda *args: configured.append(args)
        ),
    ):
        target, draft = factory.create_model_runner(
            server_args, "target", "draft", gpu_id=0, global_rank=5
        )

    # The target reads the whole checkpoint on every stage: world-wide loads.
    assert _RecordingRunner.built[0] == ("target", None, False)
    assert target.model == "target"
    if not builds_draft:
        assert draft is None
        assert _RecordingRunner.built == [("target", None, False)]
        assert configured == []
        return
    # A pipeline draft loads within its stage: other stages may build none.
    assert _RecordingRunner.built[1] == (
        "draft",
        STAGE_GROUP if has_pp else None,
        True,
    )
    assert draft.model == "draft"
    assert configured == [(server_args, target, draft)]


def test_pipeline_stage_builds_draft_follows_the_drafter_kind():
    before_last = SimpleNamespace(has_pp=True, is_last_pp_rank=False)
    last = SimpleNamespace(has_pp=True, is_last_pp_rank=True)
    single = SimpleNamespace(has_pp=False, is_last_pp_rank=True)
    for spec_algo in ("MTP", "EAGLE3", "DSPARK", "DFLASH", None):
        assert factory.pipeline_stage_builds_draft(spec_algo, last)
        assert factory.pipeline_stage_builds_draft(spec_algo, single)
    # Block drafters produce context from every stage's taps; nothing else
    # has work before the last stage.
    assert factory.pipeline_stage_builds_draft("DSPARK", before_last)
    assert factory.pipeline_stage_builds_draft("DFLASH", before_last)
    assert not factory.pipeline_stage_builds_draft("MTP", before_last)
    assert not factory.pipeline_stage_builds_draft("EAGLE3", before_last)


def test_wire_eagle3_explicit_capture_ids_override_checkpoint():
    target, draft = mock.MagicMock(), mock.MagicMock()
    target.mapping = _mapping()
    target.model.get_embed_and_head.return_value = ("E", "H")
    target.model.logits_processor = draft.model.logits_processor = _head_layout()
    draft.model_config.hf_config = {
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}
    }

    with mock.patch.object(factory, "get_drafter_impl", return_value=Eagle):
        factory.configure_draft_target(
            _server_args("EAGLE3", capture_ids=[7, 8]), target, draft
        )

    target.model.set_eagle3_layers_to_capture.assert_called_once_with([7, 8])


class _ContextDraft:
    """A draft implementing the DSparkContextModel protocol on one stage."""

    hidden_size = 8

    def __init__(self, is_last_pp_rank: bool):
        self.mapping = SimpleNamespace(
            is_first_pp_rank=not is_last_pp_rank, is_last_pp_rank=is_last_pp_rank
        )

    def project_target_tap(self, capture_idx, hidden):
        raise AssertionError("unused")

    def finalize_target_projection(self, projected):
        raise AssertionError("unused")

    def write_context_kv(self, ctx_hidden, positions, cache_locs, token_to_kv_pool):
        raise AssertionError("unused")


@pytest.mark.parametrize("is_last_pp_rank", [False, True])
def test_pipeline_dspark_draft_gets_a_context_producer_on_every_stage(is_last_pp_rank):
    from tokenspeed.runtime.execution.dspark_context import DSparkContextProducer
    from tokenspeed.runtime.execution.model_executor import (
        select_dspark_context_producer,
    )

    draft_pool = object() if is_last_pp_rank else None
    producer = select_dspark_context_producer(
        spec_algo="DSPARK",
        pp_size=4,
        draft_model=_ContextDraft(is_last_pp_rank),
        draft_token_to_kv_pool=draft_pool,
    )
    assert isinstance(producer, DSparkContextProducer)
    assert producer.token_to_kv_pool is draft_pool


@pytest.mark.parametrize(
    "draft_model,draft_pool",
    [
        # A stage before the last builds no MTP draft at all.
        (None, None),
        # The last stage drafts from captured hidden states without a producer.
        (torch.nn.Module(), object()),
    ],
)
def test_pipeline_mtp_draft_produces_no_context(draft_model, draft_pool):
    from tokenspeed.runtime.execution.model_executor import (
        select_dspark_context_producer,
    )

    assert (
        select_dspark_context_producer(
            spec_algo="MTP",
            pp_size=2,
            draft_model=draft_model,
            draft_token_to_kv_pool=draft_pool,
        )
        is None
    )


def test_pipeline_eagle3_is_refused_by_the_executor():
    # EAGLE3's aux taps live on several stages; the executor refuses it even
    # when the ServerArgs gate is bypassed.
    from tokenspeed.runtime.execution.model_executor import (
        select_dspark_context_producer,
    )

    with pytest.raises(ValueError, match="EAGLE3 cannot run on a pipeline"):
        select_dspark_context_producer(
            spec_algo="EAGLE3",
            pp_size=2,
            draft_model=torch.nn.Module(),
            draft_token_to_kv_pool=object(),
        )


def test_pipeline_block_drafter_without_context_production_is_rejected():
    from tokenspeed.runtime.execution.model_executor import (
        select_dspark_context_producer,
    )

    with pytest.raises(TypeError, match="cannot produce DSpark context"):
        select_dspark_context_producer(
            spec_algo="DSPARK",
            pp_size=2,
            draft_model=torch.nn.Module(),
            draft_token_to_kv_pool=None,
        )


def test_single_stage_never_builds_a_context_producer():
    from tokenspeed.runtime.execution.model_executor import (
        select_dspark_context_producer,
    )

    assert (
        select_dspark_context_producer(
            spec_algo="DSPARK",
            pp_size=1,
            draft_model=_ContextDraft(is_last_pp_rank=True),
            draft_token_to_kv_pool=object(),
        )
        is None
    )
    assert (
        select_dspark_context_producer(
            spec_algo=None, pp_size=2, draft_model=None, draft_token_to_kv_pool=None
        )
        is None
    )


@pytest.mark.parametrize(
    "has_pp,is_last_pp_rank,producer,drafter,expected",
    [
        # Stages before the last own no draft fields: nothing to finalize.
        (True, False, None, None, True),
        # The last stage answers for the draft-field writer, like a non-PP engine.
        # Mtp enqueues every depth's KV write inside run(), so it finalizes
        # layerwise like Eagle; the vanilla base drafter does not.
        (True, True, None, Eagle, True),
        (True, True, None, Mtp, True),
        (False, True, None, Eagle, True),
        (False, True, None, Mtp, True),
        (False, True, None, BaseDrafter, False),
        # A configured context producer owns the draft writes.
        (True, True, "producer", DSpark, True),
        # No speculation: unused, and nothing claims support.
        (False, True, None, None, False),
    ],
)
def test_pd_layerwise_finalization_is_decided_per_pipeline_stage(
    has_pp, is_last_pp_rank, producer, drafter, expected
):
    from tokenspeed.runtime.execution.device import (
        _supports_pd_layerwise_finalization,
    )
    from tokenspeed.runtime.execution.dspark_context import DSparkContextProducer
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    # The real executor property: one predicate names the draft-field writer
    # for the startup flag and for register_draft_final_step_counter alike.
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.dspark_context_producer = (
        DSparkContextProducer.__new__(DSparkContextProducer)
        if producer is not None
        else None
    )
    executor.drafter = drafter.__new__(drafter) if drafter is not None else None
    mapping = SimpleNamespace(has_pp=has_pp, is_last_pp_rank=is_last_pp_rank)
    assert _supports_pd_layerwise_finalization(executor, mapping) is expected
    if has_pp and not is_last_pp_rank:
        return
    # The owning stage's register_draft_final_step_counter agrees with the flag.
    executor.draft_attn_backend = object()
    executor.attn_backend = object()
    executor._draft_final_step_counter = None
    if expected:
        executor.register_draft_final_step_counter("counter")
        assert executor._draft_final_step_counter == "counter"
    else:
        with pytest.raises(RuntimeError, match="cannot finalize layerwise"):
            executor.register_draft_final_step_counter("counter")


def test_draft_field_writer_prefers_the_context_producer():
    from tokenspeed.runtime.execution.dspark_context import DSparkContextProducer
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    executor = ModelExecutor.__new__(ModelExecutor)
    executor.dspark_context_producer = None
    executor.drafter = None
    assert executor.draft_field_writer is None
    executor.drafter = Eagle.__new__(Eagle)
    assert executor.draft_field_writer is executor.drafter
    executor.dspark_context_producer = DSparkContextProducer.__new__(
        DSparkContextProducer
    )
    assert executor.draft_field_writer is executor.dspark_context_producer


def test_wire_dflash_keeps_own_embed_head():
    from tokenspeed.runtime.models.target_capture import TargetCaptureConfigurator

    target, draft = mock.MagicMock(), mock.MagicMock()
    draft.model = mock.MagicMock(spec=TargetCaptureConfigurator)
    draft.model_config.requires_request_token_history = False
    with mock.patch.object(factory, "get_drafter_impl", return_value=DFlash):
        factory.configure_draft_target(_server_args("DFLASH"), target, draft)
    draft.model.configure_target.assert_called_once_with(
        target.model, target.model_config.hf_text_config
    )
    target.model.get_embed_and_head.assert_not_called()


def test_multi_depth_mtp_refuses_draft_trees():
    drafter = mock.MagicMock(spec=Mtp)
    with pytest.raises(NotImplementedError, match="cannot draft trees"):
        Mtp.bind_tree(drafter, mock.MagicMock())


def test_eagle_tree_drafting_ranks_full_vocab_logits():
    """A fused distributed argmax would leave each TP rank only its vocab shard."""
    from tokenspeed.runtime.execution.tree_spec import TreeSpecConfig

    drafter = mock.MagicMock(spec=Eagle)
    drafter.draft_reads_token_history = False
    drafter.dp_size = 1
    drafter.device = torch.device("cpu")
    drafter.hot_token_ids = None
    drafter.input_buffers = SimpleNamespace(max_bs=2)
    drafter.token_to_kv_pool = SimpleNamespace(
        layer_num=1, head_num=1, head_dim=8, dtype=torch.bfloat16
    )
    processor = SimpleNamespace(do_argmax=True, config=SimpleNamespace(vocab_size=50))
    drafter.draft_model_runner = SimpleNamespace(
        model=SimpleNamespace(logits_processor=processor)
    )
    drafter.attn_backend = mock.MagicMock()
    tree_spec = SimpleNamespace(config=TreeSpecConfig(topk=2, num_steps=3, num_nodes=6))

    Eagle.bind_tree(drafter, tree_spec)

    assert processor.do_argmax is False
    full = SimpleNamespace(next_token_logits=torch.randn(2, 50, device="cuda"))
    scores, tokens = Eagle._score_candidates(drafter, full)
    assert scores.shape == tokens.shape == (2, 2)
    shard = SimpleNamespace(next_token_logits=torch.zeros(2, 25))
    with pytest.raises(RuntimeError, match="ranks all 50 draft tokens"):
        Eagle._score_candidates(drafter, shard)


def test_base_wire_target_is_a_noop():
    drafter = mock.MagicMock(spec=BaseDrafter)
    target_model = mock.MagicMock()
    BaseDrafter.wire_target(drafter, target_model)
    target_model.assert_not_called()


@pytest.mark.parametrize("drafter_class", [DFlash, DSpark])
def test_block_drafter_wire_target_only_binds_resources(drafter_class):
    drafter = drafter_class.__new__(drafter_class)
    drafter.model = SimpleNamespace()
    embedding, head, processor = object(), object(), object()
    target = SimpleNamespace(
        get_input_embeddings=lambda: embedding,
        lm_head=head,
        logits_processor=processor,
        set_dflash_layers_to_capture=mock.Mock(
            side_effect=AssertionError("capture changed during binding")
        ),
        set_dflash_aux_hidden_stream=mock.Mock(
            side_effect=AssertionError("stream changed during binding")
        ),
    )
    drafter.wire_target(target)
    assert drafter.embed_tokens is embedding
    assert drafter.lm_head is head
    assert drafter.logits_processor is processor
    target.set_dflash_layers_to_capture.assert_not_called()
    target.set_dflash_aux_hidden_stream.assert_not_called()


def test_block_drafter_binds_local_embedding_on_last_pipeline_stage():
    drafter = DSpark.__new__(DSpark)
    embedding = object()
    drafter.model = SimpleNamespace(embed_tokens=embedding)
    target = SimpleNamespace(
        get_input_embeddings=lambda: None, lm_head=object(), logits_processor=object()
    )
    drafter.wire_target(target)
    assert drafter.embed_tokens is embedding


def test_deepseek_dspark_wire_target_only_binds_resources():
    drafter = DeepseekV4DSpark.__new__(DeepseekV4DSpark)
    draft_head = object()
    drafter.draft_model = SimpleNamespace(lm_head=draft_head)
    target = SimpleNamespace(
        logits_processor=SimpleNamespace(tp_group=(0, 1)),
        set_dspark_layers_to_capture=mock.Mock(
            side_effect=AssertionError("capture changed during binding")
        ),
    )
    drafter.wire_target(target)
    assert drafter.lm_head is draft_head
    assert drafter.tp_group == (0, 1)
    target.set_dspark_layers_to_capture.assert_not_called()


@pytest.mark.parametrize("drafter_cls", [DeepseekV4DSpark, DeepseekV41DSpark])
def test_dspark_weight_update_needs_no_derived_head_refresh(drafter_cls):
    # The draft reads the target's BF16 head in place, so an in-place target
    # weight update needs no drafter-side refresh: the base no-op applies.
    assert (
        drafter_cls.on_target_weights_updated is BaseDrafter.on_target_weights_updated
    )
    drafter = mock.MagicMock(spec=drafter_cls)
    drafter.model = mock.MagicMock()
    drafter_cls.on_target_weights_updated(drafter)
    assert drafter.model.mock_calls == []


def test_device_weight_update_notifies_drafter_before_returning():
    runner = mock.MagicMock()
    runner.update_weights_from_distributed.return_value = (True, "updated")
    drafter = mock.MagicMock(spec=BaseDrafter)
    forward_thread = mock.MagicMock()
    forward_thread.run.side_effect = lambda callback: callback()
    executor = SimpleNamespace(
        model_runner=runner,
        drafter=drafter,
        forward_thread=forward_thread,
    )
    handle = DeviceHandle(executor)
    request = UpdateWeightsFromDistributedReqInput(
        names=[],
        dtype_names=[],
        shapes=[],
        group_name="weight_update_group",
        flush_cache=True,
        weight_version=None,
    )

    assert handle.update_weights(request) == (True, "updated")
    drafter.on_target_weights_updated.assert_called_once_with()


# --------------------------------------------------------------------------
# prepare_target_forward: what a drafter attaches to the target's context
# --------------------------------------------------------------------------


def _target_ctx(num_extends: int, num_tokens: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=SimpleNamespace(
            decode_window_locations=lambda: torch.arange(100, 100 + 2 * num_tokens)
        ),
        token_to_kv_pool=None,
        bs=2,
        num_extends=num_extends,
        output_layout=ForwardOutputLayout(num_extends, num_extends, 2 - num_extends, 1),
        input_num_tokens=num_tokens,
        forward_mode=ForwardMode.DECODE,
    )


def _incremental_dflash(enabled: bool) -> DFlash:
    """A DFlash with only the incremental-projection state set up."""
    drafter = DFlash.__new__(DFlash)
    drafter._incremental_proj_enabled = enabled
    drafter._fused_kv_enabled = True
    drafter._kv_aux_stream = object()
    drafter._incremental_kv_write_done = True  # stale from a previous round
    drafter._incr_num_tokens = 0
    drafter._incr_acc_buf = torch.ones(16, 3)
    drafter.input_buffers = SimpleNamespace(positions_buf=torch.arange(16))
    return drafter


def test_base_prepare_target_forward_attaches_nothing():
    ctx = _target_ctx(num_extends=0, num_tokens=4)
    BaseDrafter.prepare_target_forward(mock.MagicMock(spec=BaseDrafter), ctx)
    assert ctx.target_capture_sink is None


def test_dflash_arms_the_incremental_projection_on_the_forward_context():
    drafter = _incremental_dflash(enabled=True)
    ctx = _target_ctx(num_extends=0, num_tokens=4)

    drafter.prepare_target_forward(ctx)

    assert ctx.target_capture_sink is drafter
    assert drafter._incremental_kv_write_done is False
    assert drafter._incr_num_tokens == 4
    assert torch.equal(drafter._incr_positions, torch.arange(4))
    assert torch.equal(drafter._incr_cache_locs, torch.arange(100, 104))
    # The accumulator rows of this forward are cleared, nothing else.
    assert torch.equal(drafter._incr_acc_buf[:4], torch.zeros(4, 3))
    assert torch.equal(drafter._incr_acc_buf[4:], torch.ones(12, 3))


@pytest.mark.parametrize(
    "enabled, num_extends, graph_warmup",
    [
        (False, 0, False),  # projection disabled
        (True, 1, False),  # a mixed round: extend rows in the write vector
        (True, 0, True),  # graph warmup runs auxiliary branches serially
    ],
)
def test_dflash_leaves_the_context_alone_when_it_will_not_overlap(
    enabled, num_extends, graph_warmup
):
    import tokenspeed.runtime.execution.drafter.dflash as dflash_module

    drafter = _incremental_dflash(enabled=enabled)
    ctx = _target_ctx(num_extends=num_extends, num_tokens=4)

    with mock.patch.object(
        dflash_module, "get_is_cuda_graph_phase", return_value=graph_warmup
    ):
        drafter.prepare_target_forward(ctx)

    assert ctx.target_capture_sink is None
    # A stale "already written" from an earlier round never survives into
    # a round that did not arm.
    assert drafter._incremental_kv_write_done is False
    assert torch.equal(drafter._incr_acc_buf, torch.ones(16, 3))


def test_dflash_rejects_a_capture_of_the_wrong_width():
    drafter = _incremental_dflash(enabled=True)
    drafter.prepare_target_forward(_target_ctx(num_extends=0, num_tokens=4))
    with pytest.raises(RuntimeError, match="armed for 4"):
        drafter.on_target_capture(0, torch.zeros(3, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="aux-stream projection")
def test_dflash_sink_folds_the_taps_and_writes_the_kv_once():
    """fc over the concatenated taps == the sum of per-tap GEMMs, accumulated
    on the aux stream as each tap lands; the last tap writes the KV."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, hidden, n_taps = 4, 8, 2
    drafter = _incremental_dflash(enabled=True)
    drafter.input_buffers = SimpleNamespace(
        positions_buf=torch.arange(16, device=device)
    )
    drafter._incr_acc_buf = torch.ones(16, hidden, device=device)
    drafter._incr_slot_bufs = [
        torch.empty(16, hidden, device=device) for _ in range(n_taps)
    ]
    drafter._incr_capture_events = [torch.cuda.Event() for _ in range(n_taps)]
    drafter._incr_sub_weights_t = [
        torch.randn(hidden, hidden, device=device) for _ in range(n_taps)
    ]
    drafter._incr_n_captures = n_taps
    drafter._incr_hidden_norm = lambda acc: acc * 2
    drafter._kv_aux_stream = torch.cuda.Stream()
    drafter._kv_join_event = torch.cuda.Event()
    writes = []
    drafter._write_native_cache_fused = (
        lambda ctx_hidden, positions, locs: writes.append(
            (ctx_hidden.clone(), positions.clone(), locs.clone())
        )
    )
    ctx = _target_ctx(num_extends=0, num_tokens=num_tokens)
    ctx.attn_backend = SimpleNamespace(
        decode_window_locations=lambda: torch.arange(100, 132, device=device)
    )
    taps = [torch.randn(num_tokens, hidden, device=device) for _ in range(n_taps)]

    drafter.prepare_target_forward(ctx)
    assert ctx.target_capture_sink is drafter
    ctx.target_capture_sink.on_target_capture(0, taps[0])
    assert writes == []
    ctx.target_capture_sink.on_target_capture(1, taps[1])
    torch.cuda.synchronize()

    expected = (
        taps[0] @ drafter._incr_sub_weights_t[0]
        + taps[1] @ drafter._incr_sub_weights_t[1]
    )
    torch.testing.assert_close(drafter._incr_acc_buf[:num_tokens], expected)
    assert drafter._incremental_kv_write_done is True
    ((ctx_hidden, positions, locs),) = writes
    torch.testing.assert_close(ctx_hidden, expected * 2)
    assert torch.equal(positions, torch.arange(num_tokens, device=device))
    assert torch.equal(locs, torch.arange(100, 100 + num_tokens, device=device))


@pytest.mark.parametrize("producer_owned", [False, True])
@pytest.mark.parametrize(
    "num_extends,accept,capturing",
    [(1, 1, False), (0, 2, False), (0, 0, False), (0, 2, True), (0, 0, True)],
)
def test_context_cache_ownership_preserves_prefix_lengths(
    monkeypatch, producer_owned, num_extends, accept, capturing
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: capturing)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    drafter = DFlash.__new__(DFlash)
    drafter.spec_num_tokens = 4
    drafter.input_buffers = SimpleNamespace(
        input_lengths_buf=torch.tensor([4]),
        req_pool_indices_buf=torch.tensor([0]),
        positions_buf=torch.arange(10, 14),
    )
    drafter.runtime_states = SimpleNamespace(valid_cache_lengths=torch.tensor([10]))
    drafter.draft_seq_lens_buf = torch.zeros(1, dtype=torch.int32)
    drafter._write_native_cache = mock.Mock()
    locations = torch.arange(4)
    backend = SimpleNamespace(
        decode_window_locations=lambda: (
            locations if not num_extends else torch.empty(0, dtype=torch.int64)
        ),
        extend_span_locations=lambda: locations,
    )
    ctx = SimpleNamespace(
        dspark_context_producer=object() if producer_owned else None,
        bs=1,
        num_extends=num_extends,
        input_num_tokens=4,
        # A producer-owned update must not inspect write locations again.
        attn_backend=object() if producer_owned else backend,
    )
    hidden = torch.zeros(4, 8)
    logits = object() if producer_owned else SimpleNamespace(hidden_states=hidden)
    drafter._update_native_cache_from_target(ctx, logits, torch.tensor([accept]))
    assert drafter.draft_seq_lens_buf.item() == (14 if num_extends else 10 + accept)
    if producer_owned:
        drafter._write_native_cache.assert_not_called()
    else:
        args, kwargs = drafter._write_native_cache.call_args
        assert drafter._write_native_cache.call_count == 1
        assert args[0] is hidden
        torch.testing.assert_close(args[1], torch.arange(10, 14))
        torch.testing.assert_close(args[2], locations)
        assert kwargs == {"decode_only": num_extends == 0}


@pytest.mark.parametrize(
    "hidden,error",
    [(None, "requires target hidden"), (torch.zeros(3, 8), "token mismatch")],
)
def test_drafter_owned_context_requires_matching_target_hidden(hidden, error):
    drafter = DFlash.__new__(DFlash)
    drafter._update_draft_prefix_lengths = mock.Mock()
    drafter._write_native_cache = mock.Mock()
    ctx = SimpleNamespace(dspark_context_producer=None, input_num_tokens=4)
    with pytest.raises(RuntimeError, match=error):
        drafter._update_native_cache_from_target(
            ctx, SimpleNamespace(hidden_states=hidden), torch.tensor([1])
        )
    drafter._update_draft_prefix_lengths.assert_called_once()
    drafter._write_native_cache.assert_not_called()


@pytest.mark.parametrize("has_pp", [False, True])
def test_k3_setup_capture_survives_resource_binding(monkeypatch, has_pp):
    from tokenspeed.runtime.models import kimi_k3_dspark

    draft = kimi_k3_dspark.K3DSparkModel.__new__(kimi_k3_dspark.K3DSparkModel)
    torch.nn.Module.__init__(draft)
    draft.config = SimpleNamespace(aux_hidden_stream="attn_res")
    draft.target_capture_layer_ids = (2, 5)
    draft.hidden_size = 8
    # Only a pipeline splits the taps across stages; off the pipeline the
    # drafter keeps projecting and writing the context itself.
    draft.mapping = SimpleNamespace(has_pp=has_pp)
    validation = mock.Mock()
    monkeypatch.setattr(kimi_k3_dspark, "validate_k3_dspark_config", validation)
    target = SimpleNamespace(
        get_input_embeddings=lambda: "embedding",
        lm_head="head",
        logits_processor="processor",
        set_target_context_capture=mock.Mock(),
        set_dflash_layers_to_capture=mock.Mock(),
        set_dflash_aux_hidden_stream=mock.Mock(),
    )
    target_config = object()
    monkeypatch.setattr(factory, "get_drafter_impl", lambda algo, model: DSpark)
    factory.configure_draft_target(
        _server_args("DSPARK"),
        SimpleNamespace(
            model=target, model_config=SimpleNamespace(hf_text_config=target_config)
        ),
        _draft_runner(draft),
    )
    validation.assert_called_once_with(draft.config, target_config)
    if has_pp:
        target.set_target_context_capture.assert_called_once_with([2, 5], "attn_res", 8)
        target.set_dflash_layers_to_capture.assert_not_called()
    else:
        target.set_dflash_layers_to_capture.assert_called_once_with([2, 5])
        target.set_dflash_aux_hidden_stream.assert_called_once_with("attn_res")
        target.set_target_context_capture.assert_not_called()
    for setter in (
        target.set_target_context_capture,
        target.set_dflash_layers_to_capture,
        target.set_dflash_aux_hidden_stream,
    ):
        setter.side_effect = AssertionError(
            "resource binding must not reconfigure capture"
        )
    drafter = DSpark.__new__(DSpark)
    drafter.model = draft
    drafter.wire_target(target)
    assert drafter.embed_tokens == "embedding"
    assert drafter.lm_head == "head"


@pytest.mark.parametrize("family", ["dflash", "dflash2", "dspark"])
def test_ordinary_block_models_declare_capture_setup(family):
    from tokenspeed.runtime.models.dflash import DFlashDraftModel
    from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel
    from tokenspeed.runtime.models.dspark import DSparkDraftModel
    from tokenspeed.runtime.models.target_capture import TargetCaptureConfigurator

    model_class = {
        "dflash": DFlashDraftModel,
        "dflash2": DFlash2DraftModel,
        "dspark": DSparkDraftModel,
    }[family]
    draft = model_class.__new__(model_class)
    torch.nn.Module.__init__(draft)
    draft.config = SimpleNamespace(dflash_config={"target_layer_ids": [2, 7]})
    target = SimpleNamespace(set_dflash_layers_to_capture=mock.Mock())
    assert isinstance(draft, TargetCaptureConfigurator)
    draft.configure_target(target, None)
    target.set_dflash_layers_to_capture.assert_called_once_with([2, 7])


@pytest.mark.parametrize("family", ["v4", "v41"])
def test_deepseek_block_models_configure_capture_through_common_setup(
    monkeypatch, family
):
    from tokenspeed.runtime.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
    from tokenspeed.runtime.models.deepseek_v41_dspark import (
        DeepseekV41ForCausalLMDSpark,
    )
    from tokenspeed.runtime.models.target_capture import TargetCaptureConfigurator

    model_class = (
        DeepseekV4ForCausalLMDSpark if family == "v4" else DeepseekV41ForCausalLMDSpark
    )
    draft = model_class.__new__(model_class)
    torch.nn.Module.__init__(draft)
    draft.model = SimpleNamespace(target_layer_ids=(10, 20))
    target = SimpleNamespace(set_dspark_layers_to_capture=mock.Mock())
    assert isinstance(draft, TargetCaptureConfigurator)
    monkeypatch.setattr(
        factory,
        "get_drafter_impl",
        lambda algorithm, model: SimpleNamespace(shares_target_embed_head=False),
    )
    factory.configure_draft_target(
        _server_args("DSPARK"),
        SimpleNamespace(
            model=target, model_config=SimpleNamespace(hf_text_config=object())
        ),
        _draft_runner(draft),
    )
    target.set_dspark_layers_to_capture.assert_called_once_with([10, 20])
    target.set_dspark_layers_to_capture.side_effect = AssertionError(
        "resource binding must not reconfigure capture"
    )
    draft.lm_head = object()
    target.logits_processor = SimpleNamespace(tp_group=(0, 1))
    drafter_class = DeepseekV4DSpark if family == "v4" else DeepseekV41DSpark
    drafter = drafter_class.__new__(drafter_class)
    drafter.draft_model = draft
    drafter.wire_target(target)
    assert drafter.target_model is target
    assert drafter.lm_head is draft.lm_head
    assert drafter.tp_group == (0, 1)
    target.set_dspark_layers_to_capture.assert_called_once_with([10, 20])


# ---------------------------------------------------------------------------
# Draft model setup: capture configuration happens once per stage, before
# any drafter exists (folded in from the former test_draft_model_setup.py).
# ---------------------------------------------------------------------------


class _BlockDraft(TargetCaptureConfigurator):
    def __init__(self, config):
        self.config = config
        self.calls = []

    def configure_target(self, target_model, target_config):
        self.calls.append((target_model, target_config))


@pytest.fixture
def dflash_model():
    from tokenspeed.runtime.models.dflash import DFlashDraftModel

    model = DFlashDraftModel.__new__(DFlashDraftModel)
    torch.nn.Module.__init__(model)
    return model


@pytest.mark.parametrize("algorithm", ["DFLASH", "DSPARK"])
@pytest.mark.parametrize("stage", [0, 1, 3])
def test_setup_configures_every_stage_without_creating_a_drafter(
    monkeypatch, algorithm, stage
):
    implementation = mock.Mock(
        shares_target_embed_head=False,
        side_effect=AssertionError("drafter constructed during model setup"),
    )
    monkeypatch.setattr(factory, "get_drafter_impl", lambda algo, model: implementation)
    target = SimpleNamespace(set_dflash_layers_to_capture=mock.Mock())
    draft = _BlockDraft(
        SimpleNamespace(dflash_config={"target_layer_ids": [2, 23, 47]}, pp_rank=stage)
    )
    target_config = object()
    factory.configure_draft_target(
        SimpleNamespace(speculative_algorithm=algorithm),
        SimpleNamespace(
            model=target, model_config=SimpleNamespace(hf_text_config=target_config)
        ),
        _draft_runner(draft),
    )
    assert draft.calls == [(target, target_config)]
    implementation.assert_not_called()


def test_method_name_alone_does_not_opt_a_model_into_capture_setup(monkeypatch):
    monkeypatch.setattr(
        factory,
        "get_drafter_impl",
        lambda algo, model: SimpleNamespace(shares_target_embed_head=False),
    )
    draft = SimpleNamespace(configure_target=mock.Mock())
    with pytest.raises(TypeError, match="must implement TargetCaptureConfigurator"):
        factory.configure_draft_target(
            SimpleNamespace(speculative_algorithm="DSPARK"),
            SimpleNamespace(model=object()),
            _draft_runner(draft),
        )
    draft.configure_target.assert_not_called()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("stream", ["prefix", "attn_res"])
def test_dflash_model_capture_uses_checkpoint_fields(dflash_model, nested, stream):
    values = {"target_layer_ids": [2, 7], "aux_hidden_stream": stream}
    config = (
        SimpleNamespace(dflash_config=values) if nested else SimpleNamespace(**values)
    )
    target = SimpleNamespace(
        set_dflash_layers_to_capture=mock.Mock(),
        set_dflash_aux_hidden_stream=mock.Mock(),
    )
    dflash_model.config = config
    dflash_model.configure_target(target, None)
    target.set_dflash_layers_to_capture.assert_called_once_with([2, 7])
    target.set_dflash_aux_hidden_stream.assert_called_once_with(stream)


def test_nested_capture_config_overrides_top_level_values(dflash_model):
    target = SimpleNamespace(
        set_dflash_layers_to_capture=mock.Mock(),
        set_dflash_aux_hidden_stream=mock.Mock(),
    )
    config = SimpleNamespace(
        target_layer_ids=[1],
        aux_hidden_stream="prefix",
        dflash_config={"target_layer_ids": [2, 7], "aux_hidden_stream": "ATTN_RES"},
    )
    dflash_model.config = config
    dflash_model.configure_target(target, None)
    target.set_dflash_layers_to_capture.assert_called_once_with([2, 7])
    target.set_dflash_aux_hidden_stream.assert_called_once_with("attn_res")


def test_missing_taps_fail_during_model_setup(dflash_model):
    dflash_model.config = SimpleNamespace()
    with pytest.raises(ValueError, match="target_layer_ids"):
        dflash_model.configure_target(object(), None)


def test_target_without_capture_support_fails_during_model_setup(dflash_model):
    dflash_model.config = SimpleNamespace(target_layer_ids=[1])
    with pytest.raises(ValueError, match="set_dflash_layers_to_capture"):
        dflash_model.configure_target(object(), None)


def test_unsupported_stream_fails_before_installing_capture(dflash_model):
    setter = mock.Mock()
    target = SimpleNamespace(set_dflash_layers_to_capture=setter)
    dflash_model.config = SimpleNamespace(
        target_layer_ids=[1], aux_hidden_stream="attn_res"
    )
    with pytest.raises(ValueError, match="only supply 'prefix'"):
        dflash_model.configure_target(target, None)
    setter.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
