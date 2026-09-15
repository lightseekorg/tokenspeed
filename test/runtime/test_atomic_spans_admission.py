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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tokenspeed.runtime.engine.event_loop import EventLoop
from tokenspeed.runtime.engine.request_handler import _request_atomic_spans
from tokenspeed.runtime.engine.scheduler_utils import (
    make_spec,
    validate_atomic_spans_flat,
)


def _image(offsets):
    return SimpleNamespace(modality=SimpleNamespace(name="IMAGE"), offsets=offsets)


def test_spans_derive_whole_blocks_in_authored_order_only_for_deepseek_vision():
    multimodal = SimpleNamespace(
        mm_items=[
            _image([(20, 25)]),
            SimpleNamespace(modality=SimpleNamespace(name="AUDIO"), offsets=[(1, 2)]),
            _image([(5, 10)]),
        ]
    )

    authored = _request_atomic_spans(multimodal, deepseek_v4_vision_enabled=True)
    disabled = _request_atomic_spans(multimodal, deepseek_v4_vision_enabled=False)
    spec = make_spec("image", list(range(32)), atomic_spans=authored)
    text = make_spec("text", [1, 2, 3])

    assert authored == [(20, 25), (5, 10)]
    assert disabled == []
    assert spec.atomic_spans_flat == [20, 25, 5, 10]
    assert text.atomic_spans_flat == []
    assert not hasattr(spec, "visibility_spans_flat")


@pytest.mark.parametrize(
    ("flat", "num_tokens", "budget", "message"),
    [
        ([0], 8, 8, "even number"),
        ([-1, 2], 8, 8, "0 <= start <= end"),
        ([3, 2], 8, 8, "0 <= start <= end"),
        ([2, 8], 8, 8, "token count"),
        ([1, 3, 3, 5], 8, 8, "strictly ascending and non-overlapping"),
        ([1, 6], 8, 4, "length 6.*--chunked-prefill-size 4"),
    ],
)
def test_spans_validation_names_each_offending_condition(
    flat, num_tokens, budget, message
):
    with pytest.raises(ValueError, match=message):
        validate_atomic_spans_flat(
            flat, num_tokens=num_tokens, max_scheduled_tokens=budget
        )


class _State:
    def __init__(self, *, epd: bool = False) -> None:
        self.finished = False
        self.errors = []
        self.multimodal_inputs = (
            SimpleNamespace(mm_items=[SimpleNamespace(encode_handshake=object())])
            if epd
            else None
        )

    def set_finish_with_abort(self, message):
        self.errors.append(message)
        self.finished = True


class _OutputProcessor:
    def __init__(self) -> None:
        self.rid_to_state = {}
        self.published = []

    def sweep_pending_aborts(self):
        pass

    def register(self, rid, state):
        self.rid_to_state[rid] = state

    def publish_finished_at_admission(self, rid, state):
        self.published.append((rid, list(state.errors)))
        self.rid_to_state.pop(rid, None)


def test_spans_epd_request_is_rejected_before_staging_and_sibling_recovers():
    invalid = make_spec("invalid", list(range(16)), atomic_spans=[(4, 20)])
    valid = make_spec("valid", list(range(16)))
    invalid_state = _State(epd=True)
    valid_state = _State()
    grammar = SimpleNamespace(
        process_req_with_grammar=lambda _state: True,
        get_ready_grammar_requests=lambda: [],
        mark_abort=lambda _rid: None,
    )
    request_handler = SimpleNamespace(
        recv_reqs=lambda: [object()],
        process_requests=lambda _requests: (
            [invalid, valid],
            [invalid_state, valid_state],
            [None, None],
            [],
        ),
        grammar_manager=grammar,
    )
    staged = []
    epd_hooks = SimpleNamespace(
        try_stage=lambda spec, _state, _bootstrap, **_kwargs: staged.append(
            spec.request_id
        )
        or False
    )
    accepted = []
    scheduler = SimpleNamespace(
        submit_requests=lambda specs: accepted.extend(spec.request_id for spec in specs)
    )

    loop = object.__new__(EventLoop)
    loop.request_handler = request_handler
    loop.output_processor = _OutputProcessor()
    loop._pause = SimpleNamespace(admit_blocked=False)
    loop._pause_hooks = SimpleNamespace(
        apply_transitions=lambda _grammar: None,
        withhold_admissions=lambda _entries, _blocked: False,
    )
    loop._epd_hooks = epd_hooks
    loop._device = SimpleNamespace(role=object())
    loop.kv_transfer = None
    loop.scheduler = scheduler
    loop.max_scheduled_tokens = 16

    loop._process_new_requests()

    assert staged == ["valid"]
    assert accepted == ["valid"]
    assert loop.output_processor.published[0][0] == "invalid"
    assert "must end before the request token count 16" in invalid_state.errors[0]
