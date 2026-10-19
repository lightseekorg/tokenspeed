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

"""Graded CUDA-graph support: declaration, composition, culprit logging.

Backends declare their static graph capability as a class attribute
(``cuda_graph_support``); the executor AND-composes it over the target and
draft ``child_backends()`` trees once at startup and downgrades the graph
subsystems (docs/design/unified_path.md, "Graded CUDA-graph support").
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class _CudaGraphSupportCase:
    def setup_method(self):
        try:
            from tokenspeed.runtime.layers.attention.backends.base import (
                CudaGraphSupport,
                resolve_cuda_graph_support,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs torch: {exc}")
        self.support_cls = CudaGraphSupport
        self.resolve = resolve_cuda_graph_support

    def _stub(self, support=None, children=()):
        support_cls = self.support_cls

        class _Stub:
            cuda_graph_support = support or support_cls()

            def child_backends(self):
                return children

        return _Stub()


class TestResolutionTest(_CudaGraphSupportCase):
    def test_default_is_fully_supported(self):
        support = self.resolve(self._stub())
        assert support.decode_graph
        assert support.prefill_graph

    def test_axes_compose_by_and_and_none_roots_are_skipped(self):
        support = self.resolve(
            self._stub(self.support_cls(prefill_graph=False)),
            None,
            self._stub(self.support_cls(decode_graph=False)),
        )
        assert not (support.decode_graph)
        assert not (support.prefill_graph)

    def test_a_child_declaration_lowers_the_wrapper(self):
        child = self._stub(self.support_cls(prefill_graph=False))
        wrapper = self._stub(children=(child,))
        support = self.resolve(wrapper)
        assert support.decode_graph
        assert not (support.prefill_graph)

    def test_each_culprit_is_logged_by_class_name(self, caplog, monkeypatch):
        child = self._stub(self.support_cls(prefill_graph=False))
        wrapper = self._stub(children=(child,))
        logger_name = "tokenspeed.runtime.layers.attention.backends.support"
        monkeypatch.setattr(logging.getLogger(logger_name), "propagate", True)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=logger_name):
            self.resolve(wrapper)
        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == logger_name and record.levelno >= logging.INFO
        ]
        assert any("_Stub" in line for line in messages)
        assert any("Prefill" in line for line in messages)


class TestBackendDeclarationTest(_CudaGraphSupportCase):
    """Backend graph capabilities are declared on their classes."""

    def test_dsa_declares_no_prefill_graph(self):
        try:
            from tokenspeed.runtime.layers.attention.backends.paged.dsa import (
                DSABackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs tokenspeed_kernel: {exc}")
        assert not (DSABackend.cuda_graph_support.prefill_graph)
        assert DSABackend.cuda_graph_support.decode_graph

    def test_qwen4_exp_request_state_supports_breakable_prefill_graph(self):
        try:
            from tokenspeed.runtime.layers.attention.backends.specific.qsa_indexer import (
                QSAIndexerBackend,
            )
            from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
                Qwen4ExpPLEBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs tokenspeed_kernel: {exc}")
        for backend in (Qwen4ExpPLEBackend, QSAIndexerBackend):
            assert backend.cuda_graph_support.prefill_graph
            assert backend.cuda_graph_support.decode_graph

    def test_declarations_are_class_attributes(self):
        # Rank-uniformity gate: support must not depend on per-rank instance
        # state — every declaration lives on the class.
        for module_name, cls_name in (
            ("tokenspeed.runtime.layers.attention.backends.paged.dsa", "DSABackend"),
            (
                "tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple",
                "Qwen4ExpPLEBackend",
            ),
            (
                "tokenspeed.runtime.layers.attention.backends.specific.qsa_indexer",
                "QSAIndexerBackend",
            ),
        ):
            try:
                cls = getattr(importlib.import_module(module_name), cls_name)
            except (ImportError, ModuleNotFoundError) as exc:
                pytest.skip(f"needs optional deps: {exc}")
            _pytest_member_1 = "cuda_graph_support"
            _pytest_container_2 = vars(cls)
            _pytest_msg_3 = f"{cls_name} must declare cuda_graph_support on the class"
            assert _pytest_member_1 in _pytest_container_2, _pytest_msg_3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
