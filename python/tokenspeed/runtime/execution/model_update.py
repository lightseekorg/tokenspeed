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

"""Adapter for the Model Updater SDK behind ``/update_weights_from_mooncake``.

The RL trainer publishes a checkpoint to a Mooncake weight store and names a
committed version; each scheduler process reads its own shard of that version
and loads it into the serving model in place. The SDK is a vendor package the
runtime must not depend on, so it is named by import path
(``--model-update-sdk-module``) and imported on the first update, not at
startup. The module is expected to expose ``make_model_updater``,
``ModelUpdaterConfig``, ``EngineType``, ``MooncakeWeightStore``,
``FluentLlmEngineConfig`` and ``FluentLlmModelUpdateInitConfig``; TokenSpeed
presents itself to the SDK as the reference engine's layout
(``--model-update-engine-type``).

The SDK streams ``(name, tensor)`` pairs into each model's ``load_weights`` in
many partial calls -- it chunks the checkpoint as it reads, so no single call
sees the whole model. The caller therefore wraps the models in a weight-update
session (``models/base/weight_update.py``) so each model derives its post-load
state once, after the last chunk, rather than per chunk on a half-written
model.

``reader_rank`` is the process's global rank: the SDK shards the metadata
bundles across readers by it, and under attention-DP every rank's TP rank may
be 0, so only the global rank tells the readers apart.
"""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

from torch import nn

from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)


def model_update_adapter_for(
    server_args: ServerArgs, *, global_rank: int, is_draft_worker: bool
) -> ModelUpdateAdapter | None:
    """The adapter a ``ModelRunner`` owns, or None.

    Only the target runner holds one: a Mooncake update streams into the
    draft model through the target runner's call, so the draft runner never
    reads the store itself. None as well without ``--model-update-config``.
    """
    if server_args.model_update_config is None or is_draft_worker:
        return None
    return ModelUpdateAdapter(
        sdk_module=server_args.model_update_sdk_module,
        config_json=server_args.model_update_config,
        engine_type=server_args.model_update_engine_type,
        reader_rank=global_rank,
    )


class ModelUpdateAdapter:
    """Lazily built Model Updater SDK client for one scheduler process.

    Construction records the server arguments and nothing else; the SDK is
    imported and the weight store and updater are created on the first
    :meth:`update`, so a server that never receives a Mooncake update never
    needs the SDK installed.
    """

    def __init__(
        self,
        *,
        sdk_module: str,
        config_json: str,
        engine_type: str,
        reader_rank: int,
    ) -> None:
        """
        Args:
            sdk_module: Import path of the Model Updater SDK module.
            config_json: The ``--model-update-config`` JSON object, parsed by
                the SDK's ``FluentLlmModelUpdateInitConfig.from_dict``.
            engine_type: ``EngineType`` member name, matched case-insensitively.
            reader_rank: This process's global rank.
        """
        self._sdk_module = sdk_module
        self._config_json = config_json
        self._engine_type = engine_type
        self._reader_rank = reader_rank
        # The store is kept alive alongside the updater that reads from it.
        self._weight_store: object | None = None
        self._updater: object | None = None

    def _ensure_updater(self) -> object:
        if self._updater is not None:
            return self._updater
        try:
            sdk = importlib.import_module(self._sdk_module)
        except ImportError as exc:
            raise ImportError(
                f"--model-update-sdk-module {self._sdk_module!r} is not importable "
                "in the scheduler process; install the Model Updater SDK or fix "
                f"the module path: {exc}"
            ) from exc
        config = sdk.FluentLlmModelUpdateInitConfig.from_dict(
            json.loads(self._config_json)
        )
        try:
            engine_type = sdk.EngineType[self._engine_type.upper()]
        except KeyError as exc:
            raise ValueError(
                f"--model-update-engine-type {self._engine_type!r} is not a member "
                f"of {self._sdk_module}.EngineType"
            ) from exc
        weight_store = sdk.MooncakeWeightStore(
            config.weight_store_config, local_host=config.local_host
        )
        updater = sdk.make_model_updater(
            sdk.ModelUpdaterConfig(
                role="target",
                reader_rank=self._reader_rank,
                engine_type=engine_type,
                engine_config=sdk.FluentLlmEngineConfig(
                    hf_type=config.hf_type,
                    hf_safetensors_path=config.hf_safetensors_path,
                ),
            ),
            weight_store=weight_store,
        )
        self._weight_store = weight_store
        self._updater = updater
        logger.info(
            f"model updater ready: module={self._sdk_module!s} "
            f"engine_type={engine_type!s} reader_rank={self._reader_rank:d}"
        )
        return updater

    def update(self, models: list[nn.Module], version: int) -> str:
        """Stream one committed version into ``models``.

        Args:
            models: The modules to update, target first. Each receives the
                SDK's partial ``load_weights`` calls; wrap them in a
                weight-update session before calling.
            version: The committed weight-store version to read.

        Returns:
            The SDK's result rendered as a string, for the reply message.
        """
        updater = self._ensure_updater()
        result = updater.update_weights(models, version=version)
        return str(result)
