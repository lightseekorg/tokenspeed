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

"""Abstract engine interface for runtime entrypoints."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch


class EngineBase(ABC):
    """Abstract base class for engine interfaces.

    This interface covers generation, weight updates, and memory control for
    both HTTP-based adapters and in-process engines.
    """

    @abstractmethod
    def generate(
        self,
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        input_ids: list[list[int]] | list[int] | None = None,
        image_data: list[str] | str | None = None,
        return_logprob: list[bool] | bool | None = False,
        logprob_start_len: list[int] | int | None = None,
        top_logprobs_num: list[int] | int | None = None,
        token_ids_logprob: list[list[int]] | list[int] | None = None,
        custom_logit_processor: list[str] | str | None = None,
        return_hidden_states: bool | None = None,
        stream: bool | None = None,
        bootstrap_host: list[str] | str | None = None,
        bootstrap_port: list[int] | int | None = None,
        bootstrap_room: list[int] | int | None = None,
        data_parallel_rank: int | None = None,
    ) -> dict | Iterator[dict]:
        """Generate outputs based on given inputs."""

    @abstractmethod
    def flush_cache(self) -> None:
        """Flush the cache of the engine."""

    @abstractmethod
    def update_weights_from_tensor(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        load_format: str | None = None,
        flush_cache: bool = True,
    ) -> None:
        """Update model weights with in-memory tensor data."""

    @abstractmethod
    def release_memory_occupation(self) -> None:
        """Release GPU memory occupation temporarily."""

    @abstractmethod
    def resume_memory_occupation(self) -> None:
        """Resume GPU memory occupation which is previously released."""

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the engine and clean up resources."""

    # ------------------------------------------------------------------
    # LoRA adapter management
    # ------------------------------------------------------------------

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: str,
        pinned: bool = False,
    ) -> None:
        """Load a LoRA adapter into GPU memory and register it under ``lora_name``.

        Args:
            lora_name: Short identifier used in subsequent requests
                       (``GenerateReqInput.lora_path = lora_name``).
            lora_path: Filesystem path to the PEFT adapter directory containing
                       ``adapter_config.json`` and ``adapter_model.safetensors``.
            pinned:    If True the adapter is never evicted from GPU memory even
                       when ``max_loras`` resident adapters are exceeded.

        Raises:
            NotImplementedError: Until the full implementation is complete.
            ValueError:          If the server was not started with --enable-lora.
        """
        raise NotImplementedError(
            "LoRA adapter loading is not yet implemented. "
            "Track progress at https://github.com/qywu/tokenspeed/pull/2"
        )

    def unload_lora_adapter(self, lora_name: str) -> None:
        """Unload a previously loaded LoRA adapter and free its GPU memory.

        Args:
            lora_name: The name used when the adapter was loaded.

        Raises:
            NotImplementedError: Until the full implementation is complete.
            KeyError:            If ``lora_name`` is not currently loaded.
        """
        raise NotImplementedError(
            "LoRA adapter unloading is not yet implemented. "
            "Track progress at https://github.com/qywu/tokenspeed/pull/2"
        )
