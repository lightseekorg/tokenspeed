# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.

# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

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

import json
import multiprocessing as mp
import os
import queue
from dataclasses import dataclass
from test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, calculate_rouge_l
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from tokenspeed.runtime.entrypoints.engine import Engine
from tokenspeed.runtime.utils import get_device
from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer

DEFAULT_PROMPTS = [
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
    # the output of gemma-2-2b from SRT is unstable on the commented prompt
    # "The capital of France is",
]
dirpath = os.path.dirname(__file__)
with open(os.path.join(dirpath, "long_prompt.txt"), "r") as f:
    long_prompt = f.read()
DEFAULT_PROMPTS.append(long_prompt)

NUM_TOP_LOGPROBS = 5


def get_dtype_str(torch_dtype):
    if torch_dtype is torch.float16:
        return "float16"
    if torch_dtype is torch.float32:
        return "float32"
    if torch_dtype is torch.bfloat16:
        return "bfloat16"
    else:
        raise NotImplementedError()


def get_top_logprobs(logits, k):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    return torch.topk(logprobs, k=k, dim=-1).values


def get_token_ids_logprobs(logits, token_ids):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    logprobs = logprobs[..., token_ids]
    return logprobs


@dataclass
class ModelOutput:
    output_strs: List[str] = None
    output_ids: List[int] = None
    top_input_logprobs: List[torch.Tensor] = None
    top_output_logprobs: List[torch.Tensor] = None
    top_output_logprob_idx: List[List[int]] = None
    embed_logits: List[torch.Tensor] = None
    scores: List[float] = None
    input_token_logprobs_lst: List[List[Tuple[float, int, None]]] = None
    output_token_logprobs_lst: List[List[Tuple[float, int, None]]] = None
    token_ids_input_logprobs: List[torch.Tensor] = None
    token_ids_output_logprobs: List[torch.Tensor] = None


class HFRunner:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str = "generation",
        output_str_only: bool = False,
        trust_remote_code: bool = False,
        patch_model_do_sample_false: bool = False,
        matryoshka_dim: Optional[int] = None,
        tp_size: int = 1,
        max_model_len: Optional[int] = None,
    ):
        self.model_type = model_type
        self.output_str_only = output_str_only
        self.trust_remote_code = trust_remote_code
        self.patch_model_do_sample_false = patch_model_do_sample_false
        self.tp_size = tp_size
        self.max_model_len = max_model_len

        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()

        self.model_proc = mp.Process(
            target=self.start_model_process,
            args=(
                self.in_queue,
                self.out_queue,
                model_path,
                torch_dtype,
                matryoshka_dim,
                tp_size,
                max_model_len,
            ),
        )
        self.model_proc.start()

    def start_model_process(
        self,
        in_queue,
        out_queue,
        model_path,
        torch_dtype,
        matryoshka_dim: Optional[int] = None,
        tp_size: int = 1,
        max_model_len: Optional[int] = None,
    ):
        # Apply model-specific patches
        monkey_patch_gemma2_sdpa()

        # Disable async tensor loading to avoid CUDA illegal memory access in spawned subprocess.
        # Transformers uses a ThreadPoolExecutor to load weights in parallel, which is not safe
        # when CUDA is used from multiple threads in a subprocess started with "spawn".
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

        # Load the model and tokenizer
        if self.model_type == "generation":
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=self.trust_remote_code
            )
            if self.trust_remote_code:
                model_cls = AutoModelForCausalLM
            else:
                model_arch = getattr(config, "architectures")[0]
                model_cls = getattr(transformers, model_arch)

            # HFRunner is for reference outputs only, so load onto a single GPU.
            # Using device_map="auto" with multi-GPU in a spawned subprocess causes
            # cudaErrorIllegalAddress on B200 (CUDA 13.0) when tensors are materialized
            # on non-primary devices during MXFP4 dequantization.
            if tp_size > 1:
                self.base_model = model_cls.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    low_cpu_mem_usage=True,
                    device_map="cuda:0",
                )
            else:
                self.base_model = model_cls.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    low_cpu_mem_usage=True,
                ).to(get_device())
        else:
            raise Exception(f"Unrecognized model type {self.model_type}")

        self.max_model_len = max_model_len
        self.tokenizer = get_tokenizer(
            model_path,
            torch_dtype=torch.dtype,
            trust_remote_code=self.trust_remote_code,
            model_max_length=self.max_model_len,
        )

        # Run forward
        while True:
            prompts, image_data, max_new_tokens, lora_paths, token_ids_logprob = (
                in_queue.get()
            )
            if lora_paths is not None:
                assert len(prompts) == len(lora_paths)

            if prompts is not None:
                if self.model_type == "generation":
                    out_queue.put(
                        self.forward_generation_raw(
                            base_model=self.base_model,
                            prompts=prompts,
                            max_new_tokens=max_new_tokens,
                            tokenizer=self.tokenizer,
                            lora_paths=lora_paths,
                            torch_dtype=torch_dtype,
                            output_str_only=self.output_str_only,
                            token_ids_logprob=token_ids_logprob,
                            patch_model_do_sample_false=self.patch_model_do_sample_false,
                            max_model_len=self.max_model_len,
                        )
                    )
                else:
                    raise Exception(f"Unrecognized model type {self.model_type}")

    def forward(
        self,
        prompts: Union[
            List[List[str]], List[str], List[torch.Tensor]
        ] = DEFAULT_PROMPTS,
        image_data: Optional[List[str]] = None,
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        token_ids_logprob: Optional[int] = None,
    ):
        self.in_queue.put(
            (prompts, image_data, max_new_tokens, lora_paths, token_ids_logprob)
        )
        while True:
            try:
                return self.out_queue.get(timeout=10)
            except queue.Empty:
                if not self.model_proc.is_alive():
                    raise RuntimeError(
                        f"HFRunner subprocess died with exit code "
                        f"{self.model_proc.exitcode} (likely OOM). "
                        f"Check GPU memory availability."
                    )

    def terminate(self):
        self.model_proc.terminate()
        self.model_proc.join(timeout=10)
        if self.model_proc.is_alive():
            self.model_proc.kill()
            self.model_proc.join(timeout=5)
        self.in_queue = self.out_queue = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    @staticmethod
    def forward_generation_raw(
        base_model,
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens: int,
        tokenizer,
        torch_dtype: torch.dtype,
        lora_paths: Optional[List[str]] = None,
        output_str_only: bool = False,
        token_ids_logprob: Optional[int] = None,
        patch_model_do_sample_false: Optional[bool] = False,
        max_model_len: Optional[int] = None,
    ) -> ModelOutput:
        output_strs = []
        top_input_logprobs = []
        top_output_logprobs = []
        if token_ids_logprob is not None:
            token_ids_input_logprobs = []
            token_ids_output_logprobs = []
        else:
            token_ids_input_logprobs = token_ids_output_logprobs = None

        for i, p in enumerate(prompts):
            if isinstance(p, str):
                # Apply max_model_len truncation if specified
                if max_model_len is not None:
                    input_ids = tokenizer.encode(
                        p,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_model_len,
                    ).to(get_device())
                else:
                    input_ids = tokenizer.encode(p, return_tensors="pt").to(
                        get_device()
                    )
            else:
                input_ids = torch.tensor([p], device=get_device())
                # Apply max_model_len truncation for tensor input
                if max_model_len is not None and input_ids.shape[1] > max_model_len:
                    input_ids = input_ids[:, :max_model_len]

            if lora_paths is not None and lora_paths[i] is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    base_model,
                    lora_paths[i],
                    torch_dtype=torch_dtype,
                    is_trainable=False,
                )
            else:
                model = base_model

            if patch_model_do_sample_false:
                model.generation_config.do_sample = False
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=(not output_str_only),
                    # make sure to disable compile
                    disable_compile=True,
                ),
            )

            text = tokenizer.decode(
                outputs[0][0][len(input_ids[0]) :], skip_special_tokens=True
            )

            # Check if the text is empty or only whitespace.
            if not text.strip():
                raise ValueError(
                    "Received an empty text response. Please verify your input or model configuration."
                )
            output_strs.append(text)

            if not output_str_only:
                # outputs.scores: (num_token, 1, vocab_size)
                top_output_logprobs.append(
                    [
                        get_top_logprobs(logits[0], NUM_TOP_LOGPROBS).tolist()
                        for logits in outputs.scores
                    ]
                )
                if token_ids_logprob is not None:
                    token_ids_output_logprobs.append(
                        [
                            get_token_ids_logprobs(
                                logits[0], token_ids_logprob
                            ).tolist()
                            for logits in outputs.scores
                        ]
                    )
                del outputs

                input_logits = model.forward(input_ids).logits[0]
                top_input_logprobs.append(
                    get_top_logprobs(input_logits, NUM_TOP_LOGPROBS).tolist()
                )
                if token_ids_logprob is not None:
                    token_ids_input_logprobs.append(
                        get_token_ids_logprobs(input_logits, token_ids_logprob).tolist()
                    )
                del input_logits

            if lora_paths is not None and lora_paths[i] is not None:
                # Unload the LoRA adapter if it is used
                model.unload()

        return ModelOutput(
            output_strs=output_strs,
            top_input_logprobs=top_input_logprobs,
            top_output_logprobs=top_output_logprobs,
            token_ids_input_logprobs=token_ids_input_logprobs,
            token_ids_output_logprobs=token_ids_output_logprobs,
        )


class RTRunner:
    _port_counter = 0  # Class-level port counter

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str,
        world_size: int = 1,
        ep_size: int = 1,
        port: int = None,  # None means auto-increment
        attention_backend: Optional[str] = None,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = True,
        chunked_prefill_size: Optional[int] = None,
        max_model_len: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
        block_size: Optional[int] = 64,
        data_parallel_size: int = 1,
        tokenizer: Optional[str] = None,
        gpu_memory_utilization: float = 0.65,
        trust_remote_code: bool = False,
        speculative_draft_model_path: Optional[str] = None,
        speculative_algorithm: Optional[str] = None,
        speculative_num_steps: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
        disable_overlap_schedule: bool = False,
        disable_custom_all_reduce: bool = False,
        max_cudagraph_capture_size: int = 4,
        hf_overrides: Optional[dict[str, Any]] = None,
        disable_prefill_graph: bool = False,
        **kwargs,
    ):
        # Auto-assign port if not specified
        if port is None:
            port = DEFAULT_PORT_FOR_SRT_TEST_RUNNER + RTRunner._port_counter
            RTRunner._port_counter += 1

        self.model_type = model_type
        self.is_generation = model_type == "generation"
        if not self.is_generation:
            raise ValueError("Embedding, rerank, and reward model runners are removed.")

        spec_kwargs = {}
        if speculative_draft_model_path:
            spec_kwargs["speculative_draft_model_path"] = speculative_draft_model_path
            spec_kwargs["speculative_algorithm"] = speculative_algorithm
            spec_kwargs["speculative_num_steps"] = speculative_num_steps
            spec_kwargs["speculative_eagle_topk"] = speculative_eagle_topk
            spec_kwargs["speculative_num_draft_tokens"] = speculative_num_draft_tokens

        self.engine = Engine(
            model=model_path,
            world_size=world_size,
            ep_size=ep_size,
            dtype=get_dtype_str(torch_dtype),
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            attention_backend=attention_backend,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            chunked_prefill_size=chunked_prefill_size,
            max_model_len=max_model_len,
            max_total_tokens=max_total_tokens,
            block_size=block_size,
            data_parallel_size=data_parallel_size,
            tokenizer=tokenizer,
            disable_overlap_schedule=disable_overlap_schedule,
            max_cudagraph_capture_size=max_cudagraph_capture_size,
            disable_custom_all_reduce=disable_custom_all_reduce,
            hf_overrides=(json.dumps(hf_overrides) if hf_overrides else "{}"),
            disable_prefill_graph=disable_prefill_graph,
            **spec_kwargs,
            **kwargs,
        )

        if tokenizer is None:
            self.tokenizer = get_tokenizer(
                model_path, trust_remote_code=trust_remote_code
            )
        else:
            self.tokenizer = None

    def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
        return self.engine.load_lora_adapter(lora_name, lora_path, pinned)

    def unload_lora_adapter(self, lora_name: str):
        return self.engine.unload_lora_adapter(lora_name)

    def forward(
        self,
        prompts: Union[
            List[List[str]], List[str], List[torch.Tensor]
        ] = DEFAULT_PROMPTS,
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        logprob_start_len: int = 0,
        top_k: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
    ):
        if self.is_generation:
            return self.forward_generation_raw(
                engine=self.engine,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                lora_paths=lora_paths,
                logprob_start_len=logprob_start_len,
                top_k=top_k,
                token_ids_logprob=token_ids_logprob,
            )
        else:
            raise ValueError("Embedding, rerank, and reward model runners are removed.")

    def batch_forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        max_new_tokens=8,
    ):
        """
        testing serving by sending all prompts once
        only return output strings and no logprobs
        """
        if self.is_generation:
            return self.batch_forward_generation_raw(
                engine=self.engine,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError("Embedding, rerank, and reward model runners are removed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.shutdown()
        del self.engine

    @staticmethod
    def forward_generation_raw(
        engine: Engine,
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        logprob_start_len: int = 0,
        top_k: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
    ):
        # the return value contains logprobs from prefill
        output_strs = []
        output_ids = []
        # Input logprobs. Note that the last item in input logprob is equivalent to
        # the first item in the output logprob.
        top_input_logprobs = []
        input_token_logprobs_lst = []
        top_output_logprobs = []
        output_token_logprobs_lst = []
        top_output_logprob_idx = []
        if token_ids_logprob is not None:
            token_ids_input_logprobs = []
            token_ids_output_logprobs = []
        else:
            token_ids_input_logprobs = token_ids_output_logprobs = None

        sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
        if top_k:
            sampling_params["top_k"] = top_k

        for i, prompt in enumerate(prompts):
            response = engine.generate(
                prompt,
                sampling_params=sampling_params,
                return_logprob=True,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=NUM_TOP_LOGPROBS,
                token_ids_logprob=token_ids_logprob,
            )
            text = response["text"]

            # Check if the text is empty or only whitespace.
            if not text.strip():
                raise ValueError(
                    "Received an empty text response. Please verify your input or model configuration."
                )
            output_strs.append(text)
            output_ids.append(response["output_ids"])

            input_token_logprobs = response["meta_info"]["input_token_logprobs"]
            output_token_logprobs = response["meta_info"]["output_token_logprobs"]
            # print(i, input_token_logprobs)
            # print(i, output_token_logprobs)
            logprobs = response["meta_info"]["input_top_logprobs"]
            if token_ids_logprob is not None:
                input_token_ids_logprobs = response["meta_info"][
                    "input_token_ids_logprobs"
                ][1:]
            else:
                input_token_ids_logprobs = None

            num_prompt_tokens = response["meta_info"]["prompt_tokens"]
            # assert len(input_token_logprobs) == num_prompt_tokens - logprob_start_len
            assert len(logprobs) == num_prompt_tokens - logprob_start_len

            # The first token logprob has no meaning in tokenspeed.
            input_token_logprobs = input_token_logprobs[1:]
            logprobs = logprobs[1:]
            assert len(input_token_logprobs) == len(logprobs)

            input_token_logprobs_lst.append(
                input_token_logprobs + [output_token_logprobs[0]]
            )
            output_token_logprobs_lst.append(output_token_logprobs)

            top_input_logprobs.append(
                [[tup[0] for tup in x[:NUM_TOP_LOGPROBS]] for x in logprobs]
                + [
                    [
                        tup[0]
                        for tup in response["meta_info"]["output_top_logprobs"][0][
                            :NUM_TOP_LOGPROBS
                        ]
                    ]
                ]
            )
            top_output_logprobs.append(
                [
                    [tup[0] for tup in x[:NUM_TOP_LOGPROBS]]
                    for x in response["meta_info"]["output_top_logprobs"]
                ]
            )
            top_output_logprob_idx.append(
                [
                    [tup[1] for tup in x[:NUM_TOP_LOGPROBS]]
                    for x in response["meta_info"]["output_top_logprobs"]
                ]
            )
            if token_ids_logprob is not None:
                token_ids_input_logprobs.append(
                    [[tup[0] for tup in x] for x in input_token_ids_logprobs]
                    + [
                        [
                            tup[0]
                            for tup in response["meta_info"][
                                "output_token_ids_logprobs"
                            ][0]
                        ]
                    ]
                )
                token_ids_output_logprobs.append(
                    [
                        [tup[0] for tup in x]
                        for x in response["meta_info"]["output_token_ids_logprobs"]
                    ]
                )

        return ModelOutput(
            output_strs=output_strs,
            output_ids=output_ids,
            top_input_logprobs=top_input_logprobs,
            top_output_logprobs=top_output_logprobs,
            input_token_logprobs_lst=input_token_logprobs_lst,
            output_token_logprobs_lst=output_token_logprobs_lst,
            top_output_logprob_idx=top_output_logprob_idx,
            token_ids_input_logprobs=token_ids_input_logprobs,
            token_ids_output_logprobs=token_ids_output_logprobs,
        )

    @staticmethod
    def batch_forward_generation_raw(
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens,
        engine,
    ):
        # the return value contains logprobs from prefill
        output_strs = []
        sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
        response = engine.generate(
            prompts,
            sampling_params=sampling_params,
        )
        output_strs = [r["text"] for r in response]

        return ModelOutput(
            output_strs=output_strs,
        )


def monkey_patch_gemma2_sdpa():
    """
    Use sdpa by default to fix the OOM issue.
    Revert this commit:
    https://github.com/huggingface/transformers/commit/975b988bfe6e7ebb47390cd9a1556c6888804883#diff-5f76eac6f18f4b491521314c318a9692318feb4d19228e9576cce7bde4240834R660
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2PreTrainedModel

    def _check_and_enable_sdpa(config, hard_check_only: bool = False):
        config._attn_implementation = "sdpa"
        return config

    setattr(Gemma2PreTrainedModel, "_check_and_enable_sdpa", _check_and_enable_sdpa)


def check_close_model_outputs(
    hf_outputs: ModelOutput,
    rt_outputs: ModelOutput,
    prefill_tolerance: float,
    decode_tolerance: float,
    rouge_l_tolerance: float,
    debug_text: str = "",
    check_logprobs: bool = True,
):
    # Compare output strings
    print(f"{hf_outputs.output_strs=}")
    print(f"{rt_outputs.output_strs=}")
    rouge_l_scores = calculate_rouge_l(hf_outputs.output_strs, rt_outputs.output_strs)
    print(f"{rouge_l_scores=}")
    assert all(
        score >= rouge_l_tolerance for score in rouge_l_scores
    ), f"Not all ROUGE-L scores are greater than rouge_l_tolerance={rouge_l_tolerance}"

    if check_logprobs:
        for i in range(len(hf_outputs.output_strs)):
            # Compare input logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_input_logprobs[i])
            srt_logprobs = torch.Tensor(rt_outputs.top_input_logprobs[i])
            input_len = hf_logprobs.shape[0]
            print(
                "prefill logprobs max_diff", torch.max(abs(hf_logprobs - srt_logprobs))
            )
            if input_len <= 100:
                assert torch.all(abs(hf_logprobs - srt_logprobs) < prefill_tolerance), (
                    f"prefill logprobs are not all close with {debug_text} "
                    f"prefill_tolerance={prefill_tolerance}."
                    f"{hf_logprobs=}, {srt_logprobs=}"
                )

            # Compare output logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_output_logprobs[i])
            srt_logprobs = torch.Tensor(rt_outputs.top_output_logprobs[i])

            print(
                "decode logprobs max_diff", torch.max(abs(hf_logprobs - srt_logprobs))
            )
            if input_len <= 100:
                assert torch.all(abs(hf_logprobs - srt_logprobs) < decode_tolerance), (
                    f"decode logprobs are not all close with {debug_text} "
                    f"decode_tolerance={decode_tolerance}."
                    f"{hf_logprobs=}, {srt_logprobs=}"
                )
