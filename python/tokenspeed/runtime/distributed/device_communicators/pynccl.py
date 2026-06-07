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


from contextlib import contextmanager

# ===================== import region =====================
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.nccl import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from torch.distributed import ProcessGroup, ReduceOp

from tokenspeed.runtime.distributed.utils import StatelessProcessGroup
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class PyNcclCommunicator:

    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert (
                dist.get_backend(group) != dist.Backend.NCCL
            ), "PyNcclCommunicator should be attached to a non-NCCL group."
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            self.stream = None
            return
        try:
            self.nccl = NCCLLibrary(library_path)
        except Exception:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            self.stream = None
            return

        self.available = True
        self.disabled = False

        logger.info("Epsilon is using nccl==%s", self.nccl.ncclGetVersion())

        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = ncclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # nccl communicator and stream will use this device
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )
            self.stream = torch.cuda.Stream()

            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            self.stream.synchronize()
            del data

        # by default it is disabled, e.g. in profiling models and prefill phase.
        # to use it, use under `with obj.change_state(enable=True)`, usually
        # when we are using CUDA graph.
        self.disabled = True

    def all_reduce(
        self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = self.stream
        self.nccl.ncclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = self.stream
        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = self.stream
        self.nccl.ncclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = self.stream
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = self.stream
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = self.stream
        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self.nccl.ncclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    @contextmanager
    def change_state(
        self, enable: bool | None = None, stream: torch.cuda.Stream | None = None
    ):
        """
        A context manager to change the state of the communicator.
        """
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        if stream is None:
            stream = self.stream

        old_disable = self.disabled
        old_stream = self.stream

        self.stream = stream
        self.disabled = not enable
        yield

        self.disabled = old_disable
        self.stream = old_stream
