"""Translate native HIP stream arguments for Gluon launches."""

import torch


def native_stream(stream, device):
    """None/zero denote HIP's default stream, independent of PyTorch's current stream."""
    if stream is None or isinstance(stream, int) and stream == 0:
        return torch.cuda.default_stream(device)
    if isinstance(stream, int):
        return torch.cuda.ExternalStream(stream, device=device)
    return stream
