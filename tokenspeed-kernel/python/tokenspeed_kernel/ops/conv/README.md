# Convolution kernels

The causal prefill kernel preserves its value masks while bounding inactive
token addresses to the current sequence. This also covers suffixes of one or
two tokens when a cached convolution window is resumed. The active token
values, convolution arithmetic and state layout stay the same.
