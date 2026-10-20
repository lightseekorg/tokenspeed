# Stop conditions

`apply_stop_conditions` cuts accepted output widths and marks stopped requests
before state commit. CUDA and Ascend run the device kernel, including graph
capture; CPU uses the tensor reference. Empty output buffers and masked lanes
retain valid addresses, with the original masks deciding which values are used.
The scalar CPU reference reads tensor values on the host, so NPU graph capture
must use the device kernel instead.
