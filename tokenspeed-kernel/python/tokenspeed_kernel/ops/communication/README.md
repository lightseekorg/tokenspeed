# Peer-memory collectives

Native Lamport collectives keep persistent peer buffers, allocated before CUDA
Graph capture. Local groups use CUDA IPC. Groups spanning hosts use symmetric
memory when every peer has verified NVLink fabric access. Gather the world fabric
map during distributed initialization before constructing cross-host workspaces.
The same gather, scatter, and fused projection kernels operate on either mapping.

Workspaces belong to one serialized stream of calls. Keep their allocations and
peer mappings alive until every graph and consumer has retired; release them
collectively after device synchronization. A node boundary alone must not disable
fused projection paths when peer memory is reachable over the fabric.

`test/nvidia/ops/communication/test_shared_buffers_distributed.py` checks real
collectives, changing inputs, graph replay and collective reallocation. Launch it
with torchrun over two or four ranks on one host or across a supported NVLink
fabric. Unsupported cross-host groups must use a transport that supports their
topology instead of attempting CUDA IPC imports.
