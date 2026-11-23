# SM120 distributed transport tests

`test_sm120_elastic_distributed.py` exercises the actual Elastic transport
with zero-row peers and explicitly selected EP groups. Invoke it with
`torchrun` with 2, 4, 8, 16 or 32 processes across the intended nodes in a
matching freshly built environment. Set `TOKENSPEED_SM120_DEEPEP_NICS` in
visible-device order when NIC selection must be explicit; this exercises the
same mapping as runtime buffer construction. Two local ranks provide a small
correctness check; EP16 remains a separate full deployment validation.

These tests validate communication agreement and local expert semantics;
full-model timing, real speculation and dataset quality are owned by
[the reproducibility recipes](../../../../.agents/README.md).
