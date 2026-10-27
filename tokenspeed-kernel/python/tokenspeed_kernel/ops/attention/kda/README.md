# KDA recovery records

KDA verification computes candidate outputs before acceptance selects a token
prefix. The NVIDIA RecoverSSM entrypoints support two explicit record
representations. `retained_history=None` uses the current-round raw records:
FP32 corrections plus activation-dtype keys and raw gates. The retained-history
representation uses cache-page-owned FP32 corrections, normalized keys and
multiplicative decays, with an INT32 accepted length.

Retained history represents a dense recurrent base followed by the accepted
record prefix. Applying `S = S * decay + outer(correction, key)` in token order
reconstructs the accepted endpoint. Unaccepted or stale records beyond the
published length have no meaning and need no clearing.

## Verify and publication

The caller supplies source pages and private live pages. Each active request
must have a distinct live destination. A destination may equal its own source,
but must not alias another active request's source. Multiple requests may read
the same immutable source.

Capacity includes both accepted records and the complete candidate window.
Verify uses the actual packed query length, reserves space after the accepted
prefix, and consolidates that prefix into the dense base only when the window
would exceed capacity. Equality fits. When source and live pages differ,
verify reconstructs the source endpoint into the live base and starts there
with empty history. The source remains unchanged.

The NVIDIA implementation orders conditional reconstruction, length reset and
candidate verification on the same CUDA stream. Reconstruction finishes across
all value tiles before any old record can be overwritten. Candidate stores
then follow the accepted prefix and leave its length unchanged.

After verification completes, the attention commit publishes the old length
plus the accepted count. It does not copy records or write dense state. The
next verify therefore reuses rejected slots immediately. Accepting two of four
candidates written at slots 0–3 makes the next window start at slot 2.

Position/generation stamps describe candidate validity separately from the
cache-owned accepted history. Their ring capacity is independent of history
capacity. All page fields may have arena strides; callers must preserve the
complete checkpoint when moving pages.

`KdaRetainedHistory` validates fixed storage geometry, dtypes and devices at
construction. Reuse it across verify and commit while refreshing tensor
contents in place; rebuild it when rebinding views or changing their metadata.
The entrypoints check the current checkpoint, request and stamp arguments;
GPU kernels check length values and candidate stamps.

## Integration status

This interface implements recurrent attention only. The caller still owns
stream ordering, original QKV retention and convolution-state publication at
the accepted endpoint. Serving integration, batched retained-history commit
and AMD retained-history support are later stages. The production Kimi recipe
keeps `history_capacity=0` and continues to use its existing record workspace
until all checkpoint consumers and memory budgets support retained history.
