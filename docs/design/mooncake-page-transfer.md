# Mooncake page transfer compatibility

The engine uses Mooncake's native `batch_transfer_sync_write_pages` when it is
available. Older builds use `batch_transfer_sync_write`: the adapter expands
page/field metadata into bounded batches, without copying tensor payloads
through host memory. No deployment setting is needed. A partial failure is
returned to the caller without retrying successful writes. IPv6 initialization
and bracketed session addresses follow the upstream implementation.
