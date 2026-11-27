# EPD encode metrics

EPD encode mode runs a TokenSpeed encode loop instead of the language-model
scheduler. The loop owns admissions, cache decisions, batching, ring-slot
leases and copy completion; the EPD sender owns physical writes to prefill
ranks. Those owners emit through the same ts-metrics Metric exporter,
bounded relay and selected reporter as ordinary engine metrics.

The attention-TP group's first rank reports logical requests, image items,
queues, cache decisions, batching and copy-complete outcomes. Every encode TP
rank reports its physical writes with a tp_rank label. A write timeout is
counted as a transfer failure. Vision and ring-copy failures belong to the
logical failed-item counter and do not increment transfer failures.
The Python bootstrap exposes that group-local rank as the position of its
global rank in the ordered attention-TP group; the native host consumes the
same value when constructing transfer metrics.

ImageCount counts image items in a decoded encode request. The token offset
ranges within one image are not separate images. Encode wire requests usually
carry one item; an original HTTP request with multiple images can create
multiple encode items.

Queue snapshots refresh on state changes and at most once per second while
unchanged. Oldest age reads each FIFO queue head. Execution timing covers
data-plane submission through the already-existing device-copy completion
event; it includes forward-thread queueing and is not a synchronized GPU-only
measurement. Transfer timing covers the successful room-write worker and
excludes the wait for the full peer fanout.

Model plugins can provide EncoderSpec.make_metrics_observer. The optional
factory receives the standard metric labels, enabled flag, reporter selection
and event sink, and returns a nonblocking (phase, items) callback. The forward
thread constructs it once when metrics are enabled on the first TP rank.
Callbacks receive start, admit (including embedding-cache hits), and encoded.
A callback failure disables that observer without failing encoding. Model
counters, metadata formats and metric definitions belong in the model plugin.
The core never interprets model-specific cache fields or private metadata.

Core observations use the same Metric exporter and relay as language-model
serving. External metric names, aggregation and delivery belong to reporters.
