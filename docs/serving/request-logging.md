Request completion logging
==========================

Metadata-only request logging (level 0) omits prompt text and token IDs. It must not decode a pre-tokenized prompt just to construct a Finish log: that synchronous work delays the final streamed response even after model execution completes. Levels 1 and 2 still decode missing prompt text when it will be logged. This does not change generated output, request cleanup, or structured request statistics.

Use --no-enable-log-requests to disable request text/metadata logs explicitly. Structured --enable-log-request-stats is separate. The regression test test/runtime/test_async_llm_finish_logging.py exercises real response collection and cleanup, metadata logging, and both text-bearing verbosity levels for streaming and nonstreaming requests.
