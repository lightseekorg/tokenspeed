"""Manual E2E checklist: TokenSpeed KV events ↔ Dynamo standalone indexer.

This module is a **skip-by-default** integration stub. Dynamo is not installed
in TokenSpeed CI. Run the checklist below on a machine that has:

- TokenSpeed built with Mooncake Store support
- NVIDIA Dynamo standalone indexer (`python -m dynamo.indexer`)
- A Mooncake master reachable at ``HOST:PORT``

Docs: ``docs/configuration/server.md`` (KV Cache Events → Dynamo standalone
indexer). Contract references:

- Dynamo PR #8912 (multi-tier ``/query`` ``instances`` map)
- Mooncake RFC #1403 (gpu / cpu / disk cumulative tier fields)

Manual checklist
----------------
1. Start Mooncake master and note ``master_server_address``.
2. Launch TokenSpeed with RFC #1527 envelopes and all tiers::

       --kv-events-config '{
         "enable_kv_cache_events":true,
         "publisher":"zmq",
         "endpoint":"tcp://*:5557",
         "wire_format":"rfc1527",
         "backend_id":"ts-worker-0",
         "publish_tiers":["gpu","cpu","disk"],
         "hash_mode":"xxh3"
       }'
       --kvstore-storage-backend mooncake
       --kvstore-storage-backend-extra-config '{"master_server_address":"HOST:PORT"}'

3. Start the Dynamo indexer subscribed to the worker PUB endpoint::

       python -m dynamo.indexer --zmq-endpoint tcp://ts-worker-0:5557

   (or register via ``POST /register`` / ``--workers`` per your Dynamo version.)

4. Drive prompts that land prefixes on GPU, host (CPU), and Mooncake (disk).
5. ``POST /query`` with the same ``token_ids`` / ``model_name`` and assert the
   ``instances`` entry for the worker shows cumulative tier reach::

       gpu ≤ cpu ≤ disk
       longest_matched == max(gpu, cpu, disk)

   See the example response shape in ``docs/configuration/server.md``.
6. Confirm a single ``instances`` key for the shared ``backend_id`` (no
   duplicate workers from multi-tier publishes).

To execute this stub locally after Dynamo is available, remove the
module-level ``pytest.mark.skip`` and flesh out the HTTP assertions against a
live indexer.
"""

from __future__ import annotations

import pytest

# Not run in CI: Dynamo standalone indexer is an optional external dependency.
pytestmark = pytest.mark.skip(
    reason=(
        "Manual E2E stub: requires Dynamo standalone indexer "
        "(python -m dynamo.indexer), not installed in CI. "
        "See module docstring checklist."
    )
)


def test_dynamo_indexer_query_reports_cumulative_tier_breakdown() -> None:
    """Placeholder for live Dynamo ``POST /query`` tier assertions.

    When unskipped, this test should:

    1. Ensure the indexer has ingested gpu/cpu/disk ``BlockStored`` events for
       a shared ``backend_id``.
    2. Call ``POST /query`` (or ``/query_by_hash``) with known prompt tokens.
    3. Assert ``instances[<id>].gpu <= .cpu <= .disk`` and
       ``longest_matched == max(gpu, cpu, disk)`` per Dynamo PR #8912 /
       Mooncake RFC #1403.
    """
    raise NotImplementedError(
        "Populate against a live dynamo.indexer process; see module docstring."
    )
