"""Model recipes for shared cache layouts."""

from __future__ import annotations


def configured_token_limit(server_args) -> int | None:
    """Return the runtime token cap, including the CI-only override."""
    from tokenspeed.runtime.utils.env import envs

    limit = server_args.max_total_tokens
    ci_size = envs.TOKENSPEED_CI_SMALL_KV_SIZE.get_set_value_or(None)
    if ci_size is not None and int(ci_size) > 0:
        ci_limit = int(ci_size)
        limit = ci_limit if limit is None else min(limit, ci_limit)
    return limit
