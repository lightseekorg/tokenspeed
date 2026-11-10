#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Shared L2 host-memory KV pool: `owner` runs on the head node, `provider` on
# every node. Launched by serve_dwdp16_dep16.sh.
set -euo pipefail
role=${1:?owner or provider}
: "${AGENTX_HEAD:?}"

case "$role" in
    owner)
        exec python3 -m tokenspeed.runtime.cache.shared_l2 owner --address tcp://0.0.0.0:47100 \
            --lcm-bytes 84787200 --providers 16 --headroom-lcm-blocks 680 --client-timeout-s 600
        ;;
    provider)
        host=$(python3 -c 'from tokenspeed.runtime.utils.network import get_local_ip_by_remote as g; print(g())')
        # Two providers per node, each on half of the cores with its own NIC pair.
        # The head node, which also runs the gateway and client, contributes less.
        bytes=360777252864
        [[ "$(hostname -s)" == "$AGENTX_HEAD" ]] && bytes=200000000000
        taskset -c 0-71 python3 -m tokenspeed.runtime.cache.shared_l2 provider --owner "tcp://$AGENTX_HEAD:47100" \
            --hostname "$host" --device-name mlx5_0,mlx5_1 --bytes "$bytes" &
        taskset -c 72-143 python3 -m tokenspeed.runtime.cache.shared_l2 provider --owner "tcp://$AGENTX_HEAD:47100" \
            --hostname "$host" --device-name mlx5_2,mlx5_3 --bytes "$bytes" &
        wait
        ;;
    *)
        echo "unknown role $role" >&2
        exit 2
        ;;
esac
