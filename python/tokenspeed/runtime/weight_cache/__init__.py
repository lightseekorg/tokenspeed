# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Intentionally light: importing a weight_cache submodule (e.g.
# ``tokenspeed.runtime.weight_cache.protocol``) executes this package __init__
# first. Eagerly re-exporting daemon/ipc_loader here would pull in torch and the
# model loader on that cheap protocol import, re-introducing the circular-import
# and startup-cost problems the local-import layout avoids. Import the concrete
# symbols from their submodules instead, e.g.
#     from tokenspeed.runtime.weight_cache.protocol import CacheConfig
#     from tokenspeed.runtime.weight_cache.daemon import launch_weight_cache_daemons
#     from tokenspeed.runtime.weight_cache.ipc_loader import IpcModelLoader
