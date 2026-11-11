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

"""Decode the same Nsight launch fingerprints for Graph and eager analysis."""


def activity_resource_columns(table: str) -> tuple[str, ...]:
    return (
        (
            "mangledName",
            "demangledName",
            "shortName",
            "deviceId",
            "contextId",
            "streamId",
            "registersPerThread",
            "gridX",
            "gridY",
            "gridZ",
            "blockX",
            "blockY",
            "blockZ",
            "staticSharedMemory",
            "dynamicSharedMemory",
            "clusterX",
            "clusterY",
            "clusterZ",
            "graphId",
            "launchType",
        )
        if table == "CUPTI_ACTIVITY_KIND_KERNEL"
        else (
            "bytes",
            "deviceId",
            "contextId",
            "streamId",
            "copyKind",
            "srcKind",
            "dstKind",
            "srcDeviceId",
            "dstDeviceId",
        )
    )


def decode_activity_resources(table, columns, values, strings):
    """Return kernel or memory fields with absent native values kept as None."""
    kernel = None
    memory = None
    if table == "CUPTI_ACTIVITY_KIND_KERNEL":
        data = dict(zip(columns, values, strict=True))
        for name in ("mangledName", "demangledName", "shortName"):
            value = data[name]
            data[name] = value if isinstance(value, str) else strings.get(value)
        kernel = dict(
            name=data["mangledName"],
            demangled_name=data["demangledName"],
            short_name=data["shortName"],
            grid=[data[f"grid{axis}"] for axis in "XYZ"],
            block=[data[f"block{axis}"] for axis in "XYZ"],
            cluster=[data[f"cluster{axis}"] for axis in "XYZ"],
            registers_per_thread=data["registersPerThread"],
            static_shared_memory=data["staticSharedMemory"],
            dynamic_shared_memory=data["dynamicSharedMemory"],
            device_id=data["deviceId"],
            context_id=data["contextId"],
            stream_id=data["streamId"],
            graph_id=data["graphId"],
            launch_type=data["launchType"],
        )
    else:
        data = dict(zip(columns, values, strict=True))
        memory = dict(
            bytes=data["bytes"],
            device_id=data["deviceId"],
            context_id=data["contextId"],
            stream_id=data["streamId"],
            copy_kind=data["copyKind"],
            source_kind=data["srcKind"],
            destination_kind=data["dstKind"],
            source_device_id=data["srcDeviceId"],
            destination_device_id=data["dstDeviceId"],
        )
    return kernel, memory
