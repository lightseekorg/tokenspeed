"""Fault-test only: restart a local RPC fixture, never a serving/provider process."""

import ctypes
import json
import socket
import subprocess
import sys
import time


def main(library, provider):
    client = ctypes.CDLL(library)
    client.ts_mooncake_offload_call.argtypes = [ctypes.c_char_p] * 3
    client.ts_mooncake_offload_call.restype = ctypes.c_void_p
    client.ts_mooncake_offload_free.argtypes = [ctypes.c_void_p]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def call(**request):
        request["timeout_ms"] = 1000
        if request["mode"] == "prepare":
            request.setdefault("generations", [[0, 0]] * len(request["keys"]))
        ptr = client.ts_mooncake_offload_call(
            b"127.0.0.1", str(port).encode(), json.dumps(request).encode()
        )
        try:
            return json.loads(ctypes.string_at(ptr))
        finally:
            client.ts_mooncake_offload_free(ptr)

    def start():
        process = subprocess.Popen(
            [provider, str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("test provider exited")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        return process
                except OSError:
                    time.sleep(0.02)
            raise TimeoutError("test provider did not listen")
        except BaseException:
            process.terminate()
            process.wait(timeout=5)
            raise

    process = start()
    try:
        query = call(mode="query", tenant="test", keys=["versioned", "legacy"])
        assert query["objects"][0]["ssd"][0]["generation"] == [1, 2], query
        assert query["objects"][1]["ssd"][0]["generation"] == [0, 0], query
        versioned = call(
            mode="prepare",
            tenant="test",
            keys=["versioned"],
            sizes=[16],
            generations=[[1, 2]],
        )
        assert "addresses" in versioned, versioned
        assert call(mode="release", batch_id=versioned["batch_id"])["released"]
        for generation in [[0, 0], [2, 1]]:
            wrong = call(
                mode="prepare",
                tenant="test",
                keys=["versioned"],
                sizes=[16],
                generations=[generation],
            )
            assert wrong.get("remote_error") == -704, wrong
        mismatched = call(
            mode="prepare",
            tenant="test",
            keys=["versioned"],
            sizes=[16],
            generations=[],
        )
        assert "error" in mismatched, mismatched
        # Restart before the lease-lifetime check to make both native ids 1.
        process.terminate()
        process.wait(timeout=5)
        process = start()
        old = call(mode="prepare", tenant="test", keys=["old-generation"], sizes=[16])
        assert "error" not in old, old
        process.terminate()
        process.wait(timeout=5)
        process = start()
        new = call(mode="prepare", tenant="test", keys=["new-generation"], sizes=[16])
        assert "error" not in new, new
        # Both server processes assigned native batch id 1. Old release must
        # fail, and the new process's source must remain leased afterward.
        stale = call(mode="release", batch_id=old["batch_id"])
        assert stale.get("released") is not True, stale
        valid = call(mode="release", batch_id=new["batch_id"])
        assert valid == {"released": True}, valid
        duplicate = call(mode="release", batch_id=new["batch_id"])
        assert duplicate == {"released": False}, duplicate
        print(
            json.dumps(
                {
                    "same_port_provider_restart": "passed",
                    "stale_release_rejected": True,
                    "new_lease_preserved": True,
                    "double_release_rejected": True,
                    "storage_generation_roundtrip": True,
                    "stale_generation_rejected": True,
                    "legacy_generation_supported": True,
                }
            )
        )
    finally:
        process.terminate()
        process.wait(timeout=5)


if __name__ == "__main__":
    main(*sys.argv[1:])
