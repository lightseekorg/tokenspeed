"""Real NPU-to-NPU transfer through the C ABI, including sparse page fields."""

import ctypes as c
import json
import multiprocessing as mp
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def library():
    lib = c.CDLL(os.environ["TOKENSPEED_MOONCAKE_LIBRARY"])
    lib.ts_mooncake_abi_version.argtypes = []
    lib.ts_mooncake_abi_version.restype = c.c_int
    lib.ts_mooncake_create.argtypes = [c.c_char_p] * 4
    lib.ts_mooncake_create.restype = c.c_void_p
    for name in ("destroy", "rpc_port"):
        getattr(lib, "ts_mooncake_" + name).argtypes = [c.c_void_p]
    lib.ts_mooncake_register.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t]
    lib.ts_mooncake_unregister.argtypes = [c.c_void_p, c.c_void_p]
    lib.ts_mooncake_write.argtypes = (
        [c.c_void_p, c.c_char_p] + [c.POINTER(c.c_uint64)] * 3 + [c.c_size_t]
    )
    lib.ts_mooncake_write_pages.argtypes = [
        c.c_void_p,
        c.c_char_p,
        c.POINTER(c.c_int64),
        c.POINTER(c.c_int64),
        c.c_size_t,
        c.POINTER(c.c_int64),
        c.c_size_t,
        c.c_size_t,
    ]
    lib.ts_mooncake_read.argtypes = lib.ts_mooncake_write.argtypes
    lib.ts_mooncake_read_pages.argtypes = lib.ts_mooncake_write_pages.argtypes
    lib.ts_mooncake_destroy.restype = None
    for name in (
        "rpc_port",
        "register",
        "unregister",
        "write",
        "write_pages",
        "read",
        "read_pages",
    ):
        getattr(lib, "ts_mooncake_" + name).restype = c.c_int
    assert lib.ts_mooncake_abi_version() == 1
    return lib


def worker(rank, conn):
    import torch

    torch.npu.set_device(rank)
    data = ((torch.arange(1024 * 1024, dtype=torch.int64) * 7 + 11) % 251).to(
        torch.uint8
    )
    buffer = (data if rank == 0 else torch.full_like(data, 173)).to(f"npu:{rank}")
    torch.npu.synchronize()
    lib = library()
    host = subprocess.check_output(["hostname", "-I"], text=True).split()[0]
    engine = lib.ts_mooncake_create(
        host.encode(), b"P2PHANDSHAKE", b"ascend_direct", b""
    )
    assert engine
    address = buffer.data_ptr()
    with ThreadPoolExecutor(1) as executor:
        assert (
            executor.submit(
                lib.ts_mooncake_register, engine, address, buffer.numel()
            ).result(timeout=30)
            == 0
        )
    conn.send(
        dict(session=f"{host}:{lib.ts_mooncake_rpc_port(engine)}", address=address)
    )
    peer = receive(conn)
    if rank == 0:
        target = peer["session"].encode()
        destination = peer["address"]
        # A Rust transfer worker likewise enters the C ABI on a fresh thread.
        with ThreadPoolExecutor(1) as executor:

            def u64(values):
                return (c.c_uint64 * len(values))(*values)

            rc = executor.submit(
                lib.ts_mooncake_write,
                engine,
                target,
                u64([address]),
                u64([destination]),
                u64([4096]),
                1,
            ).result(timeout=90)
            assert rc == 0, rc

            def i64(values):
                return (c.c_int64 * len(values))(*values)

            fields = i64(
                [
                    address + 8192,
                    512,
                    destination + 16384,
                    512,
                    256,
                    address + 32768,
                    1024,
                    destination + 65536,
                    1024,
                    384,
                ]
            )
            rc = executor.submit(
                lib.ts_mooncake_write_pages,
                engine,
                target,
                i64([1, 3, 7]),
                i64([8, 2, 5]),
                3,
                fields,
                2,
                2,
            ).result(timeout=90)
            assert rc == 0, rc
            # Default batch size and a partial last batch exercise both paths.
            for batch_size, offset in [(0, 131072), (4, 196608)]:
                extra = i64(
                    [
                        address + 8192,
                        512,
                        destination + offset,
                        512,
                        256,
                        address + 32768,
                        1024,
                        destination + offset + 16384,
                        1024,
                        384,
                    ]
                )
                assert (
                    executor.submit(
                        lib.ts_mooncake_write_pages,
                        engine,
                        target,
                        i64([1, 3, 7]),
                        i64([8, 2, 5]),
                        3,
                        extra,
                        2,
                        batch_size,
                    ).result(timeout=90)
                    == 0
                )
            # Put an invalid entry after a valid one. If validation accidentally
            # submits early, the receiver's untouched guard region will change.
            guard = destination + 262144
            assert (
                lib.ts_mooncake_write(
                    engine,
                    target,
                    u64([address, 2**64 - 1]),
                    u64([guard, guard + 64]),
                    u64([16, 16]),
                    2,
                )
                < 0
            )
            for bad in (-1, 2**63 - 1):
                assert (
                    lib.ts_mooncake_write_pages(
                        engine,
                        target,
                        i64([0, bad]),
                        i64([0, 1]),
                        2,
                        i64([address, 512, guard, 512, 16]),
                        1,
                        1,
                    )
                    < 0
                )
            invalid_fields = i64(
                [address, 512, guard, 512, 16, address, 512, guard + 4096, -1, 16]
            )
            assert (
                lib.ts_mooncake_write_pages(
                    engine, target, i64([0]), i64([0]), 1, invalid_fields, 2, 1
                )
                < 0
            )
            assert lib.ts_mooncake_write(engine, target, None, None, None, 0) == 0
            assert (
                lib.ts_mooncake_write(engine, target, u64([0]), u64([0]), u64([0]), 1)
                == 0
            )
            assert (
                lib.ts_mooncake_write(engine, target, None, u64([guard]), u64([16]), 1)
                < 0
            )
            assert lib.ts_mooncake_register(engine, 2**64 - 8, 16) < 0
            assert lib.ts_mooncake_unregister(None, address) < 0
            assert (
                lib.ts_mooncake_write_pages(engine, None, None, None, 0, None, 1, 0)
                == 0
            )
        conn.send("written")
        assert receive(conn) == "verified"
    else:
        assert receive(conn) == "written"
        # Pull from the still-live sender using the public scatter SDK path.
        with ThreadPoolExecutor(1) as executor:

            def u64(values):
                return (c.c_uint64 * len(values))(*values)

            def i64(values):
                return (c.c_int64 * len(values))(*values)

            assert (
                executor.submit(
                    lib.ts_mooncake_read,
                    engine,
                    peer["session"].encode(),
                    u64([peer["address"] + 4096]),
                    u64([address + 524288]),
                    u64([4096]),
                    1,
                ).result(timeout=90)
                == 0
            )
            assert (
                executor.submit(
                    lib.ts_mooncake_read_pages,
                    engine,
                    peer["session"].encode(),
                    i64([1, 3, 7]),
                    i64([8, 2, 5]),
                    3,
                    i64([peer["address"] + 8192, 512, address + 589824, 512, 256]),
                    1,
                    2,
                ).result(timeout=90)
                == 0
            )
        expected = torch.full_like(data, 173)
        expected[:4096] = data[:4096]
        expected[524288:528384] = data[4096:8192]
        for src, dst in zip([1, 3, 7], [8, 2, 5]):
            expected[589824 + dst * 512 : 589824 + dst * 512 + 256] = data[
                8192 + src * 512 : 8192 + src * 512 + 256
            ]
        for src_base, dst_base, stride, length in [
            (8192, 16384, 512, 256),
            (32768, 65536, 1024, 384),
        ]:
            for src, dst in zip([1, 3, 7], [8, 2, 5]):
                expected[dst_base + dst * stride : dst_base + dst * stride + length] = (
                    data[src_base + src * stride : src_base + src * stride + length]
                )
        for offset in (131072, 196608):
            for src_base, dst_base, stride, length in [
                (8192, offset, 512, 256),
                (32768, offset + 16384, 1024, 384),
            ]:
                for src, dst in zip([1, 3, 7], [8, 2, 5]):
                    expected[
                        dst_base + dst * stride : dst_base + dst * stride + length
                    ] = data[src_base + src * stride : src_base + src * stride + length]
        actual = buffer.cpu()
        assert torch.equal(actual, expected), int((actual != expected).sum())
        conn.send("verified")
        assert receive(conn) == "sender-exited"
    with ThreadPoolExecutor(1) as executor:
        assert (
            executor.submit(lib.ts_mooncake_unregister, engine, address).result(
                timeout=30
            )
            == 0
        )
        print(f"rank={rank} unregister passed", flush=True)
        executor.submit(lib.ts_mooncake_destroy, engine).result(timeout=30)
    print(f"rank={rank} destroy returned", flush=True)


def receive(conn):
    if not conn.poll(120):
        raise TimeoutError("C ABI peer did not respond in 120 seconds")
    return conn.recv()


def main():
    ctx = mp.get_context("spawn")
    pairs = [ctx.Pipe() for _ in range(2)]
    workers = [
        ctx.Process(target=worker, args=(i, pair[1])) for i, pair in enumerate(pairs)
    ]
    try:
        for worker_process in workers:
            worker_process.start()
        for pair in pairs:
            pair[1].close()
        endpoints = [receive(pair[0]) for pair in pairs]
        for index, pair in enumerate(pairs):
            pair[0].send(endpoints[1 - index])
        assert receive(pairs[0][0]) == "written"
        pairs[1][0].send("written")
        assert receive(pairs[1][0]) == "verified"
        pairs[0][0].send("verified")
        # Sender must exit before the receiver tears down its transport.
        for index, worker_process in enumerate(workers):
            worker_process.join(40)
            assert worker_process.exitcode == 0, worker_process.exitcode
            if index == 0:
                pairs[1][0].send("sender-exited")
        result = dict(
            passed=True,
            devices=2,
            contiguous=True,
            read_and_write=True,
            sparse_fields=2,
            sparse_pages=3,
            max_batch_sizes=[0, 2, 4],
            untouched_guard_bytes=True,
            worker_thread=True,
            process_exit_codes=[p.exitcode for p in workers],
            library=os.environ["TOKENSPEED_MOONCAKE_LIBRARY"],
            mask=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        )
        Path("cabi-test-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(10)
        for pair in pairs:
            pair[0].close()


if __name__ == "__main__":
    main()
