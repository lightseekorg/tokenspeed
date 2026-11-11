"""Bundle CPython for the standalone host before normal auditwheel repair."""

import json
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var(
        "LDLIBRARY"
    )
    soname = run("patchelf", "--print-soname", str(library))
    # auditwheel removes libpython dependencies for extension modules, even
    # from executables. Give the host's private interpreter its own SONAME.
    private_name = soname.replace("libpython", "libtokenspeed_python", 1)
    if private_name == soname:
        raise RuntimeError(f"Unexpected CPython library: {soname}")

    with tempfile.TemporaryDirectory() as temporary:
        run(sys.executable, "-m", "wheel", "unpack", sys.argv[1], "-d", temporary)
        (root,) = Path(temporary).iterdir()
        package = root / "tokenspeed_engine"
        host = package / "bin" / "ts-host"
        if soname not in run("patchelf", "--print-needed", str(host)).splitlines():
            raise RuntimeError(f"ts-host does not link the build interpreter: {soname}")
        libraries = package / ".libs"
        libraries.mkdir()
        bundled = libraries / private_name
        shutil.copy2(library, bundled)
        run("patchelf", "--set-soname", private_name, str(bundled))
        run("patchelf", "--set-rpath", "$ORIGIN", str(bundled))
        run("patchelf", "--replace-needed", soname, private_name, str(host))
        run("patchelf", "--set-rpath", "$ORIGIN/../.libs", str(host))

        (metadata,) = root.glob("*.dist-info")
        # uv's stripped installation omits the licenses of libraries compiled
        # into libpython. Recover them from the same release's full archive.
        (download,) = json.loads(
            run(
                "uv",
                "python",
                "list",
                f"cpython@{platform.python_version()}",
                "--only-downloads",
                "--output-format",
                "json",
            )
        )
        archive_url = (
            download["url"]
            .replace(
                "https://releases.astral.sh/github/", "https://github.com/astral-sh/"
            )
            .replace("-install_only_stripped.tar.gz", "-pgo%2Blto-full.tar.zst")
        )
        archive = Path(temporary) / "python-full.tar.zst"
        run("curl", "-fsSL", "--max-time", "180", "-o", str(archive), archive_url)
        python_licenses = metadata / "licenses" / "CPython"
        python_licenses.mkdir(parents=True)
        run(
            "tar",
            "--use-compress-program=unzstd",
            "-xf",
            str(archive),
            "-C",
            str(python_licenses),
            "--strip-components=2",
            "--wildcards",
            "python/licenses/*",
        )
        licenses = {
            "OpenSSL": Path("/usr/share/licenses/openssl-libs/LICENSE"),
            "TokenSpeed": Path("LICENSE"),
        }
        for name, source in licenses.items():
            license_dir = metadata / "licenses" / name
            license_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, license_dir / "LICENSE.txt")
        Path("/tmp/engine-vendored").mkdir(exist_ok=True)
        run(
            sys.executable,
            "-m",
            "wheel",
            "pack",
            str(root),
            "-d",
            "/tmp/engine-vendored",
        )


if __name__ == "__main__":
    main()
