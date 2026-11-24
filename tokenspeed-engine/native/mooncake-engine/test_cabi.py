"""Host-only C ABI validation; fake backend headers never enter a native build."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class TokenSpeedCAbiTest(unittest.TestCase):
    def test_validation_and_failure_cleanup(self):
        compiler = shutil.which(os.environ.get("CXX", "g++"))
        if not compiler:
            self.skipTest("a C++20 compiler is required")
        with tempfile.TemporaryDirectory() as temporary:
            probe = Path(temporary) / "probe"
            subprocess.run(
                [
                    compiler,
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I" + str(ROOT / "tests"),
                    "-I" + str(ROOT),
                    str(ROOT / "tests/probe.cpp"),
                    "-o",
                    str(probe),
                ],
                check=True,
                timeout=60,
            )
            subprocess.run([probe], check=True, timeout=15)

    def test_public_header_is_c(self):
        compiler = shutil.which(os.environ.get("CC", "cc"))
        if not compiler:
            self.skipTest("a C compiler is required")
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fsyntax-only",
                "-I" + str(ROOT),
                "-x",
                "c",
                "-",
            ],
            input='#include "transfer_engine_tokenspeed.h"\n',
            text=True,
            check=True,
            timeout=15,
        )


if __name__ == "__main__":
    unittest.main()
