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

from pathlib import Path

import pytest

CI_SYSTEM_DIR = Path(__file__).parent


@pytest.mark.parametrize("installer_name", ["install_deps.sh", "install_deps_rocm.sh"])
def test_shell_scheduler_install_rebuilds_checkout(installer_name):
    installer = (CI_SYSTEM_DIR / installer_name).read_text()
    scheduler_step = installer.split(
        'echo "=== Step 5: Install TokenSpeed Scheduler ==="', 1
    )[1].split('echo "=== Step 6: Install TokenSpeed ==="', 1)[0]

    assert 'SCHEDULER_BUILD_DIR="$(mktemp -d)"' in scheduler_step
    assert "pip3 install --force-reinstall --no-deps" in scheduler_step
    assert '--config-settings="build-dir=${SCHEDULER_BUILD_DIR}"' in scheduler_step


def test_cu129_scheduler_install_rebuilds_checkout():
    installer = (CI_SYSTEM_DIR / "install_deps_cu129.py").read_text()
    scheduler_step = installer.split(
        'TemporaryDirectory(prefix="tokenspeed-cu129-scheduler-")', 1
    )[1]

    assert '"--force-reinstall"' in scheduler_step
    assert '"--no-deps"' in scheduler_step
    assert 'f"--config-settings=build-dir={build_dir}"' in scheduler_step
