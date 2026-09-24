# MI450 rocJITsu simulator setup

The MI450 simulator task pins a ROCm SDK version and a `rocm-systems` source
commit in `test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml`. The setup script
builds the rocJITsu launcher and runtime from that source. It stores the source
commit, SDK version, and SDK root path in `rocjitsu-build/.tokenspeed-build-id`.
The pinned ROCm 10.1 nightly and matching PyTorch wheel are the last known
passing toolchain for the full simulator suite. The setup script requires its
SDK version, nightly index, and source commit as explicit inputs from the task.

The setup script reuses the build only when the stamp matches and both build
outputs exist. A missing or different stamp causes a clean rebuild so a runner
cannot execute binaries left by an earlier nightly. The stamp is written only
after the build succeeds.

The source checkout resets its local config patch before changing revisions,
then reapplies the unlimited `max_ticks` setting to the selected revision.

The `amd-mi45x-cpu-test` GitHub job has a 60-minute limit to cover a cold ROCm
installation and clean rocJITsu build. The simulator suite still has its own
600-second limit, with a 300-second limit for each test.

The task selects the `kernel` scope of `install_deps_rocm.sh`. It installs
PyTorch and the in-tree AMD/kernel packages required by the selected tests,
then stops before installing the scheduler and full TokenSpeed runtime. Other
AMD tasks explicitly select `full` to retain those packages. A missing or
unknown scope fails before installation begins.

Run `python3 -m pytest test/ci_system/test_setup_mi450_sim.py
test/ci_system/test_install_deps_rocm.py` to check the cache and install scopes
without installing ROCm. Use the MI450 K8s Dispatch task for the full
simulator test suite.
