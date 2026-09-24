# MI450 rocJITsu simulator setup

The MI450 simulator task pins a ROCm SDK version and a `rocm-systems` source
commit in `test/ci/ut/ut-tokenspeed-kernel-mi450-sim.yaml`. The setup script
builds the rocJITsu launcher and runtime from that source. It stores the source
commit, SDK version, and SDK root path in `rocjitsu-build/.tokenspeed-build-id`.

The setup script reuses the build only when the stamp matches and both build
outputs exist. A missing or different stamp causes a clean rebuild so a runner
cannot execute binaries left by an earlier nightly. The stamp is written only
after the build succeeds.

The `amd-mi45x-cpu-test` GitHub job has a 60-minute limit to cover a cold ROCm
installation and clean rocJITsu build. The simulator suite still has its own
600-second limit, with a 300-second limit for each test.

Run `python3 -m pytest test/ci_system/test_setup_mi450_sim.py` to check the
cache behavior without installing ROCm. Use the MI450 K8s Dispatch task for
the full simulator test suite.
