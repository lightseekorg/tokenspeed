---
name: bisect-triton-release
description: Bisecting triton releases to figure out which commit causes regressions.
---

TokenSpeed depends on `tokenspeed-triton`, a vendor release of Triton, for
faster access to latest features.
If a new release causes a regression, we need to switch to upstream Triton and
bisect which commits caused the issue.

## Overall Steps

Always activate the local venv from the `tokenspeed/` directory to make sure
installing `triton` into it.

1. Record the version of currently installed `triton` (or `triton-rocm` for AMD)
   pip package, as a dependency to `torch`, so we can restore it later.
2. Patch `_triton.py` to switch `tokenspeed_triton` imports to direct `triton`
   imports.
3. Ask which test in TokenSpeed shows regression.
4. Ask which upstream Triton commit is the last known good, and which upstream
   Triton commit shows regression.
5. Build and install triton at the two commits gotten in previous step to
   confirm that the regression happened due to upstream commit.
   If not, then ask where the downstream triton repo is to inspect downstream
   changes.
6. Once confirmed a upstream commit causes regression, start `git bisect` flow
   to identify the exact root cause.
   If `git bisect` points to an LLVM bump commit, then bisect the LLVM commit
   range to find the problematic commit. Ask where the LLVM source codebase
   is and where the build directory is.
7. Once identified, reinstall the recorded `triton` (or `triton-rocm` package)
   to the original version and revert local changes.

## Build and Install Triton

In Triton codebase, do the following

```shell
pip install -r python/requirements.txt
pip install .
```

To build with custom LLVM, set `LLVM_LIBRARY_DIR` and `LLVM_SYSPATH` to the
LLVM build directory.
