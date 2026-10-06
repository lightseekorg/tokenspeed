The pinned FlashInfer, FA4 and Quack wheels still use APIs deprecated by
CuTe DSL 4.8. These patches migrate their imports and calls without disabling
warnings. `patch_cute_dsl_dependencies.py` checks package versions and every
affected file's original or patched SHA256, then applies the patches with no
fuzz. A repeated install is safe; changed wheels require a reviewed patch update.

The CI installers and NVIDIA source image apply the patches after installing
dependencies. The BSD-3-Clause static scheduler is copied from the repository
into each package to avoid an import cycle. V2 FastDivmod serialization carries
both the encoded divisor and its scalar value; corresponding scheduler offsets
are updated. Kernel algorithms and numerical tolerances are preserved.

Patched files differ from the wheel's RECORD hashes. Stock PyPI installations
and release images retain upstream dependency warnings until patched package
releases are available. Remove these patches when the pinned wheels include
the migrations.
