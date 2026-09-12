# SM120 bias argmax host launch contract

The partial and final Triton GPU kernels remain unchanged. This experiment uses
one CPU-built extension call to enqueue the same two kernels on the current
stream after the original live Tensor, device, layout, stride, alias, dtype,
offset and registry-selection checks. The cache retains executable handles and
immutable launch metadata; it retains no input pointers, Tensors or streams.
The two official runners remain alive in that same cache to own the executables.

The native pair is bounded to the inspected tokenspeed-triton distribution
3.8.10.post20260906 (its runtime version string is 3.8.10), one CTA per cluster,
four warps, the verified pointer-only PTX signatures, and zero global/profile
scratch. Both implicit scratch parameters are passed as null, as by the vendor
launcher. Cooperative launch, PDL, GSan, debug/pre-run instrumentation, and live
launch hooks preserve the official launcher path. Hook presence is checked at
every warm call; it is never cached. CUDA errors propagate immediately.

Eager and graph capture execute the identical dispatch. CUDA Graph records both
driver launches; replay uses the already captured nodes. No timing-mode branch,
new numerical kernel, changed output/work gate or new scheduler path is added.
The CUDA package build compiles the host extension with the selected CUDA headers and current PyTorch. Its build record binds the binary hash, exact PyTorch version and Python major/minor. CUDA wheels carry CPython/platform tags. A missing or incompatible optional extension keeps the original bias-plus-argmax fallback, including CPU/ROCm source installations. No compilation occurs on import.

C8 checks the pinned runtime HookChain call list live: an exactly typed empty chain is inactive; adding an enter/exit callback immediately restores official runner dispatch, and removal restores native dispatch. Arbitrary callables/subclasses keep the official path.

C9 releases the Python GIL only around the two CUDA driver calls, after all Python/Tensor metadata has been read. The pinned vendor driver.c uses Py_BEGIN_ALLOW_THREADS around its launch at lines 1524-1528. This restores its thread-interleaving contract; tensor argument ownership and RAII keep lifetime/error translation safe. C8 held the GIL and regressed 7.67-8.23 percent in full Engine eager while graph improved. The final C9 complete Engine eager run remained 3.168–3.825 percent slower despite positive local-region results. Releasing the GIL preserves the vendor concurrency contract; it is not evidence of a complete-model speedup.
