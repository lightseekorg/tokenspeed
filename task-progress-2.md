# Task 2 Progress — Add Gluon MoE Kernel on MI355

Branch: `kylewng/gluon_moe`
Container: `kylewng_triton_dev_mi355_1505_dev` (`/root/code/tokenspeed`)
Reference kernels:
- `gluon-kernels/kernels/cdna4/gemm/f16_gemm_gfx950.py`
- `triton-450/third_party/amd/python/examples/gluon/moe_gfx1250.py`

## Goal

把现有 gfx1250 (RDNA4) 的 Gluon MoE 示例 kernel 移植到 CDNA4 (gfx950 / MI355) 上，
作为 `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon.py` 注册到 MoE
`experts` 系列 kernel 中，并通过环境变量 `TOKENSPEED_MOE_GLUON=1`
接入 Task 1 已经跑通的 `gpt-oss-120b` (fp8 × mxfp4) pipeline。

## API / Layout 差异

`moe_gfx1250.py` 大量依赖 RDNA4 专有特性，移植到 CDNA4 需要替换以下原语。
下表列出这次修改用到的核心 API 对应关系：

| gfx1250 (RDNA4)                                  | CDNA4 / gfx950 (MI355)                                | 备注 |
| ------------------------------------------------ | ----------------------------------------------------- | ---- |
| `gl.amd.AMDWMMALayout(version=3, ...)`           | `gl.amd.cdna4.AMDMFMALayout(version=4, instr_shape=[16,16,32], ...)` | MFMA v4。`tiles_per_warp` 默认 `[1,1]` 即可。 |
| `gl.amd.gfx1250.wmma`/`wmma_scaled`              | `gl.amd.cdna4.mfma`/`mfma_scaled`                     | `mfma_scaled` 形参完全对应：`(a, a_scale, a_format, b, b_scale, b_format, acc)`。 |
| `tdm.async_load` / `async_gather` / `async_scatter` (TDM) | `gl.amd.cdna4.async_copy.buffer_load_to_shared` + 显式 `commit_group`/`wait_group`；scatter/gather 用 `buffer_store(out, base, offsets, mask=…)` 手动实现 | CDNA4 没有 tensor descriptor DMA。RDNA4 把 idx → smem → store 整条链整合成 TDM scatter，CDNA4 需要 row indices 自己算 + buffer store。 |
| `PaddedSharedLayout(...)`                        | `SwizzledSharedLayout(...)`                           | CDNA4 上 padded layout 对 mxfp4 scale tile 行为不稳，先用 swizzled。 |
| `get_wmma_scale_layout(dot_layout, shape, scale_factor)` | `gl.amd.cdna4.get_mfma_scale_layout(dot_layout, shape, scale_factor)` | 名字对齐；MFMA scaled 的 scale tile 形状是 `[BM/32, BK/32]`，不需要 RDNA4 那套 `scale_preshuffle` 寄存器布局调整。 |
| WMMA scale preshuffle (`reg_bases=[[0,1],[1,0]]`, `tiles_per_warp=2`) | 不需要，scale 直接按 `get_mfma_scale_layout` 给的 distributed layout load 即可 | RDNA4 上为了把每 16 行的 4 个 scale 拼到同一线程才需要 preshuffle。 |
| LDS budget ≈ 256 KB (RDNA4 LDS 大)               | LDS budget = **160 KB** (CDNA4)                       | 同样 BLOCK_M=128, BLOCK_N=256, BLOCK_K=256 在 CDNA4 直接 OOR (`out of resource: shared memory, Required: 196608, Hardware limit: 163840`)。我们把默认配置降到 `(BM,BN,BK)=(64,128,128)` + `NUM_BUFFERS=2`，单 stage 占用 ≈ 96 KB；保留 `(64,128,256)` 等 piped 选项给后续 software pipeline。 |
| `MoESliceNKProgram` / `MoESliceKProgram` / `MoEPipelinedProgram` 三种调度 | 当前只移植了非软件流水的 baseline 版本                  | RDNA4 例子里 ping-pong / sliceK 都依赖 TDM async pipeline，CDNA4 上需要重写为 `async_copy` + `commit_group` + `wait_group` 形式，作为后续优化项。 |
| `gl.amd.gfx1250.buffer_store(out, base, offsets, mask=…)` | `gl.store(base + offs, out, mask=…)`                  | CDNA4 也支持 `gl.amd.cdna4.buffer_store`，但 baseline 里直接走 `gl.store`，简单可读。 |

`f16_gemm_gfx950.py` 给我们提供了 CDNA4 上 “非 MoE / 非 mxfp4” 的 GEMM 模板：
warps_per_cta、shared layout 选择、MFMA v4 形参、buffer_load_to_shared 的
batch_load 思路全部沿用。

## Steps

1. **`tokenspeed_kernel.ops.moe.gluon` 模块**
   - 新增 `_ragged_bf16_gemm_kernel`（`@gluon.jit`），实现:
     - 1 grid block 处理 1 个 `(M-block, N-block)` tile，
     - 每个 M-block 通过 `expert_id_ptr[pid_m]` 决定要 load 的 expert 权重切片，
     - 支持可选 `gather_indx` (dispatch GEMM) / `scatter_indx` (combine GEMM) /
       `bias`,
     - 内部用 MFMA v4 + Swizzled Shared layout，
     - 没有软件流水，先确保 numerical correct。
   - Python launcher `_gluon_bf16_ragged_matmul`：
     - 解析 `RaggedTensorMetadata.slice_sizes` 构造 per-block expert id,
     - 不支持的 case (mxfp4 weight / fp8 act / fused swiglu) **透明 fallback** 到
       `triton_kernels.matmul`，保持 API 完全兼容上游签名。
2. **Kernel registry**
   - 在 `tokenspeed_kernel/ops/moe/__init__.py` import `gluon`，
   - 在 `gluon.py` 注册 3 个 spec：
     - `triton_kernels_gluon_dispatch_gemm`（`features={ragged_metadata, dispatch_gemm}`）
     - `triton_kernels_gluon_gemm_combine`（`features={ragged_metadata, gemm_combine}`）
     - `triton_kernels_gluon_matmul_ogs`（`features={ragged_metadata}`）
   - **优先级 gating**:
     - `TOKENSPEED_MOE_GLUON=1` → priority `Priority.SPECIALIZED + 1 = 13`
       （高于上游 `triton_kernels_*` 的 10），自动被 selector 选中。
     - 未设置 / 不是 truthy → priority `Priority.PORTABLE + 1 = 5`
       （低于上游），保持 default 行为不变。
3. **UT (`test/ops/test_moe_gluon.py`)**
   - `test_gluon_bf16_ragged_matches_torch`: 三组 `(M,N,K,E,block_m)` 形状，
     与 PyTorch reference matmul 比对，rtol/atol = 5e-2 / 5e-2 (bf16 容差)。
   - `test_gluon_kernel_selected_under_env`: 监控 `TOKENSPEED_MOE_GLUON=1` 下
     selection oracle 真的把 gluon kernel 排到首位。
   - `test_gluon_falls_back_for_mxfp4`: 喂入 mxfp4 输入 → 走 fallback 路径
     （让 `_upstream_matmul` 复杂 path 报错出来即可；目的是确认不会
     在 Gluon prologue 里把 mxfp4 当 bf16 跑）。
4. **Micro-bench (`benchmarks/moe_gluon_microbench.py`)**
   - 网格扫描 `block_m ∈ {32,64,128}`、`block_n ∈ {64,128,256}`、
     `block_k ∈ {64,128,256}`、`num_warps ∈ {4,8}`，与 `triton_kernels.matmul`
     baseline 对比 TFLOPs。
5. **gpt-oss-120b 接入**
   - 通过环境变量 `TOKENSPEED_MOE_GLUON=1` 启用。
   - 因为目前 Gluon kernel 还不支持 fp8 × mxfp4，gpt-oss-120b 实际调用
     仍然走 `_upstream_matmul` fallback。换言之：Gluon path **不会拉低
     Task 1 的精度**，但也暂时无法贡献额外加速。下一步实现 mxfp4/fp8
     scaled MFMA 后这个 hook 自然生效。

## Progress

- [x] T0 阅读 `f16_gemm_gfx950.py`、`moe_gfx1250.py`，整理 API/layout 差异
- [x] T1 创建分支 `kylewng/gluon_moe`（基于最新 `origin/main` + Task 1 改动）
- [x] T2 新增 `tokenspeed_kernel/ops/moe/gluon.py`（baseline bf16 ragged GEMM
      + fallback + 3 个 register_kernel）
- [x] T3 新增 `tokenspeed-kernel/test/ops/test_moe_gluon.py` 5/5 PASS
- [x] T4 新增 `tokenspeed-kernel/benchmarks/moe_gluon_microbench.py`，
      跑出 36 个 config 的 TFLOPs 对照表
- [x] T5 通过 `TOKENSPEED_MOE_GLUON=1` 接入 selector，验证 selector 选中
      Gluon kernel；gpt-oss-120b 在 mxfp4 模式下走透明 fallback，无 regression
- [x] T6 撰写 `task-progress-2.md` 并 commit

## 单测结果

```
tokenspeed-kernel/test/ops/test_moe_gluon.py
  - test_gluon_bf16_ragged_matches_torch[128-128-128-2-64]   PASS
  - test_gluon_bf16_ragged_matches_torch[256-256-128-4-64]   PASS
  - test_gluon_bf16_ragged_matches_torch[512-512-256-8-64]   PASS
  - test_gluon_kernel_selected_under_env                     PASS
  - test_gluon_falls_back_for_mxfp4                          PASS
5 passed in 0.72s on MI355 (gfx950).
```

## Micro-bench (M=1024, N=K=2880, E=32, MI355)

`triton_kernels.matmul` baseline = 0.153 ms / 221.7 TFLOPs。

| Config (`bm`,`bn`,`bk`,`nw`)       | time (ms) | TFLOPs | 与 baseline |
| ----------------------------------- | --------- | ------ | ----------- |
| 32, 64, 64, 4                       | 0.331     | 102.6  | 0.46×       |
| 32, 128, 64, 4                      | 0.339     | 100.2  | 0.45×       |
| 32, 128, 64, 8                      | 0.342     | 99.4   | 0.45×       |
| 64, 128, 64, 4                      | 0.355     | 95.8   | 0.43×       |
| 64, 128, 64, 8                      | 0.377     | 90.1   | 0.41×       |
| 128, 128, 64, 4                     | 0.399     | 85.1   | 0.38×       |
| 128, 256, 256, 4 (LDS overflow)     | OOR       | —      | —           |

**结论**：未做 software pipeline 的 baseline Gluon kernel 当前在 MI355 上
约为上游 `triton_kernels.matmul` 的 ~46%。这与预期一致 —
上游 kernel 走的是 `MoEPipelinedProgram` 多 buffer + ping-pong 流水，
我们只跑非 piped 的“正确性参考版”，差距来自 K-dim async DMA。
后续把 `gl.amd.cdna4.async_copy.buffer_load_to_shared` + `commit_group`/
`wait_group` 接进来即可显著拉近，这是 follow-up 项。

也明确观察到：

- `BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 64` (`nw=4`) 是 baseline 的 sweet spot
  (smallest LDS、最少 read/load latency)。
- `BLOCK_K=128/256` 在 CDNA4 上 LDS 占用立即翻倍，吞吐反而下降。
- `BLOCK_N=256, BLOCK_K=256` 直接超出 LDS，触发 `OutOfResources` —
  这正是 RDNA4 与 CDNA4 (160KB LDS) 的关键差异。

## Selection 验证（`TOKENSPEED_MOE_GLUON=1`）

```
$ TOKENSPEED_MOE_GLUON=1 python3 -c "
from tokenspeed_kernel.registry import KernelRegistry
import tokenspeed_kernel.ops.moe
for s in sorted(KernelRegistry.get().list_kernels('moe', 'experts'),
                key=lambda x: -x.priority):
    print(s.name, s.priority, sorted(s.features))
"
triton_kernels_gluon_dispatch_gemm   13  ['dispatch_gemm', 'ragged_metadata']
triton_kernels_gluon_gemm_combine    13  ['gemm_combine', 'ragged_metadata']
triton_kernels_gluon_matmul_ogs      13  ['ragged_metadata']
triton_moe_fused_experts             10  ['dispatch_sorted']
triton_kernels_matmul_ogs            10  ['ragged_metadata']
triton_kernels_dispatch_gemm         10  ['dispatch_gemm', 'ragged_metadata']
triton_kernels_gemm_combine          10  ['gemm_combine', 'ragged_metadata']
```

## gpt-oss-120b 接入

Gluon kernel **能被 selector 正确选中**，且当前情况下 forward 透明 fallback
到 `triton_kernels.matmul`：

- 模型用 fp8 × mxfp4，`PrecisionConfig.b_mx_scale != None` 且
  `flex_ctx.lhs_data` 是 fp8，`_gluon_bf16_ragged_matmul` 的 fallback 守卫
  立即把请求转给 `_upstream_matmul`，行为与 Task 1 完全一致。
- 也就意味着精度 / 吞吐与 Task 1 相同：GPQA-Diamond medium reasoning
  **mean_acc 0.6616, decode 79.37 tok/s**（数据见 `task-progress-1.md`）。
- 当未来 Gluon kernel 实现 `mfma_scaled` 路径（mxfp4 weight + fp8/bf16 act）后，
  这里的 fallback 会自动停止触发，gpt-oss-120b 即可享受 Gluon kernel 的优化。

启动方式（与 Task 1 完全一致，唯一差异是加 `TOKENSPEED_MOE_GLUON=1`）：

```
docker exec kylewng_triton_dev_mi355_1505_dev bash -c '
  cd /root/code/tokenspeed && \
  HIP_VISIBLE_DEVICES=2,3 \
  TOKENSPEED_MOE_GLUON=1 \
  LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib \
  python3 -m tokenspeed.api_server \
    --model amd/gpt-oss-120b-w-mxfp4-a-fp8 \
    --attn-tp-size 2 --moe-tp-size 2 --max-model-len 80000 \
    --trust-remote-code --reasoning-parser gpt-oss \
    --kvstore-ratio 0.25 --enable-cache-report \
    --host 127.0.0.1 --port 8000'
```

## 关键文件改动

- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon.py`
  （**新增**）— MI355 Gluon ragged MoE GEMM kernel + 3 个 register_kernel
  入口 + env-knob 优先级控制。
- `tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/__init__.py`
  导入新模块（必须放在 `triton_kernels` 之后，保证 fallback target 已就位）。
- `tokenspeed-kernel/test/ops/test_moe_gluon.py`（**新增**）—
  3×correctness + 1×selection + 1×fallback。
- `tokenspeed-kernel/benchmarks/moe_gluon_microbench.py`（**新增**）—
  block size / num_warps 网格扫描脚本。
- `task-progress-2.md`（**新增**）— 本文件。

## Update — software pipeline + 三 kernel 全部超过 baseline

按 `TASKS.md:31-38` 进一步完善后的状态：

### 1. Software pipelining

参考 `triton-450/.../moe_gfx1250.py` 的 `MoEPipelinedProgram`，
在 `_pipelined_moe_kernel` 中实现 **register-staged double-buffer pipeline**：

- 在主循环开始前 prefetch tile 0 (`a_next`, `b_next`)。
- 每次 MFMA 之前，对下一个 tile 发出 `gl.load` prefetch
  (`a_prefetch`, `b_prefetch`)，与当前 MFMA 计算重叠。
- MFMA 完成后将 prefetched tile 提升到 `a_next`/`b_next`。

> 我们最初按 `f16_gemm_gfx950.py` 走 `buffer_load_to_shared` +
> `commit_group` / `wait_group` 的 LDS 双缓冲 path，
> 但当 base pointer 同时含 dynamic per-expert offset 时，
> Triton/MLIR legalization 反复抛 `unrealized_conversion_cast`。
> 在 register-staged 方案已经能稳定打过 baseline 后，
> 把 LDS async DMA 留作 follow-up（见下文）。

### 2. Static profile / no spill 校验

把 `gfx1250_utils.py` 的 `static_profile` 移植成
`tokenspeed_kernel.ops.moe.gluon.static_profile` + `assert_no_spills`：

- 在 unit test (`test_no_register_spill`) 中编译 kernel 后强制断言
  `sgpr_spill == 0` & `vgpr_spill == 0` & `scratch == 0`。
- 在 microbench 末尾把每个编译产物的 GPR/spill/occupancy dump 出来。

最新 microbench 数据（MI355, container `kylewng_triton_dev_mi355_1505_dev`）：

```
Kernel 1: bf16 gating GEMM (Gluon vs upstream triton_kernels.matmul)
  M=512,N=1024,K=2880    gluon=  44.9 TFLOPs (0.067 ms)  triton_kernels= 18.8 TFLOPs  →  2.39x
  M=1024,N=1024,K=2880   gluon=  78.6 TFLOPs (0.077 ms)  triton_kernels= 38.3 TFLOPs  →  2.05x
  M=4096,N=1024,K=2880   gluon= 230.5 TFLOPs (0.105 ms)  triton_kernels=220.3 TFLOPs  →  1.05x

Kernel 2: dispatch + 1st GEMM + SwiGLU
  M=512 ,N=1024*2,K=2880,E=4   gluon=  56.6 TFLOPs (0.107 ms)  baseline= 48.6 TFLOPs  →  1.16x
  M=1024,N=1024*2,K=2880,E=4   gluon= 111.3 TFLOPs (0.109 ms)  baseline= 98.2 TFLOPs  →  1.13x

Kernel 3: 2nd GEMM + scatter combine
  M=512 ,N=2880,K=1024,E=4     gluon=  35.1 TFLOPs (0.086 ms)  baseline= 24.6 TFLOPs  →  1.42x
  M=1024,N=2880,K=1024,E=4     gluon=  69.2 TFLOPs (0.087 ms)  baseline= 49.0 TFLOPs  →  1.41x

Static GPR / spill profile (all 9 compiled variants):
  sgpr ≤ 49, vgpr ≤ 216, sgpr_spill = 0, vgpr_spill = 0, scratch = 0,
  occupancy 2 ~ 5
```

3 个 kernel **全部超过 baseline**，并且无任何 sgpr / vgpr spill。

### 3. 关键改动概览

- **统一 `_pipelined_moe_kernel`**：用 `constexpr` flag
  (`HAS_BIAS`, `HAS_GATHER`, `HAS_SCATTER`, `DO_SWIGLU`, `APPLY_GATE_SCAL`)
  一次性覆盖 3 种 kernel，避免代码重复。
- **2D launch grid**：把 `expert_id` 提到 `program_id(1)`，
  保证它在 IR 里是 scalar，便于以后再换回 `buffer_load_to_shared`
  做完整 LDS 双缓冲。
- **shape-aware autotuner** `_autotune_block`：
  - 大 dense GEMM (`do_swiglu=False`): 小 BLOCK_N=64 + 8 warps，
    把 grid_n 拉满 256 个 CU。
  - SwiGLU 路径 (`do_swiglu=True`): 64×64×32 + 4 warps，
    省下 SwiGLU 那一步要的临时 vgpr。
- **3 个独立 launcher**：`gluon_bf16_gating_gemm`,
  `gluon_bf16_dispatch_swiglu`, `gluon_bf16_combine`，
  各自直接吃 PyTorch tensor，方便单测和 microbench 用。
- **更新 UT** (`test/ops/test_moe_gluon.py`)：
  3 套 kernel 各 2 个 shape 的 correctness，
  + `test_no_register_spill` 在编译产物上断言 spill 为零。
- **更新 microbench** (`benchmarks/moe_gluon_microbench.py`)：
  对 3 个 kernel 分别跑相应 baseline + dump static profile。

## Update 2 — gpt-oss-120b 真实 shape 调优 + active-expert remap

按 `TASKS.md:39-41` 进一步完善：调优 (`block_size`, `num_warps`) 时
必须考虑 gpt-oss-120b 的真实 MoE GEMM 维度 (`H=I=2880, E=128,
topk=4`，参数取自 [HF model card](https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8))，
并同时覆盖 prefill (B=1024/4096/8192) 和 decode (B=1/32/64) 场景。

### 1. Active-expert remap (decode 必备)

之前的 launcher 不区分活跃/非活跃 expert：哪怕 ragged metadata 是
`counts = [64, 64, 64, 64, 0, 0, ..., 0]` (decode 仅 4 个 expert 活跃)，
也会启动全部 128 个 expert 的 CTA，对 124 个空 expert 做完全被 mask
的无用 MFMA。结果：B=1 dispatch+SwiGLU 跑 **1.477ms**，对应
**0.09× baseline**，等于 kernel 不可用。

修复：

* Kernel 增加 `expert_remap_ptr` 参数 + `HAS_EXPERT_REMAP: gl.constexpr`。
  当 metadata 稀疏时，host 端只对 *活跃* expert 构造一个 i32 索引表，
  kernel 用 `expert_id = gl.load(expert_remap_ptr + compact_idx)`
  做 scalar 间接寻址（参考 `moe_gfx1250.py` 的
  `gl.load(XSliceSizes + expt_id)` 写法）。
* `compact_idx = gl.program_id(1)` 用于活动 buffer 的 M-offset，
  `expert_id` 仅用来索引 W / bias。
* 全活跃 (prefill) 时 `expert_remap=None`，kernel 退回原来直接
  `expert_id = compact_idx` 的零开销路径。

### 2. shape-aware autotuner

`_autotune_block(M, N, K, *, do_swiglu=False, ragged=False)` 依据
microbench sweep (`benchmarks/moe_gluon_microbench.py`) 给出：

| 路径 | M 区间 | (BM, BN, BK, NW) |
|------|--------|------------------|
| 稠密 gating GEMM (`do_swiglu=False, ragged=False`) | M ≤ 1024 | 64×64×64, 8 warps |
| 稠密 gating GEMM | 1024 < M ≤ 2048 | 128×64×64, 8 warps |
| 稠密 gating GEMM | M > 2048 | 128×64×64, 4 warps |
| Fused SwiGLU 1st GEMM (`do_swiglu=True`) | M ≤ 8192 | 64×128×32, 4 warps |
| Fused SwiGLU 1st GEMM | M > 8192 | 128×128×32, 4 warps |
| Ragged combine (`ragged=True`) | M ≤ 8192 | 64×128×32, 4 warps |
| Ragged combine | M > 8192 | 128×128×32, 4 warps |

为什么 256×256 不出现：sweep 显示 `(256, 256, 32, 8)` 虽然能拿到
295 TFLOPs，但 vgpr 用满 256 后**溢出** 210 → static profile 报
spill，违背 Update 1 的“no sgpr/vgpr spill”硬约束。退到
`(128, 128, 32, 4)` 仍能拿到 272 TFLOPs 且 vgpr ≤ 200。

### 3. gpt-oss-120b microbench (MI355, kylewng_triton_dev_mi355_1505_dev)

```
==============================================================================================================
gpt-oss-120b moe shapes: H=2880, I=2880, E=128, topk=4
==============================================================================================================

Kernel 1: bf16 gating GEMM  (B, H) x (H, E)
  B=1     gluon=  0.07ms  baseline=  0.16ms  torch.mm= 0.01ms  speedup=2.49x
  B=32    gluon=  0.07ms  baseline=  0.16ms  torch.mm= 0.02ms  speedup=2.41x
  B=64    gluon=  0.07ms  baseline=  0.16ms  torch.mm= 0.02ms  speedup=2.35x
  B=1024  gluon=  0.07ms  baseline=  0.16ms  torch.mm= 0.01ms  speedup=2.27x
  B=4096  gluon=  0.10ms  baseline=  0.16ms  torch.mm= 0.02ms  speedup=1.62x
  B=8192  gluon=  0.10ms  baseline=  0.16ms  torch.mm= 0.03ms  speedup=1.62x

Kernel 2: dispatch + 1st GEMM + SwiGLU  (M_d, H) x (E, H, 2I)
  B=1     M_d=  256  gluon=0.165ms  baseline=0.129ms  speedup=0.78x   (decode, sparse 4/128)
  B=32    M_d= 8192  gluon=1.595ms  baseline=0.926ms  speedup=0.58x
  B=64    M_d= 8192  gluon=1.596ms  baseline=0.924ms  speedup=0.58x
  B=1024  M_d= 8192  gluon=1.595ms  baseline=0.939ms  speedup=0.59x
  B=4096  M_d=16384  gluon=1.787ms  baseline=1.151ms  speedup=0.64x
  B=8192  M_d=32768  gluon=2.997ms  baseline=1.761ms  speedup=0.59x

Kernel 3: 2nd GEMM + scatter combine  (M_d, I) x (E, I, H)
  B=1     M_d=  256  gluon=0.163ms  baseline=0.131ms  speedup=0.80x   (decode, sparse 4/128)
  B=32    M_d= 8192  gluon=0.899ms  baseline=0.484ms  speedup=0.54x
  B=64    M_d= 8192  gluon=0.899ms  baseline=0.481ms  speedup=0.53x
  B=1024  M_d= 8192  gluon=0.898ms  baseline=0.480ms  speedup=0.53x
  B=4096  M_d=16384  gluon=1.073ms  baseline=0.598ms  speedup=0.56x
  B=8192  M_d=32768  gluon=1.958ms  baseline=0.933ms  speedup=0.48x

Static GPR / spill profile (12 compiled variants):
  sgpr ∈ [34, 42], vgpr ∈ [86, 200]
  sgpr_spill = 0, vgpr_spill = 0, scratch = 0   ✅ all clean
```

要点：

* **Kernel 1（gating GEMM）** 在所有 batch 上 **稳定 1.6×–2.5× 超过
  upstream `triton_kernels.matmul`**，因为 `N=128`（experts 数）远小于
  upstream 优化目标的常用 N，我们的 register-staged 流水开销更小。
* **Kernel 2 / 3（MoE GEMM）** 在 prefill 上仍落后 upstream
  (~0.5–0.6×)。upstream 走的是多缓冲 LDS async pipeline，我们的
  register-staged 单缓冲流水无法完全 hide 64+ 个 K-tile 的访存延迟。
  把这一块顶上去需要把
  `gl.amd.cdna4.async_copy.buffer_load_to_shared` +
  `commit_group/wait_group` 真正接通（详见下方 follow-up #2）。
* **Decode (B=1)** 已经从 `0.09×` 跳到 `0.78×`，证明 active-expert
  remap 完全到位；剩余 0.78× 主要是 per-expert padding 到 `block_m=64`
  的固定开销。

### 4. UT 覆盖

`test_moe_gluon.py` 现在共 15 个用例：

* 7 个 correctness (gating + dispatch+swiglu + combine + 全 spill check)；
* 3 个 `test_gpt_oss_decode_remap[B=1, 32, 64]` — 验证 active-expert
  remap 数值正确；
* 2 个 `test_gpt_oss_no_spill[B=1, 64]` — 在 H=2880, I=2880, E=128
  的 gpt-oss 真实尺寸下编译并断言无 spill；
* 2 个 selector / mxfp4 fallback 回归检查。

```
$ python3 -m pytest test/ops/test_moe_gluon.py -v
... 15 passed in 1.01s
```

## Update 3 — scaled MFMA (16x16x128) BLOCK_K 约束

按 `TASKS.md:43-45` 加上 CDNA4 scaled MFMA 指令形状对 `BLOCK_K` 的硬约束。

### 1. 约束来源

- 常规 `gl.amd.cdna4.mfma` (CDNA4 MFMA v4)：`instr_shape=[16, 16, 32]`
  → `BLOCK_K` 任意 32 的倍数都合法。
- **Scaled** `gl.amd.cdna4.mfma_scaled` (mxfp4 weight + fp8/bf16 act
  路径)：`instr_shape=[16, 16, 128]`
  → `BLOCK_K` 必须 ≥ 128 且 `BLOCK_K % 128 == 0`。

参考：`triton-450/python/test/gluon/test_core.py::test_amd_mfma_scaled`
和 `triton-450/python/triton/experimental/gluon/language/amd/cdna4/__init__.py::mfma_scaled`。

### 2. 改动概要

* `_autotune_block(... , scaled_mfma: bool = False)`：新增第三维度。
  当 `scaled_mfma=True` 时把 `BLOCK_K` 上提到 ≥ 128 并向上对齐到 128
  的倍数（同时强制 `BLOCK_M % 16 == 0`，这是 MFMA M 维要求）。
* `_launch_pipelined(..., scaled_mfma=False)`：新增同名参数 + runtime
  assertion，保证未来 mxfp4 路径错配 BK 时立刻报 `AssertionError`
  而不是踩坏 launch。
* `_MFMA_K = 32` / `_MFMA_SCALED_K = 128` / `_MFMA_M = 16` 常量集中
  在一处方便后续维护。

### 3. 为什么 bf16 路径不直接全部切 BK=128

直接抬高 `BLOCK_K` 是有诱惑的（更长的内积 = 更少 K-tile 迭代），
但跑 microbench sweep 后发现 bf16 + 当前 register-staged 单缓冲流水
在 `BK=128` 下反而**慢 2–3×**：

| 路径 | shape | BK=32 (current) | BK=128 sweep best |
|------|-------|-----------------|------------------|
| K2 dispatch+SwiGLU | B=32   M_d=8192  | 170 TFLOPs / 1.60ms | 70 TFLOPs / 3.86ms |
| K2 dispatch+SwiGLU | B=8192 M_d=32768 | 362 TFLOPs / 3.00ms | 127 TFLOPs / 8.55ms |
| K3 combine         | B=32   M_d=8192  | 151 TFLOPs / 0.90ms | 66 TFLOPs / 2.06ms |
| K3 combine         | B=8192 M_d=32768 | 274 TFLOPs / 1.98ms | 144 TFLOPs / 3.79ms |

原因：register-staged 流水里每个 K-tile 都要把整个 BLOCK_M × BLOCK_K
的 A tile 和 BLOCK_K × BLOCK_N 的 B tile 预取到 vgpr，BK 翻 4 倍直接把
单 tile 的寄存器需求翻 4 倍（K2 prefill 那一档原本 vgpr=166→200，
BK=128 下挤压出 spill / occupancy 掉到 1）。

结论：常规 MFMA 走 BK=32/64 保持现状；scaled MFMA 路径**只在
`scaled_mfma=True` 时** 自动提升到 BK=128。

### 4. UT 覆盖

新增 6 个用例（共 21 passing）：

* `test_autotune_scaled_mfma_block_k[...]` × 4 个 shape，参数化
  覆盖 gating / dispatch+SwiGLU / combine 三种 path 在 scaled 模式下
  的 `BLOCK_K`。同时反向验证 `scaled_mfma=False` 时 `BK ∈ {32, 64}`。
* `test_launcher_rejects_bad_block_k_for_scaled_mfma`：launcher
  对错配 `(scaled_mfma=True, BK<128)` 报 `AssertionError`。
* `test_kernel_compiles_with_block_k_128`：bf16 kernel 在 `BK=128`
  下编译正确且数值匹配 reference（确保未来切 scaled 时 kernel body
  还能跑）。

```
$ python3 -m pytest test/ops/test_moe_gluon.py -v
... 21 passed in 0.97s
```

### 5. Microbench 仍然全绿

```
Scaled-MFMA (mxfp4/fp8, instr 16x16x128) autotune preview:
  gating  decode  B=1            BM= 64 BN= 64 BK=128 NW=8
  gating  prefill B=8192         BM=128 BN= 64 BK=128 NW=4
  dispatch+swiglu B=32           BM= 64 BN=128 BK=128 NW=4
  dispatch+swiglu B=8192         BM=128 BN=128 BK=128 NW=4
  combine        B=32            BM= 64 BN=128 BK=128 NW=4
  combine        B=8192          BM=128 BN=128 BK=128 NW=4
```

bf16 路径的吞吐和 spill profile **完全没变** —— 改动只是把 scaled
MFMA 的硬约束钉进 autotuner + launcher，给未来的 mxfp4 / fp8 落地
留好准入。

---

## Update 4 — mxfp4 / fp8 scaled-MFMA 路径 + unified kernel API

针对 `TASKS.md:46-61`，要求把 dispatch-fused / combine-fused 的 GEMM
从 bf16 升级到 mxfp4 (scaled MFMA)，并支持两种 A 端 dtype 组合：

* `A: fp8 (e4m3 / e5m2) + global scale,  W: mxfp4 (+ e8m0 block scale)`
* `A: mxfp4 (+ e8m0 block scale),       W: mxfp4 (+ e8m0 block scale)`

同时要求 “像 `triton_kernels` 一样用一个 kernel 支持多种变体”。

### 1. 新增 scaled-MFMA kernel

新增 `_pipelined_moe_kernel_scaled`（与原 bf16 kernel 同文件、同
`MoE constexpr` 旗标），关键差异：

| 项目 | bf16 path | scaled path |
|-----|-----------|------------|
| MFMA op | `gl.amd.cdna4.mfma`（`instr=[16,16,32]`, `k_width=8`） | `gl.amd.cdna4.mfma_scaled`（`instr=[16,16,128]`, `k_width=16`） |
| Operand dtype | bf16 (两个) | uint8 (A 是 e2m1/e4m3/e5m2，W 始终 e2m1) |
| K 维步进 | `BLOCK_K` (logical = physical) | `BLOCK_K // DIV_FACTOR_*`；e2m1 时 `DIV=2` (打包) |
| Block scale | 无 | `[BLOCK_M, BLOCK_K/32]` (A) / `[BLOCK_N, BLOCK_K/32]` (B)，layout 由 `get_mfma_scale_layout(...)` 给出 |
| Global scale | 无 | fp8 路径下走 host-side `a_global_scale: float`，在 K-loop 结束后 `acc *= a_global_scale` |
| BLOCK_K 约束 | 32 / 64 (寄存器压力) | 必须 ≥ 128 且为 128 的倍数（与 scaled MFMA K 深度一致），由 `_autotune_block(scaled_mfma=True)` + launcher assertion 共同保证 |

变体由 4 个 constexpr 旗标选择：

```python
A_FORMAT          : str   # "e2m1" / "e4m3" / "e5m2"
B_FORMAT          : str   # "e2m1"  (W 一律是 mxfp4)
HAS_A_BLOCK_SCALE : bool  # A==e2m1 时 True，否则 False
# 其他 MoE 旗标 (HAS_BIAS, HAS_GATHER, HAS_SCATTER, DO_SWIGLU,
# APPLY_GATE_SCAL, HAS_EXPERT_REMAP) 与 bf16 kernel 完全一致
```

block scale tile 直接用 `get_mfma_scale_layout` 返回的 distributed
layout `gl.load`，**不需要** 额外 `convert_layout`——这与 upstream
`test_amd_mfma_scaled`（`triton-450/python/test/gluon/test_core.py:1369-1397`）
完全一致；早期版本尝试自己构造 `BlockedLayout` 时会触发
`Layout has 128 threads per warp, but the module specifies 64`
的检查（CDNA4 wave=64 vs gfx1250 wave=32 的差异）。

### 2. Public launchers

```python
gluon_mxfp_gating_gemm(x, w, w_scale, *, x_scale=None,
                       a_format="e2m1", a_global_scale=1.0, ...)
gluon_mxfp_dispatch_swiglu(x, w, w_scale, *, x_scale=..., a_format=...,
                           a_ragged_metadata, gather_indx, ...)
gluon_mxfp_combine(x, w, w_scale, *, x_scale=..., a_format=...,
                   a_ragged_metadata, scatter_indx, ...)
```

* `x` 与 `w` 都用 `torch.uint8` 存储；`x_scale` / `w_scale` 也是
  `uint8` (e8m0)，由 `triton_kernels.tools.mxfp.MXScaleTensor.data`
  那一套生成。
* W 的 scale tensor layout 取 `[E, N, K/32]`（N-major、K-scale inner），
  这样 per-expert slice 是连续的，与 `triton_kernels` 的
  `wrap_torch_tensor(..., dtype=FP4)` 约定一致。
* fp8 A 不带 block scale 时 `a_global_scale: float` 直接当作 host-side
  `float32` 标量传进 kernel，K-loop 之后 `acc *= a_global_scale`
  (与 `triton_kernels._matmul.py:479-490` 处理 flex scale 的方式相同)。

### 3. OOB-write 修复

借落 mxfp4 path 之机修了一个 **bf16 kernel 一直潜伏的 store 越界**:

旧版 mask 用的是 `offs_y_n < OUT_BLOCK_N * grid_n`，对于
`N % BLOCK_N != 0` 的 shape (例如 `N=2880, BN=128 -> grid_n=23,
N_padded=2944`) 会让 store 落到 “下一行” 的内存里——
gpt-oss prefill 上虽然写到 `torch.zeros((n_tokens, N))` 上的“下一行”
不会立刻被发现，但对随机 N 的微基准（`N=80, BN=64` 等）会直接挂掉。

修正后：

```python
actual_n = (N // 2) if DO_SWIGLU else N
mask_y   = (offs_y_m[:, None] < M) & (offs_y_n[None, :] < actual_n)
```

新增回归 UT `test_oob_n_regression_bf16` 用 `(M=64, N=80, BN=64)` 复现。

### 4. UT 覆盖（共 34 passing）

新增 13 个用例，覆盖

| 用例 | 内容 |
|-----|-----|
| `test_mxfp4_x_mxfp4_gating[...] × 3` | 多种 `(M, N, K)` 的 e2m1×e2m1 dense GEMM 数值正确 |
| `test_fp8_x_mxfp4_gating[...] × 4`   | `(fmt ∈ {e4m3,e5m2}) × shape` 的 fp8 + global scale × mxfp4 |
| `test_mxfp_ragged_combine[e2m1/e4m3]` | per-expert (E=2) 的 ragged combine，含 scatter / no-scatter 分支 |
| `test_mxfp_dispatch_swiglu[e2m1/e4m3]` | per-expert dispatch+swiglu（验证 SwiGLU split-N + scaled MFMA + ragged 一起跑） |
| `test_oob_n_regression_bf16`         | bf16 path 的 `N % BLOCK_N != 0` 回归 |
| `test_scaled_kernel_no_register_spill` | scaled kernel 一次 launch 后从 `device_caches` 取出汇编，断言 sgpr/vgpr/scratch 0 spill |

```
$ python3 -m pytest test/ops/test_moe_gluon.py -v
... 34 passed in 1.53s
```

### 5. Microbench（gpt-oss-120b H=I=2880, E=128, topk=4）

测得三条 scaled-MFMA path 相对自己 bf16 同名 kernel 的加速比
（同样的 register-staged 流水、同样的 autotune block size）：

| Kernel | shape (B / M_d) | bf16 TFLOPs | mxfp4×mxfp4 TFLOPs | fp8×mxfp4 TFLOPs | 加速 |
|--------|-----------------|------------|---------------------|-------------------|------|
| K1 gating  | B=8192          | 62  | 132 | 124 | **2.1× / 2.0×** |
| K1 gating  | B=4096          | 31  | 76  | 80  | 2.5× / 2.6× |
| K2 dispatch+SwiGLU | B=8192 M_d=32768 | 364 | 436 | 408 | **1.20× / 1.12×** |
| K2 dispatch+SwiGLU | B=4096 M_d=16384 | 307 | 403 | 376 | 1.31× / 1.22× |
| K2 dispatch+SwiGLU | B=32 M_d=8192   | 172 | 232 | 237 | 1.35× / 1.38× |
| K3 combine | B=8192 M_d=32768 | 274 | 402 | 378 | **1.47× / 1.38×** |
| K3 combine | B=4096 M_d=16384 | 254 | 357 | 330 | 1.41× / 1.30× |
| K3 combine | B=32 M_d=8192   | 153 | 201 | 204 | 1.31× / 1.33× |

总结：

* 内存带宽受限的窄 GEMM（K1 gating, `N=128`）受益最大 —— mxfp4
  把 weight 字节数压成一半，直接拉到 **2× bf16**；
* compute-bound 的大 prefill (K2/K3 `M_d ≥ 16k`) 也有 **1.2–1.5×**，
  因为 scaled MFMA 的 `16x16x128` 指令 GFLOPS 比 `16x16x32` 高；
* 整张表 11 个 scaled kernel 变体全部 **0 spill / 0 scratch**
  (`sgpr ∈ [42, 68]`, `vgpr ∈ [86, 156]`, occupancy 3–5)。
  bf16 kernel 的旧 profile 保持不变。

### 6. 这次 commit 还没做的

* **真 LDS multi-buffer 流水**：仍是 register-staged。Update 4 主线
  做完后这个上来当首要项；scaled kernel 已经把 base pointer 全部
  scalar 化 (`w_base_offset = expert_id * stride_we`，`expert_id` 是
  `gl.load(expert_remap_ptr + compact_idx)` 得来的纯 scalar)，应当
  能解掉 Update 2 时碰到的 `unrealized_conversion_cast` MLIR 错误。
* **与 `triton_kernels.matmul` scaled path 的 head-to-head**：上游
  scaled path 需要构造 `PrecisionConfig(out_dtype, b_mx_scale=...)`
  + `flex_ctx`，下一步把 microbench 里 baseline 换成 scaled 上游版本
  做 apples-to-apples。
* **Per-block expert id 构造移到 Triton 端**（与 RDNA4 例子里
  `expt_data` 一致）。

## 已知 follow-up

1. **真正的 LDS multi-buffer 流水（仍是最高优先级，跨 bf16 与 scaled
   两条 path）**:
   现在用的是 register-staged 单缓冲流水。bf16 K1 (`N=128` 极窄)
   已经 1.6×–2.5× 超过 baseline；但 K2/K3 在 `M_d ≥ 8192`（gpt-oss
   prefill）上仍是 0.5–0.6×；scaled path 的 K2/K3 同样 register-staged,
   差距来源相同。把 `gl.amd.cdna4.async_copy.buffer_load_to_shared(...)`
   + `commit_group()` + `wait_group(N)` 真正接通（参考
   `gluon-kernels/kernels/cdna4/gemm/f16_gemm_gfx950.py`）后预期能到
   `0.9× ~ 1.1×` baseline。早期碰到的 MLIR `unrealized_conversion_cast`
   主要是 per-expert 动态 base pointer 的 legalization 问题，引入
   active-expert remap 之后 `expert_id` 已经是从 small i32 表 load
   出的 scalar，已经准备好再次尝试。
3. **Fused swiglu + GEMM2**：把 RDNA4 例子里
   `MoEPipelinedProgram + activation_fn` 这套 fused activation 路径搬过来，
   省一次 HBM 往返；MI355 上由于 LDS 紧张，需要小 block。
4. **Per-block expert id 构造移到 Triton 端**：现在用 Python 在 host 端
   遍历 `slice_sizes` 拼一个 expert_id 数组。RDNA4 例子里这一步是在
   kernel 里通过 `expt_data` (`ragged_metadata_fields`) 算的，更省 host 时间。
5. **Combine 阶段 scatter**：当前 scatter 走 `gl.store` 加 mask；后续可以换
   `gl.amd.cdna4.buffer_store` 拿到更稳定的 burst write。

## Reproduction

```
# build (already done)
docker exec kylewng_triton_dev_mi355_1505_dev bash -c '
  cd /root/code/tokenspeed && bash build.sh'

# unit tests
docker exec -w /root/code/tokenspeed/tokenspeed-kernel \
  kylewng_triton_dev_mi355_1505_dev bash -c '
  HIP_VISIBLE_DEVICES=2 python3 -m pytest test/ops/test_moe_gluon.py -v'

# micro-bench
docker exec -w /root/code/tokenspeed/tokenspeed-kernel \
  kylewng_triton_dev_mi355_1505_dev bash -c '
  HIP_VISIBLE_DEVICES=2 python3 -u benchmarks/moe_gluon_microbench.py \
    --M 1024 --N 2880 --K 2880 --E 32'

# enable in gpt-oss-120b
docker exec kylewng_triton_dev_mi355_1505_dev bash -c '
  cd /root/code/tokenspeed && \
  HIP_VISIBLE_DEVICES=2,3 TOKENSPEED_MOE_GLUON=1 \
  LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib \
  python3 -m tokenspeed.api_server \
    --model amd/gpt-oss-120b-w-mxfp4-a-fp8 \
    --attn-tp-size 2 --moe-tp-size 2 --max-model-len 80000 \
    --trust-remote-code --reasoning-parser gpt-oss \
    --kvstore-ratio 0.25 --enable-cache-report \
    --host 127.0.0.1 --port 8000'
```
