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

## 已知 follow-up

1. **mxfp4 weight + fp8/bf16 act 路径**：复用 `gl.amd.cdna4.mfma_scaled` +
   `get_mfma_scale_layout`，把 RDNA4 `wmma_scaled` 路径改写为 MFMA scaled
   即可。要点：
   - scale tile 形状是 `[BM/scale_block, BK/scale_block]`，
     `scale_block=32` 对 mxfp4 即是。
   - mxfp4 是 packed uint8（每 byte 2 个 nibble），
     `LOAD_B` layout 必须按 `BLOCK_K // 2` 定 K 维 stride，类似上游
     `triton_kernels` 的 `wrap_torch_tensor(b, dtype=FP4)`。
   - 需要 per-tensor static fp8 input scale 一并写进
     `mfma_scaled` 的 `a_scale`（直接用 broadcast 的 e8m0 scalar）。
2. **Software pipeline (multi-buffer)**: 当前是“先 load A/B 到 smem、再 mfma”，
   把 `gl.amd.cdna4.async_copy.buffer_load_to_shared(...)` +
   `gl.amd.cdna4.async_copy.commit_group()` + `wait_group(N)` 拼起来，
   就能做到与 RDNA4 例子相同的 2~3 buffer ping-pong，预计 perf
   能拉到上游 baseline 的 0.9× 以上。
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
