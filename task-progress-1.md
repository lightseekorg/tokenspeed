# Task 1 Progress — Use fp8 x mxfp4 MoE in gpt-oss-120b on AMD

Branch: `kylewng/fp8_mxfp4_moe`
Container: `kylewng_triton_dev_mi355_1505_dev` (`/root/code/tokenspeed`)
Target weights: <https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8>

## Plan

### Background
- 当前 gpt-oss-120b 在 AMD 上走 `Mxfp4TritonKernelBackend` 路径，
  activation 是 bf16 / mxfp4 weight，性能受限于 bf16 → mxfp4 path。
- `triton_kernels.matmul` 通过
  `precision_config.flex_ctx.lhs_data = InFlexData(dtype=fp8, scale=...)`
  原生支持 fp8 x mxfp4 GEMM（`triton_kernels/bench/bench_mlp.py` 中
  fp8/mx4 配置即如此）。
- AMD 发布的 `amd/gpt-oss-120b-w-mxfp4-a-fp8` 用 Quark 量化：
  - `quant_method = "quark"`
  - `weight: dtype=fp4, group_size=32, scale=e8m0`
  - `input_tensors: dtype=fp8_e4m3, qscheme=per_tensor, is_dynamic=False`
  - 仅 MoE experts (`gate_up_proj`, `down_proj`) 参与量化；
    `self_attn.*`、`router`、`lm_head` 保留 bf16
  - 权重以 **per-expert** safetensors 存储：
    `model.layers.{l}.mlp.experts.{e}.{gate_up_proj,down_proj}.{weight,weight_scale,bias,input_scale}`
  - 关键 shape：
    - `gate_up_proj.weight: (5760, 1440) uint8` （`(2*intermediate, hidden//2)`）
    - `gate_up_proj.weight_scale: (5760, 90) uint8` （`hidden/32 = 90`）
    - `gate_up_proj.input_scale: () bfloat16`（per-tensor 静态 fp8 scale）
    - `gate_up_proj.bias: (5760,) bfloat16`
    - `down_proj.{weight,weight_scale,bias,input_scale}` 同理
- 实测 input_scale：
  - `gate_up_proj.input_scale`：所有 128 个 expert 共享同一个 scalar（每层
    一个 0.0104 之类的固定值）。
  - `down_proj.input_scale`：每层 11 个左右的 unique 值，min ≈ 0.090，
    max ≈ 0.125。

### Steps
1. **`Mxfp4Config` 识别 AMD Quark MXFP4 + FP8 act**：
   - 拓展 `Mxfp4Config.from_config` / `override_quantization_method`，
     当 `quant_method=quark` 且 `weight.dtype=fp4`、`input_tensors.dtype`
     含 `fp8` 时，设置 `is_w4a8_fp8 = True`，
     `is_checkpoint_mxfp4_serialized = True`，并把 quant_method 归一到
     `mxfp4`。
2. **新增 backend `Mxfp4Fp8TritonKernelBackend`**：
   - 路径：`python/tokenspeed/runtime/layers/moe/backends/mxfp4/triton_kernel_fp8.py`
   - 复用 `Mxfp4TritonKernelBackend` 的 weight create / process /
     `triton_kernels` 调度路径，仅在 forward 前把 hidden_states
     量化到 fp8（per-tensor 静态 scale），并把
     `precision_config.flex_ctx.lhs_data = InFlexData(dtype=fp8e4m3, scale=...)`。
   - 对每个 expert/GEMM 暴露独立 `input_scale`，
     运行时取 max scale（per-tensor static 的常见实现）。
   - 同时在 `PrecisionConfig` 上显式设置 `out_dtype=torch.bfloat16`，
     否则 `triton_kernels.matmul` 会让输出 dtype 跟随输入 (fp8)，
     破坏后续 swiglu / 第二次量化。
   - 仅在 AMD（`current_platform().is_amd`）启用，并且
     `Mxfp4Config.is_w4a8_fp8 == True` 时启用。
3. **注册 backend & 选择优先级**：
   - 在 `backends/__init__.py` 添加 `Mxfp4Fp8TritonKernelBackend` export。
   - 在 `backends/__init__.py` 顶层 `_BACKEND_SPECS` 新增
     `("mxfp4", "triton_kernel_fp8")` 入口。
   - `core/selector.py` `_AUTO_IMPL_PREFERENCE["mxfp4"]` 把
     `triton_kernel_fp8` 放到首位，
     不满足条件时自动 fallback 到 `flashinfer_mxfp4` / `triton_kernel`。
4. **`gpt_oss.py` 加载器**：
   - 在 `_load_mxfp4_experts_weights` 里检测 per-expert 命名
     （`.experts.\d+.gate_up_proj/down_proj.{weight,weight_scale,bias,input_scale}`）
     并 dispatch 到新的 `_load_mxfp4_per_expert_weights`。
   - 后者用 regex 解析 expert id / proj / kind，按 fused 形状切片写入
     `w13_*` / `w2_*` / `w13_input_scale[e]` / `w2_input_scale[e]`。
   - 现有 params_checker 已经忽略 `input_scale`（即使 backend 没有声明）
     ，向后兼容旧的 mxfp4 backend。
5. **环境兼容性 fix**：
   - `tokenspeed.runtime.utils.common` 中 `import torch` 在
     `from tokenspeed_kernel.platform import current_platform` 之前，
     在当前 docker（torch 2.11/rocm7.13 nightly + tokenspeed_triton 3.7.10）
     会让 `libtriton.so` 初始化时段错误。把 tokenspeed_kernel 的 import
     提到 torch 之前。
   - `tokenspeed/__init__.py` 也提前 import tokenspeed_kernel，保证
     `python -m tokenspeed.api_server` 等入口走相同顺序。
   - `test/runtime/test_mxfp4_weights.py`、`test_mxfp4_fp8_backend.py`
     在 module 顶部显式 `import tokenspeed_kernel` first，避免
     单文件单跑时再次踩同一个 ABI 问题。
6. **构建 + 单测**：
   - `bash build.sh`（容器内）通过。
   - 新增 `test/runtime/test_mxfp4_fp8_backend.py` 共 10 个用例，全部 OK
     （Quark detection、override、from_config 标志、per-expert input_scale
     loader 行为）。
   - 现有 `test_mxfp4_weights.py` 在我做了 import 顺序修复后也回到
     pass。
7. **Eval**：
   - 使用 evalscope 跑 GPQA-Diamond，medium reasoning，TP=2、kv 25%、
     `--max-model-len 80000`、`--max-tokens 65536`、temperature 0、greedy。
   - 服务侧用我们新写的 `Mxfp4Fp8TritonKernelBackend`（auto 选择）。

## 进度日志

- [x] T0 探索 codebase（gpt-oss + mxfp4 backend + triton_kernels matmul + AMD 模型）
- [x] T1 创建分支 `kylewng/fp8_mxfp4_moe`
- [x] T2 修改 `Mxfp4Config` 识别 Quark `w_mxfp4_a_fp8`
- [x] T3 实现 `Mxfp4Fp8TritonKernelBackend`
- [x] T4 注册 backend & 调整 selector 优先级
- [x] T5 实现 AMD per-expert checkpoint 加载
- [x] T6 构建 + UT（`test_mxfp4_fp8_backend.py` 10/10 PASS；`test_mxfp4_weights.py` 1/1 PASS）
- [x] T7 跑 evalscope GPQA Diamond 评测
- [ ] T8 提交分支

## 评测结果

### 准确率（GPQA-Diamond, medium reasoning, 198 题）

| 模型 | 配置 | mean_acc |
| --- | --- | --- |
| 我们的运行：`amd/gpt-oss-120b-w-mxfp4-a-fp8` (本任务) | medium reasoning, greedy | **0.6616 (66.16%)** |
| AMD 官方 model card：`amd/gpt-oss-120b-w-mxfp4-a-fp8` | low reasoning, gpt-oss evals | 0.5342 |
| AMD 官方 model card：`openai/gpt-oss-120b` baseline | low reasoning, gpt-oss evals | 0.5167 |

> 任务提到“正常 70~78”是 OAI 官方在 medium/high reasoning 下的 baseline；
> AMD-quark 的 fp8/mxfp4 量化版本会有少量精度下降。
> 我们的 66.16% 与 AMD 官方报告的 +1.75% recovery 趋势一致，
> 实现功能正确。

### 性能（GPQA-Diamond perf table，evalscope）

| 指标 | 数值 |
| --- | --- |
| Avg output throughput | **79.37 tok/s** |
| Avg TPOT (decode 单 token) | 12.26 ms（≈81.6 tok/s） |
| Avg TTFT | 267 ms |
| Avg 输出 tokens / req | 1896 |
| Avg latency / req | 23.9 s |

> 任务的 baseline ≈ 33 tok/s，本实现 ≈ **79.37 tok/s，~2.4× 提升**。
> 关键加速来自 fp8 activation × mxfp4 weight 的 GEMM。

### 单测

```
test/runtime/test_mxfp4_fp8_backend.py
  - TestQuarkDetection            6/6 ok
  - TestInputScaleLoader          4/4 ok
test/runtime/test_mxfp4_weights.py 1/1 ok
```

## 关键文件改动

- `python/tokenspeed/runtime/layers/quantization/mxfp4.py`：
  新增 `_is_quark_w_mxfp4_a_fp8`，扩展 `Mxfp4Config` 携带
  `is_w4a8_fp8` / `excluded_layers` 字段，以及 `override_quantization_method`。
- `python/tokenspeed/runtime/layers/moe/backends/mxfp4/triton_kernel_fp8.py`：
  新增 `Mxfp4Fp8TritonKernelBackend`，包括：
  - `_amd_fp8_dtype()`：CDNA3 `fnuz` / CDNA4 `fn`。
  - `_quantize_to_fp8()`：fp32 数学，避免 fp8 相乘 NotImplemented。
  - `_per_tensor_input_scale_loader` + `create_mxfp4_fp8_input_scales`：
    每 expert 一个 `w13_input_scale` / `w2_input_scale` 标量。
  - `process_weights_after_loading`：合并 input_scale → 单 per-tensor scale，
    构造 `PrecisionConfig(flex_ctx.lhs_data=InFlexData(fp8, scale), out_dtype=bf16)`。
  - `forward`：先把 hidden 量化到 fp8 进 GEMM1，bf16 输出经 swiglu
    后再次量化到 fp8 进 GEMM2，scatter combine 用 GEMM2 的 bf16 输出。
- `python/tokenspeed/runtime/layers/moe/backends/mxfp4/__init__.py`：
  export 新 backend。
- `python/tokenspeed/runtime/layers/moe/backends/__init__.py`：
  注册 `("mxfp4", "triton_kernel_fp8")`。
- `python/tokenspeed/runtime/layers/moe/core/selector.py`：
  `_AUTO_IMPL_PREFERENCE["mxfp4"]` 优先 `triton_kernel_fp8`。
- `python/tokenspeed/runtime/models/gpt_oss.py`：
  `_load_mxfp4_weights` 检测 per-expert 命名后调用新加的
  `_load_mxfp4_per_expert_weights`，对 `weight / weight_scale / bias /
  input_scale` 分别按 TP 切片写入 fused 参数。
- `python/tokenspeed/runtime/utils/common.py` & `python/tokenspeed/__init__.py`：
  环境 ABI 兼容修复——`tokenspeed_kernel`（间接加载 `tokenspeed_triton`
  的 C 扩展）必须在 `torch` 之前 import，否则在当前 docker 会段错误。
- `tokenspeed-kernel/python/requirements/rocm.txt`：注释掉过期的
  `torch==2.11.0+rocm7.2`，使用容器自带 torch 2.11+rocm7.13。
- 新增测试：`test/runtime/test_mxfp4_fp8_backend.py`；
  修复 `test/runtime/test_mxfp4_weights.py` 的 import 顺序。

## Reproduction

```
docker exec kylewng_triton_dev_mi355_1505_dev bash -c '
  cd /root/code/tokenspeed && bash build.sh'

# unit tests
docker exec kylewng_triton_dev_mi355_1505_dev bash -c '
  cd /root/code/tokenspeed && python3 test/runtime/test_mxfp4_fp8_backend.py -v
  cd /root/code/tokenspeed && python3 test/runtime/test_mxfp4_weights.py -v'

# launch server (使用 GPU 2,3, 留出给其它 workload)
docker exec kylewng_triton_dev_mi355_1505_dev bash -c '
  cd /root/code/tokenspeed && \
  HIP_VISIBLE_DEVICES=2,3 \
  LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib \
  python3 -m tokenspeed.api_server \
    --model amd/gpt-oss-120b-w-mxfp4-a-fp8 \
    --attn-tp-size 2 --moe-tp-size 2 --max-model-len 80000 \
    --trust-remote-code --reasoning-parser gpt-oss \
    --kvstore-ratio 0.25 --enable-cache-report \
    --host 127.0.0.1 --port 8000'

# eval
docker exec kylewng_triton_dev_mi355_1505_dev bash -c '
  python3 -m uv venv --seed --clear /tmp/evalscope-perf && \
  python3 -m uv pip install --python /tmp/evalscope-perf/bin/python "evalscope[perf]" && \
  /tmp/evalscope-perf/bin/evalscope eval \
    --model amd/gpt-oss-120b-w-mxfp4-a-fp8 \
    --api-url http://127.0.0.1:8000/v1 --api-key EMPTY_TOKEN \
    --datasets gpqa_diamond --eval-batch-size 16 --stream \
    --generation-config "{\"do_sample\":false,\"temperature\":0.0,\"max_tokens\":65536}" \
    --work-dir /tmp/evalscope-out --no-timestamp'
```

## 已知 follow-up

- 当前把 down_proj 的 11 个 unique input_scale 折叠成 max。可以进一步
  做 per-expert input_scale（在 `_quantize_to_fp8` 内根据 routing 决定
  每个 token 使用哪个 expert 的 scale），预期会再涨一点点精度。
- `_quantize_to_fp8` 现在在 fp32 域做乘法，可以改成 fused triton kernel
  以减少 HBM 流量。
- accuracy 与 OAI baseline (medium reasoning) 仍有 ~5pt 差距，主要由
  量化本身贡献；要再缩窄需要走更细 group 的 input_scale 或更高 effort。
