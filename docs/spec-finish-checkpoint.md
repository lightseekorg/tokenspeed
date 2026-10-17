# 投机推理下 finish 检查点错位:排查、复现与修复方案

> 临时讨论文档,讨论结束后删除。排查基线:main `d20c74d58`;修复基于 main `d473d3f77`,随本 PR 提交。本文记录一次问答式排查的完整过程:问题先用单测复现,修复后同一组用例转绿。

## 结论速览

- **问题**:开投机推理时,如果停止条件(EOS、`stop_token_ids`、`max_new_tokens`)落在 verify 窗口的中间,finish 发布到前缀缓存里的条目会和它的 key 对不上。
  - **linear state(KDA / GDN / Mamba2)**:发布的检查点比 key 多吃进了 `a − e` 个 token。`a` 是这一轮被接受的 token 数,`e` 是 CPU 实际上报的 token 数。**开不开 overlap 都会发生。**
  - **KV**:只在**投机加 overlap**时出现。发布出去的页,其 hash 和页里装的 KV 不是同一串 token。
  - **只开 overlap、不开投机**:没有问题。
- **影响**:hybrid 模型下一轮如果命中 finish 检查点,会从一个错误的 state 继续往下算,输出直接出错。KV 错位要求下一轮 prompt 恰好包含"EOS 加 dummy forward 的续写",触发概率低,但违反了"发布的块与 key 一致"这条不变式。
- **根因**:finish 那一轮,CPU(`OutputProcesser`)在停止 token 处 `break`,只上报了一部分 token;GPU 的 state commit 和 `valid_cache_lengths` 却按全部接受数推进了。overlap 下已经派发出去的 dummy forward 从 GPU 的真实进度接着写,而它的 token 又被追加在截断后的 token 表后面。
- **复现**:两个单测,分别对应 state 和 KV。真实的调度器、output processor 和 runtime 状态全部参与,三种组合的结果和推演完全一致。
- **修复(本 PR)**:finish 检查点**统一不含停止 token**;在 GPU 侧按停止条件截断接受长度;已停止请求的 dummy forward 当作 padding 行处理;停止 token 表放在 `RuntimeStates` 里,容量 16,超出直接拒绝(400)。修复后复现用例全部通过,三种调度组合的端到端冒烟也一致。

---

## 1. 背景

### 1.1 state group 的两槽表与自然检查点

state group(KDA / GDN / Mamba2 这类 linear attention 的递推状态)没有 token 跨度,每个请求固定占两个槽(`layers/attention/backends/state/checkpoint.py`):

| 槽 | 含义 |
|---|---|
| slot 0 | 输入检查点:命中的前缀检查点(多个请求共享,只读)、上一次 forward 写出的块,或者全零块;用过之后置为 null |
| slot 1 | live 块,本次 forward 写入 |

规则只有一条:slot 0 不为空就读 slot 0,否则读 slot 1;永远写 slot 1。prefill、decode、verify 都遵守这条规则。

调度器每次准入时会对两槽做一个 `StateStep` 操作:prefill chunk 和 fused 角色的第一次 decode 是 `kRotate`(slot 1 挪到 slot 0,slot 1 换成新块);之后每一步 decode 是 `kRelease`(释放 slot 0,slot 1 原地演化)。

state 只在"自然检查点"处发布:prompt 结束、落在 junction 上的 chunk 结束、finish、retraction。普通的 chunk 结束和普通的 decode 步不发布 state。发布用的 key 是"精确 token 数 `T`"的 hash,不要求页对齐。

### 1.2 什么时候插入前缀缓存

插入分成两半:**hash 是调度器在 CPU 上用自己的 token 表算的,块里的内容是 GPU 写进去的。** 命中要正确,就要求每个位置 p 上,token 表记录的 token 等于 GPU 在 p 处实际写入的 token。

| 时机 | 插入的内容 | 用的位置 T |
|---|---|---|
| prefill 的下一个 chunk 准入时(或 fused 的第一次 decode 准入时) | 填满的整页。chunk 正好结束在 junction 上,或这是 prompt 结束时,还会插入尾页和 state | 调度器排定的 chunk 结束位置(数据面是 FIFO,命中者的 forward 一定排在写入者后面) |
| 每次 decode 准入时 | 只插新填满的整页,不插尾页,也不插 state | 已落地结果对应的 `TokenSize − 1` |
| finish 结算(`Finishing` 状态,且在飞计数归零) | 整页、尾页、state 全部插入 | `TokenSize − 1` |
| retraction(只挑没有 forward 在飞的请求) | 同 finish | 同 finish |

其中 `NumComputedTokens() = TokenSize() − 1`(`tokenspeed-scheduler/csrc/scheduler/request.cpp`):token 表里除最后一个之外都已经被计算过,最后一个是采样出来、还没被任何已落地 forward 消费的 token。

### 1.3 overlap 调度与 dummy forward

开 overlap 时,调度器比 GPU 领先一步:第 k 步还在 GPU 上跑,第 k+1 步就已经派发出去了。如果第 k 步落地后 CPU 判定请求结束,第 k+1 步这时已经在 GPU 上了。它会以停止 token 作为输入跑一遍,本文把这一步叫做 **dummy forward**。

finish 不会立刻释放资源。请求先进入 `Finishing` 状态,等 dummy forward 也落地、在飞计数归零之后,`settleFinishedRequests` 才发布并释放。dummy forward 的结果会通过 output processor 的 untracked 分支原样上报,追加进 token 表,这样 `TokenSize − 1` 就能和 GPU 实际消费的进度精确对应。

### 1.4 投机 verify 怎么提交 state

两种提交方式都只在 commit 时写 slot 1,verify 过程中不写任何缓存块:

| | 关闭 ReplaySSM(per-position scratch) | 开启 ReplaySSM |
|---|---|---|
| verify 期间 | 每个 draft 位置的递推状态写进 scratch | 不写 state,只把每个 token 的紧凑投影存进 payload |
| commit | 把 scratch 里第 steps 行拷到 slot 1 | 从读页重放 steps 个 token,结果写进 slot 1 |

`--disable-replay-ssm` 只对 Qwen GDN 和 Nemotron-H Mamba2 生效。KDA(Kimi-K3、GLM-5.3-Flash)是否走 replay 由 kernel 能力自动决定。

---

## 2. 排查过程

| 步骤 | 问题 | 结论 |
|---|---|---|
| 1 | 投机和 ReplaySSM 下,KDA 怎么做前缀缓存? | 三者正交:缓存什么、何时发布由两槽表和自然检查点决定;投机只决定读哪块、写哪块;ReplaySSM 只决定 commit 怎么实现 |
| 2 | 开 overlap 时遇到 EOS 或 `max_tokens`,发布的 state 是哪个位置的? | 位置是 `TokenSize − 1`。开 overlap 时,dummy forward 吃进了 EOS,所以检查点位于 EOS 之后。**同时发现疑点**:投机时 CPU 在停止 token 处截断了上报,GPU 却按全部接受数提交 |
| 3 | 只开 overlap、不开投机,是不是也有 bug? | 不是。逐 token 对照后,token 表、KV、state 三者一致,因为每步只产出一个 token,`break` 截不掉任何东西 |
| 4 | dummy forward 会不会写坏已经发布的 state? | 不会。在飞时发布的块随即被 rotate 到 slot 0,后续只读;finish 要等在飞计数归零才发布;retraction 只挑没有 forward 在飞的请求。已有测试 `FinishWithAForwardInFlightSettlesAtTheFrontierThatForwardLands` 覆盖了这个时序 |
| 5 | 插入前缀缓存的时机是什么?用具体例子推演 | 推演出投机加 overlap 时 KV 的 hash 和内容错位(见第 3 节) |
| 6 | 写单测复现 state 问题 | 两个投机用例失败,只开 overlap 的用例通过 |
| 7 | KV 部分能不能也写单测复现? | 起初认为不行:KDA harness 没有 KV,输入按位置取,dummy forward 的写入位置还是测试自己指定的,会构成循环论证。后来改为让写入位置由真实的 `InputBuffers` 和 `advance_accepted_frontier` 决定,成功复现 |
| 8 | 修法 | 否掉"finish 时全部上报"(会丢命中);改为 GPU 截断;统一不含停止 token;dummy forward 当 padding 行处理;确定停止 token 表的设计 |

---

## 3. 根因与例子

### 3.1 根因

finish 那一轮,假设 verify 接受了 a 个 token,其中第 e 个(e < a)是停止 token:

- **GPU**:state commit 推进 a 步(`commit_state_pages`,steps 等于接受数);`advance_accepted_frontier` 也把 `valid_cache_lengths` 推进 a。
- **CPU**:`OutputProcesser.post_process_forward_op` 逐个 token 检查 finish,在第 e 个 token 处 `break`,只上报 e 个。调度器的 `TokenSize` 只增加了 e。

开 overlap 时,dummy forward 的输入来自 `future_input_map`,位置来自 `valid_cache_lengths`(`InputBuffers.fill_input_buffers` 中的 `fused_decode_input_prep`)。所以它接着 GPU 的真实进度写,它的输出却被追加在截断后的 token 表后面。

### 3.2 具体例子(投机加 overlap,draft 宽度 4)

第 k 步之前,token 表有 x₀…x₁₁ 共 12 个 token,位置 0 到 10 的 KV 已经写好,x₁₁ 还没被消费。

- **第 k 步**:verify `[x₁₁, d₁, d₂, d₃]`,写在位置 11 到 14,接受 3 个,输出 o₁ o₂ o₃,其中 o₁ 是 EOS。
  - GPU:位置 11、12、13 的 KV 分别是 x₁₁、o₁、o₂;state 吃到 o₂;`valid_cache_lengths` 变成 14。
  - CPU:o₁ 是 EOS,`break`,只上报 [o₁]。token 表变成 x₀…x₁₁, o₁,共 13 个。
- **第 k+1 步(dummy forward)**:从位置 14 开始,输入 [o₃, …],接受 2 个;位置 14、15 的 KV 是 o₃、g₁;输出 g₁ g₂,原样上报。token 表变成 x₀…x₁₁, o₁, g₁, g₂,共 15 个。
- **结算**:`T = 15 − 1 = 14`,插入位置 [0, 14) 的 KV 和 state。

| 位置 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|
| token 表(用来算 hash) | x₁₁ | o₁ | **g₁** | g₂ | — |
| GPU 实际写入的 KV | x₁₁ | o₁ | **o₂** | o₃ | g₁ |

- **KV**:位置 13 在插入范围内,hash 是按 g₁ 算的,块里装的却是 o₂ 的 KV。
- **state**:key 声称吃完了位置 0 到 13,slot 1 实际吃到了位置 15,多吃了 2 个,也就是 a − e = 3 − 1。

### 3.3 三种组合对比

| 组合 | KV | state | 原因 |
|---|---|---|---|
| 只开投机 | 正确 | **超前 a − e** | 没有 dummy forward,`T` 落在已经正确写入的位置之内;但 state 已经多吃了 a − e 个 token |
| 投机加 overlap | **错位** | **超前 a − e** | dummy forward 从 GPU 进度续写,它的 token 接在截断后的表后面 |
| 只开 overlap | 正确 | 正确 | 每步只产出 1 个 token,不会被截断;dummy forward 的 token 全量上报 |

---

## 4. 单测复现

以下"结果"列是修复前的结果;修复后两个文件的全部用例通过。两个用例都把修复后的执行顺序(`bind_stop_criteria`、`apply_stop_conditions`、`masked_state_tables`)按 `ModelExecutor` 的顺序调用,因此去掉修复的任何一半都会转红:去掉截断,六个用例全部失败;去掉 padding,两个 overlap 用例失败(检查点比 key 多吃一个 token)。

### 4.1 state:`test/runtime/test_kda_state_eviction_spec.py::test_the_checkpoint_published_at_finish_holds_the_state_its_key_names`

- 用真实的 C++ 调度器、真实的 KDA backend(replay commit),以及逐 token 计算的 oracle。
- 发给调度器的事件由真实的 `OutputProcesser` 生成,所以截断就来自被测代码。
- 判定方法:另一个请求按调度器的 token 表命中,确认命中位置 `T`,再检查发布的检查点是否等于 oracle 在 `T` 处的状态。conv state 和 oracle 是逐位相等的,可以精确判断检查点实际停在第几个 token。

| 用例 | 结果 |
|---|---|
| spec(宽度 3) | 失败:`the checkpoint published under the key of 10 tokens holds the state after [12] tokens` |
| spec_overlap | 失败:key 覆盖 12 个 token,state 实际吃到 14 个 |
| overlap(宽度 1) | 通过 |

### 4.2 KV:`test/runtime/test_spec_finish_kv_publication.py::test_a_prefix_hit_after_a_finish_reads_the_kv_of_its_own_tokens`

为了避免循环论证,**dummy forward 写在哪里必须由真实代码决定**:

- **真实组件**:调度器、`OutputProcesser`、`RuntimeStates`、`InputBuffers.fill_input_buffers`(决定每一步的输入 token 和位置)、`advance_accepted_frontier`(按 GPU 侧的接受数推进 `valid_cache_lengths`)。调用顺序和 `ModelExecutor` 一致。
- **测试替代的部分**(本机缺的只有这些):模型换成一条固定的贪心 token 流,第 pos 个位置"写 KV"就是记下当时的输入 token,输出取 token 流的第 pos + 1 个;drafter 按真实契约,把最后一个被接受的 token 和后续 draft 写进 `future_input_map`。
- **判定方法**:probe 命中后,按调度器给它的页表(包括尾页拷贝)逐位置比较 KV 内容和 probe 的 token。

| 用例 | 结果 |
|---|---|
| spec | 通过 |
| overlap | 通过 |
| spec_overlap | 失败:`[(13, 115, 113)]`,位置 13 的 key 记的是 115,slot 里装的却是 113 的 KV |

---

## 5. 修复方案

### 5.1 被否掉的方案

| 方案 | 否掉的原因 |
|---|---|
| finish 那一轮把所有被接受的 token 全部上报 | 结果正确,但检查点的 key 会包含 EOS 之后、客户端从未见过的 token,下一轮不可能命中,hybrid 模型就得重算整段输出 |
| 把"检查点含不含 EOS"做成配置 | "不含 EOS"在两种模板下都能命中(模板保留 EOS 时多重算 1 个 token;去掉 EOS 时正好是前缀);"含 EOS"只在模板保留 EOS 时才能命中,而且在 EOS 是 bonus token、或不开 overlap 时根本做不到。做成配置还会让检查点位置随 overlap 变化,违反"overlap 是同一条路径上的参数"这条原则 |
| dummy forward 强制按单 token verify | 接受数会变成 1,吃进 EOS 并上报 1 个 token,检查点重新包含 EOS,又会随 overlap 变化 |

### 5.2 实现

1. **finish 检查点统一不含停止 token**,不加配置。三种组合下检查点的位置相同。
2. **在 GPU 侧截断接受长度**:`ModelExecutor._run_sampling` 在 sample/verify 之后、任何消费者读取之前,调用 `tokenspeed_kernel.ops.metadata.apply_stop_conditions`(一个 Triton kernel,在 decode CUDA graph 里一起 capture),原地按第一个停止 token 和剩余 `max_new_tokens` 预算截断。预算等于 `token_limit − valid_len − 1`,完全由已有量算出,不需要逐步递减。截断后的值同时用于 state commit、`advance_accepted_frontier`、上报的输出长度和 drafter(drafter 拿到的是下限为 1 的副本,因为它按"最后一个被接受位置"取下标)。这样 CPU 判定 finish 时,停止 token 一定是本轮最后一个被接受的 token,`break` 截不掉任何东西,token 表和 GPU 自然一致。prefill 行只在补完 prompt 的那个 chunk 上判断(新增的 `extend_prompt_lens_buf`),采样到停止 token 或 `max_new_tokens == 1` 时置"已停止"。
3. **已停止请求的 dummy forward 当作 padding 行处理**:截断触发时,在设备上置"已停止"标记。已经派发的 dummy forward 读到这个标记后:
   - state 的读写目标都设成 −1,和 graph padding 行的约定一样。做法是 `RuntimeStates.masked_state_tables` 在 forward 前把该行所有 state group 的两个槽都改成 −1;之后的 eager、graph replay、verify 和 commit 都从表里取块号,统一看到 −1。所有 state kernel 本来就会跳过负的索引:decode kernel 遵守 `pad_slot_id`;`copy_state_rows` 遇到负的目标行不写;GDN 和 KDA 的 replay commit 都是 `write_idx >= 0` 才写。`compute_state_block_indices` 的调试校验把"两个槽都是 −1"认作 padding 行,不再报错。
   - 接受长度设成 0:`valid_cache_lengths` 不推进,上报空的 `ExtendResult`,`Finishing` 状态只会把在飞计数减 1。`commit_state_pages` 里"步数至少为 1"的下限保留,因为该行的写入页是 −1,commit 什么都不写。
   - 如果 dummy forward 是 fused 角色的第一次 decode(prefill 就采样到了停止 token),它是 rotate 步,slot 1 是从没写过的新块。finish 在 `T` = prompt 长度处发布时,这个 key 已经有 prompt-end 检查点(该请求的 slot 0 仍持有它,不会被淘汰),前缀索引按"已有的 canonical 块优先"去重,所以发布出去的是正确的 prompt-end 状态。
   - KV、draft KV、index_k 照常写,都只会写到 `T` 及之后的位置,不在 key 的范围内;命中的请求会先对尾页做 CoW 再接着写。
   - 不需要强制单 token verify,也不改变 verify 宽度和 graph 形状。
4. **停止 token 表**:
   - 放在 `RuntimeStates` 里(贪心等无状态 sampling backend 没有按 slot 存的状态,不能依赖它)。新增 `stop_token_ids [slot, 16]`(int32,不足的用 −1 填充)、`token_limits [slot]` 和 `stopped [slot]` 三个张量。event loop 每一步把各请求的 `StopCriteria` 随 `PlannedForward` 交给 forward 线程,`bind_stop_criteria` 只重写换了主人的 slot(新 prefill、D 角色落地、restore 都覆盖到);三个张量通过 `SlotStateExporter` 导出,victim 的 `stopped` 永远是 False,导入时顺带清掉上一任的标记。
   - 内存可以忽略:每个 rank 的 slot 数等于 `--max-num-seqs` 除以 DP size,再加 2。即使单卡 1024 个并发,也只有大约 64KB。
   - **设备表和 CPU 的 `check_finished` 共用同一个 `resolve_stop_token_ids`**(`RequestState.stop_token_ids`,构造时算一次)。合并集合是 `stop_token_ids`、模型 EOS、tokenizer 的 EOS 和 `additional_stop_token_ids` 的并集;`ignore_eos` 时整段为空(注意,它连 `stop_token_ids` 也会关掉)。
   - **合并后超过 16 个就直接拒绝(400)**,接受和 OpenAI API 在这一点上不兼容。在 `RequestHandler` 里检查,错误信息写明用户实际可用的数量,即 16 减去引擎 EOS 的个数。
5. **兜底**:stop string、grammar、客户端 abort 这类只有 CPU 能判断的停止条件,如果 CPU 判定的 finish 早于本轮最后一个被接受的 token,就把本轮接受的 token 全部上报(`OutputProcesser` 的 `reported_ids`)。结果正确,只是放弃命中(stop string 通常会从返回文本里裁掉,下一轮本来也命中不了)。另外,一个仍被跟踪的请求如果某个 decode 一个 token 都没留,直接报错:只有已经被设备停掉的请求才会留 0 个,而它已经结束。

### 5.3 修复后的行为

| 场景 | finish 检查点位置 | 下一轮能否命中 |
|---|---|---|
| 投机,不开 overlap | EOS 之前 | 能 |
| 投机加 overlap | EOS 之前 | 能 |
| 只开 overlap | EOS 之前(原来是 EOS 之后) | 能 |
| `max_new_tokens` 截断 | 最后一个可见 token 之前 | 能 |
| stop string | 超出可见范围 | 不能,但结果正确 |

---

## 6. 验证

- **kernel**:`tokenspeed-kernel/test/ops/metadata/test_stop_conditions.py`,Triton kernel 与 tensor 路径对齐(含 mixed batch、padding 行、已停止行),以及 `assert_no_triton_compile`(batch、extend、输出个数变化不触发重编译)。
- **复现用例**:第 4 节两个文件,修复后全部通过;去掉修复的任一半会转红(见第 4 节开头)。
- **控制面**:`test_generation_output_processor.py` 新增停止集合合并、兜底全量上报、0 token 报错、超限拒绝四个用例;`test_gdn_state_paging.py` 新增 padding 行用例;`test_slot_state.py` 覆盖新增的按 slot 张量。
- **端到端冒烟**(H20,Qwen2.5-0.5B-Instruct,`DUMMY` drafter 加 `--synthetic-acceptance-length 4`,每轮满额接受):投机加 overlap、只开 overlap、投机不开 overlap 三种组合下分别检查 `max_new_tokens` 在窗口中间截断、prefill 采样到停止 token、decode 轮里遇到停止 token 三种情况。下一轮 prompt 带上完整输出(含停止 token)时,`cached_tokens` 都恰好等于停止 token 的位置(或最后一个保留 token 的位置)。decode graph 正常 capture,stop kernel 只在启动时编译一次。
- **未覆盖**:没有在真实的 hybrid(KDA/GDN)模型上端到端跑过;state 侧由第 4.1 节的用例在真实 KDA kernel 上验证。
