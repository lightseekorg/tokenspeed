# Flat KV-Cache

## 1. C++ 侧的组件和几条关键路径

### CacheBlock 和 BlockPool

`CacheBlock` 是块的元数据:引用计数 + 内容哈希 + 所属池。`BlockPool` 管一个存储
层级的所有块:一条 LRU free list (管理可被分配的 block), 加一个哈希到块的索引。引用计数掉到 0 的块不清空, 而是带着哈希回到 free list 尾部, free list 中的 block 有两种复用方式:1. 被同前缀的请求命中,从 free list 中间摘走再次使用; 2. 走到 LRU 头被新分配拿走(驱逐), 这时才抹掉哈希。所以"缓存"和"空闲"不是两个集合,是同一条 free list。

block 0 是保留的 null 块, 只用来在块表里占位.

### BlockRef:引用的全部机制

拿块放块只有一条通道:`BlockRef`,move-only 的 RAII 句柄。全部生命周期:

* **诞生只有两种**:`Adopt(pool, block)` 接管一个刚从 free list 弹出的块(分配时
  引用已计 1,Adopt 不再加);`Share(pool, block)` 给一个已存在的块加一个引用——
  内部走 `TouchBlock`:若该块正躺在 free list 上(ref 0 的缓存块),先把它从
  free list 中间摘走(存的迭代器,O(1)),再 +1。
* **死亡只有两种**:析构自动调 `FreeBlock`(-1,归 0 则回 free list 尾);或
  `Release()` 交出裸指针放弃所有权——只给批量释放用:把一批 Release 出来的指针
  一次 `FreeBlocks` 逆序归还,因为**归还顺序就是将来被驱逐的顺序**,不能交给
  vector 析构的实现定义顺序。
* **谁在持有**:BlockTable(请求活着期间,结束时走 Release + FreeBlocks 批量
  路);load/store ticket(L2 传输在飞期间,走析构路——`LoadBackDone` 的全部
  释放动作就是 `flat_load_ops_.erase(op_id)`,ticket 里的 `vector<BlockRef>`
  随之析构,引用自动归还,代码里没有 unpin 函数,析构就是 unpin)。中途失败
  的路径同理:局部 BlockRef 出作用域即归还,不需要清理代码。
* **引用与哈希的组合就是块的状态机**:ref>0 且无哈希 = 正在写入;ref>0 且有哈希 =
  在用且可命中;ref=0 且有哈希 = 缓存中、可驱逐;null 块不参与计数。
* 匹配(查询)不产生引用,函数签名收 `const BlockPool&`。

### 块表和策略

`BlockTable` 是每请求每组一张的块表:逻辑块号到物理块的映射。滑动窗口把过期块
换成 null 块占位(`EvictToNull`),后面块的槽位号不变。

匹配和滑动的规则因注意力类型而异,写在三个 manager 里:`FullAttnManager`、
`SwaManager`、`MambaStateManager`。manager 不持有池,方法把池当参数收,同一份
匹配代码既查 device 池也查 host 池。`KvCacheCoordinator` 构造时把 manager 和池
绑起来(device 必选,host 可选),对外一个 `MatchPrefix` 入口。

### Acquire / Cache / Reclaim / Free:各组的同与异

一个 chunk 进来(T 个新 token),coordinator 对每个组依次做四件事,**前两件全组
一致,后两件因组而异**:

**Acquire(全组一致)**:每组按自己块表的尾块余量算需求——`BlocksNeededFor` =
`ceil((T - tail_avail) / block_size)`,先吃尾块空 slot,再从 free list 取新块
`Adopt` 进表。跨 group 遵循 all-or-nothing: 先按总需求
查询 free list 余量,任何一组不够则整步失败,已有状态不动。**mamba 组也照常按
token 数取块**——它的块表和别人一样逐块增长,差异全在后面的 Reclaim。这也是
"block_size 是记账粒度、不是内容承诺"的第一处体现:attention 块装这些 token
各自的 KV;mamba 块装一份"截至本块边界"的状态快照,里面没有 slot 阵列,但
记账仍按覆盖的 token 数走。

**CacheFullBlocks(注册,差异开始)**:chunk 写满的块按链式哈希(内容 + 前序块
哈希)登记进池索引,key 里编入组号。差异:

| 组 | 注册哪些块 |
|---|---|
| full / SWA | 范围内**全部**满块 |
| mamba | 仅当 chunk 末端恰好块对齐时,注册**最后一块**;其余一律跳过——chunk 中途跨过的块里没有物化过状态,登记即发布零快照 |

SWA 注册的块随后可能被滑动 punch 出块表,但池索引里的条目不受影响(哈希完好、
ref 归 0 后仍可命中,直到被驱逐)。

**ReclaimExpired(滑动,差异最大)**:

| 组 | 回收什么 | 一个请求的稳态驻留 |
|---|---|---|
| full | 不回收 | ceil(n / block_size),随序列线性涨 |
| SWA W | punch 掉完全滑出窗口的块(skipped = n - W + 1) | 约 ceil((W + chunk) / block_size),不随序列涨 |
| mamba | 只留最后一块(skipped = n - 1,即 W=2 的滑动公式) | 1-2 块 |

被 punch 的块若已注册,回 free list 后仍是缓存条目;若未注册(比如 mamba 组的
中间块),回去就是纯空闲块。

**Free(全组一致)**:请求结束,整表 Release 收批,一次 `FreeBlocks` 逆序归还——
前缀链的尾块先进 free list、先被驱逐,留下更短更通用的前缀。

### 例子:三种注意力的前缀匹配

设 block_size = 4,一个 24 token 的 prompt 对应 6 块,池里已缓存块 0、1、2 和
块 5(内容哈希相同意义上)。三个组各自匹配:

* full attention:从左往右逐块查哈希,块 3 miss,停。命中 3 块,边界 12 token。
  full 的匹配截短一段仍有效,收敛时可以直接截。
* SWA,窗口 8 token:恢复计算需要边界前 ceil(7/4) = 2 块连续在缓存。从右往左扫,
  块 5 只有一块连续,不够;块 0-2 这段够,边界同样落在 12 token。命中段前面的
  缺口用 null 块占位。窗口匹配截短后可能不再满足"边界前连续",不能截,只能按
  新边界重匹配。

  需要的块数为什么是 ceil((W-1)/block_size):窗口含自己,恢复位置自身的 KV 由这次前向
  现算,必须在缓存里的只有前 W-1 个 token;块整块命中,边界又块对齐,W-1 个
  token 从块边缘往回铺,正好占 ceil((W-1)/block_size) 块。
* mamba(GDN):命中 = 从右往左找最近的快照块,形如 [null, null, 快照块]。上面
  公式取 W=2 恰好给出这两条:恢复需要 ceil(1/block_size) = 1 块(那份快照),滑动
  skipped = n-1(只留最后一块)——`MambaStateManager` 就是 `SwaManager(block_size, 2)`
  加一条注册规则。W=1 则两头皆空(恢复不需要任何块、最后一块也滑掉),是
  "无跨 token 依赖"的退化情形,不是 mamba。

多组边界不一致时,`SweepThenConverge` 取交集:先让可截短的组定上界,窗口组在
上界内匹配;窗口组把上界压得更低时,已匹配的窗口组按新上界重来,直到稳定。

### 各组的 block_size:可以不同,怎么做到的

各组的 block_size 可以不同,只要都是某个基数(base = 各组 block_size 的 GCD)的
整数倍。构造断言从"必须相等"改成了"必须整除 base"。这套机制天生对单组透明:
单组或全组相等时 base = block_size,下面的折叠退化成恒等,一切和从前一样。

分两个粒度:内容哈希按 base 切一次算,各组按自己的粒度取。具体:

* 内容哈希链按 base 粒度推进(`scheduler.cpp` 建 Request 时 `page_size =
  BaseBlockSize()`),整条序列只算一次 SHA,组号不进哈希;
* 组 g 的 block_size = m·base。它的第 j 个粗块 = base 序列上第 j·m … (j+1)·m−1
  这 m 个细页折一次(`FoldBaseHashes`,链式防重排),再包组号成 key。折叠只在
  匹配/注册时按组做,哈希链本身不动。`m = 1` 的组直接用细页原样,不折叠;
* 匹配逐组按自己的值折算(`matchTierWithKeys` 里 `bound_tokens /
  group_block_size`),边界以 token 计,`SweepThenConverge` 交出 token 边界,
  跨粒度通用;
* 准入里索引 base 哈希数组的除法用 base(`forward.cpp` 的
  `matchFlatPrefixAtAdmission`);`BlocksNeededFor` 早已扇到各组,按各自粒度数块。

约束是"整除 base",不是任意粒度:这样所有组的块边界都落在同一张 base 细网格上,
哈希只需一条细链、各组在细边界上取子集,省掉了重复 SHA。base(GCD)是折叠和
索引的粒度;coordinator 还持一个 lcm(各组 block_size 的 LCM),预留给将来需要
"取整到所有组共同块边界"的对齐点(chunk/reserved 记账),目前 flat 准入路径上
还没有这样的点,故暂无消费者。

mamba 组的 block_size 若与 base 不同,它的粗块只在自己的对齐边界成块
(`RegistersAlignedFinalPageOnly` + 折叠),命中粒度天然变粗——这与状态快照
本就只在块边界物化一致,不额外做细粒度前缀命中。

`BlockPool`/`BlockRef`/`BlockTable` 对粒度全盲(纯块号空间),引用、驱逐、
L2 传输一个字节不碰。Python 侧每块总账按块的归属组算(slab 段几何各表自理)。
适配新的混合模型时,各组声明自己的 block_size 即可,C++ 不用改。

## 2. Python 侧的显存组织

C++ 只管块号,字节都在 Python。装字节的东西叫 slab:服务启动时
一次申请出来的大张量,
之后整个服务生命周期不再分配、不再搬动,变的只有"哪些块在用"。slab 一词来自操作系统内核的
slab allocator : 预分配一大块,切成等大的格子,分配和释放都只是记账。这里
更简单,记账全在 C++ 的 `BlockPool` 里(free list 弹出块号就是分配),Python 的
slab 自己不含任何分配逻辑,就是一张任人按块号读写的表。

slab 当前只有两种形状(下式里的 num_blocks 是全局块数,即块号空间的大小):

* KV slab:`(num_blocks × block_size, heads, head_dim)`,一行是一个 token 的
  K(或 V),块 k 占从第 k×block_size 行起的连续 block_size 行。一层 K、V 各
  一张(GPT-OSS 里两层共用,见下)。
* 状态 slab:`(num_blocks + 1, *状态形状)`,一行是一个块的整份快照;第 0 行留给
  null 块,恒零。每个 GDN 层 conv、ssm 各一张。

一个块占多少字节,每张 slab 各有自己的答案(下文叫"每块字节数"):KV slab 里
是 block_size × heads × head_dim × 元素字节,随 block_size 线性;状态 slab 里
就是一份快照的大小,与 block_size 无关。下面 GPT-OSS、Qwen3.5、DSv4 三节的
分岔全在这一条上。

所有 slab 共用同一个块号空间:C++ 把块 k 分给谁,谁就同时拿到块 k 在每张 slab
里的那一段。寻址不要求各张 slab 的每块字节数相等,但不等有两个后果:

1. 两层想共用一张 slab(省显存),每块字节数必须相等;
2. 每个块号在每张 slab 里都占着一段,不管归谁用,所以一个块的真实开销是它在
   全部 slab 里所占字节之和。这笔总账里若混进一个巨大的常数项,每 token 的
   有效显存就被它稀释掉。

### GPT-OSS-20b: 每块字节数相同,所以共用

24 层,full 和滑窗各 12 层,每块字节数完全相等,于是 full 第 j 层和滑窗第 j 层
共用一对 KV slab:`k_buffer[full第j层]` 和 `k_buffer[滑窗第j层]` 是同一个
tensor 对象,没有偏移、没有切分。

共用的是表,不是数据——两层分租同一张表的不同段。哪一段能写由块号说了算,而
块号来自 C++ 池的分配,一个块号同一时刻只属于一个请求的一个组。full 组和滑窗组
各自拿块,拿到的号必然不同:

```
共用的一张 k_slab(num_blocks 个段)
段号:   0     1     2     3     4     5    ...
归属: null  full  full  full  滑窗  滑窗   (空闲)
写它的: —    full 第 j 层      滑窗第 j 层
```

full 第 j 层只会拿着 full 组的块号(1、2、3)来写,滑窗第 j 层只会拿着滑窗组的
块号(4、5)来写,落在不相交的段上,永远不冲突。

前提只有一个实质条件:两层每块字节数相等(段的几何一样,块 k 在谁看来都是同一
段字节)。`hybrid_slab_group_size` 开机验证,不满足就退回每层一对,只损失显存
效率不损失功能。

### Qwen3.5-35B-A3B :巨大的常数项,所以 pad block_size

Qwen3.5-35B-A3B 共 40 层:30 个 GDN 层 + 10 个 full 层。GDN 状态一份约 2.1 MB
(conv + ssm),与 token 数无关。如果 block_size 停在 64,每个 block 只覆盖
64 token 却要对应 30 层 × 2.1 MB ≈ 63 MB 的状态——每 token 均摊约 1 MB,10 GB
预算只装得下约 1 万 token。所以把 block_size pad 到 KV 的每块字节数不小于状态的
(64 → 1088, 见`registry.create_attn_components`),每 token 均摊降到 ~80 KB,
同样预算约 12 万 token。这是一笔经济账,不是硬约束.

抬完之后:

* **GDN 层没有 KV slab**:`k_buffer` 里状态层的槽位是 None,只有 10 个 full 层
  真正分配(`_create_buffers`)。误取状态层的 KV(`get_key_buffer`)直接
  ValueError;PD 传输在入口拒绝(状态层无逐层 KV 可传)。attention 后端只对
  full 层调 PagedAttention,GDN 层走 MambaAttnBackend 读状态 slab,所以消费方
  不需要层号重映射;
* 每个 GDN 层一对状态 slab(conv、ssm),形状见 §2 开头;
* 块数 = 显存预算 ÷ 每块总账,总账由 `plan_component_tensors` 按实际组件算:
  10 层 KV(各 1088×2048 ≈ 2.23 MB)+ 30 层 conv+ssm(各 ~2.1 MB)+ MTP draft
  预留 ≈ 86.7 MB/块(起服日志实测 block_bytes=86,671,360、70 张组件张量)。
  每个块号在每张 slab 里仍占一段(§2 事实 2),full 块载着状态段、状态块载着
  KV 段的跨组死重还在。各组可以带不同 block_size,但在共享池下这笔死重消不掉,
  要回收得靠等字节/分池的池模型(见下)。

**pad block_size 的两个已知代价:**

* flashinfer/trtllm 的 decode 内核要求每块 token 数为 2 的幂,1088 不满足,
  Qwen3.5 目前走 Triton 后端。治本是内核块与管理块分离:内核用一个 2 的幂的小块、
  管理仍用 1088。尚未实现。
* 命中/准入粒度变粗:前缀命中只能落在 1088 的倍数上。待优化

## 3. 以后的模型怎么适配

### DSv4:组件多、字节数杂
各 group 不同的 block size + slab 分桶, cpp 侧极少改动. 

### 假设: 四种层混合:SWA-4 + SWA-128 + mamba + full

C++ 侧不用改,四种层落到已有三个 kind:

| 层 | KvCacheSpec | manager |
|---|---|---|
| full | {kFull, block_size, 0} | FullAttnManager |
| SWA 4 | {kSlidingWindow, block_size, 4} | SwaManager(block_size, 4) |
| SWA 128 | {kSlidingWindow, block_size, 128} | SwaManager(block_size, 128) |
| mamba | {kMambaState, block_size, 0} | MambaStateManager(block_size) |

不同窗口互相压边界、级联重匹配,`SweepThenConverge` 的测试矩阵已覆盖;准入、
滑窗信用、host 匹配都按组循环,不关心组数和种类。

Python 侧每个新模型写两处:层标签到组的映射(`paged_cache_spec.py`),和每层的
组件字节声明(`components_from_layers`:KV 字节随 block_size 线性,状态是常数)。
`solve_page_geometry` 自动裁决:有常数项就抬 block_size,全是线性项就维持原值,
想共享的等字节 slab 按 GPT-OSS 方式别名,状态 slab 单列。

什么时候才需要动 C++:某种层的复用规则没法表达成"从右往左找可恢复边界 + 保留
最后 W 个 token"。真遇到了,加一个 manager 子类和一个 kind 枚举值;块号、token
边界、收敛骨架、引用规则都不用碰。各组可以带不同 block_size,所以异构粒度的新
模型也不用动 C++。尚未实现的方向:等字节/分池的池模型(回收跨组死重)、内核块
与管理块分离、块粒度状态快照(需要内核吐分块中间态)。
