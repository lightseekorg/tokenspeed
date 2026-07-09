# Flat KV-Cache

本目录是 flat KV-cache 的 C++ 实现:按固定大小的块管理 KV 显存,用哈希做前缀复用,
思路和 vLLM V1 的 BlockPool 一路相同。下面先钉术语,再按三个问题展开:C++ 各个类
怎么分工、Python 显存怎么摆、更复杂的混合模型要改哪里。

## 0. 术语

* **块(block)**:分配、缓存、传输的最小单位。中文文档里也叫"页",同一个东西。
* **block_size**:一个块覆盖的 token 数,也叫每块 slot 数。这是**记账粒度**,不是
  内容承诺——attention 块装这些 token 各自的 KV,mamba 块装一份"截至本块边界"的
  状态快照,里面没有 slot 阵列。配置入口是 `server_args.block_size`,C++ 侧同名;
  Python 池属性 `page_size` 是历史遗留的同义词,radix 删除时统一。
* **行表 / 行宽**:一层的一个组件对应一张行表——shape 为 `(num_blocks, ...)` 的
  连续显存,块号 k 就是第 k 行;**行宽 = 一行的字节数,是张量的属性**。一层可以有
  多个组件(多张行表),层的每块开销 = 它全部组件行宽之和。
* **num_blocks**:全局唯一的块数,即块号空间的大小。

## 1. C++ 侧的组件和几条关键路径

### 块和池

`CacheBlock` 是块的元数据:引用计数 + 内容哈希 + 所属池。`BlockPool` 管一个存储
层级的所有块:一条 LRU free list,加一个哈希到块的索引。引用计数掉到 0 的块不清
空,带着哈希回到 free list 尾部,之后两种下场:被同前缀的请求命中,从 free list
中间摘走再次使用;或者走到 LRU 头被新分配拿走,这时才抹掉哈希。所以"缓存"和
"空闲"不是两个集合,是同一条 free list。

device 显存一个 `BlockPool`,host 内存一个 `BlockPool`,同一个类,没有子类。
块 0 是保留的 null 块,只用来在块表里占位,永远不参与分配。

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
* **谁在持有**:块表槽位(请求活着期间)、load/store ticket(传输在飞期间)、
  store 候选(注册到下沉确认之间)。三处都是 BlockRef,不存在裸 unpin。
* **引用与哈希的组合就是块的状态机**:ref>0 且无哈希 = 正在写入;ref>0 且有哈希 =
  在用且可命中;ref=0 且有哈希 = 缓存中、可驱逐;null 块不参与计数。
* 匹配(查询)不产生引用,函数签名收 `const BlockPool&`,想改也改不了。跨池误用
  由块上的 owner 断言当场击落。

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
`Adopt` 进表。统一 block_size 下各组取的块数相同。跨组 all-or-nothing:先按总需求
查询 free list 余量,任何一组不够则整步失败,已有状态不动。**mamba 组也照常按
token 数取块**——它的块表和别人一样逐块增长,差异全在后面的 Reclaim。

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

  需要的块数为什么是 ceil((W-1)/P):窗口含自己,恢复位置自身的 KV 由这次前向
  现算,必须在缓存里的只有前 W-1 个 token;块整块命中,边界又块对齐,W-1 个
  token 从块边缘往回铺,正好占 ceil((W-1)/P) 块。vLLM 同一公式
  (single_type_kv_cache_manager.py 的 cdiv(window_size - 1, block_size))。
* mamba(GDN):命中 = 从右往左找最近的快照块,形如 [null, null, 快照块]。上面
  公式取 W=2 恰好给出这两条:恢复需要 ceil(1/P) = 1 块(那份快照),滑动
  skipped = n-1(只留最后一块)——`MambaStateManager` 就是 `SwaManager(P, 2)`
  加一条注册规则。W=1 则两头皆空(恢复不需要任何块、最后一块也滑掉),是
  "无跨 token 依赖"的退化情形,不是 mamba。

多组边界不一致时,`SweepThenConverge` 取交集:先让可截短的组定上界,窗口组在
上界内匹配;窗口组把上界压得更低时,已匹配的窗口组按新上界重来,直到稳定。
边界以 token 计,为将来各组 block_size 不同预留。

### 例子:host 层(L2)

`MatchPrefix` 一次查两层:先 device,再以 device 边界为起点查 host,host 段允许
更短但必须接在 device 段上(窗口连续性由两层拼起来满足)。装载时给每个 host
命中块配一个新分配的 device 块,H2D 期间两侧的块都由 BlockRef 引用住,拷完按序
归还。下沉方向是每轮调度末尾把新注册的块批成一个 D2H 操作,host 块在拷贝完成的
回执里才登记进 host 池索引:查到即字节已就位。

## 2. Python 侧的显存组织

C++ 只管块号,字节在 Python。核心是**块号全局一份,行宽张量自理**:每张行表拿
块号 k 找自己的第 k 行,不同行表的行宽互不相干,寻址不需要它们相等。

但"不必相同"不等于"不同没有代价",两个事实:

1. **等宽才能别名**:两层想共用同一张物理行表(省显存),行宽必须相等;
2. **每个块号在每张行表里都占一行**,不管当前归谁用。每块总账 = Σ 全部行表行宽,
   一个巨大的常数行会稀释每 token 的显存效率。

### GPT-OSS:等宽 → 别名

48 层,full 和滑窗各 24 层,行宽完全相等 → full 第 j 层和滑窗第 j 层共用一张
行表(块号同一时刻只属于一个请求的一个组,不会写同一行)。48 层只要 24 张表,
同样预算块数翻倍。`hybrid_slab_group_size` 判定,条件不满足退回每层一张。

### Qwen3.5:常数大行 → 抬 block_size(经济性)

GDN 状态一份约 2.1 MB(conv + ssm),与 token 数无关。如果 block_size 停在 64,
每块只覆盖 64 token 却要养 36 层 × 2.1 MB ≈ 75 MB 的状态行——10 GB 预算只装得下
约 7800 token,显存全花在状态上。所以把 block_size 抬到 KV 行字节不小于状态行
(64 → 1088,`registry.create_attn_components` 里改,有日志),状态开销占比压到
一半以下,同样预算约 6 万 token。这是**经济性**决策;vLLM 对 Qwen3-Next 的膨胀
则是结构性的(单一共享缓冲要求块字节硬性相等)。抬完之后:

* KV 行表照旧覆盖全部 48 层(GDN 层的 KV 行暂空置,收敛进 plan 执行器是后手);
* 每个 GDN 层两张状态行表(conv、ssm),第 k 行 = 块 k 的快照,行数 num_blocks+1,
  第 0 行对应 null 块恒零;
* 块数 = 显存预算 ÷ 每块总账。

GDN 前向每步读上一个位置的状态、写当前位置的状态,读写行号都从状态组的块表查:
同一块内原地更新;跨块那步读旧块写新块,旧块里留下的正好是边界快照,不需要额外
拷贝;命中恢复第一步从快照块(共享只读)读、写自己的新块。行号和 KV 块表同源,
CUDA graph 无需特殊处理。

已知现状:flashinfer/trtllm 的 decode 内核要求每块 token 数为 2 的幂,1088 不
满足,Qwen3.5 目前走 Triton 后端;治本是 kernel 块与管理块分离(vLLM 的
kernel_block_size 方案),已立后手。

host 镜像(`FlatHostMirror`)给每张行表配 pinned 内存,块 k 的 host 副本 =
各行表第 k 行拼起来;搬运按行拷、不认识字节内容,状态块的下沉装载零专门代码。

## 3. 以后的模型怎么适配

### DSv4:组件多、行宽杂 → 多行表,按需 pad

DSv4 一层就有行宽差近 10 倍的组件:MLA latent(每 slot 1152 B)+ indexer 量化
key(每 slot ~132 B),部分层另有压缩状态。做法:每组件一张行表,块号贯穿全部
(host 传输"每块一行跨全部行表"的性质正好用上);全部主要组件随 P 线性,
**block_size 不用动**;pad 只在想跨层共享同一张行表时才需要(vLLM 给 DSv4 的
专用路径同样只 pad 不膨胀)。压缩状态若语义同 mamba(每块边界一份),匹配侧映射
现成的 `kMambaState`。C++ 零或近零新增。

### 四种层混合:SWA-4 + SWA-128 + mamba + full

C++ 侧不用改,四种层落到已有三个 kind:

| 层 | KvCacheSpec | manager |
|---|---|---|
| full | {kFull, P, 0} | FullAttnManager |
| SWA 4 | {kSlidingWindow, P, 4} | SwaManager(P, 4) |
| SWA 128 | {kSlidingWindow, P, 128} | SwaManager(P, 128) |
| mamba | {kMambaState, P, 0} | MambaStateManager(P) |

不同窗口互相压边界、级联重匹配,`SweepThenConverge` 的测试矩阵已覆盖;准入、
滑窗信用、host 匹配都按组循环,不关心组数和种类。

Python 侧每个新模型写两处:层标签到组的映射(`paged_cache_spec.py`),和每层的
组件字节声明(`components_from_layers`:KV 行随 block_size 线性,状态行是常数)。
`solve_page_geometry` 自动裁决:有常数行就抬 block_size,只有线性行就维持原值,
想共享的等宽行表按 GPT-OSS 方式别名,状态行表单列。

什么时候才需要动 C++:某种层的复用规则没法表达成"从右往左找可恢复边界 + 保留
最后 W 个 token"。真遇到了,加一个 manager 子类和一个 kind 枚举值;块号、token
边界、收敛骨架、引用规则都不用碰。已知后手:状态组独立 block_size(匹配代码按
token 算边界就是为它留的)、kernel 块与管理块分离、块粒度状态快照(需要 kernel
吐分块中间态)。
