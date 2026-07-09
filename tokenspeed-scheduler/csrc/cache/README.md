# Flat KV-Cache

本目录是 flat KV-cache 的 C++ 实现:按固定大小的页管理 KV 显存,用哈希做前缀复用,
思路和 vLLM V1 的 BlockPool 一路相同。下面按三个问题展开:C++ 这边各个类怎么分工、
Python 那边显存怎么摆、以后来了更复杂的混合模型要改哪里。

## 1. C++ 侧的组件和几条关键路径

### 页和池

一页(`CacheBlock`)是分配、缓存、传输的最小单位,对应序列里连续 `page_size` 个
token 的覆盖范围。页里装什么由层类型决定:attention 层装这些 token 各自的 KV,
mamba 层装一份"截至本页边界"的状态快照(见下文)。`page_size` 的配置入口是
`server_args.block_size`,同一个量在 C++ 侧叫 page_size。
`BlockPool` 管一层存储的所有页:一条 LRU free list,加一个 hash 到页的索引。
引用计数掉到 0 的页不清空,带着 hash 回到 free list 尾部。之后它有两种下场:
被同前缀的请求命中,从 free list 中间摘走再次使用;或者走到 LRU 头,被新的分配
拿走,这时才抹掉 hash。这就是"缓存"和"空闲"不是两个集合,而是同一个 free list
的原因。

device 显存一个 `BlockPool`,host 内存一个 `BlockPool`,同一个类,没有子类。
页 0 是保留的 null 页,用来在页表里占位,永远不会被分配。

### 引用

拿页放页只有一条通道:`BlockRef`。它是个 move-only 的 RAII 句柄,三个操作:
`Share` 给一个已缓存的页加一个引用,`Adopt` 接管一个刚分配的页(分配时已经计了 1),
析构时自动归还。页表的槽位存的是 BlockRef,in-flight 传输的 ticket 里存的也是
BlockRef,所以不存在忘记归还的路径。查询(匹配)不产生引用,函数签名收
`const BlockPool&`,想改也改不了。

### 页表和策略

`BlockTable` 是每个请求每个组一张的页表,逻辑页号到物理页的映射。滑动窗口把
过期页换成 null 页占位(`EvictToNull`),这样后面页的槽位号不变。

匹配和滑动的规则因注意力类型而异,写在三个 manager 里:`FullAttnManager`、
`SwaManager`、`MambaStateManager`。manager 不持有池,所有方法把池当参数收,
所以同一份匹配代码既能查 device 池也能查 host 池。`KvCacheCoordinator` 在构造时
把 manager 和池绑起来(device 必选,host 可选),对外提供 `MatchPrefix` 一个入口。

### 例子:分配与释放

一个请求要写 3 页。`Acquire` 先看页表尾页还有没有空 slot,不够的部分调
`AllocateBlocks(n)`:从 free list 头上弹 n 页,弹到带 hash 的页就把它从 hash
索引里删掉(驱逐),然后 `Adopt` 进页表。任何一个组不够页,整个请求这一步失败,
已有状态不动,不需要回滚。

请求结束时 `Free` 把页表里所有 BlockRef 收成一批,一次 `FreeBlocks` 逆序归还。
逆序是有意的:一条前缀链的尾页先进 free list,先被驱逐,留下的是更短、更通用的
前缀。

### 例子:三种注意力的前缀匹配

设 page_size = 4,一个 24 token 的 prompt 对应 6 页,池里已经缓存了页 0、1、2
和页 5(指内容 hash 相同的页)。三个组各自匹配:

* full attention:从左往右逐页查 hash,页 3 miss,停。命中 3 页,边界在 12 token。
  full 的匹配结果截短一段仍然有效,后面收敛时可以直接截。
* SWA,窗口 8 token:恢复计算需要边界前 ceil(7/4) = 2 页连续在缓存。从右往左扫,
  页 5 只有一页连续,不够;页 0-2 这段够,边界同样落在 12 token。如果命中段前面
  有缺页,用 null 页占位,槽位号不乱。窗口匹配截短之后可能不再满足"边界前连续",
  所以不能截,只能按新边界重新匹配。

  需要的页数为什么是 ceil((W-1)/P):窗口含自己,恢复位置自身的 KV 由这次前向现算,
  必须在缓存里的只有前 W-1 个 token;页整页命中,边界又页对齐,W-1 个 token 从页
  边缘往回铺,正好占 ceil((W-1)/P) 页。vLLM 是同一个公式
  (single_type_kv_cache_manager.py 的 cdiv(window_size - 1, block_size)),
  W=2 时它等于 1,这就是下面 mamba 的"一份快照"。
* mamba(GDN):一页存一份该页边界时刻的状态快照,不是 4 个 token 的 KV。命中
  就是从右往左找最近的一份快照,结果形如 [null, null, 快照页]。滑动规则是只留
  最后一页(等价于窗口为 2 的 SWA,`MambaStateManager` 就是这么实现的)。注册时
  只登记"某次前向恰好写到页边界"的那一页,中途跨过去的页里没有物化过状态,
  登记了就是把零快照当真数据发布。

多个组的边界不一样时,`SweepThenConverge` 取交集:先让 full 这类可截短的组定一个
上界,窗口组在上界内匹配;窗口组把上界压得更低时,已经匹配过的窗口组按新上界重来,
直到稳定。边界以 token 计,因为各组的 page_size 将来可能不同。

### 例子:host 层(L2)

`MatchPrefix` 一次查两层:先 device,再以 device 边界为起点查 host,host 段允许
更短但必须接在 device 段上(窗口的连续性由两层拼起来满足)。装载时给每个 host
命中页配一个新分配的 device 页,H2D 拷贝期间两边的页都用 BlockRef 引用住,
拷完按序归还。写方向(下沉)是每轮调度末尾把新注册的页批成一个 D2H 操作,
host 页在拷贝完成的回执里才登记进 host 池的 hash 索引:查到即字节已就位。

## 2. Python 侧的显存组织

C++ 只管页号,字节在 Python。三个量:

* `block_size`:一页几个 token,即上文 page_size,全部组统一;
* 每层每页的字节数:随层类型不同;
* `num_pages`:全局一份的页数,也就是页号空间的大小。

显存按"slab"分配。一张 slab 是一块连续显存,shape 是 (num_pages × page_size,
head 数, head 维),页 k 占第 k×page_size 到 (k+1)×page_size 行。页号是全局的,
每层拿页号乘上自己的行宽去找字节,所以不同层的每页字节数不必相同。

### GPT-OSS

48 层,full 和滑窗各 24 层,两种层的每页字节完全一样。于是 full 的第 j 层和滑窗的
第 j 层共用同一张 slab:调度器保证一个页号同一时刻只属于一个请求的一个组,两层
不会写同一行。48 层只需要 24 张 slab,同样的显存预算页数翻倍。这个共享由
`hybrid_slab_group_size` 判定,条件不满足(层数不等、窗口不一)就退回每层一张。

### Qwen3.5

GDN 层的状态(conv + ssm)一份约 2.1 MB,和 token 数无关,塞不进 64 token 的
KV 页。做法和 vLLM 对 Qwen3-Next 一样:把 page_size 抬到一个 KV 页的字节不小于
一份状态,64 抬到 1088(`registry.create_attn_components` 里改,有日志)。抬完之后:

* KV slab 照旧,覆盖全部 48 层(GDN 层的 KV 行暂时空着,以后收);
* 每个 GDN 层另配两张状态张量(conv、ssm),第 k 行就是页 k 的快照,行数
  num_pages + 1,第 0 行对应 null 页,恒零;
* 页数 = 显存预算 ÷ 每页总字节(48 层 KV 行加全部状态行)。

GDN 前向每一步读上一个位置的状态、写当前位置的状态。读哪一行、写哪一行都从
状态组的页表里查:位置还在同一页,读写同一行,原地更新;跨页那一步读旧页写新页,
旧页里留下的正好是页边界时刻的快照,不需要额外的拷贝。命中恢复也是同一个动作:
第一步从命中的快照页(共享,只读)读,写到自己的新页。读写行号和 KV 的页表走
同一条路,CUDA graph 里不需要特殊处理。

host 侧的镜像(`FlatHostMirror`)给每张 slab 和每张状态张量配一块 pinned 内存,
页 k 的 host 副本就是各张量第 k 行拼起来。搬运按行拷,不认识字节内容,所以状态页
的下沉和装载不需要专门代码。

## 3. 以后的模型怎么适配

假设一个模型混着四种层:窗口 4 的 SWA、窗口 128 的 SWA、mamba、full。

C++ 侧不用改。四种层落到已有的三个 kind 上:

| 层 | KvCacheSpec | manager |
|---|---|---|
| full | {kFull, P, 0} | FullAttnManager |
| SWA 4 | {kSlidingWindow, P, 4} | SwaManager(P, 4) |
| SWA 128 | {kSlidingWindow, P, 128} | SwaManager(P, 128) |
| mamba | {kMambaState, P, 0} | MambaStateManager(P) |

两个不同窗口的组互相压边界、级联重匹配的情况,`SweepThenConverge` 的测试里已经
有(窗口 6 和窗口 2 混跑的用例)。准入、滑窗信用、host 匹配都是按组循环,不关心
组的数量和种类。

Python 侧每个新模型要写两处:层类型标签到组的映射(`paged_cache_spec.py`,
SWA 标签带各自的窗口值),和每层的字节声明(`components_from_layers`:KV 行是
随 page_size 线性的,状态行是常数)。page_size 的计算会自动处理:有常数行就抬
page_size,只有线性行就把窄的 pad 到宽的。四种层里 SWA 4、SWA 128 和 full 的
KV 行同宽,照 GPT-OSS 的方式共享 slab;mamba 状态张量单列。

什么时候才需要动 C++:某种层的复用规则没法表达成"从右往左找一个可恢复边界 +
保留最后 W 个 token"。真遇到了,加一个 manager 子类和一个 kind 枚举值,页号、
token 边界、收敛这些都不用碰。已知但没做的:状态组用更大的独立 page_size
(匹配代码按 token 算边界就是为这个留的)、页粒度的状态快照(需要 kernel 吐出
分块中间态)。
