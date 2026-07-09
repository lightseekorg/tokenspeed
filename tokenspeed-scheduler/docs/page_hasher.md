# page_hasher.h — 前缀缓存分页哈希

`csrc/scheduler/page_hasher.h` 实现 tokenspeed 前缀缓存(prefix cache)的分页哈希。
目标是与 vLLM V1 的 block 哈希**功能对齐**(不要求数值/字节一致)。本文档记录算法、
设计取舍,以及与 vLLM 的对应关系。

对应的 vLLM 实现:`vllm/v1/core/kv_cache_utils.py`(`hash_block_tokens`、
`generate_block_hash_extra_keys`、`BlockHashWithGroupId`)。

---

## 1. 总览

一个请求的 token 序列按固定大小切成若干 **page(块)**。每个 page 算一个哈希,用作
前缀缓存的查找键。核心性质是**链式(chain hashing)**:每页的哈希都折入前一页的哈希,
所以同样的 token 出现在不同前缀之后会得到不同的 page 哈希,缓存不会错配。

哈希输出是 **64 字符的小写 hex 字符串**(而非裸 32 字节)。这是 tokenspeed 的刻意选择,
方便跨 nanobind / Python / 序列化边界传递;代价是链式时下一页要先 `HexToBytes` 解回字节
再喂入,比 vLLM 的裸 bytes 多一次编解码,但功能等价。

底层摘要算法是 OpenSSL SHA-256(`SHA256_Init/Update/Final`)。OpenSSL 3.0 将这组接口标记
为 deprecated;通过 CMake 的 `OPENSSL_SUPPRESS_DEPRECATED` 编译定义压掉警告,行为不变
(未迁移到 `EVP_*`,因为无性能/功能收益)。

---

## 2. 单页哈希:`HashPage`（page_hasher.h:91）

```cpp
std::string HashPage(std::span<const std::int32_t> tokens,
                     const std::string& prior_hash,
                     std::span<const std::string> extra_keys = {});
```

整条输入做成**全前缀分帧(fully length-prefixed framing)**,喂入 SHA 流的字节布局是:

```
[prior_len][prior][token_count][tokens...][extra_count][extra_key_len][extra_key]...
```

逻辑顺序仍是 **`prior_hash → tokens → extra_keys`**,但每一段都自带定界信息,任意两个不同的
`(prior, tokens, extra_keys)` 三元组都不会喂出同一条字节流(见 §2「为什么要全前缀分帧」)。

### (1) 折入前一页哈希(链式)— page_hasher.h:96
`prior_hash`(上一页的 64-hex 输出)先 `HexToBytes` 解回字节,**先写 4 字节 LE 的 `prior_len`**
(字节数,空 prior 写 0),再写 prior 内容本身。这是前缀链的本质。第一页 `prior_hash` 为空,
此时只写 `prior_len = 0`、不写内容(**不是整段跳过**)。

### (2) 喂入 token — page_hasher.h:102
**先写 4 字节 LE 的 `token_count`**,再逐个把 token id 编成**固定 4 字节小端(LE)**。token 定长,
彼此之间无需分隔;count 前缀把整个 token 块和后面的 extra_keys 块隔开。

### (3) 喂入 extra_keys(可选,带 framing）— page_hasher.h:107
`extra_keys` 是一个 **key 列表**,framing 规则:
- 先写 4 字节 LE 的 **key 数量**(count-prefix);
- 再对每个 key 写 4 字节 LE 的**长度前缀**,然后写 key 内容。

extra_keys 是**最后一段**,所以为空时可以**整段跳过**而不产生歧义(非空列表必先写一个 ≥1 的
count;空着什么都不写,二者天然区分)。

### (4) 收尾 — page_hasher.h:117
`SHA256_Final` 出 32 字节摘要,`DigestToHex` 编码成 64-hex 返回。

### 为什么要全前缀分帧
SHA 的 `Update` 只追加字节,块与块之间的边界不进哈希。若不显式编码边界,三处会糊在一起产生
**结构性碰撞**(都是合法输入可触发,只是概率极低):

1. **空 prior vs 链中页**:若 page 0(无 prior)的前 8 个 token 恰好 LE-编码成某 32 字节 digest,
   它与「该 digest 作 prior + 无这些 token」的某条链中页字节流相同。靠 `prior_len` 区分(0 vs 32)。
2. **token 块漏进 extra_keys**:token 块若无 count,末尾的 token 可与「`count / len / key`」的字节
   重合,使「N token + 1 key」与「(N+k) token + 0 key」碰撞。靠 `token_count` 区分。
3. **key 之间被重新切分**:`{"ab","c"}` 与 `{"a","bc"}`、或单 key 与多 key,靠 count + 每 key 长度
   前缀区分。

> **注意**:全前缀分帧改变了 digest 的字节流,因此与本次改动**之前**产出的旧 page hash **不再字节
> 兼容**(纯内存缓存重启即失效,无碍;若有落盘/持久化的旧哈希则需失效重算)。空输入
> (`HashPage({}, "")`)现在喂的是 `prior_len=0 ++ token_count=0`(8 个零字节),其 SHA-256 是
> `af5570f5…e83dfc`,单测里钉住了这个向量。

---

## 3. 多页链式:`ComputePagedHashes`（page_hasher.h:122）

把 `HashPage` 在多页上滚一遍:`current_prior` 初值是入参 `prior`,每页算完把输出回填成
下一页的 `prior`,串成前缀链。`extra_keys_per_page[i]` 给第 i 页,缺省为空。

---

## 4. extra_keys:区分键

`extra_keys` 用于区分「token 相同但语义不同」的页。对应 vLLM 的四种来源
(`generate_block_hash_extra_keys`,kv_cache_utils.py:525,固定顺序拼接):

| 顺序 | 来源 | 内容 | 触发条件 |
|---|---|---|---|
| 1 | LoRA | `lora_request.lora_name`(字符串,是 name 不是 id) | 请求带 LoRA,每页都加 |
| 2 | 多模态 | `(mm_identifier, offset - start_token_idx)` 二元组 | 页落在某 MM 输入范围内 |
| 3 | cache_salt | `request.cache_salt` 字符串 | **仅第一页**(start_token_idx==0)且非空 |
| 4 | prompt_embeds | 该页 embedding 切片的 SHA-256 digest | 请求用 prompt embeds 而非 token id |

### 责任边界（与 vLLM 的关键差异）
vLLM 把结构化的 extra_keys 整个塞进序列化器(pickle/cbor),分界由 tuple 结构免费得到。
C++ 这边把职责切成两层:

| 层 | 谁负责 |
|---|---|
| 选哪些 key 进哪页(salt 只进 page 0、MM 的 offset 窗口、LoRA 每页带) | **调用方**——依赖整个 Request 元数据,搬进 kernel 会破坏 vendor-neutral |
| 把单个值编码成 key(如 `(identifier, offset)` 编进一个 string) | **调用方** |
| 多 key 之间、key 与 token 之间的无歧义分界(framing) | **C++**(`HashPage` 内的 count + 长度前缀) |

> cache_salt「只进第一页」这个不对称,在本接口里就是:`extra_keys_per_page` 只给 index 0
> 填 salt、其余页留空。salt 锚在前缀链头上,靠链式往后传染,后续页不必重复带。

---

## 5. group_id:混合 KV cache 的分组标签

对应 vLLM 的 `BlockHashWithGroupId`。**group_id 不进 SHA 流** —— 它不是页内容的一部分,
而是「这段内容属于哪个 KV cache group」的标签。算完内容哈希后,把 group_id **缀在 hex 串尾巴**
上构成查找键。

### `MakeKeyWithGroupId`（page_hasher.h:146）
group_id 编成 **4 字节大端(BE)**,追加为 8 个 hex 字符:**64-hex 内容哈希 → 72-hex 键**。

> 注意字节序在本文件里是混用的,且都是故意的:**token / extra_keys 长度前缀用小端 LE**
> (只自己读,内部自洽即可);**group_id 用大端 BE**,为了对齐 vLLM 的
> `group_id.to_bytes(4, "big")`。

### 加 group_id 前后的哈希形态
```
之前(内容哈希,64 hex):   a3f1...e9c2
                          └──── 64 hex ────┘
之后(查找键,72 hex):     a3f1...e9c2 00000007
                          └─ 64 hex ─┘└ 8 hex ┘
                            内容哈希    group_id=7,大端
```
- **前 64 位 hex 一字不差**还是原内容哈希;只在尾部多缀 8 hex 的 group_id。
- 同内容、不同 group → 只有尾 8 位不同(group 0 → `00000000`,group 1 → `00000001`)。
  因此**SHA 只算一次**,多个 group 复用同一份内容哈希。

### 反解与隔离
- `GetBlockHashFromKey`(page_hasher.h:159):砍掉尾 8 hex,取回内容哈希。
- `GetGroupIdFromHashKey`(page_hasher.h:167):取尾 8 hex,大端还原成 uint32。
- `ComputePagedHashesWithGroup`(page_hasher.h:181):先 `ComputePagedHashes` 算裸内容哈希
  链,**再逐页** `MakeKeyWithGroupId` wrap。因此 **group_id 不渗进前缀链** —— 链上滚的始终
  是裸内容哈希,不同 group 的同一前缀每步都匹配,缓存前缀匹配照常工作。

> tokenspeed 内部的 group_id 实际是 `std::string`;此处按 **uint32** 处理(用户选定),
> 是功能对齐所需的最小表示,不追求与内部类型一一对应。

---

## 6. 与 vLLM 的对应小结

| 维度 | vLLM | tokenspeed (page_hasher.h) | 是否数值一致 |
|---|---|---|---|
| 单页哈希 | `hash_block_tokens((parent, tokens, extra_keys))` | `HashPage(tokens, prior, extra_keys)` | 否(功能对齐) |
| 喂入顺序 | (parent, tokens, extra_keys) | prior → tokens → extra_keys | 一致 |
| 输出表示 | 裸 32 字节 `BlockHash` | 64-hex 字符串 | 否(刻意) |
| 输入分界 | 序列化器从 tuple 结构得到 | 全前缀分帧(prior_len + token_count + extra count/长度前缀) | 否(等价抗碰撞) |
| 空 extra_keys | 仍哈希一个 `None` 槽 | 末段整段跳过(由 token_count 保证无歧义) | 否 |
| group_id | `BlockHash` 后接 4 字节大端 | 64-hex 后接 8-hex 大端 | 结构同构 |

**结论**:按「只要功能对齐、不要求数值对齐」的标准,两者等价。唯一被显式转移给调用方的责任,
是「每页放哪些 key / 单个值如何编码」;块与块之间的所有分界(prior / token / 各 extra_key)都已由
C++ 的全前缀分帧承担,任意合法输入都无结构性碰撞。
