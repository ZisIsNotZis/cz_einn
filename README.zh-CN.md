<p align="center">
  <img src="assets/logo.svg" alt="EINN — 爱因斯坦记号神经网络" width="820">
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<h1 align="center">EINN</h1>

<p align="center"><strong>把神经网络写成一行爱因斯坦求和式，再交给 PyTorch 搭出来。</strong></p>

<p align="center">2020 年 fastai 中文语言模型研究存档。</p>

> [!IMPORTANT]
> **这是冻结的研究存档，不是仍在维护的软件。** 所有 notebook 的输出都是空的；仓库里没有实验结果、模型权重、语料、测试，也没有训练脚本。它被公开，只是为了让这条思路留下记录——它不是基线，不是库，也不是任何承诺。

EINN 是一次单人实验：用 `#1mek->m?k` 这样的短文本描述网络结构，把它解析成真正的层，维度的对应关系交给 `torch.einsum`。存档沿着这个想法一路走完：从合成数据上的最小验证，到 CLUE 中文分类任务，再到字符级金字塔自编码器。

**第一次来？** 先打开 [`einn.ipynb`](einn.ipynb)，它是整个想法最完整也最短的表述。完整的事实、证据和边界在 [`docs/project-status.md`](docs/project-status.md)。

---

## 快速开始

**只想读——不用安装。** 在 GitHub 上打开 [`einn.ipynb`](einn.ipynb)。第二个代码单元用九行 spec 说清了全部思路。

**运行唯一一个不依赖语料的 notebook**（合成随机 token，假定有 CUDA）：

```bash
git clone https://github.com/ZisIsNotZis/cz_einn.git
cd cz_einn
python -m venv .venv && . .venv/bin/activate
pip install "fastai>=2.7" "torch>=2.0" numpy jupyter
jupyter lab einn.ipynb
```

`einn5*` 语言模型 notebook 需要外部语料，见[如何开发](#如何开发)。除 `einn.ipynb` 外，这里没有任何东西可以直接打包、固定依赖或开箱运行。

---

## 现在能做什么

- **用字符串描述网络。** `net(dict)` 把单字符符号映射到层工厂，再从多行 spec 构建 fastai `Sequential`。
- **一个可用的 einsum 层。** 默认的 `Dot` 层持有 `ParameterList`，用 `torch.einsum` 执行 `A->B` 维度标注。
- **可复用的构件。** `MGLinear`（群等变多头线性层）、`MLinear`、`Red`（对数深度的成对序列归约）、`Bn`、`Get`、`Call`、`Add`、`View`、`Rec`。
- **任务 notebook。** CLUE 格式的中文分类实验：`iflytek`、`tnews`、`cmnli`、`afqmc`、`cluewsc2020`。
- **金字塔自编码器。** `PyramidAutoEncoder` 与 `PyramidCrossEntropyLoss`，对 U 形字符模型的每一层都施加监督。
- **语料构建配方。** 字符 ID 流的生成代码原样保存在 notebook 里，并在[下文](#语料不在仓库里)复述。

一切从 [`einn.ipynb`](einn.ipynb) 里的这段 spec 开始：

```python
m = """?
#1mek->m?k
->m?K
@MGLinear(m,g,k,k,0)->m?kG
~mk
@Swish()
@lambda x:x.sum(3)->m?k
@MLinear(m,k,e,1)
"""
m = learn(m, m=1024, e=16, k=4, K=16, g=4, l=4)
m.fit_one_cycle(20, 0.001, 9, 99, 0.01)
```

### Notebook 序列

| 文件 | 内容 |
|---|---|
| [`einn.ipynb`](einn.ipynb) | **正典。** einsum 字符串 DSL、完整算子注册表、`MGLinear` / `MLinear` / `Red`。在合成随机 token 上训练——这是机器的单元测试，不是任务。 |
| `einn2*.ipynb` | 同一套结构改写成函数式生成器（`F`、`Red`、`MGLinear`、`MBatchNorm`、`MLinear`）；仍是合成数据。 |
| `einn3*.ipynb`、`einn4*.ipynb` | 转向真实的 CLUE JSON 中文分类：`iflytek`、`tnews`、`cmnli`、`afqmc`、`cluewsc2020`。用 `Red` 处理变长句子。 |
| `einn5-autoencoder*.ipynb`（22 个） | 在 memmap 中文新闻语料上做字符级语言模型 / 自编码。终点是 `PyramidAutoEncoder`；`einn5-autoencoder-Copy20.ipynb` 是走得最远的状态。 |
| `einn5-autoencoder-Copy17.ipynb` | 一条无关分支，引入了预训练 `resnet26d`（仓库中没有）。 |

带 `Copy` 编号的文件是刻意的：它们是工作过程，包括走不通的路。公开整理时删掉了一个完全重复的文件，剩下的 45 个 notebook 与私有存档逐字节一致。

---

## 架构

```
spec 文本
   │
   ▼
net(dict)  ── 单字符符号 ──▶  层工厂
   │
   ▼
fastai Sequential  ── Dot: torch.einsum(A,B->C) ──▶  张量
   │
   ▼
原版 fastai Learner / DataLoaders / fit_one_cycle
```

这个仓库提供的是层和数据迭代器，不提供训练机制：每个 notebook 都用 fastai 的 `Learner`、`fit_one_cycle`、`lr_find`、`ShowGraphCallback` 和 `SaveModelCallback`。

### 算子注册表

| 符号 | 类 | 作用 |
|---|---|---|
| *(默认)* | `Dot` | `ParameterList` + 对 `A->B` 标注执行 `torch.einsum` |
| `#` | `Get` | 沿首维 gather / select |
| `~` | `Bn` | 通过 permute/reshape 在指定维度子集上做 `BatchNorm1d` |
| `@` | `Call` | 从 spec 上下文求值任意表达式（如 `@Swish()`） |
| `+` | `Add` | 带学习的偏置加 |
| `-` | `View` | reshape；`?` 折叠为 `-1` |
| `%` | `Rec` | 复用前 *n* 层（权重共享递归） |

后期 notebook 里真正干活的是 DSL 之外的两个模块：

- **`MGLinear`** — 群等变 / 多头线性层，docstring 里的签名是 `m?(gk), mK@gk, @Gg, +mK -> m?KG`：`m` 个头共享一个 `k→K` 投影，混合前对头分组施加 `g` 种置换。
- **`Red`** — 成对归约包装器，每次把序列轴的两个元素折进同一个 `Sequential`，即手写的对数深度树归约，适用于变长序列。

逐行证据和更完整的分析在 [`docs/project-status.md`](docs/project-status.md)。

---

## 如何开发

### 环境

没有 `requirements.txt`，没有 lockfile，没有 `__init__.py`，也没有包结构。唯一的机器学习依赖是 fastai，而且只用到六个训练辅助函数：

```bash
python -m venv .venv && . .venv/bin/activate
pip install "fastai>=2.7" "torch>=2.0" numpy jupyter
```

- `fastai` — **必需。** 每个 notebook 都以 `from fastai.vision.all import *` 开头，会把 torchvision、PIL、spaCy 和 matplotlib 拖进一个纯文本项目。
- `torch` — `nn.Module`、`Parameter`、`torch.einsum`、`NLLLoss`、`Embedding`、`Conv1d`。
- `numpy` — `np.memmap`、`np.fromiter`、`np.add.at` 以及 dtype 技巧。
- `jupyter` — 打开 `.ipynb`。
- 默认有 CUDA GPU。Notebook 硬编码 `.cuda()`，有些还调用 `torch.cuda.set_device(1)`；不支持 CPU 路径。
- `einn3*` / `einn4*` 需要 CLUE JSON 目录；`einn5-autoencoder-Copy17.ipynb` 需要 `../resnet26d`。两者都不在仓库里。

原始环境的确切版本从未记录。这里写的版本号只是下限，不是可复现的固定值。

### 语料不在仓库里

`einn5*` notebook 从仓库外读取 memmap 的 token 流：

```python
class Dat:
    a = torch.from_numpy(np.memmap(f"../tmp/news2016zhMin100.H", "h"))
    m = np.array([open("../tmp/news2016zhMin100.txt").read()]).view("U1")
```

`../tmp/` 相对于 notebook 所在目录，也就是本目录的上一级。**这个路径在这里不存在**，构建好的语料和上游 `news2016zh_{train,valid}.json` 都不在仓库里。不重新生成语料，`einn5*` 一个都训练不了。

两个路径记录在 [`.env.example`](.env.example) 里，但 notebook 本身**不读环境变量**——它们写死了 `../tmp/...`，作为存档的一部分保持逐字节不变。

| 变量 | 含义 | Notebook 默认值 |
|---|---|---|
| `CZ_EINN_SOURCE_DIR` | 存放 `news2016zh_train.json`、`news2016zh_valid.json` 的目录（逐行 JSON，文本在 `content` 字段） | `../tmp/` |
| `CZ_EINN_CORPUS` | 构建好的语料对 `<NAME>.H` + `<NAME>.txt` 的主名（不含扩展名） | `../tmp/news2016zhMin100` |

#### 语料的格式

| 文件 | 格式 |
|---|---|
| `<NAME>.H` | 原始 **uint16 小端** token ID，连续存放，一个字符一个 ID。用 `np.memmap(path, "H")` 读取。Notebook 用的是 `"h"`（int16），在所有 ID 小于 32768 时字节完全相同。 |
| `<NAME>.txt` | 按 ID 顺序排列的字符。读法是 `np.array([open(path).read()]).view("U1")`——NumPy UTF-32 字符串数组，每字符 4 字节，*不是* UTF-8 字节。 |

已知词表大小：**21482**（完整的 `news2016zh` 字符集）和 **684**（语言模型 notebook 实际使用的 `Min100` 精简集）。少数单元把 `Embedding(684, …)` 和 685 宽的输出层配在一起，这个差一错误在存档里没有解释。

<details>
<summary><strong>语料重建配方（从 notebook 原样抄录）</strong></summary>

```python
# 1. 从源 JSON 的 "content" 字段直接生成 uint16 流。
from re import compile
from json import loads
m = {}
r = compile("[\u4e00-\u9fa5]")
for i in "train", "valid":
    open(f"../tmp/news2016zh_{i}.H", "wb").writelines(
        m.setdefault(i, len(m)).to_bytes(2, "little")
        for i in open(f"../tmp/news2016zh_{i}.json")
        for i in r.findall(loads(i)["content"])
    )
len(m)                                       # -> 21482

# 2. "Min100"：只保留频率高于 max_freq / 100 的字符，
#    重新映射成稠密词表，写回过滤后的流和解码表。
import numpy as np
from multiprocessing.pool import ThreadPool

def f(i):
    c = np.zeros(21482, "I")
    np.add.at(c, a[len(a) * i // 12 : len(a) * (i + 1) // 12], 1)
    return c

a = np.memmap("../tmp/news2016zh.H", "H")
with ThreadPool(12) as p:
    p = np.sum(p.map(f, range(12)), 0)

t = p > p.max() // 100
m = np.zeros(21482, "H")
m[t] = range(t.sum())
m[a[t[a]]].tofile("../tmp/news2016zhMin100.H")
open("../tmp/news2016zhMin100.txt", "w").write(
    np.array([open("../tmp/news2016zh.txt").read()]).view("U1")[t].view(f"U{t.sum()}")[0]
)
```

两点来自读代码而非运行的说明：第 2 步读的是 `news2016zh.H` / `.txt`，而第 1 步写的是 `news2016zh_train.H` / `news2016zh_valid.H`，两者对不上；阈值是 `max_freq / 100`，所以叫 "Min100"。这个构建过程从未写成脚本——只能照着配方手工执行。

</details>

### 目录结构

```
.
├── README.md                     # 英文说明
├── README.zh-CN.md               # 本文件
├── LICENSE                       # AGPL-3.0-only
├── .env.example                  # 语料路径说明（notebook 不读取）
├── .gitignore
├── assets/
│   └── logo.svg
├── docs/
│   └── project-status.md         # 分类、证据、边界
└── einn*.ipynb                   # 45 个 notebook，按实验顺序排列
```

---

## 如何贡献

这是**冻结的存档**，没有可以加入的功能路线。欢迎以下类型的修正：

- 事实错误、失效的相对链接、无法解析的 notebook；
- 对语料格式或运行环境的澄清；
- 指出缺失数据集或权重的上游来源。

请不要重排、重命名或"清理" notebook——Copy 编号的序列和写死的路径本身就是记录。开一个带证据的小 PR 或 issue，并把文档修改和代码修改分开。

---

## 路线图

### 已实现

- [x] einsum 字符串 DSL 与 `net(dict)` 解析器 — [`einn.ipynb`](einn.ipynb)
- [x] 完整算子注册表：`Dot`、`Get`、`Bn`、`Call`、`Add`、`View`、`Rec`
- [x] `MGLinear`、`MLinear`、`Red`、`MBatchNorm`
- [x] `iflytek`、`tnews`、`cmnli`、`afqmc`、`cluewsc2020` 的 CLUE 任务 notebook
- [x] 在 `Min100` 语料上做字符级自编码
- [x] `PyramidAutoEncoder` + 逐层 `PyramidCrossEntropyLoss`
- [x] 语料构建配方，原样保存在 notebook 中
- [x] 公开存档文档、许可证与静态审查

### 计划中——*仅在项目重启时考虑*

这些是方向，不是承诺。存档已冻结。

- [ ] 把语料配方变成带测试的脚本，并为期望词表加校验和
- [ ] 用简短的显式训练循环或 Lightning 模块替换 `fastai.vision.all` 星号导入
- [ ] 把 DSL 移植到 `einops` / `torch.fx`，与手写解析器对比
- [ ] 增加 CPU 路径，去掉硬编码的 `.cuda()` / `set_device(1)` 假设
- [ ] 重跑一个任务并保存输出，给出可复现的结果
- [ ] 固定环境，并加一个解析全部 notebook 的冒烟测试

## 未来设想

如果 EINN 重启，它最合适的定位是**教学工具和紧凑记号的实验**，而不是训练框架。那个有意思的问题仍然开放：一个网络到底有多少能写成带类型的字符串，解析器又比手写层多带来了什么？现代版本应该保留 spec，去掉 `fastai` 依赖，用 `einops` 做缩并，用 `torch.fx` 检查字符串生成的图。最值得继续的是金字塔自编码器——对 U 形字符模型的每一层都施加监督。其余部分是一段 2020 年思路的记录，保留下来，是因为走不通的路和能走通的路同样有信息量。

---

## 许可证

**AGPL-3.0-only。** 见 [`LICENSE`](LICENSE)。Notebook 从私有存档迁移而来；本仓库现在采用 GNU Affero 通用公共许可证第 3 版。

还有两项第三方义务：语料派生自 2016 年中文新闻数据集，任何再分发的 `.H` / `.txt` 提取物都受其条款约束（本仓库没有包含这些文件）；`einn5-autoencoder-Copy17.ipynb` 引用了一个来源未记录的 `resnet26d` 权重。
