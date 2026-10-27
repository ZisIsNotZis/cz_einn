<p align="center">
  <img src="assets/logo.svg" alt="EINN — Einstein-notation neural networks" width="820">
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<h1 align="center">EINN</h1>

<p align="center"><strong>Write a neural network as an Einstein-summation string; EINN turns it into PyTorch layers.</strong></p>

<p align="center">A 2020 fastai notebook archive on character-level Chinese language modelling.</p>

> [!IMPORTANT]
> **Frozen archive, not maintained software.** The notebooks are stored with empty outputs. There are no results, checkpoints, corpus, tests, or training script in this repository. It is published so the line of thinking is on the record — not as a baseline, a library, or a promise.

EINN was a one-person experiment: describe an architecture as short text such as `#1mek->m?k`, parse it into real modules, and let `torch.einsum` handle the dimension bookkeeping. The archive follows the idea from a synthetic unit test through CLUE classification tasks to a character-level pyramid autoencoder.

**New here?** Open [`einn.ipynb`](einn.ipynb) — it is the smallest complete statement of the idea. The full evidence trail, including what is missing, is in [`docs/project-status.md`](docs/project-status.md).

---

## Quick start

**Read it — no install.** Open [`einn.ipynb`](einn.ipynb) on GitHub. The second cell is the whole idea in nine lines of spec.

**Run the one self-contained notebook** (synthetic random tokens, no corpus needed; CUDA assumed):

```bash
git clone https://github.com/ZisIsNotZis/cz_einn.git
cd cz_einn
python -m venv .venv && . .venv/bin/activate
pip install "fastai>=2.7" "torch>=2.0" numpy jupyter
jupyter lab einn.ipynb
```

The `einn5*` language-model notebooks need the external corpus described in [How to develop](#how-to-develop). Nothing here is packaged, pinned, or runnable as-is beyond `einn.ipynb`.

---

## What works today

- **A string DSL for architectures.** `net(dict)` maps one-character symbols to layer factories and builds a fastai `Sequential` from a multi-line spec.
- **A working einsum layer.** The default `Dot` layer holds a `ParameterList` and executes the `A->B` dimension annotation with `torch.einsum`.
- **Reusable building blocks.** `MGLinear` (group-equivariant multi-head linear), `MLinear`, `Red` (log-depth pairwise sequence reduction), `Bn`, `Get`, `Call`, `Add`, `View`, `Rec`.
- **Task notebooks.** CLUE-format classification experiments (`iflytek`, `tnews`, `cmnli`, `afqmc`, `cluewsc2020`).
- **A pyramid autoencoder.** `PyramidAutoEncoder` plus `PyramidCrossEntropyLoss`, supervising every level of a U-shaped character model.
- **The corpus recipe.** The character-ID stream builder is preserved verbatim inside the notebooks and reproduced [below](#the-corpus-is-not-here).

The spec that started it, from [`einn.ipynb`](einn.ipynb):

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

### The notebook sequence

| File(s) | What it does |
|---|---|
| [`einn.ipynb`](einn.ipynb) | **Canonical.** The einsum-string DSL, full op registry, `MGLinear` / `MLinear` / `Red`. Trained on synthetic random tokens — a unit test of the machinery, not a task. |
| `einn2*.ipynb` | The same architecture re-expressed as functional generators (`F`, `Red`, `MGLinear`, `MBatchNorm`, `MLinear`); still synthetic data. |
| `einn3*.ipynb`, `einn4*.ipynb` | Real Chinese classification via CLUE-format JSON: `iflytek`, `tnews`, `cmnli`, `afqmc`, `cluewsc2020`. Uses `Red` for variable-length sentences. |
| `einn5-autoencoder*.ipynb` (22 files) | Character-level language modelling / autoencoding on a memmapped Chinese-news corpus. Ends at `PyramidAutoEncoder`; `einn5-autoencoder-Copy20.ipynb` is the furthest-progressed state. |
| `einn5-autoencoder-Copy17.ipynb` | An unrelated branch that pulls in a pretrained `resnet26d` (absent from the repo). |

The copy-numbered files are deliberate: they are the working record, including dead ends. One exact duplicate was removed during the public-release cleanup; the remaining 45 notebooks are byte-identical to the private archive.

---

## Architecture

```
spec text
   │
   ▼
net(dict)  ── one-char symbol ──▶  layer factory
   │
   ▼
fastai Sequential  ── Dot: torch.einsum(A,B->C) ──▶  tensors
   │
   ▼
stock fastai Learner / DataLoaders / fit_one_cycle
```

The repo contributes layers and a data iterator. It contributes no training machinery: every notebook uses fastai's `Learner`, `fit_one_cycle`, `lr_find`, `ShowGraphCallback`, and `SaveModelCallback`.

### The op registry

| Symbol | Class | Role |
|---|---|---|
| *(default)* | `Dot` | `ParameterList` + `torch.einsum` on the `A->B` annotation |
| `#` | `Get` | gather/select along a leading index |
| `~` | `Bn` | `BatchNorm1d` over a chosen subset of dims via permute/reshape |
| `@` | `Call` | evaluate an arbitrary expression (e.g. `@Swish()`) from the spec context |
| `+` | `Add` | learned bias add |
| `-` | `View` | reshape; `?` collapses to `-1` |
| `%` | `Rec` | reuse the previous *n* layers (weight-shared recursion) |

Two blocks outside the DSL do the heavy lifting in later notebooks:

- **`MGLinear`** — a group-equivariant / multi-head linear with the docstring signature `m?(gk), mK@gk, @Gg, +mK -> m?KG`: `m` heads share a `k→K` projection, and `g` permutations of the head group are applied before mixing.
- **`Red`** — a pairwise-reduction wrapper that folds the sequence axis two elements at a time through a shared `Sequential`, i.e. a hand-rolled log-depth tree reduction over a variable-length sequence.

Deeper reading, with line-level evidence: [`docs/project-status.md`](docs/project-status.md).

---

## How to develop

### Environment

There is no `requirements.txt`, lockfile, `__init__.py`, or package structure. The only ML dependency is fastai, and it is used for six training-loop helpers:

```bash
python -m venv .venv && . .venv/bin/activate
pip install "fastai>=2.7" "torch>=2.0" numpy jupyter
```

- `fastai` — **required.** Every notebook opens with `from fastai.vision.all import *`, which drags torchvision, PIL, spaCy and matplotlib into a pure-text project.
- `torch` — `nn.Module`, `Parameter`, `torch.einsum`, `NLLLoss`, `Embedding`, `Conv1d`.
- `numpy` — `np.memmap`, `np.fromiter`, `np.add.at`, dtype tricks.
- `jupyter` — to open `.ipynb`.
- A CUDA GPU is assumed. Notebooks hard-code `.cuda()` and sometimes `torch.cuda.set_device(1)`; CPU is not a supported path.
- `einn3*` / `einn4*` expect a CLUE JSON directory; `einn5-autoencoder-Copy17.ipynb` expects `../resnet26d`. Both are absent.

The original session's exact versions were never recorded. Any version pin here is a floor, not a reproduction.

### The corpus is not here

The `einn5*` notebooks read a memmapped token stream from outside the repository:

```python
class Dat:
    a = torch.from_numpy(np.memmap(f"../tmp/news2016zhMin100.H", "h"))
    m = np.array([open("../tmp/news2016zhMin100.txt").read()]).view("U1")
```

`../tmp/` resolves to a sibling of this directory. **That path does not exist here**, and neither the built corpus nor the upstream `news2016zh_{train,valid}.json` is included. Nothing in `einn5*` can be trained without regenerating it.

The two paths are documented in [`.env.example`](.env.example), but the notebooks themselves do **not** read environment variables — they contain literal `../tmp/...` strings, left byte-identical as part of the archive.

| Variable | Meaning | Notebook default |
|---|---|---|
| `CZ_EINN_SOURCE_DIR` | directory holding `news2016zh_train.json`, `news2016zh_valid.json` (line-delimited JSON; text in `content`) | `../tmp/` |
| `CZ_EINN_CORPUS` | basename (no extension) of the built corpus pair `<NAME>.H` + `<NAME>.txt` | `../tmp/news2016zhMin100` |

#### Expected corpus shape

| File | Format |
|---|---|
| `<NAME>.H` | raw **uint16 little-endian** token IDs, contiguous, one per character. Read with `np.memmap(path, "H")`. The notebooks use `"h"` (int16); byte-identical for every ID below 32768. |
| `<NAME>.txt` | the character for each ID, in ID order. Read as `np.array([open(path).read()]).view("U1")` — a NumPy UTF-32 string array, 4 bytes per character, *not* UTF-8 bytes. |

Known vocabulary sizes: **21482** (full `news2016zh` character set) and **684** (the `Min100` reduced set used by the language-model notebooks). A few cells pair `Embedding(684, …)` with a 685-wide output layer; the off-by-one is unexplained in the archive.

<details>
<summary><strong>Canonical regeneration recipe (verbatim from the notebooks)</strong></summary>

```python
# 1. Raw char -> uint16 stream, straight from the source JSON's "content" field.
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

# 2. "Min100": keep only characters whose frequency exceeds max_freq / 100,
#    remap to a dense vocabulary, and write the filtered stream + decode table.
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

Two notes read from the code, not from a run: step 2 consumes `news2016zh.H` / `.txt`, which step 1 does not produce (step 1 writes `news2016zh_train.H` / `news2016zh_valid.H`); and the threshold is `max_freq / 100`, hence "Min100". The build was never scripted — this is a recipe to follow by hand.

</details>

### Repository layout

```
.
├── README.md                     # this file
├── README.zh-CN.md               # Simplified Chinese version
├── LICENSE                       # AGPL-3.0-only
├── .env.example                  # documented corpus paths (not read by notebooks)
├── .gitignore
├── assets/
│   └── logo.svg
├── docs/
│   └── project-status.md         # classification, evidence, boundaries
└── einn*.ipynb                   # 45 notebooks, archive-ordered by experiment
```

---

## How to contribute

This is a **frozen archive**, so there is no feature roadmap to join. Corrections are welcome:

- a factually wrong statement, a broken relative link, or a notebook that will not parse;
- a clarification about the corpus format or the environment;
- an issue that points to the upstream source of a missing dataset or checkpoint.

Please do **not** reformat, rename, or "clean up" the notebooks — the copy-numbered sequence and the literal paths are the record. Open a small PR or issue with the evidence, and keep changes separate from code edits.

---

## Roadmap

### Implemented

- [x] Einsum-string DSL and `net(dict)` parser — [`einn.ipynb`](einn.ipynb)
- [x] Full op registry: `Dot`, `Get`, `Bn`, `Call`, `Add`, `View`, `Rec`
- [x] `MGLinear`, `MLinear`, `Red`, `MBatchNorm`
- [x] CLUE task notebooks for `iflytek`, `tnews`, `cmnli`, `afqmc`, `cluewsc2020`
- [x] Character-level autoencoding on the `Min100` corpus
- [x] `PyramidAutoEncoder` + per-level `PyramidCrossEntropyLoss`
- [x] Corpus-building recipe, preserved in the notebooks
- [x] Public archive documentation, license, and static review

### Planned — *if the project is ever revived*

These are directions, not commitments. The archive is frozen.

- [ ] Turn the corpus recipe into a tested script with a checksum for the expected vocabulary
- [ ] Replace the `fastai.vision.all` star import with a small explicit training loop or Lightning module
- [ ] Port the DSL to `einops` / `torch.fx` and compare it against the hand-rolled parser
- [ ] Add a CPU path and remove the hard-coded `.cuda()` / `set_device(1)` assumptions
- [ ] Re-run one task with stored outputs and publish a reproducible result
- [ ] Pin the environment and add a smoke test that parses every notebook

## Future vision

If EINN is revived, its best use is as a **teaching tool and a compact notation experiment**, not as a training framework. The interesting question is still open: how much of a network can be expressed as a typed string, and what does a parser buy over writing layers by hand? A modern version would keep the spec, drop the `fastai` dependency, lean on `einops` for the contractions, and use `torch.fx` to inspect the graph the string produced. The pyramid autoencoder — supervising every level of a U-shaped character model — is the part most worth carrying forward. The rest is a record of a 2020 line of thought, kept because the dead ends are as informative as the code.

---

## License

**AGPL-3.0-only.** See [`LICENSE`](LICENSE). The notebooks were migrated from a private archive; the repository now carries the GNU Affero General Public License v3.0.

Two third-party obligations remain: the corpus is derived from the 2016 Chinese news dataset, whose terms would govern any redistributed `.H` / `.txt` extracts (none are included here), and `einn5-autoencoder-Copy17.ipynb` references a `resnet26d` checkpoint whose provenance is not recorded.
