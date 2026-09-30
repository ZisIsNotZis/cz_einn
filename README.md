# cz_einn

**Status: deprecated (2020-era fastai notebook research) — kept as an archive.**

EINN was a one-person research sketch: a "define the network as a string" experiment
for character-level Chinese language modelling, written as 46 fastai/Jupyter
notebooks against the 2016 Chinese news corpus. It exists to answer a single
question — *can an architecture be expressed in Einstein-summation notation and
searched as text, instead of being hand-written layer by layer?* — and it is
published here only as a record of that line of thinking, not as working software.

- **Author:** zisisnotzis (private archive)
- **Era:** notebook idioms are ~2020; files last saved 2026‑02‑13 (single squashed git commit in the source archive)
- **Language:** Python / Jupyter, fastai v2 + PyTorch
- **Size:** 46 notebooks, 273 cells, ~5.4k code-cell lines
- **Corpus:** **not included** — see [How to run](#how-to-run)
- **License:** none declared — see [License](#license)

---

## What EINN actually is

The archive never expands the acronym; the name is inferred from the code. The
canonical notebook [`einn.ipynb`](einn.ipynb) declares a network as a short text
spec:

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

A `net(dict)` registry maps one-character symbols to layer factories, and a
default `Dot` layer executes the `A->B` dimension annotations with
`torch.einsum` on a translated index string. That is the whole idea: **EINN =
"Einstein-notation Neural Network"** — a network whose layers are literally
Einstein-summation expressions. (Inferred, not asserted by the author anywhere in
the repo.)

### The pieces you will actually recognise

| Symbol | Class | Role |
|---|---|---|
| *(default)* | `Dot` | `ParameterList` + `torch.einsum` on the `A->B` annotation |
| `#` | `Get` | gather/select along a leading index |
| `~` | `Bn` | `BatchNorm1d` applied over a chosen subset of dims via permute/reshape |
| `@` | `Call` | evaluate an arbitrary expression (e.g. `@Swish()`) from the spec context |
| `+` | `Add` | learned bias add |
| `-` | `View` | reshape; `?` collapses to `-1` |
| `%` | `Rec` | reuse the previous *n* layers (weight-shared recursion) |

Two non-DSL building blocks do the real work in the later notebooks:

- **`MGLinear`** — a group-equivariant / multi-head linear, documented by its own
  docstring as an einsum signature `m?(gk), mK@gk, @Gg, +mK -> m?KG`: `m` heads
  share a `k→K` projection, and `g` permutations of the head group are applied
  before mixing.
- **`Red`** — a pairwise-reduction wrapper that folds the sequence axis two
  elements at a time through a shared `Sequential`, i.e. a hand-rolled
  log-depth tree reduction over a variable-length sequence.

The training loop is pure stock fastai: `Learner` + `DataLoaders`,
`fit_one_cycle`, `lr_find`, `ShowGraphCallback`, `SaveModelCallback`. Nothing in
this repo is a training framework; it is a pile of layer experiments assembled
inside fastai's `Learner`.

### The notebooks, in order

| File(s) | What it does |
|---|---|
| [`einn.ipynb`](einn.ipynb) | **Canonical.** The einsum-string DSL, full op registry, `MGLinear`/`MLinear`/`Red`. Trained on *synthetic random tokens* — a unit test of the machinery, not a task. |
| `einn2*.ipynb` | Same architecture re-expressed as functional generators (`F`, `Red`, `MGLinear`, `MBatchNorm`, `MLinear`); still synthetic data. |
| `einn3*.ipynb`, `einn4*.ipynb` | Pivot to real Chinese classification via CLUE-format JSON: `iflytek`, `tnews`, `cmnli`, `afqmc`, `cluewsc2020`. Uses the `Red` reduction for variable-length sentences. |
| `einn5-autoencoder*.ipynb` (23 files) | Pivot to **character-level language modelling / autoencoding** on the memmapped Chinese-news corpus. Ends at `PyramidAutoEncoder` — a U-shaped encoder/decoder with per-level skips and a `PyramidCrossEntropyLoss` that supervises *every* pyramid level. `einn5-autoencoder-Copy20.ipynb` is the furthest-progressed state. |

`einn5-autoencoder-Copy17.ipynb` is an unrelated branch that pulls in a
pretrained `resnet26d` (absent from the repo) — an experiment that did not go
anywhere.

---

## How to run

> **Nothing here runs as-is.** There is no `requirements.txt`, no pinned
> environment, no `__init__.py`, no package structure, and no driver script. These
> are interactive notebooks that were run top-to-bottom in a Jupyter session with
> a local corpus on disk. Treat "how to run" as *how to reconstruct the session*.

### 1. Environment

Only one import is external to PyTorch, and it is the one this repo is now
defined by:

```bash
python -m venv .venv && . .venv/bin/activate
pip install "fastai>=2.7" "torch>=2.0" numpy jupyter
```

- `fastai` — **required.** Every notebook opens with
  `from fastai.vision.all import *`. It is used for exactly six things:
  `Learner`, `DataLoaders`, `fit_one_cycle`, `lr_find`, `ShowGraphCallback`,
  `SaveModelCallback`. Note that importing `fastai.vision.all` drags in
  torchvision, PIL and (via fastai's own deps) spaCy for a model that is pure
  text.
- `torch` — the actual modelling: `nn.Module`, `Parameter`, `torch.einsum`,
  `NLLLoss`, `Embedding`, `Conv1d`.
- `numpy` — only for `np.memmap`, `np.fromiter`, `np.add.at` and dtype tricks.
- `jupyter` — to open `.ipynb`.
- A CUDA GPU is assumed. The notebooks hard-code `.cuda()` and, in places,
  `torch.cuda.set_device(1)`; `dat` in `einn.ipynb` defaults to `dev="cuda"`.
  They will not run unchanged on CPU.
- `einn3`/`einn4` additionally expect a CLUE JSON directory; `einn5-autoencoder-Copy17.ipynb`
  expects `../resnet26d`. Both are absent.

The original session's exact library versions were **not recorded** — no
`pip freeze`, no `conda.yaml`, no notebook metadata beyond a kernel name. Any
version pin in this README is a floor, not a reproduction.

### 2. The corpus is NOT in this repo

This is the important caveat. The `einn5*` notebooks read a memmapped token
stream from *outside* the repository:

```python
class Dat:
    a = torch.from_numpy(np.memmap(f"../tmp/news2016zhMin100.H", "h"))
    m = np.array([open("../tmp/news2016zhMin100.txt").read()]).view("U1")
```

`../tmp/` resolves relative to the notebook, i.e. to the sibling of this
directory. **Verified: that path does not exist.** Neither `news2016zhMin100.H`
nor any `news2016zh*` file is present anywhere under the source tree, and the
upstream raw corpus (`news2016zh_train.json` / `news2016zh_valid.json`) is absent
too. Nothing in this repo can be trained without regenerating or supplying it.

If you want to run the notebooks, point them at a corpus using these variables
(the notebooks themselves are left byte-identical to the archive and still
contain the literal `../tmp/...` strings — see
[Paths and portability](#paths-and-portability)):

| Variable | Meaning | Default assumed by notebooks |
|---|---|---|
| `CZ_EINN_SOURCE_DIR` | directory holding the raw corpus `news2016zh_train.json`, `news2016zh_valid.json` (one JSON object per line, `content` field holds the text) | `../tmp/` |
| `CZ_EINN_CORPUS` | basename (no extension) of the built corpus pair `<NAME>.H` + `<NAME>.txt` | `../tmp/news2016zhMin100` |

A copy of `.env.example` documents these.

#### Expected shape of the corpus

Both files are flat, single-column, **not** embeddings — they are a character
token-ID stream plus its decode table:

| File | Format |
|---|---|
| `<NAME>.H` | raw **uint16 little-endian** token IDs, contiguous, one ID per character. Read with `np.memmap(path, "H")` (dtype `uint16`). The notebooks use `np.memmap(path, "h")` — signed `int16` — which is byte-identical for any ID below 32768; the whole vocabulary is far below that, so the two readings agree. |
| `<NAME>.txt` | the corresponding character for each ID, in ID order. Read as `np.array([open(path).read()]).view("U1")` — a NumPy UTF-32 string array, 4 bytes per character (note: *not* UTF-8 bytes). |

Known vocabulary sizes appearing in the code: **21482** (full news2016zh
character set) and **684** (the `Min100` reduced set actually used by the
language-model notebooks). A few cells pair `Embedding(684, …)` with an output
layer of `685`; that off-by-one is unexplained in the archive and may be a
padding class or a bug.

#### Canonical regeneration recipe

The exact build code is preserved verbatim in the notebooks (notably
`einn5-autoencoder.ipynb` cell 4 and the markdown cells of
`einn5-autoencoder-Copy4-Last.ipynb`). Reproduced unchanged:

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

Two notes on that recipe, both from reading it rather than running it: step 2
consumes `news2016zh.H` and `news2016zh.txt`, which step 1 does *not* produce
(step 1 writes `news2016zh_train.H` / `news2016zh_valid.H`); and the filtering
threshold is `max_freq / 100`, which is where the "Min100" in the filename comes
from. The build was never turned into a script, so it is a recipe to follow by
hand, not a command to run.

---

## What is here vs. what is not

**Here:**

- 46 notebooks: the full working record of the EINN idea, including every
  dead-end and copy-numbered variant (`-Copy1` … `-Copy20`).
- The einsum-string DSL, its op registry, `MGLinear` / `MLinear` / `Red`, and the
  later `PyramidAutoEncoder` / `PyramidCrossEntropyLoss`.
- The corpus-building recipe, inline in the notebooks.
- The one numeric result that survives anywhere in the archive, left as a code
  comment rather than an output: `t @ np.log2(t) #-9.655227239498798`, a corpus
  entropy figure in `einn5-autoencoder-Copy4-Last.ipynb`.

**Not here:**

- **No results.** Every notebook cell in the archive has an empty `outputs`
  list. There is not a single stored metric, loss curve, printed accuracy or
  `m.recorder.values` dump anywhere in the 46 files. The notebooks *call*
  `m.recorder.values` and `m.validate()`; they never keep the answer.
- **No checkpoints.** `SaveModelCallback(fname="model5-autoencoder-Copy20", …)`
  and a `%cd models` + `rename(..., i+'.pth')` cell reference `models/*.pth`
  files. The directory does not exist here; the weights stayed in the original
  working tree.
- **No write-up.** No paper, no report, no experiment log, no README (this one is
  new), no conclusion about whether the einsum-DSL idea worked. The archive stops
  mid-experiment at `fit_one_cycle(100, .01, 9, 99, .01)` followed by
  `fine_tune(300)`.
- **No data.** The corpus and the CLUE datasets are outside the repo.
- **No tests, no packaging, no CI, no license file.**

So: this is a lab notebook, not a project. Nothing here demonstrates that EINN
beat anything, and the archive should not be read as claiming that it did.

---

## Deprecation note

**This is dated, and the label at the top of this README is deliberate.**

*The `fastai` dependency.* In 2026, `fastai` is a maintenance-mode library: the
latest release is 2.8.12 (Apache-2.0), it still classifies itself
`Development Status :: 4 - Beta` on PyPI
([release metadata](https://pypi.org/pypi/fastai/json)), and it has seen no
substantive API work in years. Two specific problems for *this* code:

1. It is the wrong half of the library. Everything here is text and tensors, yet
   every notebook does `from fastai.vision.all import *`, pulling torchvision,
   PIL, spaCy and matplotlib into the process to obtain six training-loop
   helpers. A modern equivalent is ~30 lines of plain PyTorch plus a `tqdm`
   loop, or Lightning / HF `Trainer`.
2. It is a star-import oral tradition. `from fastai.vision.all import *` defines
   `Learner`, `accuracy`, `init`, `Swish`, `set_num_threads` and more by
   side-effect, so the notebooks are not independently readable and cannot be
   statically checked. It also makes `fastai` an unpinnable hard requirement for
   code that barely uses it.

*The approach.* The core bet — describe a network as an Einstein-notation string
and have code build it — is a 2019–2020 idea that has since been done properly by
[`einops`](https://github.com/arogozhnikov/einops) (compositional, typed,
well-tested, widely adopted) and by `torch.fx` / `torch.compile` for actually
capturing and transforming graphs. `MGLinear`'s group-equivariant hand-rolled
`einsum` construction has meanwhile been absorbed into `torch.nn.functional.conv2d`
grouped convolutions and the equivariant-NN literature. The `Red` pairwise
reduction is a hand-rolled stand-in for what `torch.nn.utils.rnn.pack_padded_sequence`
or a plain masked mean now covers. And the entire workflow — memory-mapping a
raw `uint16` file and hand-writing `Dat.iterate` — predates `torch.utils.data`
being used the way it is used today.

None of this makes the archive worthless: the DSL is a neat, compact
demonstration, and `PyramidAutoEncoder` with per-level supervision is a real
idea. But it should not be reintroduced as a dependency on `fastai`, and it
should not be presented as a current baseline. **Read it as history.**

---

## Paths and portability

Requested hardening was applied and is reported honestly:

- **Absolute `/home/<user>/...` paths: none existed.** A recursive scan of the
  copied tree for `/home/`, `/Users/`, `/root/`, `/mnt/` and Windows drive
  prefixes returned zero matches. All data references were already relative
  (`../tmp/...`, `../bert/albert/CLUEdataset/...`, `../resnet26d`), so **no code
  was rewritten** for absolute paths — there was nothing to genericise.
- **Relative external paths were left as-is, on purpose.** The notebooks are an
  archive, and the `../tmp/news2016zhMin100.H` strings *are* the evidence of
  what the corpus was and where it lived. Rewriting 40+ duplicated `Dat` class
  definitions across copy-numbered notebooks would destroy that record for no
  benefit. Instead the mapping to `CZ_EINN_CORPUS` / `CZ_EINN_SOURCE_DIR` is
  documented above and in `.env.example`.
- **Copy fidelity was verified**: the notebooks are byte-identical to the source
  archive (`diff -r` clean, excluding only `README.md`, `docs/`, `.gitignore`
  and `.env.example`, which are new).

---

## Security scan

The archive was scanned for private keys, API keys, tokens, passwords, intranet
hostnames, employer-domain (`@<company>`) email addresses and bare IP literals.
**The scan is clean** — zero matches. See
[`docs/project-status.md`](docs/project-status.md) for the exact command and
result. No credentials were found, so no rotation is needed.

---

## License

**None.** There is no `LICENSE`, no `COPYING`, and no license header in any
notebook; the source archive has no license metadata either. Absent an explicit
grant, the default is *all rights reserved* — the code is source-available for
reading, not licensed for reuse, modification or redistribution. If you are the
author and want this to be genuinely publishable, add a license file; if you are
not the author, ask before reusing anything here.

Two third-party obligations are worth flagging even if you do add a license:
the corpus is derived from the 2016 Chinese news dataset, whose own terms govern
any redistributed `.H`/`.txt` files (the current repo redistributes none), and
`einn5-autoencoder-Copy17.ipynb` references a `resnet26d` checkpoint whose
provenance is not recorded.

---

## Layout

```
.
├── README.md                     # this file (new)
├── .env.example                  # documented corpus env vars (new)
├── .gitignore                    # new
├── docs/
│   └── project-status.md         # Classification / Status / Evidence / Boundaries (new)
└── einn*.ipynb                   # 46 notebooks, byte-identical to the private archive
```
