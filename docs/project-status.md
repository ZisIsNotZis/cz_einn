# cz_einn project status

## Classification

Archived personal research notebooks. Not a library, not a package, not an
application: 46 standalone Jupyter notebooks exploring one idea — describing a
neural network as an Einstein-summation string and building it from that string
("EINN", inferred to mean *Einstein-notation Neural Network*). Character-level
Chinese language modelling, prototyped against the 2016 Chinese news corpus.
No API, no entry point, no installable surface.

## Status

**Deprecated (2020-era fastai notebook research) — kept as an archive.**

Not abandoned in the sense of being deleted, and not complete in any sense. The
record stops mid-experiment: the last notebook in the sequence
(`einn5-autoencoder-Copy20.ipynb`) ends on `m.fit_one_cycle(100, .01, 9, 99, .01)`
followed by `m.fine_tune(300)`, with no captured result. Nothing here is
maintained, and the `fastai` dependency it is built on is no longer a reasonable
choice for new work (see Evidence → Deprecation).

This status is deliberate, not incidental. The repo is published so the line of
thinking is on record, not so that it can be run or extended.

## Evidence

**The architecture.** `einn.ipynb` is the canonical file. It defines `net(dict)`,
a one-character-symbol registry that parses a multi-line spec such as
`#1mek->m?k` / `@MGLinear(m,g,k,k,0)->m?kG` / `~mk` into a `Sequential`, with the
default `Dot` layer executing the dimension annotation through `torch.einsum`.
Around it sit `MGLinear` (group-equivariant multi-head linear, docstring
`m?(gk), mK@gk, @Gg, +mK -> m?KG`), `MLinear`, `Red` (log-depth pairwise
sequence reduction through a shared `Sequential`), `Bn`, `Get`, `Call`, `Add`,
`View` and `Rec`. Later notebooks replace the DSL with functional generators
(`einn2*`) and then move to real tasks (`einn3*`, `einn4*`: `iflytek`, `tnews`,
`cmnli`, `afqmc`, `cluewsc2020`) and to character-level autoencoding
(`einn5*`) with `PyramidAutoEncoder` + `PyramidCrossEntropyLoss` (supervision at
every level of the pyramid).

**Training loop.** Stock fastai throughout — `Learner(DataLoaders(...), model,
loss, metrics=..., cbs=[ShowGraphCallback, SaveModelCallback(...)])`, then
`lr_find()`, `fit_one_cycle(...)`, `fine_tune(...)`. The repo contributes layers
and a data iterator; it contributes no training machinery.

**The memmapped data is a char-ID stream, not embeddings.** The `einn5*`
notebooks do `torch.from_numpy(np.memmap("../tmp/news2016zhMin100.H", "h"))` at
class-definition time, with `m = np.array([open("../tmp/news2016zhMin100.txt").read()]).view("U1")`
as the decode table. `.H` is raw uint16 little-endian character token IDs, one
per character, contiguous; `.txt` is the character for each ID in ID order (a
NumPy UTF-32 `U1` array, 4 bytes per character). The builder (preserved verbatim
in `einn5-autoencoder.ipynb` cell 4 and in the markdown cells of
`einn5-autoencoder-Copy4-Last.ipynb`) extracts `[\u4e00-\u9fa5]` from the
`content` field of `news2016zh_{train,valid}.json`, assigns IDs by first-seen
order, then keeps only characters whose frequency exceeds `max_freq / 100` —
hence "Min100" — and remaps to a dense vocabulary. Vocabulary sizes seen in the
code: 21482 (full charset) and 684 (the reduced set actually used).

**Scale.** 46 notebooks, 273 cells, ~5430 code-cell lines (including IPython
magics), ~376 KB.

**Validation run at archive time.**

```
# every notebook parses as JSON
ipynb JSON ok=46 bad=0

# .py files: 0 exist, so `python3 -m py_compile` has nothing to check
find . -name "*.py" | wc -l   ->  0

# supplementary: parse each code cell as Python after stripping !/%/? magics
code cells parsed ok: 210   failed: 5
```

The 5 failures are expected notebook artefacts, not corruption: incomplete
line-wrapped cells (`einn5-autoencoder.ipynb` cell 3), IPython-only `obj??`
syntax not stripped by the filter (`einn3-Copy1.ipynb`, `einn5-autoencoder-Copy17.ipynb`),
and two malformed cells in `einn5-autoencoder-Copy7.ipynb`. All 46 files are
valid JSON.

**Security scan.** Requested gate, run against the destination copy:

```
grep -rniE -l -e "-----BEGIN [A-Z ]*PRIVATE KEY-----" \
  -e "api[_-]?key *[:=] *['\"][A-Za-z0-9_-]{12,}" \
  -e "token *[:=] *['\"][A-Za-z0-9_.-]{20,}" \
  -e "password *[:=] *['\"][^'\"]{6,}" \
  -e "intranet\." -e "@<employer-domain>" -e "[0-9]{1,3}(\.[0-9]{1,3}){3}" \
  --exclude-dir=.git /home/z/vibe/cz_einn
-> no matches (exit 1)
```

(The employer-domain alternative is written as a placeholder so that this
document does not itself trip the rule it documents.)

A wider sweep for the employer name, `intranet`, `/home/`, `/Users/`, `C:\`, `/root/`,
`/mnt/`, `secret`, `passwd`, `bearer`, `ghp_`, `sk-` and any dotted-quad literal
also returned zero matches. No credentials, no internal hostnames, no PII. No
rotation needed.

**Portability.** No absolute `/home/<user>/...` paths existed to genericise —
zero matches across the tree. Every external reference was already relative
(`../tmp/...`, `../bert/albert/CLUEdataset/...`, `../resnet26d`). Those relative
paths were left byte-identical rather than rewritten, because in an archive they
are the evidence of what the data was; the mapping to `CZ_EINN_CORPUS` /
`CZ_EINN_SOURCE_DIR` is documented in `README.md` and `.env.example` instead.
Copy fidelity confirmed with `diff -r` (clean).

**Deprecation.** Two independent reasons.

*Tooling:* `fastai` is in maintenance mode — latest release 2.8.12, Apache-2.0,
self-classified `Development Status :: 4 - Beta` on PyPI
([release metadata](https://pypi.org/pypi/fastai/json)) — and the dependency is
misused here: every text-model notebook does `from fastai.vision.all import *`
purely for `Learner`, `DataLoaders`, `fit_one_cycle`, `lr_find`,
`ShowGraphCallback` and `SaveModelCallback`, dragging torchvision, PIL and spaCy
into the process. A star import also makes the notebooks un-analysable and the
requirement unpinnable.

*Approach:* the "network as an einsum string" bet is a 2019–2020 idea since done
properly by [`einops`](https://github.com/arogozhnikov/einops), with graph
capture handled by `torch.fx` / `torch.compile`. The hand-rolled
group-equivariant `MGLinear` is subsumed by grouped convolutions and the
equivariant-NN literature; `Red` is a stand-in for
`pack_padded_sequence` / masked pooling; the `np.memmap` + hand-written
`__iter__` data path predates present-day `torch.utils.data` practice.

The *files* were last saved 2026-02-13, but the idioms are roughly five years
older than that date. The status label reflects the idioms, not the mtime.

## Boundaries

- **No results of any kind.** Every code cell in all 46 notebooks has an empty
  `outputs` list. There is no stored metric, loss curve, accuracy, or
  `m.recorder.values` dump anywhere. Cells that *compute* results
  (`m.recorder.values`, `m.validate()`) do not keep them. This repo cannot be
  cited for any empirical claim, including a negative one.
- **No checkpoints.** `SaveModelCallback(fname="model5-autoencoder-Copy20", ...)`
  and a `%cd models` + `.pth` rename cell reference weight files that are not
  here.
- **No corpus.** `../tmp/news2016zhMin100.H` / `.txt` are absent (verified: no
  `news2016zh*` file exists under the source tree). The upstream
  `news2016zh_{train,valid}.json` is absent too. Not one notebook can be trained
  as-is. Regeneration is a copy-by-hand recipe, never a script, and there is no
  test that the recipe still reproduces the expected vocabulary.
- **No other datasets.** `einn3*`/`einn4*` need a CLUE JSON directory and
  `einn5-autoencoder-Copy17.ipynb` needs a `resnet26d` checkpoint; both are
  outside the repo and their provenance is unrecorded.
- **Not reproducible.** No `requirements.txt`, no lockfile, no environment
  record, no CI. Library versions from the original session are unknown; any
  floor in `README.md` is an estimate. CUDA is assumed and hard-coded
  (`.cuda()`, `torch.cuda.set_device(1)`), so CPU execution is not a supported
  path.
- **No write-up.** No paper, report, experiment log or conclusion about whether
  the DSL idea worked. The acronym "EINN" is never expanded in the archive; the
  Einstein-notation reading in this document is an inference from the code.
- **Unresolved detail.** Some cells pair `Embedding(684, ...)` with a `685`-wide
  output layer. The off-by-one is unexplained — possibly a padding class,
  possibly a bug. Left as found.
- **License: none.** No `LICENSE`, no `COPYING`, no headers. Default is all
  rights reserved. The corpus's own terms would also govern any redistributed
  `.H`/`.txt` extracts; none are redistributed here.
- **Scope of this review.** Conclusions are from reading the notebooks and the
  source tree, not from executing them — no corpus and no GPU were available.
  Claims about runtime behaviour are marked as read-inference throughout.
