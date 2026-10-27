# Contributing to cz_einn

EINN is a **frozen archive**: 45 standalone notebooks kept as the record of a 2020
line of thought. There is no roadmap to join, and the notebooks are not going to
be modernised in place. What the project does want is accuracy.

## Scope — what a contribution may touch

Welcome:

- a factually wrong statement in `README.md`, `README.zh-CN.md` or
  [`docs/project-status.md`](docs/project-status.md), with evidence;
- a broken relative link or a notebook that no longer parses as JSON;
- a clarification about the corpus format, the `../tmp/` paths, or the environment;
- a pointer to the upstream source of a missing dataset or checkpoint.

Not welcome:

- reformatting, renaming, re-running (which would fill in `outputs`), or
  "cleaning up" the notebooks — the copy-numbered sequence, the literal
  `../tmp/...` paths and the empty outputs are the record;
- new dependencies, packaging, a `requirements.txt`, or CI;
- claims of results. No cell in this repository stores a metric, and nothing in
  it may be cited for an empirical claim.

## How to check your change

There is no build, no test suite and no linter — there is nothing to install
either. If you touch a notebook, only verify that the tree is still valid JSON:

```bash
# every notebook must parse
for f in *.ipynb; do python -m json.tool "$f" > /dev/null || echo "BAD $f"; done
```

If you touch Markdown, check that every relative link resolves inside the
repository (no absolute URLs to files that exist locally) and that a heading
anchor you reference still exists.

## Pull requests and issues

- One concern per PR, and keep documentation edits separate from any notebook
  change.
- Say what you verified and how. "Read from the code" and "ran it" are different
  claims, and the docs mark them differently.
- For a notebook change, explain why the archive's fidelity is not harmed.
- Open an issue first for anything larger than a one-line correction; the
  maintainer may prefer to leave a defect documented as a defect.

File corrections as an issue at
<https://github.com/ZisIsNotZis/cz_einn/issues>.

## License

Contributions are accepted under the repository's license,
**AGPL-3.0-only** — see [`LICENSE`](LICENSE).
