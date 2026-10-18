# Project Audit — Data Mining & Machine Learning Curriculum (Indonesian)

Phase 1 · Date: 2026-10-18 · Scope: read-only inspection. Nothing was installed, committed, pushed, deployed or deleted.

## 1. What was inspected

Local repository `D:\ucup\Workspace\blog\materi-machine-learning` (not the remote; the remote was deliberately **not** fetched or assumed).

| Area | Files |
|---|---|
| Config | `mkdocs.yml`, `requirements.txt`, `.gitignore` |
| Theme assets | `docs/stylesheets/extra.css`, `docs/javascripts/mathjax.js`, `docs/javascripts/mermaid-config.js`, `docs/assets/vendor/mermaid/{VERSION.txt,LICENSE}` (the 3.5 MB `mermaid.min.js` was not read) |
| Content | `docs/index.md`, `getting-started/`, `curriculum/{index,sample-chapter}.md`, `authoring/{index,code,math,components}.md`, `examples/{index,mermaid}.md` |
| Snippet sources | `examples/programming/*` (7 files), `examples/mathematics/sigmoid.md` |
| Project docs | `README.md`, `PRD.md`, `TECHNICAL_DESIGN.md`, `VALIDATION_REPORT.md`, `PROMPT_MKDOCS_TEMPLATE.md`, `LICENSE` |
| Environment | `git log/remote`, `pip list` (global), `py -0` |

Not present: `CLAUDE.md`, `.github/` (no workflows), `overrides/`, `.venv/`, `site/`, any Jupyter notebook, any ML content.

## 2. Local vs. remote — do not assume they match

- Local Git: branch `main`, **zero commits, zero remotes**. Everything is untracked. The local tree is a freshly generated copy of the template, not a clone of `TopekoX/mkdocs-template`.
- `README.md` and `mkdocs.yml` point at `topekox.github.io/mkdocs-template`; the remote state (extra commits, workflows, `gh-pages`) is unknown. **Unresolved: confirm which is the source of truth** (see §8).
- The existing `PRD.md` / `TECHNICAL_DESIGN.md` / `VALIDATION_REPORT.md` describe the **template** (written 2026-10-17 at `D:\Workspace\blog\mkdocs-template`), not this curriculum. They are kept; curriculum material is added as a clearly separated "Part B".

## 3. Existing capabilities and reuse verdict

| Capability | Evidence | Verdict |
|---|---|---|
| Material theme, System/Light/Dark palette | `mkdocs.yml` theme.palette | **Reuse as-is** |
| Navigation: sections, indexes, breadcrumbs (`navigation.path`), prev/next footer, top button, tracking | theme.features | **Reuse** |
| Search (`search`, highlight, suggest) | plugins/features | **Reuse**; Indonesian stemming unverified (§5) |
| Code: highlight, copy, annotations, line numbers, titles, tabs | `content.code.*`, `pymdownx.highlight/superfences/tabbed` | **Reuse** |
| Snippet include from `examples/` (`check_paths: true`) | `pymdownx.snippets` | **Reuse** — ideal for single-sourcing runnable `.py` files |
| Admonitions, `details`, tabs, task lists, tables, footnotes, def lists, `attr_list`, `md_in_html`, grid cards | extensions list | **Reuse** (`details` = collapsible exercise solutions) |
| Math: MathJax 3.2.2 via arithmatex `generic`, re-typeset on `document$` | `mathjax.js`, CDN script | **Reuse**; online-only (CDN) |
| Mermaid 11.17.2 vendored, strict, theme-synced | `mermaid-config.js` | **Reuse** (workflow/pipeline diagrams) |
| CSS (≈70 lines): reading width, zebra tables, focus ring, scrollable equations, reduced motion | `extra.css` | **Reuse**; add only what ML content proves necessary |
| Strict validation of links/anchors | `validation:` block | **Reuse** |
| Authoring patterns (objectives → summary) | `docs/curriculum/sample-chapter.md` | **Reuse as base**, extend (§4 of PRD Part B) |
| Prior validation evidence (Chrome, 3 widths, 2 schemes) | `VALIDATION_REPORT.md` | Reference only; **stale** for this tree (no re-run yet) |

No theme redesign is justified. All needs below are met by content conventions and small additive config.

## 4. Must change (identity/config, not design)

| Item | Current | Needed |
|---|---|---|
| `site_name`, `site_description` | "Programming Course Template" | Indonesian ML/Data Mining title |
| `theme.language` | `en` | `id` (Material ships Indonesian UI strings — **verify at build**) |
| `site_url`, `repo_url`, `repo_name` | `topekox/mkdocs-template` | real curriculum repo (decision pending) |
| `nav` | template sample | five-chapter nav (see TD Part B) |
| `docs/index.md`, `curriculum/*` | English template text | replaced/extended in implementation phases, **not deleted** — move template demo pages under an `authoring/` reference or keep as-is until approved |
| `README.md` | template instructions | add "Curriculum" section; keep template how-to |

## 5. Gaps and risks

| # | Gap / risk | Severity | Mitigation |
|---|---|---|---|
| G1 | No ML/data stack installed globally (numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter all absent) and **no `.venv`** exists | High | Separate `requirements-ml.txt` + venv in the implementation phase; user approval required to install |
| G2 | Python **3.14.7** only. scikit-learn/pandas/numpy/statsmodels/mlxtend wheel availability for 3.14 is unverified; template claims 3.12+ untested | High | Check wheels with `pip download --dry-run`/`pip install --dry-run` in the venv phase; if gaps, install a 3.12/3.13 alongside via `py` launcher, document supported version |
| G3 | Examples will rot silently if code in Markdown is not executed | High | Single-source runnable scripts in `examples/`, include via snippets, run all in CI/local test; commit expected outputs |
| G4 | Jupyter Notebook is a stated requirement, but MkDocs has no notebook support; `.ipynb` JSON diffs poorly | Medium | Decision D2: scripts + notebooks generated from them (jupytext) vs. `mkdocs-jupyter`. Default: scripts are source of truth, notebooks derived and downloadable |
| G5 | Rendered plots: Markdown pages cannot run code | Medium | Pre-generate PNG/SVG (fixed seed) into `docs/assets/img/<chapter>/` by a build script; light/dark readability of images must be checked (use transparent bg or neutral palette) |
| G6 | Datasets: licence, size, availability. Network fetchers (`fetch_openml`, `fetch_california_housing`) break reproducibility | Medium | Prefer scikit-learn bundled sets and seeded synthetic generators; small CSVs in `examples/data/` with `README` listing source + licence; no large files in Git |
| G7 | Indonesian search quality: Material's `search` plugin language list must include `id`; unverified | Low | Set `plugins.search.lang: [id, en]` if supported, else keep default; test with "klasifikasi" |
| G8 | MathJax is online-only; offline learners (and CI browsers) see raw LaTeX | Low-Med | Accept, or vendor `tex-mml-chtml.js` (decision D5) |
| G9 | Mermaid (3.5 MB) loaded on every page | Low | Accept; revisit only if page-weight complaints |
| G10 | Material 9.x in maintenance; MkDocs 2.0 notice. `mkdocs<2` cap holds | Low | Keep cap; revisit when migrating |
| G11 | `$` in arithmatex: currency ("$5") may be parsed as math; Indonesian text uses "Rp", low risk | Low | Authoring rule: escape `\$` |
| G12 | Stray root file `.shots$4.png` (80 KB, `$` in name) referenced by `README.md` image link; not ignored by `.gitignore` (which ignores `.shots/`) | Low | Decide: keep (README relies on it) or move to `docs/assets/`; **not touched** |
| G13 | Global `pymdown-extensions` is 11.0.2, `VALIDATION_REPORT` records 12.3 (from a venv that no longer exists) | Info | Fresh venv will resolve; re-validate |
| G14 | No `CLAUDE.md` — Claude Code has no project conventions (language, tone, chapter template, commands) | Medium | Create in implementation Phase 0 (see plan); not created now because it is not in the audit deliverable list |
| G15 | No CI/deploy workflow; deployment only documented as manual `gh-deploy` | Low | Add Actions workflow after content stabilises; **no deploy now** |
| G16 | Pedagogical risks: data leakage (scaling before split), accuracy on imbalanced data, random split on time series, treating clustering as supervised | High (quality) | Dedicated "common mistakes" blocks and per-task metric tables (TD B.4) |
| G17 | Large `Part B` content in Indonesian: consistent terminology (e.g. *pelatihan/training*, *galat/error*) | Medium | Glossary page + term policy (keep English term in italics at first use) |

## 6. Verdict

The template is a sound, already-validated base. **Reuse everything; change only identity config and nav; add conventions, runnable-example tooling and a Python ML environment.** The genuine work is outside the theme: environment, dataset policy, executable examples, and pedagogical structure.

## 7. Prior art claims not independently re-verified in this phase

Build cleanliness (`mkdocs build --strict`), Chrome rendering, and Python 3.14 compatibility of the **template** come from `VALIDATION_REPORT.md` (2026-10-17). They were not re-run because that would require creating a venv/installing, prohibited in Phase 1.

## 8. Unresolved questions (need the user's decision)

1. **Source of truth:** is this folder meant to become a new repo (new remote) or replace/overwrite `TopekoX/mkdocs-template`? Repo name and `site_url`?
2. **Language policy:** full Indonesian prose with English technical terms and English code identifiers/comments? (Proposed: yes; code comments in Indonesian.)
3. **Notebooks (D2):** scripts-first with generated `.ipynb` (proposed), `.ipynb`-first with `mkdocs-jupyter`, or notebooks only as downloads?
4. **Python version:** accept 3.14 if wheels exist, or standardise on 3.12/3.13 for the ML env?
5. **Forecasting scope:** classical only (naive, moving average, exponential smoothing, ARIMA via `statsmodels`) or also ML-based (lag features + GradientBoosting)? Proposed: both, classical first.
6. **Extra libraries:** OK to add `statsmodels`, `mlxtend` (Apriori/FP-Growth), `joblib`, `scipy`? (Proposed: yes, justified per chapter.)
7. **Deployment target:** GitHub Pages via Actions or manual `gh-deploy`?
8. **Audience level:** assumed undergraduate with basic Python; confirm.
9. **Existing template pages:** keep `docs/authoring/` and `docs/examples/` as hidden/reference, or move out of the learner nav?
10. **`.shots$4.png`:** keep at root or relocate?
