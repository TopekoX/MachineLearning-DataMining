# PRD — Reusable MkDocs Documentation Template

Status: Draft for approval · Date: 2026-10-17

## 0. Phase 1 Audit Summary

| Item | Finding |
|---|---|
| Repository | `D:\Workspace\blog\mkdocs-template`, Git initialised on `main`, **no commits, no remote** |
| Existing content | Only `PROMPT_MKDOCS_TEMPLATE.md` (the task spec). No `mkdocs.yml`, `docs/`, CSS, JS, requirements, or deploy scripts |
| Existing work to preserve | `PROMPT_MKDOCS_TEMPLATE.md` (left untouched). No deployment scripts or safeguards exist |
| GitHub Pages | Not currently used: no remote, no `gh-pages` branch, no workflow |
| Global Python | 3.14.8 (the only interpreter listed by `py -0`); the spec targets 3.12 compatibility |
| Globally installed | mkdocs 1.6.1, mkdocs-material 9.7.7, mkdocs-material-extensions 1.3.1, pymdown-extensions 12.1, Markdown 3.10.3, Pygments 2.21.0 |
| Node | v24.10.0 (available for optional browser-based checks; not a runtime dependency) |

**Conclusion:** greenfield implementation. Nothing needs migrating or overwriting.

**Compatibility issues and risks found**

1. Python 3.14 is installed, not 3.12. The template must be developed in an isolated venv and its dependencies pinned with ranges known to work on 3.12 as well. A 3.12 interpreter is not installed, so 3.12 can be claimed only as "compatible by constraint", not as tested.
2. Material for MkDocs 9.7.x is in maintenance mode and emits a notice about MkDocs 2.0 at build time. The template must cap `mkdocs<2` and must verify that the notice does not fail `--strict`.
3. Global packages are polluted with unrelated tools. The venv must be created fresh so `requirements.txt` reflects what the template actually needs.
4. No browser-automation tooling has been verified. Visual/UI checks may be limited to what can be exercised (see §11).
5. The repo has no `.gitignore`; `site/` and `.venv/` must be excluded.

## 1. Product Overview and Vision

A copy-and-go MkDocs + Material template for long-form educational and technical documentation. One shared, consistent look and feel; each subject (Java, C, C++, Python, JS, ML, Data Science, …) lives in its own repository cloned from this template, changing only metadata, navigation, content, and branding.

## 2. Problem Statement

Separate course sites drift apart in configuration, styling, math support, and deployment. Setting up light/dark mode, MathJax, code highlighting, and GitHub Pages deployment correctly each time is repetitive and error-prone.

## 3. Target Users and Use Cases

- **Author/maintainer** (primary): creates a new subject site from the template in minutes, writes Markdown only.
- **Learner/reader**: reads on desktop, tablet, or phone; searches; switches theme; copies code; reads equations.
- Use cases: programming tutorials, math-heavy ML notes, exercise sets, reference guides.

## 4. Goals and Non-Goals

**Goals:** reuse, consistency, minimal custom code, verified builds, GitHub Pages compatibility.

**Non-goals:** code execution engine; custom frontend framework; template engine or multi-repo framework; full course curricula; custom domain assumptions; external fonts/third-party services; performing any deployment or commit.

## 5. Functional Requirements

### Mandatory
| ID | Requirement |
|---|---|
| F1 | `mkdocs.yml` identifies site name, description, site URL, repo URL/edit links, palette, extensions, nav, extra CSS/JS, deployment-relevant settings, each with comments |
| F2 | Material theme with sidebar hierarchical nav, search, breadcrumbs (path), previous/next footer navigation |
| F3 | Palette with three preferences: System, Light, Dark, using Material's native `palette` toggle (three entries with `media` schemes) |
| F4 | MathJax: inline `$…$` / `\(…\)` and display `$$…$$` / `\[…\]`; fractions, scripts, sums, integrals, matrices, Greek, `aligned`; works inside admonitions and tabs; re-typesets on instant navigation if enabled |
| F5 | Code blocks with highlighting for Java, C, C++, Python, JS/TS, SQL, Bash, YAML, JSON; copy button; optional line numbers/titles/annotations |
| F6 | Admonitions, collapsible details, content tabs, tables, task lists, footnotes, abbreviations, definition lists |
| F7 | Mermaid diagrams via Material's native `pymdownx.superfences` custom fence. Mermaid JS is **vendored and pinned** (11.17.2, MIT) under `docs/assets/vendor/mermaid/`, not loaded from a CDN. Supported: flowchart, sequence, class, state, ER, Gantt, pie, Git graph, mindmap. Follows Light/Dark palette; `securityLevel: strict`; a malformed diagram must not break the page. Demo page `docs/examples/mermaid.md` |
| F8 | Sample content: landing page, getting started, sample curriculum page, programming example, inline + display LaTeX, admonition, table, tabs |
| F9 | Authoring patterns documented and demonstrated: learning objectives, concept, syntax, expected output, tutorial steps, exercises, common mistakes, summary |
| F10 | `README.md` covering install, local dev, customization, new-project creation, deployment |
| F11 | Validation report (`VALIDATION_REPORT.md`) listing tests actually run, results, and unverified items |
| F12 | `requirements.txt` with documented, bounded version ranges (`mkdocs>=1.6,<2`, `mkdocs-material>=9.5,<10`, `pymdown-extensions`) |
| F13 | `.gitignore` excluding `site/`, `.venv/`, caches |

### Optional enhancements (only if native and low cost)
- `navigation.instant`, `navigation.tracking`, `navigation.top`, `content.tabs.link`.
- GitHub Actions workflow template for Pages (provided as documentation/example only, not enabled).
- `examples/` directory of standalone source files outside `docs_dir`, optionally included via `pymdownx.snippets`.
- Social cards, blog plugin, versioning (`mike`) — explicitly deferred.

## 6. Non-Functional Requirements

- Maintainability: custom CSS ≲ 150 lines, custom JS limited to the MathJax config (+ one re-typeset hook).
- No unnecessary dependency upgrades; no `mkdocs>=2`.
- No external fonts: use system font stack (`theme.font: false`) to avoid Google Fonts requests; MathJax loaded from a pinned CDN URL (documented trade-off, see §13) unless vendoring is chosen in the technical design.
- Privacy: no analytics enabled by default.
- Build is reproducible and `mkdocs build --strict` is clean.

## 7. UI/UX Requirements

Per the task spec: responsive layouts; readable typography (line-height ~1.7, content width bounded); hierarchical nav; search; breadcrumbs and prev/next; clear heading hierarchy; wide tables scroll horizontally; styled code blocks with copy; visible focus indicators; sufficient contrast in both schemes; consistent admonition/tab/diagram styling; no clipped or overlapping content. Custom colours only via Material CSS variables (`--md-*`) or `[data-md-color-scheme]` selectors.

## 8. Technical Stack and Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Generator | MkDocs 1.6.x | Required baseline; installed 1.6.1 |
| Theme | Material for MkDocs 9.x (installed 9.7.7) | Native palette, search, copy, tabs |
| Extensions | `pymdownx.*` (arithmatex generic, superfences, tabbed, highlight, details, snippets, tasklist) + `admonition`, `attr_list`, `md_in_html`, `tables`, `footnotes`, `toc` | Native Material features |
| Math | MathJax 3 via `pymdownx.arithmatex` `generic: true` | Avoids Markdown mangling backslashes |
| Highlighting | Pygments via `pymdownx.highlight` | Build-time, no JS |
| Python | Venv; constraints compatible with 3.12 | Spec baseline |
| Separation | Shared look in `mkdocs.yml` + `docs/stylesheets` + `docs/javascripts`; subject content in `docs/` | Reuse by copy |

## 9. Proposed Directory Structure

```
mkdocs-template/
├── mkdocs.yml
├── requirements.txt
├── README.md
├── PRD.md
├── TECHNICAL_DESIGN.md
├── VALIDATION_REPORT.md
├── .gitignore
├── PROMPT_MKDOCS_TEMPLATE.md      (existing, untouched)
├── docs/
│   ├── index.md
│   ├── getting-started/index.md
│   ├── curriculum/index.md
│   ├── authoring/                 (patterns: code, math, admonitions/tabs/tables)
│   ├── stylesheets/extra.css
│   ├── javascripts/mathjax.js
│   ├── javascripts/mermaid-config.js
│   ├── examples/mermaid.md
│   └── assets/vendor/mermaid/    (pinned Mermaid + LICENSE + VERSION.txt)
└── examples/
    ├── programming/
    └── mathematics/
```

`examples/` sits outside `docs_dir` so it is not published unless snippet-included. No `overrides/` directory unless a feature truly requires it.

## 10. Accessibility and Responsive Requirements

- Keyboard navigable (Material default) with visible `:focus-visible` outlines in both schemes.
- Colour contrast ≥ WCAG AA (4.5:1 body text) in light and dark for custom colours.
- Layout verified at ~1280, ~768, ~375 px widths.
- Tables in `overflow-x: auto` containers; code blocks scroll horizontally; no page-level horizontal scroll.
- Respect `prefers-reduced-motion`; no decorative animation added.
- Images require alt text; equations remain readable by MathJax's assistive MathML.

## 11. Testing and Acceptance Criteria

| Area | Criterion | Method |
|---|---|---|
| Build | deps install in fresh venv; `mkdocs build --strict` exit 0, no warnings | CLI |
| Links | no broken internal links/anchors | strict build (`validation` config) |
| Output | `site/` contains pages, CSS, JS, search index | script/inspection |
| Math | delimiters survive in built HTML (`arithmatex` spans, no mangled `\\`) | HTML grep + browser if available |
| Math rendering | inline/display/aligned/matrix, inside admonition and tab | browser check |
| Theme | System/Light/Dark switching, contrast | browser check |
| Layout | desktop/tablet/mobile overflow check | browser at three widths |
| Interaction | search, copy button, tables | browser check |

If no browser automation is available, only the CLI/HTML checks are reported as passed; the rest are listed as **unverified** in `VALIDATION_REPORT.md`. No test is claimed without evidence.

## 12. Deployment Requirements

- Compatible with `mkdocs gh-deploy` (documented; **not executed**).
- README explains: create repo, set `site_url` and `repo_url`, `gh-deploy`, Settings → Pages → source `gh-pages` branch; project site URL form `https://<user>.github.io/<repo>/`.
- All asset paths relative so the site works under a sub-path; no custom domain (`CNAME`) assumed.
- No commit, push, or deploy is performed during this task.

## 13. Risks, Assumptions, Constraints

- **Mermaid (added later):** the vendored `mermaid.min.js` is ~3.5 MB and is only needed on pages with diagrams (Material mounts it lazily, but `extra_javascript` still downloads it on every page; accepted trade-off, cached after first load). Upgrading Mermaid is a manual file replacement. Known limitations: Gantt/sequence text is small on phone widths; Git-graph branch label colours are fixed for branches 0 and 1 only; Mermaid renders inside a closed shadow root, so page CSS cannot style diagrams (only CSS variables inherit).
- **GitHub API call:** when `repo_url` is set, Material fetches repository facts from `api.github.com` in the browser. Remove `repo_url` to avoid it.

- **Python 3.12 untested** (only 3.14 present) — mitigated by version ranges; stated in validation report.
- **MkDocs 2.0 notice / maintenance-mode Material** — cap `<2`; check strict build.
- **MathJax CDN** is a third-party load. Spec forbids unjustified third-party services; justification: MathJax is explicitly required and Material's documented approach. Alternative (vendoring into `docs/assets`) to be decided in the technical design; default is a pinned jsDelivr URL, documented, with offline vendoring instructions.
- Browser-based verification availability is unknown.
- Assumption: approval of this PRD is implied by the instruction to proceed to technical design; implementation will follow once both documents exist.

---

# Part B — Data Mining & Machine Learning Curriculum (Indonesian)

Status: Draft for approval · Date: 2026-10-18 · Part A above documents the reusable template and remains valid as the platform baseline. See `PROJECT_AUDIT.md` for findings.

## B.1 Vision and goals

An Indonesian-language, Python-based learning site on Data Mining / Machine Learning, built on the existing Material template and authored with Claude Code. Learners finish each chapter able to define a problem, build a reproducible pipeline, evaluate it with the *right* metric and validation scheme, and explain the result.

**Goals:** five chapters with one consistent structure; every code example runnable and tested; correct evaluation practice (no leakage); reuse of the existing theme.
**Non-goals:** deep learning, NLP/vision, MLOps platforms, an in-browser code runner, a new theme, publishing/deployment in this phase.

## B.2 Audience (assumed — to confirm)

Indonesian-speaking undergraduates/self-learners with basic Python and high-school-level statistics. Math is introduced intuitively first, formally second.

## B.3 Curriculum

| Part | Content |
|---|---|
| 0. Fondasi (taught once) | Data mining vs ML vs AI; the CRISP-DM-style workflow; environment setup; NumPy/pandas essentials; visualisation (Matplotlib/Seaborn); EDA toolkit; preprocessing toolkit (missing values, encoding, scaling, outliers) with `Pipeline`/`ColumnTransformer`; train/validation/test and cross-validation; leakage; reproducibility (seeds); model persistence (joblib); glossary |
| 1. Klasifikasi | Logistic regression, k-NN, Naive Bayes, decision tree, random forest, SVM; confusion matrix, precision/recall/F1, ROC-AUC, PR curve; class imbalance; tuning |
| 2. Estimasi (Regresi) | Linear/Ridge/Lasso, polynomial features, tree/forest regressors; MAE/MSE/RMSE/R²; residual analysis; regularisation; tuning |
| 3. Forecasting | Time-series anatomy (trend/seasonality), naive & seasonal-naive baselines, moving average, exponential smoothing, ARIMA/SARIMA, lag-feature ML; chronological split, rolling-origin CV; MAE/RMSE/MAPE/sMAPE/MASE |
| 4. Klaster | k-Means, hierarchical, DBSCAN (optionally GMM); scaling is essential; inertia/elbow, silhouette, Davies-Bouldin; cluster profiling; optional ARI when labels exist |
| 5. Aturan Asosiasi | Transactions, support/confidence/lift, Apriori, FP-Growth; threshold selection; rule pruning and interpretation |

## B.4 Shared chapter template (content requirement)

Every chapter follows: Tujuan pembelajaran → Konsep → Intuisi & matematika (collapsible where heavy) → Dataset → Implementasi bertahap (runnable) → Evaluasi → Optimasi → Kesalahan umum → Latihan (with collapsible solutions) → Ringkasan → Bacaan lanjutan. A chapter index shows prerequisites, estimated time, datasets, libraries and the workflow-applicability table (B.5).

## B.5 Workflow applicability (requirement)

Legend: **R** required · **O** optional · **X** not applicable · **D** applicable but different. The 12-stage workflow must not be forced uniformly.

| # | Stage | Klasifikasi | Estimasi | Forecasting | Klaster | Asosiasi |
|---|---|---|---|---|---|---|
| 1 | Problem definition | R | R | R (horizon, frequency) | R (goal is exploratory) | R (business question) |
| 2 | Data collection | R | R | R (regular timestamps) | R | R (transaction format) |
| 3 | EDA | R | R | **D** (plots over time, decomposition, ACF) | R | **D** (item frequencies, basket size) |
| 4 | Preprocessing | R | R | **D** (resample, gap fill, no random imputation) | R (**scaling mandatory**) | **D** (one-hot transaction matrix; dedupe) |
| 5 | Feature engineering | O | O | R for ML approach (lags, calendar); X for classical | O (selection/PCA) | X (O: item grouping) |
| 6 | Data splitting | R (stratified) | R | **D** (chronological, rolling origin; never shuffle) | **X** (no test set; stability checks instead) | **X** (no split; optional hold-out to check rule stability) |
| 7 | Model selection | R | R | R (baseline first) | R | R (algorithm + thresholds) |
| 8 | Training | R | R | R | R (unsupervised `fit`) | R (mining) |
| 9 | Prediction & evaluation | R (confusion matrix, F1, AUC…) | R (MAE, RMSE, R², residuals) | **D** (MAE, RMSE, sMAPE, MASE vs baseline) | **D** (silhouette, DB, profiling; ARI only with labels) | **D** (support, confidence, lift; no "accuracy") |
| 10 | Optimization | R (grid/random search, CV) | R | R (time-aware CV) | O (choose k/eps) | O (min-support/confidence sweep) |
| 11 | Model saving | R | R | R | O | O (save rules table) |
| 12 | Deployment & monitoring | O (concept + simple example) | O | O (retraining cadence) | O | O |

## B.6 Technical requirements

- Python (version per audit decision), NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Jupyter — **required**.
- Justified additions: `statsmodels` (ARIMA/ETS/decomposition/ACF), `mlxtend` (Apriori/FP-Growth), `scipy` (statistics), `joblib` (persistence; ships with scikit-learn), `jupytext`/`nbconvert` (notebook tooling, dev-only), `pytest` (example tests, dev-only).
- Datasets: scikit-learn bundled sets and seeded synthetic generators first; small committed CSVs only with documented source and licence; no runtime downloads required to follow a lesson.
- Reproducibility: fixed `random_state`, version-stamped output, pinned dependency ranges in `requirements-ml.txt`.

## B.7 Functional requirements (curriculum)

| ID | Requirement |
|---|---|
| C1 | Five chapters + foundation part, Indonesian, navigable via sidebar sections with index pages |
| C2 | Each code listing is included from `examples/` (single source) and has a recorded expected output |
| C3 | Every chapter has ≥ 3 graded exercises (basic / applied / challenge) with solutions in collapsible blocks |
| C4 | Every evaluation section names the metric, why it fits, and when it misleads |
| C5 | Pipelines never fit preprocessing on test data; leakage is demonstrated once and warned about |
| C6 | Mermaid used for the workflow overview in each chapter; math via existing MathJax |
| C7 | Downloadable notebooks per lesson (derived) |
| C8 | Glossary (ID↔EN) and term policy |
| C9 | `mkdocs build --strict` clean; all examples pass their smoke test |

## B.8 Acceptance criteria

| Area | Criterion | Method |
|---|---|---|
| Build | `mkdocs build --strict` exit 0 | CLI |
| Examples | every `examples/**/*.py` runs from a clean venv in < 60 s with fixed seed and exits 0 | test runner |
| Output drift | printed output matches committed expected-output blocks (numeric tolerance) | test script |
| Metrics | no chapter reports accuracy alone for imbalanced data; forecasting always compared with a naive baseline | review checklist |
| Leakage | scalers/encoders only inside `Pipeline` or fitted on train | review + test |
| Structure | each chapter has every section in B.4 | lint script checking headings |
| Language | prose Indonesian; identifiers English; glossary terms consistent | review |
| Rendering | math, Mermaid, plots legible in light/dark at 1280/768/375 px | browser check |
| Assets | no dataset > 5 MB in Git; each dataset has a source+licence line | script/review |
| Safety | nothing committed/pushed/deployed without an explicit request | process |

## B.9 Risks

See `PROJECT_AUDIT.md` §5 (G1–G17). Highest: ML wheel availability on Python 3.14, example rot, notebook strategy, pedagogical leakage errors.
