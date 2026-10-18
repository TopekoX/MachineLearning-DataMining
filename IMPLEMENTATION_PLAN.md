# Implementation Plan — Data Mining & ML Curriculum

Date: 2026-10-18 · Depends on `PROJECT_AUDIT.md`, `PRD.md` Part B, `TECHNICAL_DESIGN.md` Part B.
Each phase ends with a review gate; no phase starts without approval. Commit/push/deploy happen only on explicit request.

## Phase 0 — Decisions & environment (needs user approval)

- Resolve audit questions Q1–Q10 (`PROJECT_AUDIT.md` §8), especially repo/remote, notebook strategy, Python version.
- Create `.venv`; install `requirements.txt`; create `requirements-ml.txt` / `requirements-dev.txt`; verify ML wheels on the chosen Python (`pip check`).
- Create `CLAUDE.md` (language policy, chapter template, commands, "never commit unless asked", example-first rule).
- Re-run `mkdocs build --strict` on the untouched template as a baseline.
- **Exit:** clean venv, baseline build green, decisions recorded.

## Phase 1 — Skeleton & identity

- `mkdocs.yml`: site identity, `theme.language: id` (verify), full `nav` with placeholder-free pages only for what exists.
- Rewrite `docs/index.md`; create `panduan/`, chapter folders with `index.md` (objectives, prerequisites, applicability table, Mermaid roadmap).
- Move template reference pages (`authoring/`, `examples/`) to bottom of nav; delete nothing.
- Add `tools/lint_chapters.py`, `tools/run_examples.py`, `tests/`.
- **Exit:** strict build green; navigation reviewed in browser.

## Phase 2 — Fondasi

- Write foundation lessons + examples (`examples/fondasi/`), glossary, dataset sources, leakage demo, CV, persistence.
- Prove the toolchain end-to-end on one pilot lesson (script → expected output → figure → notebook → page).
- **Exit:** pilot lesson passes all quality gates; template pattern approved before scaling.

## Phase 3 — Klasifikasi (reference chapter)

- Full chapter incl. metrics, imbalance, tuning, saving, exercises. This chapter fixes the pattern for the rest.
- **Exit:** gates green; content review of Indonesian prose and metric correctness.

## Phase 4 — Estimasi, then Phase 5 — Klaster

- Reuse Fondasi + Klasifikasi pattern; keep differences explicit per applicability matrix.

## Phase 6 — Forecasting (highest conceptual risk)

- Chronological split, rolling origin, baselines first; `statsmodels` verified on the chosen Python.

## Phase 7 — Aturan Asosiasi

- Synthetic transactions, Apriori/FP-Growth (`mlxtend` compatibility check), rule pruning.

## Phase 8 — Hardening & release prep

- Full browser validation (3 widths × 2 schemes), link/anchor pass, Indonesian search check, glossary consistency, accessibility spot checks, page-weight review.
- Update `README.md`, append results to `VALIDATION_REPORT.md`.
- Optional: GitHub Actions workflow (build + example tests); **deploy only on explicit request**.

## Order rationale

Classification first because it establishes metrics, pipelines and splitting that Estimation and Clustering reuse; Forecasting after Estimation because lag-feature ML builds on regression; Association last because it is the most independent.

## Testing summary

| Gate | Command |
|---|---|
| Docs build | `mkdocs build --strict` |
| Examples | `python tools/run_examples.py` (or `pytest`) |
| Structure | `python tools/lint_chapters.py` |
| Notebooks (dev) | `jupytext` conversion + `nbconvert --execute` smoke |
| Browser | per-chapter spot check, light/dark, 1280/768/375 px |

## Per-chapter Definition of Done

All sections of PRD B.4 present · every code listing sourced from tested `examples/` file · expected output recorded · ≥ 3 exercises with solutions · metric choice justified and failure modes noted · no leakage · Indonesian prose reviewed · glossary updated · strict build and example tests green.

## Risks to track

Python 3.14 wheel gaps (G2) · example rot (G3) · notebook diffs (G4) · version-dependent numeric output (TD B.6) · scope creep into deep learning.
