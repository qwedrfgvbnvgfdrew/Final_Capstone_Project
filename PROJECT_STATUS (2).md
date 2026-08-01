# Project Status

**Project:** Student Learning Segmentation (EDU-01)
**Track:** Track 2 — Field-Based Scenario (EdTech)
**Student:** Nilufar Azimjonova
**Status:** 🟢 Green Zone — mentor-required correction applied and verified

---

## 1. Where this stands right now

The project was reviewed once by the mentor (🟡 Yellow Zone) with one mandatory
conceptual correction: the clustering model was using demographic/context
columns (gender, nationality, grade level, course topic, parent-related
fields) alongside behavioral ones, even though the project was framed as
pure behavioral segmentation.

**That correction has been made and retrained.** The clustering model now
uses **behavior only** — attendance plus the four engagement counts.
Demographic fields are used strictly *after* clustering, for interpretation
and a fairness audit, never as model inputs. See `reports/results.md` for
the full before/after write-up.

## 2. One obvious entry point

Start here: **`demo.ipynb`** (Colab-first). It installs dependencies, loads
the saved model, predicts a cluster for a new student using behavior only,
and demonstrates invalid-input handling. No other setup is required to see
the project work end to end.

## 3. Deliverables checklist

| Deliverable | Status | Where |
|---|---|---|
| Problem definition & ML framing | ✅ Done | `README.md` |
| EDA | ✅ Done | `notebooks/01_eda...ipynb` |
| Preprocessing (behavior-only, leakage-safe) | ✅ Done | `src/preprocessing.py`, `src/data.py` |
| Naive + simple baselines | ✅ Done | `src/train.py`, `reports/results.md` §1 |
| ≥2 algorithms compared (K-Means, Agglomerative, DBSCAN) | ✅ Done | `src/train.py`, `reports/results.md` §3 |
| Experiment tracking | ✅ Done | MLflow runs logged during `src/train.py`; summarized in `reports/experiment_results.csv` |
| Final model trained & justified | ✅ Done | `models/final_kmeans_model.joblib`, `reports/results.md` §2 |
| Evaluation on held-out/unseen data | ✅ Done | `src/evaluate.py`, `reports/test_evaluation.json` |
| Error analysis | ✅ Done | `reports/borderline_students_test.csv` |
| Inference function + input validation | ✅ Done | `src/predict.py` |
| Reproducible Colab demo | ✅ Done | `demo.ipynb` |
| Responsible AI / fairness analysis | ✅ Done | `reports/results.md` §8, post-hoc gender/nationality/grade audit |
| Documentation (README, data card, model card) | ✅ Done | `README.md`, `data/README.md`, `models/README.md` |
| Pinned exact library versions | ✅ Done | `requirements.txt` (6 of 9 packages verified by re-running the pipeline; 3 flagged as best-known-good, see file comments) |
| LMS submission file | ✅ Done | `submission/Capstone Submission Form - Nilufar Azimjonova.docx` |
| Presentation slides for defense | ✅ Done | prepared separately for the live defense (not part of the LMS submission) |

## 4. Terminology mapping (for anyone checking against a generic repo-structure reference)

Some course reference materials use different folder/file names than this
repository does. Same concepts, different labels — nothing is missing:

| Reference term | This repository |
|---|---|
| `artifacts/` (saved model outputs) | `models/` — see `models/README.md`, titled "Saved Model Artifacts" |
| `src/inference.py` | Both exist — `src/inference.py` is a thin alias re-exporting `src/predict.py` (the real logic, used everywhere else in this repo) |
| `notebooks/03_demo.ipynb` | `demo.ipynb` at repo root |
| `reports/results/` (folder) | `reports/results.md` (file) |

## 5. Key metrics snapshot

| Metric | Value |
|---|---|
| Final model | K-Means, k=4, seed=42, behavior-only features |
| Silhouette (train, n=384) | 0.3038 |
| Silhouette (unseen test, n=96) | 0.3031 |
| Stability (mean pairwise ARI, 5 seeds) | 0.982 |
| Poorly-fit test students (negative silhouette) | 5.2% |
| Borderline test students (smallest 10% centroid gap) | 10 |

Full detail, including the honest disclosure that k=2 scores higher on pure
silhouette than the selected k=4, is in `reports/results.md`.



- [ ] Clean up the file-naming leftovers noted above
- [ ] Confirm the repository is public (or mentors have explicit access)
- [ ] Do one final live run-through of `demo.ipynb` in a fresh Colab session
