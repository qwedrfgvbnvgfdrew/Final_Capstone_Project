# EDU-01 — Student Learning Segmentation

**Student:** Nilufar Azimjonova
**Track:** Track 2 — Field-Based Scenario (EdTech, Scenario EDU-01: Student Learning Segmentation)
**Project brief:** the official scenario brief is included in this repository as
[`FIELD-BASED_CAPSTONE_SCENARIO.pdf`](FIELD-BASED_CAPSTONE_SCENARIO.pdf); every
design decision below traces back to that document.

**Repository URL:** https://github.com/qwedrfgvbnvgfdrew/Final_Capstone_Project.git

---

## 1. Problem Statement

An online education platform gives every student the same generic support,
regardless of how differently they actually behave in the course — some
students engage constantly (raising hands, visiting resources, joining
discussions), others barely engage at all. Without a way to group students by
behavior, the platform's educators cannot target interventions, and student
disengagement goes unnoticed until it shows up in final grades. The goal of
this project is to **discover meaningful, interpretable learner groups from
behavioral data alone**, so a student-success team can act on engagement
patterns before they become poor outcomes.

## 2. Selected Project Track

**Track 2 — Field-Based Scenario**, EdTech domain, Scenario EDU-01 (Student
Learning Segmentation), as defined in `FIELD-BASED_CAPSTONE_SCENARIO.pdf`.

## 3. Dataset Source

- **xAPI-Edu-Data** (Students' Academic Performance Dataset), 480 students, 17 columns.
- Source: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- License: CC BY-SA 4.0.
- Full data documentation, including the exact columns used and why
  `PlaceofBirth` and `Class` are excluded from clustering: see
  [`data/README.md`](data/README.md).

## 4. ML Task Type

**Unsupervised learning — clustering.** There is no ground-truth label for
"learner type"; the model discovers natural groups of students from
**behavioral inputs only**: the 4 numeric engagement counts (raised hands,
resources visited, announcements viewed, discussion posts) plus
`StudentAbsenceDays` (attendance, also a behavior signal), using K-Means as
the final algorithm. **Demographic/contextual columns (gender, nationality,
grade, section, topic, semester, parent-related fields) are explicitly
excluded from the clustering inputs** — see Section 4a below for why, and
`src/data.py` (`CLUSTERING_FEATURES` vs. `DEMOGRAPHIC_CONTEXT_FEATURES`) for
the exact split. The `Class` academic-performance label exists in the data
but is deliberately **excluded from training** and used only afterward, to
sanity-check whether the discovered behavioral clusters relate to real
academic outcomes.

- **Input at inference time:** one student's behavioral fields (the 4 counts
  + `StudentAbsenceDays`) are *required*; demographic fields are *optional*
  and, if supplied, are never used to compute the assignment — see Section
  10, "Example Input and Output," for the exact schema.
- **Output:** a cluster ID (0–3), a human-readable learner-profile name
  (e.g. "At-Risk Learners"), a recommended intervention, a confidence note,
  and (if demographic fields were supplied) a `demographic_context` echo for
  fairness reporting.

### 4a. Feature-scope correction (mentor-flagged issue, now fixed)

An earlier version of this project fit the clustering model on the 4
behavioral counts **plus 11 demographic/contextual categorical columns**
(gender, nationality, grade, section, topic, semester, parent fields,
etc.), even though the project was framed as "segmentation based on
learning behavior." A mentor review correctly flagged that this let cluster
assignment be influenced by who a student is, not just how they behave, and
that a warning in the Responsible AI section alone wasn't a sufficient fix.

**What changed:** `build_preprocessor()` in `src/preprocessing.py` now only
ever sees `CLUSTERING_FEATURES` (the 4 counts + `StudentAbsenceDays`).
Demographic columns are a completely separate list
(`DEMOGRAPHIC_CONTEXT_FEATURES`) that is never passed to the
`ColumnTransformer` and is joined back onto cluster labels only *after*
`.fit_predict()`/`.predict()`, for interpretation and the fairness check
described in `reports/results.md`, Section 7. Every model, metric, figure,
and report in this repository was regenerated from that corrected pipeline.

## 5. Project Pipeline / System Architecture

```
data/xAPI-Edu-Data.csv
        │
        ▼
 src/data.py            → load + schema validation + data-quality checks
        │
        ▼
 src/preprocessing.py   → 80/20 train/test split (random, fixed seed)
                           ColumnTransformer over BEHAVIORAL inputs ONLY:
                             OneHotEncoder (StudentAbsenceDays)
                             + StandardScaler (4 engagement counts)
                           demographic/context columns are NEVER passed in
                           fit ONLY on train
        │
        ▼
 src/train.py            → naive baseline (random labels)
                           → simple baseline (K-Means, k=2)
                           → K-Means grid k=2..8 (Elbow + Silhouette), logged to MLflow
                           → Agglomerative (Ward) and DBSCAN comparisons, logged to MLflow
                           → 5-seed stability check (Adjusted Rand Index)
                           → final model selection (k=4, justified in reports/results.md)
                           → saves models/*.joblib + models/model_metadata.json
                           → saves reports/figures/*.png
        │
        ▼
 src/evaluate.py         → loads saved artifacts
                           → evaluates on the held-out 20% test split ("unseen students")
                           → error analysis: borderline / poorly-fit students
                           → saves reports/test_evaluation.json, reports/borderline_students_test.csv
        │
        ▼
 src/predict.py          → predict_student_cluster(dict) -> cluster + name + recommendation
                           → input validation (missing fields, wrong types, out-of-range values)
        │
        ▼
 demo.ipynb (Colab)      → clones/installs, runs the pipeline end-to-end,
                           demonstrates valid AND invalid inference inputs
```

Directory layout:

```
capstone-project/
├── README.md                      <- this file
├── requirements.txt
├── .gitignore
├── FIELD-BASED_CAPSTONE_SCENARIO.pdf   <- official scenario brief
├── demo.ipynb                      <- Colab-first, reproducible, end-to-end demo
├── data/
│   ├── xAPI-Edu-Data.csv
│   └── README.md                   <- dataset documentation
├── notebooks/
│   ├── 01_eda.ipynb                <- exploratory data analysis
│   └── 02_experiments.ipynb        <- full modeling/experiment walkthrough
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── preprocessor.joblib
│   ├── final_kmeans_model.joblib
│   ├── pca_projector.joblib
│   ├── model_metadata.json
│   └── README.md
├── reports/
│   ├── results.md                  <- full write-up of every experiment and metric
│   ├── experiment_results.csv
│   ├── test_evaluation.json
│   ├── borderline_students_test.csv
│   └── figures/
│       ├── elbow_plot.png
│       ├── silhouette_vs_k.png
│       └── pca_clusters_train.png
├── mlflow.db                       <- (generated locally by src/train.py; not committed)
├── mlruns/                         <- (generated locally by src/train.py; not committed)
└── submission/
    └── Submission_Details.docx     <- LMS submission file
```

> **Terminology note:** some course reference materials use the folder name
> `artifacts/` for saved model outputs (the trained model, preprocessor, and
> metadata). In this repository, that same content lives in **`models/`** —
> same purpose, different name. See [`models/README.md`](models/README.md)
> for exactly what each file is.

## 6. Models / Approaches Tested

| Model | Role |
|---|---|
| Random cluster assignment | Naive baseline — confirms real structure exists (near-zero silhouette) |
| K-Means, k=2 | Simple baseline |
| K-Means, k=2..8 | Main experiment grid (Elbow Method + Silhouette Score) |
| Agglomerative Clustering (Ward linkage), k=4 | Comparison approach |
| DBSCAN, eps grid 0.5–3.0 | Comparison approach |
| **K-Means, k=4 (final)** | **Selected final model** |

All experiments are logged to MLflow (SQLite backend at `mlflow.db`, model
artifacts under `mlruns/`) whenever `python -m src.train` is run. These two
are **not** committed to the repository — every run creates a fresh,
timestamped model snapshot under `mlruns/`, so committing it would mean an
ever-growing pile of near-duplicate folders. Instead:
- **Evidence of every experiment** (parameters + metrics for every model
  tried) is committed in the human-readable
  [`reports/experiment_results.csv`](reports/experiment_results.csv).
- To inspect the live MLflow UI yourself, just re-run `python -m src.train`
  locally, then: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## 7. Final Model and Justification

**Final model: K-Means, k=4, random_state=42, fit on behavioral inputs only.**

This is not the numerically "best" clustering by silhouette score alone —
that would be k=2 (silhouette 0.3735 vs. 0.3038 at k=4). **This is not a
claim that the data "naturally" forms four clusters** — k=4 was deliberately
selected, despite being statistically weaker than k=2, because:

1. The scenario brief's functional requirements and expected deliverables
   explicitly call for **four named, actionable learner profiles** (Highly
   Engaged / Consistent / Struggling but Active / At-Risk Learners) that
   educators can act on — a 2-cluster split is too coarse to route four
   different intervention strategies.
2. K-Means is the only one of the three algorithms compared that supports
   `.predict()` on brand-new, unseen students without retraining — essential
   for the "assign a new student" requirement in the brief.
3. It is highly stable: mean pairwise Adjusted Rand Index of **0.9818** across
   5 different random seeds.
4. Agglomerative Clustering at k=4 scored close but slightly lower
   (silhouette 0.2927 vs. 0.3038). DBSCAN could not find a usable,
   low-noise global clustering at any tested density threshold — see
   `reports/results.md`, Section 3, for the exact numbers.

Full quantitative justification, including the honest disclosure that k=2 has
the best pure silhouette score, is in [`reports/results.md`](reports/results.md).

## 8. Evaluation Metrics and Results

| Metric | Train (n=384) | Test / unseen (n=96) |
|---|---|---|
| Silhouette Score | 0.3038 | 0.3031 |
| Calinski-Harabasz Index | 214.85 | 53.19 |
| Davies-Bouldin Index | 1.2620 | 1.2060 |

- **Silhouette Score** measures how well-separated and internally cohesive
  clusters are (higher is better); it is the primary metric because it works
  without ground-truth labels, exactly as the clustering task requires.
- **Calinski-Harabasz** and **Davies-Bouldin** are reported as secondary,
  corroborating internal-validation metrics.
- **Stability**: mean pairwise Adjusted Rand Index across 5 seeds = **0.9818**.
- **Post-hoc validation against `Class`** (never used in training): the
  At-Risk Learners cluster contains 2.9% high-performing and the highest
  share (64.7%) of low-performing students on the unseen test split; Highly
  Engaged and Consistent Learners contain 0% low-performing students. Full
  crosstab in `reports/results.md`, Section 6.
- **Post-hoc fairness check** (demographics, never used in training): see
  `reports/results.md`, Section 7 — a gender skew was found in the At-Risk
  cluster and is flagged for human follow-up rather than hidden.

Full breakdown, all experiment numbers, and the complete cluster-profile
table are in **[`reports/results.md`](reports/results.md)** — read this file
for the complete evaluation write-up.

## 9. Installation Instructions

```bash
git clone <PASTE-YOUR-GITHUB-REPO-URL-HERE>
cd capstone-project
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## 10. Training Instructions

Everything below is also demonstrated, cell by cell, in `demo.ipynb` and
`notebooks/02_experiments.ipynb` — you do not have to use the command line if
you prefer Colab.

```bash
# From the repository root:
python -m src.train      # fits preprocessing + all clustering models,
                          # logs every experiment to MLflow, saves the
                          # final model + figures + metadata
python -m src.evaluate   # evaluates the saved final model on the held-out
                          # test split and runs the error analysis
```

Both scripts print their results to the console and also save them to
`reports/` and `models/` so nothing is lost if you close the terminal.

To inspect the logged MLflow experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 11. Demo and Inference Run Instructions (Colab-first)

Open **`demo.ipynb`** in Google Colab (or Jupyter). It is self-contained: it
installs dependencies, loads the dataset and saved model directly from this
repository, and demonstrates:

1. Loading the saved preprocessing pipeline and final K-Means model.
2. Assigning a **new, previously unseen** student to a learner segment.
3. Rejecting an **invalid** student record (missing field, wrong type,
   out-of-range value) with a clear error message instead of crashing.
4. Visualizing the four discovered learner segments in PCA space.

No application or API layer is used for this project — the Colab notebook is
the demo/inference interface, as permitted by the scenario brief.

## 12. Example Input and Output

**Input** (a dict passed to `src.predict.predict_student_cluster`). Only the
5 behavioral fields are *required*; demographic fields are *optional* and
never influence the assignment — see Section 4a:

```python
{
    # Required (behavioral):
    "raisedhands": 55,
    "VisITedResources": 60,
    "AnnouncementsView": 30,
    "Discussion": 40,
    "StudentAbsenceDays": "Under-7",
    # Optional (demographic/context) -- NOT used to compute the cluster,
    # only echoed back under "demographic_context" for fairness reporting:
    "gender": "F",
    "NationalITy": "Jordan",
}
```

**Output:**

```python
{
    "cluster_id": 3,
    "cluster_name": "Consistent Learners",
    "recommendation": "Maintain regular learning support and encourage continued engagement.",
    "confidence_note": "Moderate confidence in this cluster assignment (centroid gap in the middle third of the training distribution).",
    "demographic_context": {"gender": "F", "NationalITy": "Jordan"},
}
```

**Invalid input example** (`raisedhands` as a string instead of a number):

```python
predict_student_cluster({..., "raisedhands": "a lot"})
# Raises: InvalidStudentInputError: Field 'raisedhands' must be numeric
# (0-100 scale in the original dataset), got 'a lot' (<class 'str'>).
```

## 13. Known Limitations

- **k=4 is a business-interpretability choice, not the statistically
  tightest clustering, and is not a claim that four learner types "naturally"
  exist** — silhouette score is actually highest at k=2 on this dataset
  (0.3735 vs. 0.3038 at k=4). This is disclosed, not hidden (see
  `reports/results.md`).
- **Small dataset**: 480 students from one platform; clusters may not
  generalize to other institutions or LMS platforms.
- **DBSCAN did not find a usable, low-noise global clustering** of this
  feature space at any tested density threshold — most points were labeled
  noise, or eps was loose enough that everything collapsed into one cluster.
  This suggests the behavioral feature space does not have naturally
  density-separated clusters (K-Means/Agglomerative's centroid-based
  assumption fits this data better than density-based clustering).
- **Borderline students exist**: on the held-out test set, 5.2% of students
  have a negative per-sample silhouette (poorly fit to their assigned
  cluster), and 10 students are essentially equidistant between two clusters.
  `predict_student_cluster` surfaces this via a `confidence_note`, using
  thresholds computed from the training centroid-gap distribution (not
  manually chosen), rather than presenting every assignment as equally
  certain.
- **Hard cluster assignment**: K-Means gives every student exactly one label,
  even though real learning behavior is a spectrum. A soft-clustering method
  (e.g. Gaussian Mixture Models) could represent partial membership, and is a
  natural direction for future work.

## 14. Responsible AI Considerations

- **Bias / fairness:** demographic fields (`gender`, `NationalITy`, grade,
  topic, parent fields, etc.) are **excluded from the clustering model
  entirely** — see Section 4a. They are used only *after* clustering, in a
  concrete, regenerated-every-run fairness check
  (`src.train.fairness_breakdown`, `reports/results.md` Section 7 /
  `reports/test_evaluation.json`) that cross-tabulates every discovered
  cluster against every demographic column. That check is not just a
  disclaimer: on the current test split it surfaced a real gender skew in
  the At-Risk Learners cluster (79.4% male vs. a 63.5% male dataset
  baseline) that should be reviewed by an educator/domain expert before this
  tool is used operationally, to rule out systemic causes unrelated to
  genuine behavior (e.g., access differences, reporting artifacts).
- **Privacy:** the dataset is fully anonymized, publicly available under
  CC BY-SA 4.0, and contains no personally identifiable information. No new
  personal data is collected by this project.
- **Appropriate use:** cluster assignments are meant to **support** an
  educator's judgment (e.g., flag a student for outreach), never to
  **replace** it or to make automated decisions about grading, admission, or
  discipline. The `Class` academic-performance label is used only to
  interpret clusters after the fact — it is never fed into the model, and the
  model's output should never be reframed as a prediction of a student's
  grade.
- **Misuse risk:** this is a prototype trained on 480 students from one
  platform; deploying it operationally on a different, larger, or more
  diverse student population without re-validating cluster quality and
  fairness would be inappropriate.

## 15. Colab Setup Troubleshooting

`demo.ipynb` supports two ways to get the project into Colab:

- **Option A (recommended):** run the first code cell as-is — it prompts you
  to upload `capstone-project.zip` directly and extracts it in the Colab
  session. No GitHub required.
- **Option B:** if you have a working GitHub repo, comment in the `git clone`
  line in the second code cell instead.

**Common cause of errors:** uploading files to GitHub one at a time (or
dragging loose files instead of the whole folder) silently drops the
`src/`, `data/`, `models/`, and `reports/` subfolder structure, and can even
cause same-named files (e.g. multiple `requirements.txt`) to overwrite each
other. If that happens, `demo.ipynb`'s Step 2 ("Auto-repair the folder
layout") detects flattened files and reconstructs the correct structure
automatically — you do not need to manually fix your repo for the demo to
run. To fix the GitHub repo itself for submission, delete its contents and
re-upload by **dragging the entire extracted `capstone-project` folder**
(not individual files) onto GitHub's "Add file → Upload files" drop zone, or
push it with `git` from a local clone.

## 16. Reproducibility Notes

- All random seeds are fixed (`random_state=42` for the train/test split and
  the final model; stability is additionally checked across seeds
  `[0, 1, 2, 3, 42]`).
- The train/test split is a **plain random 80/20 split**: each row is one
  independent student record with no time ordering or repeated-user
  structure to respect, so a random split is the logically correct and
  simplest strategy here.
- No leakage: the `Class` outcome column and the redundant `PlaceofBirth`
  column are dropped by the `ColumnTransformer` itself (`remainder="drop"`)
  before any model ever sees the data; the preprocessing pipeline (encoder +
  scaler) is fit **only** on the train split and only ever `.transform()`-ed
  on the test split.
- Every script (`src/train.py`, `src/evaluate.py`, `src/predict.py`) can be
  re-run from a clean clone of this repository with no hidden local state —
  the only inputs are `data/xAPI-Edu-Data.csv` (included) and the Python
  packages in `requirements.txt`.
