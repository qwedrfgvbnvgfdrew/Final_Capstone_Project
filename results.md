# Results & Interpretation — EDU-01 Student Learning Segmentation

All numbers on this page come directly from running `python -m src.train`
followed by `python -m src.evaluate` (see `models/model_metadata.json`,
`reports/experiment_results.csv`, and `reports/test_evaluation.json` for the
raw machine-readable versions of everything below).

## Feature-scope correction (mentor feedback)

**Earlier version of this project fit the clustering model on 11 categorical
demographic/context columns (gender, nationality, grade, topic, parent
fields, ...) plus the 4 behavioral counts.** A mentor review correctly
flagged that this contradicts the project's own framing ("segmentation based
on learning behavior") and could let cluster assignment be driven by who a
student is rather than how they behave.

**Fix applied:** the clustering model (`build_preprocessor()` /
`CLUSTERING_FEATURES` in `src/data.py`) now uses **only**:

- `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`
  (standardized numeric counts)
- `StudentAbsenceDays` (one-hot encoded; a behavioral/attendance signal, not
  a demographic one)

Every demographic/contextual column (`gender`, `NationalITy`, `StageID`,
`GradeID`, `SectionID`, `Topic`, `Semester`, `Relation`,
`ParentAnsweringSurvey`, `ParentschoolSatisfaction` —
`src.data.DEMOGRAPHIC_CONTEXT_FEATURES`) is now **excluded from the
ColumnTransformer entirely** and is joined back onto the cluster labels only
**after** `.fit_predict()` / `.predict()`, purely for interpretation and the
fairness check in Section 4 below. All numbers on this page were regenerated
from that corrected pipeline — none of the pre-correction numbers survive on
this page.

Data split: 480 students → **384 train / 96 test** (80/20 random split,
`random_state=42`). The preprocessing pipeline (one-hot + scaling on the 5
behavioral inputs only) and every clustering model were fit **only** on the
384-student train split; the 96-student test split was held out and only
ever `.transform()`-ed and `.predict()`-ed, never used to fit anything.

## 1. Baselines

| Model | k | Silhouette (train) |
|---|---|---|
| Naive baseline — random cluster assignment | 4 | **-0.0361** |
| Simple baseline — K-Means, minimal k | 2 | **0.3735** |

The random baseline's silhouette near zero confirms there is genuine,
non-trivial cluster structure in the behavioral data. Note that, compared to
the old (demographics-included) baseline of 0.200, K-Means on
behavior-only features finds a **much cleaner** structure (0.3735) — a
direct, measurable consequence of removing the ~40+ mostly-uninformative
one-hot demographic columns that were previously diluting the distance
metric.

## 2. Model search: K-Means, k = 2 to 8 (behavioral features only)

| k | Silhouette (train) | Inertia |
|---|---|---|
| **2** | **0.3735** | 970.3 |
| 3 | 0.3332 | 732.9 |
| **4** | **0.3038** | 636.6 |
| 5 | 0.2762 | 569.2 |
| 6 | 0.2845 | 511.8 |
| 7 | 0.2818 | 468.0 |
| 8 | 0.2985 | 414.6 |

See `reports/figures/elbow_plot.png` and `reports/figures/silhouette_vs_k.png`.

**Honest disclosure:** the numerically best K-Means solution by silhouette
alone is **k=2** (0.3735), not k=4. This is reported transparently, exactly
as it was before the feature-scope correction — the correction changed the
input features and therefore the absolute numbers, but did **not** change
the fact that k=2 statistically outperforms k=4 on this dataset.

### Why k=4 was still selected as the final model

k=4 (silhouette 0.3038) is an **18.7% relative drop** from the k=2 optimum
(down from a 32.2% drop under the old, demographics-included pipeline — the
gap between the "statistically best" and "chosen" k actually narrowed after
removing demographics). That drop is deliberately accepted because:

- The Capstone Brief's functional requirements and expected deliverables
  explicitly call for **four named, actionable learner profiles** (Highly
  Engaged / Consistent / Struggling but Active / At-Risk), not the minimum
  mathematically-tightest split.
- The brief's own criterion for cluster usefulness (Section 11, Q4) is
  interpretability and actionable interventions, not silhouette maximization
  alone.
- **This is explicitly NOT a claim that the dataset "naturally" contains
  four clusters.** Choosing k=4 is a documented business/interpretability
  decision made *despite* k=2 being statistically preferable — it is not
  presented as ground truth about how many learner types exist.

This trade-off is a limitation, not a hidden flaw — see Section 7.

## 3. Comparison models (at k=4, behavioral features only)

| Model | Silhouette (train) | Notes |
|---|---|---|
| K-Means (final) | 0.3038 | Selected — supports `.predict()` on new students |
| Agglomerative Clustering (Ward linkage) | 0.2927 | Close to K-Means; no native `.predict()` for new points |
| DBSCAN (best config found, eps=0.5, min_samples=5) | 0.2938* | *215 of 384 train points (56.0%) were labeled noise at this setting, split across 9 small clusters. Looser eps values (1.5+) collapsed everything into a single cluster (silhouette undefined). DBSCAN did not find a usable, low-noise global clustering of this feature space at any tested eps. |

**Conclusion:** K-Means remains the best-separated *usable* clustering and
the only one of the three that supports assigning new, unseen students
without retraining — this is why it is the model saved to
`models/final_kmeans_model.joblib`.

## 4. Cluster stability

The final K-Means (k=4) was re-fit with 5 different random seeds
(`[0, 1, 2, 3, 42]`). The mean pairwise Adjusted Rand Index (ARI) across all
seed pairs was **0.9818**, indicating the four learner groups are highly
reproducible and not an artifact of a particular random initialization
(improved from 0.926 under the old pipeline).

## 5. Final model performance: train vs. unseen test set

| Metric | Train (n=384) | Test / unseen (n=96) |
|---|---|---|
| Silhouette | 0.3038 | **0.3031** |
| Calinski-Harabasz | 214.85 | 53.19 |
| Davies-Bouldin | 1.2620 | 1.2060 |

Silhouette barely moves (0.3038 → 0.3031) on students the model never saw
during fitting — the cluster structure generalizes cleanly. The drop in
Calinski-Harabasz is expected, since that metric scales with sample size
(96 vs 384 points) rather than only cluster quality.

## 6. Cluster profiles (learner segments, behavioral features only)

Mean behavioral feature values per cluster, computed on the **train** split
(raw 0–100 activity-count scale):

| Cluster | raisedhands | VisitedResources | AnnouncementsView | Discussion |
|---|---|---|---|---|
| **Highly Engaged Learners** | 77.0 | 81.2 | 68.6 | 73.2 |
| **Consistent Learners** | 58.7 | 78.6 | 39.9 | 25.7 |
| **Struggling but Active Learners** | 26.5 | 30.4 | 22.0 | 69.9 |
| **At-Risk Learners** | 16.9 | 16.9 | 16.6 | 23.4 |

Post-hoc interpretation against `Class` (academic performance — **never**
used to fit the clusters), on the test / unseen split (n=96):

| Cluster | % Low class | % Middle class | % High class |
|---|---|---|---|
| At-Risk Learners (n=34) | 64.7% | 32.4% | 2.9% |
| Consistent Learners (n=30) | 0.0% | 70.0% | 30.0% |
| Highly Engaged Learners (n=23) | 0.0% | 47.8% | 52.2% |
| Struggling but Active Learners (n=9) | 44.4% | 55.6% | 0.0% |

This cross-tabulation is still a strong qualitative validation signal
after the feature-scope correction: **At-Risk Learners contain the highest
share of low-performing students (64.7%) and almost no high performers
(2.9%)**, while Highly Engaged and Consistent Learners contain 0%
low-performing students. The clusters, discovered purely from behavior, line
up sensibly with actual academic outcomes without ever having seen those
outcomes during training.

**Note on "Struggling but Active Learners":** this group has the *second*
highest Discussion rate (69.9, close to Highly Engaged's 73.2) but low
raisedhands/resources/announcements activity — they participate a lot in
one channel (discussion posts) while being largely disengaged elsewhere.
This is a distinct, actionable pattern: these students are talking, but not
attending to material or class-level cues.

## 7. Post-hoc fairness analysis (demographics, NEVER used for training)

Demographic/contextual columns played no role in fitting the preprocessor or
the K-Means model. After clustering, `src/train.py::fairness_breakdown()`
(and the equivalent block in `src/evaluate.py`) cross-tabulates each
discovered cluster against every demographic column, purely to check whether
any group ended up disproportionately concentrated in a cluster for reasons
unrelated to their actual logged behavior. Full breakdown (all demographic
columns) is saved in `models/model_metadata.json`
(`fairness_breakdown_train`) and `reports/test_evaluation.json`
(`fairness_breakdown_test`); highlights on the test split (n=96):

- **Gender:** the dataset overall is 63.5% male / 36.5% female. At-Risk
  Learners are 79.4% male vs. 20.6% female — noticeably more skewed than the
  dataset baseline. This is flagged, not explained away: it may reflect a
  genuine behavioral pattern in this specific dataset, or it may reflect
  something else (e.g. reporting/labeling artifacts in the source data). It
  is exactly the kind of signal this post-hoc check exists to surface, and
  it should be investigated by an educator/domain expert before the tool is
  used operationally, per the Responsible AI section.
- **Grade level:** lower grades (G-02, G-04) are somewhat overrepresented in
  At-Risk Learners (29.4% + 20.6% = 50.0% of that cluster vs. a smaller share
  of the overall test set), consistent with the earlier (pre-correction)
  finding — this pattern persisted after removing demographics from the
  clustering inputs, which is a modest piece of evidence it reflects a real
  early-grade engagement pattern rather than an artifact of demographic
  leakage into the clustering itself.
- **Nationality, topic, section, semester, parent fields:** no cluster is
  overwhelmingly dominated by a single category beyond what's expected from
  the dataset's own base rates; see the full tables in
  `reports/test_evaluation.json` for every category.

This section replaces the old "a warning in the Responsible AI section
alone" approach the mentor flagged as insufficient — it is now a concrete,
regenerated-every-run, machine-readable check, not just prose.

## 8. Data-driven confidence thresholds (replaces manually-chosen 0.5 / 1.5)

`src/train.py::confidence_thresholds_from_gaps()` computes, on the **train**
split, the gap between each student's distance to their assigned centroid
and their distance to the second-closest centroid, then takes the 33rd and
67th percentiles of that distribution as the low/high confidence cutoffs.
These are saved to `models/model_metadata.json["confidence_thresholds"]` and
loaded by `src/predict.py` at inference time — no constants are hard-coded
in `predict.py` anymore.

| Threshold | Value (train centroid-gap distribution) |
|---|---|
| Minimum gap | 0.0098 |
| p33 (low confidence cutoff) | **0.6822** |
| Median gap | 0.9804 |
| p67 (high confidence cutoff) | **1.2937** |
| Maximum gap | 2.1095 |

A new student's gap below 0.6822 → "Low confidence"; between 0.6822 and
1.2937 → "Moderate confidence"; above 1.2937 → "High confidence." If the
training data or model changes, re-running `python -m src.train`
automatically recomputes these thresholds — they are not manually re-tuned.

## 9. Error analysis (on the unseen test set, n=96)

- **5 of 96 students (5.2%)** have a negative per-sample silhouette value,
  meaning they are on average closer to a *different* cluster's centroid
  than to their own.
- **10 students** (bottom 10% by centroid-distance gap) are borderline — the
  full list is in `reports/borderline_students_test.csv`.
- These borderline cases are a natural, expected consequence of hard
  assignment (K-Means) on behavioral data that doesn't fall into perfectly
  separated groups. The `confidence_note` field returned by
  `src/predict.py::predict_student_cluster` flags exactly this situation
  using the data-driven thresholds from Section 8, so downstream users know
  to treat a borderline assignment as provisional.

## 10. Limitations and assumptions (summary — see README for the full list)

- Silhouette score alone favors k=2; k=4 is a deliberate, documented
  interpretability trade-off, not the statistically tightest split, and is
  never claimed to be the "natural" number of learner types.
- DBSCAN could not find a usable, low-noise clustering of the full dataset
  at any tested eps.
- 480 students from one platform is a small sample; discovered clusters may
  not transfer to a different institution's engagement patterns.
- Demographic features are excluded from clustering inputs entirely and are
  used only post-hoc, for interpretation and the fairness check in Section 7
  — never to imply ability, and never fed to the model. `Class` is likewise
  used only post-hoc, never for training or cluster assignment.
- The post-hoc fairness check surfaced a real gender skew in the At-Risk
  cluster (Section 7) that warrants human follow-up before operational use.
