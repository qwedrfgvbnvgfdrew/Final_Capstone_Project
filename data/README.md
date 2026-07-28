# Dataset: xAPI-Edu-Data (Students' Academic Performance Dataset)

## Source and license

- **Original source:** Kaggle — https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- **License:** CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike)
- **File in this repo:** `data/xAPI-Edu-Data.csv`
- **Size:** 480 records, 17 columns, no missing values, 0 duplicate rows.

The file included here is the same publicly distributed dataset, byte-for-byte
consistent with the row count and column names of the original Kaggle release.
No personally identifiable information is present; every field is either an
anonymized behavioral count or a coarse demographic category.

## What one record represents

Each row is **one individual student** enrolled in an online learning
environment (Kalboard 360, an LMS used in the original study this dataset
comes from), described by:

- **Demographics / context (used ONLY for post-hoc interpretation and
  fairness analysis, NEVER for clustering):** `gender`, `NationalITy`,
  `PlaceofBirth`, `StageID`, `GradeID`, `SectionID`, `Topic`, `Semester`,
  `Relation` (parent responsible for the student), `ParentAnsweringSurvey`,
  `ParentschoolSatisfaction`. See `src.data.DEMOGRAPHIC_CONTEXT_FEATURES`.
- **Behavioral / engagement features (the ONLY inputs used for
  clustering — `src.data.CLUSTERING_FEATURES`):**
  - `raisedhands` — number of times the student raised their hand in class.
  - `VisITedResources` — number of times the student visited course content.
  - `AnnouncementsView` — number of times the student checked new announcements.
  - `Discussion` — number of times the student participated in discussion groups.
  - `StudentAbsenceDays` — categorical: `Under-7` or `Above-7` days absent.
    Included in clustering because it's an attendance/behavior signal, not a
    demographic one.
- **Outcome label (NOT used for clustering):** `Class` — academic performance
  band (`L`ow, `M`iddle, `H`igh), used only *after* clustering to help
  interpret the discovered learner segments (see `src/train.py` and
  `reports/results.md`).

## Column dropped during feature selection

`PlaceofBirth` is dropped before modeling. In this dataset a student's stated
nationality (`NationalITy`) and place of birth co-occur almost one-to-one, so
keeping both would add a highly correlated, redundant categorical feature
without adding new information — this is the "highly correlated variables"
concern raised in the Capstone Brief (Section 11, Q3). This is a separate
decision from excluding `Class`, which is excluded for leakage reasons, not
correlation reasons.

## Known limitations

- Only 480 students from a small number of course sections, so discovered
  clusters may not generalize to every online learning platform.
- Demographic fields (`gender`, `NationalITy`, etc.) are **excluded from the
  clustering model entirely** (mentor-corrected feature scope) and are used
  only *after* clustering, for interpretation and a fairness check —
  never to imply ability or potential, and never fed to the model. See
  `reports/results.md`, Section 7.
- Behavioral counts are raw activity counts, not normalized by course length
  or number of sessions available, so they are treated as relative signals
  of engagement rather than absolute measures.

