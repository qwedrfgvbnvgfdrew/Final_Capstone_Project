"""
predict.py
----------
Inference interface: assign a brand-new student to one of the discovered
learner segments using the saved preprocessing pipeline + final K-Means model.

Mentor correction: the model is fit on LEARNING-BEHAVIOR inputs only
(`src.data.CLUSTERING_FEATURES` -- the 4 engagement counts + attendance).
Demographic/contextual fields are OPTIONAL here and are never used to decide
the cluster assignment; if supplied, they are only echoed back in the
`demographic_context` key of the result for downstream fairness reporting,
exactly as demonstrated in `src/train.py::fairness_breakdown`.

This satisfies:
  - Expected Deliverable: "A simple prediction function ... that assigns a new
    student to one of the discovered clusters."
  - Acceptance Criterion: "The solution can process previously unseen input."

Usage (after running `python -m src.train` at least once so models/ exists):

    from src.predict import predict_student_cluster

    student = {
        "raisedhands": 55,
        "VisITedResources": 60,
        "AnnouncementsView": 30,
        "Discussion": 40,
        "StudentAbsenceDays": "Under-7",
    }
    result = predict_student_cluster(student)
    print(result)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from src.data import CLUSTERING_FEATURES, BEHAVIOR_NUMERIC_FEATURES, CLUSTERING_CATEGORICAL_FEATURES, DEMOGRAPHIC_CONTEXT_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Only behavioral inputs are REQUIRED to assign a cluster.
REQUIRED_FIELDS = CLUSTERING_FEATURES
# Demographic fields are OPTIONAL and never influence the assignment -- see
# module docstring. If present, they're returned under "demographic_context"
# for fairness reporting only.
OPTIONAL_CONTEXT_FIELDS = DEMOGRAPHIC_CONTEXT_FEATURES

RECOMMENDATIONS = {
    "Highly Engaged Learners": "Offer advanced learning materials and leadership opportunities.",
    "Consistent Learners": "Maintain regular learning support and encourage continued engagement.",
    "Struggling but Active Learners": "Provide tutoring, personalized feedback, and additional practice resources.",
    "At-Risk Learners": "Send early-warning notifications, increase mentor communication, and recommend academic support services.",
}


class InvalidStudentInputError(ValueError):
    """Raised when a student record is missing fields or has an invalid type/value."""


def _validate_student(student: dict) -> None:
    if not isinstance(student, dict):
        raise InvalidStudentInputError(f"Expected a dict of student features, got {type(student)}.")

    missing = [f for f in REQUIRED_FIELDS if f not in student]
    if missing:
        raise InvalidStudentInputError(f"Missing required field(s): {missing}")

    for col in BEHAVIOR_NUMERIC_FEATURES:
        value = student[col]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidStudentInputError(
                f"Field '{col}' must be numeric (0-100 scale in the original dataset), got {value!r} ({type(value)})."
            )
        if value < 0 or value > 100:
            raise InvalidStudentInputError(
                f"Field '{col}' = {value} is outside the expected 0-100 range used by this dataset."
            )

    for col in CLUSTERING_CATEGORICAL_FEATURES:
        value = student[col]
        if not isinstance(value, str) or not value.strip():
            raise InvalidStudentInputError(f"Field '{col}' must be a non-empty string, got {value!r}.")

    for col in OPTIONAL_CONTEXT_FIELDS:
        if col in student and (not isinstance(student[col], str) or not student[col].strip()):
            raise InvalidStudentInputError(
                f"Optional field '{col}' was provided but must be a non-empty string, got {student[col]!r}."
            )


_preprocessor = None
_model = None
_metadata = None


def _load_artifacts():
    """Lazy-load artifacts once per process."""
    global _preprocessor, _model, _metadata
    if _preprocessor is None:
        preproc_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        model_path = os.path.join(MODELS_DIR, "final_kmeans_model.joblib")
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        for p in (preproc_path, model_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Could not find '{p}'. Run `python -m src.train` first to generate the saved model artifacts."
                )
        _preprocessor = joblib.load(preproc_path)
        _model = joblib.load(model_path)
        with open(meta_path) as f:
            _metadata = json.load(f)
    return _preprocessor, _model, _metadata


def predict_student_cluster(student: dict) -> dict:
    """
    Assign a single new student to a discovered learner segment, using
    BEHAVIORAL inputs only (`CLUSTERING_FEATURES`).

    Parameters
    ----------
    student : dict
        Must contain every field in `REQUIRED_FIELDS` (the 4 behavioral
        counts + StudentAbsenceDays). Behavioral count fields must be
        numeric in the 0-100 range used by the original dataset.
        `StudentAbsenceDays` must be "Under-7" or "Above-7".
        Demographic/contextual fields (gender, NationalITy, etc.) are
        OPTIONAL and, if present, are NOT used to compute the cluster --
        they are only echoed back for fairness reporting (see
        `demographic_context` in the return value).

    Returns
    -------
    dict with keys: cluster_id, cluster_name, recommendation,
    confidence_note, demographic_context
    """
    _validate_student(student)
    preprocessor, model, metadata = _load_artifacts()
    cluster_name_map = {int(k): v for k, v in metadata["cluster_name_map"].items()}

    row = pd.DataFrame([{k: student[k] for k in REQUIRED_FIELDS}])
    X = preprocessor.transform(row)
    cluster_id = int(model.predict(X)[0])
    cluster_name = cluster_name_map.get(cluster_id, f"Cluster {cluster_id}")

    # Confidence note: how much closer is the assigned centroid than the
    # runner-up, compared against thresholds DERIVED FROM THE TRAINING
    # centroid-gap distribution (see src/train.py::confidence_thresholds_from_gaps)
    # -- not manually chosen constants.
    thresholds = metadata.get("confidence_thresholds", {})
    low_threshold = thresholds.get("low_threshold", 0.5)
    high_threshold = thresholds.get("high_threshold", 1.5)

    distances = model.transform(X)[0]
    sorted_dist = np.sort(distances)
    gap = float(sorted_dist[1] - sorted_dist[0])
    if gap <= low_threshold:
        confidence_note = (
            "Low confidence: this student's behavior sits nearly equidistant between two "
            "learner profiles (centroid gap in the bottom third of the training distribution). "
            "Treat the assignment as provisional."
        )
    elif gap < high_threshold:
        confidence_note = (
            "Moderate confidence in this cluster assignment (centroid gap in the middle "
            "third of the training distribution)."
        )
    else:
        confidence_note = (
            "High confidence: this student is clearly closest to one learner profile "
            "(centroid gap in the top third of the training distribution)."
        )

    # Demographic/contextual fields are NEVER used to compute the cluster --
    # they are only surfaced here, unchanged, for fairness auditing by the
    # caller (e.g. "are At-Risk assignments concentrated in one nationality?").
    demographic_context = {k: student[k] for k in OPTIONAL_CONTEXT_FIELDS if k in student}

    return {
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "recommendation": RECOMMENDATIONS.get(cluster_name, "Review student profile manually."),
        "confidence_note": confidence_note,
        "demographic_context": demographic_context,
    }


def predict_students_batch(students: list) -> pd.DataFrame:
    """Convenience wrapper: run predict_student_cluster over a list of student dicts."""
    rows = []
    for i, s in enumerate(students):
        try:
            result = predict_student_cluster(s)
            result["row_index"] = i
            result["error"] = None
        except InvalidStudentInputError as e:
            result = {"row_index": i, "cluster_id": None, "cluster_name": None,
                      "recommendation": None, "confidence_note": None, "error": str(e)}
        rows.append(result)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    example_student = {
        # Required (behavioral) fields:
        "raisedhands": 55,
        "VisITedResources": 60,
        "AnnouncementsView": 30,
        "Discussion": 40,
        "StudentAbsenceDays": "Under-7",
        # Optional (demographic/context) fields -- NOT used in the assignment,
        # only echoed back under "demographic_context" for fairness reporting:
        "gender": "F",
        "NationalITy": "Jordan",
    }
    print(predict_student_cluster(example_student))

    invalid_student = dict(example_student)
    invalid_student["raisedhands"] = "a lot"  # wrong type on purpose
    try:
        predict_student_cluster(invalid_student)
    except InvalidStudentInputError as e:
        print(f"\nCorrectly rejected invalid input: {e}")
