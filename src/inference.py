"""
inference.py
------------
Thin alias for src/predict.py.

Some course reference materials expect the inference entry point to be
named `inference.py` (e.g. `src/inference.py`). This project's real,
documented inference logic lives in `src/predict.py` -- this file adds no
independent logic of its own. It simply re-exports everything from
predict.py, so:

  - `from src.inference import predict_student_cluster` works, AND
  - `from src.predict import predict_student_cluster` (used throughout the
    rest of this repo -- train.py, evaluate.py, demo.ipynb) also still works.

There is a single source of truth (predict.py). This file exists purely so
the filename itself matches a naming convention some reference materials
expect, without duplicating logic that could drift out of sync between two
copies over time.

See src/predict.py for full documentation of every function below.
"""

from src.predict import (
    predict_student_cluster,
    predict_students_batch,
    InvalidStudentInputError,
    REQUIRED_FIELDS,
    OPTIONAL_CONTEXT_FIELDS,
    RECOMMENDATIONS,
)

__all__ = [
    "predict_student_cluster",
    "predict_students_batch",
    "InvalidStudentInputError",
    "REQUIRED_FIELDS",
    "OPTIONAL_CONTEXT_FIELDS",
    "RECOMMENDATIONS",
]


if __name__ == "__main__":
    # Same smoke test as predict.py, callable via `python -m src.inference` too.
    example_student = {
        "raisedhands": 55,
        "VisITedResources": 60,
        "AnnouncementsView": 30,
        "Discussion": 40,
        "StudentAbsenceDays": "Under-7",
        "gender": "F",
        "NationalITy": "Jordan",
    }
    print(predict_student_cluster(example_student))
