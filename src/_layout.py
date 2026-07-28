"""
_layout.py
----------
Defensive utility: if this project's files ever end up "flattened" into a
single folder (e.g. because a file was dragged into GitHub's web upload one
file at a time instead of as a folder, which silently drops subfolder
structure), this function reorganizes them back into the expected layout
so every other script/notebook in this repo keeps working unmodified.

This is defensive infrastructure, not part of the core ML pipeline.
"""

import os
import shutil

# Maps: filename as it would appear in a flattened upload -> its correct
# relative path in the proper repository layout.
EXPECTED_LAYOUT = {
    "xAPI-Edu-Data.csv": "data/xAPI-Edu-Data.csv",
    "preprocessor.joblib": "models/preprocessor.joblib",
    "final_kmeans_model.joblib": "models/final_kmeans_model.joblib",
    "pca_projector.joblib": "models/pca_projector.joblib",
    "model_metadata.json": "models/model_metadata.json",
    "elbow_plot.png": "reports/figures/elbow_plot.png",
    "silhouette_vs_k.png": "reports/figures/silhouette_vs_k.png",
    "pca_clusters_train.png": "reports/figures/pca_clusters_train.png",
    "eda_class_distribution.png": "reports/figures/eda_class_distribution.png",
    "eda_behavior_distributions.png": "reports/figures/eda_behavior_distributions.png",
    "eda_behavior_correlation.png": "reports/figures/eda_behavior_correlation.png",
    "eda_behavior_by_class.png": "reports/figures/eda_behavior_by_class.png",
    "experiment_results.csv": "reports/experiment_results.csv",
    "test_evaluation.json": "reports/test_evaluation.json",
    "borderline_students_test.csv": "reports/borderline_students_test.csv",
    "results.md": "reports/results.md",
    "data.py": "src/data.py",
    "preprocessing.py": "src/preprocessing.py",
    "train.py": "src/train.py",
    "evaluate.py": "src/evaluate.py",
    "predict.py": "src/predict.py",
}


def repair_flat_layout(root: str = ".") -> list:
    """
    If files from EXPECTED_LAYOUT exist directly in `root` (flattened),
    copy each one into its correct subfolder (creating subfolders as
    needed) without deleting the original. Safe to call every time --
    it only acts on files it finds sitting in the wrong place, and never
    overwrites a file that already exists at the correct destination.

    Also guarantees src/__init__.py exists so `import src.xxx` works.

    Returns the list of paths that were repaired (for logging/inspection).
    """
    repaired = []
    for fname, correct_rel_path in EXPECTED_LAYOUT.items():
        flat_path = os.path.join(root, fname)
        correct_path = os.path.join(root, correct_rel_path)
        if os.path.exists(flat_path) and not os.path.exists(correct_path):
            os.makedirs(os.path.dirname(correct_path), exist_ok=True)
            shutil.copy2(flat_path, correct_path)
            repaired.append(correct_rel_path)

    src_init = os.path.join(root, "src", "__init__.py")
    if os.path.isdir(os.path.join(root, "src")) and not os.path.exists(src_init):
        open(src_init, "a").close()
        repaired.append("src/__init__.py (created empty)")

    return repaired


if __name__ == "__main__":
    fixed = repair_flat_layout(".")
    if fixed:
        print(f"Repaired {len(fixed)} file(s):")
        for f in fixed:
            print(f"  -> {f}")
    else:
        print("No flattened files found -- layout already looks correct.")
