"""
evaluate.py
-----------
Loads the artifacts saved by train.py and evaluates the FINAL model on the
20% test split that was never used for fitting the preprocessor or the
clustering model (the "unseen students").

This directly satisfies:
  - Acceptance Criterion: "The solution can process previously unseen input."
  - Grading Criterion: evaluation results reported on data not used in fitting.

What this script reports:
  1. Test-set cluster assignment using the saved model's .predict() (K-Means
     is the only one of the three algorithms compared in train.py that
     supports predicting cluster membership for brand-new points without
     re-fitting -- Agglomerative and DBSCAN do not have a native .predict(),
     which is itself a key reason K-Means was chosen as the FINAL model, on
     top of the interpretability reasoning already logged in train.py).
  2. Test-set Silhouette / Calinski-Harabasz / Davies-Bouldin, compared to the
     train-set values, to check whether cluster quality holds up on new data.
  3. Test-set cluster size distribution and Class crosstab (interpretation
     only, never used to fit anything).
  4. Error analysis: for every test student, the distance to their assigned
     centroid vs. the distance to the second-closest centroid. Students whose
     two nearest centroids are nearly equidistant are "borderline" cases --
     the clustering is genuinely ambiguous about where they belong. Students
     with a negative per-sample silhouette value are flagged as poorly-fit.

Run with: python -m src.evaluate   (after running python -m src.train)
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score

from src.data import load_raw_data, BEHAVIOR_NUMERIC_FEATURES, CLUSTERING_CATEGORICAL_FEATURES
from src.preprocessing import split_data, get_feature_frame, RANDOM_STATE
from src.train import fairness_breakdown

DATA_PATH = "data/xAPI-Edu-Data.csv"
MODELS_DIR = "models"


def load_artifacts():
    preprocessor = joblib.load(f"{MODELS_DIR}/preprocessor.joblib")
    model = joblib.load(f"{MODELS_DIR}/final_kmeans_model.joblib")
    with open(f"{MODELS_DIR}/model_metadata.json") as f:
        metadata = json.load(f)
    return preprocessor, model, metadata


def borderline_analysis(X, model, top_n=10):
    """For each point, find the gap between distance-to-1st and distance-to-2nd nearest centroid.
    A small gap (near 0) means the point sits almost exactly between two clusters."""
    distances = model.transform(X)  # (n_samples, n_clusters) distance to each centroid
    sorted_dist = np.sort(distances, axis=1)
    gap = sorted_dist[:, 1] - sorted_dist[:, 0]
    return gap


def main():
    preprocessor, model, metadata = load_artifacts()
    cluster_name_map = {int(k): v for k, v in metadata["cluster_name_map"].items()}

    df = load_raw_data(DATA_PATH)
    # IMPORTANT: same random_state / split logic as train.py so this is the
    # exact same held-out test set that the model never saw during fitting.
    train_df, test_df = split_data(df, random_state=RANDOM_STATE)

    X_test = preprocessor.transform(get_feature_frame(test_df))
    test_labels = model.predict(X_test)

    # ---------- 1 & 2: test-set internal validation metrics ----------
    test_silhouette = silhouette_score(X_test, test_labels)
    test_ch = calinski_harabasz_score(X_test, test_labels)
    test_db = davies_bouldin_score(X_test, test_labels)

    print("=== Test-set (unseen students, n={}) evaluation ===".format(len(test_df)))
    print(f"Train silhouette: {metadata['final_model_train_metrics']['silhouette']:.4f}")
    print(f"Test  silhouette: {test_silhouette:.4f}")
    print(f"Train Calinski-Harabasz: {metadata['final_model_train_metrics']['calinski_harabasz']:.2f}")
    print(f"Test  Calinski-Harabasz: {test_ch:.2f}")
    print(f"Train Davies-Bouldin: {metadata['final_model_train_metrics']['davies_bouldin']:.4f}")
    print(f"Test  Davies-Bouldin: {test_db:.4f}")

    # ---------- 3: cluster sizes + Class crosstab on test ----------
    test_profile = test_df.copy()
    test_profile["cluster"] = test_labels
    test_profile["cluster_name"] = test_profile["cluster"].map(cluster_name_map)
    cluster_sizes = test_profile["cluster_name"].value_counts()
    class_crosstab_test = pd.crosstab(test_profile["cluster_name"], test_profile["Class"], normalize="index").round(3)

    print("\nTest-set cluster sizes:")
    print(cluster_sizes)
    print("\nTest-set cluster vs. Class crosstab (interpretation only):")
    print(class_crosstab_test)

    # ---------- 3b: POST-HOC fairness check (demographics, never used in training) ----------
    fairness_test = fairness_breakdown(test_profile, cluster_col="cluster_name")
    print("\nPost-hoc fairness check (test set) -- demographic composition per cluster:")
    for col, table in fairness_test.items():
        print(f"  {col}: {table}")

    # ---------- 4: error analysis ----------
    per_sample_silhouette = silhouette_samples(X_test, test_labels)
    gap = borderline_analysis(X_test, model)

    test_profile["silhouette_sample"] = per_sample_silhouette
    test_profile["centroid_gap"] = gap

    n_poorly_fit = int((per_sample_silhouette < 0).sum())
    borderline_threshold = np.percentile(gap, 10)  # bottom 10% smallest gaps = most ambiguous
    n_borderline = int((gap <= borderline_threshold).sum())

    print(f"\nStudents with negative per-sample silhouette (poorly fit to their cluster): "
          f"{n_poorly_fit} / {len(test_df)} ({n_poorly_fit/len(test_df)*100:.1f}%)")
    print(f"Most ambiguous / borderline students (smallest 10% centroid gap): {n_borderline}")

    borderline_students = test_profile.sort_values("centroid_gap").head(10)[
        ["cluster_name", "centroid_gap", "silhouette_sample"] + BEHAVIOR_NUMERIC_FEATURES + ["Class"]
    ]

    # ---------- Save everything for the README / results.md ----------
    eval_summary = {
        "n_test": len(test_df),
        "test_silhouette": float(test_silhouette),
        "test_calinski_harabasz": float(test_ch),
        "test_davies_bouldin": float(test_db),
        "train_silhouette": metadata["final_model_train_metrics"]["silhouette"],
        "cluster_sizes_test": cluster_sizes.to_dict(),
        "class_crosstab_test": class_crosstab_test.to_dict(),
        "fairness_breakdown_test": fairness_test,
        "n_poorly_fit_negative_silhouette": n_poorly_fit,
        "pct_poorly_fit": round(n_poorly_fit / len(test_df) * 100, 1),
        "n_borderline_students": n_borderline,
    }
    with open("reports/test_evaluation.json", "w") as f:
        json.dump(eval_summary, f, indent=2, default=str)

    borderline_students.to_csv("reports/borderline_students_test.csv", index=False)

    print("\nSaved reports/test_evaluation.json and reports/borderline_students_test.csv")
    return eval_summary


if __name__ == "__main__":
    main()
