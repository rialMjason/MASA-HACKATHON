import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a K-Means clustering model on unique indicator labels."
    )
    parser.add_argument(
        "--input-csv",
        default="WB_WDI_WIDEF.csv",
        help="Path to CSV containing indicator labels.",
    )
    parser.add_argument(
        "--input-column",
        default="INDICATOR_LABEL",
        help="Column containing indicator labels to cluster.",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.8,
        help=(
            "Maximum allowed missing ratio per indicator across numeric time columns. "
            "Indicators above this threshold are removed before clustering."
        ),
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="Minimum cluster count to evaluate when selecting k",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=15,
        help="Maximum cluster count to evaluate when selecting k",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum TF-IDF vocabulary size",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=100,
        help="TruncatedSVD component count for dense clustering features",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible clustering",
    )
    parser.add_argument(
        "--out-dir",
        default="model_outputs",
        help="Directory to save model artifacts and outputs",
    )
    return parser.parse_args()


def compute_indicator_missing_ratio(df: pd.DataFrame, input_column: str) -> pd.DataFrame:
    if input_column not in df.columns:
        raise ValueError(f"Missing required column '{input_column}'.")

    year_like_columns = [c for c in df.columns if str(c).isdigit()]
    if not year_like_columns:
        return pd.DataFrame({"indicator_label": pd.Series(dtype=str), "missing_ratio": pd.Series(dtype=float)})

    tmp = df[[input_column] + year_like_columns].copy()
    tmp = tmp[tmp[input_column].notna()].copy()
    tmp[input_column] = tmp[input_column].astype(str).map(str.strip)
    tmp = tmp[tmp[input_column] != ""]

    indicator_missing_ratio = (
        tmp.groupby(input_column)[year_like_columns]
        .apply(lambda g: float(g.isna().mean().mean()))
        .rename("missing_ratio")
        .reset_index()
        .rename(columns={input_column: "indicator_label"})
    )
    return indicator_missing_ratio


def choose_best_k(
    features: np.ndarray, min_k: int, max_k: int, random_state: int
) -> tuple[int, list[tuple[int, float]]]:
    n_rows = features.shape[0]
    effective_min_k = max(2, min_k)
    effective_max_k = min(max_k, n_rows - 1)

    if effective_min_k > effective_max_k:
        raise ValueError(
            f"Unable to evaluate k in range [{min_k}, {max_k}] with {n_rows} rows. "
            "Need at least 3 rows and min_k < row_count."
        )

    inertias: list[tuple[int, float]] = []
    for k in range(effective_min_k, effective_max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        model.fit(features)
        inertias.append((k, float(model.inertia_)))

    if not inertias:
        raise ValueError("No valid k values were evaluated.")

    # If only one k is available, use it directly.
    if len(inertias) == 1:
        return inertias[0][0], inertias

    ks = np.array([k for k, _ in inertias], dtype=float)
    ys = np.array([v for _, v in inertias], dtype=float)

    # Elbow method: choose the point with max distance from the line joining first and last points.
    p1 = np.array([ks[0], ys[0]], dtype=float)
    p2 = np.array([ks[-1], ys[-1]], dtype=float)
    line = p2 - p1
    line_norm = np.linalg.norm(line)

    if line_norm == 0:
        # Degenerate case: all inertias identical across tested k.
        return int(ks[0]), inertias

    points = np.column_stack((ks, ys))
    diff = points - p1
    # 2D perpendicular distance to the baseline using determinant magnitude.
    distances = np.abs(line[0] * diff[:, 1] - line[1] * diff[:, 0]) / line_norm
    best_idx = int(np.argmax(distances))
    best_k = int(ks[best_idx])
    return best_k, inertias


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.input_column not in df.columns:
        raise ValueError(
            f"Missing required column '{args.input_column}' in {args.input_csv}."
        )

    max_missing_ratio = args.max_missing_ratio
    if max_missing_ratio < 0 or max_missing_ratio > 1:
        raise ValueError("--max-missing-ratio must be between 0 and 1.")

    indicator_labels = (
        df[args.input_column]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    indicator_labels = (
        indicator_labels[indicator_labels != ""].drop_duplicates().reset_index(drop=True)
    )

    missing_ratio_by_indicator = compute_indicator_missing_ratio(df, args.input_column)
    if not missing_ratio_by_indicator.empty:
        keepers = set(
            missing_ratio_by_indicator.loc[
                missing_ratio_by_indicator["missing_ratio"] <= max_missing_ratio, "indicator_label"
            ].tolist()
        )
        indicator_labels = indicator_labels[indicator_labels.isin(keepers)].reset_index(drop=True)

    if len(indicator_labels) < 3:
        raise ValueError("Need at least 3 unique labels to run elbow-based K-Means.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=args.max_features,
    )
    tfidf = vectorizer.fit_transform(indicator_labels)

    max_possible_components = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if max_possible_components < 2:
        raise ValueError(
            "Unable to build dense embedding for clustering: not enough TF-IDF dimensions."
        )

    n_components = min(args.svd_components, max_possible_components)
    svd = TruncatedSVD(n_components=n_components, random_state=args.random_state)
    normalizer = Normalizer(copy=False)
    features = normalizer.fit_transform(svd.fit_transform(tfidf))

    best_k, inertia_table = choose_best_k(features, args.min_k, args.max_k, args.random_state)

    kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=args.random_state)
    cluster_ids = kmeans.fit_predict(features)
    final_silhouette = float(silhouette_score(features, cluster_ids))

    clustered = pd.DataFrame(
        {
            "indicator_label": indicator_labels.astype(str),
            "cluster": cluster_ids.astype(int),
        }
    )

    component_cols = [f"component_{i + 1}" for i in range(kmeans.cluster_centers_.shape[1])]
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=component_cols)
    centers.insert(0, "cluster", centers.index.astype(int))

    counts = clustered["cluster"].value_counts().sort_index()

    metrics_lines = [
        f"rows: {len(clustered)}",
        f"unique_labels: {len(clustered)}",
        f"input_column: {args.input_column}",
        f"max_missing_ratio: {max_missing_ratio}",
        f"svd_components: {n_components}",
        f"selected_k: {best_k}",
        f"silhouette_score: {final_silhouette}",
        "",
        "k_search_inertia:",
    ]
    metrics_lines.extend([f"k={k}: {inertia}" for k, inertia in inertia_table])
    metrics_lines.append("")
    metrics_lines.append("cluster_sizes:")
    metrics_lines.extend([f"cluster_{idx}: {size}" for idx, size in counts.items()])

    model_artifact = {
        "model": kmeans,
        "vectorizer": vectorizer,
        "svd": svd,
        "normalizer": normalizer,
        "input_column": args.input_column,
        "max_missing_ratio": max_missing_ratio,
        "selected_k": best_k,
        "silhouette_score": final_silhouette,
        "k_search_inertia": inertia_table,
        "cluster_centers": centers,
        "input_csv": args.input_csv,
    }

    joblib.dump(model_artifact, out_dir / "kmeans_model.joblib")
    clustered.sort_values(["cluster", "indicator_label"], ascending=[True, True]).to_csv(
        out_dir / "kmeans_clusters.csv", index=False
    )
    centers.to_csv(out_dir / "kmeans_cluster_centers.csv", index=False)
    (out_dir / "kmeans_metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    print("Saved:")
    print(out_dir / "kmeans_model.joblib")
    print(out_dir / "kmeans_clusters.csv")
    print(out_dir / "kmeans_cluster_centers.csv")
    print(out_dir / "kmeans_metrics.txt")
    print("\n" + "\n".join(metrics_lines))


if __name__ == "__main__":
    main()