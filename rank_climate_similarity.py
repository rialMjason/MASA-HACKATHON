import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def parse_args() -> argparse.Namespace:
    default_candidate_column = "INDICATOR_LABEL"

    parser = argparse.ArgumentParser(
        description=(
            "Tokenize indicator names and rank indicators by closeness to the climate indicator set."
        )
    )
    parser.add_argument(
        "--reference-csv",
        default="correlation_with_climate_all.csv",
        help="CSV containing climate indicators to use as the reference set.",
    )
    parser.add_argument(
        "--reference-column",
        default="indicator",
        help="Column name in the reference CSV containing indicator names.",
    )
    parser.add_argument(
        "--candidate-csv",
        default="WB_WDI_WIDEF.csv",
        help="CSV containing indicators to score.",
    )
    parser.add_argument(
        "--candidate-column",
        default=default_candidate_column,
        help=(
            "Column name in the candidate CSV containing indicator names "
            f"(default: {default_candidate_column})."
        ),
    )
    parser.add_argument(
        "--negative-csv",
        default="correlation_with_climate_nonclimate.csv",
        help="Optional CSV of known non-climate indicators for evaluation.",
    )
    parser.add_argument(
        "--out-dir",
        default="model_outputs",
        help="Directory to save the similarity model and rankings.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of closest indicators to print.",
    )
    return parser.parse_args()


def load_indicator_series(csv_path: str, column: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Missing column '{column}' in {csv_path}")
    values = df[column].dropna().astype(str).map(str.strip)
    values = values[values != ""]
    return values.drop_duplicates().reset_index(drop=True)


def build_vectorizer() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "svd",
                TruncatedSVD(n_components=100, random_state=42),
            ),
            ("normalize", Normalizer(copy=False)),
        ]
    )


def evaluate_reference_vs_negative(
    vectorizer: Pipeline,
    reference_texts: pd.Series,
    negative_texts: pd.Series | None,
) -> dict:
    reference_matrix = vectorizer.transform(reference_texts)
    reference_similarity = cosine_similarity(reference_matrix, reference_matrix)
    reference_mask = np.equal.outer(reference_texts.to_numpy(), reference_texts.to_numpy())
    np.fill_diagonal(reference_mask, True)
    reference_similarity[reference_mask] = -1.0
    reference_scores = reference_similarity.max(axis=1)

    metrics: dict[str, float] = {
        "reference_mean_best_similarity": float(reference_scores.mean()),
        "reference_median_best_similarity": float(np.median(reference_scores)),
    }

    if negative_texts is not None and len(negative_texts) > 0:
        negative_matrix = vectorizer.transform(negative_texts)
        negative_similarity = cosine_similarity(negative_matrix, reference_matrix)
        negative_mask = np.equal.outer(negative_texts.to_numpy(), reference_texts.to_numpy())
        negative_similarity[negative_mask] = -1.0
        negative_scores = negative_similarity.max(axis=1)
        metrics.update(
            {
                "negative_mean_best_similarity": float(negative_scores.mean()),
                "negative_median_best_similarity": float(np.median(negative_scores)),
            }
        )

        y_true = np.concatenate(
            [np.ones(len(reference_scores), dtype=int), np.zeros(len(negative_scores), dtype=int)]
        )
        y_score = np.concatenate([reference_scores, negative_scores])
        metrics["roc_auc_reference_vs_negative"] = float(roc_auc_score(y_true, y_score))

    return metrics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_texts = load_indicator_series(args.reference_csv, args.reference_column)
    candidate_texts = load_indicator_series(args.candidate_csv, args.candidate_column)
    try:
        negative_texts = load_indicator_series(args.negative_csv, "indicator")
    except Exception:
        negative_texts = None

    reference_set = set(reference_texts.tolist())
    candidate_texts = candidate_texts[~candidate_texts.isin(reference_set)].reset_index(drop=True)

    all_texts = pd.Index(reference_texts.tolist() + candidate_texts.tolist()).drop_duplicates().tolist()

    vectorizer = build_vectorizer()
    vectorizer.fit(all_texts)

    reference_matrix = vectorizer.transform(reference_texts)
    candidate_matrix = vectorizer.transform(candidate_texts)

    neighbor_count = min(3, len(reference_texts))
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=neighbor_count)
    neighbors.fit(reference_matrix)
    distances, indices = neighbors.kneighbors(candidate_matrix)
    similarities = 1.0 - distances

    reference_list = reference_texts.tolist()
    rankings = pd.DataFrame(
        {
            "indicator": candidate_texts,
            "closest_climate_indicator": [reference_list[row[0]] for row in indices],
            "climate_similarity": similarities[:, 0],
            "second_closest_climate_indicator": [reference_list[row[1]] if len(row) > 1 else None for row in indices],
            "second_closest_similarity": similarities[:, 1] if similarities.shape[1] > 1 else np.nan,
            "third_closest_climate_indicator": [reference_list[row[2]] if len(row) > 2 else None for row in indices],
            "third_closest_similarity": similarities[:, 2] if similarities.shape[1] > 2 else np.nan,
        }
    ).sort_values("climate_similarity", ascending=False)

    rankings["nearest_rank"] = np.arange(1, len(rankings) + 1)
    rankings["nearest_rank"] = rankings["nearest_rank"].astype(int)

    rankings.to_csv(out_dir / "climate_similarity_rankings.csv", index=False)

    model_artifact = {
        "vectorizer": vectorizer,
        "reference_indicators": reference_list,
        "reference_csv": args.reference_csv,
        "reference_column": args.reference_column,
        "candidate_csv": args.candidate_csv,
        "candidate_column": args.candidate_column,
        "neighbor_count": neighbor_count,
        "description": "TF-IDF + TruncatedSVD semantic embedding similarity against climate indicator references",
    }
    joblib.dump(model_artifact, out_dir / "climate_similarity_model.joblib")

    metrics = evaluate_reference_vs_negative(vectorizer, reference_texts, negative_texts)
    metrics_lines = [f"{key}: {value}" for key, value in metrics.items()]
    (out_dir / "climate_similarity_metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    print("Saved:")
    print(out_dir / "climate_similarity_model.joblib")
    print(out_dir / "climate_similarity_rankings.csv")
    print(out_dir / "climate_similarity_metrics.txt")
    print("\nTop matches:")
    print(rankings.head(args.top_k)[["indicator", "closest_climate_indicator", "climate_similarity"]].to_string(index=False))
    print("\nMetrics:")
    print("\n".join(metrics_lines))


if __name__ == "__main__":
    main()