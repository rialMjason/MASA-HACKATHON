import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a classifier that labels indicators as informative or non-informative "
            "for climate-change prediction."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="correlation_with_climate_all.csv",
        help="Path to CSV with columns: indicator, corr, n",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.25,
        help="Absolute correlation threshold to define informative indicators",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion for test split",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible split",
    )
    parser.add_argument(
        "--out-dir",
        default="model_outputs",
        help="Directory to save model and predictions",
    )
    return parser.parse_args()


def train_classifier(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    features = df[["indicator", "n"]].copy()
    target = df["informative"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "indicator_text",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000),
                "indicator",
            ),
            ("coverage_n", StandardScaler(with_mean=False), ["n"]),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    return model, x_test, y_test, x_train, y_train


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    required_columns = {"indicator", "corr", "n"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["indicator", "corr", "n"]).copy()
    df["abs_corr"] = df["corr"].abs()
    df["informative"] = (df["abs_corr"] >= args.corr_threshold).astype(int)

    model, x_test, y_test, x_train, y_train = train_classifier(
        df, test_size=args.test_size, random_state=args.random_state
    )

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "informative_rate_train": float(y_train.mean()),
        "informative_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    report = classification_report(y_test, y_pred, digits=4)

    model_artifact = {
        "model": model,
        "corr_threshold": args.corr_threshold,
        "feature_columns": ["indicator", "n"],
        "label_definition": "informative = abs(corr) >= corr_threshold",
    }
    joblib.dump(model_artifact, out_dir / "indicator_classifier.joblib")

    predictions = df[["indicator", "corr", "n", "abs_corr", "informative"]].copy()
    predictions["informative_probability"] = model.predict_proba(df[["indicator", "n"]])[:, 1]
    predictions["predicted_label"] = (predictions["informative_probability"] >= 0.5).astype(int)
    predictions["predicted_class"] = predictions["predicted_label"].map(
        {1: "informative", 0: "non_informative"}
    )
    predictions.to_csv(out_dir / "indicator_predictions.csv", index=False)

    metrics_lines = [f"{key}: {value}" for key, value in metrics.items()]
    metrics_text = "\n".join(metrics_lines)
    (out_dir / "metrics.txt").write_text(
        metrics_text + "\n\nClassification report:\n" + report,
        encoding="utf-8",
    )

    print("Saved:")
    print(out_dir / "indicator_classifier.joblib")
    print(out_dir / "indicator_predictions.csv")
    print(out_dir / "metrics.txt")
    print("\n" + metrics_text)
    print("\nClassification report:\n" + report)


if __name__ == "__main__":
    main()
