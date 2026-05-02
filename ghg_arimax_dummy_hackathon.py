"""Hackathon submission script for GHG forecasting with ARIMAX_DUMMY.

This script compares the original univariate ARIMA baseline against a final
ARIMAX model that uses policy / penalty dummy variables as exogenous inputs.

Outputs:
- model_outputs/ghg_arimax_dummy_summary.csv
- model_outputs/ghg_arimax_dummy_detailed.csv
- model_outputs/ghg_arimax_dummy_report.txt

Usage:
    ./.venv-compare/bin/python ghg_arimax_dummy_hackathon.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from compare_arima_vs_regression import (
    COUNTRY_NAMES,
    SEA_COUNTRIES,
    TARGET_NAME,
    build_country_backtest,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "model_outputs"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(repo_root / "WB_WDI_WIDEF.csv")
    best_indicators_df = pd.read_csv(output_dir / "sea_countries_best_indicators.csv")

    summary_rows = []
    detail_frames = []

    for country_code in SEA_COUNTRIES:
        summary, detail_df = build_country_backtest(country_code, df, best_indicators_df)
        summary_rows.append(summary.__dict__)
        detail_frames.append(detail_df)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(detail_frames, ignore_index=True)

    summary_df["arimax_dummy_delta_rmse"] = summary_df["arimax_dummy_rmse"] - summary_df["arima_rmse"]
    summary_df["arimax_dummy_improved"] = summary_df["arimax_dummy_delta_rmse"] < 0
    summary_df = summary_df.sort_values("arimax_dummy_delta_rmse")

    summary_path = output_dir / "ghg_arimax_dummy_summary.csv"
    detail_path = output_dir / "ghg_arimax_dummy_detailed.csv"
    report_path = output_dir / "ghg_arimax_dummy_report.txt"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    mean_arima_rmse = float(summary_df["arima_rmse"].mean())
    mean_arimax_dummy_rmse = float(summary_df["arimax_dummy_rmse"].mean())
    mean_delta = float(summary_df["arimax_dummy_delta_rmse"].mean())
    improved_count = int(summary_df["arimax_dummy_improved"].sum())
    total_countries = int(len(summary_df))
    recommended_model = "ARIMAX_DUMMY" if mean_arimax_dummy_rmse < mean_arima_rmse else "ARIMA"

    lines = []
    lines.append("=" * 100)
    lines.append("GHG FORECASTING HACKATHON SUBMISSION")
    lines.append("Final model: ARIMAX_DUMMY")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Target variable: {TARGET_NAME}")
    lines.append(f"Countries analyzed: {total_countries}")
    lines.append("Baseline model: univariate ARIMA")
    lines.append("Final model: ARIMAX with policy / penalty dummy exogenous variables")
    lines.append("")
    lines.append("OVERALL COMPARISON")
    lines.append("-" * 100)
    lines.append(f"Mean ARIMA RMSE:        {mean_arima_rmse:.6f}")
    lines.append(f"Mean ARIMAX_DUMMY RMSE:  {mean_arimax_dummy_rmse:.6f}")
    lines.append(f"Mean delta RMSE:         {mean_delta:.6f}  (ARIMAX_DUMMY - ARIMA)")
    lines.append(f"Countries improved:      {improved_count}/{total_countries}")
    lines.append(f"Recommended model:       {recommended_model}")
    lines.append("")
    lines.append("COUNTRY-LEVEL RESULTS")
    lines.append("-" * 100)
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.country_name:<14} | ARIMA RMSE={row.arima_rmse:.6f} | "
            f"ARIMAX_DUMMY RMSE={row.arimax_dummy_rmse:.6f} | delta={row.arimax_dummy_delta_rmse:.6f}"
        )
    lines.append("")
    lines.append("FILES WRITTEN")
    lines.append("-" * 100)
    lines.append(f"- {summary_path.relative_to(repo_root)}")
    lines.append(f"- {detail_path.relative_to(repo_root)}")
    lines.append(f"- {report_path.relative_to(repo_root)}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- ARIMAX_DUMMY adds policy / penalty dummy variables as exogenous regressors.")
    lines.append("- The dummy inputs include event-style series such as Singapore carbon tax, Indonesia ETS, PT Kallista Alam, Shell ruling, and country-level net-zero announcement proxies.")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {detail_path}")
    print(f"Saved report:  {report_path}")
    print(f"Mean ARIMA RMSE: {mean_arima_rmse:.6f}")
    print(f"Mean ARIMAX_DUMMY RMSE: {mean_arimax_dummy_rmse:.6f}")
    print(f"Countries improved: {improved_count}/{total_countries}")
    print(f"Recommended model: {recommended_model}")


if __name__ == "__main__":
    main()
