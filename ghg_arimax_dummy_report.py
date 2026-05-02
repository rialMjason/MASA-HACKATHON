"""Generate a hackathon-ready markdown report for the ARIMA GHG model.

This script reads the summary CSV written by ghg_arimax_dummy_hackathon.py and
produces a concise explanation of the comparison, the dummy variables used, and
the country-level performance against the original ARIMA baseline.

Output:
- reports/ghg_arimax_dummy_model_report.md

Usage:
    ./.venv-compare/bin/python ghg_arimax_dummy_report.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COUNTRY_LABELS = {
    "BRN": "Brunei",
    "KHM": "Cambodia",
    "IDN": "Indonesia",
    "LAO": "Laos",
    "MYS": "Malaysia",
    "MMR": "Myanmar",
    "PHL": "Philippines",
    "SGP": "Singapore",
    "THA": "Thailand",
    "TLS": "Timor-Leste",
    "VNM": "Vietnam",
}

DUMMY_DESCRIPTION_ROWS = [
    ("Global Shell ruling 2021", "1 in 2021 only", "All countries"),
    ("Net-zero announcement", "1 from the announcement year onward", "Country-specific proxy"),
    ("Singapore carbon tax", "1 from 2019 onward", "Singapore"),
    ("Indonesia ETS", "1 from 2023 onward", "Indonesia"),
    ("PT Kallista Alam penalty", "1 in 2014 only", "Indonesia"),
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "model_outputs"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    summary_path = output_dir / "ghg_arimax_dummy_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}. Run ghg_arimax_dummy_hackathon.py first.")

    df = pd.read_csv(summary_path)
    df = df.sort_values("arimax_dummy_delta_rmse")

    mean_arima_rmse = float(df["arima_rmse"].mean())
    mean_arimax_dummy_rmse = float(df["arimax_dummy_rmse"].mean())
    mean_delta = float(df["arimax_dummy_delta_rmse"].mean())
    improved = int((df["arimax_dummy_delta_rmse"] < 0).sum())
    worsened = int((df["arimax_dummy_delta_rmse"] > 0).sum())

    lines: list[str] = []
    lines.append("# GHG Forecasting Report: ARIMA")
    lines.append("")
    lines.append("## 1. Model Summary")
    lines.append(
        "The final hackathon model for GHG is ARIMA. ARIMAX_DUMMY was evaluated as a policy-aware comparison model using policy and penalty dummy variables as exogenous inputs, and it is compared against the original univariate ARIMA baseline on the same rolling-origin backtest."
    )
    lines.append("")
    lines.append("## 2. Why Dummies Were Added")
    lines.append(
        "Dummy variables let the model react to known interventions and one-off policy shocks without forcing those "
        "events to be learned only from the emission history. They are encoded as binary indicators that switch on "
        "at the event year and, for announcement-style variables, stay on afterward."
    )
    lines.append("")
    lines.append("## 3. Dummy Variables Used")
    lines.append("")
    lines.append("| Dummy | Encoding | Scope |")
    lines.append("|---|---|---|")
    for name, encoding, scope in DUMMY_DESCRIPTION_ROWS:
        lines.append(f"| {name} | {encoding} | {scope} |")
    lines.append("")
    lines.append("## 4. Evaluation Setup")
    lines.append("- Target: Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)")
    lines.append("- Countries: 11 Southeast Asian countries")
    lines.append("- Validation: rolling-origin backtest")
    lines.append("- Comparison: ARIMA vs ARIMAX_DUMMY")
    lines.append("")
    lines.append("## 5. Results")
    lines.append(f"- Mean ARIMA RMSE: {mean_arima_rmse:.6f}")
    lines.append(f"- Mean ARIMAX_DUMMY RMSE: {mean_arimax_dummy_rmse:.6f}")
    lines.append(f"- Mean delta RMSE (dummy - original): {mean_delta:.6f}")
    lines.append(f"- Countries improved: {improved}")
    lines.append(f"- Countries worsened: {worsened}")
    lines.append("")
    lines.append("### Country-level comparison")
    lines.append("")
    lines.append("| Country | ARIMA RMSE | ARIMAX_DUMMY RMSE | Delta RMSE |")
    lines.append("|---|---:|---:|---:|")
    for row in df.itertuples(index=False):
        country_name = COUNTRY_LABELS.get(row.country_code, row.country_code)
        lines.append(
            f"| {country_name} | {row.arima_rmse:.6f} | {row.arimax_dummy_rmse:.6f} | {row.arimax_dummy_delta_rmse:.6f} |"
        )
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append(
        "Even though ARIMAX_DUMMY can edge out ARIMA on mean RMSE in this run, the final submission for GHG is ARIMA to keep the model simpler and more stable across countries."
    )
    lines.append("")
    lines.append("## 7. Practical Takeaway")
    lines.append(
        "Use ARIMAX_DUMMY as a benchmark to measure the impact of policy events, but keep ARIMA as the final GHG forecasting choice for the hackathon submission."
    )

    report_path = reports_dir / "ghg_arimax_dummy_model_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved report: {report_path}")
    print(f"Mean ARIMA RMSE: {mean_arima_rmse:.6f}")
    print(f"Mean ARIMAX_DUMMY RMSE: {mean_arimax_dummy_rmse:.6f}")
    print(f"Countries improved: {improved}")

if __name__ == "__main__":
    main()
