import sys
from pathlib import Path
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent
out_dir = repo_root / "model_outputs"
exc_file = out_dir / "final_forecast_2030.xlsx"

if not exc_file.exists():
    print(f"Forecast file not found: {exc_file}\nRun final_forecast_2030_improved.py first to generate forecasts.")
    sys.exit(1)

print(f"Loading forecasts from: {exc_file}")
df = pd.read_excel(exc_file, sheet_name="Forecasts")

# Ensure Year is int
if df["Year"].dtype != int:
    df["Year"] = df["Year"].astype(int)

# Summary for 2030 per country and variable
summary_rows = []
for country in sorted(df["Country"].unique()):
    for var in sorted(df[df["Country"] == country]["Variable"].unique()):
        sub = df[(df["Country"] == country) & (df["Variable"] == var)]
        if sub.empty:
            continue
        row_2030 = sub[sub["Year"] == 2030]
        forecast_2030 = float(row_2030["Forecast"].values[0]) if not row_2030.empty else np.nan
        model = row_2030["Model"].values[0] if not row_2030.empty else ""

        # Check constancy across forecast horizon
        fvals = sub["Forecast"].dropna().values
        is_const = len(set(np.round(fvals, 6))) == 1 if len(fvals) > 0 else False

        # CI width: average and last year
        ci_widths = (sub["Upper_CI"] - sub["Lower_CI"]).dropna().values
        avg_ci_width = float(ci_widths.mean()) if len(ci_widths) > 0 else np.nan
        last_ci_width = float(ci_widths[-1]) if len(ci_widths) > 0 else np.nan

        summary_rows.append({
            "Country": country,
            "Variable": var,
            "Forecast_2030": forecast_2030,
            "Model": model,
            "Is_Constant_Forecast": is_const,
            "Avg_CI_Width": avg_ci_width,
            "Last_CI_Width": last_ci_width,
        })

summary_df = pd.DataFrame(summary_rows)

# Save summary
csv_out = out_dir / "forecast_summary.csv"
summary_df.to_csv(csv_out, index=False)

# Print concise summary
print("\nForecast summary (per country-variable for 2030):\n")
print(summary_df.to_string(index=False))
print(f"\nSaved summary to: {csv_out}")
