"""Plot facet grids for GHG per-capita and Forest area for all 11 SEA countries.

Saves two PNG files to `model_outputs/figures/`: `ghg_facets.png` and `forest_facets.png`.

Usage:
    python scripts/plot_ghg_forest_facets.py

Requirements:
    pandas, matplotlib, seaborn
"""
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SEA_CODES = ["MYS","IDN","THA","VNM","PHL","SGP","BRN","KHM","LAO","MMR","TLS"]

# Map code -> display name (keeps nice facet titles)
CODE_TO_NAME = {
    "MYS": "Malaysia",
    "IDN": "Indonesia",
    "THA": "Thailand",
    "VNM": "Viet Nam",
    "PHL": "Philippines",
    "SGP": "Singapore",
    "BRN": "Brunei Darussalam",
    "KHM": "Cambodia",
    "LAO": "Lao PDR",
    "MMR": "Myanmar",
    "TLS": "Timor-Leste",
}

# Normalize name variants between forecast outputs and plot labels
COUNTRY_ALIASES = {
    "Brunei": "Brunei Darussalam",
    "Laos": "Lao PDR",
    "Vietnam": "Viet Nam",
}

GHG_IND = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
FOREST_IND = "Forest area (% of land area)"


def year_columns(df):
    return [c for c in df.columns if re.match(r"^\d{4}$", c)]


def melt_indicator(df, indicator_label, value_name):
    yrs = year_columns(df)
    sub = df[df["INDICATOR_LABEL"] == indicator_label].copy()
    if sub.empty:
        return pd.DataFrame()
    long = sub.melt(id_vars=["REF_AREA", "INDICATOR_LABEL"], value_vars=yrs,
                    var_name="Year", value_name=value_name)
    long = long.rename(columns={"REF_AREA": "Code"})
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
    long = long[long["Code"].isin(SEA_CODES)]
    long["Country"] = long["Code"].map(CODE_TO_NAME)
    return long


def plot_facets(df_long, value_col, out_path, ylabel, title, forecasts=None):
    sns.set_style("whitegrid")
    plot_df = df_long.dropna(subset=[value_col, "Year"]).copy()
    if plot_df.empty:
        raise ValueError(f"No data available for {value_col}")

    countries = [CODE_TO_NAME[c] for c in SEA_CODES]
    n = len(countries)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
    axes_flat = axes.flatten()

    for i, country in enumerate(countries):
        ax = axes_flat[i]
        sub_obs = plot_df[plot_df["Country"] == country]
        if not sub_obs.empty:
            ax.plot(sub_obs["Year"], sub_obs[value_col], marker="o", label="Observed", color="C0")

        # overlay forecast if provided
        if forecasts is not None:
            fsub = forecasts[(forecasts["Country"] == country) & (forecasts["Variable"].str.lower() == ("ghg" if value_col=="ghg" else "forest"))]
            if not fsub.empty:
                ax.plot(fsub["Year"], fsub["Forecast"], marker="o", linestyle="--", color="C1", label="Forecast")
                if "Lower_CI" in fsub.columns and "Upper_CI" in fsub.columns:
                    ax.fill_between(fsub["Year"], fsub["Lower_CI"], fsub["Upper_CI"], color="C1", alpha=0.2)

        ax.set_title(country)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize="small")

    # hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    wb = repo_root / "WB_WDI_WIDEF.csv"
    if not wb.exists():
        raise FileNotFoundError("WB_WDI_WIDEF.csv not found in repository root")

    df = pd.read_csv(wb, low_memory=False)

    ghg_long = melt_indicator(df, GHG_IND, "ghg")
    forest_long = melt_indicator(df, FOREST_IND, "forest")

    # Optional forecast overlay (with confidence intervals)
    forecast_file = repo_root / "model_outputs" / "final_forecast_2030.xlsx"
    forecasts = None
    if forecast_file.exists():
        forecasts = pd.read_excel(forecast_file, sheet_name="Forecasts")
        for col in ["Year", "Forecast", "Lower_CI", "Upper_CI"]:
            if col in forecasts.columns:
                forecasts[col] = pd.to_numeric(forecasts[col], errors="coerce")
        if "Country" in forecasts.columns:
            forecasts["Country"] = forecasts["Country"].replace(COUNTRY_ALIASES)

    out_dir = repo_root / "model_outputs" / "figures"
    plot_facets(ghg_long, "ghg", out_dir / "ghg_facets.png",
                ylabel="t CO2e per capita", title="GHG per-capita (all SEA countries)", forecasts=forecasts)

    plot_facets(forest_long, "forest", out_dir / "forest_facets.png",
                ylabel="% of land area", title="Forest area (% of land area) (all SEA countries)", forecasts=forecasts)

    print(f"Saved: {out_dir / 'ghg_facets.png'}")
    print(f"Saved: {out_dir / 'forest_facets.png'}")


if __name__ == '__main__':
    main()
