"""
Converted from R: panel negative-binomial regression

This block prepares a panel dataset joining EMDAT event frequencies with
World Bank indicators (GHG per-capita and forest area), creates lagged
predictors, and fits a Negative Binomial GLM similar to R's glm.nb.

Usage: the script will look for `WB_WDI_WIDEF.csv` and an EMDAT export
file starting with `public_emdat_custom_request` in the repository root.
Run the NB portion directly with:

    python -c "from final_forecast_2030_improved import main_nb; main_nb()"

Or run the whole `final_forecast_2030_improved.py` as before; this block
is safe to call independently.
"""

import glob
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# Indicator labels (must match strings in WB_WDI_WIDEF.csv)
GHG_IND = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
FOREST_IND = "Forest area (% of land area)"

SEA_CODES = ["MYS","IDN","THA","VNM","PHL","SGP","BRN","KHM","LAO","MMR","TLS"]

def find_emdat_file(repo_root: Path):
    patterns = [
        str(repo_root / "public_emdat_custom_request*.csv"),
        str(repo_root / "public_emdat_custom_request*.txt"),
        str(repo_root / "data" / "public_emdat_custom_request*.csv"),
    ]
    for p in patterns:
        for f in glob.glob(p):
            return Path(f)
    return None

def load_and_pivot_wb(wb_path: Path):
    df = pd.read_csv(wb_path, low_memory=False)
    year_cols = [c for c in df.columns if re.match(r"^\d{4}$", c)]

    ghg_df = df[(df["REF_AREA"].isin(SEA_CODES)) & (df["INDICATOR_LABEL"] == GHG_IND)].copy()
    ghg_long = ghg_df.melt(id_vars=["REF_AREA","INDICATOR_LABEL"], value_vars=year_cols,
                            var_name="Year", value_name="ghg")
    ghg_long = ghg_long.rename(columns={"REF_AREA": "Country"})
    ghg_long["Year"] = pd.to_numeric(ghg_long["Year"], errors="coerce")

    forest_df = df[(df["REF_AREA"].isin(SEA_CODES)) & (df["INDICATOR_LABEL"] == FOREST_IND)].copy()
    forest_long = forest_df.melt(id_vars=["REF_AREA","INDICATOR_LABEL"], value_vars=year_cols,
                                 var_name="Year", value_name="forest")
    forest_long = forest_long.rename(columns={"REF_AREA": "Country"})
    forest_long["Year"] = pd.to_numeric(forest_long["Year"], errors="coerce")

    return ghg_long, forest_long

def build_panel_and_fit(wb_path: Path, repo_root: Path):
    ghg_long, forest_long = load_and_pivot_wb(wb_path)

    emdat_file = find_emdat_file(repo_root)
    if emdat_file is None:
        raise FileNotFoundError("Could not find EMDAT file (public_emdat_custom_request...csv). Place it in the repo root.")

    emdat = pd.read_csv(emdat_file)
    if 'Start Year' not in emdat.columns:
        # try variant names
        possible = [c for c in emdat.columns if 'Start' in c and 'Year' in c]
        if possible:
            emdat = emdat.rename(columns={possible[0]: 'Start Year'})
        else:
            raise KeyError("EMDAT file missing 'Start Year' column")

    freq_table = (
        emdat.groupby(['Country', 'Start Year'])
             .size()
             .reset_index(name='Frequency')
             .rename(columns={'Start Year': 'Year'})
    )

    country_map = pd.DataFrame({
        'Country': ["Malaysia","Indonesia","Thailand","Viet Nam","Philippines",
                    "Singapore","Brunei Darussalam","Cambodia","Lao People's Democratic Republic","Myanmar","Timor-Leste"],
        'REF_AREA': ["MYS","IDN","THA","VNM","PHL","SGP","BRN","KHM","LAO","MMR","TLS"]
    })

    freq_table = freq_table.merge(country_map, on='Country', how='left')
    freq_table = freq_table.drop(columns=['Country']).rename(columns={'REF_AREA': 'Country'})

    panel = (
        freq_table.merge(ghg_long[['Country','Year','ghg']], on=['Country','Year'], how='left')
                  .merge(forest_long[['Country','Year','forest']], on=['Country','Year'], how='left')
                  .sort_values(['Country','Year'])
    )

    panel['ghg_lag9'] = panel.groupby('Country')['ghg'].shift(9)
    panel['forest_lag1'] = panel.groupby('Country')['forest'].shift(1)

    panel_model = panel[panel['ghg_lag9'].notna()].copy()
    if panel_model.empty:
        raise ValueError('No rows with ghg_lag9 available to fit the model')

    formula = 'Frequency ~ ghg_lag9 + forest_lag1 + Year + C(Country)'
    model = smf.glm(formula=formula, data=panel_model, family=sm.families.NegativeBinomial())
    res = model.fit()
    return res, panel_model

def main_nb(repo_root: Path = Path(__file__).resolve().parent):
    wb_path = repo_root / 'WB_WDI_WIDEF.csv'
    if not wb_path.exists():
        raise FileNotFoundError('WB_WDI_WIDEF.csv not found in repository root')

    res, panel_model = build_panel_and_fit(wb_path, repo_root)
    print(res.summary())

warnings.filterwarnings("ignore")

SEA_COUNTRIES = {
    "Brunei Darussalam": "Brunei",
    "Cambodia": "Cambodia",
    "Indonesia": "Indonesia",
    "Lao PDR": "Laos",
    "Malaysia": "Malaysia",
    "Myanmar": "Myanmar",
    "Philippines": "Philippines",
    "Singapore": "Singapore",
    "Thailand": "Thailand",
    "Timor-Leste": "Timor-Leste",
    "Vietnam": "Vietnam",
}

GHG_IND = "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
FOREST_IND = "Forest area (% of land area)"

# Policy dummies for GHG (similar to forest)
POLICY_DUMMIES = {
    "all": [2021],  # Shell 2021
    "SGP": [2019],  # Carbon tax 2019
    "IDN": [2023],  # ETS 2023
    "VNM": [2023],  # Net-Zero 2023
    "THA": [2023],  # Net-Zero 2023
    "MYS": [2023],  # Net-Zero 2023
}

def extract_series(df, country_label, indicator):
    mask = (df["REF_AREA_LABEL"] == country_label) & (df["INDICATOR_LABEL"] == indicator)
    if not mask.any():
        return None
    row = df[mask].iloc[0]
    years = sorted([int(c) for c in row.index if c.isdigit()])
    values = [float(row[str(y)]) if pd.notna(row[str(y)]) else np.nan for y in years]
    ts = pd.Series(values, index=years).dropna()
    return ts if len(ts) > 0 else None

def build_policy_dummies(country_name, start_year, end_year):
    """Build policy dummy variables."""
    dummies = {}
    
    # Map country to code
    code_map = {
        "Brunei": "SGP", "Singapore": "SGP", "Indonesia": "IDN", "Laos": "LAO",
        "Vietnam": "VNM", "Thailand": "THA", "Malaysia": "MYS", "Philippines": "PHL",
        "Cambodia": "KHM", "Myanmar": "MMR", "Timor-Leste": "TLS",
    }
    code = code_map.get(country_name, "SGP")
    
    # Global dummies
    if "all" in POLICY_DUMMIES:
        for year in POLICY_DUMMIES["all"]:
            dummies[f"shell_{year}"] = np.array([1.0 if y == year else 0.0 for y in range(start_year, end_year + 1)])
    
    # Country specific
    if code in POLICY_DUMMIES:
        for year in POLICY_DUMMIES[code]:
            dummies[f"{code.lower()}_policy_{year}"] = np.array([1.0 if y >= year else 0.0 for y in range(start_year, end_year + 1)])
    
    if not dummies:
        return None
    return pd.DataFrame(dummies, index=range(start_year, end_year + 1))

def fit_arimax_ghg(ts, country_name):
    """Fit ARIMAX model for GHG with policy dummies."""
    if ts is None or len(ts) < 10:
        return None, None
    
    start_year = int(ts.index.min())
    end_year = int(ts.index.max())
    
    # Build exog dummies
    exog = build_policy_dummies(country_name, start_year, end_year)
    
    orders = [(0,1,0), (1,1,0), (1,1,1), (0,1,1), (2,1,0)]
    best_model, best_aic, best_order = None, np.inf, None
    
    for order in orders:
        try:
            if exog is not None:
                m = SARIMAX(ts, exog=exog, order=order, seasonal_order=(0,0,0,0)).fit(disp=False)
            else:
                m = ARIMA(ts, order=order).fit()
            
            if m.aic < best_aic:
                best_aic, best_model, best_order = m.aic, m, order
        except:
            pass
    
    return best_model, best_order

def forecast_to_2030(ts, model, exog_future=None):
    """Forecast to 2030."""
    if model is None or ts is None:
        return None
    
    last_year = int(ts.index.max())
    steps = 2030 - last_year
    
    if steps <= 0:
        return None
    
    try:
        # Check if model has exogenous variables
        has_exog = hasattr(model, "model") and hasattr(model.model, "exog") and model.model.exog is not None
        
        if has_exog and exog_future is not None:
            # Make sure exog_future has right shape
            if exog_future.shape[0] != steps:
                exog_future = exog_future.iloc[:steps]
            fcast = model.get_forecast(steps=steps, exog=exog_future)
        else:
            fcast = model.get_forecast(steps=steps)
        
        conf = fcast.conf_int()
        return pd.DataFrame({
            "Year": range(last_year + 1, 2031),
            "Forecast": fcast.predicted_mean.values[:steps],
            "Lower_CI": conf.iloc[:, 0].values[:steps],
            "Upper_CI": conf.iloc[:, 1].values[:steps],
        })
    except Exception as e:
        return None

def main():
    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "model_outputs"
    
    print("Loading World Bank data...")
    df = pd.read_csv("WB_WDI_WIDEF.csv", low_memory=False)
    
    all_results = []
    
    for country_label, country_name in SEA_COUNTRIES.items():
        print(f"  {country_name}...", end="", flush=True)
        
        # GHG with policy dummies
        ghg_ts = extract_series(df, country_label, GHG_IND)
        if ghg_ts is not None:
            ghg_model, ghg_order = fit_arimax_ghg(ghg_ts, country_name)
            
            # Build future exog
            exog_future = None
            if ghg_model is not None and hasattr(ghg_model, "exog"):
                last_year = int(ghg_ts.index.max())
                start_year = int(ghg_ts.index.min())
                exog_future = build_policy_dummies(country_name, last_year + 1, 2030)
            
            ghg_fcast = forecast_to_2030(ghg_ts, ghg_model, exog_future)
            if ghg_fcast is not None:
                ghg_fcast["Country"] = country_name
                ghg_fcast["Variable"] = "GHG"
                ghg_fcast["Model"] = f"ARIMAX{ghg_order}" if ghg_order else "ARIMA"
                all_results.append(ghg_fcast)
        
        # Forest (unchanged)
        forest_ts = extract_series(df, country_label, FOREST_IND)
        if forest_ts is not None:
            orders = [(0,1,0), (1,1,0), (1,1,1), (0,1,1)]
            best_model, best_aic, best_order = None, np.inf, None
            for order in orders:
                try:
                    m = ARIMA(forest_ts, order=order).fit()
                    if m.aic < best_aic:
                        best_aic, best_model, best_order = m.aic, m, order
                except:
                    pass
            
            if best_model is not None:
                forest_fcast = forecast_to_2030(forest_ts, best_model)
                if forest_fcast is not None:
                    forest_fcast["Country"] = country_name
                    forest_fcast["Variable"] = "Forest"
                    forest_fcast["Model"] = str(best_order)
                    all_results.append(forest_fcast)
        
        print(" ✓")
    
    if not all_results:
        print("No forecasts generated")
        return
    
    df_out = pd.concat(all_results, ignore_index=True)
    
    # Excel
    output_file = output_dir / "final_forecast_2030.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Forecasts", index=False)
        
        # Summary
        summary = []
        for country_name in sorted(SEA_COUNTRIES.values()):
            ghg_rows = df_out[(df_out["Country"] == country_name) & (df_out["Variable"] == "GHG") & (df_out["Year"] == 2030)]
            forest_rows = df_out[(df_out["Country"] == country_name) & (df_out["Variable"] == "Forest") & (df_out["Year"] == 2030)]
            
            summary.append({
                "Country": country_name,
                "GHG_2030": ghg_rows["Forecast"].values[0] if len(ghg_rows) > 0 else np.nan,
                "GHG_Model": ghg_rows["Model"].values[0] if len(ghg_rows) > 0 else "",
                "Forest_2030": forest_rows["Forecast"].values[0] if len(forest_rows) > 0 else np.nan,
                "Forest_Model": forest_rows["Model"].values[0] if len(forest_rows) > 0 else "",
            })
        
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary_2030", index=False)
    
    print(f"\n✅ Updated: {output_file}")
    print(f"   GHG now uses ARIMAX with policy dummies (instead of plain ARIMA)")
    print(f"   Total forecasts: {len(df_out)} rows")
    
    # Show which countries had constant vs varying GHG
    ghg_df = df_out[df_out["Variable"] == "GHG"]
    print("\nGHG Forecast Patterns:")
    for country in sorted(SEA_COUNTRIES.values()):
        c_data = ghg_df[ghg_df["Country"] == country]["Forecast"].values
        is_const = len(set(np.round(c_data, 2))) == 1
        status = "CONSTANT" if is_const else "VARYING"
        print(f"  {country}: {status}")

if __name__ == "__main__":
    main()
