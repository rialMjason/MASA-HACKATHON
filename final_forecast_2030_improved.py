from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    raise SystemExit("statsmodels required") from exc

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
