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

FOREST_POLICY_DUMMIES = {
    "all": [2021],
    "Brunei Darussalam": [2023],
    "Cambodia": [2023],
    "Indonesia": [2014, 2023],
    "Lao PDR": [2023],
    "Malaysia": [2023],
    "Myanmar": [2023],
    "Philippines": [2023],
    "Singapore": [2019],
    "Thailand": [2023],
    "Timor-Leste": [2023],
    "Vietnam": [2023],
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

def fit_arima(ts):
    if ts is None or len(ts) < 10:
        return None, None
    orders = [(0,1,0), (1,1,0), (1,1,1), (0,1,1)]
    best_model, best_aic, best_order = None, np.inf, None
    for order in orders:
        try:
            m = ARIMA(ts, order=order).fit()
            if m.aic < best_aic:
                best_aic, best_model, best_order = m.aic, m, order
        except:
            pass
    return best_model, best_order

def build_forest_exog(country_label, start_year, end_year):
    dummies = {}
    for policy_year in FOREST_POLICY_DUMMIES.get("all", []):
        dummies[f"global_{policy_year}"] = np.array([1.0 if y == policy_year else 0.0 for y in range(start_year, end_year + 1)])
    for policy_year in FOREST_POLICY_DUMMIES.get(country_label, []):
        dummies[f"{country_label.lower().replace(' ', '_')}_{policy_year}"] = np.array([1.0 if y >= policy_year else 0.0 for y in range(start_year, end_year + 1)])
    if not dummies:
        return None
    return pd.DataFrame(dummies, index=range(start_year, end_year + 1))

def fit_forest_arimax(ts, country_label):
    if ts is None or len(ts) < 10:
        return None, None, None

    start_year = int(ts.index.min())
    end_year = int(ts.index.max())
    exog = build_forest_exog(country_label, start_year, end_year)

    if exog is None:
        return fit_arima(ts)[0], fit_arima(ts)[1], None

    best_model, best_aic, best_order = None, np.inf, None
    for order in [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]:
        try:
            fitted = SARIMAX(ts, exog=exog, order=order, seasonal_order=(0, 0, 0, 0)).fit(disp=False)
            if fitted.aic < best_aic:
                best_model, best_aic, best_order = fitted, fitted.aic, order
        except:
            pass

    if best_model is None:
        return fit_arima(ts)[0], fit_arima(ts)[1], None

    return best_model, best_order, exog.columns.tolist()

def build_future_forest_exog(country_label, start_year, end_year, exog_columns):
    if not exog_columns:
        return None

    future_years = range(start_year, end_year + 1)
    dummies = {}
    for col in exog_columns:
        if col.startswith("global_"):
            policy_year = int(col.split("_")[-1])
            dummies[col] = np.array([1.0 if y == policy_year else 0.0 for y in future_years])
        else:
            policy_year = int(col.rsplit("_", 1)[-1])
            dummies[col] = np.array([1.0 if y >= policy_year else 0.0 for y in future_years])
    return pd.DataFrame(dummies, index=future_years)

def forecast_to_2030(ts, model, exog_future=None):
    if model is None or ts is None:
        return None
    last_year = int(ts.index.max())
    steps = 2030 - last_year
    if steps <= 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if exog_future is not None:
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
        
        # GHG
        ghg_ts = extract_series(df, country_label, GHG_IND)
        if ghg_ts is not None:
            ghg_model = ARIMA(ghg_ts, order=(1, 1, 0)).fit()
            ghg_order = (1, 1, 0)
            ghg_fcast = forecast_to_2030(ghg_ts, ghg_model)
            if ghg_fcast is not None:
                ghg_fcast["Country"] = country_name
                ghg_fcast["Variable"] = "GHG"
                ghg_fcast["Model"] = str(ghg_order)
                all_results.append(ghg_fcast)
        
        # Forest
        forest_ts = extract_series(df, country_label, FOREST_IND)
        if forest_ts is not None:
            forest_model, forest_order, forest_exog_cols = fit_forest_arimax(forest_ts, country_label)
            forest_exog_future = None
            if forest_exog_cols is not None:
                forest_exog_future = build_future_forest_exog(
                    country_label,
                    int(forest_ts.index.max()) + 1,
                    2030,
                    forest_exog_cols,
                )
            forest_fcast = forecast_to_2030(forest_ts, forest_model, exog_future=forest_exog_future)
            if forest_fcast is not None:
                forest_fcast["Country"] = country_name
                forest_fcast["Variable"] = "Forest"
                forest_fcast["Model"] = f"ARIMAX{forest_order}" if forest_exog_cols is not None else str(forest_order)
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
                "Forest_2030": forest_rows["Forecast"].values[0] if len(forest_rows) > 0 else np.nan,
            })
        
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary_2030", index=False)
    
    print(f"\n✅ Saved: {output_file}")
    print(f"   - GHG + Forest forecasts generated")
    print(f"   - Total forecast rows: {len(df_out)}")
    print(f"   - Variables: {df_out['Variable'].unique().tolist()}")

if __name__ == "__main__":
    main()
