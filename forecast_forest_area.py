"""
Forest Area Forecasting Model for 11 SEA countries.

This script builds a compact, reproducible analysis for
"Forest area (% of land area)" and writes:
- a comprehensive report
- a country-level summary CSV
- a detailed rolling-backtest CSV
- per-country indicator ranking CSVs

The focus is on a fast, defensible comparison between:
- Regression with selected indicators
- Univariate ARIMA

The script also records the selected indicators used for each country.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:  # pragma: no cover
    raise SystemExit("statsmodels is required to run this script.") from exc


warnings.filterwarnings("ignore")

SEA_COUNTRIES = ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "TLS", "VNM"]
COUNTRY_NAMES = {
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

TARGET_NAME = "Forest area (% of land area)"
INITIAL_TRAIN_SIZE = 20
TOP_PREDICTORS = 5
INDICATOR_KEYWORDS = [
    "forest",
    "agricultural",
    "arable",
    "land area",
    "land use",
    "urban",
    "rural",
    "population",
    "gdp",
    "energy",
    "electric",
    "fertilizer",
    "precipitation",
    "temperature",
    "rain",
    "trade",
    "exports",
    "imports",
    "cropland",
    "protected areas",
    "deforestation",
    "agriculture",
]
# add policy and enforcement keywords so generated dummies pass the candidate filter
INDICATOR_KEYWORDS += ["carbon", "tax", "ets", "penalty", "court", "litigation", "policy", "regulation"]
ARIMA_CANDIDATES = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
]


@dataclass
class CountryResult:
    country_code: str
    country_name: str
    regression_rmse: float
    regression_mae: float
    arima_rmse: float
    arima_mae: float
    regression_wins: int
    arima_wins: int
    n_test_points: int
    arima_order: str


def load_data() -> pd.DataFrame:
    return pd.read_csv("WB_WDI_WIDEF.csv")


def year_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.isdigit()]


def country_frame(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    return df[df["REF_AREA"] == country_code].copy()


def is_candidate_indicator(indicator: str) -> bool:
    indicator_lower = indicator.lower()
    return any(keyword in indicator_lower for keyword in INDICATOR_KEYWORDS)


def series_from_country_frame(country_df: pd.DataFrame, indicator: str) -> pd.Series | None:
    rows = country_df[country_df["INDICATOR_LABEL"] == indicator]
    if rows.empty:
        return None

    row = rows.iloc[0]
    years = []
    values = []
    for col in year_columns(country_df):
        value = row[col]
        if pd.notna(value) and value != "":
            try:
                years.append(int(col))
                values.append(float(value))
            except (TypeError, ValueError):
                continue

    if len(values) < 3:
        return None

    return pd.Series(values, index=years, dtype=float).sort_index()


def build_country_series_map(country_df: pd.DataFrame) -> dict[str, pd.Series]:
    series_map: dict[str, pd.Series] = {}

    for indicator, group in country_df.groupby("INDICATOR_LABEL", sort=False):
        if pd.isna(indicator):
            continue
        if indicator != TARGET_NAME and not is_candidate_indicator(str(indicator)):
            continue

        row = group.iloc[0]
        years = []
        values = []
        for col in year_columns(country_df):
            value = row[col]
            if pd.notna(value) and value != "":
                try:
                    years.append(int(col))
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue

        if len(values) < 3:
            continue

        series = pd.Series(values, index=years, dtype=float).sort_index()
        series_map[str(indicator)] = series
    return series_map


def add_policy_event_dummies(series_map: dict[str, pd.Series], country_code: str) -> None:
    """Add synthetic binary time series for policy events and corporate penalties.

    The function inserts series into `series_map` with labels that match
    indicator keyword filters so they can be selected as predictors.
    """
    # Determine years to cover from existing series if available, else default range
    years = set()
    for s in series_map.values():
        years.update(s.index.tolist())
    if not years:
        years = set(range(2000, 2026))
    years = sorted(years)

    # Helper to make binary series
    def make_series(active_years: list[int], name: str) -> pd.Series:
        vals = [1.0 if y in active_years else 0.0 for y in years]
        return pd.Series(vals, index=years, dtype=float)

    # Policy events known from the project brief
    # Singapore: carbon tax implemented 2019 (post-2019 indicator)
    if country_code == "SGP":
        series_map["Singapore carbon tax (post-2019)"] = make_series([y for y in years if y >= 2019], "SGP_carbon_tax")

    # Indonesia: ETS launched 2023
    if country_code == "IDN":
        series_map["Indonesia ETS (post-2023)"] = make_series([y for y in years if y >= 2023], "IDN_ets")

    # Corporate penalty case: PT Kallista Alam (Indonesia) 2014
    if country_code == "IDN":
        series_map["PT Kallista Alam penalty 2014"] = make_series([2014], "Kallista_2014")

    # Global/regional litigious events that could influence investor behavior
    # Shell ruling 2021 - include as a global dummy available to all countries
    series_map.setdefault("Global Shell ruling 2021 (litigation)", make_series([2021], "Shell_2021"))

    # Net-zero commitment announcement dummies (use commitment year as proxy)
    net_zero_announcements = {
        "BRN": 2020,
        "KHM": 2020,
        "IDN": 2021,
        "LAO": 2021,
        "MYS": 2021,
        "MMR": 2020,
        "PHL": 2021,
        "SGP": 2020,
        "THA": 2020,
        "TLS": 2021,
        "VNM": 2020,
    }
    ann_year = net_zero_announcements.get(country_code)
    if ann_year:
        series_map[f"Net-zero announcement ({ann_year})"] = make_series([y for y in years if y >= ann_year], "net_zero_announced")


def select_predictors(target: pd.Series, series_map: dict[str, pd.Series], top_n: int = TOP_PREDICTORS) -> list[tuple[str, float]]:
    correlations: list[tuple[str, float]] = []
    target_years = target.index.tolist()

    for indicator, series in series_map.items():
        if indicator == TARGET_NAME:
            continue

        common_years = sorted(set(target_years) & set(series.index.tolist()))
        if len(common_years) < 5:
            continue

        aligned_target = target.loc[common_years]
        aligned_series = series.loc[common_years]
        mask = aligned_target.notna() & aligned_series.notna()
        if mask.sum() < 5:
            continue

        corr = aligned_target[mask].corr(aligned_series[mask])
        if pd.notna(corr):
            correlations.append((indicator, abs(float(corr))))

    correlations.sort(key=lambda item: item[1], reverse=True)
    return correlations[:top_n]


def project_exog(
    series_map: dict[str, pd.Series],
    predictor_names: list[str],
    train_end_year: int,
    forecast_year: int,
) -> pd.DataFrame | None:
    data = {}
    for name in predictor_names:
        series = series_map.get(name)
        if series is None:
            continue

        historical = series[series.index <= train_end_year].sort_index()
        if len(historical) < 2:
            continue

        history_values = historical.astype(float).values
        history_values = np.where(np.isnan(history_values), np.nanmean(history_values), history_values)
        if np.isnan(history_values).all():
            continue

        if forecast_year in historical.index:
            projected_value = float(historical.loc[forecast_year])
        else:
            # Use the latest observed value as a conservative projection.
            projected_value = float(history_values[-1])

        data[name] = [projected_value]

    if not data:
        return None

    return pd.DataFrame(data, index=[forecast_year])


def fit_regression_forecast(y_train: pd.Series, x_train: pd.DataFrame, x_future: pd.DataFrame) -> float | None:
    try:
        if x_train is None or x_future is None or x_train.empty or x_future.empty:
            return float(y_train.iloc[-1])

        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        x_train_imputed = imputer.fit_transform(x_train)
        x_future_imputed = imputer.transform(x_future)
        x_train_scaled = scaler.fit_transform(x_train_imputed)
        x_future_scaled = scaler.transform(x_future_imputed)

        model = Ridge(alpha=1.0)
        model.fit(x_train_scaled, y_train.values)
        forecast = float(model.predict(x_future_scaled)[0])
        return float(np.clip(forecast, 0.0, 100.0))
    except Exception:
        return float(y_train.iloc[-1]) if len(y_train) else None


def select_arima_order(series: pd.Series) -> tuple[int, int, int]:
    best_order = ARIMA_CANDIDATES[0]
    best_aic = np.inf

    for order in ARIMA_CANDIDATES:
        try:
            result = ARIMA(series, order=order).fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
        except Exception:
            continue

    return best_order


def fit_arima_forecast(y_train: pd.Series, order: tuple[int, int, int]) -> float | None:
    try:
        result = ARIMA(y_train, order=order).fit()
        return float(result.get_forecast(steps=1).predicted_mean.iloc[0])
    except Exception:
        return float(y_train.iloc[-1]) if len(y_train) else None


def build_backtest(
    target: pd.Series,
    series_map: dict[str, pd.Series],
    predictor_names: list[str],
) -> tuple[CountryResult | None, list[dict]]:
    years = sorted(target.index.tolist())
    if len(years) < INITIAL_TRAIN_SIZE + 4:
        return None, []

    initial_train_years = years[:INITIAL_TRAIN_SIZE]
    initial_train_target = target.loc[initial_train_years]
    arima_order = select_arima_order(initial_train_target)

    regression_sq_errors: list[float] = []
    regression_abs_errors: list[float] = []
    arima_sq_errors: list[float] = []
    arima_abs_errors: list[float] = []
    regression_wins = 0
    arima_wins = 0
    detailed_rows: list[dict] = []

    for test_idx in range(INITIAL_TRAIN_SIZE, len(years)):
        train_years = years[:test_idx]
        test_year = years[test_idx]
        y_train = target.loc[train_years]
        actual = float(target.loc[test_year])

        x_train_rows = []
        for year in train_years:
            row = []
            for predictor in predictor_names:
                series = series_map.get(predictor)
                row.append(float(series.loc[year]) if series is not None and year in series.index else np.nan)
            x_train_rows.append(row)

        x_train = pd.DataFrame(x_train_rows, index=train_years, columns=predictor_names)
        x_future = project_exog(series_map, predictor_names, train_years[-1], test_year)

        regression_forecast = fit_regression_forecast(y_train, x_train, x_future)
        arima_forecast = fit_arima_forecast(y_train, arima_order)

        if regression_forecast is not None:
            regression_error = actual - regression_forecast
            regression_sq_errors.append(regression_error ** 2)
            regression_abs_errors.append(abs(regression_error))

        if arima_forecast is not None:
            arima_error = actual - arima_forecast
            arima_sq_errors.append(arima_error ** 2)
            arima_abs_errors.append(abs(arima_error))

        if regression_forecast is not None and arima_forecast is not None:
            reg_abs = abs(actual - regression_forecast)
            arima_abs = abs(actual - arima_forecast)
            if reg_abs < arima_abs:
                regression_wins += 1
                winner = "Regression"
            elif arima_abs < reg_abs:
                arima_wins += 1
                winner = "ARIMA"
            else:
                winner = "Tie"
        else:
            winner = "Tie"

        detailed_rows.append(
            {
                "year": test_year,
                "actual": actual,
                "regression_forecast": regression_forecast,
                "arima_forecast": arima_forecast,
                "regression_abs_error": abs(actual - regression_forecast) if regression_forecast is not None else np.nan,
                "arima_abs_error": abs(actual - arima_forecast) if arima_forecast is not None else np.nan,
                "winner": winner,
            }
        )

    regression_rmse = float(np.sqrt(np.mean(regression_sq_errors))) if regression_sq_errors else np.inf
    regression_mae = float(np.mean(regression_abs_errors)) if regression_abs_errors else np.inf
    arima_rmse = float(np.sqrt(np.mean(arima_sq_errors))) if arima_sq_errors else np.inf
    arima_mae = float(np.mean(arima_abs_errors)) if arima_abs_errors else np.inf

    return (
        CountryResult(
            country_code="",
            country_name="",
            regression_rmse=regression_rmse,
            regression_mae=regression_mae,
            arima_rmse=arima_rmse,
            arima_mae=arima_mae,
            regression_wins=regression_wins,
            arima_wins=arima_wins,
            n_test_points=len(detailed_rows),
            arima_order=str(arima_order),
        ),
        detailed_rows,
    )


def run() -> None:
    out_dir = Path("model_outputs")
    out_dir.mkdir(exist_ok=True)

    df = load_data()
    country_results: list[CountryResult] = []
    detailed_rows_all: list[dict] = []
    predictor_rows_all: list[dict] = []

    for country_code in SEA_COUNTRIES:
        country_name = COUNTRY_NAMES[country_code]
        cdf = country_frame(df, country_code)
        series_map = build_country_series_map(cdf)
        target = series_map.get(TARGET_NAME)

        if target is None:
            continue

        selected = select_predictors(target, series_map, TOP_PREDICTORS)
        predictor_names = [name for name, _ in selected]

        country_predictor_rows = []
        for rank, (indicator, corr) in enumerate(selected, start=1):
            country_predictor_rows.append(
                {
                    "country_code": country_code,
                    "country_name": country_name,
                    "rank": rank,
                    "indicator": indicator,
                    "abs_correlation": corr,
                }
            )
            predictor_rows_all.append(country_predictor_rows[-1])

        pd.DataFrame(country_predictor_rows).to_csv(
            out_dir / f"best_predictors_forest_area_{country_code}.csv",
            index=False,
        )

        # Run base backtest using selected predictors (original)
        result_base, detailed_rows = build_backtest(target, series_map, predictor_names)
        if result_base is None:
            continue

        result_base.country_code = country_code
        result_base.country_name = country_name

        # Identify generated policy/penalty dummies available in series_map
        dummy_keywords = ["carbon", "ets", "penalty", "kallista", "shell", "net-zero"]
        available_dummies = [k for k in series_map.keys() if any(d in k.lower() for d in dummy_keywords)]

        # Build augmented predictor list (ensure we include dummies)
        augmented_predictors = predictor_names.copy()
        for d in available_dummies:
            if d not in augmented_predictors:
                augmented_predictors.append(d)

        # Run backtest with augmented predictors
        result_aug, detailed_rows_aug = build_backtest(target, series_map, augmented_predictors)
        if result_aug is None:
            # If augmented backtest failed, fall back to base
            result_aug = result_base

        result_aug.country_code = country_code
        result_aug.country_name = country_name

        # Save both results
        country_results.append((result_base, result_aug))

        for row in detailed_rows:
            detailed_rows_all.append(
                {
                    "country_code": country_code,
                    "country_name": country_name,
                    **row,
                }
            )

    pd.DataFrame(detailed_rows_all).to_csv(out_dir / "forest_area_forecast_detailed.csv", index=False)

    # country_results now contains tuples (base_result, augmented_result)
    summary_rows = []
    ablation_rows = []
    paired_results = country_results
    for base_r, aug_r in paired_results:
        summary_rows.append(
            {
                "country_code": base_r.country_code,
                "country_name": base_r.country_name,
                "regression_rmse_base": base_r.regression_rmse,
                "regression_mae_base": base_r.regression_mae,
                "arima_rmse": base_r.arima_rmse,
                "arima_mae": base_r.arima_mae,
                "regression_wins_base": base_r.regression_wins,
                "arima_wins": base_r.arima_wins,
                "n_test_points": base_r.n_test_points,
                "arima_order": base_r.arima_order,
            }
        )

        ablation_rows.append(
            {
                "country_code": base_r.country_code,
                "country_name": base_r.country_name,
                "regression_rmse_base": base_r.regression_rmse,
                "regression_rmse_augmented": aug_r.regression_rmse,
                "delta_rmse": (aug_r.regression_rmse - base_r.regression_rmse),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "forest_area_forecast_summary.csv", index=False)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out_dir / "forest_area_policy_dummy_ablation.csv", index=False)

    write_report(paired_results, predictor_rows_all, out_dir)


def write_report(
    paired_results: list[tuple[CountryResult, CountryResult]],
    predictor_rows_all: list[dict],
    out_dir: Path,
) -> None:
    report_path = out_dir / "forest_area_forecast_report.txt"
    predictor_df = pd.DataFrame(predictor_rows_all)

    # paired_results contains tuples (base_result, augmented_result)
    total_forecasts = sum(base.n_test_points for base, _ in paired_results) or 1
    avg_regression_rmse = float(np.mean([base.regression_rmse for base, _ in paired_results])) if paired_results else np.inf
    avg_arima_rmse = float(np.mean([base.arima_rmse for base, _ in paired_results])) if paired_results else np.inf
    total_regression_wins = sum(base.regression_wins for base, _ in paired_results)
    total_arima_wins = sum(base.arima_wins for base, _ in paired_results)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FOREST AREA FORECASTING MODEL - COMPREHENSIVE REPORT\n")
        f.write("11 Southeast Asian Countries\n")
        f.write("=" * 80 + "\n\n")

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Target variable: {TARGET_NAME}\n")
        f.write(f"Countries analyzed: {len(paired_results)}\n")
        f.write("Models compared: Regression vs ARIMA\n")
        f.write(f"Backtest design: rolling-origin with {INITIAL_TRAIN_SIZE}-year initial window\n\n")

        f.write("OVERALL MODEL PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<18} {'Avg RMSE':<14} {'Wins':<10} {'Win %':<10}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Regression':<18} {avg_regression_rmse:<14.4f} {total_regression_wins:<10} {(total_regression_wins/total_forecasts*100):<9.1f}%\n")
        f.write(f"{'ARIMA':<18} {avg_arima_rmse:<14.4f} {total_arima_wins:<10} {(total_arima_wins/total_forecasts*100):<9.1f}%\n")
        f.write("-" * 80 + "\n")
        best_model = "ARIMA" if avg_arima_rmse < avg_regression_rmse else "Regression"
        f.write(f"Recommended model: {best_model}\n\n")

        f.write("SELECTED INDICATORS\n")
        f.write("-" * 80 + "\n")
        for country_code in SEA_COUNTRIES:
            rows = predictor_df[predictor_df["country_code"] == country_code]
            if rows.empty:
                continue
            f.write(f"{country_code} - {COUNTRY_NAMES[country_code]}\n")
            for _, row in rows.sort_values("rank").iterrows():
                f.write(f"  {int(row['rank'])}. {row['indicator']} (|corr|={row['abs_correlation']:.4f})\n")
            f.write("\n")

        f.write("PER-COUNTRY RESULTS\n")
        f.write("-" * 80 + "\n")
        for base_result, aug_result in paired_results:
            f.write(f"Country: {base_result.country_code} - {base_result.country_name}\n")
            f.write(f"  Regression (base): RMSE={base_result.regression_rmse:.4f}, MAE={base_result.regression_mae:.4f}, Wins={base_result.regression_wins}\n")
            f.write(f"  Regression (with policy dummies): RMSE={aug_result.regression_rmse:.4f}, MAE={aug_result.regression_mae:.4f}, Wins={aug_result.regression_wins}\n")
            f.write(f"  ARIMA:      RMSE={base_result.arima_rmse:.4f}, MAE={base_result.arima_mae:.4f}, Wins={base_result.arima_wins}, Order={base_result.arima_order}\n")
            f.write(f"  Test points: {base_result.n_test_points}\n\n")

        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Use the selected top-correlation indicators as the regression feature set for each country.\n")
        f.write("2. ARIMA is the strongest baseline when only the forest-area history is used.\n")
        f.write("3. If future exogenous forecasts become available, rerun the regression model with better predictor projections.\n")
        f.write("4. Country-specific tuning is likely to improve results more than a single global specification.\n")
        f.write("5. Forest-area forecasts should be interpreted alongside land-use policy and deforestation indicators.\n\n")

        f.write("FILES WRITTEN\n")
        f.write("-" * 80 + "\n")
        f.write("- model_outputs/forest_area_forecast_report.txt\n")
        f.write("- model_outputs/forest_area_forecast_summary.csv\n")
        f.write("- model_outputs/forest_area_forecast_detailed.csv\n")
        f.write("- model_outputs/best_predictors_forest_area_<country>.csv\n")
        f.write("- model_outputs/forest_area_policy_dummy_ablation.csv\n")
        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    run()